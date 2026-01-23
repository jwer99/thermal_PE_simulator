from fpdf import FPDF
import datetime
import os
import tempfile
import base64
import numpy as np

# --- Colors (RGB) ---
COLOR_PRIMARY = (44, 62, 80)      # Dark Blue/Grey
COLOR_SECONDARY = (52, 152, 219)  # Light Blue
COLOR_ACCENT = (231, 76, 60)      # Red
COLOR_TEXT_HEADER = (255, 255, 255)
COLOR_TEXT_BODY = (50, 50, 50)
COLOR_TABLE_HEADER_BG = (220, 220, 220)
COLOR_TABLE_ROW_ODD = (255, 255, 255)
COLOR_TABLE_ROW_EVEN = (245, 245, 245)

class ReportPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.alias_nb_pages()

    def header(self):
        # Top colored strip
        self.set_fill_color(*COLOR_PRIMARY)
        self.rect(0, 0, 210, 20, 'F')

        # Title text
        self.set_xy(10, 5)
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(*COLOR_TEXT_HEADER)
        self.cell(0, 10, 'THERMAL SIMULATION REPORT', 0, 0, 'L')

        # Date on right
        self.set_xy(150, 5)
        self.set_font('Helvetica', '', 10)
        self.cell(50, 10, datetime.datetime.now().strftime('%Y-%m-%d %H:%M'), 0, 0, 'R')

        # Reset colors and position
        self.set_text_color(*COLOR_TEXT_BODY)
        self.ln(20) # Space after header strip

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')

    def section_title(self, label):
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(*COLOR_PRIMARY)
        self.cell(0, 10, label, 0, 1, 'L')
        # Underline
        line_y = self.get_y()
        self.set_draw_color(*COLOR_SECONDARY)
        self.set_line_width(0.5)
        self.line(10, line_y, 200, line_y)
        self.ln(5)
        self.set_text_color(*COLOR_TEXT_BODY)

    def chapter_body_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.multi_cell(0, 6, text)
        self.ln()

    def key_value_table(self, data, col_widths=(80, 110)):
        """
        Renders a dictionary as a clean table.
        """
        self.set_font('Helvetica', '', 10)
        self.set_line_width(0.1)
        self.set_draw_color(200, 200, 200)

        fill = False
        for key, value in data.items():
            if fill:
                self.set_fill_color(*COLOR_TABLE_ROW_EVEN)
            else:
                self.set_fill_color(*COLOR_TABLE_ROW_ODD)

            self.set_font('Helvetica', 'B', 10)
            self.cell(col_widths[0], 8, f"  {key}", 1, 0, 'L', True)

            self.set_font('Helvetica', '', 10)
            # Handle long text? For now assume it fits or simple wrap manually if needed,
            # but standard cell doesn't wrap. Using multi_cell in a table is tricky.
            # We'll stick to single line for parameters usually.
            val_str = str(value)
            self.cell(col_widths[1], 8, f"  {val_str}", 1, 1, 'L', True)

            fill = not fill
        self.ln(5)

    def add_image_from_base64(self, b64_string, w=180, caption=None):
        if not b64_string or 'base64,' not in b64_string:
            return

        try:
            image_data = base64.b64decode(b64_string.split('base64,')[1])
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img_file:
                temp_img_file.write(image_data)
                temp_img_path = temp_img_file.name

            # Center the image
            x_pos = (210 - w) / 2
            self.image(temp_img_path, x=x_pos, w=w)

            if caption:
                self.set_font('Helvetica', 'I', 9)
                self.cell(0, 8, caption, 0, 1, 'C')

            self.ln(5)

            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

        except Exception as e:
            print(f"Error adding image to PDF: {e}")

def format_val(val, unit, precision=2):
    if val is None or (isinstance(val, (int, float)) and (np.isnan(val) or np.isinf(val))):
        return 'N/A'
    try:
        return f'{float(val):.{precision}f} {unit}'
    except (ValueError, TypeError):
        return str(val)

def create_pdf_report(data):
    """
    Generates a professional PDF report.
    """
    inputs = data.get('inputs', {})
    outputs = data.get('outputs', {})

    pdf = ReportPDF()
    pdf.add_page()

    # 1. Executive Summary / Results Highlights
    pdf.section_title('1. Simulation Results Summary')

    results_data = {
        'Status': outputs.get('status', 'Unknown'),
        'Convergence': "Yes" if outputs.get('convergence') else "No",
        'Calculated Heatsink Rth': format_val(outputs.get('rth_heatsink_calculated'), 'K/W', 4),
        'Max Base Temperature': format_val(outputs.get('t_max_base'), '°C'),
        'Max Junction Temperature': f"{format_val(outputs.get('t_max_junction'), '°C')} ({outputs.get('t_max_junction_chip', 'N/A')})",
        'Max NTC Temperature': format_val(outputs.get('t_max_ntc'), '°C'),
        'Outlet Air Temperature': format_val(outputs.get('t_air_outlet'), '°C'),
    }
    pdf.key_value_table(results_data)

    # 2. Input Parameters
    pdf.section_title('2. Input Parameters')

    hs_params = inputs.get('heatsink_params', {})
    env_params = inputs.get('environment_params', {})

    # Heatsink
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, 'Heatsink Geometry & Material', 0, 1, 'L')

    hs_data = {
        'Dimensions (Lx x Ly)': f"{hs_params.get('lx')} m x {hs_params.get('ly')} m",
        'Base Thickness': f"{hs_params.get('t')} m",
        'Conductivity': f"{hs_params.get('k_base')} W/mK",
    }
    if hs_params.get('use_manual_rth'):
        hs_data['Rth Mode'] = f"Manual ({hs_params.get('rth_heatsink_manual')} K/W)"
    else:
        hs_data['Rth Mode'] = "Auto-Calculated (Fins)"
        hs_data['Fins'] = f"{hs_params.get('num_fins')} fins (h={hs_params.get('h_fin')}m, t={hs_params.get('t_fin')}m)"

    pdf.key_value_table(hs_data)

    # Environment
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, 'Environmental Conditions', 0, 1, 'L')
    env_data = {
        'Inlet Air Temp': f"{env_params.get('t_ambient_inlet')} °C",
        'Air Flow Rate': f"{env_params.get('Q_total_m3_h')} m³/h"
    }
    pdf.key_value_table(env_data)

    # Modules
    pdf.section_title('3. Module Configuration')
    module_defs = inputs.get('module_definitions', [])
    powers = inputs.get('powers', {})

    if not module_defs:
        pdf.chapter_body_text("No modules defined.")
    else:
        # Custom table for modules
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_fill_color(*COLOR_TABLE_HEADER_BG)
        # Headers: ID, Pos (X,Y), Power (IGBT1/D1/IGBT2/D2)
        pdf.cell(40, 8, "Module ID", 1, 0, 'C', True)
        pdf.cell(50, 8, "Position (x, y) [m]", 1, 0, 'C', True)
        pdf.cell(100, 8, "Power Losses [W]", 1, 1, 'C', True)

        pdf.set_font('Helvetica', '', 9)
        fill = False
        for mod in module_defs:
            if fill: pdf.set_fill_color(*COLOR_TABLE_ROW_EVEN)
            else: pdf.set_fill_color(*COLOR_TABLE_ROW_ODD)

            mid = mod.get('id', 'N/A')
            pos = f"({mod.get('center_x')}, {mod.get('center_y')})"

            # Construct power string
            p_str = []
            for chip in ['IGBT1', 'Diode1', 'IGBT2', 'Diode2']:
                val = powers.get(f"{mid}_{chip}", 0)
                if val > 0:
                    p_str.append(f"{chip}:{val}")
            p_text = ", ".join(p_str) if p_str else "0 W"

            pdf.cell(40, 8, mid, 1, 0, 'C', True)
            pdf.cell(50, 8, pos, 1, 0, 'C', True)
            pdf.cell(100, 8, p_text, 1, 1, 'L', True)
            fill = not fill
        pdf.ln(5)

    # 4. Visualizations
    pdf.add_page()
    pdf.section_title('4. Visualizations')

    plot_base_uri = outputs.get('plot_base_data_uri')
    if plot_base_uri:
        pdf.add_image_from_base64(plot_base_uri, w=190, caption="Figure 1: Heatsink Base Temperature Distribution & Air Temperature Rise")

    plot_zoom_uri = outputs.get('plot_zoom_data_uri')
    if plot_zoom_uri:
        if pdf.get_y() > 200: pdf.add_page()
        pdf.add_image_from_base64(plot_zoom_uri, w=190, caption="Figure 2: Detailed Temperature Maps (Module Footprints)")

    # fpdf2 output() with no arguments writes to file, but we need bytes.
    # In fpdf2, we can use output() to get bytearray directly.
    return bytes(pdf.output())

# app.py
from flask import Flask, render_template, request, jsonify, Response, url_for, session, send_from_directory
import numpy as np
import os
import time
import json
import datetime
import logging
import base64
import io
import tempfile
# +++ NUEVA IMPORTACIÓN PARA PDF +++
from fpdf import FPDF

# IMPORTAR LA NUEVA FUNCIÓN MAESTRA Y OTRAS NECESARIAS
from simulador_core import run_simulation_with_ntc_compensation, chip_rel_positions

# calculate_h_convection_detailed ya no se llama directamente desde app.py
# run_thermal_simulation tampoco se llama directamente desde app.py para el flujo principal

app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Default Configuration (sin cambios) ---
DEFAULT_HEATSINK_PARAMS = {
    'lx': 0.25, 'ly': 0.35, 't': 0.02,  # t es espesor de la base
    'k_base': 218.0,
    'h_fin': 0.112, 't_fin': 0.005, 'num_fins': 31,
    'w_hollow': 0.003, 'h_hollow': 0.05, 'num_hollow_per_fin': 2
}
DEFAULT_ENVIRONMENT_PARAMS = {
    't_ambient_inlet': 40.0,
    'Q_total_m3_h': 750.0
}
static_images_dir = os.path.join(app.static_folder, 'images')
os.makedirs(static_images_dir, exist_ok=True)
PDF_FOLDER = os.path.join(app.root_path, 'static')
PDF_FILENAME = 'thermal_simulator_intro.pdf'


# --- Helper Function for JSON Serialization (sin cambios) ---
def make_serializable(data):
    if isinstance(data, dict):
        return {k: make_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [make_serializable(item) for item in data]
    elif isinstance(data, (np.floating, float)):
        return None if np.isnan(data) or np.isinf(data) else float(data)
    elif isinstance(data, (np.integer, int)):
        return int(data)
    elif isinstance(data, (np.bool_, bool)):
        return bool(data)
    elif isinstance(data, np.ndarray):
        return make_serializable(data.tolist())
    elif data is None:
        return None
    try:
        json.dumps(data)
        return data
    except TypeError:
        return str(data)


# +++ INICIO: NUEVAS FUNCIONES PARA GENERAR PDF +++

class PDF(FPDF):
    """Clase personalizada para añadir cabecera y pie de página al PDF."""

    def header(self):
        self.set_font('helvetica', 'B', 12)
        self.cell(0, 10, 'Technical Simulation Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('helvetica', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, data_dict):
        self.set_font('helvetica', '', 10)
        for key, value in data_dict.items():
            self.set_font('helvetica', 'B', 10)
            self.cell(60, 6, f'{key}:')
            self.set_font('helvetica', '', 10)
            self.multi_cell(0, 6, str(value))
        self.ln()

    def add_image_from_base64(self, b64_string, x, y, w):
        """
        Función robusta para decodificar una imagen Base64, guardarla en un archivo
        temporal y añadirla al PDF desde el archivo. Esto es más fiable que
        usar streams en memoria con fpdf.
        """
        if not b64_string or 'base64,' not in b64_string:
            print("[PDF Gen] Cadena Base64 vacía o mal formada. No se añade la imagen.")
            return

        try:
            # 1. Extraer los datos puros de la imagen
            image_data = base64.b64decode(b64_string.split('base64,')[1])

            # 2. Crear un archivo temporal con la extensión correcta (.png)
            #    'delete=False' es crucial para poder pasar la ruta a fpdf.
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img_file:
                temp_img_file.write(image_data)
                temp_img_path = temp_img_file.name # Guardar la ruta del archivo

            print(f"[PDF Gen] Imagen guardada en archivo temporal: {temp_img_path}")

            # 3. Añadir la imagen al PDF usando la ruta del archivo temporal
            self.image(temp_img_path, x=x, y=y, w=w)

            print(f"[PDF Gen] Imagen '{temp_img_path}' añadida al PDF correctamente desde archivo.")

        except Exception as e:
            print(f"!!! Error Crítico al procesar imagen Base64 para PDF: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 4. Asegurarse de eliminar el archivo temporal después de su uso
            if 'temp_img_path' in locals() and os.path.exists(temp_img_path):
                os.remove(temp_img_path)
                print(f"[PDF Gen] Archivo temporal eliminado: {temp_img_path}")


def create_pdf_report(data):
    """Genera el contenido del PDF a partir de los datos de la simulación."""
    inputs = data.get('inputs', {})
    outputs = data.get('outputs', {})

    pdf = PDF()
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 16)
    pdf.cell(0, 10, 'Thermal Analysis Detailed Report', 0, 1, 'C')
    pdf.set_font('helvetica', '', 10)
    pdf.cell(0, 6, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
    pdf.ln(10)

    # --- Sección de Parámetros de Entrada ---
    pdf.chapter_title('1. Input Parameters')
    hs_params = inputs.get('heatsink_params', {})
    env_params = inputs.get('environment_params', {})

    pdf.set_font('helvetica', 'B', 11)
    pdf.cell(0, 8, '1.1 Heatsink & Fin Design', 0, 1, 'L')
    heatsink_design_data = {
        'Length (Lx)': f"{hs_params.get('lx', 'N/A')} m",
        'Width (Ly)': f"{hs_params.get('ly', 'N/A')} m",
        'Base Thickness (t)': f"{hs_params.get('t', 'N/A')} m",
        'Base Conductivity (k)': f"{hs_params.get('k_base', 'N/A')} W/mK",
    }
    if hs_params.get('use_manual_rth', False) and hs_params.get('rth_heatsink_manual') is not None:
        heatsink_design_data['Rth Mode'] = "Manual Input"
        heatsink_design_data['Manual Rth Value'] = f"{hs_params.get('rth_heatsink_manual')} °C/W"
    else:
        heatsink_design_data['Rth Mode'] = "Calculated from Geometry/Flow"
        heatsink_design_data['Fin Height (h_fin)'] = f"{hs_params.get('h_fin', 'N/A')} m"
        heatsink_design_data['Fin Thickness (t_fin)'] = f"{hs_params.get('t_fin', 'N/A')} m"
        heatsink_design_data['Number of Fins'] = f"{hs_params.get('num_fins', 'N/A')}"
        # Optionally, add hollow parameters if they are relevant when not using manual Rth
        if int(hs_params.get('num_fins', 0)) > 0 and int(hs_params.get('num_hollow_per_fin', 0)) > 0 :
            heatsink_design_data['Hollow Width (w_hollow)'] = f"{hs_params.get('w_hollow', 'N/A')} m"
            heatsink_design_data['Hollow Height (h_hollow)'] = f"{hs_params.get('h_hollow', 'N/A')} m"
            heatsink_design_data['Hollows per Fin'] = f"{hs_params.get('num_hollow_per_fin', 'N/A')}"


    pdf.chapter_body(heatsink_design_data)

    pdf.set_font('helvetica', 'B', 11)
    pdf.cell(0, 8, '1.2 Environment', 0, 1, 'L')
    pdf.chapter_body({
        'Inlet Air Temperature': f"{env_params.get('t_ambient_inlet', 'N/A')} °C",
        'Air Flow Rate': f"{env_params.get('Q_total_m3_h', 'N/A')} m³/h",
    })

    pdf.set_font('helvetica', 'B', 11)
    pdf.cell(0, 8, '1.3 Module Configuration', 0, 1, 'L')
    pdf.set_font('helvetica', '', 10)
    module_defs = inputs.get('module_definitions', [])
    if module_defs:
        for mod in module_defs:
            pdf.multi_cell(0, 6,
                           f" - Module '{mod.get('id', 'N/A')}': Center at (X={mod.get('center_x')}, Y={mod.get('center_y')}) m")
            for chip in ['IGBT1', 'Diode1', 'IGBT2', 'Diode2']:
                power = inputs.get('powers', {}).get(f"{mod.get('id', '')}_{chip}", "N/A")
                pdf.multi_cell(0, 5, f"     - {chip} Power: {power} W")
    else:
        pdf.multi_cell(0, 6, " - No modules were placed in this simulation.")
    pdf.ln()

    # --- Sección de Resultados ---
    pdf.add_page()
    pdf.chapter_title('2. Simulation Results')

    def format_val(val, unit, precision=2):
        if val is None or (isinstance(val, (int, float)) and (np.isnan(val) or np.isinf(val))):
            return f'N/A {unit}'
        try:
            return f'{float(val):.{precision}f} {unit}'
        except (ValueError, TypeError):
            return str(val)

    pdf.chapter_body({
        'Simulation Status': outputs.get('status', 'Unknown'),
        'Convergence': "Yes" if outputs.get('convergence') else "No",
        'Heatsink Convergence Iterations': f"{outputs.get('h_iterations', 'N/A')}",
        'Max Base Temperature': format_val(outputs.get('t_max_base'), '°C'),
        'Average Base Temperature': format_val(outputs.get('t_avg_base'), '°C'),
        'Outlet Air Temperature': format_val(outputs.get('t_air_outlet'), '°C'),
        'Max Junction Temperature': f"{format_val(outputs.get('t_max_junction'), '°C')} ({outputs.get('t_max_junction_chip', 'N/A')})",
        'Max NTC Temperature': format_val(outputs.get('t_max_ntc'), '°C'),
        'Calculated Heatsink Rth': format_val(outputs.get('rth_heatsink_calculated'), 'K/W', 4),
    })

    pdf.chapter_title('3. Visualizations')
    page_width = pdf.w - 2 * pdf.l_margin

    plot_base_uri = outputs.get('plot_base_data_uri')
    if plot_base_uri:
        pdf.set_font('helvetica', 'B', 11)
        pdf.cell(0, 8, '3.1 Overview (Base & Air Temperature Maps)', 0, 1, 'L')
        img_width = page_width * 0.9
        img_x_pos = pdf.l_margin + (page_width - img_width) / 2
        pdf.add_image_from_base64(plot_base_uri, x=img_x_pos, y=pdf.get_y(), w=img_width)
        pdf.ln(95)
    else:
        pdf.set_font('helvetica', 'I', 10)
        pdf.cell(0, 8, '3.1 Overview plot not available.', 0, 1, 'L')
        pdf.ln(5)

    plot_zoom_uri = outputs.get('plot_zoom_data_uri')
    if plot_zoom_uri:
        if pdf.get_y() > (pdf.h - 120):
            pdf.add_page()
        pdf.set_font('helvetica', 'B', 11)
        pdf.cell(0, 8, '3.2 Module Detail (Module Baseplate Temperature)', 0, 1, 'L')
        img_width = page_width * 0.9
        img_x_pos = pdf.l_margin + (page_width - img_width) / 2
        pdf.add_image_from_base64(plot_zoom_uri, x=img_x_pos, y=pdf.get_y(), w=img_width)
    else:
        pdf.set_font('helvetica', 'I', 10)
        pdf.cell(0, 8, '3.2 Module detail plot not available.', 0, 1, 'L')

    return pdf.output(dest='S').encode('latin-1')


# +++ FIN: NUEVAS FUNCIONES PARA GENERAR PDF +++


@app.route('/')
def index():
    print("Accessing main route ('/') - Dynamic Mode")
    return render_template('index.html',
                           initial_heatsink=DEFAULT_HEATSINK_PARAMS,
                           initial_environment=DEFAULT_ENVIRONMENT_PARAMS,
                           initial_results=None)


@app.route('/update_simulation', methods=['POST'])
def update_simulation():
    print(">>> update_simulation FUNCTION START (ahora llama a iterador de 'h')")
    start_request_time = time.time()
    try:
        data = request.get_json()
        if not data:
            print("!!! ERROR: No JSON data received or data is None.")
            return jsonify({'status': 'Error', 'message': 'No JSON data received'}), 400
        print(f"Received data: {data}")

        # --- Get Heatsink Parameters ---
        heatsink_data = data.get('heatsink_params')
        if not heatsink_data or not isinstance(heatsink_data, dict):
            return jsonify({'status': 'Error', 'message': 'Missing or invalid heatsink_params'}), 400

        try:
            lx_val = float(heatsink_data['lx'])
            ly_val = float(heatsink_data['ly'])
            t_base_val = float(heatsink_data['t'])
            k_base_val = float(heatsink_data['k_base'])

            # Get new manual Rth parameters
            rth_heatsink_manual_str = heatsink_data.get('rth_heatsink_manual')
            use_manual_rth_val = heatsink_data.get('use_manual_rth', False)
            rth_heatsink_manual_val = None

            if use_manual_rth_val:
                if rth_heatsink_manual_str is None or rth_heatsink_manual_str.strip() == "":
                    raise ValueError("Manual Rth value is required when 'Use Manual Rth' is checked.")
                try:
                    rth_heatsink_manual_val = float(rth_heatsink_manual_str)
                    if rth_heatsink_manual_val <= 1e-9:
                        raise ValueError("Manual Rth value must be positive (> 0).")
                except ValueError as e:
                    raise ValueError(f"Invalid Manual Rth value: {e}")

            if not (lx_val > 0 and ly_val > 0 and t_base_val > 0 and k_base_val > 1e-9):
                raise ValueError("Heatsink base parameters (Lx, Ly, t, k) must be positive (> 0).")

        except (KeyError, ValueError, TypeError) as e:
            return jsonify({'status': 'Error', 'message': f'Invalid heatsink base parameters: {e}'}), 400

        try:
            h_fin_val_str = heatsink_data.get('h_fin')
            h_fin_val = float(h_fin_val_str) if h_fin_val_str and h_fin_val_str.strip() != "" else DEFAULT_HEATSINK_PARAMS['h_fin']

            t_fin_val_str = heatsink_data.get('t_fin')
            t_fin_val = float(t_fin_val_str) if t_fin_val_str and t_fin_val_str.strip() != "" else DEFAULT_HEATSINK_PARAMS['t_fin']

            num_fins_val_str = heatsink_data.get('num_fins')
            num_fins_val = int(num_fins_val_str) if num_fins_val_str and num_fins_val_str.strip() != "" else DEFAULT_HEATSINK_PARAMS['num_fins']

            w_hollow_val = float(heatsink_data.get('w_hollow', DEFAULT_HEATSINK_PARAMS['w_hollow']))
            h_hollow_val = float(heatsink_data.get('h_hollow', DEFAULT_HEATSINK_PARAMS['h_hollow']))
            num_hollow_per_fin_val = int(
                heatsink_data.get('num_hollow_per_fin', DEFAULT_HEATSINK_PARAMS['num_hollow_per_fin']))

            if not (h_fin_val >= 0 and t_fin_val >= 0 and num_fins_val >= 0 and
                    w_hollow_val >= 0 and h_hollow_val >= 0 and num_hollow_per_fin_val >= 0):
                raise ValueError("Fin parameters must be non-negative.")
            if num_fins_val > 0 and (t_fin_val <= 1e-9 or h_fin_val <= 1e-9):
                raise ValueError("If num_fins > 0, t_fin and h_fin must be > 0.")
            if num_hollow_per_fin_val > 0:
                if num_fins_val <= 0: raise ValueError("num_hollow_per_fin > 0 requires num_fins > 0.")
                if w_hollow_val <= 1e-9 or h_hollow_val <= 1e-9: raise ValueError(
                    "If num_hollow_per_fin > 0, w_hollow and h_hollow must be > 0.")
                if w_hollow_val >= t_fin_val or h_hollow_val >= h_fin_val:
                    print(
                        f"Warning: Hollow dimensions ({w_hollow_val}, {h_hollow_val}) might be too large for fin ({t_fin_val}, {h_fin_val}).")
        except (KeyError, ValueError, TypeError) as e:
            return jsonify({'status': 'Error', 'message': f'Invalid fin parameters: {e}'}), 400

        fin_params_for_core = {
            'h_fin': h_fin_val, 't_fin': t_fin_val, 'num_fins': num_fins_val,
            'w_hollow': w_hollow_val, 'h_hollow': h_hollow_val, 'num_hollow_per_fin': num_hollow_per_fin_val
        }
        print(f"Parsed fin_params for core: {fin_params_for_core}")

        # --- Get Environmental Parameters ---
        environment_data = data.get('environment_params')
        if not environment_data or not isinstance(environment_data, dict):
            return jsonify({'status': 'Error', 'message': 'Missing or invalid environment_params'}), 400
        try:
            t_ambient_inlet_val = float(environment_data['t_ambient_inlet'])
            q_total_m3_h_val = float(environment_data['Q_total_m3_h'])
            if q_total_m3_h_val <= 1e-9: raise ValueError("Air flow rate (Q) must be positive (> 0).")
        except (KeyError, ValueError, TypeError) as e:
            return jsonify({'status': 'Error', 'message': f'Invalid environment parameters: {e}'}), 400

        # --- Get Powers ---
        powers_data = data.get('powers')
        current_chip_powers = {}
        if not powers_data or not isinstance(powers_data, dict):
            print("Warning: No power data ('powers') received. Assuming 0W for all chips.")
        else:
            for chip_id, power_str in powers_data.items():
                try:
                    current_chip_powers[chip_id] = max(0.0, float(power_str))
                except (ValueError, TypeError):
                    current_chip_powers[chip_id] = 0.0

        # --- Get Module Definitions ---
        module_definitions_data = data.get('module_definitions')
        if module_definitions_data is None or not isinstance(module_definitions_data, list):
            return jsonify({'status': 'Error', 'message': 'Module definitions missing or invalid'}), 400
        validated_module_defs = []
        for mod_def in module_definitions_data:
            if isinstance(mod_def, dict) and all(k in mod_def for k in ['id', 'center_x', 'center_y']):
                try:
                    validated_module_defs.append({
                        'id': str(mod_def['id']),
                        'center_x': float(mod_def['center_x']),
                        'center_y': float(mod_def['center_y'])
                    })
                except (ValueError, TypeError, KeyError):
                    print(f"Warning: Invalid data in module def {mod_def}")
            else:
                print(f"Warning: Module def with invalid structure: {mod_def}")

        # --- Parámetros adicionales ---
        nz_base_sim_val = 1
        if t_base_val < 0.002 and nz_base_sim_val > 1:
            print(
                f"Nota: Base del disipador muy delgada (t={t_base_val * 1000}mm). nz_base={nz_base_sim_val} podría ser excesivo.")

        assumed_duct_height_for_h_calc_val = 0.05
        if h_fin_val > 0:
            assumed_duct_height_for_h_calc_val = h_fin_val + 0.005
        print(f"assumed_duct_height_for_h_calc_val = {assumed_duct_height_for_h_calc_val:.4f} m")

        # --- Llamada a la función maestra ---
        print(">>> update_simulation LLAMANDO A run_simulation_with_ntc_compensation")
        start_core_time = time.time()
        results_from_core = run_simulation_with_ntc_compensation(
            max_h_iterations=2, h_convergence_tolerance=10,
            lx_base_h_calc=lx_val, ly_base_h_calc=ly_val, q_total_m3_h_h_calc=q_total_m3_h_val,
            t_ambient_inlet_h_calc=t_ambient_inlet_val, assumed_duct_height_h_calc=assumed_duct_height_for_h_calc_val,
            k_heatsink_material_h_calc=k_base_val, fin_params_h_calc=fin_params_for_core,
            rth_heatsink_manual_val_h_calc=rth_heatsink_manual_val, # Pass the manual Rth value
            use_manual_rth_h_calc=use_manual_rth_val, # Pass the checkbox state
            specific_chip_powers_sim=current_chip_powers, lx_sim=lx_val, ly_sim=ly_val,
            t_sim_base_fdm=t_base_val, module_definitions_sim=validated_module_defs,
            nz_base_sim=nz_base_sim_val
        )
        end_core_time = time.time()

        # --- Procesamiento de Resultados ---
        serializable_results = make_serializable(results_from_core)

        if 'T_solution_matrix' in serializable_results:
            serializable_results['temperature_matrix'] = serializable_results.pop('T_solution_matrix')
        if 'x_coordinates_vector' in serializable_results:
            serializable_results['x_coordinates'] = serializable_results.pop('x_coordinates_vector')
        if 'y_coordinates_vector' in serializable_results:
            serializable_results['y_coordinates'] = serializable_results.pop('y_coordinates_vector')

        if 'sim_params_dict' in serializable_results and isinstance(serializable_results.get('sim_params_dict'), dict):
            sim_p = serializable_results.pop('sim_params_dict')
            serializable_results['sim_lx'] = sim_p.get('lx', lx_val)
            serializable_results['sim_ly'] = sim_p.get('ly', ly_val)
            serializable_results['sim_nx'] = sim_p.get('nx')
            serializable_results['sim_ny'] = sim_p.get('ny')
        else:
            serializable_results['sim_lx'] = lx_val
            serializable_results['sim_ly'] = ly_val

        core_sim_time = round(end_core_time - start_core_time, 2)
        if 'simulation_time_s' not in serializable_results:
            serializable_results['simulation_time_s'] = core_sim_time

        serializable_results['report_generated_utc'] = datetime.datetime.utcnow().isoformat() + 'Z'
        serializable_results['request_processing_time_s'] = round(time.time() - start_request_time, 2)

        # +++ INICIO DE LA MODIFICACIÓN CLAVE +++
        # Empaquetar los inputs validados que se usaron en la simulación
        # para devolverlos al frontend. Esto asegura que el informe PDF use los datos correctos.
        if serializable_results.get('status', '').startswith('Success'):
            validated_inputs_for_report = {
                'heatsink_params': {
                    'lx': lx_val, 'ly': ly_val, 't': t_base_val, 'k_base': k_base_val,
                    'rth_heatsink_manual': rth_heatsink_manual_val, # Manual Rth
                    'use_manual_rth': use_manual_rth_val, # Checkbox state
                    'h_fin': h_fin_val, 't_fin': t_fin_val,
                    'num_fins': num_fins_val, 'w_hollow': w_hollow_val, 'h_hollow': h_hollow_val,
                    'num_hollow_per_fin': num_hollow_per_fin_val
                },
                'environment_params': {
                    't_ambient_inlet': t_ambient_inlet_val,
                    'Q_total_m3_h': q_total_m3_h_val
                },
                'module_definitions': validated_module_defs,
                'powers': current_chip_powers
            }
            serializable_results['simulation_inputs'] = validated_inputs_for_report
        # +++ FIN DE LA MODIFICACIÓN CLAVE +++

        print(">>> update_simulation PREPARANDO RESPUESTA")
        return jsonify(serializable_results)

    except Exception as e:
        import traceback
        error_id = str(time.time())
        print(f"!!! CRITICAL ERROR in /update_simulation (ID: {error_id}): {str(e)}")
        traceback.print_exc()
        return jsonify({'status': 'Error', 'message': f'Internal server error (ID: {error_id}). Check logs.'}), 500


# +++ INICIO: NUEVO ENDPOINT PARA EL INFORME PDF +++
@app.route('/generate_report', methods=['POST'])
def generate_report():
    print(">>> /generate_report endpoint hit")
    try:
        # Recibir todos los datos de la simulación (inputs + outputs)
        data = request.get_json()
        if not data or 'inputs' not in data or 'outputs' not in data:
            return jsonify({'status': 'Error', 'message': 'Missing data for report generation.'}), 400

        # Crear el contenido del PDF usando la función definida
        pdf_content = create_pdf_report(data)

        # Devolver el PDF como respuesta de archivo
        return Response(
            pdf_content,
            mimetype="application/pdf",
            headers={"Content-Disposition": "attachment;filename=thermal_report.pdf"}
        )

    except Exception as e:
        import traceback
        error_id = str(time.time())
        print(f"!!! CRITICAL ERROR in /generate_report (ID: {error_id}): {str(e)}")
        traceback.print_exc()
        return jsonify({'status': 'Error', 'message': f'Internal server error generating PDF (ID: {error_id}).'}), 500


# +++ FIN: NUEVO ENDPOINT PARA EL INFORME PDF +++


@app.route('/view_pdf')
def view_pdf():

    print(f"Request to view PDF: {PDF_FILENAME}")
    try:
        return send_from_directory(PDF_FOLDER, PDF_FILENAME, as_attachment=False)
    except FileNotFoundError:
        logging.error(f"PDF file not found at: {os.path.join(PDF_FOLDER, PDF_FILENAME)}");
        return "Error: PDF file not found on server.", 404


@app.route('/creator_info.html')
def creator_info_page():

    print("Accessing creator info page ('/creator_info.html')")

    return render_template('creator_info.html')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)  # Configurar logging básico
    # logging.getLogger('matplotlib').setLevel(logging.WARNING) # Reducir verbosidad de Matplotlib
    print("Starting Flask server (Dynamic Mode with 'h' iteration)...")
    app.run(debug=True, host='0.0.0.0', port=5000)

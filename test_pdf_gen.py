import pdf_generator
import os

# Dummy data structure similar to what `app.py` passes
dummy_data = {
    'inputs': {
        'heatsink_params': {
            'lx': 0.25, 'ly': 0.35, 't': 0.02, 'k_base': 218.0,
            'h_fin': 0.112, 't_fin': 0.005, 'num_fins': 31,
            'use_manual_rth': False
        },
        'environment_params': {
            't_ambient_inlet': 40.0,
            'Q_total_m3_h': 750.0
        },
        'module_definitions': [
            {'id': 'Mod1', 'center_x': 0.1, 'center_y': 0.1},
            {'id': 'Mod2', 'center_x': 0.2, 'center_y': 0.2}
        ],
        'powers': {
            'Mod1_IGBT1': 100, 'Mod1_Diode1': 50,
            'Mod2_IGBT1': 120, 'Mod2_Diode1': 60
        }
    },
    'outputs': {
        'status': 'Success',
        'convergence': True,
        'rth_heatsink_calculated': 0.05,
        't_max_base': 85.5,
        't_max_junction': 110.2,
        't_max_junction_chip': 'Mod2_IGBT1',
        't_max_ntc': 90.0,
        't_air_outlet': 55.0,
        # 'plot_base_data_uri': '...', # Omitted for basic layout check
        # 'plot_zoom_data_uri': '...'
    }
}

try:
    print("Generating PDF...")
    pdf_bytes = pdf_generator.create_pdf_report(dummy_data)
    with open("test_report.pdf", "wb") as f:
        f.write(pdf_bytes)
    print("PDF generated successfully: test_report.pdf")

    # Optional: Check file size
    size = os.path.getsize("test_report.pdf")
    print(f"File size: {size} bytes")
    if size < 1000:
        print("Warning: File size seems too small.")

except Exception as e:
    print(f"FAILED to generate PDF: {e}")
    exit(1)

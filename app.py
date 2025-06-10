# app.py
from flask import Flask, render_template, request, jsonify, Response, url_for, session, send_from_directory
import numpy as np
import os
import time
import json
import datetime
import logging

# IMPORTAR LA NUEVA FUNCIÓN MAESTRA Y OTRAS NECESARIAS
from simulador_core import run_simulation_with_h_iteration, chip_rel_positions

# calculate_h_convection_detailed ya no se llama directamente desde app.py
# run_thermal_simulation tampoco se llama directamente desde app.py para el flujo principal

app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Default Configuration (sin cambios) ---
DEFAULT_HEATSINK_PARAMS = {
    'lx': 0.25, 'ly': 0.35, 't': 0.02,  # t es espesor de la base
    'k_base': 218.0,
    'rth_heatsink': 0.015,  # Rth global, usado como fallback o para caso sin módulos
    'h_fin': 0.035, 't_fin': 0.0015, 'num_fins': 20,
    'w_hollow': 0.0, 'h_hollow': 0.0, 'num_hollow_per_fin': 0
}
DEFAULT_ENVIRONMENT_PARAMS = {
    't_ambient_inlet': 40.0,
    'Q_total_m3_h': 1000.0
}
static_images_dir = os.path.join(app.static_folder, 'images')
os.makedirs(static_images_dir, exist_ok=True)
PDF_FOLDER = os.path.join(app.root_path, 'static')
PDF_FILENAME = 'thermal_simulator_intro.pdf'


# --- Helper Function for JSON Serialization (sin cambios) ---
def make_serializable(data):
    # ... (código make_serializable sin cambios)
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
        json.dumps(data);  # Verificar si es serializable por json directamente
        return data
    except TypeError:
        return str(data)  # Último recurso


@app.route('/')
def index():
    # ... (sin cambios)
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
            rth_heatsink_val = float(heatsink_data['rth_heatsink'])
            if not (lx_val > 0 and ly_val > 0 and t_base_val > 0 and k_base_val > 1e-9 and rth_heatsink_val > 1e-9):
                raise ValueError("Heatsink base parameters must be positive (> 0).")
        except (KeyError, ValueError, TypeError) as e:
            return jsonify({'status': 'Error', 'message': f'Invalid heatsink base parameters: {e}'}), 400

        try:
            h_fin_val = float(heatsink_data.get('h_fin', DEFAULT_HEATSINK_PARAMS['h_fin']))
            t_fin_val = float(heatsink_data.get('t_fin', DEFAULT_HEATSINK_PARAMS['t_fin']))
            num_fins_val = int(heatsink_data.get('num_fins', DEFAULT_HEATSINK_PARAMS['num_fins']))
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

        # --- Parámetros adicionales para el cálculo de 'h' y simulación ---
        # nz_base_sim_val: Número de capas en Z para el FDM de la base del disipador.
        # nz_base_sim_val = 1 para 2.5D (base como una sola capa con espesor t_base_val)
        # nz_base_sim_val > 1 para 3D (base dividida en nz_base_sim_val capas)
        nz_base_sim_val = 1  # O tomarlo de la UI si se añade un control para esto. Por ahora, fijo a 5 para 3D.
        # Si se quiere 2.5D por defecto, poner 1.
        if t_base_val < 0.002 and nz_base_sim_val > 1:  # Si la base es muy delgada, 3D puede no ser necesario.
            print(
                f"Nota: Base del disipador muy delgada (t={t_base_val * 1000}mm). nz_base={nz_base_sim_val} podría ser excesivo.")
            # nz_base_sim_val = 1 # Podría forzarse a 2.5D

        # Altura del ducto para el cálculo de h.
        # Esta es una simplificación importante. Si hay aletas, el flujo se canaliza por ellas.
        # Si no hay aletas, es un ducto sobre la placa base.
        assumed_duct_height_for_h_calc_val = 0.05  # Valor por defecto si no hay aletas (5cm)
        if h_fin_val > 0:  # Si hay aletas, la altura del ducto podría relacionarse con h_fin
            assumed_duct_height_for_h_calc_val = h_fin_val + 0.005  # e.g. 5mm clearance over fins
            # O simplemente h_fin_val si las aletas llenan el ducto.
        print(f"assumed_duct_height_for_h_calc_val = {assumed_duct_height_for_h_calc_val:.4f} m")

        # --- Llamada a la nueva función maestra en simulador_core.py ---
        print(">>> update_simulation LLAMANDO A run_simulation_with_h_iteration")
        start_core_time = time.time()
        results_from_core = run_simulation_with_h_iteration(
            max_h_iterations=10,  # Número de iteraciones para convergencia de 'h'
            h_convergence_tolerance=0.5,  # Tolerancia para 'h' en W/m^2.K
            # Args para calculate_h_convection_detailed
            lx_base_h_calc=lx_val,
            ly_base_h_calc=ly_val,
            q_total_m3_h_h_calc=q_total_m3_h_val,
            t_ambient_inlet_h_calc=t_ambient_inlet_val,
            assumed_duct_height_h_calc=assumed_duct_height_for_h_calc_val,
            k_heatsink_material_h_calc=k_base_val,  # Usar k_base del UI
            fin_params_h_calc=fin_params_for_core,
            rth_heatsink_fallback_h_calc=rth_heatsink_val,  # Para el fallback de h
            # Args para run_thermal_simulation (dentro del iterador de h)
            specific_chip_powers_sim=current_chip_powers,
            lx_sim=lx_val,
            ly_sim=ly_val,
            t_sim_base_fdm=t_base_val,  # Espesor de la base para el FDM
            module_definitions_sim=validated_module_defs,
            # nx_sim, ny_sim se usarán los por defecto de simulador_core
            nz_base_sim=nz_base_sim_val  # Número de capas en Z para FDM de la base
        )
        end_core_time = time.time()

        # --- Procesamiento de Resultados (similar a antes) ---
        serializable_results = make_serializable(results_from_core)

        # Renombrar claves si es necesario para consistencia con el frontend (JS)
        if 'T_solution_matrix' in serializable_results:  # Esta es la superficie para el plot interactivo
            serializable_results['temperature_matrix'] = serializable_results.pop('T_solution_matrix')
        if 'x_coordinates_vector' in serializable_results:
            serializable_results['x_coordinates'] = serializable_results.pop('x_coordinates_vector')
        if 'y_coordinates_vector' in serializable_results:
            serializable_results['y_coordinates'] = serializable_results.pop('y_coordinates_vector')

        # Añadir sim_params si no vienen del core (aunque deberían)
        if 'sim_params_dict' in serializable_results and isinstance(serializable_results.get('sim_params_dict'), dict):
            sim_p = serializable_results.pop('sim_params_dict')
            serializable_results['sim_lx'] = sim_p.get('lx', lx_val)
            serializable_results['sim_ly'] = sim_p.get('ly', ly_val)
            serializable_results['sim_nx'] = sim_p.get('nx')  # Dejar que el core lo reporte
            serializable_results['sim_ny'] = sim_p.get('ny')
        else:  # Fallback si sim_params_dict no está
            serializable_results['sim_lx'] = lx_val
            serializable_results['sim_ly'] = ly_val
            # nx, ny no se conocen aquí si no los devuelve el core

        core_sim_time = round(end_core_time - start_core_time, 2)
        if 'simulation_time_s' not in serializable_results:  # El iterador de 'h' puede no reportar esto
            serializable_results['simulation_time_s'] = core_sim_time

        serializable_results['report_generated_utc'] = datetime.datetime.utcnow().isoformat() + 'Z'
        serializable_results['request_processing_time_s'] = round(time.time() - start_request_time, 2)

        print(">>> update_simulation PREPARANDO RESPUESTA (después de iteración de 'h')")
        return jsonify(serializable_results)

    except Exception as e:
        import traceback
        error_id = str(time.time())
        print(f"!!! CRITICAL ERROR in /update_simulation (ID: {error_id}): {str(e)}")
        traceback.print_exc()
        return jsonify({'status': 'Error', 'message': f'Internal server error (ID: {error_id}). Check logs.'}), 500


@app.route('/view_pdf')
def view_pdf():
    # ... (sin cambios)
    print(f"Request to view PDF: {PDF_FILENAME}")
    try:
        return send_from_directory(PDF_FOLDER, PDF_FILENAME, as_attachment=False)
    except FileNotFoundError:
        logging.error(f"PDF file not found at: {os.path.join(PDF_FOLDER, PDF_FILENAME)}");
        return "Error: PDF file not found on server.", 404


@app.route('/creator_info')
def creator_info_page():
    # ... (sin cambios)
    print("Accessing creator info page ('/creator_info')")
    return render_template('creator_info.html')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)  # Configurar logging básico
    # logging.getLogger('matplotlib').setLevel(logging.WARNING) # Reducir verbosidad de Matplotlib
    print("Starting Flask server (Dynamic Mode with 'h' iteration)...")
    app.run(debug=True, host='0.0.0.0', port=5000)

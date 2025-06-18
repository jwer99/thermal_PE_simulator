# app.py

from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import os
import time
import json
import datetime
import logging

# Importar las funciones del core
from simulador_core import (
    calculate_h_effective_and_components,
    run_thermal_simulation,
    chip_rel_positions
)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Configuraciones por Defecto ---
DEFAULT_HEATSINK_PARAMS = {
    'lx': 0.25, 'ly': 0.35, 't': 0.02,
    'k_base': 218.0,
    'rth_heatsink': 0.015,
    'h_fin': 0.035, 't_fin': 0.0015, 'num_fins': 20,
    'flow_maldistribution_factor': 0.9,
    'R_contact_base_fin_per_fin': 0.0005,
    'heatsink_emissivity': 0.85
}
DEFAULT_ENVIRONMENT_PARAMS = {
    't_ambient_inlet': 40.0,
    'Q_total_m3_h': 1000.0
}
PDF_FOLDER = os.path.join(app.root_path, 'static')
PDF_FILENAME = 'thermal_simulator_intro.pdf'


# --- Helper Function for JSON Serialization (CORREGIDA) ---
def make_serializable(data):
    """
    Convierte recursivamente los tipos de datos (especialmente de NumPy) a formatos
    compatibles con JSON.
    """
    if isinstance(data, dict):
        return {k: make_serializable(v) for k, v in data.items()}
    if isinstance(data, list):
        return [make_serializable(item) for item in data]
    # --- INICIO DE LA CORRECCIÓN ---
    # Añadido para manejar booleanos de NumPy. Debe ir ANTES de la comprobación de enteros.
    if isinstance(data, np.bool_):
        return bool(data)
    # --- FIN DE LA CORRECCIÓN ---
    if isinstance(data, (np.floating, float)):
        return None if np.isnan(data) or np.isinf(data) else float(data)
    if isinstance(data, (np.integer, int)):
        return int(data)
    if isinstance(data, np.ndarray):
        return make_serializable(data.tolist())
    if isinstance(data, (datetime.datetime, datetime.date)):
        return data.isoformat()
    return data


@app.route('/')
def index():
    return render_template('index.html',
                           initial_heatsink=DEFAULT_HEATSINK_PARAMS,
                           initial_environment=DEFAULT_ENVIRONMENT_PARAMS,
                           initial_results=None)


@app.route('/update_simulation', methods=['POST'])
def update_simulation():
    print(">>> update_simulation INICIADO (con bucle de iteración de 'h' explícito)")
    start_request_time = time.time()
    try:
        data = request.get_json()

        # --- Obtener Parámetros del Disipador ---
        heatsink_data = data.get('heatsink_params', {})
        lx_val = float(heatsink_data.get('lx', DEFAULT_HEATSINK_PARAMS['lx']))
        ly_val = float(heatsink_data.get('ly', DEFAULT_HEATSINK_PARAMS['ly']))
        t_base_val = float(heatsink_data.get('t', DEFAULT_HEATSINK_PARAMS['t']))
        k_base_val = float(heatsink_data.get('k_base', DEFAULT_HEATSINK_PARAMS['k_base']))
        h_fin_val = float(heatsink_data.get('h_fin', DEFAULT_HEATSINK_PARAMS['h_fin']))
        t_fin_val = float(heatsink_data.get('t_fin', DEFAULT_HEATSINK_PARAMS['t_fin']))
        num_fins_val = int(heatsink_data.get('num_fins', DEFAULT_HEATSINK_PARAMS['num_fins']))
        fin_params_for_core = {'h_fin': h_fin_val, 't_fin': t_fin_val, 'num_fins': num_fins_val}
        flow_maldist_factor_val = float(
            heatsink_data.get('flow_maldistribution_factor', DEFAULT_HEATSINK_PARAMS['flow_maldistribution_factor']))
        r_contact_per_fin_val = float(
            heatsink_data.get('R_contact_base_fin_per_fin', DEFAULT_HEATSINK_PARAMS['R_contact_base_fin_per_fin']))
        emissivity_val = float(heatsink_data.get('heatsink_emissivity', DEFAULT_HEATSINK_PARAMS['heatsink_emissivity']))

        # --- Obtener Parámetros Ambientales y de Potencia ---
        environment_data = data.get('environment_params', {})
        t_ambient_inlet_val = float(
            environment_data.get('t_ambient_inlet', DEFAULT_ENVIRONMENT_PARAMS['t_ambient_inlet']))
        q_total_m3_h_val = float(environment_data.get('Q_total_m3_h', DEFAULT_ENVIRONMENT_PARAMS['Q_total_m3_h']))
        powers_data = data.get('powers', {})
        current_chip_powers = {chip_id: max(0.0, float(power_str)) for chip_id, power_str in powers_data.items()}
        validated_module_defs = data.get('module_definitions', [])

        # --- Flujo de ejecución explícito con iteración de 'h' ---
        max_h_iterations = 3
        h_convergence_tolerance = 0.5
        t_surface_avg_estimate = t_ambient_inlet_val + 15.0
        h_eff_calculated = -1.0
        final_results = {}
        converged_h = False

        print("\n--- INICIANDO BUCLE DE ITERACIÓN 'h' <-> 'T_superficie' ---")
        for i in range(max_h_iterations):
            print(f"--- Iteración de 'h' #{i + 1}/{max_h_iterations} ---")
            print(f"Usando T_superficie_promedio_estimada = {t_surface_avg_estimate:.2f} °C para calcular 'h'")

            is_finned_heatsink = fin_params_for_core['num_fins'] > 0
            if not is_finned_heatsink:
                print("ADVERTENCIA: Modelo de placa plana no implementado en este flujo. Usando un h_eff fijo.")
                h_to_use_in_sim = 50.0
            else:
                h_to_use_in_sim = calculate_h_effective_and_components(
                    lx_base=lx_val, ly_base=ly_val, q_total_m3_h=q_total_m3_h_val,
                    t_ambient_inlet=t_ambient_inlet_val, k_heatsink_material=k_base_val,
                    fin_params=fin_params_for_core,
                    t_surface_avg_estimate=t_surface_avg_estimate,
                    flow_maldistribution_factor=flow_maldist_factor_val,
                    R_contact_base_fin_per_fin=r_contact_per_fin_val,
                    heatsink_emissivity=emissivity_val,
                    t_surroundings_rad=t_ambient_inlet_val
                )

            if h_eff_calculated > 0 and abs(h_to_use_in_sim - h_eff_calculated) < h_convergence_tolerance:
                print(f"CONVERGENCIA DE 'h' ALCANZADA.")
                converged_h = True
                break
            h_eff_calculated = h_to_use_in_sim

            print(f"Ejecutando simulación FDM con h_eff = {h_eff_calculated:.2f} W/m^2.K")
            results_this_iter = run_thermal_simulation(
                specific_chip_powers=current_chip_powers, lx=lx_val, ly=ly_val, t=t_base_val,
                rth_heatsink=1.0 / (h_eff_calculated * lx_val * ly_val),
                module_definitions=validated_module_defs,
                t_ambient_inlet_arg=t_ambient_inlet_val, q_total_m3_h_arg=q_total_m3_h_val,
                h_eff_FDM_heatsink_arg=h_eff_calculated,
                nx=50, ny=75, nz_base=1
            )
            final_results = results_this_iter

            if not final_results.get('status', 'Error').startswith('Success'):
                print("ERROR: Simulación FDM falló. Abortando.")
                return jsonify(make_serializable(final_results))

            new_t_surface_avg = final_results.get('t_avg_base')
            if new_t_surface_avg is not None and not np.isnan(new_t_surface_avg):
                relaxation_factor = 0.7
                t_surface_avg_estimate = (relaxation_factor * new_t_surface_avg) + (
                            (1 - relaxation_factor) * t_surface_avg_estimate)
                print(
                    f"T_superficie promedio resultante: {new_t_surface_avg:.2f}°C. Nueva estimación para 'h': {t_surface_avg_estimate:.2f}°C")
            else:
                print("ADVERTENCIA: No se pudo obtener t_avg_base.")
                break

        # --- Procesamiento de Resultados Finales ---
        final_results['h_eff_converged_value'] = h_eff_calculated
        final_results['h_iterations'] = i + 1
        final_results['h_converged_status'] = converged_h

        serializable_results = make_serializable(final_results)

        if 'T_solution_matrix' in serializable_results:
            serializable_results['temperature_matrix'] = serializable_results.pop('T_solution_matrix')
        if 'x_coordinates_vector' in serializable_results:
            serializable_results['x_coordinates'] = serializable_results.pop('x_coordinates_vector')
        if 'y_coordinates_vector' in serializable_results:
            serializable_results['y_coordinates'] = serializable_results.pop('y_coordinates_vector')

        if 'sim_params_dict' in serializable_results:
            params = serializable_results.pop('sim_params_dict')
            serializable_results['sim_lx'] = params.get('lx')
            serializable_results['sim_ly'] = params.get('ly')
            serializable_results['sim_nx'] = params.get('nx')
            serializable_results['sim_ny'] = params.get('ny')
            print(f"Añadidos al JSON para el frontend: sim_lx, sim_ly, sim_nx, sim_ny")

        serializable_results['report_generated_utc'] = datetime.datetime.now(datetime.UTC).isoformat()
        serializable_results['simulation_time_s'] = round(time.time() - start_request_time, 2)

        print(">>> update_simulation PREPARANDO RESPUESTA FINAL")
        print(f"Claves finales enviadas al frontend: {list(serializable_results.keys())}")

        return jsonify(serializable_results)

    except Exception as e:
        import traceback
        error_id = str(time.time())
        print(f"!!! CRITICAL ERROR in /update_simulation (ID: {error_id}): {str(e)}")
        traceback.print_exc()
        return jsonify({'status': 'Error', 'message': f'Internal server error (ID: {error_id}). Check logs.'}), 500


@app.route('/view_pdf')
def view_pdf():
    return send_from_directory(PDF_FOLDER, PDF_FILENAME, as_attachment=False)


@app.route('/creator_info')
def creator_info_page():
    return render_template('creator_info.html')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Starting Flask server (compatible con modelo de 'h' avanzado)...")
    app.run(debug=True, host='0.0.0.0', port=5000)

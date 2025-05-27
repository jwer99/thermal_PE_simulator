# app.py
from flask import Flask, render_template, request, jsonify, Response, url_for, session, send_from_directory
import numpy as np
import os
import time
# from fpdf import FPDF # Can be removed if not used for reports here
import json
import datetime
import logging

# Assuming simulador_core and chip_rel_positions are correctly imported
from simulador_core import run_thermal_simulation, chip_rel_positions
from eulerian_fluid_simulator import run_1d_eulerian_simulation # New Import

app = Flask(__name__)
app.secret_key = os.urandom(24) # Needed for sessions if used

# --- Default Configuration ---
DEFAULT_HEATSINK_PARAMS = {
    'lx': 0.25, 'ly': 0.35, 't': 0.02,
    'k_base': 218.0,    # W/mK (Aluminum)
    'rth_heatsink': 0.015 # °C/W
}
DEFAULT_ENVIRONMENT_PARAMS = {
    't_ambient_inlet': 40.0, # °C
    'Q_total_m3_h': 1000.0   # m³/h
}

static_images_dir = os.path.join(app.static_folder, 'images')
os.makedirs(static_images_dir, exist_ok=True)
PDF_FOLDER = os.path.join(app.root_path, 'static')
PDF_FILENAME = 'thermal_simulator_intro.pdf' # Keep filename or rename if needed
# --- End Configuration ---

# --- Helper Function for JSON Serialization ---
def make_serializable(data):
    # (Keep this function as is - it handles data types, not text)
    if isinstance(data, dict):
        return {k: make_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [make_serializable(item) for item in data]
    elif isinstance(data, (np.floating, float)):
        if np.isnan(data) or np.isinf(data):
            return None
        return float(data)
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

# --- Flask Routes ---
@app.route('/')
def index():
    print("Accessing main route ('/') - Dynamic Mode") # Changed
    return render_template('index.html',
                           initial_heatsink=DEFAULT_HEATSINK_PARAMS,
                           initial_environment=DEFAULT_ENVIRONMENT_PARAMS,
                           initial_results=None
                           )

@app.route('/update_simulation', methods=['POST'])
def update_simulation():
    print("Received POST request at /update_simulation (Dynamic)") # Changed
    start_request_time = time.time()
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'Error', 'message': 'No JSON data received'}), 400

        # --- Get Heatsink Parameters ---
        heatsink_data = data.get('heatsink_params')
        if not heatsink_data or not isinstance(heatsink_data, dict):
            return jsonify({'status': 'Error', 'message': 'Missing heatsink parameters (heatsink_params)'}), 400 # Changed
        try:
            lx = float(heatsink_data['lx'])
            ly = float(heatsink_data['ly'])
            t = float(heatsink_data['t'])
            k_base = float(heatsink_data['k_base'])
            rth_heatsink = float(heatsink_data['rth_heatsink'])
            if lx <= 0 or ly <= 0 or t <= 0 or k_base <= 1e-9 or rth_heatsink <= 1e-9:
                raise ValueError("Heatsink parameters must be positive (> 0).") # Changed
        except (KeyError, ValueError, TypeError) as e:
             return jsonify({'status': 'Error', 'message': f'Invalid heatsink parameters: {e}'}), 400 # Changed

        # --- Get Environmental Parameters ---
        environment_data = data.get('environment_params')
        if not environment_data or not isinstance(environment_data, dict):
            return jsonify({'status': 'Error', 'message': 'Missing environment parameters (environment_params)'}), 400 # Changed
        try:
            t_ambient_inlet = float(environment_data['t_ambient_inlet'])
            Q_total_m3_h = float(environment_data['Q_total_m3_h'])
            if Q_total_m3_h <= 1e-9:
                 raise ValueError("Air flow rate (Q) must be positive (> 0).") # Changed
        except (KeyError, ValueError, TypeError) as e:
             return jsonify({'status': 'Error', 'message': f'Invalid environment parameters: {e}'}), 400 # Changed

        # --- Get Powers ---
        powers_data = data.get('powers')
        if not powers_data or not isinstance(powers_data, dict):
             print("Warning: No power data ('powers') received or not a dictionary. Assuming 0W.") # Changed
             current_chip_powers = {}
        else:
            current_chip_powers = {}
            for chip_id, power_str in powers_data.items():
                try:
                    power_val = float(power_str)
                    current_chip_powers[chip_id] = max(0.0, power_val)
                except (ValueError, TypeError):
                    print(f"Warning: Invalid power value for {chip_id} ('{power_str}'). Using 0.") # Changed
                    current_chip_powers[chip_id] = 0.0

        # --- Get Module Definitions ---
        module_definitions_data = data.get('module_definitions')
        if module_definitions_data is None or not isinstance(module_definitions_data, list):
             print("Error: Module definitions missing ('module_definitions') or not a list.") # Changed
             return jsonify({'status': 'Error', 'message': 'Module definitions missing or invalid structure'}), 400

        validated_module_defs = []
        module_ids_seen = set()
        for mod_def in module_definitions_data:
            if isinstance(mod_def, dict) and all(k in mod_def for k in ['id', 'center_x', 'center_y']):
                 try:
                     mod_id = str(mod_def['id'])
                     center_x = float(mod_def['center_x'])
                     center_y = float(mod_def['center_y'])
                     validated_module_defs.append({'id': mod_id, 'center_x': center_x, 'center_y': center_y})
                     module_ids_seen.add(mod_id)
                 except (ValueError, TypeError, KeyError) as e:
                      print(f"Warning: Invalid data in module definition {mod_def}. Skipping. Error: {e}") # Changed
            else:
                print(f"Warning: Module definition with invalid structure: {mod_def}. Skipping.") # Changed

        if not validated_module_defs and module_definitions_data:
             print("Warning: None of the submitted module definitions were valid.") # Changed
        elif not validated_module_defs:
             print("Warning: No modules defined for simulation.") # Changed


        # --- Execute Simulation ---
        print(f"Running simulation with {len(validated_module_defs)} valid modules:") # Changed
        print(f"  Heatsink: Lx={lx}, Ly={ly}, t={t}, k={k_base}, Rth={rth_heatsink}")
        print(f"  Environment: T_in={t_ambient_inlet}, Q={Q_total_m3_h}")
        start_sim_time = time.time()

        results = run_thermal_simulation(
            specific_chip_powers=current_chip_powers,
            lx=lx, ly=ly, t=t, k_base=k_base, rth_heatsink=rth_heatsink,
            t_ambient_inlet=t_ambient_inlet, Q_total_m3_h=Q_total_m3_h,
            module_definitions=validated_module_defs
        )
        end_sim_time = time.time()
        sim_time = round(end_sim_time - start_sim_time, 2)
        total_time = round(end_sim_time - start_request_time, 2)
        print(f"Simulation completed in {sim_time}s (Total request: {total_time}s).") # Changed

        # --- Return Results ---
        serializable_results = make_serializable(results)
        if 'simulation_time_s' not in serializable_results:
             serializable_results['simulation_time_s'] = sim_time
        serializable_results['report_generated_utc'] = datetime.datetime.utcnow().isoformat() + 'Z'

        return jsonify(serializable_results)

    except Exception as e:
        import traceback
        error_id = str(time.time())
        print(f"Critical error in /update_simulation (ID: {error_id}): {e}") # Changed
        traceback.print_exc()
        return jsonify({'status': 'Error', 'message': f'Internal server error (ID: {error_id}). Check logs.'}), 500 # Changed

# --- Route for PDF ---
@app.route('/view_pdf')
def view_pdf():
    """Serves the introductory PDF file."""
    print(f"Request to view PDF: {PDF_FILENAME}") # Changed
    try:
        return send_from_directory(PDF_FOLDER, PDF_FILENAME, as_attachment=False)
    except FileNotFoundError:
        logging.error(f"PDF file not found at: {os.path.join(PDF_FOLDER, PDF_FILENAME)}")
        return "Error: PDF file not found on server.", 404 # Changed

# --- NEW ROUTE: Creator Information Page ---
@app.route('/creator_info')
def creator_info_page():
    """Displays the page with creator information and bibliography."""
    print("Accessing creator info page ('/creator_info')") # Changed
    # Simply renders the new HTML template
    return render_template('creator_info.html')
# --- END NEW ROUTE ---


# --- Routes for 1D Eulerian Fluid Simulator ---
@app.route('/fluid_simulation')
def fluid_simulation_page():
    """Renders the 1D Eulerian Fluid Simulator page."""
    print("Accessing 1D Eulerian Fluid Simulator page ('/fluid_simulation')")
    return render_template('fluid_simulator.html')

@app.route('/run_fluid_simulation', methods=['POST'])
def handle_run_fluid_simulation():
    """Handles the request to run the 1D Eulerian fluid simulation."""
    print("Received POST request at /run_fluid_simulation")
    start_request_time = time.time()
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'Error', 'message': 'No JSON data received'}), 400

        # Extract parameters
        sim_params = {
            'domain_length_m': float(data.get('domain_length_m', 1.0)),
            'num_cells': int(data.get('num_cells', 100)),
            'total_sim_time_s': float(data.get('total_sim_time_s', 0.1)),
            'initial_density_kg_m3': float(data.get('initial_density_kg_m3', 1.225)),
            'initial_velocity_m_s': float(data.get('initial_velocity_m_s', 0.0)),
            'initial_pressure_Pa': float(data.get('initial_pressure_Pa', 101325.0)),
            'boundary_condition_type': str(data.get('boundary_condition_type', 'transmissive')),
            'cfl_number': float(data.get('cfl_number', 0.5)),
            'output_time_steps': int(data.get('output_time_steps', 10))
        }
        
        # Basic validation for critical parameters (can be expanded)
        if sim_params['num_cells'] <= 0:
            return jsonify({'status': 'Error', 'message': 'Number of cells must be positive.'}), 400
        if sim_params['domain_length_m'] <= 0:
            return jsonify({'status': 'Error', 'message': 'Domain length must be positive.'}), 400
        # output_time_steps in run_1d_eulerian_simulation is handled to be >= 1
        
        print(f"Running 1D Eulerian simulation with params: {sim_params}")
        
        results = run_1d_eulerian_simulation(**sim_params)
        
        end_request_time = time.time()
        total_time = round(end_request_time - start_request_time, 3)
        print(f"Eulerian simulation request processed in {total_time}s.")

        # Ensure results are JSON serializable (run_1d_eulerian_simulation should already handle numpy types within its dict)
        # The make_serializable can be used if there are still direct numpy types at this level,
        # but the current run_1d_eulerian_simulation is designed to return a dict with basic types or lists of basic types.
        return jsonify(results)

    except ValueError as ve: # Catch specific errors from float/int conversion or direct validation
        error_id = str(time.time())
        print(f"ValueError in /run_fluid_simulation (ID: {error_id}): {ve}")
        return jsonify({'status': 'Error', 'message': f'Invalid parameter value: {ve} (ID: {error_id})'}), 400
    except Exception as e:
        import traceback
        error_id = str(time.time())
        print(f"Critical error in /run_fluid_simulation (ID: {error_id}): {e}")
        traceback.print_exc()
        # Ensure the response from run_1d_eulerian_simulation is used if it's an error dict itself
        if isinstance(e, dict) and 'status' in e and e['status'] == 'Error':
            return jsonify(e), 500 # Or an appropriate error code from the simulation
        return jsonify({'status': 'Error', 'message': f'Internal server error during fluid simulation (ID: {error_id}). Check logs.'}), 500
# --- END Routes for 1D Eulerian Fluid Simulator ---


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Starting Flask server (Dynamic Mode)...") # Changed
    app.run(debug=True, host='0.0.0.0', port=5000)
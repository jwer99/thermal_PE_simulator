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
from simulador_core import run_thermal_simulation, chip_rel_positions, calculate_heatsink_contact_area # simulador_core V2
from eulerian_fluid_simulator import run_1d_eulerian_simulation  # New Import

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Needed for sessions if used

# --- Default Configuration ---
# These defaults are for the UI, the actual values will be passed to the simulation
DEFAULT_HEATSINK_PARAMS = {
    'lx': 0.25, 'ly': 0.35, 't_base': 0.02, # 't_base' is the key used by the frontend for heatsink thickness
    'k_base': 218.0,  # W/mK (Aluminum)
    'rth_heatsink': 0.015  # °C/W
}
DEFAULT_ENVIRONMENT_PARAMS = {
    't_ambient_inlet': 40.0,  # °C
    'Q_total_m3_h': 1000.0  # m³/h
}

static_images_dir = os.path.join(app.static_folder, 'images')
os.makedirs(static_images_dir, exist_ok=True)
PDF_FOLDER = os.path.join(app.root_path, 'static')
PDF_FILENAME = 'thermal_simulator_intro.pdf'  # Keep filename or rename if needed


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
    print("Accessing main route ('/') - Dynamic Mode")  # Changed
    return render_template('index.html',
                           initial_heatsink=DEFAULT_HEATSINK_PARAMS,
                           initial_environment=DEFAULT_ENVIRONMENT_PARAMS,
                           initial_results=None
                           )


@app.route('/update_simulation', methods=['POST'])
def update_simulation():
    print("Received POST request at /update_simulation (Dynamic)")  # Changed
    start_request_time = time.time()
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'Error', 'message': 'No JSON data received'}), 400

        # --- Get Heatsink Parameters ---
        heatsink_data = data.get('heatsink_params')
        if not heatsink_data or not isinstance(heatsink_data, dict):
            return jsonify(
                {'status': 'Error', 'message': 'Missing heatsink parameters (heatsink_params)'}), 400  # Changed
        try:
            lx_val = float(heatsink_data['lx'])
            ly_val = float(heatsink_data['ly'])
            # 't_base' is what the frontend sends for heatsink thickness
            t_heatsink_thickness_val = float(heatsink_data['t']) # RENAMED for clarity and matching simulador_core
            k_base_val = float(heatsink_data['k_base'])
            rth_heatsink_val = float(heatsink_data['rth_heatsink'])
            if lx_val <= 0 or ly_val <= 0 or t_heatsink_thickness_val <= 0 or k_base_val <= 1e-9 or rth_heatsink_val <= 1e-9:
                raise ValueError("Heatsink parameters must be positive (> 0).")  # Changed
        except (KeyError, ValueError, TypeError) as e:
            # Ensure to use the correct key 't_base' in error message if it's missing
            if isinstance(e, KeyError) and 't_base' not in heatsink_data and 't' in heatsink_data:
                message = f"Invalid heatsink parameters: Missing 't_base' (heatsink thickness), received 't'. Error: {e}"
            elif isinstance(e, KeyError) and 't_base' not in heatsink_data:
                message = f"Invalid heatsink parameters: Missing 't_base' (heatsink thickness). Error: {e}"
            else:
                message = f'Invalid heatsink parameters: {e}'
            return jsonify({'status': 'Error', 'message': message}), 400

        # --- Get Environmental Parameters ---
        environment_data = data.get('environment_params')
        if not environment_data or not isinstance(environment_data, dict):
            return jsonify(
                {'status': 'Error', 'message': 'Missing environment parameters (environment_params)'}), 400  # Changed
        try:
            t_ambient_inlet_val = float(environment_data['t_ambient_inlet'])
            q_total_m3_h_val = float(environment_data['Q_total_m3_h'])
            if q_total_m3_h_val <= 1e-9:
                raise ValueError("Air flow rate (Q) must be positive (> 0).")  # Changed
        except (KeyError, ValueError, TypeError) as e:
            return jsonify({'status': 'Error', 'message': f'Invalid environment parameters: {e}'}), 400  # Changed

        # --- Get Powers ---
        powers_data = data.get('powers')
        if not powers_data or not isinstance(powers_data, dict):
            print("Warning: No power data ('powers') received or not a dictionary. Assuming 0W.")  # Changed
            current_chip_powers = {}
        else:
            current_chip_powers = {}
            for chip_id, power_str in powers_data.items():
                try:
                    power_val = float(power_str)
                    current_chip_powers[chip_id] = max(0.0, power_val)
                except (ValueError, TypeError):
                    print(f"Warning: Invalid power value for {chip_id} ('{power_str}'). Using 0.")  # Changed
                    current_chip_powers[chip_id] = 0.0

        # --- Get Module Definitions ---
        module_definitions_data = data.get('module_definitions')
        if module_definitions_data is None or not isinstance(module_definitions_data, list):
            print("Error: Module definitions missing ('module_definitions') or not a list.")  # Changed
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
                    print(f"Warning: Invalid data in module definition {mod_def}. Skipping. Error: {e}")  # Changed
            else:
                print(f"Warning: Module definition with invalid structure: {mod_def}. Skipping.")  # Changed

        if not validated_module_defs and module_definitions_data:
            print("Warning: None of the submitted module definitions were valid.")  # Changed
        elif not validated_module_defs:
            print("Warning: No modules defined for simulation.")  # Changed

        # --- Execute Simulation ---
        print(f"Running simulation with {len(validated_module_defs)} valid modules:")  # Changed
        print(f"  Heatsink: Lx={lx_val}, Ly={ly_val}, t_hs={t_heatsink_thickness_val}, k_hs={k_base_val}, Rth_global={rth_heatsink_val}")
        print(f"  Environment: T_in={t_ambient_inlet_val}, Q={q_total_m3_h_val}")
        start_sim_time = time.time()

        # MODIFIED: Call to run_thermal_simulation with updated argument names
        results = run_thermal_simulation(
            specific_chip_powers=current_chip_powers,
            lx=lx_val,
            ly=ly_val,
            t_heatsink_thickness=t_heatsink_thickness_val,       # MATCHES simulador_core V2
            k_main_heatsink_base_arg=k_base_val,                 # MATCHES simulador_core V2
            rth_heatsink=rth_heatsink_val,
            t_ambient_inlet_arg=t_ambient_inlet_val,             # MATCHES simulador_core V2
            q_total_m3_h_arg=q_total_m3_h_val,                   # MATCHES simulador_core V2
            module_definitions=validated_module_defs
            # nx, ny can be passed if you want to control FDM resolution from frontend
            # nx=data.get('nx', Nx_base_default_from_sim_core), # Example
            # ny=data.get('ny', Ny_base_default_from_sim_core)  # Example
        )
        end_sim_time = time.time()
        sim_time = round(end_sim_time - start_sim_time, 2)
        total_time = round(end_sim_time - start_request_time, 2)
        print(f"Simulation completed in {sim_time}s (Total request: {total_time}s).")  # Changed

        # --- Return Results ---
        serializable_results = make_serializable(results)
        if 'simulation_time_s' not in serializable_results:
            serializable_results['simulation_time_s'] = sim_time
        serializable_results['report_generated_utc'] = datetime.datetime.utcnow().isoformat() + 'Z'

        return jsonify(serializable_results)

    except Exception as e:
        import traceback
        error_id = str(time.time())
        print(f"Critical error in /update_simulation (ID: {error_id}): {e}")  # Changed
        traceback.print_exc()
        return jsonify(
            {'status': 'Error', 'message': f'Internal server error (ID: {error_id}). Check logs.'}), 500


# --- Route for PDF ---
@app.route('/view_pdf')
def view_pdf():
    """Serves the introductory PDF file."""
    print(f"Request to view PDF: {PDF_FILENAME}")  # Changed
    try:
        return send_from_directory(PDF_FOLDER, PDF_FILENAME, as_attachment=False)
    except FileNotFoundError:
        logging.error(f"PDF file not found at: {os.path.join(PDF_FOLDER, PDF_FILENAME)}")
        return "Error: PDF file not found on server.", 404  # Changed


# --- NEW ROUTE: Creator Information Page ---
@app.route('/creator_info')
def creator_info_page():
    """Displays the page with creator information and bibliography."""
    print("Accessing creator info page ('/creator_info')")  # Changed
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

    except ValueError as ve:  # Catch specific errors from float/int conversion or direct validation
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
            return jsonify(e), 500  # Or an appropriate error code from the simulation
        return jsonify({'status': 'Error',
                        'message': f'Internal server error during fluid simulation (ID: {error_id}). Check logs.'}), 500


# --- END Routes for 1D Eulerian Fluid Simulator ---


@app.route('/calculate_heatsink_area', methods=['POST'])
def api_calculate_heatsink_area():
    """
    API endpoint to calculate the wetted contact area of a heatsink.
    Accepts JSON data with heatsink dimensions.
    """
    print("Received POST request at /calculate_heatsink_area")
    try:
        data = request.get_json()
        if not data:
            print("Error: No JSON data received for /calculate_heatsink_area")
            return jsonify({'status': 'Error', 'message': 'No JSON data received'}), 400

        # Parameter names from frontend javascript (heatsinkAreaData)
        # mapped to function arguments for calculate_heatsink_contact_area
        param_map = {
            'heatsink_designer_lx': 'hs_length',
            'heatsink_designer_ly': 'hs_width',
            'fin_height': 'fin_height',
            'fin_width': 'fin_width',
            'num_fins': 'num_fins',
            'hollow_fin_length': 'hollow_fin_length',
            'hollow_fin_width': 'hollow_fin_width',
            'num_hollow_fins': 'num_hollow_fins'
        }

        expected_params_info = {
            'heatsink_designer_lx': float,
            'heatsink_designer_ly': float,
            'fin_height': float,
            'fin_width': float,
            'num_fins': int,
            'hollow_fin_length': float, # Optional in JS if num_hollow_fins is 0
            'hollow_fin_width': float,  # Optional in JS if num_hollow_fins is 0
            'num_hollow_fins': int      # Optional in JS (defaults to 0 if not present)
        }

        sim_core_params = {}

        for js_key, py_key in param_map.items():
            expected_type = expected_params_info[js_key]
            value = data.get(js_key)

            # Handle optional parameters that might not be sent if num_hollow_fins is 0
            # The core function expects them, so we default to 0.0 or 0 if missing.
            is_hollow_param = js_key in ['hollow_fin_length', 'hollow_fin_width', 'num_hollow_fins']
            if value is None and is_hollow_param:
                 # If num_hollow_fins itself is missing, it defaults to 0.
                 # If other hollow params are missing, they default to 0.0 for float, 0 for int.
                 # The core function will validate if num_hollow_fins > 0 but others are 0.
                sim_core_params[py_key] = 0.0 if expected_type == float else 0
                print(f"Info: Optional param '{js_key}' not found, defaulting to {sim_core_params[py_key]}.")
                continue # Skip further checks for this optional missing param

            if value is None and not is_hollow_param: # Required param missing
                print(f"Error: Missing required parameter '{js_key}' for /calculate_heatsink_area")
                return jsonify({'status': 'Error', 'message': f"Missing required parameter: {js_key}"}), 400

            try:
                sim_core_params[py_key] = expected_type(value)
            except (ValueError, TypeError) as e:
                print(f"Error: Invalid data type for parameter '{js_key}'. Expected {expected_type.__name__}, got '{value}'. Error: {e}")
                return jsonify({'status': 'Error', 'message': f"Invalid data type for parameter {js_key}. Expected {expected_type.__name__}."}), 400

        print(f"Parameters for calculate_heatsink_contact_area: {sim_core_params}")

        # Call the core calculation function
        result = calculate_heatsink_contact_area(**sim_core_params)

        if 'error' in result:
            print(f"Error from calculate_heatsink_contact_area: {result['error']}")
            return jsonify({'status': 'Error', 'message': result['error']}), 400
        elif 'contact_area' in result:
            print(f"Success from calculate_heatsink_contact_area: Area = {result['contact_area']}")
            return jsonify({'status': 'Success', 'contact_area': result['contact_area']})
        else:
            # Should not happen if calculate_heatsink_contact_area behaves as expected
            print("Error: Unexpected result format from calculate_heatsink_contact_area.")
            return jsonify({'status': 'Error', 'message': 'Internal server error: Unexpected result format from calculation.'}), 500

    except Exception as e:
        import traceback
        error_id = str(time.time())
        print(f"Critical error in /calculate_heatsink_area (ID: {error_id}): {e}")
        traceback.print_exc()
        return jsonify({'status': 'Error', 'message': f'Internal server error during area calculation (ID: {error_id}). Check logs.'}), 500


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Starting Flask server (Dynamic Mode)...")  # Changed
    app.run(debug=True, host='0.0.0.0', port=5000)

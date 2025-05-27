import numpy as np
import matplotlib.pyplot as plt
import io
import base64

# Constants for air
GAMMA = 1.4  # Specific heat ratio
R_AIR = 287.0  # Specific gas constant for air (J/kgK)

def initialize_grid(num_cells, initial_density, initial_velocity, initial_pressure):
    """
    Initializes the computational grid with uniform conditions.

    Args:
        num_cells (int): The number of cells in the grid.
        initial_density (float): Initial density value for all cells.
        initial_velocity (float): Initial velocity value for all cells.
        initial_pressure (float): Initial pressure value for all cells.

    Returns:
        tuple: A tuple containing NumPy arrays for density, momentum, and total_energy.
    """
    density = np.full(num_cells, initial_density)
    velocity = np.full(num_cells, initial_velocity)
    pressure = np.full(num_cells, initial_pressure)

    momentum = density * velocity

    # Calculate internal energy per unit mass (e)
    # e = P / ((GAMMA - 1.0) * ρ)
    # Avoid division by zero if initial_density is very small
    if initial_density > 1e-6:
        internal_energy_per_unit_mass = pressure / ((GAMMA - 1.0) * initial_density)
    else:
        internal_energy_per_unit_mass = np.zeros(num_cells)
        # Or handle as an error, depending on desired behavior for zero density

    # Calculate total energy (ρE_total)
    # ρE_total = ρ * (e + 0.5 * u^2)
    total_energy = density * (internal_energy_per_unit_mass + 0.5 * velocity**2)
    
    return density, momentum, total_energy

def calculate_primitive_variables(density, momentum, total_energy):
    """
    Calculates primitive variables (pressure, velocity, temperature) from conservative variables.

    Args:
        density (np.ndarray): Density (ρ).
        momentum (np.ndarray): Momentum (ρu).
        total_energy (np.ndarray): Total energy (ρE_total).

    Returns:
        tuple: A tuple containing NumPy arrays for pressure, velocity, and temperature.
    """
    # Avoid division by zero by checking if density is close to zero
    velocity = np.where(density > 1e-6, momentum / density, 0.0)
    
    # Pressure calculation: P = (GAMMA - 1.0) * (total_energy - 0.5 * momentum**2 / density)
    # Ensure density is not zero before division
    kinetic_energy_density = np.where(density > 1e-6, 0.5 * momentum**2 / density, 0.0)
    internal_energy_density = total_energy - kinetic_energy_density
    pressure = (GAMMA - 1.0) * internal_energy_density
    
    # Temperature calculation: T = pressure / (density * R_AIR)
    # Ensure density and R_AIR are not zero
    temperature = np.where((density > 1e-6) & (R_AIR != 0), pressure / (density * R_AIR), 0.0)
    
    return pressure, velocity, temperature

def calculate_fluxes(density, momentum, total_energy, pressure):
    """
    Calculates fluxes for the 1D Euler equations.

    Args:
        density (np.ndarray): Density (ρ) at each cell.
        momentum (np.ndarray): Momentum (ρu) at each cell.
        total_energy (np.ndarray): Total energy (ρE_total) at each cell.
        pressure (np.ndarray): Pressure (P) at each cell.

    Returns:
        tuple: A tuple containing NumPy arrays for flux_density (F1),
               flux_momentum (F2), and flux_total_energy (F3).
    """
    velocity = np.where(density > 1e-6, momentum / density, 0.0)

    flux_density = momentum  # F1 = ρu
    
    # F2 = ρu^2 + P = (ρu)^2/ρ + P
    flux_momentum = np.where(density > 1e-6, momentum**2 / density + pressure, pressure) # if density is near zero, flux is just P

    # F3 = (ρE_total + P)u = (ρE_total + P) * (ρu/ρ)
    flux_total_energy = np.where(density > 1e-6, (total_energy + pressure) * velocity, 0.0)

    return flux_density, flux_momentum, flux_total_energy

def calculate_dt(density, velocity, pressure, dx, cfl_number):
    """
    Calculates the time step 'dt' based on the CFL condition.

    Args:
        density (np.ndarray): Density (ρ) array.
        velocity (np.ndarray): Velocity (u) array.
        pressure (np.ndarray): Pressure (P) array.
        dx (float): Spatial step size.
        cfl_number (float): Courant-Friedrichs-Lewy number.

    Returns:
        float: The calculated time step 'dt'.
    """
    # Ensure density is positive for speed of sound calculation
    safe_density = np.maximum(density, 1e-9)
    # Ensure pressure is non-negative for speed of sound calculation
    safe_pressure = np.maximum(pressure, 0.0) # Pressure can be zero, but not negative for sqrt

    speed_of_sound = np.sqrt(GAMMA * safe_pressure / safe_density)
    
    max_signal_speed = np.max(np.abs(velocity) + speed_of_sound)
    
    if max_signal_speed == 0:
        # Avoid division by zero if system is static; choose a small dt or handle as error
        # For now, returning a small, somewhat arbitrary dt if max_signal_speed is 0.
        # This case might need more sophisticated handling depending on the simulation context.
        return 1e-6 # Or raise an error, or use a default dt if appropriate
        
    dt = cfl_number * dx / max_signal_speed
    return dt

def maccormack_step(density, momentum, total_energy, dx, dt):
    """
    Performs a single MacCormack step to update conservative variables.
    This implementation updates interior points only. Boundary conditions
    must be applied separately.

    Args:
        density (np.ndarray): Current density (ρ) array.
        momentum (np.ndarray): Current momentum (ρu) array.
        total_energy (np.ndarray): Current total energy (ρE_total) array.
        dx (float): Spatial step size.
        dt (float): Time step size.

    Returns:
        tuple: A tuple containing the updated (corrected) NumPy arrays for
               density, momentum, and total_energy. The first and last
               elements of these arrays are not updated by this function.
    """
    N = len(density)
    
    # --- Predictor Step ---
    # Calculate current primitive variables for flux calculation
    pressure_curr, velocity_curr, _ = calculate_primitive_variables(density, momentum, total_energy)
    
    # Calculate fluxes F(U) based on current state
    flux_density_curr, flux_momentum_curr, flux_total_energy_curr = calculate_fluxes(
        density, momentum, total_energy, pressure_curr
    )

    # Initialize predicted state arrays (full size, but only interior will be valid)
    density_pred = density.copy() # Or np.zeros_like(density) if we don't want to carry over boundaries
    momentum_pred = momentum.copy()
    total_energy_pred = total_energy.copy()

    # Update interior points for predicted state (U_pred)
    # U_pred[i] = U[i] - (dt/dx) * (F[i+1] - F[i])
    # This computes N-1 elements, from index 0 to N-2
    density_pred[:-1] = density[:-1] - (dt/dx) * (flux_density_curr[1:] - flux_density_curr[:-1])
    momentum_pred[:-1] = momentum[:-1] - (dt/dx) * (flux_momentum_curr[1:] - flux_momentum_curr[:-1])
    total_energy_pred[:-1] = total_energy[:-1] - (dt/dx) * (flux_total_energy_curr[1:] - flux_total_energy_curr[:-1])

    # Calculate primitive variables for the predicted state
    pressure_pred, velocity_pred, _ = calculate_primitive_variables(
        density_pred, momentum_pred, total_energy_pred
    )

    # --- Corrector Step ---
    # Calculate fluxes F(U_pred) based on predicted state
    # Note: For F(U_pred), we need fluxes at all points based on U_pred values.
    # The slicing for F_pred[i] - F_pred[i-1] means we need flux_..._pred to be defined at these points.
    # The primitive variables were calculated based on the full _pred arrays.
    flux_density_pred, flux_momentum_pred, flux_total_energy_pred = calculate_fluxes(
        density_pred, momentum_pred, total_energy_pred, pressure_pred
    )

    # Initialize corrected state arrays (these will be the output)
    density_corr = density.copy() 
    momentum_corr = momentum.copy()
    total_energy_corr = total_energy.copy()
    
    # Update interior points for corrected state (U_corr)
    # U_corr[i] = 0.5 * (U[i] + U_pred[i] - (dt/dx) * (F_pred[i] - F_pred[i-1]))
    # This computes N-1 elements, from index 1 to N-1
    # U_pred[:-1] corresponds to U_pred[i] in the formula if U_corr[1:] means i starts from 1.
    # flux_..._pred[1:] corresponds to F_pred[i]
    # flux_..._pred[:-1] corresponds to F_pred[i-1]
    density_corr[1:] = 0.5 * (density[1:] + density_pred[:-1] - \
        (dt/dx) * (flux_density_pred[1:] - flux_density_pred[:-1]))
    
    momentum_corr[1:] = 0.5 * (momentum[1:] + momentum_pred[:-1] - \
        (dt/dx) * (flux_momentum_pred[1:] - flux_momentum_pred[:-1]))
    
    total_energy_corr[1:] = 0.5 * (total_energy[1:] + total_energy_pred[:-1] - \
        (dt/dx) * (flux_total_energy_pred[1:] - flux_total_energy_pred[:-1]))

    return density_corr, momentum_corr, total_energy_corr

def apply_boundary_conditions(density, momentum, total_energy, bc_type):
    """
    Applies boundary conditions to the conservative variable arrays in-place.

    Args:
        density (np.ndarray): Density (ρ) array.
        momentum (np.ndarray): Momentum (ρu) array.
        total_energy (np.ndarray): Total energy (ρE_total) array.
        bc_type (str): Type of boundary condition ("transmissive" or "reflective").
    """
    if bc_type == "transmissive":
        # Left boundary (index 0)
        density[0] = density[1]
        momentum[0] = momentum[1]
        total_energy[0] = total_energy[1]
        
        # Right boundary (index -1)
        density[-1] = density[-2]
        momentum[-1] = momentum[-2]
        total_energy[-1] = total_energy[-2]
        
    elif bc_type == "reflective":
        # Left boundary (index 0)
        density[0] = density[1]
        momentum[0] = -momentum[1]  # Reflects velocity
        total_energy[0] = total_energy[1]
        
        # Right boundary (index -1)
        density[-1] = density[-2]
        momentum[-1] = -momentum[-2]  # Reflects velocity
        total_energy[-1] = total_energy[-2]
        
    # else:
    #     # Optionally, raise an error for unsupported bc_type
    #     # print(f"Warning: Boundary condition type '{bc_type}' not recognized.")
    #     pass

def fig_to_data_uri(fig):
    """Converts a Matplotlib figure to a Data URI PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=90) # Lower DPI for smaller image size
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig) # Close the figure to free memory
    return f"data:image/png;base64,{img_base64}"

def run_1d_eulerian_simulation(
    domain_length_m, 
    num_cells, 
    total_sim_time_s, 
    initial_density_kg_m3, 
    initial_velocity_m_s, 
    initial_pressure_Pa, 
    boundary_condition_type, 
    cfl_number=0.5, 
    output_time_steps=10
):
    """
    Runs a 1D Eulerian fluid simulation using the MacCormack scheme.

    Args:
        domain_length_m (float): Length of the computational domain in meters.
        num_cells (int): Number of cells in the computational domain.
        total_sim_time_s (float): Total simulation time in seconds.
        initial_density_kg_m3 (float): Initial density in kg/m^3.
        initial_velocity_m_s (float): Initial velocity in m/s.
        initial_pressure_Pa (float): Initial pressure in Pascals.
        boundary_condition_type (str): Type of boundary condition ("transmissive" or "reflective").
        cfl_number (float, optional): Courant-Friedrichs-Lewy number for time step calculation. Defaults to 0.5.
        output_time_steps (int, optional): Number of time steps to store for output, including initial and final. 
                                         Minimum 1 (only final state). If 2, initial and final. If >2, initial, final and intermediates.
                                         Defaults to 10.

    Returns:
        dict: A dictionary containing:
            - 'status' (str): 'Success' or 'Error'.
            - 'message' (str): A message describing the outcome or error.
            - 'parameters' (dict): A dictionary of the input simulation parameters.
            - 'results_history' (list): A list of tuples. Each tuple contains:
                (time_s, x_coords_m, density_kg_m3, velocity_m_s, pressure_Pa, temperature_K).
                The list contains results at specified output intervals.
            - 'plots_data_uris' (dict): A dictionary where keys are plot types (e.g., 'density') 
                                        and values are base64 encoded PNG data URIs of the plots for the final state.
    """
    sim_params = {
        'domain_length_m': domain_length_m, 'num_cells': num_cells, 'total_sim_time_s': total_sim_time_s,
        'initial_density_kg_m3': initial_density_kg_m3, 'initial_velocity_m_s': initial_velocity_m_s,
        'initial_pressure_Pa': initial_pressure_Pa, 'boundary_condition_type': boundary_condition_type,
        'cfl_number': cfl_number, 'output_time_steps': output_time_steps
    }

    results_history = []
    plots_data_uris = {}

    try:
        if output_time_steps < 1:
            output_time_steps = 1
        
        # Handle invalid physical parameters before grid initialization
        if num_cells <= 0 or domain_length_m <= 0:
            # Prepare initial state for plotting even with invalid parameters for consistency
            dx_placeholder = 1.0 if num_cells <=0 else domain_length_m / (num_cells if num_cells > 0 else 1)
            x_coords = np.linspace(0, domain_length_m if domain_length_m > 0 else dx_placeholder, num_cells if num_cells > 0 else 1) # Avoid division by zero
            density = np.full_like(x_coords, initial_density_kg_m3)
            velocity = np.full_like(x_coords, initial_velocity_m_s)
            pressure = np.full_like(x_coords, initial_pressure_Pa)
            # Temp calculation might be problematic if R_AIR or density is zero, default to 0 or NaN.
            temperature = np.zeros_like(x_coords)
            if initial_density_kg_m3 > 1e-6 and R_AIR != 0:
                 temperature = pressure / (density * R_AIR)

            initial_state_tuple = (0.0, x_coords, density, velocity, pressure, temperature)
            results_history.append(initial_state_tuple)
            
            # Generate plots for this "initial" or "invalid setup" state
            t_plot, x_plot, d_plot, v_plot, p_plot, temp_plot = results_history[-1]
            fig_d, ax_d = plt.subplots(); ax_d.plot(x_plot, d_plot); ax_d.set_title(f'Density at t={t_plot:.4f}s'); ax_d.set_xlabel('Position (m)'); ax_d.set_ylabel('Density (kg/m^3)'); ax_d.grid(True); plots_data_uris['density'] = fig_to_data_uri(fig_d)
            fig_v, ax_v = plt.subplots(); ax_v.plot(x_plot, v_plot); ax_v.set_title(f'Velocity at t={t_plot:.4f}s'); ax_v.set_xlabel('Position (m)'); ax_v.set_ylabel('Velocity (m/s)'); ax_v.grid(True); plots_data_uris['velocity'] = fig_to_data_uri(fig_v)
            fig_p, ax_p = plt.subplots(); ax_p.plot(x_plot, p_plot); ax_p.set_title(f'Pressure at t={t_plot:.4f}s'); ax_p.set_xlabel('Position (m)'); ax_p.set_ylabel('Pressure (Pa)'); ax_p.grid(True); plots_data_uris['pressure'] = fig_to_data_uri(fig_p)
            fig_t, ax_t = plt.subplots(); ax_t.plot(x_plot, temp_plot); ax_t.set_title(f'Temperature at t={t_plot:.4f}s'); ax_t.set_xlabel('Position (m)'); ax_t.set_ylabel('Temperature (K)'); ax_t.grid(True); plots_data_uris['temperature'] = fig_to_data_uri(fig_t)
            
            return {
                'status': 'Success', 'message': 'Simulation not run due to non-positive domain_length_m or num_cells. Initial state plotted.',
                'parameters': sim_params, 'results_history': results_history, 'plots_data_uris': plots_data_uris
            }

        dx = domain_length_m / num_cells
        x_coords = np.linspace(0, domain_length_m, num_cells) # Cell centers or faces? Assumed centers for now.

        density, momentum, total_energy = initialize_grid(
            num_cells, initial_density_kg_m3, initial_velocity_m_s, initial_pressure_Pa
        )
        pressure, velocity, temperature = calculate_primitive_variables(density, momentum, total_energy)
        
        current_time_s = 0.0
        initial_state_tuple = (current_time_s, x_coords.copy(), density.copy(), velocity.copy(), pressure.copy(), temperature.copy())
        results_history.append(initial_state_tuple)

        if total_sim_time_s <= 0:
            # Plot initial state and return if no simulation time
            t_plot, x_plot, d_plot, v_plot, p_plot, temp_plot = results_history[-1]
            fig_d, ax_d = plt.subplots(); ax_d.plot(x_plot, d_plot); ax_d.set_title(f'Density at t={t_plot:.4f}s'); ax_d.set_xlabel('Position (m)'); ax_d.set_ylabel('Density (kg/m^3)'); ax_d.grid(True); plots_data_uris['density'] = fig_to_data_uri(fig_d)
            fig_v, ax_v = plt.subplots(); ax_v.plot(x_plot, v_plot); ax_v.set_title(f'Velocity at t={t_plot:.4f}s'); ax_v.set_xlabel('Position (m)'); ax_v.set_ylabel('Velocity (m/s)'); ax_v.grid(True); plots_data_uris['velocity'] = fig_to_data_uri(fig_v)
            fig_p, ax_p = plt.subplots(); ax_p.plot(x_plot, p_plot); ax_p.set_title(f'Pressure at t={t_plot:.4f}s'); ax_p.set_xlabel('Position (m)'); ax_p.set_ylabel('Pressure (Pa)'); ax_p.grid(True); plots_data_uris['pressure'] = fig_to_data_uri(fig_p)
            fig_t, ax_t = plt.subplots(); ax_t.plot(x_plot, temp_plot); ax_t.set_title(f'Temperature at t={t_plot:.4f}s'); ax_t.set_xlabel('Position (m)'); ax_t.set_ylabel('Temperature (K)'); ax_t.grid(True); plots_data_uris['temperature'] = fig_to_data_uri(fig_t)
            
            return {
                'status': 'Success', 'message': 'Simulation not run as total_sim_time_s <= 0. Initial state plotted.',
                'parameters': sim_params, 'results_history': results_history, 'plots_data_uris': plots_data_uris
            }

        # output_time_steps includes initial and final. So N-1 intervals.
        output_interval = total_sim_time_s / (output_time_steps -1) if output_time_steps > 1 else float('inf')
        next_output_target_time = output_interval
        num_steps_taken = 0

        while current_time_s < total_sim_time_s:
            dt = calculate_dt(density, velocity, pressure, dx, cfl_number)
            dt = min(dt, total_sim_time_s - current_time_s) # Don't overshoot total_sim_time_s

            if dt < 1e-12: # Simulation stalled or effectively completed
                break 

            density_new, momentum_new, total_energy_new = maccormack_step(
                density, momentum, total_energy, dx, dt
            )
            density, momentum, total_energy = density_new.copy(), momentum_new.copy(), total_energy_new.copy()
            
            apply_boundary_conditions(density, momentum, total_energy, boundary_condition_type)
            
            pressure, velocity, temperature = calculate_primitive_variables(density, momentum, total_energy)
            
            current_time_s += dt
            num_steps_taken +=1

            # Store results if output_time_steps > 1 (more than just initial and final)
            # and if we are close to the next target time or it's the very last step.
            is_last_step_before_or_at_total_time = abs(current_time_s - total_sim_time_s) < 1e-9 or current_time_s >= total_sim_time_s
            
            if output_time_steps > 1 and \
               (current_time_s >= next_output_target_time - 1e-9 or is_last_step_before_or_at_total_time): # -1e-9 for float precision
                if len(results_history) < output_time_steps -1: # Still space for intermediate results
                    results_history.append((current_time_s, x_coords.copy(), density.copy(), velocity.copy(), pressure.copy(), temperature.copy()))
                    next_output_target_time += output_interval
                # If it's the last step and we haven't filled up to output_time_steps-1, it will be caught by final state storage.

        # Final state storage logic
        final_state_tuple = (current_time_s, x_coords.copy(), density.copy(), velocity.copy(), pressure.copy(), temperature.copy())
        if output_time_steps == 1:
            results_history = [final_state_tuple] # Only the final state
        else: # output_time_steps > 1
            # Ensure the very last computed state is the last item in results_history.
            # This might replace the last "intermediate" if it was very close to total_sim_time_s,
            # or append if fewer than output_time_steps were collected.
            if len(results_history) < output_time_steps:
                results_history.append(final_state_tuple)
            else: 
                results_history[-1] = final_state_tuple # Replace the last stored item with the true final state

        # Plotting the final state (or the state at which simulation ended)
        t_plot, x_plot, d_plot, v_plot, p_plot, temp_plot = results_history[-1]

        fig_d, ax_d = plt.subplots(); ax_d.plot(x_plot, d_plot); ax_d.set_title(f'Density at t={t_plot:.4f}s'); ax_d.set_xlabel('Position (m)'); ax_d.set_ylabel('Density (kg/m^3)'); ax_d.grid(True); plots_data_uris['density'] = fig_to_data_uri(fig_d)
        fig_v, ax_v = plt.subplots(); ax_v.plot(x_plot, v_plot); ax_v.set_title(f'Velocity at t={t_plot:.4f}s'); ax_v.set_xlabel('Position (m)'); ax_v.set_ylabel('Velocity (m/s)'); ax_v.grid(True); plots_data_uris['velocity'] = fig_to_data_uri(fig_v)
        fig_p, ax_p = plt.subplots(); ax_p.plot(x_plot, p_plot); ax_p.set_title(f'Pressure at t={t_plot:.4f}s'); ax_p.set_xlabel('Position (m)'); ax_p.set_ylabel('Pressure (Pa)'); ax_p.grid(True); plots_data_uris['pressure'] = fig_to_data_uri(fig_p)
        fig_t, ax_t = plt.subplots(); ax_t.plot(x_plot, temp_plot); ax_t.set_title(f'Temperature at t={t_plot:.4f}s'); ax_t.set_xlabel('Position (m)'); ax_t.set_ylabel('Temperature (K)'); ax_t.grid(True); plots_data_uris['temperature'] = fig_to_data_uri(fig_t)

        message = f"Simulation completed successfully after {num_steps_taken} steps. Reached t={current_time_s:.4f}s."
        if abs(current_time_s - total_sim_time_s) > 1e-6 and dt < 1e-12 : # Check if stalled
             message = f"Simulation stalled or dt too small after {num_steps_taken} steps. Reached t={current_time_s:.4f}s."


        return {
            'status': 'Success', 'message': message,
            'parameters': sim_params, 'results_history': results_history, 'plots_data_uris': plots_data_uris
        }

    except Exception as e:
        # Ensure plots_data_uris is populated with empty strings or error messages if plots failed
        for key in ['density', 'velocity', 'pressure', 'temperature']:
            if key not in plots_data_uris:
                plots_data_uris[key] = "Error generating plot."
        return {
            'status': 'Error', 'message': f"An error occurred: {str(e)}", 
            'parameters': sim_params, 'results_history': results_history, # results_history might have partial data
            'plots_data_uris': plots_data_uris
        }

if __name__ == '__main__':
    print("Running local test for 1D Eulerian Fluid Simulator...")

    # Revised Test Case Parameters
    test_params = {
        'domain_length_m': 1.0,
        'num_cells': 100,
        'total_sim_time_s': 0.005,  # Very short, for a fast moving wave
        'initial_density_kg_m3': 1.0,
        'initial_velocity_m_s': 100.0, # A significant initial velocity
        'initial_pressure_Pa': 101325.0,
        'boundary_condition_type': "transmissive",
        'cfl_number': 0.5,
        'output_time_steps': 3  # Initial, middle, final
    }

    print(f"\nTest Parameters:")
    for key, value in test_params.items():
        print(f"  {key}: {value}")
    print("\nStarting simulation test...")

    results = run_1d_eulerian_simulation(**test_params)

    print("\n--- Simulation Results ---")
    print(f"Simulation Status: {results['status']}")
    print(f"Simulation Message: {results['message']}")
    
    if 'parameters' in results:
        print(f"Ran with {results['parameters']['num_cells']} cells for {results['parameters']['total_sim_time_s']}s.")

    if 'results_history' in results and results['results_history']:
        print(f"Number of history snapshots: {len(results['results_history'])}")
        
        # Print details from the final state
        final_state = results['results_history'][-1]
        t_final, x_final, d_final, v_final, p_final, temp_final = final_state
        
        print(f"\nFinal state at t = {t_final:.6f}s:")
        print(f"  Spatial points: {len(x_final)}")
        if len(d_final) > 0:
            print(f"  Density  | Min: {np.min(d_final):.3f}, Max: {np.max(d_final):.3f} | First: {d_final[0]:.3f}, Mid: {d_final[len(d_final)//2]:.3f}, Last: {d_final[-1]:.3f}")
        if len(v_final) > 0:
            print(f"  Velocity | Min: {np.min(v_final):.2f}, Max: {np.max(v_final):.2f} | First: {v_final[0]:.2f}, Mid: {v_final[len(v_final)//2]:.2f}, Last: {v_final[-1]:.2f}")
        if len(p_final) > 0:
            print(f"  Pressure | Min: {np.min(p_final):.0f}, Max: {np.max(p_final):.0f} | First: {p_final[0]:.0f}, Mid: {p_final[len(p_final)//2]:.0f}, Last: {p_final[-1]:.0f}")
        if len(temp_final) > 0:
            print(f"  Temp.    | Min: {np.min(temp_final):.1f}, Max: {np.max(temp_final):.1f} | First: {temp_final[0]:.1f}, Mid: {temp_final[len(temp_final)//2]:.1f}, Last: {temp_final[-1]:.1f}")

    if 'plots_data_uris' in results:
        print(f"\nPlot URIs generated: {list(results['plots_data_uris'].keys())}")
        # Example: To save one of the plots if needed (requires further setup if not running in Flask context)
        # if results['plots_data_uris'].get('density'):
        #     import re
        #     img_data = re.sub(r'^data:image/png;base64,', '', results['plots_data_uris']['density'])
        #     with open("density_test_plot.png", "wb") as fh:
        #         fh.write(base64.b64decode(img_data))
        #     print("Saved 'density_test_plot.png' for inspection (example).")

    print("\nLocal test finished.")
    print("Note: This test runs with uniform initial conditions.")
    print("For a shock tube or contact discontinuity, the initialization would need to support non-uniform states,")
    print("or the core solver functions would be called directly with custom initial arrays.")

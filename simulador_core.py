# simulador_core.py (Versión Actualizada)

import matplotlib

matplotlib.use('Agg')  # Usar backend no interactivo para Flask
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu
import time
import math
import os
import io
import base64
import traceback  # Importar para trazas de error detalladas

# --- Constantes del Nuevo Modelo (algunas serán argumentos de función o valores por defecto) ---
NTC_REL_X = +0.021
NTC_REL_Y = -0.036

# Estos serán los valores por defecto si no se pasan como argumentos
DEFAULT_K_MAIN_HEATSINK_BASE = 218.0
DEFAULT_T_AMBIENT_INLET = 40.0
DEFAULT_Q_TOTAL_M3_H = 785.0

cp_aire = 1005.0  # J/kgK
rho_aire = 1.109  # kg/m³ (aproximado a 50°C)
w_igbt_footprint, h_igbt_footprint = 0.062, 0.122  # Dimensiones físicas módulo (m)

K_MODULE_BASEPLATE = 390.0  # Conductividad de la base de cobre del módulo
T_MODULE_BASEPLATE = 0.003  # Espesor de la base de cobre del módulo
K_TIM_INTERFACE = 5.0  # Conductividad del TIM entre módulo y disipador
T_TIM_INTERFACE = 0.000125  # Espesor del TIM
H_MODULE_TO_HEATSINK_INTERFACE = K_TIM_INTERFACE / T_TIM_INTERFACE if T_TIM_INTERFACE > 1e-9 else float('inf')

# Dimensiones y Rth junction-heatsink de los chips (aproximados)
h_chip_igbt_m, w_chip_igbt_m = 0.036, 0.022
A_chip_igbt = w_chip_igbt_m * h_chip_igbt_m
h_chip_diode_m, w_chip_diode_m = 0.036, 0.011
A_chip_diode = w_chip_diode_m * h_chip_diode_m
Rth_jhs_igbt = 0.0721  # °C/W (Junction to Heatsink Surface under chip) - Usado para Tj desde T_base_chip
Rth_jhs_diode = 0.121  # °C/W

Nx_base_default, Ny_base_default = 250, 350  # Resolución FDM por defecto (puede ser sobreescrita)

# Posiciones relativas de los centros de los chips respecto al centro del módulo
chip_rel_positions = {
    "IGBT1": (-0.006, +0.023), "Diode1": (+0.012, +0.023),
    "IGBT2": (+0.006, -0.023), "Diode2": (-0.012, -0.023),
}
EXPERIMENTAL_RTH_NTC_DATA = {
    "IGBT1": [([0.00, 0.00, 1.00, 0.00], 0.1041915), ([0.50, 0.00, 0.50, 0.00], 0.093628),
              ([0.25, 0.25, 0.25, 0.25], 0.1204493), ([0.00, 0.00, 0.50, 0.50], 0.1521716)],
    "IGBT2": [([1.00, 0.00, 0.00, 0.00], 0.0768713), ([0.50, 0.00, 0.50, 0.00], 0.093628),
              ([0.25, 0.25, 0.25, 0.25], 0.1191141), ([0.50, 0.50, 0.00, 0.00], 0.1053138)],
    "Diode1": [([0.00, 0.00, 0.00, 1.00], 0.1441531), ([0.00, 0.50, 0.00, 0.50], 0.1502391),
               ([0.25, 0.25, 0.25, 0.25], 0.1894391), ([0.00, 0.00, 0.50, 0.50], 0.2024668)],
    "Diode2": [([0.00, 1.00, 0.00, 0.00], 0.1466934), ([0.00, 0.50, 0.00, 0.50], 0.1502391),
               ([0.25, 0.25, 0.25, 0.25], 0.1883709), ([0.50, 0.50, 0.00, 0.00], 0.1640444)]
}
CHIP_ORDER_FOR_RTH_TABLE = ["IGBT2", "Diode2", "IGBT1", "Diode1"]


# --- FIN CONSTANTES ---

def fig_to_data_uri(fig, tight=True, pad_inches=0.1, transparent=False):
    """Convierte una figura de Matplotlib a Data URI PNG."""
    buf = io.BytesIO()
    save_kwargs = {'format': 'png', 'dpi': 90, 'transparent': transparent}
    if tight:
        save_kwargs['bbox_inches'] = 'tight'
        save_kwargs['pad_inches'] = pad_inches
    fig.savefig(buf, **save_kwargs)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)  # Importante cerrar la figura
    return f"data:image/png;base64,{img_base64}"


def find_closest_experimental_rth(actual_power_distribution_normalized, experimental_rth_entries):
    if not experimental_rth_entries:
        return np.nan, None
    min_distance = float('inf')
    best_rth = np.nan
    closest_experimental_distribution = None
    actual_power_distribution_normalized_np = np.array(actual_power_distribution_normalized)
    for exp_dist, rth_val in experimental_rth_entries:
        exp_dist_np = np.array(exp_dist)
        if len(exp_dist_np) != len(actual_power_distribution_normalized_np):
            continue
        distance = np.sum(np.abs(actual_power_distribution_normalized_np - exp_dist_np))  # L1 norm
        if distance < min_distance:
            min_distance = distance
            best_rth = rth_val
            closest_experimental_distribution = exp_dist
        elif distance == min_distance:  # En caso de empate, podríamos tener una lógica (ej. tomar el promedio, o el primero)
            pass  # Por ahora, se queda con el primero encontrado con esa distancia mínima
    return best_rth, closest_experimental_distribution


def run_thermal_simulation(specific_chip_powers,
                           lx, ly, t_heatsink_thickness,  # 't' es el espesor del disipador principal
                           k_main_heatsink_base_arg,  # Nuevo argumento para K del disipador
                           rth_heatsink,
                           t_ambient_inlet_arg,  # Nuevo argumento para T ambiente
                           q_total_m3_h_arg,  # Nuevo argumento para Caudal
                           module_definitions,
                           nx=Nx_base_default, ny=Ny_base_default):
    """
    Ejecuta simulación FDM de dos etapas (módulos + disipador),
    genera URIs para plots y añade datos numéricos.
    """
    results = {
        'status': 'Processing', 'convergence': False, 'iterations': 0,
        't_max_base': np.nan, 't_avg_base': np.nan, 't_air_outlet': np.nan,
        't_max_junction': np.nan, 't_max_junction_chip': '', 't_max_ntc': np.nan,
        'module_results': [],
        'plot_base_data_uri': None,  # Combinado T_Base (Disipador) + T_Aire (Estático)
        'plot_zoom_data_uri': None,  # Zoom T_Superficie_Módulos (Estático)
        'plot_interactive_raw_uri': None,  # SOLO contourf T_Base (Disipador) (Interactivo)
        'error_message': None,
        'temperature_matrix': None, 'x_coordinates': None, 'y_coordinates': None,
        'sim_lx': None, 'sim_ly': None, 'sim_nx': None, 'sim_ny': None,
    }
    start_time_sim = time.time()
    print(f"[SimCoreV2] Iniciando con {len(module_definitions)} módulos. Simulación detallada de módulo y disipador.")
    print(f"[SimCoreV2] Tj IGBTs desde T_HÍBRIDA(max,avg)_hs_chip. Tj Diodos desde T_AVG_hs_chip.")
    print(f"[SimCoreV2] T_NTC será calculada desde Tj_simulada y Rth_j-NTC experimentales ponderadas.")
    print(
        f"[SimCoreV2] Disipador: Lx={lx:.3f}, Ly={ly:.3f}, t_hs={t_heatsink_thickness:.4f}, k_hs={k_main_heatsink_base_arg:.1f}, Rth_global={rth_heatsink:.4f}")
    print(f"[SimCoreV2] Ambiente: T_in={t_ambient_inlet_arg:.1f}°C, Q={q_total_m3_h_arg:.1f} m³/h")
    print(f"[SimCoreV2] FDM (Disipador): Nx={nx}, Ny={ny}")

    # --- Validaciones ---
    if not isinstance(specific_chip_powers, dict): results['status'] = 'Error'; results[
        'error_message'] = 'Powers dict invalido.'; return results
    if not all(isinstance(d, (int, float)) and d > 1e-9 for d in
               [lx, ly, t_heatsink_thickness, k_main_heatsink_base_arg, rth_heatsink, q_total_m3_h_arg]):
        results['status'] = 'Error';
        results['error_message'] = 'Dimensiones, k_base, Rth y Caudal (Q) deben ser números > 0.';
        return results
    if not isinstance(t_ambient_inlet_arg, (int, float)): results['status'] = 'Error'; results[
        'error_message'] = 'T_ambient_inlet debe ser un número.'; return results
    if not isinstance(module_definitions, list): results['status'] = 'Error'; results[
        'error_message'] = 'module_definitions debe ser lista.'; return results
    if not all(isinstance(n, int) and n > 1 for n in [nx, ny]): results['status'] = 'Error'; results[
        'error_message'] = 'Nx/Ny deben ser > 1.'; return results

    # --- Manejar caso sin módulos ---
    if not module_definitions:
        print("[SimCoreV2] No se proporcionaron módulos. Simulando solo convección base del disipador (simplificado).")
        try:
            # Corregido: A_conv_disipador_no_mod dinámico
            A_conv_disipador_no_mod = lx * ly * 2 + (2 * lx * t_heatsink_thickness) + (2 * ly * t_heatsink_thickness)
            if rth_heatsink > 1e-9 and A_conv_disipador_no_mod > 1e-9:
                h_avg_disipador_no_mod = 1.0 / (rth_heatsink * A_conv_disipador_no_mod)
            else:
                h_avg_disipador_no_mod = 85.0  # Fallback

            A_placa_base_FDM_no_mod = lx * ly
            if A_placa_base_FDM_no_mod > 1e-9:
                h_eff_FDM_no_mod = h_avg_disipador_no_mod * (A_conv_disipador_no_mod / A_placa_base_FDM_no_mod)
            else:
                h_eff_FDM_no_mod = 10.0  # Fallback

            dx_no_mod = lx / (nx - 1) if nx > 1 else lx
            dy_no_mod = ly / (ny - 1) if ny > 1 else ly
            N_nodes_no_mod = nx * ny
            dA_no_mod = dx_no_mod * dy_no_mod

            C_x_no_mod = k_main_heatsink_base_arg * t_heatsink_thickness * dy_no_mod / dx_no_mod if dx_no_mod > 1e-12 else 0
            C_y_no_mod = k_main_heatsink_base_arg * t_heatsink_thickness * dx_no_mod / dy_no_mod if dy_no_mod > 1e-12 else 0

            A_mat_no_mod = lil_matrix((N_nodes_no_mod, N_nodes_no_mod))
            b_vec_no_mod = np.zeros(N_nodes_no_mod)

            for i_node in range(nx):
                for j_node in range(ny):
                    idx = i_node * ny + j_node
                    C_h_local_no_mod = h_eff_FDM_no_mod * dA_no_mod
                    coef_diag_no_mod = -C_h_local_no_mod
                    b_vec_no_mod[idx] = - C_h_local_no_mod * t_ambient_inlet_arg  # Usar T_ambiente del argumento
                    if i_node > 0: A_mat_no_mod[idx, idx - ny] = C_x_no_mod; coef_diag_no_mod -= C_x_no_mod
                    if i_node < nx - 1: A_mat_no_mod[idx, idx + ny] = C_x_no_mod; coef_diag_no_mod -= C_x_no_mod
                    if j_node > 0: A_mat_no_mod[idx, idx - 1] = C_y_no_mod; coef_diag_no_mod -= C_y_no_mod
                    if j_node < ny - 1: A_mat_no_mod[idx, idx + 1] = C_y_no_mod; coef_diag_no_mod -= C_y_no_mod
                    A_mat_no_mod[idx, idx] = coef_diag_no_mod

            LU_no_mod = splu(A_mat_no_mod.tocsr())
            T_flat_no_mod = LU_no_mod.solve(b_vec_no_mod)
            T_sol_no_mod = T_flat_no_mod.reshape((nx, ny))
            T_air_no_mod = np.full((nx, ny), t_ambient_inlet_arg)  # Aire no se calienta en este modelo simple

            results['t_max_base'] = np.max(T_sol_no_mod)
            results['t_avg_base'] = np.mean(T_sol_no_mod)
            results['t_air_outlet'] = t_ambient_inlet_arg  # No hay calentamiento de aire modelado aquí
            results['convergence'] = True;
            results['iterations'] = 0
            results['status'] = 'Success'

            x_coords_gfx_no_mod = np.linspace(0, lx, nx)
            y_coords_gfx_no_mod = np.linspace(0, ly, ny)
            X_gfx_no_mod, Y_gfx_no_mod = np.meshgrid(x_coords_gfx_no_mod, y_coords_gfx_no_mod, indexing='ij')

            min_T_plot_no_mod = t_ambient_inlet_arg
            max_T_plot_no_mod = np.max(T_sol_no_mod) if not np.isnan(T_sol_no_mod).all() else t_ambient_inlet_arg + 1
            max_T_plot_no_mod = max(min_T_plot_no_mod + 0.1, max_T_plot_no_mod)
            levels_main_no_mod = np.linspace(min_T_plot_no_mod, max_T_plot_no_mod, 30) if abs(
                max_T_plot_no_mod - min_T_plot_no_mod) > 1e-3 else np.linspace(min_T_plot_no_mod - 0.5,
                                                                               max_T_plot_no_mod + 0.5, 2)
            T_plot_base_no_mod = np.nan_to_num(T_sol_no_mod, nan=min_T_plot_no_mod - 10)

            # --- Gráfico Combinado Base/Aire (Estático) - Caso sin módulos ---
            fig_main_no_mod, axes_no_mod = plt.subplots(1, 2, figsize=(11, 4.5))
            title_fontsize = 12;
            label_fontsize = 10;
            tick_fontsize = 9;
            suptitle_fontsize = 14

            cbar_base_no_mod = fig_main_no_mod.colorbar(
                axes_no_mod[0].contourf(X_gfx_no_mod, Y_gfx_no_mod, T_plot_base_no_mod, levels=levels_main_no_mod,
                                        cmap="hot", extend='max'), ax=axes_no_mod[0])
            cbar_base_no_mod.set_label("T Base (°C)", size=label_fontsize);
            cbar_base_no_mod.ax.tick_params(labelsize=tick_fontsize)
            axes_no_mod[0].set_title("T Base (Sin Módulos)", fontsize=title_fontsize);
            axes_no_mod[0].set_aspect('equal')
            axes_no_mod[0].set_xlabel("x (m)", fontsize=label_fontsize);
            axes_no_mod[0].set_ylabel("y (m)", fontsize=label_fontsize)
            axes_no_mod[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)

            levels_air_no_mod = np.array([t_ambient_inlet_arg - 0.5, t_ambient_inlet_arg + 0.5])
            cbar_air_no_mod = fig_main_no_mod.colorbar(
                axes_no_mod[1].contourf(X_gfx_no_mod, Y_gfx_no_mod, T_air_no_mod, levels=levels_air_no_mod,
                                        cmap="coolwarm"), ax=axes_no_mod[1])
            cbar_air_no_mod.set_label("T Aire (°C)", size=label_fontsize, ticks=[t_ambient_inlet_arg]);
            cbar_air_no_mod.ax.tick_params(labelsize=tick_fontsize)
            axes_no_mod[1].set_title("T Aire (Entrada)", fontsize=title_fontsize);
            axes_no_mod[1].set_aspect('equal')
            axes_no_mod[1].set_xlabel("x (m)", fontsize=label_fontsize);
            axes_no_mod[1].set_ylabel("y (m)", fontsize=label_fontsize)
            axes_no_mod[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)

            fig_main_no_mod.suptitle(f"Visión General (Sin Módulos, T_in={t_ambient_inlet_arg:.1f}°C)",
                                     fontsize=suptitle_fontsize)
            fig_main_no_mod.tight_layout(rect=[0, 0.03, 1, 0.93])
            results['plot_base_data_uri'] = fig_to_data_uri(fig_main_no_mod, tight=True, pad_inches=0.1)

            # --- Gráfico RAW SOLO T_Base (Interactivo) - Caso sin módulos ---
            fig_raw_no_mod, ax_raw_no_mod = plt.subplots(
                figsize=(lx * 100 / 2.54 / 90 * 2.5, ly * 100 / 2.54 / 90 * 2.5), dpi=90)  # Ajustar figsize
            ax_raw_no_mod.contourf(X_gfx_no_mod, Y_gfx_no_mod, T_plot_base_no_mod, levels=levels_main_no_mod,
                                   cmap="hot", extend='neither')
            ax_raw_no_mod.set_xlim(0, lx);
            ax_raw_no_mod.set_ylim(0, ly)
            ax_raw_no_mod.set_aspect('equal');
            ax_raw_no_mod.axis('off')
            fig_raw_no_mod.subplots_adjust(left=0, right=1, top=1, bottom=0)
            results['plot_interactive_raw_uri'] = fig_to_data_uri(fig_raw_no_mod, tight=False, transparent=True)

            # Añadir datos numéricos para interactividad - Caso sin módulos
            results['temperature_matrix'] = T_sol_no_mod
            results['x_coordinates'] = x_coords_gfx_no_mod
            results['y_coordinates'] = y_coords_gfx_no_mod
            results['sim_lx'] = lx;
            results['sim_ly'] = ly
            results['sim_nx'] = nx;
            results['sim_ny'] = ny
            print("[SimCoreV2] Gráficos (sin módulos) y datos interactivos generados.")

        except Exception as e_base_no_mod:
            results['status'] = 'Error';
            results['error_message'] = f"Error en setup sin módulos: {e_base_no_mod}"
            traceback.print_exc()

        print(f"[SimCoreV2] Simulación (sin módulos) completada en {time.time() - start_time_sim:.2f}s.")
        return results

    # --- Ejecución con módulos ---
    # Inicialización de T_solution (disipador) y T_air_solution
    T_solution = np.full((nx, ny), t_ambient_inlet_arg + 10.0)
    T_air_solution = np.full((nx, ny), t_ambient_inlet_arg)

    try:
        k_hs = k_main_heatsink_base_arg  # Conductividad del disipador principal desde argumento
        # t_heatsink_thickness es el espesor del disipador principal (ya es un argumento)

        Q_total_m3_s = q_total_m3_h_arg / 3600.0  # Caudal desde argumento
        m_dot_total_kgs = Q_total_m3_s * rho_aire

        # Corregido: A_conv_disipador dinámico
        A_conv_disipador = lx * ly * 2 + (2 * lx * t_heatsink_thickness) + (2 * ly * t_heatsink_thickness)
        if rth_heatsink > 1e-9 and A_conv_disipador > 1e-9:
            h_avg_disipador = 1.0 / (rth_heatsink * A_conv_disipador)
        else:
            h_avg_disipador = 85.0  # Fallback

        A_placa_base_FDM = lx * ly
        if A_placa_base_FDM > 1e-9:
            h_eff_FDM_heatsink = h_avg_disipador * (A_conv_disipador / A_placa_base_FDM)
        else:
            h_eff_FDM_heatsink = 10.0  # Fallback

        print(f"[SimCoreV2] h_avg_disipador (calculado desde Rth_hs y A_conv): {h_avg_disipador:.2f} W/m^2K")
        print(f"[SimCoreV2] h_eff_FDM_heatsink (aplicado a nodos FDM disipador): {h_eff_FDM_heatsink:.2f} W/m^2K")

        dx = lx / (nx - 1) if nx > 1 else lx
        dy = ly / (ny - 1) if ny > 1 else ly
        dA = dx * dy

        if lx > 1e-9 and nx > 0 and dx > 1e-12:
            m_dot_per_meter_width = m_dot_total_kgs / lx
            m_dot_column_kgs = m_dot_per_meter_width * dx
            m_dot_cp_column = m_dot_column_kgs * cp_aire
        else:
            m_dot_cp_column = 1e-12  # Evitar división por cero
        if m_dot_cp_column < 1e-12: m_dot_cp_column = 1e-12

        N_nodes = nx * ny  # Nodos totales en el disipador principal
        P_total_fuentes_nominal = sum(
            p_val for p_val in specific_chip_powers.values() if p_val is not None and p_val > 0)

        # --- Procesar geometría de módulos y chips ---
        modules_data_sim = []
        Nx_module_footprint_global = max(1, int(round(w_igbt_footprint / dx))) if dx > 1e-12 else 1
        Ny_module_footprint_global = max(1, int(round(h_igbt_footprint / dy))) if dy > 1e-12 else 1
        nx_module_half_global = Nx_module_footprint_global // 2
        ny_module_half_global = Ny_module_footprint_global // 2

        for mod_def_idx, mod_def in enumerate(module_definitions):
            module_id_base = mod_def['id']
            module_center_x, module_center_y = mod_def['center_x'], mod_def['center_y']

            half_w_footprint, half_h_footprint = w_igbt_footprint / 2, h_igbt_footprint / 2
            module_center_x = max(half_w_footprint, min(lx - half_w_footprint, module_center_x))
            module_center_y = max(half_h_footprint, min(ly - half_h_footprint, module_center_y))

            module_center_x_idx_global = int(round(module_center_x / dx)) if dx > 1e-12 else 0
            module_center_y_idx_global = int(round(module_center_y / dy)) if dy > 1e-12 else 0
            module_center_x_idx_global = max(0, min(nx - 1, module_center_x_idx_global))
            module_center_y_idx_global = max(0, min(ny - 1, module_center_y_idx_global))

            footprint_i_min_global = max(0, module_center_x_idx_global - nx_module_half_global)
            footprint_i_max_global = min(nx - 1, module_center_x_idx_global + nx_module_half_global)
            footprint_j_min_global = max(0, module_center_y_idx_global - ny_module_half_global)
            footprint_j_max_global = min(ny - 1, module_center_y_idx_global + ny_module_half_global)

            nx_mod_local = footprint_i_max_global - footprint_i_min_global + 1
            ny_mod_local = footprint_j_max_global - footprint_j_min_global + 1

            module_info = {
                'id': module_id_base, 'center_x_m': module_center_x, 'center_y_m': module_center_y,
                'center_x_idx_global': module_center_x_idx_global, 'center_y_idx_global': module_center_y_idx_global,
                'footprint_i_min_global': footprint_i_min_global, 'footprint_i_max_global': footprint_i_max_global,
                'footprint_j_min_global': footprint_j_min_global, 'footprint_j_max_global': footprint_j_max_global,
                'nx_mod_local': nx_mod_local, 'ny_mod_local': ny_mod_local,
                'chips': [],
                'ntc_abs_x': None, 'ntc_abs_y': None,
                'ntc_i_idx_global': None, 'ntc_j_idx_global': None,
                'ntc_i_idx_local': None, 'ntc_j_idx_local': None,  # NTC index in local module grid
                'T_ntc_final_experimental': np.nan,
                'T_module_internal_solution': np.full((nx_mod_local, ny_mod_local), t_ambient_inlet_arg + 10.0)
                # Temp. on module baseplate surface
            }

            ntc_abs_x_val = module_center_x + NTC_REL_X;
            ntc_abs_y_val = module_center_y + NTC_REL_Y
            module_info['ntc_abs_x'] = ntc_abs_x_val;
            module_info['ntc_abs_y'] = ntc_abs_y_val
            if 0 <= ntc_abs_x_val <= lx and 0 <= ntc_abs_y_val <= ly and dx > 1e-12 and dy > 1e-12:
                ntc_i_idx_global_val = int(round(ntc_abs_x_val / dx))
                ntc_j_idx_global_val = int(round(ntc_abs_y_val / dy))
                ntc_i_idx_global_val = max(0, min(nx - 1, ntc_i_idx_global_val))
                ntc_j_idx_global_val = max(0, min(ny - 1, ntc_j_idx_global_val))
                module_info['ntc_i_idx_global'] = ntc_i_idx_global_val
                module_info['ntc_j_idx_global'] = ntc_j_idx_global_val
                # Check if NTC is within this module's footprint to get local index
                if footprint_i_min_global <= ntc_i_idx_global_val <= footprint_i_max_global and \
                        footprint_j_min_global <= ntc_j_idx_global_val <= footprint_j_max_global:
                    module_info['ntc_i_idx_local'] = ntc_i_idx_global_val - footprint_i_min_global
                    module_info['ntc_j_idx_local'] = ntc_j_idx_global_val - footprint_j_min_global

            for chip_label_suffix, (rel_x_chip, rel_y_chip) in chip_rel_positions.items():
                chip_full_id = f"{module_id_base}_{chip_label_suffix}"
                chip_center_x_m = module_center_x + rel_x_chip;
                chip_center_y_m = module_center_y + rel_y_chip
                chip_center_x_m = max(0, min(lx, chip_center_x_m));
                chip_center_y_m = max(0, min(ly, chip_center_y_m))

                chip_x_idx_global_val = int(round(chip_center_x_m / dx)) if dx > 1e-12 else 0
                chip_y_idx_global_val = int(round(chip_center_y_m / dy)) if dy > 1e-12 else 0
                chip_x_idx_global_val = max(0, min(nx - 1, chip_x_idx_global_val))
                chip_y_idx_global_val = max(0, min(ny - 1, chip_y_idx_global_val))

                is_igbt_chip = "IGBT" in chip_label_suffix;
                chip_type_val = "IGBT" if is_igbt_chip else "Diode"
                if is_igbt_chip:
                    chip_w_m_val, chip_h_m_val, chip_A_val, chip_Rth_val = w_chip_igbt_m, h_chip_igbt_m, A_chip_igbt, Rth_jhs_igbt
                else:
                    chip_w_m_val, chip_h_m_val, chip_A_val, chip_Rth_val = w_chip_diode_m, h_chip_diode_m, A_chip_diode, Rth_jhs_diode

                # Chip cell extent on GLOBAL heatsink grid
                chip_Nx_cells_global_val = max(1, int(round(chip_w_m_val / dx))) if dx > 1e-12 else 1
                chip_Ny_cells_global_val = max(1, int(round(chip_h_m_val / dy))) if dy > 1e-12 else 1
                chip_nx_half_global_val, chip_ny_half_global_val = chip_Nx_cells_global_val // 2, chip_Ny_cells_global_val // 2

                # Chip cell extent on LOCAL module grid (dx, dy are the same for both grids in this model)
                chip_Nx_cells_local_val = chip_Nx_cells_global_val
                chip_Ny_cells_local_val = chip_Ny_cells_global_val
                chip_nx_half_local_val, chip_ny_half_local_val = chip_Nx_cells_local_val // 2, chip_Ny_cells_local_val // 2

                chip_power_val = specific_chip_powers.get(chip_full_id, 0.0)
                chip_q_source_val = chip_power_val / chip_A_val if chip_A_val > 1e-12 else 0.0  # W/m^2 source term (not used directly in this FDM)

                # Chip center index on LOCAL module grid
                chip_x_idx_local_val, chip_y_idx_local_val = -1, -1  # Default if chip center not in module footprint
                if footprint_i_min_global <= chip_x_idx_global_val <= footprint_i_max_global and \
                        footprint_j_min_global <= chip_y_idx_global_val <= footprint_j_max_global:
                    chip_x_idx_local_val = chip_x_idx_global_val - footprint_i_min_global
                    chip_y_idx_local_val = chip_y_idx_global_val - footprint_j_min_global

                chip_data = {'label': chip_full_id, 'suffix': chip_label_suffix, 'type': chip_type_val,
                             'center_x_m': chip_center_x_m, 'center_y_m': chip_center_y_m,
                             'center_x_idx_global': chip_x_idx_global_val, 'center_y_idx_global': chip_y_idx_global_val,
                             'nx_half_global': chip_nx_half_global_val, 'ny_half_global': chip_ny_half_global_val,
                             'center_x_idx_local': chip_x_idx_local_val, 'center_y_idx_local': chip_y_idx_local_val,
                             'nx_half_local': chip_nx_half_local_val, 'ny_half_local': chip_ny_half_local_val,
                             'power': chip_power_val, 'q_source_W_per_m2': chip_q_source_val,  # q_source for reference
                             'Rth_jhs': chip_Rth_val,
                             'T_base_chip_on_heatsink': np.nan, 'T_base_chip_on_module_surface': np.nan,
                             'Tj_from_heatsink': np.nan, 'Tj_from_module_surface': np.nan,
                             # Tj from module surface (verification)
                             'Tj': np.nan,  # Official Tj (from heatsink temp for now)
                             'w_m': chip_w_m_val, 'h_m': chip_h_m_val, 'Area_m2': chip_A_val}
                module_info['chips'].append(chip_data)
            modules_data_sim.append(module_info)

        # --- FDM Setup for Main Heatsink ---
        C_x_cond_hs = k_hs * t_heatsink_thickness * dy / dx if dx > 1e-12 else 0
        C_y_cond_hs = k_hs * t_heatsink_thickness * dx / dy if dy > 1e-12 else 0

        A_matrix_heatsink = lil_matrix((N_nodes, N_nodes))
        # Fill constant part of A_matrix_heatsink (conduction and convection terms)
        for i_hs_node in range(nx):
            for j_hs_node in range(ny):
                idx_hs = i_hs_node * ny + j_hs_node
                C_h_local_hs = h_eff_FDM_heatsink * dA  # Convection from heatsink to air
                coef_diag_hs = -C_h_local_hs
                if i_hs_node > 0: A_matrix_heatsink[idx_hs, idx_hs - ny] = C_x_cond_hs; coef_diag_hs -= C_x_cond_hs
                if i_hs_node < nx - 1: A_matrix_heatsink[idx_hs, idx_hs + ny] = C_x_cond_hs; coef_diag_hs -= C_x_cond_hs
                if j_hs_node > 0: A_matrix_heatsink[idx_hs, idx_hs - 1] = C_y_cond_hs; coef_diag_hs -= C_y_cond_hs
                if j_hs_node < ny - 1: A_matrix_heatsink[idx_hs, idx_hs + 1] = C_y_cond_hs; coef_diag_hs -= C_y_cond_hs
                A_matrix_heatsink[idx_hs, idx_hs] = coef_diag_hs

        A_csr_heatsink = A_matrix_heatsink.tocsr()
        try:
            LU_heatsink = splu(A_csr_heatsink)
        except RuntimeError as e_lu:
            results['status'] = 'Error';
            results['error_message'] = f"Error LU para disipador principal: {e_lu}";
            traceback.print_exc();
            return results

        # --- Iterative Solver: Main Heatsink <-> IGBT Modules ---
        max_iterations = 100;
        convergence_tolerance = 0.01;
        iteration = 0;
        converged = False
        print("[SimCoreV2] Iniciando bucle iterativo Disipador Principal <-> Módulos IGBT...")
        T_solution_old_iter = T_solution.copy()

        while not converged and iteration < max_iterations:
            iteration += 1
            iter_start_time = time.time()

            q_source_map_for_heatsink = np.zeros((nx, ny))  # Heat flux from modules to heatsink nodes (W/m^2)

            # --- 1. Solve for each Module's Baseplate Temperature ---
            for module_item in modules_data_sim:
                nx_mod = module_item['nx_mod_local'];
                ny_mod = module_item['ny_mod_local']
                N_nodes_mod = nx_mod * ny_mod
                if N_nodes_mod == 0: continue  # Skip if module has no nodes (e.g. too small for grid)

                # Heat sources from chips onto the module's baseplate (W/m^2)
                q_chip_source_map_module = np.zeros((nx_mod, ny_mod))
                for chip_item in module_item['chips']:
                    if chip_item['power'] <= 0 or chip_item[
                        'center_x_idx_local'] < 0: continue  # Chip outside or no power

                    # Chip footprint on local module grid
                    i_min_chip_loc = max(0, chip_item['center_x_idx_local'] - chip_item['nx_half_local'])
                    i_max_chip_loc = min(nx_mod - 1, chip_item['center_x_idx_local'] + chip_item['nx_half_local'])
                    j_min_chip_loc = max(0, chip_item['center_y_idx_local'] - chip_item['ny_half_local'])
                    j_max_chip_loc = min(ny_mod - 1, chip_item['center_y_idx_local'] + chip_item['ny_half_local'])

                    num_nodes_chip_on_module = (i_max_chip_loc - i_min_chip_loc + 1) * \
                                               (j_max_chip_loc - j_min_chip_loc + 1)
                    if num_nodes_chip_on_module > 0 and dA > 1e-12:
                        power_per_node_chip = chip_item['power'] / num_nodes_chip_on_module
                        q_density_on_node_chip = power_per_node_chip / dA  # W/m^2
                        q_chip_source_map_module[i_min_chip_loc: i_max_chip_loc + 1, \
                        j_min_chip_loc: j_max_chip_loc + 1] += q_density_on_node_chip

                # FDM for the module's baseplate
                A_matrix_module = lil_matrix((N_nodes_mod, N_nodes_mod))
                b_vector_module = np.zeros(N_nodes_mod)

                C_x_cond_mod = K_MODULE_BASEPLATE * T_MODULE_BASEPLATE * dy / dx if dx > 1e-12 else 0
                C_y_cond_mod = K_MODULE_BASEPLATE * T_MODULE_BASEPLATE * dx / dy if dy > 1e-12 else 0

                for i_mod_node in range(nx_mod):
                    for j_mod_node in range(ny_mod):
                        idx_mod = i_mod_node * ny_mod + j_mod_node

                        # Corresponding global heatsink node index
                        global_i = module_item['footprint_i_min_global'] + i_mod_node
                        global_j = module_item['footprint_j_min_global'] + j_mod_node
                        T_heatsink_under_module_node = T_solution[
                            global_i, global_j]  # Temp of heatsink node below this module node

                        C_h_interface_mod = H_MODULE_TO_HEATSINK_INTERFACE * dA  # Convection from module base to heatsink
                        coef_diag_mod = -C_h_interface_mod

                        q_chip_W_node = q_chip_source_map_module[i_mod_node, j_mod_node] * dA  # Heat from chip in Watts
                        b_vector_module[idx_mod] = -q_chip_W_node - C_h_interface_mod * T_heatsink_under_module_node

                        if i_mod_node > 0: A_matrix_module[
                            idx_mod, idx_mod - ny_mod] = C_x_cond_mod; coef_diag_mod -= C_x_cond_mod
                        if i_mod_node < nx_mod - 1: A_matrix_module[
                            idx_mod, idx_mod + ny_mod] = C_x_cond_mod; coef_diag_mod -= C_x_cond_mod
                        if j_mod_node > 0: A_matrix_module[
                            idx_mod, idx_mod - 1] = C_y_cond_mod; coef_diag_mod -= C_y_cond_mod
                        if j_mod_node < ny_mod - 1: A_matrix_module[
                            idx_mod, idx_mod + 1] = C_y_cond_mod; coef_diag_mod -= C_y_cond_mod
                        A_matrix_module[idx_mod, idx_mod] = coef_diag_mod

                try:
                    LU_mod = splu(A_matrix_module.tocsr())
                    T_flat_mod = LU_mod.solve(b_vector_module)
                    module_item['T_module_internal_solution'] = T_flat_mod.reshape((nx_mod, ny_mod))
                except (RuntimeError, ValueError) as e_solve_mod:
                    results['status'] = 'Error';
                    results[
                        'error_message'] = f"Error solver módulo {module_item['id']} iter {iteration}: {e_solve_mod}";
                    traceback.print_exc();
                    return results

                # Calculate heat transferred from this module to the main heatsink
                for i_mod_node in range(nx_mod):
                    for j_mod_node in range(ny_mod):
                        global_i = module_item['footprint_i_min_global'] + i_mod_node
                        global_j = module_item['footprint_j_min_global'] + j_mod_node
                        T_mod_base_node = module_item['T_module_internal_solution'][i_mod_node, j_mod_node]
                        T_heatsink_node = T_solution[global_i, global_j]  # Heatsink temp from PREVIOUS global iteration

                        P_node_mod_to_hs = H_MODULE_TO_HEATSINK_INTERFACE * dA * (
                                    T_mod_base_node - T_heatsink_node)  # Watts
                        if dA > 1e-12:
                            q_source_map_for_heatsink[global_i, global_j] += P_node_mod_to_hs / dA  # W/m^2

            # --- 2. Solve for Main Heatsink Temperature ---
            b_vector_heatsink = np.zeros(N_nodes)
            for i_hs_node in range(nx):
                for j_hs_node in range(ny):
                    idx_hs = i_hs_node * ny + j_hs_node
                    q_source_from_modules_W_node = q_source_map_for_heatsink[i_hs_node, j_hs_node] * dA
                    T_air_local_hs = T_air_solution[i_hs_node, j_hs_node]  # Air temp from PREVIOUS air iteration
                    b_vector_heatsink[idx_hs] = -q_source_from_modules_W_node - h_eff_FDM_heatsink * dA * T_air_local_hs

            try:
                T_flat_heatsink = LU_heatsink.solve(b_vector_heatsink)
                T_solution_new = T_flat_heatsink.reshape((nx, ny))
            except (RuntimeError, ValueError) as e_solve_hs:
                results['status'] = 'Error';
                results['error_message'] = f"Error solver disipador iter {iteration}: {e_solve_hs}";
                traceback.print_exc();
                return results

            # --- 3. Update Air Temperature Profile ---
            T_air_next_iter = np.full_like(T_air_solution, t_ambient_inlet_arg)  # Start with inlet temp
            delta_T_air_nodes_iter = np.zeros_like(T_solution)  # Temp rise of air per node

            for i_node_tair in range(nx):  # Iterate over columns
                for j_node_tair in range(ny):  # Iterate over rows (flow direction)
                    P_conv_iter = h_eff_FDM_heatsink * dA * \
                                  (T_solution_new[i_node_tair, j_node_tair] - T_air_solution[
                                      i_node_tair, j_node_tair])  # T_air from previous iter
                    P_conv_nodes_iter = max(0, P_conv_iter)  # Only consider heat transferred TO air
                    if m_dot_cp_column > 1e-12:
                        delta_T_air_nodes_iter[i_node_tair, j_node_tair] = P_conv_nodes_iter / m_dot_cp_column
                    else:
                        delta_T_air_nodes_iter[i_node_tair, j_node_tair] = 0

            for i_node_acc in range(nx):  # For each column
                for j_node_acc in range(1, ny):  # Accumulate temperature rise along flow path (y-dir)
                    T_air_next_iter[i_node_acc, j_node_acc] = T_air_next_iter[i_node_acc, j_node_acc - 1] + \
                                                              delta_T_air_nodes_iter[i_node_acc, j_node_acc - 1]

            # --- Check Convergence (based on max change in Heatsink temperature) ---
            if iteration > 1:  # Start checking after the first iteration
                max_diff_T_heatsink = np.max(np.abs(T_solution_new - T_solution_old_iter))
                converged = max_diff_T_heatsink < convergence_tolerance

            T_solution_old_iter = T_solution_new.copy()  # For next iteration's convergence check
            T_solution = T_solution_new.copy()  # Update heatsink temperature
            T_air_solution = T_air_next_iter.copy()  # Update air temperature

            if iteration % 10 == 0 or iteration == 1 or (converged and iteration > 1):
                max_T_sol_iter = np.max(T_solution) if not np.isnan(T_solution).all() else np.nan
                diff_str = f"{max_diff_T_heatsink:.4f}°C" if iteration > 1 else "N/A (1st iter)"
                print(
                    f"[SimCoreV2] Iter {iteration}/{max_iterations} (took {time.time() - iter_start_time:.2f}s): Max ΔT_disipador = {diff_str}. T_max_disipador = {max_T_sol_iter:.2f}°C. Converged: {converged}")

        print(f"[SimCoreV2] Bucle terminado en {iteration} iteraciones. Converged: {converged}")
        results['convergence'] = converged;
        results['iterations'] = iteration

        # --- Post-procesamiento (Cálculo de métricas, Tj, T_ntc) ---
        if np.isnan(T_solution).any():
            results['t_max_base'] = np.nan;
            results['t_avg_base'] = np.nan
            print("[SimCoreV2] ADVERTENCIA: NaN encontrado en la solución del disipador.")
        else:
            results['t_max_base'] = np.max(T_solution)  # Max temp on heatsink surface
            results['t_avg_base'] = np.mean(T_solution)  # Avg temp on heatsink surface

        if ny > 0 and not np.isnan(T_air_solution[:, ny - 1]).any():
            results['t_air_outlet'] = np.mean(T_air_solution[:, ny - 1])  # Avg air outlet temp
        else:
            results['t_air_outlet'] = np.nan

        max_tj_overall = -float('inf');
        max_tj_chip_label = "";
        max_t_ntc_overall = -float('inf')
        max_t_module_surface_overall = -float('inf')  # For plotting zoom range

        # --- Funciones para obtener T_base_chip específicas ---
        def get_hybrid_temp_in_area_local(T_matrix_func, center_x_idx_func, center_y_idx_func,
                                          nx_half_func, ny_half_func, Max_Nx_func, Max_Ny_func):
            i_min_func = max(0, center_x_idx_func - nx_half_func);
            i_max_func = min(Max_Nx_func - 1, center_x_idx_func + nx_half_func)
            j_min_func = max(0, center_y_idx_func - ny_half_func);
            j_max_func = min(Max_Ny_func - 1, center_y_idx_func + ny_half_func)
            if i_min_func > i_max_func or j_min_func > j_max_func: return np.nan

            i_max_adj_func = min(i_max_func, T_matrix_func.shape[0] - 1);
            j_max_adj_func = min(j_max_func, T_matrix_func.shape[1] - 1)
            if i_min_func > i_max_adj_func or j_min_func > j_max_adj_func: return np.nan  # Check after adjustment too

            area_nodes_func = T_matrix_func[i_min_func: i_max_adj_func + 1, j_min_func: j_max_adj_func + 1]
            if area_nodes_func.size == 0: return np.nan

            t_max_area = np.nanmax(area_nodes_func)
            t_avg_area = np.nanmean(area_nodes_func)
            if np.isnan(t_max_area) or np.isnan(t_avg_area): return np.nan
            return (t_max_area + t_avg_area) / 2.0

        def get_avg_temp_in_area_local(T_matrix_func, center_x_idx_func, center_y_idx_func,
                                       nx_half_func, ny_half_func, Max_Nx_func, Max_Ny_func):
            i_min_func = max(0, center_x_idx_func - nx_half_func);
            i_max_func = min(Max_Nx_func - 1, center_x_idx_func + nx_half_func)
            j_min_func = max(0, center_y_idx_func - ny_half_func);
            j_max_func = min(Max_Ny_func - 1, center_y_idx_func + ny_half_func)
            if i_min_func > i_max_func or j_min_func > j_max_func: return np.nan

            i_max_adj_func = min(i_max_func, T_matrix_func.shape[0] - 1);
            j_max_adj_func = min(j_max_func, T_matrix_func.shape[1] - 1)
            if i_min_func > i_max_adj_func or j_min_func > j_max_adj_func: return np.nan

            area_nodes_func = T_matrix_func[i_min_func: i_max_adj_func + 1, j_min_func: j_max_adj_func + 1]
            if area_nodes_func.size == 0: return np.nan
            return np.nanmean(area_nodes_func)

        module_results_list = []
        print("\n--- [NTC Calculation Details] ---")
        for module_item_post in modules_data_sim:
            module_result_entry = {'id': module_item_post['id'], 'chips': [], 't_ntc': np.nan}
            T_module_internal_map = module_item_post['T_module_internal_solution']
            if not np.isnan(T_module_internal_map).all():
                max_t_module_surface_overall = np.nanmax(
                    [max_t_module_surface_overall, np.nanmax(T_module_internal_map)])

            # Get power distribution for NTC Rth selection
            module_chip_powers_ordered = {}
            for chip_item_post_pwr in module_item_post['chips']:
                module_chip_powers_ordered[chip_item_post_pwr['suffix']] = chip_item_post_pwr['power']

            actual_power_values = [module_chip_powers_ordered.get(s, 0.0) for s in CHIP_ORDER_FOR_RTH_TABLE]
            total_module_power = sum(actual_power_values)
            actual_power_distribution_normalized = [0.0] * len(CHIP_ORDER_FOR_RTH_TABLE)
            if total_module_power > 1e-6:
                actual_power_distribution_normalized = [p / total_module_power for p in actual_power_values]

            print(f"Module ID: {module_item_post['id']}")
            print(f"  Actual Powers (Order: {CHIP_ORDER_FOR_RTH_TABLE}): {actual_power_values}")
            print(f"  Total Module Power: {total_module_power:.2f} W")
            print(f"  Normalized Power Distribution: {[f'{p:.2f}' for p in actual_power_distribution_normalized]}")
            all_ntc_temps_for_module = []

            for chip_item_post in module_item_post['chips']:
                is_igbt_chip_type = chip_item_post['type'] == "IGBT"

                # T_base on HEATSINK surface under the chip
                if is_igbt_chip_type:
                    T_base_chip_on_heatsink_val = get_hybrid_temp_in_area_local(
                        T_solution, chip_item_post['center_x_idx_global'], chip_item_post['center_y_idx_global'],
                        chip_item_post['nx_half_global'], chip_item_post['ny_half_global'], nx, ny)
                else:  # Diode
                    T_base_chip_on_heatsink_val = get_avg_temp_in_area_local(
                        T_solution, chip_item_post['center_x_idx_global'], chip_item_post['center_y_idx_global'],
                        chip_item_post['nx_half_global'], chip_item_post['ny_half_global'], nx, ny)
                chip_item_post['T_base_chip_on_heatsink'] = T_base_chip_on_heatsink_val

                # T_base on MODULE surface under the chip (for verification/info)
                if chip_item_post['center_x_idx_local'] >= 0:  # Chip is within module footprint
                    if is_igbt_chip_type:
                        T_base_chip_on_module_surface_val = get_hybrid_temp_in_area_local(
                            T_module_internal_map, chip_item_post['center_x_idx_local'],
                            chip_item_post['center_y_idx_local'],
                            chip_item_post['nx_half_local'], chip_item_post['ny_half_local'],
                            module_item_post['nx_mod_local'], module_item_post['ny_mod_local'])
                    else:  # Diode
                        T_base_chip_on_module_surface_val = get_avg_temp_in_area_local(
                            T_module_internal_map, chip_item_post['center_x_idx_local'],
                            chip_item_post['center_y_idx_local'],
                            chip_item_post['nx_half_local'], chip_item_post['ny_half_local'],
                            module_item_post['nx_mod_local'], module_item_post['ny_mod_local'])
                else:
                    T_base_chip_on_module_surface_val = np.nan
                chip_item_post['T_base_chip_on_module_surface'] = T_base_chip_on_module_surface_val

                # Tj calculated from T_base_chip_on_heatsink (this is the "official" Tj)
                Tj_from_heatsink_val = np.nan
                if not np.isnan(T_base_chip_on_heatsink_val):
                    if chip_item_post['power'] > 1e-6 and not np.isnan(chip_item_post['Rth_jhs']):
                        Tj_from_heatsink_val = T_base_chip_on_heatsink_val + chip_item_post['Rth_jhs'] * chip_item_post[
                            'power']
                    else:  # No power or no Rth, Tj = Tbase
                        Tj_from_heatsink_val = T_base_chip_on_heatsink_val
                chip_item_post['Tj_from_heatsink'] = Tj_from_heatsink_val
                chip_item_post['Tj'] = Tj_from_heatsink_val  # Set official Tj

                # Tj calculated from T_base_chip_on_module_surface (for verification/info)
                Tj_from_module_surface_val = np.nan
                if not np.isnan(T_base_chip_on_module_surface_val):
                    if chip_item_post['power'] > 1e-6 and not np.isnan(chip_item_post['Rth_jhs']):
                        Tj_from_module_surface_val = T_base_chip_on_module_surface_val + chip_item_post['Rth_jhs'] * \
                                                     chip_item_post['power']
                    else:
                        Tj_from_module_surface_val = T_base_chip_on_module_surface_val
                chip_item_post['Tj_from_module_surface'] = Tj_from_module_surface_val

                if not np.isnan(chip_item_post['Tj']):
                    if chip_item_post['Tj'] > max_tj_overall:
                        max_tj_overall = chip_item_post['Tj']
                        max_tj_chip_label = f"{chip_item_post['label']} (P={chip_item_post['power']:.1f}W)"

                # NTC Temperature Estimation
                chip_suffix = chip_item_post['suffix']
                chip_power_actual = chip_item_post['power']
                Tj_chip_simulated = chip_item_post['Tj']  # Use the official Tj

                if chip_suffix in EXPERIMENTAL_RTH_NTC_DATA and chip_power_actual > 1e-6 and not np.isnan(
                        Tj_chip_simulated):
                    experimental_data_for_source_chip = EXPERIMENTAL_RTH_NTC_DATA[chip_suffix]
                    selected_rth_j_ntc, closest_exp_dist = find_closest_experimental_rth(
                        actual_power_distribution_normalized, experimental_data_for_source_chip)

                    if not np.isnan(selected_rth_j_ntc):
                        T_ntc_estimate_from_chip = Tj_chip_simulated - chip_power_actual * selected_rth_j_ntc
                        all_ntc_temps_for_module.append(T_ntc_estimate_from_chip)
                        print(f"    Chip {chip_suffix} (Tj={Tj_chip_simulated:.2f}°C, P={chip_power_actual:.1f}W):")
                        print(
                            f"      Selected Rth_{chip_suffix}-NTC = {selected_rth_j_ntc:.5f} K/W (for exp. dist: {[f'{p:.2f}' for p in closest_exp_dist]})")
                        print(f"      Estimated T_NTC from this chip = {T_ntc_estimate_from_chip:.2f}°C")
                    else:
                        print(
                            f"    Chip {chip_suffix}: No suitable Rth_j-NTC found for its power distribution or P=0/Tj=NaN.")
                # ... (else clauses for NTC skipping reasons) ...

                module_result_entry['chips'].append({
                    'suffix': chip_item_post['suffix'],
                    't_base_heatsink': T_base_chip_on_heatsink_val,
                    't_base_module_surface': T_base_chip_on_module_surface_val,
                    'tj_from_heatsink': Tj_from_heatsink_val,  # Official Tj source
                    'tj_from_module_surface': Tj_from_module_surface_val,  # Verification
                    'tj': chip_item_post['Tj']  # Official Tj
                })

            T_ntc_final_experimental_module = np.nan
            if all_ntc_temps_for_module:
                T_ntc_final_experimental_module = np.mean(all_ntc_temps_for_module)
                min_T_ntc_est = np.min(all_ntc_temps_for_module);
                max_T_ntc_est = np.max(all_ntc_temps_for_module)
                print(
                    f"  -> Module {module_item_post['id']} T_NTC Estimates (Min,Max,Range): {min_T_ntc_est:.2f}, {max_T_ntc_est:.2f}, {max_T_ntc_est - min_T_ntc_est:.2f}°C")
                print(
                    f"  ==> Module {module_item_post['id']} Final Avg T_NTC (Experimental) = {T_ntc_final_experimental_module:.2f}°C")
            else:
                print(f"  -> Module {module_item_post['id']}: No T_NTC estimates generated. Final T_NTC is NaN.")

            module_item_post['T_ntc_final_experimental'] = T_ntc_final_experimental_module
            module_result_entry['t_ntc'] = T_ntc_final_experimental_module
            if not np.isnan(T_ntc_final_experimental_module):
                max_t_ntc_overall = np.nanmax([max_t_ntc_overall, T_ntc_final_experimental_module])
            print("-" * 30)
            module_results_list.append(module_result_entry)

        print("--- [End NTC Calculation Details] ---\n")
        results['t_max_junction'] = max_tj_overall if not np.isinf(max_tj_overall) else np.nan
        results['t_max_junction_chip'] = max_tj_chip_label
        results['t_max_ntc'] = max_t_ntc_overall if not np.isinf(max_t_ntc_overall) else np.nan
        results['module_results'] = module_results_list
        if np.isinf(max_t_module_surface_overall) or np.isnan(max_t_module_surface_overall):
            max_t_module_surface_overall = t_ambient_inlet_arg + 10  # Fallback for plot scaling

        # --- Generación de Gráficos ---
        print("[SimCoreV2] Generating Graphics...")
        x_coords_gfx_main_hs = np.linspace(0, lx, nx)  # Coords for main heatsink plot
        y_coords_gfx_main_hs = np.linspace(0, ly, ny)
        X_gfx_main_hs, Y_gfx_main_hs = np.meshgrid(x_coords_gfx_main_hs, y_coords_gfx_main_hs, indexing='ij')

        # --- 1. Gráfico Combinado Disipador Principal / Aire (Estático) ---
        try:
            fig_main_gfx, axes_main_gfx = plt.subplots(1, 2, figsize=(11, 4.5))  # Standard size
            title_fontsize = 12;
            label_fontsize = 10;
            tick_fontsize = 9;
            suptitle_fontsize = 11  # smaller suptitle

            # Subplot T Disipador Principal
            min_T_plot_hs = t_ambient_inlet_arg
            max_T_plot_hs_val = results['t_max_base'] if not np.isnan(
                results['t_max_base']) else t_ambient_inlet_arg + 1
            max_T_plot_hs_val = max(max_T_plot_hs_val, min_T_plot_hs + 1.0)  # Ensure some range
            max_T_plot_hs_val += (max_T_plot_hs_val - min_T_plot_hs) * 0.05  # Add margin
            levels_hs_plot = np.linspace(min_T_plot_hs, max_T_plot_hs_val, 30) if abs(
                max_T_plot_hs_val - min_T_plot_hs) > 1e-3 else np.linspace(min_T_plot_hs - 0.5, max_T_plot_hs_val + 0.5,
                                                                           2)
            T_solution_plot_hs = np.nan_to_num(T_solution, nan=min_T_plot_hs - 10)  # T_solution is heatsink temp matrix

            cbar_hs = fig_main_gfx.colorbar(
                axes_main_gfx[0].contourf(X_gfx_main_hs, Y_gfx_main_hs, T_solution_plot_hs, levels=levels_hs_plot,
                                          cmap="hot", extend='max'), ax=axes_main_gfx[0])
            cbar_hs.set_label("T Disipador (°C)", size=label_fontsize);
            cbar_hs.ax.tick_params(labelsize=tick_fontsize)
            tmax_hs_str = f"{results['t_max_base']:.1f}°C" if not np.isnan(results['t_max_base']) else "N/A"
            axes_main_gfx[0].set_title(f"T Disipador Principal (Max={tmax_hs_str})", fontsize=title_fontsize);
            axes_main_gfx[0].set_aspect('equal')
            axes_main_gfx[0].set_xlabel("x (m)", fontsize=label_fontsize);
            axes_main_gfx[0].set_ylabel("y (m)", fontsize=label_fontsize)
            axes_main_gfx[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)

            # Subplot T Aire
            min_T_air_plot = t_ambient_inlet_arg
            T_air_max_plot_val = np.nanmax(T_air_solution) if not np.isnan(
                T_air_solution).all() else t_ambient_inlet_arg
            max_T_air_plot_val = max(T_air_max_plot_val, min_T_air_plot + 0.1)
            max_T_air_plot_val += (max_T_air_plot_val - min_T_air_plot) * 0.05
            levels_air_plot = np.linspace(min_T_air_plot, max_T_air_plot_val, 30) if abs(
                max_T_air_plot_val - min_T_air_plot) > 1e-3 else np.linspace(min_T_air_plot - 0.5,
                                                                             max_T_air_plot_val + 0.5, 2)
            T_air_solution_plot = np.nan_to_num(T_air_solution, nan=min_T_air_plot - 10)

            cbar_air = fig_main_gfx.colorbar(
                axes_main_gfx[1].contourf(X_gfx_main_hs, Y_gfx_main_hs, T_air_solution_plot, levels=levels_air_plot,
                                          cmap="coolwarm", extend='max'), ax=axes_main_gfx[1])
            cbar_air.set_label("T Aire (°C)", size=label_fontsize);
            cbar_air.ax.tick_params(labelsize=tick_fontsize)
            tairout_str = f"{results['t_air_outlet']:.1f}°C" if not np.isnan(results['t_air_outlet']) else "N/A"
            axes_main_gfx[1].set_title(f"T Aire (Salida Prom={tairout_str})", fontsize=title_fontsize);
            axes_main_gfx[1].set_aspect('equal')
            axes_main_gfx[1].set_xlabel("x (m)", fontsize=label_fontsize);
            axes_main_gfx[1].set_ylabel("y (m)", fontsize=label_fontsize)
            axes_main_gfx[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)

            # Overlays de módulos y NTC
            for ax_item_main_gfx in axes_main_gfx:
                for module_item_main_gfx in modules_data_sim:
                    center_x_mod_main_gfx, center_y_mod_main_gfx = module_item_main_gfx['center_x_m'], \
                    module_item_main_gfx['center_y_m']
                    rect_x_mod_main_gfx, rect_y_mod_main_gfx = center_x_mod_main_gfx - w_igbt_footprint / 2, center_y_mod_main_gfx - h_igbt_footprint / 2
                    rect_mod_main_plot_gfx = plt.Rectangle((rect_x_mod_main_gfx, rect_y_mod_main_gfx), w_igbt_footprint,
                                                           h_igbt_footprint, edgecolor='gray', facecolor='none', lw=0.8,
                                                           ls='--')
                    ax_item_main_gfx.add_patch(rect_mod_main_plot_gfx)
                    if module_item_main_gfx['ntc_abs_x'] is not None and module_item_main_gfx[
                        'ntc_abs_y'] is not None and \
                            0 <= module_item_main_gfx['ntc_abs_x'] <= lx and 0 <= module_item_main_gfx[
                        'ntc_abs_y'] <= ly:
                        ax_item_main_gfx.plot(module_item_main_gfx['ntc_abs_x'], module_item_main_gfx['ntc_abs_y'],
                                              'wo', markersize=3, markeredgecolor='black')

            fig_main_gfx.suptitle(
                f"Simulación ({len(modules_data_sim)} Módulos, P={P_total_fuentes_nominal:.0f}W, T_in={t_ambient_inlet_arg:.1f}°C, Q={q_total_m3_h_arg:.0f}m³/h)\nTj(IGBT:híbrido,Diode:avg), T_NTC(exp.)",
                fontsize=suptitle_fontsize)
            fig_main_gfx.tight_layout(rect=[0, 0.03, 1, 0.90])  # Adjust top for suptitle
            results['plot_base_data_uri'] = fig_to_data_uri(fig_main_gfx, tight=True, pad_inches=0.1)
        except Exception as e_gfx_main:
            print(f"[SimCoreV2] Error GFX Main: {e_gfx_main}");
            traceback.print_exc()
            results['error_message'] = (results.get('error_message') or "") + f"; GFX Main Err: {e_gfx_main}"

        # --- 2. Gráfico RAW SOLO T_Disipador (Interactivo) ---
        try:
            print("[SimCoreV2] Generating RAW Interactive plot for Heatsink Base...")
            # T_solution es la temperatura del disipador principal
            # x_coords_gfx_main_hs, y_coords_gfx_main_hs, X_gfx_main_hs, Y_gfx_main_hs son las correctas
            # levels_hs_plot son los niveles correctos para T_solution

            fig_width_inches_interactive = lx * (90 / 2.54) / 2.5  # Ajustado ligeramente
            fig_height_inches_interactive = ly * (90 / 2.54) / 2.5
            min_fig_dim_inches = 1.0
            fig_width_inches_interactive = max(min_fig_dim_inches, fig_width_inches_interactive)
            fig_height_inches_interactive = max(min_fig_dim_inches, fig_height_inches_interactive)

            fig_raw_interactive, ax_raw_interactive = plt.subplots(
                figsize=(fig_width_inches_interactive, fig_height_inches_interactive), dpi=90)

            ax_raw_interactive.contourf(X_gfx_main_hs, Y_gfx_main_hs, T_solution_plot_hs, levels=levels_hs_plot,
                                        cmap="hot", extend='neither')  # Use T_solution_plot_hs
            ax_raw_interactive.set_xlim(0, lx);
            ax_raw_interactive.set_ylim(0, ly)
            ax_raw_interactive.set_aspect('equal');
            ax_raw_interactive.axis('off')
            fig_raw_interactive.subplots_adjust(left=0, right=1, top=1, bottom=0)

            results['plot_interactive_raw_uri'] = fig_to_data_uri(fig_raw_interactive, tight=False, transparent=True)
            print("[SimCoreV2] Gráfico RAW interactivo (Heatsink) generado.")
        except Exception as e_gfx_raw_interactive:
            print(f"[SimCoreV2] Error GFX RAW Interactivo (Heatsink): {e_gfx_raw_interactive}");
            traceback.print_exc()
            results['error_message'] = (results.get(
                'error_message') or "") + f"; GFX Interactive RAW Err: {e_gfx_raw_interactive}"
            results['plot_interactive_raw_uri'] = None

        # --- 3. Gráfico Zoom por Módulo (Temperatura Superficie Módulo) (Estático) ---
        try:
            zoom_title_fontsize = 10;
            zoom_label_fontsize = 8;
            zoom_tick_fontsize = 7;
            zoom_suptitle_fontsize = 10;
            chip_text_fontsize = 6;
            ntc_text_fontsize = 7;
            ntc_markersize = 4
            num_modules_plot_zoom = len(modules_data_sim)
            if num_modules_plot_zoom == 0: raise ValueError("No modules to plot zoom for.")

            ncols_zoom_gfx = min(3, num_modules_plot_zoom);
            nrows_zoom_gfx = math.ceil(num_modules_plot_zoom / ncols_zoom_gfx)
            fig_zoom_gfx, axes_zoom_list_gfx = plt.subplots(nrows=nrows_zoom_gfx, ncols=ncols_zoom_gfx,
                                                            figsize=(ncols_zoom_gfx * 3.8, nrows_zoom_gfx * 3.5),
                                                            squeeze=False)
            axes_flat_zoom_gfx = axes_zoom_list_gfx.flatten();
            contour_zoom_ref_plot_gfx = None

            min_T_plot_zoom_mod_gfx = t_ambient_inlet_arg  # Base para la escala de colores
            # Usar max_t_module_surface_overall y Tj max para una buena escala
            max_temp_for_scale = max(max_t_module_surface_overall if not (
                        np.isnan(max_t_module_surface_overall) or np.isinf(
                    max_t_module_surface_overall)) else min_T_plot_zoom_mod_gfx,
                                     results.get('t_max_junction', min_T_plot_zoom_mod_gfx))
            if np.isnan(max_temp_for_scale) or np.isinf(
                max_temp_for_scale): max_temp_for_scale = min_T_plot_zoom_mod_gfx + 20

            max_T_plot_zoom_mod_val_gfx = max(max_temp_for_scale, min_T_plot_zoom_mod_gfx + 1.0)
            levels_zoom_plot_mod_gfx = np.linspace(min_T_plot_zoom_mod_gfx, max_T_plot_zoom_mod_val_gfx, 30) if abs(
                max_T_plot_zoom_mod_val_gfx - min_T_plot_zoom_mod_gfx) > 1e-3 else np.linspace(
                min_T_plot_zoom_mod_gfx - 0.5, max_T_plot_zoom_mod_val_gfx + 0.5, 2)

            for idx_zoom_gfx, module_item_zoom_gfx in enumerate(modules_data_sim):
                if idx_zoom_gfx >= len(axes_flat_zoom_gfx): break
                ax_z_plot_gfx = axes_flat_zoom_gfx[idx_zoom_gfx]

                T_module_internal_plot = module_item_zoom_gfx['T_module_internal_solution']
                nx_mod_loc_plot = module_item_zoom_gfx['nx_mod_local'];
                ny_mod_loc_plot = module_item_zoom_gfx['ny_mod_local']

                # Coordenadas para el plot del módulo individual
                mod_origin_x_global = module_item_zoom_gfx['center_x_m'] - w_igbt_footprint / 2
                mod_origin_y_global = module_item_zoom_gfx['center_y_m'] - h_igbt_footprint / 2
                mod_end_x_global = module_item_zoom_gfx['center_x_m'] + w_igbt_footprint / 2
                mod_end_y_global = module_item_zoom_gfx['center_y_m'] + h_igbt_footprint / 2

                x_coords_mod_local_gfx = np.linspace(mod_origin_x_global, mod_end_x_global, nx_mod_loc_plot)
                y_coords_mod_local_gfx = np.linspace(mod_origin_y_global, mod_end_y_global, ny_mod_loc_plot)
                X_mod_local_gfx, Y_mod_local_gfx = np.meshgrid(x_coords_mod_local_gfx, y_coords_mod_local_gfx,
                                                               indexing='ij')

                T_module_plot_safe = np.nan_to_num(T_module_internal_plot, nan=min_T_plot_zoom_mod_gfx - 10)
                contour_zoom_gfx_detail = ax_z_plot_gfx.contourf(X_mod_local_gfx, Y_mod_local_gfx, T_module_plot_safe,
                                                                 levels=levels_zoom_plot_mod_gfx, cmap="hot",
                                                                 extend='max')
                if idx_zoom_gfx == 0: contour_zoom_ref_plot_gfx = contour_zoom_gfx_detail

                # Footprint del módulo
                rect_mod_zoom_plot_gfx_obj = plt.Rectangle((mod_origin_x_global, mod_origin_y_global), w_igbt_footprint,
                                                           h_igbt_footprint, edgecolor='black', facecolor='none',
                                                           lw=1.0, ls='--')
                ax_z_plot_gfx.add_patch(rect_mod_zoom_plot_gfx_obj)

                current_module_result_for_plot = next(
                    (mr for mr in module_results_list if mr['id'] == module_item_zoom_gfx['id']), None)
                if current_module_result_for_plot:
                    for chip_plot_info in current_module_result_for_plot['chips']:
                        original_chip_item = next(
                            (ci for ci in module_item_zoom_gfx['chips'] if ci['suffix'] == chip_plot_info['suffix']),
                            None)
                        if not original_chip_item: continue

                        center_x_phys_zoom_gfx, center_y_phys_zoom_gfx = original_chip_item['center_x_m'], \
                        original_chip_item['center_y_m']
                        chip_w_plot_zoom_gfx, chip_h_plot_zoom_gfx = original_chip_item['w_m'], original_chip_item[
                            'h_m']
                        rect_x_phys_zoom_gfx, rect_y_phys_zoom_gfx = center_x_phys_zoom_gfx - chip_w_plot_zoom_gfx / 2, center_y_phys_zoom_gfx - chip_h_plot_zoom_gfx / 2
                        is_igbt_zoom_plot_gfx = original_chip_item['type'] == "IGBT"

                        rect_chip_zoom_plot_gfx_obj = plt.Rectangle((rect_x_phys_zoom_gfx, rect_y_phys_zoom_gfx),
                                                                    chip_w_plot_zoom_gfx, chip_h_plot_zoom_gfx,
                                                                    edgecolor='cyan' if is_igbt_zoom_plot_gfx else 'lime',
                                                                    facecolor='none', lw=1.0, ls=':')
                        ax_z_plot_gfx.add_patch(rect_chip_zoom_plot_gfx_obj)

                        tj_val = chip_plot_info['tj']  # Official Tj
                        tbase_mod_val = chip_plot_info['t_base_module_surface']  # T en la sup. del modulo
                        tj_str_zoom_gfx = f"Tj={tj_val:.1f}" if not np.isnan(tj_val) else "Tj=N/A"
                        base_label_mod = "Tb_mod_HÍB" if is_igbt_zoom_plot_gfx else "Tb_mod_AVG"
                        tbase_str_zoom_gfx = f"{base_label_mod}={tbase_mod_val:.1f}" if not np.isnan(
                            tbase_mod_val) else f"{base_label_mod}=N/A"
                        ax_z_plot_gfx.text(center_x_phys_zoom_gfx, center_y_phys_zoom_gfx,
                                           f"{original_chip_item['suffix']}\n{tj_str_zoom_gfx}\n({tbase_str_zoom_gfx})",
                                           color='white', ha='center', va='center', fontsize=chip_text_fontsize,
                                           bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.6))

                # NTC point
                if module_item_zoom_gfx['ntc_abs_x'] is not None and module_item_zoom_gfx['ntc_abs_y'] is not None:
                    ntc_x_plot_zoom_gfx, ntc_y_plot_zoom_gfx = module_item_zoom_gfx['ntc_abs_x'], module_item_zoom_gfx[
                        'ntc_abs_y']
                    ax_z_plot_gfx.plot(ntc_x_plot_zoom_gfx, ntc_y_plot_zoom_gfx, 'ro', markersize=ntc_markersize,
                                       markeredgecolor='white')
                    t_ntc_val_plot_zoom_gfx = module_item_zoom_gfx['T_ntc_final_experimental']
                    t_ntc_str_plot_zoom_gfx = f"NTC≈{t_ntc_val_plot_zoom_gfx:.1f}" if not np.isnan(
                        t_ntc_val_plot_zoom_gfx) else "NTC=N/A"
                    ax_z_plot_gfx.text(ntc_x_plot_zoom_gfx + (mod_end_x_global - mod_origin_x_global) * 0.03,
                                       ntc_y_plot_zoom_gfx, t_ntc_str_plot_zoom_gfx, color='red', ha='left',
                                       va='center', fontsize=ntc_text_fontsize,
                                       bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.7))

                ax_z_plot_gfx.set_xlim(mod_origin_x_global, mod_end_x_global)
                ax_z_plot_gfx.set_ylim(mod_origin_y_global, mod_end_y_global)
                ax_z_plot_gfx.set_title(f"{module_item_zoom_gfx['id']} (T Módulo Sup.)", fontsize=zoom_title_fontsize)
                ax_z_plot_gfx.set_xlabel("x (m)", fontsize=zoom_label_fontsize);
                ax_z_plot_gfx.set_ylabel("y (m)", fontsize=zoom_label_fontsize)
                ax_z_plot_gfx.tick_params(axis='both', which='major', labelsize=zoom_tick_fontsize);
                ax_z_plot_gfx.set_aspect('equal', adjustable='box');
                ax_z_plot_gfx.grid(True, linestyle=':', alpha=0.3)

            for i_ax_flat_zoom_gfx in range(idx_zoom_gfx + 1, len(axes_flat_zoom_gfx)): axes_flat_zoom_gfx[
                i_ax_flat_zoom_gfx].axis('off')
            if contour_zoom_ref_plot_gfx is not None and num_modules_plot_zoom > 0:
                fig_zoom_gfx.subplots_adjust(right=0.85, top=0.90, bottom=0.1);
                cbar_ax_zoom_gfx = fig_zoom_gfx.add_axes([0.88, 0.15, 0.03, 0.7])
                cbar_zoom_gfx = fig_zoom_gfx.colorbar(contour_zoom_ref_plot_gfx, cax=cbar_ax_zoom_gfx)
                cbar_zoom_gfx.set_label("T Módulo Sup. (°C)", size=zoom_label_fontsize)
                cbar_zoom_gfx.ax.tick_params(labelsize=zoom_tick_fontsize)
            fig_zoom_gfx.suptitle(f"Detalle Módulos (Temperatura Superficie Módulo)", fontsize=zoom_suptitle_fontsize)
            results['plot_zoom_data_uri'] = fig_to_data_uri(fig_zoom_gfx, tight=True, pad_inches=0.1)
        except Exception as e_gfx_zoom:
            print(f"[SimCoreV2] Error GFX Zoom: {e_gfx_zoom}");
            traceback.print_exc()
            results['error_message'] = (results.get('error_message') or "") + f"; GFX Zoom Err: {e_gfx_zoom}"

        results['status'] = 'Success'
        print("[SimCoreV2] Generación de gráficos completada.")

        # --- Añadir datos numéricos para interactividad (Disipador Principal) ---
        if results['status'] == 'Success':
            try:
                results['temperature_matrix'] = T_solution  # Matriz de temp del disipador
                results['x_coordinates'] = x_coords_gfx_main_hs  # Coordenadas del disipador
                results['y_coordinates'] = y_coords_gfx_main_hs
                results['sim_lx'] = lx;
                results['sim_ly'] = ly
                results['sim_nx'] = nx;
                results['sim_ny'] = ny  # Dimensiones de la malla del disipador
                print("[SimCoreV2] Datos numéricos del disipador principal añadidos para interactividad.")
            except NameError as e_data:
                print(f"[SimCoreV2] Advertencia: No se pudieron añadir datos numéricos interactivos. Error: {e_data}")
                if results.get('error_message') is None: results['error_message'] = ''
                results['error_message'] += "; Warn: Datos interactivos no generados"

    except Exception as e_general:
        print(f"[SimCoreV2] Error general en simulación: {e_general}")
        traceback.print_exc()
        results['status'] = 'Error'
        if results.get('error_message') is None: results[
            'error_message'] = f"Error inesperado en simulador: {e_general}"

    print(f"[SimCoreV2] Simulación completada en {time.time() - start_time_sim:.2f}s. Estado: {results['status']}")
    return results


def calculate_heatsink_contact_area(hs_length: float, hs_width: float,
                                    fin_height: float, fin_width: float, num_fins: int,
                                    hollow_fin_length: float, hollow_fin_width: float, num_hollow_fins: int):
    """
    Calculates the total wetted contact area of a heatsink with optional fins and hollows within those fins.

    Args:
        hs_length: Length of the heatsink base (X-direction, typically airflow direction).
        hs_width: Width of the heatsink base (Y-direction, along which fins are placed).
        fin_height: Height of a single fin.
        fin_width: Thickness of a single fin.
        num_fins: Total number of fins on the heatsink.
        hollow_fin_length: Length of a single hollow channel along the main fin's length (hs_length).
        hollow_fin_width: Width of a single hollow channel (dimension penetrating fin_height).
        num_hollow_fins: Number of identical hollow channels per main fin.

    Returns:
        A dictionary containing 'contact_area' (float) on success, or 'error' (str) on failure.
    """

    # --- Input Validation ---
    if not (isinstance(hs_length, (int, float)) and hs_length > 0):
        return {'error': 'Invalid input: Heatsink Length (hs_length) must be a positive number.'}
    if not (isinstance(hs_width, (int, float)) and hs_width > 0):
        return {'error': 'Invalid input: Heatsink Width (hs_width) must be a positive number.'}
    # fin_height and fin_width are validated later if num_fins > 0

    if not isinstance(num_fins, int) or num_fins < 0:
        return {'error': 'Invalid input: Number of Fins (num_fins) must be a non-negative integer.'}

    if num_fins > 0:
        if not (isinstance(fin_height, (int, float)) and fin_height > 0):
            return {'error': 'Invalid input: Fin Height (fin_height) must be positive if num_fins > 0.'}
        if not (isinstance(fin_width, (int, float)) and fin_width > 0):
            return {'error': 'Invalid input: Fin Width (fin_width) must be positive if num_fins > 0.'}
        # Fin Placement Constraint
        if (num_fins * fin_width) > hs_width:
            return {'error': f'Fins are wider than the heatsink. Total fin width: {num_fins * fin_width:.4f}m, Heatsink width: {hs_width:.4f}m.'}

    if not isinstance(num_hollow_fins, int) or num_hollow_fins < 0:
        return {'error': 'Invalid input: Number of Hollow Fins (num_hollow_fins) must be a non-negative integer.'}

    if num_hollow_fins > 0:
        if num_fins == 0: # Cannot have hollow fins without main fins
             return {'error': 'Invalid input: Cannot have hollow fins if there are no main fins (num_fins is 0).'}
        if not (isinstance(hollow_fin_length, (int, float)) and hollow_fin_length > 0):
            return {'error': 'Invalid input: Hollow Fin Length must be positive if num_hollow_fins > 0.'}
        if not (isinstance(hollow_fin_width, (int, float)) and hollow_fin_width > 0):
            return {'error': 'Invalid input: Hollow Fin Width must be positive if num_hollow_fins > 0.'}

        # Hollow Fin Placement Constraints
        # Total length of hollow sections on one side of a fin cannot exceed fin length
        if (num_hollow_fins * hollow_fin_length) > hs_length:
            return {'error': f'Hollow fin total length exceeds main fin length. Hollows: {num_hollow_fins * hollow_fin_length:.4f}m, Main fin: {hs_length:.4f}m.'}
        # Width of a single hollow channel cannot exceed fin height
        if hollow_fin_width > fin_height:
            return {'error': f'Hollow fin width exceeds main fin height. Hollow width: {hollow_fin_width:.4f}m, Main fin height: {fin_height:.4f}m.'}

    # --- Handle num_fins == 0 case ---
    if num_fins == 0:
        # Area is just the top and bottom surface of the base plate
        area_base_no_fins = 2 * hs_length * hs_width
        return {'contact_area': area_base_no_fins}

    # --- Calculate Heatsink Base Area (with fins) ---
    area_base_bottom = hs_length * hs_width
    area_covered_by_fins_on_top = num_fins * fin_width * hs_length
    area_base_top_exposed = (hs_length * hs_width) - area_covered_by_fins_on_top
    # Ensure exposed area is not negative (should be caught by earlier validation, but good for safety)
    area_base_top_exposed = max(0, area_base_top_exposed)
    total_base_area = area_base_bottom + area_base_top_exposed

    # --- Calculate Main Fin Surface Area (before hollows) ---
    # Area of one fin's two large sides (along hs_length and fin_height)
    area_one_fin_sides = 2 * hs_length * fin_height
    # Area of one fin's top edge (along hs_length and fin_width)
    area_one_fin_top_edge = hs_length * fin_width
    # Area of one fin's two end edges (front and back, along fin_width and fin_height)
    area_one_fin_end_edges = 2 * fin_width * fin_height
    surface_area_one_fin_no_hollows = area_one_fin_sides + area_one_fin_top_edge + area_one_fin_end_edges
    total_fin_surface_area_no_hollows = num_fins * surface_area_one_fin_no_hollows

    # --- Calculate Net Area Change Due to Hollow Fins ---
    total_hollow_fin_area_contribution = 0.0
    if num_hollow_fins > 0 and fin_width > 0 : # Already ensured num_fins > 0 if num_hollow_fins > 0
        # Area removed from the two large sides of the main fin (hs_length x fin_height faces)
        # Each hollow channel (hollow_fin_length x hollow_fin_width) removes area from these two faces.
        area_removed_per_hollow_from_sides = 2 * hollow_fin_length * hollow_fin_width

        # Internal surface area added by the channel walls.
        # The channel passes through the main fin's thickness (fin_width).
        # Two walls are (hollow_fin_length x fin_width)
        # Other two walls are (hollow_fin_width x fin_width)
        internal_area_added_per_hollow = (2 * hollow_fin_length * fin_width) + (2 * hollow_fin_width * fin_width)

        net_area_change_per_hollow = internal_area_added_per_hollow - area_removed_per_hollow_from_sides
        total_hollow_fin_area_contribution = num_fins * num_hollow_fins * net_area_change_per_hollow

    # --- Total Contact Area ---
    total_contact_area = total_base_area + total_fin_surface_area_no_hollows + total_hollow_fin_area_contribution

    return {'contact_area': total_contact_area}


# --- Bloque para pruebas locales ---
if __name__ == '__main__':
    print(
        "Ejecutando prueba local de simulador_core.py (Modelo V2: Tj IGBT:híbrida, Tj Diode:avg, T_NTC EXPERIMENTAL)...")
    test_module_defs = [
        {'id': 'Mod_A', 'center_x': 0.1, 'center_y': 0.20},
        {'id': 'Mod_B', 'center_x': 0.1, 'center_y': 0.08},
    ]
    test_chip_powers = {
        'Mod_A_IGBT1': 100.0, 'Mod_A_Diode1': 25.0, 'Mod_A_IGBT2': 100.0, 'Mod_A_Diode2': 25.0,
        'Mod_B_IGBT1': 50.0, 'Mod_B_Diode1': 50.0, 'Mod_B_IGBT2': 0.0, 'Mod_B_Diode2': 0.0,
    }

    test_lx = 0.2
    test_ly = 0.3
    test_t_heatsink_thickness_val = 0.005  # Espesor del disipador
    test_k_main_heatsink_val = 200.0  # K del disipador
    test_rth_heatsink_val = 0.020  # Rth global del disipador
    test_t_ambient_val = 35.0  # T ambiente entrada
    test_q_total_m3_h_val = 700.0  # Caudal

    test_nx_val = 60  # Bajar resolución para prueba rápida
    test_ny_val = 90

    print(f"\n--- PARÁMETROS DE PRUEBA (Modelo V2) ---")
    print(f"Lx={test_lx}m, Ly={test_ly}m, t_hs={test_t_heatsink_thickness_val}m")
    print(f"K_MAIN_HEATSINK_BASE = {test_k_main_heatsink_val} W/mK")
    print(f"Rth_heatsink (global) = {test_rth_heatsink_val} K/W")
    print(f"T_ambient_inlet = {test_t_ambient_val}°C")
    print(f"Q_total_m3_h = {test_q_total_m3_h_val} m³/h")
    print(f"Nx={test_nx_val}, Ny={test_ny_val}")
    print(f"K_MODULE_BASEPLATE = {K_MODULE_BASEPLATE} W/mK, T_MODULE_BASEPLATE = {T_MODULE_BASEPLATE}m")
    print(
        f"K_TIM_INTERFACE = {K_TIM_INTERFACE} W/mK, T_TIM_INTERFACE = {T_TIM_INTERFACE}m => H_TIM = {H_MODULE_TO_HEATSINK_INTERFACE:.1f} W/m^2K")
    print(f"Rth_jhs_igbt = {Rth_jhs_igbt} K/W, Rth_jhs_diode = {Rth_jhs_diode} K/W")
    print(f"--- Potencias de prueba: ---");
    {print(f" {k}: {v}W") for k, v in test_chip_powers.items()}
    print(f"---------------------------\n")

    results_test = run_thermal_simulation(
        specific_chip_powers=test_chip_powers,
        lx=test_lx, ly=test_ly, t_heatsink_thickness=test_t_heatsink_thickness_val,
        k_main_heatsink_base_arg=test_k_main_heatsink_val,
        rth_heatsink=test_rth_heatsink_val,
        t_ambient_inlet_arg=test_t_ambient_val,
        q_total_m3_h_arg=test_q_total_m3_h_val,
        module_definitions=test_module_defs,
        nx=test_nx_val, ny=test_ny_val
    )

    print("\nResultados de la prueba (Modelo V2):")
    print(f"Status: {results_test['status']}")
    if results_test['status'] == 'Success':
        print(f"  Simulación con {len(test_module_defs)} módulos.")
        print(f"  Convergencia: {results_test['convergence']} en {results_test['iterations']} iteraciones")
        print(f"  T Max Disipador Principal: {results_test.get('t_max_base', np.nan):.2f} °C")
        print(
            f"  T Max Juntura (Oficial): {results_test.get('t_max_junction', np.nan):.2f} °C ({results_test.get('t_max_junction_chip', '')})")
        print(f"  T Max NTC (EXPERIMENTAL): {results_test.get('t_max_ntc', np.nan):.2f} °C")
        print(f"  T Aire Salida Promedio: {results_test.get('t_air_outlet', np.nan):.2f} °C")
        print("  Resultados por Módulo:")
        for mod_res_test in results_test.get('module_results', []):
            print(f"   - {mod_res_test.get('id', 'N/A')}: T_NTC_exp_avg={mod_res_test.get('t_ntc', np.nan):.2f}°C")
            for chip_res in mod_res_test.get('chips', []):
                is_igbt_chip_res = "IGBT" in chip_res.get('suffix', '')
                base_label_hs_res = "Tb_hs_HÍB" if is_igbt_chip_res else "Tb_hs_AVG"
                base_label_mod_res = "Tb_mod_HÍB" if is_igbt_chip_res else "Tb_mod_AVG"
                print(f"     Chip {chip_res.get('suffix')}:")
                print(
                    f"       {base_label_hs_res} = {chip_res.get('t_base_heatsink', np.nan):.2f}°C -> Tj_hs = {chip_res.get('tj_from_heatsink', np.nan):.2f}°C (Oficial)")
                print(
                    f"       {base_label_mod_res}= {chip_res.get('t_base_module_surface', np.nan):.2f}°C -> Tj_mod = {chip_res.get('tj_from_module_surface', np.nan):.2f}°C (Verif.)")

        print(
            f"  Plot Combinado (Disipador/Aire) URI: {'Presente' if results_test.get('plot_base_data_uri') else 'Ausente'}")
        print(
            f"  Plot Interactivo RAW (Disipador) URI: {'Presente' if results_test.get('plot_interactive_raw_uri') else 'Ausente'}")
        print(f"  Plot Zoom (T Módulo Sup.) URI: {'Presente' if results_test.get('plot_zoom_data_uri') else 'Ausente'}")
        print(
            f"  Datos Interactivos (Matriz T Disipador): {'Presente' if results_test.get('temperature_matrix') is not None else 'Ausente'}")

        plot_dir = "test_plots_SimCoreV2"
        os.makedirs(plot_dir, exist_ok=True)
        if results_test.get('plot_base_data_uri'):
            img_data_test = results_test['plot_base_data_uri'].split(',')[1]
            with open(os.path.join(plot_dir, "test_plot_base_V2.png"), "wb") as fh_test: fh_test.write(
                base64.b64decode(img_data_test))
            print(f"   -> {os.path.join(plot_dir, 'test_plot_base_V2.png')} guardado.")
        if results_test.get('plot_interactive_raw_uri'):
            img_data_test_raw = results_test['plot_interactive_raw_uri'].split(',')[1]
            with open(os.path.join(plot_dir, "test_plot_interactive_raw_V2.png"),
                      "wb") as fh_test_raw: fh_test_raw.write(base64.b64decode(img_data_test_raw))
            print(f"   -> {os.path.join(plot_dir, 'test_plot_interactive_raw_V2.png')} guardado.")
        if results_test.get('plot_zoom_data_uri'):
            img_data_test_zoom = results_test['plot_zoom_data_uri'].split(',')[1]
            with open(os.path.join(plot_dir, "test_plot_zoom_module_V2.png"), "wb") as fh_test_zoom: fh_test_zoom.write(
                base64.b64decode(img_data_test_zoom))
            print(f"   -> {os.path.join(plot_dir, 'test_plot_zoom_module_V2.png')} guardado.")
    else:
        print(f"  Error: {results_test.get('error_message', 'Desconocido')}")
    if results_test.get('error_message') and 'Warn:' in results_test.get('error_message', ''):
        print(f"  Advertencia: {results_test.get('error_message')}")
    print("\nPrueba local (Modelo V2) finalizada.")

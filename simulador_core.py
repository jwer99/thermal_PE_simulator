# simulador_core.py
import warnings
import matplotlib

matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu
import time
import math
import io
import base64
from scipy.interpolate import NearestNDInterpolator

# --- Constantes del Simulador ---
NTC_REL_X = +0.021
NTC_REL_Y = -0.036
K_MAIN_HEATSINK_BASE = 218.0
CP_AIR_REF_300K = 1007.0
RHO_AIR_REF_300K = 1.1614
MU_AIR_REF_300K = 1.846e-5
K_AIR_REF_300K = 0.0263
PR_AIR_REF_300K = (MU_AIR_REF_300K * CP_AIR_REF_300K) / K_AIR_REF_300K if K_AIR_REF_300K > 1e-9 else 0.707
w_igbt_footprint, h_igbt_footprint = 0.062, 0.122
K_MODULE_BASEPLATE = 390.0
T_MODULE_BASEPLATE = 0.003
K_TIM_INTERFACE = 5.0
T_TIM_INTERFACE = 0.000125
H_MODULE_TO_HEATSINK_INTERFACE = K_TIM_INTERFACE / T_TIM_INTERFACE if T_TIM_INTERFACE > 1e-9 else float('inf')
h_chip_igbt_m, w_chip_igbt_m = 0.036, 0.022
A_chip_igbt = w_chip_igbt_m * h_chip_igbt_m
h_chip_diode_m, w_chip_diode_m = 0.036, 0.011
A_chip_diode = w_chip_diode_m * h_chip_diode_m
Rth_jhs_igbt = 0.0731
Rth_jhs_diode = 0.131
Nx_base_default, Ny_base_default = 50, 70
Nz_base_default = 1
chip_rel_positions = {
    "IGBT1": (-0.006, +0.023), "Diode1": (+0.012, +0.023),
    "IGBT2": (+0.006, -0.023), "Diode2": (-0.012, -0.023),
}


def _get_air_properties(T_celsius):
    T_kelvin = T_celsius + 273.15
    temps_k = np.array([250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0])
    rhos_kg_m3 = np.array([1.3947, 1.1614, 0.9950, 0.8711, 0.7740, 0.6964, 0.6329, 0.5804])
    cps_J_kgK = np.array([1006.0, 1007.0, 1009.0, 1014.0, 1021.0, 1029.0, 1038.0, 1047.0])
    mus_Pa_s = np.array([1.596e-5, 1.846e-5, 2.082e-5, 2.301e-5, 2.507e-5, 2.701e-5, 2.884e-5, 3.058e-5])
    ks_W_mK = np.array([0.0223, 0.0263, 0.0300, 0.0338, 0.0373, 0.0407, 0.0439, 0.0469])
    prs_calc = (mus_Pa_s * cps_J_kgK) / ks_W_mK
    T_kelvin_clipped = np.clip(T_kelvin, temps_k[0], temps_k[-1])
    rho = np.interp(T_kelvin_clipped, temps_k, rhos_kg_m3)
    cp = np.interp(T_kelvin_clipped, temps_k, cps_J_kgK)
    mu = np.interp(T_kelvin_clipped, temps_k, mus_Pa_s)
    k = np.interp(T_kelvin_clipped, temps_k, ks_W_mK)
    Pr = np.interp(T_kelvin_clipped, temps_k, prs_calc)
    return rho, mu, k, Pr, cp


CHIP_ORDER_FOR_RTH_TABLE = ["IGBT1", "Diode1", "IGBT2", "Diode2"]
raw_experimental_data = [
    # ... (pega aquí la lista completa de tus datos de ensayo) ...
    {'chips_data': {'IGBT2': (207.49, 52.7)}, 'Tntc': 38.3},
    {'chips_data': {'IGBT2': (413.90, 83.5)}, 'Tntc': 54.0},
    {'chips_data': {'IGBT2': (623.82, 110.8)}, 'Tntc': 66.7},
    {'chips_data': {'IGBT2': (834.21, 140.3)}, 'Tntc': 80.7},
    {'chips_data': {'IGBT2': (903.68, 152.1)}, 'Tntc': 86.2},
    {'chips_data': {'IGBT2': (1019.04, 173.0)}, 'Tntc': 97.0},
    {'chips_data': {'Diode2': (105.96, 45.6)}, 'Tntc': 31.6},
    {'chips_data': {'Diode2': (201.92, 63.5)}, 'Tntc': 36.7},
    {'chips_data': {'Diode2': (307.91, 83.0)}, 'Tntc': 42.2},
    {'chips_data': {'Diode2': (404.39, 101.5)}, 'Tntc': 47.5},
    {'chips_data': {'Diode2': (497.38, 119.0)}, 'Tntc': 52.7},
    {'chips_data': {'Diode2': (597.82, 137.3)}, 'Tntc': 57.5},
    {'chips_data': {'IGBT1': (210.82, 52.5)}, 'Tntc': 32.15},
    {'chips_data': {'IGBT1': (412.81, 80.5)}, 'Tntc': 38.8},
    {'chips_data': {'IGBT1': (607.53, 105.0)}, 'Tntc': 45.5},
    {'chips_data': {'IGBT1': (810.11, 142.0)}, 'Tntc': 53.5},
    {'chips_data': {'IGBT1': (990.72, 175.0)}, 'Tntc': 59.5},
    {'chips_data': {'Diode1': (102.17, 41.0)}, 'Tntc': 27.5},
    {'chips_data': {'Diode1': (212.64, 60.5)}, 'Tntc': 30.8},
    {'chips_data': {'Diode1': (308.51, 78.3)}, 'Tntc': 34.5},
    {'chips_data': {'Diode1': (415.40, 98.0)}, 'Tntc': 37.8},
    {'chips_data': {'Diode1': (522.56, 118.0)}, 'Tntc': 41.5},
    {'chips_data': {'Diode1': (618.11, 136.5)}, 'Tntc': 46.1},
    {'chips_data': {'IGBT2': (110.175, 67.0), 'IGBT1': (110.175, 67.0)}, 'Tntc': 48.0},
    {'chips_data': {'IGBT2': (203.92, 108.0), 'IGBT1': (203.92, 108.0)}, 'Tntc': 70.0},
    {'chips_data': {'IGBT2': (300.31, 149.0), 'IGBT1': (300.31, 149.0)}, 'Tntc': 92.0},
    {'chips_data': {'IGBT2': (359.05, 175.2), 'IGBT1': (359.05, 175.2)}, 'Tntc': 106.0},
    {'chips_data': {'Diode2': (49.78, 47.2), 'Diode1': (49.78, 47.2)}, 'Tntc': 33.15},
    {'chips_data': {'Diode2': (101.195, 74.5), 'Diode1': (101.195, 74.5)}, 'Tntc': 44.5},
    {'chips_data': {'Diode2': (164.605, 105.7), 'Diode1': (164.605, 105.7)}, 'Tntc': 55.25},
    {'chips_data': {'Diode2': (196.74, 122.6), 'Diode1': (196.74, 122.6)}, 'Tntc': 60.7},
    {'chips_data': {'Diode2': (254.345, 159.2), 'Diode1': (254.345, 159.2)}, 'Tntc': 75.0},
    {'chips_data': {'IGBT2': (193.70, 85.5), 'Diode1': (191.57, 101.0)}, 'Tntc': 61.6},
    {'chips_data': {'IGBT2': (241.90, 124.0), 'Diode2': (190.62, 135.0)}, 'Tntc': 78.8},
    {'chips_data': {'IGBT2': (252.55, 135.0), 'Diode2': (211.89, 146.2)}, 'Tntc': 82.8},
    {'chips_data': {'IGBT1': (225.50, 116.0), 'Diode1': (181.98, 126.0)}, 'Tntc': 56.2},
    {'chips_data': {'IGBT1': (248.85, 132.0), 'Diode1': (231.93, 144.0)}, 'Tntc': 61.9},
    {'chips_data': {'IGBT2': (96.55, 129.0), 'Diode2': (96.55, 140.0), 'IGBT1': (96.55, 127.0),
                    'Diode1': (96.55, 137.5)}, 'Tntc': 86.7},
    {'chips_data': {'IGBT2': (97.15, 157.0), 'Diode2': (97.15, 167.0), 'IGBT1': (97.15, 156.0),
                    'Diode1': (97.15, 166.2)}, 'Tntc': 101.4},
]


def process_raw_data_to_rth_table(raw_data, chip_order):
    rth_data_points = {chip_name: [] for chip_name in chip_order}
    for experiment in raw_data:
        chip_powers = {name: data[0] for name, data in experiment['chips_data'].items()}
        total_power = sum(chip_powers.values())
        if total_power < 1e-6:
            continue
        power_dist_vector = [chip_powers.get(chip, 0.0) / total_power for chip in chip_order]
        for chip_name, (power, tj) in experiment['chips_data'].items():
            if power > 1e-6:
                t_ntc = experiment['Tntc']
                rth_j_ntc = (tj - t_ntc) / power
                rth_data_points[chip_name].append((power_dist_vector, rth_j_ntc))
    return rth_data_points


EXPERIMENTAL_RTH_NTC_DATA = process_raw_data_to_rth_table(raw_experimental_data, CHIP_ORDER_FOR_RTH_TABLE)


def _create_rth_j_ntc_interpolators(experimental_data):
    interpolators = {}
    print("[NTC_Model] Creando modelos de interpolación ROBUSTOS para Rth_j_ntc...")
    for chip_suffix, rth_entries in experimental_data.items():
        if not rth_entries:
            print(f"  - ADVERTENCIA: No hay datos experimentales para '{chip_suffix}'. Se omitirá.")
            continue
        points = np.array([entry[0] for entry in rth_entries])
        values = np.array([entry[1] for entry in rth_entries])
        unique_points, inverse_indices = np.unique(points, axis=0, return_inverse=True)
        unique_values = np.array([np.mean(values[inverse_indices == i]) for i in range(len(unique_points))])
        if len(unique_points) > 0:
            interp = NearestNDInterpolator(unique_points, unique_values)
            fallback_avg = np.mean(unique_values)
            interpolators[chip_suffix] = (interp, fallback_avg)
            # Descomentar para log detallado
            # print(
            #     f"  - Interpolador (Nearest) para '{chip_suffix}' creado con {len(unique_points)} puntos de datos únicos.")
            # print(
            #     f"    Rango de Rth para {chip_suffix}: Min={np.min(unique_values):.4f}, Max={np.max(unique_values):.4f}, Promedio={fallback_avg:.4f}")
        else:
            print(f"  - ADVERTENCIA: No hay suficientes datos únicos para '{chip_suffix}'.")
            interpolators[chip_suffix] = (None, np.nan)
    return interpolators


rth_j_ntc_interpolators = _create_rth_j_ntc_interpolators(EXPERIMENTAL_RTH_NTC_DATA)


def calculate_ntc_temperature_advanced(module_item, power_dist_normalized, interpolators_dict):
    # NOTA: Esta función ya no se usa para el cálculo final de T_ntc, pero se mantiene en el código por referencia.
    all_ntc_estimates = []
    power_weights = []
    total_module_power = sum(c['power'] for c in module_item['chips'])

    if total_module_power < 1e-6:
        return np.nan

    for chip in module_item['chips']:
        chip_suffix = chip['suffix']
        power = chip['power']
        tj_calc = chip['Tj']

        if power <= 1e-6 or np.isnan(tj_calc):
            continue

        interp_data = interpolators_dict.get(chip_suffix)
        if not interp_data or not interp_data[0]:
            continue

        interpolator, fallback_avg_rth = interp_data
        predicted_rth_j_ntc = interpolator([power_dist_normalized])[0]

        if np.isnan(predicted_rth_j_ntc):
            predicted_rth_j_ntc = fallback_avg_rth

        temperature_drop = power * predicted_rth_j_ntc
        t_ntc_est_from_chip = tj_calc - temperature_drop

        all_ntc_estimates.append(t_ntc_est_from_chip)
        power_weights.append(power)

    if not all_ntc_estimates:
        return np.nan

    final_ntc_temp = np.average(all_ntc_estimates, weights=power_weights)
    return final_ntc_temp


def _calculate_nu_for_channel(Re_ch, Pr_fl, type_info=""):
    Nu_ch = 0.0;
    Re_ch_crit_lower = 2300
    if Re_ch < 1.0: return 3.66
    if Re_ch < Re_ch_crit_lower:
        Nu_ch = 4.36
    else:
        if Re_ch < 10:
            Nu_ch = 4.36
        else:
            try:
                if Re_ch < 4000:
                    f_darcy = 0.3164 * (Re_ch ** (-0.25))
                else:
                    f_darcy = (0.790 * math.log(Re_ch) - 1.64) ** -2.0
                numerator = (f_darcy / 8.0) * (Re_ch - 1000.0) * Pr_fl
                denominator = 1.0 + 12.7 * (f_darcy / 8.0) ** 0.5 * (Pr_fl ** (2.0 / 3.0) - 1.0)
                if abs(denominator) < 1e-9:
                    Nu_ch = 0.027 * (Re_ch ** 0.805) * (Pr_fl ** 0.33)
                else:
                    Nu_ch = numerator / denominator
                if Nu_ch < 0: Nu_ch = 0.027 * (Re_ch ** 0.805) * (Pr_fl ** 0.33)
            except (ValueError, OverflowError):
                Nu_ch = 0.027 * (Re_ch ** 0.805) * (Pr_fl ** 0.33)
    return Nu_ch


def calculate_h_array_or_eff(lx_base, ly_base, q_total_m3_h, t_ambient_inlet,
                             assumed_duct_height, k_heatsink_material, fin_params,
                             t_surface_avg_estimate=None, nx_grid=None, ny_grid=None,
                             return_array_for_flat_plate=False):
    if not all(isinstance(v, (int, float)) for v in
               [lx_base, ly_base, q_total_m3_h, t_ambient_inlet, assumed_duct_height,
                k_heatsink_material]): return np.nan
    if lx_base <= 1e-6 or ly_base <= 1e-6 or q_total_m3_h <= 1e-9 or k_heatsink_material <= 1e-9 or assumed_duct_height <= 1e-6: return np.nan
    h_f = fin_params.get('h_fin', 0.0);
    t_f = fin_params.get('t_fin', 0.0);
    N_f = int(fin_params.get('num_fins', 0))
    w_hf = fin_params.get('w_hollow', 0.0);
    h_hf = fin_params.get('h_hollow', 0.0);
    N_hpf = int(fin_params.get('num_hollow_per_fin', 0))
    if N_f > 0 and (t_f <= 1e-9 or h_f <= 1e-9): return np.nan
    if N_f * t_f > lx_base - 1e-6 and N_f > 0: return np.nan
    valid_hollows = False
    if N_hpf > 0 and N_f > 0 and w_hf > 1e-9 and h_hf > 1e-9 and w_hf < t_f and h_hf < h_f:
        valid_hollows = True
    elif N_hpf > 0:
        N_hpf = 0
    t_surface_actual_estimate = t_ambient_inlet + 15.0 if t_surface_avg_estimate is None else t_surface_avg_estimate
    T_film_C = (t_surface_actual_estimate + t_ambient_inlet) / 2.0
    rho_fluid, mu_fluid, k_fluid, Pr_fluid, cp_fluid = _get_air_properties(T_film_C)
    if k_fluid <= 1e-9 or mu_fluid <= 1e-12 or Pr_fluid <= 1e-3: return np.nan
    Q_total_m3_s = q_total_m3_h / 3600.0
    if N_f == 0 or h_f <= 1e-6 or t_f <= 1e-6:
        A_flow_duct_gross = lx_base * assumed_duct_height
        if A_flow_duct_gross < 1e-9: return np.nan
        U_avg_air_duct = Q_total_m3_s / A_flow_duct_gross
        if not return_array_for_flat_plate or nx_grid is None or ny_grid is None or nx_grid < 1 or ny_grid < 1:
            L_char_flat_plate_avg = ly_base;
            Re_L_avg = (rho_fluid * U_avg_air_duct * L_char_flat_plate_avg) / mu_fluid if mu_fluid > 1e-12 else 0;
            Nu_L_avg = 0.0
            if Re_L_avg < 1.0:
                h_avg = 5.0
            else:
                Re_crit_flat_plate = 5e5
                if Re_L_avg <= Re_crit_flat_plate:
                    Nu_L_avg = 0.664 * (Re_L_avg ** 0.5) * (Pr_fluid ** (1.0 / 3.0))
                else:
                    Nu_L_avg = (0.037 * Re_L_avg ** 0.8 - 871) * Pr_fluid ** (1.0 / 3.0)
                if Nu_L_avg < 0: Nu_L_avg = 0.664 * (Re_L_avg ** 0.5) * (Pr_fluid ** (1.0 / 3.0))
                h_avg = (
                                Nu_L_avg * k_fluid) / L_char_flat_plate_avg if L_char_flat_plate_avg > 1e-9 and Nu_L_avg > 0 else 5.0
            return max(h_avg, 5.0)
        else:
            h_matrix_flat = np.full((nx_grid, ny_grid), 5.0);
            dy_grid = ly_base / (ny_grid - 1) if ny_grid > 1 else ly_base;
            Re_crit_local = 5e5
            for j in range(ny_grid):
                y_coord = (j + 0.5) * dy_grid;
                if y_coord < 1e-6: y_coord = 1e-6
                Re_y_local = (rho_fluid * U_avg_air_duct * y_coord) / mu_fluid if mu_fluid > 1e-12 else 0;
                Nu_y_local = 0.0
                if Re_y_local < 1.0:
                    h_val_local = 5.0
                else:
                    if Re_y_local <= Re_crit_local:
                        Nu_y_local = 0.332 * (Re_y_local ** 0.5) * (Pr_fluid ** (1.0 / 3.0))
                    else:
                        Nu_y_local = 0.0296 * (Re_y_local ** 0.8) * (Pr_fluid ** (1.0 / 3.0))
                h_val_local = (Nu_y_local * k_fluid) / y_coord if y_coord > 1e-9 and Nu_y_local > 0 else 5.0
                h_matrix_flat[:, j] = max(h_val_local, 5.0)
            return h_matrix_flat
    h_coeff_main_channels = 0.0;
    h_coeff_hollow_channels = 0.0;
    A_flow_main_channels_total = 0.0;
    A_flow_hollow_channels_total = 0.0
    s_fin_clear = (lx_base - N_f * t_f) / (N_f - 1) if N_f > 1 else (lx_base - t_f);
    H_canal_eff_main = min(h_f, assumed_duct_height)
    num_main_channels = (N_f - 1) if N_f > 1 else 1;
    A_flow_main_channels_total = num_main_channels * s_fin_clear * H_canal_eff_main
    if valid_hollows: A_flow_hollow_channels_total = N_f * N_hpf * (w_hf * h_hf)
    A_flow_grand_total = A_flow_main_channels_total + A_flow_hollow_channels_total
    Nu_main = 0.0
    if A_flow_grand_total < 1e-9:
        h_coeff_main_channels = 2.0
    else:
        Q_main = Q_total_m3_s * (A_flow_main_channels_total / A_flow_grand_total);
        Q_hollow = Q_total_m3_s * (A_flow_hollow_channels_total / A_flow_grand_total)
        if A_flow_main_channels_total > 1e-9 and Q_main > 1e-9:
            U_main = Q_main / A_flow_main_channels_total;
            D_h_main = (4 * s_fin_clear * H_canal_eff_main) / (
                    2 * (s_fin_clear + H_canal_eff_main)) if s_fin_clear + H_canal_eff_main > 1e-9 else 0.0
            if D_h_main <= 1e-9:
                h_coeff_main_channels = 2.0
            else:
                Re_main = (rho_fluid * U_main * D_h_main) / mu_fluid
                Nu_main = _calculate_nu_for_channel(Re_main, Pr_fluid, "MainCh")
                h_coeff_main_channels = (Nu_main * k_fluid) / D_h_main
            h_coeff_main_channels = max(h_coeff_main_channels, 2.0)
        else:
            h_coeff_main_channels = 2.0
        if valid_hollows and A_flow_hollow_channels_total > 1e-9 and Q_hollow > 1e-9:
            U_hollow = Q_hollow / A_flow_hollow_channels_total;
            D_h_hollow = (4 * w_hf * h_hf) / (2 * (w_hf + h_hf)) if w_hf + h_hf > 1e-9 else 0.0
            if D_h_hollow <= 1e-9:
                h_coeff_hollow_channels = 2.0
            else:
                Re_hollow = (rho_fluid * U_hollow * D_h_hollow) / mu_fluid
                Nu_hollow = _calculate_nu_for_channel(Re_hollow, Pr_fluid, "HollowCh")
                h_coeff_hollow_channels = (Nu_hollow * k_fluid) / D_h_hollow
            h_coeff_hollow_channels = max(h_coeff_hollow_channels, 2.0)
        else:
            h_coeff_hollow_channels = 0.0
    A_base_primary = lx_base * ly_base;
    A_base_exposed = max(0, (lx_base - N_f * t_f) * ly_base);
    A_fins_sides = N_f * (2 * h_f * ly_base)
    A_hollow_walls = N_f * N_hpf * (2 * (w_hf + h_hf)) * ly_base if valid_hollows else 0.0
    denominator_h_eff = A_base_exposed + A_flow_hollow_channels_total + A_hollow_walls
    if abs(denominator_h_eff) > 1e-9 and k_fluid > 1e-9 and Nu_main > 0:
        h_eff_scalar = (Nu_main * k_heatsink_material) / denominator_h_eff
    else:
        h_eff_scalar = 5.0
    h_eff_scalar = max(h_eff_scalar, 1.0)
    if return_array_for_flat_plate and (nx_grid is not None and ny_grid is not None):
        return np.full((nx_grid, ny_grid), h_eff_scalar)
    else:
        return h_eff_scalar


calculate_h_scalar_eff = calculate_h_array_or_eff


def run_thermal_simulation(specific_chip_powers, lx, ly, t, rth_heatsink, module_definitions,
                           t_ambient_inlet_arg, q_total_m3_h_arg,
                           h_eff_FDM_heatsink_arg,
                           nx=Nx_base_default, ny=Ny_base_default, nz_base=Nz_base_default,
                           get_hybrid_temp_in_area_local_func_ptr=None,
                           initial_T_solution=None, initial_T_air_solution=None):
    results = {
        'status': 'Processing', 'convergence': False, 'iterations': 0,
        't_max_base': np.nan, 't_avg_base': np.nan, 't_air_outlet': np.nan,
        't_max_junction': np.nan, 't_max_junction_chip': '', 't_max_ntc': np.nan,
        'module_results': [], 'plot_base_data_uri': None, 'plot_interactive_raw_uri': None,
        'plot_zoom_data_uri': None, 'error_message': None, 'T_solution_matrix': None,
        'x_coordinates_vector': None, 'y_coordinates_vector': None,
        'T_solution_full': None, 'T_air_solution_full': None,
        'sim_params_dict': {'lx': lx, 'ly': ly, 't': t, 'nx': nx, 'ny': ny, 'nz_base': nz_base}
    }
    start_time_sim = time.time()
    sim_type = "3D" if nz_base > 1 else "2D (o 2.5D)"
    print(f"[SimCore] Iniciando Simulación {sim_type} con {len(module_definitions)} módulos.")
    print(f"[SimCore] Dimensiones Disipador (base): Lx={lx:.3f}m, Ly={ly:.3f}m, Espesor_base_FDM={t:.4f}m")
    print(f"[SimCore] Discretización Base Disipador: Nx={nx}, Ny={ny}, Nz_base_FDM={nz_base}")

    is_h_array = isinstance(h_eff_FDM_heatsink_arg, np.ndarray) and h_eff_FDM_heatsink_arg.ndim == 2
    if is_h_array:
        if h_eff_FDM_heatsink_arg.shape != (nx, ny):
            results['status'] = 'Error';
            results[
                'error_message'] = f"Forma de h_array ({h_eff_FDM_heatsink_arg.shape}) no coincide con malla ({nx},{ny}).";
            return results
        if np.any(h_eff_FDM_heatsink_arg <= 1e-9):
            results['status'] = 'Error';
            results['error_message'] = "h_array contiene valores no positivos.";
            return results
        print(
            f"[SimCore] Usando h_eff_FDM_heatsink como ARRAY. Min={np.min(h_eff_FDM_heatsink_arg):.2f}, Max={np.max(h_eff_FDM_heatsink_arg):.2f} W/m^2K")
    else:
        if not isinstance(h_eff_FDM_heatsink_arg, (int, float)) or h_eff_FDM_heatsink_arg <= 1e-9:
            results['status'] = 'Error';
            results['error_message'] = 'h_eff_FDM_heatsink_arg (escalar) debe ser numérico > 0.';
            return results
        print(f"[SimCore] Usando h_eff_FDM_heatsink (escalar) = {h_eff_FDM_heatsink_arg:.2f} W/m^2K")

    if not isinstance(specific_chip_powers, dict): results['status'] = 'Error'; results[
        'error_message'] = 'Powers dict invalido.'; return results
    if not (isinstance(lx, (int, float)) and lx > 1e-6 and isinstance(ly, (int, float)) and ly > 1e-6 and isinstance(t,
                                                                                                                     (
                                                                                                                             int,
                                                                                                                             float)) and t > 1e-6):
        results['status'] = 'Error';
        results['error_message'] = 'Lx, Ly, t deben ser numéricos > 0.';
        return results
    if not isinstance(nz_base, int) or nz_base < 1: nz_base = 1; sim_type = "2D (o 2.5D)"
    results['sim_params_dict']['nz_base'] = nz_base
    rho_fluid_inlet, _, _, _, cp_fluid_inlet = _get_air_properties(t_ambient_inlet_arg)
    dx = lx / (nx - 1) if nx > 1 else lx;
    dy = ly / (ny - 1) if ny > 1 else ly
    dz = t / (nz_base - 1) if nz_base > 1 else t
    dA_xy = dx * dy

    T_solution_shape = (nx, ny, nz_base) if nz_base > 1 else (nx, ny)

    if initial_T_solution is not None and initial_T_solution.shape == T_solution_shape:
        T_solution = initial_T_solution.copy()
        print("[SimCore] Warm start: Usando T_solution de la iteración 'h' anterior.")
    else:
        T_solution = np.full(T_solution_shape, t_ambient_inlet_arg + 10.0)
        print("[SimCore] Cold start: Inicializando T_solution a valor por defecto.")

    if initial_T_air_solution is not None and initial_T_air_solution.shape == (nx, ny):
        T_air_solution = initial_T_air_solution.copy()
        print("[SimCore] Warm start: Usando T_air_solution de la iteración 'h' anterior.")
    else:
        T_air_solution = np.full((nx, ny), t_ambient_inlet_arg)
        print("[SimCore] Cold start: Inicializando T_air_solution a valor por defecto.")

    def local_fig_to_data_uri(fig_obj, dpi=90, pad_inches=0.1, bbox_inches='tight'):
        buf = io.BytesIO()
        fig_obj.savefig(buf, format='png', bbox_inches=bbox_inches, dpi=dpi, pad_inches=pad_inches)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig_obj)
        return f"data:image/png;base64,{img_base64}"

    # --- Caso SIN MÓDULOS ---
    if not module_definitions:
        print("[SimCore] No módulos. Simulación simplificada.")
        h_eff_no_mod_scalar = np.mean(h_eff_FDM_heatsink_arg) if is_h_array else h_eff_FDM_heatsink_arg
        h_eff_no_mod_scalar = max(h_eff_no_mod_scalar, 2.0)
        try:
            N_nodes_no_mod = nx * ny
            C_x_no_mod = K_MAIN_HEATSINK_BASE * t * dy / dx if dx > 1e-12 else 0
            C_y_no_mod = K_MAIN_HEATSINK_BASE * t * dx / dy if dy > 1e-12 else 0
            A_mat_no_mod = lil_matrix((N_nodes_no_mod, N_nodes_no_mod));
            b_vec_no_mod = np.zeros(N_nodes_no_mod)
            for i_node in range(nx):
                for j_node in range(ny):
                    idx = i_node * ny + j_node
                    C_h_local_no_mod = h_eff_no_mod_scalar * dA_xy
                    coef_diag_no_mod = -C_h_local_no_mod
                    b_vec_no_mod[idx] = -C_h_local_no_mod * t_ambient_inlet_arg
                    if i_node > 0: A_mat_no_mod[idx, idx - ny] = C_x_no_mod; coef_diag_no_mod -= C_x_no_mod
                    if i_node < nx - 1: A_mat_no_mod[idx, idx + ny] = C_x_no_mod; coef_diag_no_mod -= C_x_no_mod
                    if j_node > 0: A_mat_no_mod[idx, idx - 1] = C_y_no_mod; coef_diag_no_mod -= C_y_no_mod
                    if j_node < ny - 1: A_mat_no_mod[idx, idx + 1] = C_y_no_mod; coef_diag_no_mod -= C_y_no_mod
                    A_mat_no_mod[idx, idx] = coef_diag_no_mod
            LU_no_mod = splu(A_mat_no_mod.tocsr());
            T_flat_no_mod = LU_no_mod.solve(b_vec_no_mod)
            T_sol_no_mod = T_flat_no_mod.reshape((nx, ny))
            results['t_max_base'] = np.max(T_sol_no_mod);
            results['t_avg_base'] = np.mean(T_sol_no_mod)
            results['t_air_outlet'] = t_ambient_inlet_arg;
            results['convergence'] = True;
            results['iterations'] = 0
            results['T_solution_matrix'] = T_sol_no_mod;
            results['x_coordinates_vector'] = np.linspace(0, lx, nx);
            results['y_coordinates_vector'] = np.linspace(0, ly, ny)
            results['status'] = 'Success_NoInteractiveData'
        except Exception as e_base_no_mod:
            results['status'] = 'Error';
            results['error_message'] = f"Error setup sin módulos: {e_base_no_mod}"
        return results

    # --- PROCESAMIENTO CON MÓDULOS ---
    try:
        # Nota: esta función se ejecuta para inicializar los interpoladores, pero ya no se usan
        # para el cálculo de T_ntc. Se mantiene por si se reutiliza en el futuro.
        _create_rth_j_ntc_interpolators(EXPERIMENTAL_RTH_NTC_DATA)

        k_hs_base_fdm = K_MAIN_HEATSINK_BASE
        Q_total_m3_s_iter = q_total_m3_h_arg / 3600.0
        m_dot_total_kgs = Q_total_m3_s_iter * rho_fluid_inlet
        m_dot_per_meter_width = m_dot_total_kgs / lx if lx > 0 else 0
        m_dot_column_kgs = m_dot_per_meter_width * dx
        m_dot_cp_column = max(1e-12, m_dot_column_kgs * cp_fluid_inlet)
        modules_data_sim = []
        Nx_module_footprint_global = max(1, int(round(w_igbt_footprint / dx))) if dx > 1e-12 else 1
        Ny_module_footprint_global = max(1, int(round(h_igbt_footprint / dy))) if dy > 1e-12 else 1
        nx_module_half_global = Nx_module_footprint_global // 2;
        ny_module_half_global = Ny_module_footprint_global // 2
        for mod_def_idx, mod_def in enumerate(module_definitions):
            module_id_base = mod_def['id'];
            module_center_x = mod_def['center_x'];
            module_center_y = mod_def['center_y']
            half_w_footprint = w_igbt_footprint / 2;
            half_h_footprint = h_igbt_footprint / 2
            module_center_x = max(half_w_footprint, min(lx - half_w_footprint, module_center_x))
            module_center_y = max(half_h_footprint, min(ly - half_h_footprint, module_center_y))
            module_center_x_idx_global = int(round(module_center_x / dx)) if dx > 1e-12 else 0
            module_center_y_idx_global = int(round(module_center_y / dy)) if dy > 1e-12 else 0
            module_center_x_idx_global = max(0, min(nx - 1, module_center_x_idx_global))
            module_center_y_idx_global = max(0, min(ny - 1, module_center_y_idx_global))
            footprint_i_min_global = max(0, module_center_x_idx_global - nx_module_half_global);
            footprint_i_max_global = min(nx - 1, module_center_x_idx_global + nx_module_half_global)
            footprint_j_min_global = max(0, module_center_y_idx_global - ny_module_half_global);
            footprint_j_max_global = min(ny - 1, module_center_y_idx_global + ny_module_half_global)
            nx_mod_local = footprint_i_max_global - footprint_i_min_global + 1;
            ny_mod_local = footprint_j_max_global - footprint_j_min_global + 1
            module_info = {'id': module_id_base, 'center_x_m': module_center_x, 'center_y_m': module_center_y,
                           'center_x_idx_global': module_center_x_idx_global,
                           'center_y_idx_global': module_center_y_idx_global,
                           'footprint_i_min_global': footprint_i_min_global,
                           'footprint_i_max_global': footprint_i_max_global,
                           'footprint_j_min_global': footprint_j_min_global,
                           'footprint_j_max_global': footprint_j_max_global, 'nx_mod_local': nx_mod_local,
                           'ny_mod_local': ny_mod_local, 'chips': [], 'ntc_abs_x': None, 'ntc_abs_y': None,
                           'ntc_i_idx_global': None, 'ntc_j_idx_global': None, 'ntc_i_idx_local': None,
                           'ntc_j_idx_local': None,
                           'T_module_internal_solution': np.full((nx_mod_local, ny_mod_local),
                                                                 t_ambient_inlet_arg + 10.0),
                           'is_on_heatsink_map': np.zeros((nx_mod_local, ny_mod_local), dtype=bool)}
            for i_loc_mod in range(nx_mod_local):
                for j_loc_mod in range(ny_mod_local):
                    g_i = footprint_i_min_global + i_loc_mod;
                    g_j = footprint_j_min_global + j_loc_mod
                    if 0 <= g_i < nx and 0 <= g_j < ny: module_info['is_on_heatsink_map'][i_loc_mod, j_loc_mod] = True
            ntc_abs_x_val = module_center_x + NTC_REL_X;
            ntc_abs_y_val = module_center_y + NTC_REL_Y
            module_info['ntc_abs_x'] = ntc_abs_x_val;
            module_info['ntc_abs_y'] = ntc_abs_y_val
            if 0 <= ntc_abs_x_val <= lx and 0 <= ntc_abs_y_val <= ly and dx > 1e-12 and dy > 1e-12:
                ntc_i_idx_global_val = int(round(ntc_abs_x_val / dx));
                ntc_j_idx_global_val = int(round(ntc_abs_y_val / dy))
                ntc_i_idx_global_val = max(0, min(nx - 1, ntc_i_idx_global_val));
                ntc_j_idx_global_val = max(0, min(ny - 1, ntc_j_idx_global_val))
                module_info['ntc_i_idx_global'] = ntc_i_idx_global_val;
                module_info['ntc_j_idx_global'] = ntc_j_idx_global_val
                if footprint_i_min_global <= ntc_i_idx_global_val <= footprint_i_max_global and footprint_j_min_global <= ntc_j_idx_global_val <= footprint_j_max_global:
                    module_info['ntc_i_idx_local'] = ntc_i_idx_global_val - footprint_i_min_global;
                    module_info['ntc_j_idx_local'] = ntc_j_idx_global_val - footprint_j_min_global
            for chip_label_suffix, (rel_x_chip, rel_y_chip) in chip_rel_positions.items():
                chip_full_id = f"{module_id_base}_{chip_label_suffix}";
                chip_center_x_m = module_center_x + rel_x_chip;
                chip_center_y_m = module_center_y + rel_y_chip
                chip_center_x_m = max(0, min(lx, chip_center_x_m));
                chip_center_y_m = max(0, min(ly, chip_center_y_m))
                chip_x_idx_global_val = int(round(chip_center_x_m / dx)) if dx > 1e-12 else 0;
                chip_y_idx_global_val = int(round(chip_center_y_m / dy)) if dy > 1e-12 else 0
                chip_x_idx_global_val = max(0, min(nx - 1, chip_x_idx_global_val));
                chip_y_idx_global_val = max(0, min(ny - 1, chip_y_idx_global_val))
                is_igbt_chip = "IGBT" in chip_label_suffix;
                chip_type_val = "IGBT" if is_igbt_chip else "Diode"
                if is_igbt_chip:
                    chip_w_m_val, chip_h_m_val, chip_A_val, chip_Rth_val = w_chip_igbt_m, h_chip_igbt_m, A_chip_igbt, Rth_jhs_igbt
                else:
                    chip_w_m_val, chip_h_m_val, chip_A_val, chip_Rth_val = w_chip_diode_m, h_chip_diode_m, A_chip_diode, Rth_jhs_diode
                chip_Nx_cells_global_val = max(1, int(round(chip_w_m_val / dx))) if dx > 1e-12 else 1;
                chip_Ny_cells_global_val = max(1, int(round(chip_h_m_val / dy))) if dy > 1e-12 else 1
                chip_nx_half_global_val, chip_ny_half_global_val = chip_Nx_cells_global_val // 2, chip_Ny_cells_global_val // 2
                chip_nx_half_local_val, chip_ny_half_local_val = chip_nx_half_global_val, chip_ny_half_global_val
                chip_power_val = specific_chip_powers.get(chip_full_id, 0.0);
                chip_q_source_val = chip_power_val / chip_A_val if chip_A_val > 1e-12 else 0.0
                chip_x_idx_local_val, chip_y_idx_local_val = -1, -1
                if footprint_i_min_global <= chip_x_idx_global_val <= footprint_i_max_global and footprint_j_min_global <= chip_y_idx_global_val <= footprint_j_max_global:
                    chip_x_idx_local_val = chip_x_idx_global_val - footprint_i_min_global;
                    chip_y_idx_local_val = chip_y_idx_global_val - footprint_j_min_global
                chip_data = {'label': chip_full_id, 'suffix': chip_label_suffix, 'type': chip_type_val,
                             'center_x_m': chip_center_x_m, 'center_y_m': chip_center_y_m,
                             'center_x_idx_global': chip_x_idx_global_val, 'center_y_idx_global': chip_y_idx_global_val,
                             'nx_half_global': chip_nx_half_global_val, 'ny_half_global': chip_ny_half_global_val,
                             'center_x_idx_local': chip_x_idx_local_val, 'center_y_idx_local': chip_y_idx_local_val,
                             'nx_half_local': chip_nx_half_local_val, 'ny_half_local': chip_ny_half_local_val,
                             'power': chip_power_val, 'q_source_W_per_m2': chip_q_source_val, 'Rth_jhs': chip_Rth_val,
                             'T_base_chip_on_heatsink': np.nan, 'T_base_chip_on_module_surface': np.nan,
                             'Tj_from_heatsink': np.nan, 'Tj_from_module_surface': np.nan, 'Tj': np.nan,
                             'w_m': chip_w_m_val, 'h_m': chip_h_m_val, 'Area_m2': chip_A_val}
                module_info['chips'].append(chip_data)
            modules_data_sim.append(module_info)
        heatsink_covered_by_module_map = np.zeros((nx, ny), dtype=bool)
        for mod_item_map in modules_data_sim: heatsink_covered_by_module_map[
                                              mod_item_map['footprint_i_min_global']:mod_item_map[
                                                                                         'footprint_i_max_global'] + 1,
                                              mod_item_map['footprint_j_min_global']:mod_item_map[
                                                                                         'footprint_j_max_global'] + 1] = True

        N_nodes_heatsink = nx * ny * nz_base if nz_base > 1 else nx * ny
        A_matrix_heatsink = lil_matrix((N_nodes_heatsink, N_nodes_heatsink))
        LU_heatsink = None

        if nz_base == 1:  # 2.5D
            print("[SimCore] Ensamblando matriz 2D para base (nz_base=1).")
            C_x_cond_hs_2D = k_hs_base_fdm * t * dy / dx if dx > 1e-12 else 0;
            C_y_cond_hs_2D = k_hs_base_fdm * t * dx / dy if dy > 1e-12 else 0
            for i_hs in range(nx):
                for j_hs in range(ny):
                    idx_hs_2D = i_hs * ny + j_hs
                    h_local = h_eff_FDM_heatsink_arg[i_hs, j_hs] if is_h_array else h_eff_FDM_heatsink_arg
                    C_h_local_hs = h_local * dA_xy
                    coef_diag_hs = -C_h_local_hs
                    if i_hs > 0: A_matrix_heatsink[
                        idx_hs_2D, idx_hs_2D - ny] = C_x_cond_hs_2D; coef_diag_hs -= C_x_cond_hs_2D
                    if i_hs < nx - 1: A_matrix_heatsink[
                        idx_hs_2D, idx_hs_2D + ny] = C_x_cond_hs_2D; coef_diag_hs -= C_x_cond_hs_2D
                    if j_hs > 0: A_matrix_heatsink[
                        idx_hs_2D, idx_hs_2D - 1] = C_y_cond_hs_2D; coef_diag_hs -= C_y_cond_hs_2D
                    if j_hs < ny - 1: A_matrix_heatsink[
                        idx_hs_2D, idx_hs_2D + 1] = C_y_cond_hs_2D; coef_diag_hs -= C_y_cond_hs_2D
                    A_matrix_heatsink[idx_hs_2D, idx_hs_2D] = coef_diag_hs
        else:  # 3D
            print("[SimCore] Ensamblando matriz 3D para base.")
            dA_yz = dy * dz;
            dA_xz = dx * dz
            C_x_cond_hs_3D = k_hs_base_fdm * dA_yz / dx if dx > 1e-12 else 0;
            C_y_cond_hs_3D = k_hs_base_fdm * dA_xz / dy if dy > 1e-12 else 0;
            C_z_cond_hs_3D = k_hs_base_fdm * dA_xy / dz if dz > 1e-12 else 0
            for i_hs in range(nx):
                for j_hs in range(ny):
                    for k_hs in range(nz_base):
                        idx_hs_3D = (i_hs * ny + j_hs) * nz_base + k_hs;
                        coef_diag_hs = 0.0
                        if i_hs > 0: A_matrix_heatsink[idx_hs_3D, ((
                                                                           i_hs - 1) * ny + j_hs) * nz_base + k_hs] = C_x_cond_hs_3D; coef_diag_hs -= C_x_cond_hs_3D
                        if i_hs < nx - 1: A_matrix_heatsink[idx_hs_3D, ((
                                                                                i_hs + 1) * ny + j_hs) * nz_base + k_hs] = C_x_cond_hs_3D; coef_diag_hs -= C_x_cond_hs_3D
                        if j_hs > 0: A_matrix_heatsink[idx_hs_3D, (i_hs * ny + (
                                j_hs - 1)) * nz_base + k_hs] = C_y_cond_hs_3D; coef_diag_hs -= C_y_cond_hs_3D
                        if j_hs < ny - 1: A_matrix_heatsink[idx_hs_3D, (i_hs * ny + (
                                j_hs + 1)) * nz_base + k_hs] = C_y_cond_hs_3D; coef_diag_hs -= C_y_cond_hs_3D
                        if k_hs == 0:
                            if nz_base > 1: A_matrix_heatsink[idx_hs_3D, (i_hs * ny + j_hs) * nz_base + (
                                    k_hs + 1)] = C_z_cond_hs_3D; coef_diag_hs -= C_z_cond_hs_3D
                            h_local = h_eff_FDM_heatsink_arg[i_hs, j_hs] if is_h_array else h_eff_FDM_heatsink_arg
                            coef_diag_hs -= h_local * dA_xy
                        elif k_hs == nz_base - 1:
                            A_matrix_heatsink[idx_hs_3D, (i_hs * ny + j_hs) * nz_base + (k_hs - 1)] = C_z_cond_hs_3D;
                            coef_diag_hs -= C_z_cond_hs_3D
                            if heatsink_covered_by_module_map[
                                i_hs, j_hs]: coef_diag_hs -= H_MODULE_TO_HEATSINK_INTERFACE * dA_xy
                        else:
                            A_matrix_heatsink[idx_hs_3D, (i_hs * ny + j_hs) * nz_base + (k_hs - 1)] = C_z_cond_hs_3D;
                            coef_diag_hs -= C_z_cond_hs_3D
                            A_matrix_heatsink[idx_hs_3D, (i_hs * ny + j_hs) * nz_base + (k_hs + 1)] = C_z_cond_hs_3D;
                            coef_diag_hs -= C_z_cond_hs_3D
                        A_matrix_heatsink[idx_hs_3D, idx_hs_3D] = coef_diag_hs

        LU_heatsink = splu(A_matrix_heatsink.tocsr())

        max_iterations = 100;
        convergence_tolerance = 0.01;
        iteration = 0;
        converged = False
        T_solution_old_iter = T_solution.copy()
        T_module_base_map_global = np.full((nx, ny), t_ambient_inlet_arg + 5.0)
        print(f"[SimCore] Iniciando bucle iterativo ({sim_type} Disipador <-> Módulos IGBT <-> Aire)...")

        while not converged and iteration < max_iterations:
            iteration += 1
            # PASO 1: Módulos
            for module_item in modules_data_sim:
                nx_mod, ny_mod = module_item['nx_mod_local'], module_item['ny_mod_local'];
                N_nodes_mod = nx_mod * ny_mod
                if N_nodes_mod == 0: continue
                q_chip_source_map_module = np.zeros((nx_mod, ny_mod))
                for chip_item in module_item['chips']:
                    if chip_item['power'] <= 0 or chip_item['center_x_idx_local'] < 0: continue
                    i_min_chip_loc = max(0, chip_item['center_x_idx_local'] - chip_item['nx_half_local']);
                    i_max_chip_loc = min(nx_mod - 1, chip_item['center_x_idx_local'] + chip_item['nx_half_local'])
                    j_min_chip_loc = max(0, chip_item['center_y_idx_local'] - chip_item['ny_half_local']);
                    j_max_chip_loc = min(ny_mod - 1, chip_item['center_y_idx_local'] + chip_item['ny_half_local'])
                    num_nodes_chip_on_module = (i_max_chip_loc - i_min_chip_loc + 1) * (
                            j_max_chip_loc - j_min_chip_loc + 1)
                    if num_nodes_chip_on_module > 0: q_chip_source_map_module[i_min_chip_loc:i_max_chip_loc + 1,
                                                     j_min_chip_loc:j_max_chip_loc + 1] += chip_item[
                                                                                               'power'] / num_nodes_chip_on_module
                A_mod = lil_matrix((N_nodes_mod, N_nodes_mod));
                b_mod = np.zeros(N_nodes_mod)
                C_x_mod = K_MODULE_BASEPLATE * T_MODULE_BASEPLATE * dy / dx if dx > 1e-12 else 0;
                C_y_mod = K_MODULE_BASEPLATE * T_MODULE_BASEPLATE * dx / dy if dy > 1e-12 else 0
                for r_loc in range(nx_mod):
                    for c_loc in range(ny_mod):
                        if not module_item['is_on_heatsink_map'][
                            r_loc, c_loc]: continue
                        idx_m = r_loc * ny_mod + c_loc;
                        g_r, g_c = module_item['footprint_i_min_global'] + r_loc, module_item[
                            'footprint_j_min_global'] + c_loc
                        T_hs_under_mod_node = T_solution[g_r, g_c, nz_base - 1] if nz_base > 1 else T_solution[g_r, g_c]
                        C_h_int_mod = H_MODULE_TO_HEATSINK_INTERFACE * dA_xy;
                        coef_d_mod = -C_h_int_mod;
                        P_chip_W_node = q_chip_source_map_module[r_loc, c_loc]
                        b_mod[idx_m] = -P_chip_W_node - C_h_int_mod * T_hs_under_mod_node
                        if r_loc > 0 and module_item['is_on_heatsink_map'][r_loc - 1, c_loc]: A_mod[
                            idx_m, idx_m - ny_mod] = C_x_mod; coef_d_mod -= C_x_mod
                        if r_loc < nx_mod - 1 and module_item['is_on_heatsink_map'][r_loc + 1, c_loc]: A_mod[
                            idx_m, idx_m + ny_mod] = C_x_mod; coef_d_mod -= C_x_mod
                        if c_loc > 0 and module_item['is_on_heatsink_map'][r_loc, c_loc - 1]: A_mod[
                            idx_m, idx_m - 1] = C_y_mod; coef_d_mod -= C_y_mod
                        if c_loc < ny_mod - 1 and module_item['is_on_heatsink_map'][r_loc, c_loc + 1]: A_mod[
                            idx_m, idx_m + 1] = C_y_mod; coef_d_mod -= C_y_mod
                        A_mod[idx_m, idx_m] = coef_d_mod
                LU_m = splu(A_mod.tocsr());
                T_f_m = LU_m.solve(b_mod);
                module_item['T_module_internal_solution'] = T_f_m.reshape((nx_mod, ny_mod))
                for r_loc in range(nx_mod):
                    for c_loc in range(ny_mod):
                        g_r, g_c = module_item['footprint_i_min_global'] + r_loc, module_item[
                            'footprint_j_min_global'] + c_loc
                        T_module_base_map_global[g_r, g_c] = module_item['T_module_internal_solution'][r_loc, c_loc]

            # PASO 2: Disipador
            b_vector_heatsink = np.zeros(N_nodes_heatsink)
            if nz_base == 1:
                q_source_from_modules_W_per_m2_2D = np.zeros((nx, ny))
                for mod_item_b in modules_data_sim:
                    for r_loc_b in range(mod_item_b['nx_mod_local']):
                        for c_loc_b in range(mod_item_b['ny_mod_local']):
                            if mod_item_b['is_on_heatsink_map'][r_loc_b, c_loc_b]:
                                g_r_b, g_c_b = mod_item_b['footprint_i_min_global'] + r_loc_b, mod_item_b[
                                    'footprint_j_min_global'] + c_loc_b
                                T_mod_base_node = mod_item_b['T_module_internal_solution'][r_loc_b, c_loc_b];
                                T_hs_node = T_solution[g_r_b, g_c_b]
                                P_node_mod_to_hs = H_MODULE_TO_HEATSINK_INTERFACE * dA_xy * (
                                        T_mod_base_node - T_hs_node)
                                q_source_from_modules_W_per_m2_2D[g_r_b, g_c_b] += P_node_mod_to_hs / dA_xy
                for i_hs in range(nx):
                    for j_hs in range(ny):
                        idx_hs_2D = i_hs * ny + j_hs
                        q_src_W_node_hs = q_source_from_modules_W_per_m2_2D[i_hs, j_hs] * dA_xy
                        h_local = h_eff_FDM_heatsink_arg[i_hs, j_hs] if is_h_array else h_eff_FDM_heatsink_arg
                        b_vector_heatsink[idx_hs_2D] = -q_src_W_node_hs - h_local * dA_xy * T_air_solution[i_hs, j_hs]
                T_flat_hs = LU_heatsink.solve(b_vector_heatsink);
                T_solution_new = T_flat_hs.reshape((nx, ny))
            else:  # 3D
                for i_hs in range(nx):
                    for j_hs in range(ny):
                        for k_hs in range(nz_base):
                            idx_hs_3D = (i_hs * ny + j_hs) * nz_base + k_hs
                            if k_hs == 0:
                                h_local = h_eff_FDM_heatsink_arg[i_hs, j_hs] if is_h_array else h_eff_FDM_heatsink_arg
                                b_vector_heatsink[idx_hs_3D] -= h_local * dA_xy * T_air_solution[i_hs, j_hs]
                            elif k_hs == nz_base - 1:
                                if heatsink_covered_by_module_map[i_hs, j_hs]: b_vector_heatsink[
                                    idx_hs_3D] -= H_MODULE_TO_HEATSINK_INTERFACE * dA_xy * T_module_base_map_global[
                                    i_hs, j_hs]
                T_flat_hs = LU_heatsink.solve(b_vector_heatsink);
                T_solution_new = T_flat_hs.reshape((nx, ny, nz_base))

            # PASO 3: Temperaturas del Aire
            delta_T_air_nodes = np.zeros_like(T_air_solution)
            T_hs_convecting_surface = T_solution_new[:, :, 0] if nz_base > 1 else T_solution_new
            for r_ta in range(nx):
                for c_ta in range(ny):
                    h_local_for_air = h_eff_FDM_heatsink_arg[r_ta, c_ta] if is_h_array else h_eff_FDM_heatsink_arg
                    P_conv_node_hs_to_air = h_local_for_air * dA_xy * (
                            T_hs_convecting_surface[r_ta, c_ta] - T_air_solution[r_ta, c_ta])
                    delta_T_air_nodes[r_ta, c_ta] = max(0,
                                                        P_conv_node_hs_to_air) / m_dot_cp_column if m_dot_cp_column > 1e-12 else 0
            T_air_next = np.full_like(T_air_solution, t_ambient_inlet_arg)
            for r_acc in range(nx):
                for c_acc in range(1, ny): T_air_next[r_acc, c_acc] = T_air_next[r_acc, c_acc - 1] + delta_T_air_nodes[
                    r_acc, c_acc - 1]

            # Convergencia y actualización
            if iteration > 1: max_diff_T_hs = np.max(
                np.abs(T_solution_new - T_solution_old_iter)); converged = max_diff_T_hs < convergence_tolerance
            T_solution_old_iter = T_solution_new.copy();
            T_solution = T_solution_new.copy();
            T_air_solution = T_air_next.copy()
            if iteration % 10 == 0 or iteration == 1 or (converged and iteration > 1):
                max_T_sol_iter = np.max(T_solution) if not np.isnan(T_solution).all() else np.nan;
                diff_str = f"{max_diff_T_hs:.4f}°C" if iteration > 1 else "N/A"
                print(
                    f"[SimCore] Iter {iteration}: Max ΔT_hs = {diff_str}. T_max_hs = {max_T_sol_iter:.2f}°C. T_max_air = {np.max(T_air_solution[:, -1]):.2f}°C")

        print(f"[SimCore] Bucle terminado en {iteration} iteraciones. Converged: {converged}")
        results['convergence'] = converged;
        results['iterations'] = iteration
        if np.isnan(T_solution).any():
            results['t_max_base'] = np.nan;
            results['t_avg_base'] = np.nan
        else:
            results['t_max_base'] = np.max(T_solution);
            results['t_avg_base'] = np.mean(T_solution)
        if ny > 0 and not np.isnan(T_air_solution[:, ny - 1]).any():
            results['t_air_outlet'] = np.mean(T_air_solution[:, ny - 1])
        else:
            results['t_air_outlet'] = np.nan

        # Post-procesamiento Tj, T_ntc
        max_tj_overall = -float('inf');
        max_tj_chip_label = "";
        max_t_ntc_overall = -float('inf');
        max_t_module_surface_overall = -float('inf')
        T_hs_surface_for_chips = T_solution[:, :,
                                 nz_base - 1] if nz_base > 1 else T_solution

        def get_max_temp_in_area_local(T_matrix_func, center_x_idx_func, center_y_idx_func, nx_half_func,
                                       ny_half_func, Max_Nx_func, Max_Ny_func):
            """
            Calcula la TEMPERATURA MÁXIMA en un área rectangular de una matriz de temperaturas.
            Este método es más conservador y físicamente representativo para el cálculo de Tj.
            """
            i_min_func = max(0, center_x_idx_func - nx_half_func)
            i_max_func = min(Max_Nx_func - 1, center_x_idx_func + nx_half_func)
            j_min_func = max(0, center_y_idx_func - ny_half_func)
            j_max_func = min(Max_Ny_func - 1, center_y_idx_func + ny_half_func)

            if i_min_func > i_max_func or j_min_func > j_max_func: return np.nan

            i_start_safe = max(0, min(i_min_func, T_matrix_func.shape[0] - 1));
            i_end_safe = max(0, min(i_max_func, T_matrix_func.shape[0] - 1))
            j_start_safe = max(0, min(j_min_func, T_matrix_func.shape[1] - 1));
            j_end_safe = max(0, min(j_max_func, T_matrix_func.shape[1] - 1))

            if i_start_safe > i_end_safe or j_start_safe > j_end_safe: return np.nan

            area_nodes_func = T_matrix_func[i_start_safe:i_end_safe + 1, j_start_safe:j_end_safe + 1]
            if area_nodes_func.size == 0: return np.nan

            return np.nanmax(area_nodes_func)

        _get_max_temp_in_area_local = get_max_temp_in_area_local

        module_results_list = []
        print("\n--- [Post-Procesamiento: Cálculo de Tj y T_NTC] ---")
        for module_item_post in modules_data_sim:
            module_result = {'id': module_item_post['id'], 'chips': [], 't_ntc': np.nan}
            T_module_internal_map = module_item_post['T_module_internal_solution']
            max_t_module_surface_overall = np.nanmax([max_t_module_surface_overall, np.nanmax(T_module_internal_map)])

            for chip_item_post in module_item_post['chips']:
                T_base_chip_on_hs_val = _get_max_temp_in_area_local(
                    T_hs_surface_for_chips,
                    chip_item_post['center_x_idx_global'],
                    chip_item_post['center_y_idx_global'],
                    chip_item_post['nx_half_global'],
                    chip_item_post['ny_half_global'], nx, ny)
                chip_item_post['T_base_chip_on_heatsink'] = T_base_chip_on_hs_val

                if chip_item_post['center_x_idx_local'] >= 0 and chip_item_post['center_y_idx_local'] >= 0:
                    T_base_chip_on_mod_surf_val = _get_max_temp_in_area_local(
                        T_module_internal_map,
                        chip_item_post['center_x_idx_local'],
                        chip_item_post['center_y_idx_local'],
                        chip_item_post['nx_half_local'],
                        chip_item_post['ny_half_local'],
                        module_item_post['nx_mod_local'],
                        module_item_post['ny_mod_local'])
                else:
                    T_base_chip_on_mod_surf_val = np.nan
                chip_item_post['T_base_chip_on_module_surface'] = T_base_chip_on_mod_surf_val

                Tj_calc_val = T_base_chip_on_mod_surf_val + chip_item_post['Rth_jhs'] * chip_item_post['power'] \
                    if not np.isnan(T_base_chip_on_mod_surf_val) and chip_item_post['power'] > 1e-6 \
                    else T_base_chip_on_mod_surf_val
                chip_item_post['Tj'] = Tj_calc_val

                if not np.isnan(Tj_calc_val) and Tj_calc_val > max_tj_overall:
                    max_tj_overall = Tj_calc_val
                    max_tj_chip_label = f"{chip_item_post['label']} (P={chip_item_post['power']:.1f}W)"

                module_result['chips'].append(
                    {'suffix': chip_item_post['suffix'], 't_base_heatsink': T_base_chip_on_hs_val,
                     't_base_module_surface': T_base_chip_on_mod_surf_val, 'tj': chip_item_post['Tj']})

            # ##################################################################################
            # ### INICIO DE MODIFICACIÓN: Cálculo de T_NTC directo desde el mapa T_base_mod  ###
            # ##################################################################################

            # El método anterior basado en interpoladores (calculate_ntc_temperature_advanced)
            # se reemplaza por una lectura directa de la temperatura en la base del módulo.

            T_ntc_final_module = np.nan  # Valor por defecto si el NTC no es válido

            # Obtenemos los índices locales del NTC, que fueron pre-calculados al inicio
            ntc_i_idx_local = module_item_post.get('ntc_i_idx_local')
            ntc_j_idx_local = module_item_post.get('ntc_j_idx_local')

            if ntc_i_idx_local is not None and ntc_j_idx_local is not None:
                # 'T_module_internal_map' es el "plano T_base_mod" solicitado.
                # Se obtiene la temperatura directamente del nodo correspondiente.
                T_ntc_final_module = T_module_internal_map[ntc_i_idx_local, ntc_j_idx_local]
                # Log para verificar el cálculo (opcional)
                # print(f"  - Módulo '{module_item_post['id']}': T_ntc leída de T_base_mod[{ntc_i_idx_local}, {ntc_j_idx_local}] = {T_ntc_final_module:.2f}°C")
            else:
                # Esto ocurre si las coordenadas del NTC caen fuera del footprint del módulo.
                # print(f"  - ADVERTENCIA: NTC para el módulo '{module_item_post['id']}' no tiene índices locales válidos. T_ntc será NaN.")
                pass

            # Asignamos el valor calculado al resultado del módulo.
            module_result['t_ntc'] = T_ntc_final_module

            # ##################################################################################
            # ### FIN DE MODIFICACIÓN                                                        ###
            # ##################################################################################

            if not np.isnan(T_ntc_final_module): max_t_ntc_overall = np.nanmax([max_t_ntc_overall, T_ntc_final_module])
            module_results_list.append(module_result)

        results['t_max_junction'] = max_tj_overall if not np.isinf(max_tj_overall) else np.nan
        results['t_max_junction_chip'] = max_tj_chip_label
        results['t_max_ntc'] = max_t_ntc_overall if not np.isinf(max_t_ntc_overall) else np.nan
        results['module_results'] = module_results_list

        # --- Generación de Plots ---
        x_coords_gfx = np.linspace(0, lx, nx);
        y_coords_gfx = np.linspace(0, ly, ny)
        X_gfx, Y_gfx = np.meshgrid(x_coords_gfx, y_coords_gfx, indexing='ij')
        T_solution_surface_plot = T_solution[:, :,
                                  0] if nz_base > 1 else T_solution
        plot_title_suffix_hs = "(Cara Inferior Disipador)" if nz_base > 1 else "(Base Disipador 2D)"

        try:
            fig_main, axes_main = plt.subplots(1, 2, figsize=(11, 4.5), dpi=100);
            ax_base, ax_air = axes_main[0], axes_main[1]
            min_T_plot_hs = t_ambient_inlet_arg
            max_T_from_fdm_volume = results.get('t_max_base', t_ambient_inlet_arg + 1.0)
            if np.isnan(max_T_from_fdm_volume): max_T_from_fdm_volume = t_ambient_inlet_arg + 1.0
            max_T_plot_hs_val = np.nanmax([np.nanmax(T_solution_surface_plot) if not np.isnan(
                T_solution_surface_plot).all() else t_ambient_inlet_arg + 1.0, max_T_from_fdm_volume])
            max_T_plot_hs_val = max(max_T_plot_hs_val, min_T_plot_hs + 1.0);
            max_T_plot_hs_val += (max_T_plot_hs_val - min_T_plot_hs) * 0.05
            levels_main_hs = np.linspace(min_T_plot_hs, max_T_plot_hs_val, 30) if abs(
                max_T_plot_hs_val - min_T_plot_hs) > 1e-3 else np.array([min_T_plot_hs - 0.5, max_T_plot_hs_val + 0.5])
            contour_base = ax_base.contourf(X_gfx, Y_gfx,
                                            np.nan_to_num(T_solution_surface_plot, nan=min_T_plot_hs - 10),
                                            levels=levels_main_hs, cmap="hot", extend='max')
            fig_main.colorbar(contour_base, ax=ax_base, label=f"T {plot_title_suffix_hs} (°C)")
            tmax_surf_str = f"{np.nanmax(T_solution_surface_plot):.1f}°C" if not np.isnan(
                T_solution_surface_plot).all() else "N/A"
            ax_base.set_title(f"Temp. {plot_title_suffix_hs} (Max Sup. Graficada={tmax_surf_str})");
            ax_base.set_aspect('equal')
            T_air_max_plot = np.nanmax(T_air_solution) if not np.isnan(T_air_solution).all() else t_ambient_inlet_arg
            min_T_air_plot = t_ambient_inlet_arg;
            max_T_air_plot_val = max(T_air_max_plot, min_T_air_plot + 0.1);
            max_T_air_plot_val += (max_T_air_plot_val - min_T_air_plot) * 0.05
            levels_air = np.linspace(min_T_air_plot, max_T_air_plot_val, 30) if abs(
                max_T_air_plot_val - min_T_air_plot) > 1e-3 else np.array(
                [min_T_air_plot - 0.5, max_T_air_plot_val + 0.5])
            contour_air = ax_air.contourf(X_gfx, Y_gfx, np.nan_to_num(T_air_solution, nan=min_T_air_plot - 10),
                                          levels=levels_air, cmap="coolwarm", extend='max')
            fig_main.colorbar(contour_air, ax=ax_air, label="T Aire (°C)")
            tairout_str = f"{results['t_air_outlet']:.1f}°C" if not np.isnan(results['t_air_outlet']) else "N/A"
            ax_air.set_title(f"Temp. Aire (Salida Prom={tairout_str})");
            ax_air.set_aspect('equal')
            for ax_item in axes_main:
                ax_item.set_xlabel("x (m)", fontsize=7);
                ax_item.set_ylabel("y (m)", fontsize=7);
                ax_item.tick_params(axis='both', which='major', labelsize=6)
                for mod_item_gfx in modules_data_sim:
                    cxm, cym = mod_item_gfx['center_x_m'], mod_item_gfx['center_y_m'];
                    rxm, rym = cxm - w_igbt_footprint / 2, cym - h_igbt_footprint / 2
                    rect_mod = plt.Rectangle((rxm, rym), w_igbt_footprint, h_igbt_footprint, edgecolor='gray',
                                             facecolor='none', lw=0.8, ls='--');
                    ax_item.add_patch(rect_mod)
                    if mod_item_gfx['ntc_abs_x'] is not None and mod_item_gfx['ntc_abs_y'] is not None and 0 <= \
                            mod_item_gfx['ntc_abs_x'] <= lx and 0 <= mod_item_gfx['ntc_abs_y'] <= ly:
                        ax_item.plot(mod_item_gfx['ntc_abs_x'], mod_item_gfx['ntc_abs_y'], 'wo', markersize=3,
                                     markeredgecolor='black')
            total_P_sim = sum(c['power'] for m in modules_data_sim for c in m['chips'])
            fig_main.suptitle(
                f"Simulación {sim_type} ({len(modules_data_sim)} Módulos, P_total={total_P_sim:.0f}W)\nh_eff_base={'Array' if is_h_array else f'{h_eff_FDM_heatsink_arg:.1f}'} W/m²K, T_amb={t_ambient_inlet_arg}°C, Q={q_total_m3_h_arg} m³/h",
                fontsize=8)
            fig_main.tight_layout(rect=[0, 0.03, 1, 0.90]);
            results['plot_base_data_uri'] = local_fig_to_data_uri(fig_main, dpi=100)
        except Exception as e_gfx_main:
            print(f"[SimCore] Error GFX Main: {e_gfx_main}");
            results['plot_base_data_uri'] = None

        try:
            num_modules_plot_zoom = len(modules_data_sim)
            if num_modules_plot_zoom == 0: raise ValueError("No modules to plot zoom for.")
            ncols_zoom_gfx = min(3, num_modules_plot_zoom);
            nrows_zoom_gfx = math.ceil(num_modules_plot_zoom / ncols_zoom_gfx)
            fig_zoom_gfx, axes_zoom_list_gfx = plt.subplots(nrows=nrows_zoom_gfx, ncols=ncols_zoom_gfx,
                                                            figsize=(ncols_zoom_gfx * 3.8, nrows_zoom_gfx * 3.5),
                                                            squeeze=False, dpi=100)
            axes_flat_zoom_gfx = axes_zoom_list_gfx.flatten();
            contour_zoom_ref_plot_gfx = None
            min_T_plot_zoom_mod_gfx = t_ambient_inlet_arg
            max_T_plot_zoom_mod_val_gfx = max(max_t_module_surface_overall if not np.isinf(
                max_t_module_surface_overall) else t_ambient_inlet_arg + 10,
                                              results.get('t_max_junction', t_ambient_inlet_arg + 10))
            if np.isnan(max_T_plot_zoom_mod_val_gfx) or np.isinf(
                    max_T_plot_zoom_mod_val_gfx): max_T_plot_zoom_mod_val_gfx = t_ambient_inlet_arg + 20
            max_T_plot_zoom_mod_val_gfx = max(max_T_plot_zoom_mod_val_gfx, min_T_plot_zoom_mod_gfx + 1.0);
            max_T_plot_zoom_mod_val_gfx += (max_T_plot_zoom_mod_val_gfx - min_T_plot_zoom_mod_gfx) * 0.05
            levels_zoom_plot_mod_gfx = np.linspace(min_T_plot_zoom_mod_gfx, max_T_plot_zoom_mod_val_gfx, 30) if abs(
                max_T_plot_zoom_mod_val_gfx - min_T_plot_zoom_mod_gfx) > 1e-3 else np.array(
                [min_T_plot_zoom_mod_gfx - 0.5, max_T_plot_zoom_mod_val_gfx + 0.5])
            for idx_zoom_gfx, module_item_zoom_gfx in enumerate(modules_data_sim):
                if idx_zoom_gfx >= len(axes_flat_zoom_gfx): break
                ax_z_plot_gfx = axes_flat_zoom_gfx[idx_zoom_gfx]
                T_module_internal_plot = module_item_zoom_gfx['T_module_internal_solution'];
                nx_mod_loc_plot = module_item_zoom_gfx['nx_mod_local'];
                ny_mod_loc_plot = module_item_zoom_gfx['ny_mod_local']
                mod_origin_x_global = module_item_zoom_gfx['center_x_m'] - w_igbt_footprint / 2;
                mod_origin_y_global = module_item_zoom_gfx['center_y_m'] - h_igbt_footprint / 2
                x_coords_mod_local_gfx = np.linspace(mod_origin_x_global, mod_origin_x_global + w_igbt_footprint,
                                                     nx_mod_loc_plot) if nx_mod_loc_plot > 1 else np.array(
                    [mod_origin_x_global])
                y_coords_mod_local_gfx = np.linspace(mod_origin_y_global, mod_origin_y_global + h_igbt_footprint,
                                                     ny_mod_loc_plot) if ny_mod_loc_plot > 1 else np.array(
                    [mod_origin_y_global])
                if nx_mod_loc_plot > 1 and ny_mod_loc_plot > 1:
                    X_mod_local_gfx, Y_mod_local_gfx = np.meshgrid(x_coords_mod_local_gfx, y_coords_mod_local_gfx,
                                                                   indexing='ij')
                    T_module_plot_safe = np.nan_to_num(T_module_internal_plot, nan=min_T_plot_zoom_mod_gfx - 10)
                    contour_zoom_gfx_detail = ax_z_plot_gfx.contourf(X_mod_local_gfx, Y_mod_local_gfx,
                                                                     T_module_plot_safe,
                                                                     levels=levels_zoom_plot_mod_gfx, cmap="hot",
                                                                     extend='max')
                    if idx_zoom_gfx == 0: contour_zoom_ref_plot_gfx = contour_zoom_gfx_detail
                elif T_module_internal_plot.size > 0:
                    ax_z_plot_gfx.text(0.5, 0.5, f"AvgT: {np.nanmean(T_module_internal_plot):.1f}°C",
                                       transform=ax_z_plot_gfx.transAxes, ha='center', va='center')
                center_x_mod_zoom_gfx, center_y_mod_zoom_gfx = module_item_zoom_gfx['center_x_m'], module_item_zoom_gfx[
                    'center_y_m']
                rect_x_mod_zoom_gfx, rect_y_mod_zoom_gfx = center_x_mod_zoom_gfx - w_igbt_footprint / 2, center_y_mod_zoom_gfx - h_igbt_footprint / 2
                rect_mod_zoom_plot_gfx_obj = plt.Rectangle((rect_x_mod_zoom_gfx, rect_y_mod_zoom_gfx), w_igbt_footprint,
                                                           h_igbt_footprint, edgecolor='black', facecolor='none',
                                                           lw=1.0, ls='--');
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

                        chip_suffix_for_plot = original_chip_item['suffix']
                        rect_chip_color = 'gray'
                        if "IGBT" in chip_suffix_for_plot: rect_chip_color = 'cyan'
                        if "Diode" in chip_suffix_for_plot: rect_chip_color = 'lime'

                        rect_chip_zoom_plot_gfx_obj = plt.Rectangle((rect_x_phys_zoom_gfx, rect_y_phys_zoom_gfx),
                                                                    chip_w_plot_zoom_gfx, chip_h_plot_zoom_gfx,
                                                                    edgecolor=rect_chip_color,
                                                                    facecolor='none', lw=1.0, ls=':');
                        ax_z_plot_gfx.add_patch(rect_chip_zoom_plot_gfx_obj)
                        tj_val = chip_plot_info['tj'];
                        tbase_mod_surf_val = chip_plot_info['t_base_module_surface']
                        tj_str_zoom_gfx = f"Tj={tj_val:.1f}" if not np.isnan(tj_val) else "Tj=N/A";
                        tbase_label = "Tb_max_surf"
                        tbase_str_zoom_gfx = f"{tbase_label}={tbase_mod_surf_val:.1f}" if not np.isnan(
                            tbase_mod_surf_val) else f"{tbase_label}=N/A"

                        # ######################################################################
                        # ### INICIO DE LA MODIFICACIÓN CORREGIDA ################################
                        # ######################################################################

                        y_offset = chip_h_plot_zoom_gfx * 0.15

                        # Lógica robusta: comprobar el sufijo del chip directamente
                        if "Diode" in chip_suffix_for_plot:
                            # Para TODOS los Diodos: desplazar hacia ARRIBA
                            vertical_alignment = 'bottom'
                            text_y_position = center_y_phys_zoom_gfx + y_offset
                        elif "IGBT" in chip_suffix_for_plot:
                            # Para TODOS los IGBTs (transistores): desplazar hacia ABAJO
                            vertical_alignment = 'top'
                            text_y_position = center_y_phys_zoom_gfx - y_offset
                        else:
                            # Posición por defecto si no es ni diodo ni IGBT
                            vertical_alignment = 'center'
                            text_y_position = center_y_phys_zoom_gfx

                        ax_z_plot_gfx.text(
                            center_x_phys_zoom_gfx,
                            text_y_position,
                            f"{chip_suffix_for_plot}\n{tj_str_zoom_gfx}\n({tbase_str_zoom_gfx})",
                            color='white',
                            ha='center',
                            va=vertical_alignment,
                            fontsize=5.5,
                            bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.6)
                        )
                        # ######################################################################
                        # ### FIN DE LA MODIFICACIÓN CORREGIDA #################################
                        # ######################################################################

                margin_factor_zoom_gfx = 0.1
                zoom_x_min_plot_gfx = module_item_zoom_gfx['center_x_m'] - (w_igbt_footprint / 2) * (
                        1 + margin_factor_zoom_gfx);
                zoom_x_max_plot_gfx = module_item_zoom_gfx['center_x_m'] + (w_igbt_footprint / 2) * (
                        1 + margin_factor_zoom_gfx)
                zoom_y_min_plot_gfx = module_item_zoom_gfx['center_y_m'] - (h_igbt_footprint / 2) * (
                        1 + margin_factor_zoom_gfx);
                zoom_y_max_plot_gfx = module_item_zoom_gfx['center_y_m'] + (h_igbt_footprint / 2) * (
                        1 + margin_factor_zoom_gfx)
                if module_item_zoom_gfx['ntc_abs_x'] is not None and module_item_zoom_gfx['ntc_abs_y'] is not None:
                    ntc_x_plot_zoom_gfx, ntc_y_plot_zoom_gfx = module_item_zoom_gfx['ntc_abs_x'], module_item_zoom_gfx[
                        'ntc_abs_y']
                    if zoom_x_min_plot_gfx <= ntc_x_plot_zoom_gfx <= zoom_x_max_plot_gfx and zoom_y_min_plot_gfx <= ntc_y_plot_zoom_gfx <= zoom_y_max_plot_gfx:
                        ax_z_plot_gfx.plot(ntc_x_plot_zoom_gfx, ntc_y_plot_zoom_gfx, 'ro', markersize=4,
                                           markeredgecolor='white')

                        # Se busca el T_ntc final en el diccionario de resultados para mostrarlo en el plot
                        t_ntc_val_plot_zoom_gfx = np.nan
                        if current_module_result_for_plot:
                            t_ntc_val_plot_zoom_gfx = current_module_result_for_plot.get('t_ntc', np.nan)

                        t_ntc_str_plot_zoom_gfx = f"NTC≈{t_ntc_val_plot_zoom_gfx:.1f}" if not np.isnan(
                            t_ntc_val_plot_zoom_gfx) else "NTC=N/A"
                        ax_z_plot_gfx.text(ntc_x_plot_zoom_gfx + (zoom_x_max_plot_gfx - zoom_x_min_plot_gfx) * 0.03,
                                           ntc_y_plot_zoom_gfx, t_ntc_str_plot_zoom_gfx, color='red', ha='left',
                                           va='center', fontsize=6,
                                           bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.7))
                ax_z_plot_gfx.set_xlim(zoom_x_min_plot_gfx, zoom_x_max_plot_gfx);
                ax_z_plot_gfx.set_ylim(zoom_y_min_plot_gfx, zoom_y_max_plot_gfx)
                ax_z_plot_gfx.set_title(f"{module_item_zoom_gfx['id']} (T Baseplate Módulo)", fontsize=8)
                ax_z_plot_gfx.set_xlabel("x (m)", fontsize=7);
                ax_z_plot_gfx.set_ylabel("y (m)", fontsize=7);
                ax_z_plot_gfx.tick_params(axis='both', which='major', labelsize=6)
                ax_z_plot_gfx.set_aspect('equal', adjustable='box');
                ax_z_plot_gfx.grid(True, linestyle=':', alpha=0.3)
            for i_ax_flat_zoom_gfx in range(idx_zoom_gfx + 1, len(axes_flat_zoom_gfx)): axes_flat_zoom_gfx[
                i_ax_flat_zoom_gfx].axis('off')
            if contour_zoom_ref_plot_gfx is not None and num_modules_plot_zoom > 0:
                fig_zoom_gfx.subplots_adjust(right=0.85, top=0.85, bottom=0.1)
                cbar_ax_zoom_gfx = fig_zoom_gfx.add_axes([0.88, 0.15, 0.03, 0.7])
                cbar_zoom_gfx = fig_zoom_gfx.colorbar(contour_zoom_ref_plot_gfx, cax=cbar_ax_zoom_gfx,
                                                      label="T Baseplate Módulo (°C)")
                cbar_zoom_gfx.ax.tick_params(labelsize=7);
                cbar_zoom_gfx.set_label("T Baseplate Módulo (°C)", size=8)
            fig_zoom_gfx.suptitle(f"Detalle Módulos (Temperatura Superficie Baseplate Módulo)", fontsize=10)
            results['plot_zoom_data_uri'] = local_fig_to_data_uri(fig_zoom_gfx, dpi=100)
        except Exception as e_gfx_zoom:
            print(f"[SimCore] Error GFX Zoom: {e_gfx_zoom}");
            results['plot_zoom_data_uri'] = None

        try:
            desired_dpi_interactive = 100;
            fig_width_inches = max(1, nx / desired_dpi_interactive if nx > desired_dpi_interactive else lx * 10);
            fig_height_inches = max(1, ny / desired_dpi_interactive if ny > desired_dpi_interactive else ly * 10)
            fig_width_inches = min(fig_width_inches, 12);
            fig_height_inches = min(fig_height_inches, 12)
            fig_interactive_raw, ax_interactive_raw = plt.subplots(figsize=(fig_width_inches, fig_height_inches),
                                                                   dpi=desired_dpi_interactive)
            ax_interactive_raw.set_position([0, 0, 1, 1]);
            ax_interactive_raw.axis('off')
            ax_interactive_raw.contourf(X_gfx, Y_gfx, np.nan_to_num(T_solution_surface_plot, nan=min_T_plot_hs - 10),
                                        levels=levels_main_hs, cmap="hot", extend='max')
            results['plot_interactive_raw_uri'] = local_fig_to_data_uri(fig_interactive_raw,
                                                                        dpi=desired_dpi_interactive, pad_inches=0,
                                                                        bbox_inches=None)
            print("[SimCore] Gráfico RAW interactivo generado.")
        except Exception as e_gfx_interactive:
            print(f"[SimCore] Error GFX Interactivo RAW: {e_gfx_interactive}");
            results['plot_interactive_raw_uri'] = None
            if results['status'] != 'Error': results['status'] = 'Success_NoInteractiveData'; results[
                'error_message'] = (
                    results.get('error_message', '') + f" | Error GFX Interactivo: {e_gfx_interactive}").strip(' |')

        if results['plot_interactive_raw_uri']:
            results['T_solution_matrix'] = T_solution_surface_plot;
            results['x_coordinates_vector'] = x_coords_gfx;
            results['y_coordinates_vector'] = y_coords_gfx
            if nz_base > 1: results['T_solution_full_3D_matrix'] = T_solution
        else:
            results['T_solution_matrix'] = None;
            results['x_coordinates_vector'] = None;
            results['y_coordinates_vector'] = None
        if results['status'] != 'Error': results['status'] = 'Success_NoInteractiveData' if not results[
            'plot_interactive_raw_uri'] else 'Success'

    except Exception as e_general:
        import traceback
        print(f"[SimCore] Error general en simulación: {e_general}");
        traceback.print_exc()
        results['status'] = 'Error';
        results['error_message'] = f"Error general: {str(e_general)}"
        results['plot_base_data_uri'] = None;
        results['plot_zoom_data_uri'] = None;
        results['plot_interactive_raw_uri'] = None
        results['T_solution_matrix'] = None;
        results['x_coordinates_vector'] = None;
        results['y_coordinates_vector'] = None

    results['T_solution_full'] = T_solution
    results['T_air_solution_full'] = T_air_solution

    print(
        f"[SimCore] Simulación ({sim_type}) completada en {time.time() - start_time_sim:.2f}s. Estado: {results['status']}")
    return results

def run_simulation_with_h_iteration(
        max_h_iterations=10, h_convergence_tolerance=10,
        lx_base_h_calc=None, ly_base_h_calc=None, q_total_m3_h_h_calc=None,
        t_ambient_inlet_h_calc=None, assumed_duct_height_h_calc=None,
        k_heatsink_material_h_calc=None, fin_params_h_calc=None,
        rth_heatsink_fallback_h_calc=None,
        specific_chip_powers_sim=None, lx_sim=None, ly_sim=None, t_sim_base_fdm=None,
        module_definitions_sim=None,
        nx_sim=Nx_base_default, ny_sim=Ny_base_default, nz_base_sim=Nz_base_default,
        use_local_h_for_flat_plate=True
):
    print("\n" + "=" * 50 + "\nINICIANDO SIMULACIÓN CON ITERACIÓN DE 'h' (CON WARM START)\n" + "=" * 50)
    t_surface_avg_current_estimate = t_ambient_inlet_h_calc + 20.0
    last_h_field_or_scalar = -1.0
    final_simulation_results = None
    converged_h = False

    last_T_solution_field = None
    last_T_air_solution_field = None

    for i_h_iter in range(max_h_iterations):
        print(f"\n--- Iteración de 'h' #{i_h_iter + 1} / {max_h_iterations} ---")
        print(f"Usando T_superficie_promedio_estimada = {t_surface_avg_current_estimate:.2f} °C para calcular 'h'")

        is_flat_plate = (fin_params_h_calc.get('num_fins', 0) == 0 or fin_params_h_calc.get('h_fin', 0.0) <= 1e-6)
        should_return_array = use_local_h_for_flat_plate and is_flat_plate

        current_h_field_or_scalar = calculate_h_array_or_eff(
            lx_base=lx_base_h_calc, ly_base=ly_base_h_calc, q_total_m3_h=q_total_m3_h_h_calc,
            t_ambient_inlet=t_ambient_inlet_h_calc, assumed_duct_height=assumed_duct_height_h_calc,
            k_heatsink_material=k_heatsink_material_h_calc, fin_params=fin_params_h_calc,
            t_surface_avg_estimate=t_surface_avg_current_estimate,
            nx_grid=nx_sim if should_return_array else None,
            ny_grid=ny_sim if should_return_array else None,
            return_array_for_flat_plate=should_return_array
        )

        h_to_use_in_sim = current_h_field_or_scalar
        is_current_h_array = isinstance(h_to_use_in_sim, np.ndarray)

        area_base_heatsink = lx_base_h_calc * ly_base_h_calc

        h_avg_for_rth = np.nan
        if is_current_h_array:
            h_avg_for_rth = np.mean(h_to_use_in_sim)
            print(
                f"  h (array): Min={np.min(h_to_use_in_sim):.2f}, Promedio={h_avg_for_rth:.2f}, Max={np.max(h_to_use_in_sim):.2f} W/m²K")
        else:
            h_avg_for_rth = h_to_use_in_sim
            print(f"  h (scalar): {h_avg_for_rth:.2f} W/m²K")

        if area_base_heatsink > 1e-9 and h_avg_for_rth > 1e-9:
            rth_heatsink_calculated = 1 / (h_avg_for_rth * area_base_heatsink)
            print(
                f"  Rth_heatsink (estimada en esta iter.) = 1 / ({h_avg_for_rth:.2f} W/m²K * {area_base_heatsink:.4f} m²) = {rth_heatsink_calculated:.4f} K/W")
        else:
            print("  No se puede calcular Rth_heatsink (área o h son cero).")

        if (is_current_h_array and (np.any(np.isnan(h_to_use_in_sim)) or np.any(h_to_use_in_sim <= 1e-9))) or \
                (not is_current_h_array and (np.isnan(h_to_use_in_sim) or h_to_use_in_sim <= 1e-9)):
            print(f"ADVERTENCIA: Cálculo de 'h' falló o dio valor no positivo.")
            if not (isinstance(rth_heatsink_fallback_h_calc, (float, int)) and rth_heatsink_fallback_h_calc > 1e-6):
                h_scalar_fallback = 10.0
            else:
                A_base_fb = lx_base_h_calc * ly_base_h_calc
                if A_base_fb <= 1e-9:
                    h_scalar_fallback = 10.0
                else:
                    denom_fb = rth_heatsink_fallback_h_calc * A_base_fb
                    h_scalar_fallback = 1.0 / denom_fb if abs(denom_fb) > 1e-12 else 1e6
            h_scalar_fallback = max(10.0, h_scalar_fallback)
            print(f"Usando h_eff_fallback (escalar) = {h_scalar_fallback:.2f} W/m^2.K")
            h_to_use_in_sim = h_scalar_fallback
            current_h_field_or_scalar = h_scalar_fallback
            is_current_h_array = False

        simulation_results = run_thermal_simulation(
            specific_chip_powers=specific_chip_powers_sim, lx=lx_sim, ly=ly_sim, t=t_sim_base_fdm,
            rth_heatsink=rth_heatsink_fallback_h_calc, module_definitions=module_definitions_sim,
            t_ambient_inlet_arg=t_ambient_inlet_h_calc, q_total_m3_h_arg=q_total_m3_h_h_calc,
            h_eff_FDM_heatsink_arg=h_to_use_in_sim,
            nx=nx_sim, ny=ny_sim, nz_base=nz_base_sim,
            initial_T_solution=last_T_solution_field,
            initial_T_air_solution=last_T_air_solution_field
        )

        final_simulation_results = simulation_results
        if not simulation_results.get('status', 'Error').startswith('Success'):
            print("ERROR: Simulación térmica interna falló.")
            final_simulation_results['h_eff_used_at_failure'] = h_to_use_in_sim
            return final_simulation_results

        last_T_solution_field = simulation_results.get('T_solution_full')
        last_T_air_solution_field = simulation_results.get('T_air_solution_full')

        new_avg_surface_temp_from_sim = simulation_results.get('t_avg_base', np.nan)
        if not np.isnan(new_avg_surface_temp_from_sim):
            print(f"T_avg_base resultante: {new_avg_surface_temp_from_sim:.2f} °C")
            relaxation_factor = 0.7
            t_surface_avg_current_estimate = (relaxation_factor * new_avg_surface_temp_from_sim) + \
                                             ((1 - relaxation_factor) * t_surface_avg_current_estimate)
            print(f"Nueva T_surf_estimada para próxima iter 'h': {t_surface_avg_current_estimate:.2f} °C")
        else:
            print("ADVERTENCIA: No se pudo obtener T_promedio. Estimación de T_surf no actualizada.")

        if i_h_iter > 0:
            diff_h = 0.0
            if is_current_h_array and isinstance(last_h_field_or_scalar,
                                                 np.ndarray) and last_h_field_or_scalar.shape == current_h_field_or_scalar.shape:
                diff_h = np.max(np.abs(current_h_field_or_scalar - last_h_field_or_scalar))
            elif not is_current_h_array and not isinstance(last_h_field_or_scalar,
                                                           np.ndarray) and last_h_field_or_scalar > 0:
                diff_h = abs(current_h_field_or_scalar - last_h_field_or_scalar)
            else:
                diff_h = float('inf')

            if diff_h < h_convergence_tolerance:
                converged_h = True
                print("\n" + "*" * 50 + f"\nCONVERGENCIA DE 'h' ALCANZADA en iter {i_h_iter + 1}.")
                if is_current_h_array:
                    print(f"  - h_array final: Min={np.min(h_to_use_in_sim):.2f}, Max={np.max(h_to_use_in_sim):.2f}")
                else:
                    print(f"  - h_scalar final: {h_to_use_in_sim:.2f}")
                print("*" * 50 + "\n")
                final_simulation_results['h_eff_converged_value'] = h_to_use_in_sim
                final_simulation_results['h_iterations'] = i_h_iter + 1
                final_simulation_results['h_converged_status'] = True

                final_h = final_simulation_results.get('h_eff_converged_value')
                final_rth_heatsink = np.nan
                if final_h is not None:
                    area_base_hs = lx_base_h_calc * ly_base_h_calc
                    h_avg_final = np.mean(final_h) if isinstance(final_h, np.ndarray) else final_h
                    if area_base_hs > 1e-9 and h_avg_final > 1e-9:
                        final_rth_heatsink = 1 / (h_avg_final * area_base_hs)
                final_simulation_results['rth_heatsink_calculated'] = final_rth_heatsink

                return final_simulation_results

        last_h_field_or_scalar = current_h_field_or_scalar

    if not converged_h: print("\nADVERTENCIA: Máx iter 'h' alcanzadas sin convergencia.")
    final_simulation_results['h_eff_converged_value'] = h_to_use_in_sim
    final_simulation_results['h_iterations'] = max_h_iterations
    final_simulation_results['h_converged_status'] = False

    final_h = final_simulation_results.get('h_eff_converged_value')
    final_rth_heatsink = np.nan
    if final_h is not None:
        area_base_hs = lx_base_h_calc * ly_base_h_calc
        h_avg_final = np.mean(final_h) if isinstance(final_h, np.ndarray) else final_h
        if area_base_hs > 1e-9 and h_avg_final > 1e-9:
            final_rth_heatsink = 1 / (h_avg_final * area_base_hs)
    final_simulation_results['rth_heatsink_calculated'] = final_rth_heatsink

    return final_simulation_results


def run_simulation_with_ntc_compensation(
        max_h_iterations=10, h_convergence_tolerance=10,
        lx_base_h_calc=None, ly_base_h_calc=None, q_total_m3_h_h_calc=None,
        t_ambient_inlet_h_calc=None, assumed_duct_height_h_calc=None,
        k_heatsink_material_h_calc=None, fin_params_h_calc=None,
        rth_heatsink_fallback_h_calc=None,
        specific_chip_powers_sim=None, lx_sim=None, ly_sim=None, t_sim_base_fdm=None,
        module_definitions_sim=None,
        nx_sim=Nx_base_default, ny_sim=Ny_base_default, nz_base_sim=Nz_base_default,
        use_local_h_for_flat_plate=True
):
    print("\n" + "=" * 60)
    print("INICIANDO SIMULACIÓN CON COMPENSACIÓN DE NTC (N+1 simulaciones)")
    print("=" * 60 + "\n")

    start_time_total = time.time()
    original_module_definitions = module_definitions_sim
    original_chip_powers = specific_chip_powers_sim

    if not original_module_definitions:
        print("No hay módulos definidos. Ejecutando simulación estándar sin compensación.")
        return run_simulation_with_h_iteration(
            max_h_iterations=max_h_iterations, h_convergence_tolerance=h_convergence_tolerance,
            lx_base_h_calc=lx_base_h_calc, ly_base_h_calc=ly_base_h_calc, q_total_m3_h_h_calc=q_total_m3_h_h_calc,
            t_ambient_inlet_h_calc=t_ambient_inlet_h_calc, assumed_duct_height_h_calc=assumed_duct_height_h_calc,
            k_heatsink_material_h_calc=k_heatsink_material_h_calc, fin_params_h_calc=fin_params_h_calc,
            rth_heatsink_fallback_h_calc=rth_heatsink_fallback_h_calc,
            specific_chip_powers_sim=original_chip_powers, lx_sim=lx_sim, ly_sim=ly_sim, t_sim_base_fdm=t_sim_base_fdm,
            module_definitions_sim=original_module_definitions,
            nx_sim=nx_sim, ny_sim=ny_sim, nz_base_sim=nz_base_sim,
            use_local_h_for_flat_plate=use_local_h_for_flat_plate
        )

    print("\n--- PASO 1: Ejecutando simulación completa con TODOS los módulos ---")
    full_simulation_results = run_simulation_with_h_iteration(
        max_h_iterations=max_h_iterations, h_convergence_tolerance=h_convergence_tolerance,
        lx_base_h_calc=lx_base_h_calc, ly_base_h_calc=ly_base_h_calc, q_total_m3_h_h_calc=q_total_m3_h_h_calc,
        t_ambient_inlet_h_calc=t_ambient_inlet_h_calc, assumed_duct_height_h_calc=assumed_duct_height_h_calc,
        k_heatsink_material_h_calc=k_heatsink_material_h_calc, fin_params_h_calc=fin_params_h_calc,
        rth_heatsink_fallback_h_calc=rth_heatsink_fallback_h_calc,
        specific_chip_powers_sim=original_chip_powers, lx_sim=lx_sim, ly_sim=ly_sim, t_sim_base_fdm=t_sim_base_fdm,
        module_definitions_sim=original_module_definitions,
        nx_sim=nx_sim, ny_sim=ny_sim, nz_base_sim=nz_base_sim,
        use_local_h_for_flat_plate=use_local_h_for_flat_plate
    )

    if not full_simulation_results or not full_simulation_results.get('status', '').startswith('Success'):
        print("ERROR: La simulación completa falló. Abortando el proceso de compensación.")
        return full_simulation_results

    dx = lx_sim / (nx_sim - 1) if nx_sim > 1 else lx_sim
    dy = ly_sim / (ny_sim - 1) if ny_sim > 1 else ly_sim
    ntc_grid_coords_map = {}
    for mod_def in original_module_definitions:
        ntc_abs_x = mod_def['center_x'] + NTC_REL_X
        ntc_abs_y = mod_def['center_y'] + NTC_REL_Y
        ntc_i_idx = int(round(ntc_abs_x / dx)) if dx > 1e-12 else 0
        ntc_j_idx = int(round(ntc_abs_y / dy)) if dy > 1e-12 else 0
        ntc_i_idx_clamped = max(0, min(nx_sim - 1, ntc_i_idx))
        ntc_j_idx_clamped = max(0, min(ny_sim - 1, ntc_j_idx))
        ntc_grid_coords_map[mod_def['id']] = (ntc_i_idx_clamped, ntc_j_idx_clamped)

    delta_t_compensation_map = {}
    print("\n--- PASO 2: Ejecutando simulaciones parciales para calcular Delta_T de compensación ---")

    for module_to_remove in original_module_definitions:
        mod_id_to_remove = module_to_remove['id']
        print(f"\n   -> Simulando escenario SIN el Módulo: '{mod_id_to_remove}'")

        temp_module_defs = [m for m in original_module_definitions if m['id'] != mod_id_to_remove]
        temp_chip_powers = {k: v for k, v in original_chip_powers.items() if not k.startswith(mod_id_to_remove)}

        partial_sim_results = run_simulation_with_h_iteration(
            max_h_iterations=max_h_iterations, h_convergence_tolerance=h_convergence_tolerance,
            lx_base_h_calc=lx_base_h_calc, ly_base_h_calc=ly_base_h_calc, q_total_m3_h_h_calc=q_total_m3_h_h_calc,
            t_ambient_inlet_h_calc=t_ambient_inlet_h_calc, assumed_duct_height_h_calc=assumed_duct_height_h_calc,
            k_heatsink_material_h_calc=k_heatsink_material_h_calc, fin_params_h_calc=fin_params_h_calc,
            rth_heatsink_fallback_h_calc=rth_heatsink_fallback_h_calc,
            specific_chip_powers_sim=temp_chip_powers, lx_sim=lx_sim, ly_sim=ly_sim, t_sim_base_fdm=t_sim_base_fdm,
            module_definitions_sim=temp_module_defs,
            nx_sim=nx_sim, ny_sim=ny_sim, nz_base_sim=nz_base_sim,
            use_local_h_for_flat_plate=use_local_h_for_flat_plate
        )

        if partial_sim_results and partial_sim_results.get('status', '').startswith('Success'):
            T_heatsink_surface_partial = partial_sim_results.get('T_solution_matrix')
            if T_heatsink_surface_partial is not None:
                ntc_i_idx, ntc_j_idx = ntc_grid_coords_map[mod_id_to_remove]
                T_hs_at_ntc_location = T_heatsink_surface_partial[ntc_i_idx, ntc_j_idx]
                delta_T = T_hs_at_ntc_location - t_ambient_inlet_h_calc
                delta_t_compensation_map[mod_id_to_remove] = delta_T
                print(f"      - T_disipador en pos. NTC de '{mod_id_to_remove}': {T_hs_at_ntc_location:.2f}°C")
                print(f"      - T_ambiente: {t_ambient_inlet_h_calc:.2f}°C")
                print(f"      - Delta_T para '{mod_id_to_remove}' = {delta_T:.2f}°C")
            else:
                print(
                    f"      - ADVERTENCIA: No se pudo obtener la matriz de T. para la simulación sin '{mod_id_to_remove}'. Delta_T será 0.")
                delta_t_compensation_map[mod_id_to_remove] = 0.0
        else:
            print(f"      - ERROR: La simulación parcial sin '{mod_id_to_remove}' falló. Delta_T será 0.")
            delta_t_compensation_map[mod_id_to_remove] = 0.0

    print("\n--- PASO 3: Aplicando compensación a los resultados de la simulación completa ---")

    max_ntc_compensated = -float('inf')
    for module_result in full_simulation_results.get('module_results', []):
        mod_id = module_result.get('id')
        delta_T_to_add = delta_t_compensation_map.get(mod_id, 0.0)
        original_t_ntc = module_result.get('t_ntc')
        if original_t_ntc is not None and not np.isnan(original_t_ntc):
            compensated_t_ntc = original_t_ntc + delta_T_to_add
            module_result['t_ntc_original'] = original_t_ntc
            module_result['delta_t_compensation'] = delta_T_to_add
            module_result['t_ntc'] = compensated_t_ntc
            print(
                f"  - Módulo '{mod_id}': T_NTC original={original_t_ntc:.2f}°C, Delta_T={delta_T_to_add:.2f}°C -> T_NTC Compensada={compensated_t_ntc:.2f}°C")
            if compensated_t_ntc > max_ntc_compensated:
                max_ntc_compensated = compensated_t_ntc
        else:
            print(f"  - Módulo '{mod_id}': T_NTC original no disponible. No se aplica compensación.")

    if max_ntc_compensated > -float('inf'):
        full_simulation_results['t_max_ntc'] = max_ntc_compensated
    full_simulation_results['compensation_method_applied'] = "N+1 Superposition"
    end_time_total = time.time()
    total_duration = end_time_total - start_time_total
    full_simulation_results['total_compensation_sim_time_s'] = total_duration

    print("\n" + "=" * 60)
    print(f"SIMULACIÓN CON COMPENSACIÓN DE NTC FINALIZADA (Duración total: {total_duration:.2f}s)")
    print("=" * 60 + "\n")

    return full_simulation_results


if __name__ == '__main__':
    print("Ejecutando prueba local de simulador_core.py con iteración de 'h' (CON WARM START)...")
    test_module_defs = [{'id': 'Mod_A', 'center_x': 0.1, 'center_y': 0.20},
                        {'id': 'Mod_B', 'center_x': 0.1, 'center_y': 0.08}]
    test_chip_powers = {'Mod_A_IGBT1': 100.0, 'Mod_A_Diode1': 25.0, 'Mod_A_IGBT2': 100.0, 'Mod_A_Diode2': 25.0,
                        'Mod_B_IGBT1': 50.0, 'Mod_B_Diode1': 50.0, 'Mod_B_IGBT2': 0.0, 'Mod_B_Diode2': 0.0}
    lx_val = 0.20;
    ly_val = 0.30;
    t_base_val = 0.010;
    k_base_material_val = K_MAIN_HEATSINK_BASE;
    rth_hs_global_val = 0.015
    test_fin_params_config_no_fins = {'h_fin': 0.0, 't_fin': 0.0, 'num_fins': 0, 'w_hollow': 0.0, 'h_hollow': 0.0,
                                      'num_hollow_per_fin': 0}
    test_fin_params_config_with_fins = {'h_fin': 0.030, 't_fin': 0.0032, 'num_fins': 25, 'w_hollow': 0.001,
                                        'h_hollow': 0.010, 'num_hollow_per_fin': 2}
    assumed_duct_h_val_flat = 0.020
    assumed_duct_h_val_finned = test_fin_params_config_with_fins['h_fin'] + 0.005
    t_amb_inlet_val = 25.0;
    q_total_m3h_val = 750.0
    nx_fdm = 30;
    ny_fdm = 45;
    nz_base_fdm_3D = 1

    print("\n--- PRUEBA PLACA PLANA CON h LOCAL (use_local_h_for_flat_plate=True) ---")
    results_flat_local_h = run_simulation_with_h_iteration(
        max_h_iterations=5, h_convergence_tolerance=0.5,
        lx_base_h_calc=lx_val, ly_base_h_calc=ly_val, q_total_m3_h_h_calc=q_total_m3h_val,
        t_ambient_inlet_h_calc=t_amb_inlet_val, assumed_duct_height_h_calc=assumed_duct_h_val_flat,
        k_heatsink_material_h_calc=k_base_material_val, fin_params_h_calc=test_fin_params_config_no_fins,
        rth_heatsink_fallback_h_calc=rth_hs_global_val,
        specific_chip_powers_sim=test_chip_powers, lx_sim=lx_val, ly_sim=ly_val, t_sim_base_fdm=t_base_val,
        module_definitions_sim=test_module_defs,
        nx_sim=nx_fdm, ny_sim=ny_fdm, nz_base_sim=nz_base_fdm_3D,
        use_local_h_for_flat_plate=True
    )
    if not results_flat_local_h:
        print("Sim placa plana (h local) no produjo resultados.")
    else:
        print("\nResultados Placa Plana con h Local:");
        print(f"Status: {results_flat_local_h.get('status', 'Error')}")
        h_val = results_flat_local_h.get('h_eff_converged_value', np.nan)
        if isinstance(h_val, np.ndarray):
            print(
                f"  h_array Conv: {results_flat_local_h.get('h_converged_status', 'N/A')}, Min={np.min(h_val):.2f}, Max={np.max(h_val):.2f} en {results_flat_local_h.get('h_iterations', 'N/A')} iter.")
        else:
            print(
                f"  h_scalar Conv: {results_flat_local_h.get('h_converged_status', 'N/A')}, Val={h_val:.2f} en {results_flat_local_h.get('h_iterations', 'N/A')} iter.")
        print(f"  T_max_base: {results_flat_local_h.get('t_max_base', 'N/A'):.2f} C")
        print(
            f"  T_max_junction: {results_flat_local_h.get('t_max_junction', 'N/A'):.2f} C ({results_flat_local_h.get('t_max_junction_chip', 'N/A')})")
        print(f"  T_max_ntc: {results_flat_local_h.get('t_max_ntc', 'N/A'):.2f} C")
        rth_calc = results_flat_local_h.get('rth_heatsink_calculated')
        if rth_calc is not None and not np.isnan(rth_calc):
            print(f"  Rth_heatsink (final): {rth_calc:.4f} K/W")

    print("\n--- PRUEBA CON ALETAS Y h EFECTIVO ESCALAR (use_local_h_for_flat_plate=False) ---")
    results_finned_scalar_h = run_simulation_with_h_iteration(
        max_h_iterations=5, h_convergence_tolerance=2.0,
        lx_base_h_calc=lx_val, ly_base_h_calc=ly_val, q_total_m3_h_h_calc=q_total_m3h_val,
        t_ambient_inlet_h_calc=t_amb_inlet_val, assumed_duct_height_h_calc=assumed_duct_h_val_finned,
        k_heatsink_material_h_calc=k_base_material_val, fin_params_h_calc=test_fin_params_config_with_fins,
        rth_heatsink_fallback_h_calc=rth_hs_global_val,
        specific_chip_powers_sim=test_chip_powers, lx_sim=lx_val, ly_sim=ly_val, t_sim_base_fdm=t_base_val,
        module_definitions_sim=test_module_defs,
        nx_sim=nx_fdm, ny_sim=ny_fdm, nz_base_sim=nz_base_fdm_3D,
        use_local_h_for_flat_plate=False
    )
    if not results_finned_scalar_h:
        print("Sim con aletas (h escalar) no produjo resultados.")
    else:
        print("\nResultados Con Aletas y h Escalar:")
        print(f"Status: {results_finned_scalar_h.get('status', 'Error')}")
        h_val_f = results_finned_scalar_h.get('h_eff_converged_value', np.nan)
        if isinstance(h_val_f, np.ndarray):
            print(
                f"  h_array Conv: {results_finned_scalar_h.get('h_converged_status', 'N/A')}, Min={np.min(h_val_f):.2f}, Max={np.max(h_val_f):.2f} en {results_finned_scalar_h.get('h_iterations', 'N/A')} iter.")
        else:
            print(
                f"  h_scalar Conv: {results_finned_scalar_h.get('h_converged_status', 'N/A')}, Val={h_val_f:.2f} en {results_finned_scalar_h.get('h_iterations', 'N/A')} iter.")
        print(f"  T_max_base: {results_finned_scalar_h.get('t_max_base', 'N/A'):.2f} C")
        print(
            f"  T_max_junction: {results_finned_scalar_h.get('t_max_junction', 'N/A'):.2f} C ({results_finned_scalar_h.get('t_max_junction_chip', 'N/A')})")
        print(f"  T_max_ntc: {results_finned_scalar_h.get('t_max_ntc', 'N/A'):.2f} C")
        rth_calc_f = results_finned_scalar_h.get('rth_heatsink_calculated')
        if rth_calc_f is not None and not np.isnan(rth_calc_f):
            print(f"  Rth_heatsink (final): {rth_calc_f:.4f} K/W")

    print("\nPrueba local finalizada.")

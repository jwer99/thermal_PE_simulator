matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu
import time
import math
import os
import io
import base64

# --- Constantes ---
# ... (sin cambios, omitido por brevedad) ...
NTC_REL_X = +0.021
NTC_REL_Y = -0.036
K_MAIN_HEATSINK_BASE = 218.0
T_ambient_inlet = 40.0
Q_total_m3_h = 785.0
cp_aire = 1005.0
rho_aire = 1.109
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
Rth_jhs_igbt = 0.0721
Rth_jhs_diode = 0.121
Nx_base, Ny_base = 250, 350
chip_rel_positions = {
"IGBT1": (-0.006, +0.023), "Diode1": (+0.012, +0.023),
"IGBT2": (+0.006, -0.023), "Diode2": (-0.012, -0.023),
}
EXPERIMENTAL_RTH_NTC_DATA = {
"IGBT1": [
([0.00, 0.00, 1.00, 0.00], 0.1041915), ([0.50, 0.00, 0.50, 0.00], 0.093628),
([0.25, 0.25, 0.25, 0.25], 0.1204493), ([0.00, 0.00, 0.50, 0.50], 0.1521716)
],
"IGBT2": [
([1.00, 0.00, 0.00, 0.00], 0.0768713), ([0.50, 0.00, 0.50, 0.00], 0.093628),
([0.25, 0.25, 0.25, 0.25], 0.1191141), ([0.50, 0.50, 0.00, 0.00], 0.1053138)
],
"Diode1": [
([0.00, 0.00, 0.00, 1.00], 0.1441531), ([0.00, 0.50, 0.00, 0.50], 0.1502391),
([0.25, 0.25, 0.25, 0.25], 0.1894391), ([0.00, 0.00, 0.50, 0.50], 0.2024668)
],
"Diode2": [
([0.00, 1.00, 0.00, 0.00], 0.1466934), ([0.00, 0.50, 0.00, 0.50], 0.1502391),
([0.25, 0.25, 0.25, 0.25], 0.1883709), ([0.50, 0.50, 0.00, 0.00], 0.1640444)
]
}
CHIP_ORDER_FOR_RTH_TABLE = ["IGBT2", "Diode2", "IGBT1", "Diode1"]


# --- FIN CONSTANTES ---

def find_closest_experimental_rth(actual_power_distribution_normalized, experimental_rth_entries):
# ... (sin cambios, omitido por brevedad)
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
distance = np.sum(np.abs(actual_power_distribution_normalized_np - exp_dist_np)) # L1 norm
if distance < min_distance:
min_distance = distance
best_rth = rth_val
closest_experimental_distribution = exp_dist
elif distance == min_distance:
pass
return best_rth, closest_experimental_distribution


def run_thermal_simulation(specific_chip_powers, lx, ly, t, rth_heatsink, module_definitions, nx=Nx_base, ny=Ny_base):
results = {
'status': 'Processing', 'convergence': False, 'iterations': 0,
't_max_base': np.nan, 't_avg_base': np.nan, 't_air_outlet': np.nan,
't_max_junction': np.nan, 't_max_junction_chip': '', 't_max_ntc': np.nan,
'module_results': [],
'plot_base_data_uri': None, 'plot_air_data_uri': None, 'plot_zoom_data_uri': None,
'error_message': None
}
start_time_sim = time.time()
print(f"[SimCore] Iniciando con {len(module_definitions)} módulos. Simulación detallada de módulo.")
# --- CAMBIO EN EL TEXTO DEL LOG ---
print(f"[SimCore] Tj IGBTs desde T_HÍBRIDA(max,avg)_hs_chip. Tj Diodos desde T_AVG_hs_chip.")
print(f"[SimCore] T_NTC será calculada desde Tj_simulada y Rth_j-NTC experimentales ponderadas.")
print(
f"[SimCore] Params: Lx={lx:.3f}, Ly={ly:.3f}, t(disipador)={t:.4f}, Rth(disipador)={rth_heatsink:.4f}, Nx={nx}, Ny={ny}")

# ... (validaciones y caso sin módulos, sin cambios, omitido por brevedad) ...
if not isinstance(specific_chip_powers, dict): results['status'] = 'Error'; results[
'error_message'] = 'Powers dict invalido.'; return results
if not all(isinstance(d, (int, float)) and d > 1e-9 for d in [lx, ly, t, rth_heatsink]): results[
'status'] = 'Error'; results['error_message'] = 'Dimensiones/Rth deben ser > 0.'; return results
if not isinstance(module_definitions, list): results['status'] = 'Error'; results[
'error_message'] = 'module_definitions debe ser lista.'; return results
if not all(isinstance(n, int) and n > 1 for n in [nx, ny]): results['status'] = 'Error'; results[
'error_message'] = 'Nx/Ny deben ser > 1.'; return results
if not module_definitions:
print("[SimCore] No se proporcionaron módulos. Simulando solo convección base del disipador (simplificado).")
# ... (código idéntico omitido) ...
try:
A_conv_disipador_no_mod = 0.7717
h_avg_disipador_no_mod = 1.0 / (
rth_heatsink * A_conv_disipador_no_mod) if rth_heatsink * A_conv_disipador_no_mod > 1e-9 else 85.0
A_placa_base_FDM_no_mod = lx * ly
h_eff_FDM_no_mod = h_avg_disipador_no_mod * (
A_conv_disipador_no_mod / A_placa_base_FDM_no_mod) if A_placa_base_FDM_no_mod > 1e-9 else 10.0
dx_no_mod = lx / (nx - 1) if nx > 1 else lx;
dy_no_mod = ly / (ny - 1) if ny > 1 else ly
N_nodes_no_mod = nx * ny;
dA_no_mod = dx_no_mod * dy_no_mod
C_x_no_mod = K_MAIN_HEATSINK_BASE * t * dy_no_mod / dx_no_mod if dx_no_mod > 1e-12 else 0
C_y_no_mod = K_MAIN_HEATSINK_BASE * t * dx_no_mod / dy_no_mod if dy_no_mod > 1e-12 else 0
A_mat_no_mod = lil_matrix((N_nodes_no_mod, N_nodes_no_mod));
b_vec_no_mod = np.zeros(N_nodes_no_mod)
for i_node in range(nx):
for j_node in range(ny):
idx = i_node * ny + j_node;
C_h_local_no_mod = h_eff_FDM_no_mod * dA_no_mod;
coef_diag_no_mod = -C_h_local_no_mod
b_vec_no_mod[idx] = - C_h_local_no_mod * T_ambient_inlet
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
results['t_air_outlet'] = T_ambient_inlet
results['convergence'] = True;
results['iterations'] = 0;
results['status'] = 'Success'
x_coords_gfx_no_mod = np.linspace(0, lx, nx);
y_coords_gfx_no_mod = np.linspace(0, ly, ny)
X_gfx_no_mod, Y_gfx_no_mod = np.meshgrid(x_coords_gfx_no_mod, y_coords_gfx_no_mod, indexing='ij')

def fig_to_data_uri_gfx_local(fig_obj_gfx):
buf_gfx = io.BytesIO();
fig_obj_gfx.savefig(buf_gfx, format='png', bbox_inches='tight', dpi=90)
buf_gfx.seek(0);
img_base64_gfx = base64.b64encode(buf_gfx.getvalue()).decode('utf-8')
plt.close(fig_obj_gfx);
return f"data:image/png;base64,{img_base64_gfx}"

fig_main_no_mod, ax_no_mod = plt.subplots(1, 1, figsize=(5.5, 4.5))
contour_base_no_mod = ax_no_mod.contourf(X_gfx_no_mod, Y_gfx_no_mod, T_sol_no_mod,
levels=np.linspace(T_ambient_inlet, np.max(T_sol_no_mod) + 1, 15),
cmap="hot", extend='max')
fig_main_no_mod.colorbar(contour_base_no_mod, ax=ax_no_mod, label="T Base (°C)")
ax_no_mod.set_title("T Base (Sin Módulos)");
ax_no_mod.set_aspect('equal')
results['plot_base_data_uri'] = fig_to_data_uri_gfx_local(fig_main_no_mod)
except Exception as e_base_no_mod:
results['status'] = 'Error';
results['error_message'] = f"Error en setup sin módulos: {e_base_no_mod}"
print(f"[SimCore] Simulación (sin módulos) completada en {time.time() - start_time_sim:.2f}s.")
return results

T_solution = np.full((nx, ny), T_ambient_inlet + 10.0)
T_air_solution = np.full((nx, ny), T_ambient_inlet)

try:
# ... (inicialización de parámetros de simulación y bucle iterativo principal, sin cambios, omitido por brevedad) ...
k_hs = K_MAIN_HEATSINK_BASE
t_hs = t
Q_total_m3_s = Q_total_m3_h / 3600.0
m_dot_total_kgs = Q_total_m3_s * rho_aire
A_conv_disipador = 0.7716
if rth_heatsink > 1e-9 and A_conv_disipador > 1e-9:
h_avg_disipador = 1.0 / (rth_heatsink * A_conv_disipador)
else:
h_avg_disipador = 85.0
A_placa_base_FDM = lx * ly
if A_placa_base_FDM > 1e-9:
h_eff_FDM_heatsink = h_avg_disipador * (A_conv_disipador / A_placa_base_FDM)
else:
h_eff_FDM_heatsink = 10.0
print(f"[SimCore] h_avg_disipador (calculado desde Rth_hs y A_conv): {h_avg_disipador:.2f} W/m^2K")
print(f"[SimCore] h_eff_FDM_heatsink (aplicado a nodos FDM): {h_eff_FDM_heatsink:.2f} W/m^2K")
dx = lx / (nx - 1) if nx > 1 else lx
dy = ly / (ny - 1) if ny > 1 else ly
dA = dx * dy
if lx > 1e-9 and nx > 0 and dx > 1e-12:
m_dot_per_meter_width = m_dot_total_kgs / lx
m_dot_column_kgs = m_dot_per_meter_width * dx
m_dot_cp_column = m_dot_column_kgs * cp_aire
else:
m_dot_cp_column = 1e-12
if m_dot_cp_column < 1e-12: m_dot_cp_column = 1e-12
N_nodes = nx * ny
P_total_fuentes_nominal = sum(
p_val for p_val in specific_chip_powers.values() if p_val is not None and p_val > 0)
modules_data_sim = []
Nx_module_footprint_global = max(1, int(round(w_igbt_footprint / dx))) if dx > 1e-12 else 1
Ny_module_footprint_global = max(1, int(round(h_igbt_footprint / dy))) if dy > 1e-12 else 1
nx_module_half_global = Nx_module_footprint_global // 2
ny_module_half_global = Ny_module_footprint_global // 2
for mod_def_idx, mod_def in enumerate(module_definitions):
module_id_base = mod_def['id']
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
'ntc_i_idx_local': None, 'ntc_j_idx_local': None,
'T_ntc_final_experimental': np.nan,
'T_module_internal_solution': np.full((nx_mod_local, ny_mod_local), T_ambient_inlet + 10.0)
}
ntc_abs_x_val = module_center_x + NTC_REL_X;
ntc_abs_y_val = module_center_y + NTC_REL_Y
module_info['ntc_abs_x'] = ntc_abs_x_val;
module_info['ntc_abs_y'] = ntc_abs_y_val
if 0 <= ntc_abs_x_val <= lx and 0 <= ntc_abs_y_val <= ly and dx > 1e-12 and dy > 1e-12:
ntc_i_idx_global_val = int(round(ntc_abs_x_val / dx));
ntc_j_idx_global_val = int(round(ntc_abs_y_val / dy))
ntc_i_idx_global_val = max(0, min(nx - 1, ntc_i_idx_global_val));
ntc_j_idx_global_val = max(0, min(ny - 1, ntc_j_idx_global_val));
module_info['ntc_i_idx_global'] = ntc_i_idx_global_val;
module_info['ntc_j_idx_global'] = ntc_j_idx_global_val
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
chip_x_idx_global_val = max(0, min(nx - 1, chip_x_idx_global_val));
chip_y_idx_global_val = max(0, min(ny - 1, chip_y_idx_global_val));
is_igbt_chip = "IGBT" in chip_label_suffix;
chip_type_val = "IGBT" if is_igbt_chip else "Diode"
if is_igbt_chip:
chip_w_m_val, chip_h_m_val, chip_A_val, chip_Rth_val = w_chip_igbt_m, h_chip_igbt_m, A_chip_igbt, Rth_jhs_igbt
else:
chip_w_m_val, chip_h_m_val, chip_A_val, chip_Rth_val = w_chip_diode_m, h_chip_diode_m, A_chip_diode, Rth_jhs_diode
chip_Nx_cells_global_val = max(1, int(round(chip_w_m_val / dx))) if dx > 1e-12 else 1
chip_Ny_cells_global_val = max(1, int(round(chip_h_m_val / dy))) if dy > 1e-12 else 1
chip_nx_half_global_val, chip_ny_half_global_val = chip_Nx_cells_global_val // 2, chip_Ny_cells_global_val // 2
chip_Nx_cells_local_val = max(1, int(round(chip_w_m_val / dx))) if dx > 1e-12 else 1
chip_Ny_cells_local_val = max(1, int(round(chip_h_m_val / dy))) if dy > 1e-12 else 1
chip_nx_half_local_val, chip_ny_half_local_val = chip_Nx_cells_local_val // 2, chip_Ny_cells_local_val // 2
chip_power_val = specific_chip_powers.get(chip_full_id, 0.0)
chip_q_source_val = chip_power_val / chip_A_val if chip_A_val > 1e-12 else 0.0
chip_x_idx_local_val, chip_y_idx_local_val = -1, -1
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
'power': chip_power_val, 'q_source_W_per_m2': chip_q_source_val,
'Rth_jhs': chip_Rth_val,
'T_base_chip_on_heatsink': np.nan, 'T_base_chip_on_module_surface': np.nan,
'Tj_from_heatsink': np.nan, 'Tj_from_module_surface': np.nan,
'Tj': np.nan,
'w_m': chip_w_m_val, 'h_m': chip_h_m_val, 'Area_m2': chip_A_val}
module_info['chips'].append(chip_data)
modules_data_sim.append(module_info)
C_x_cond_hs = k_hs * t_hs * dy / dx if dx > 1e-12 else 0
C_y_cond_hs = k_hs * t_hs * dx / dy if dy > 1e-12 else 0
A_matrix_heatsink = lil_matrix((N_nodes, N_nodes))
for i_hs_node in range(nx):
for j_hs_node in range(ny):
idx_hs = i_hs_node * ny + j_hs_node;
C_h_local_hs = h_eff_FDM_heatsink * dA;
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
return results
max_iterations = 100;
convergence_tolerance = 0.01;
iteration = 0;
converged = False
print("[SimCore] Iniciando bucle iterativo Disipador Principal <-> Módulos IGBT...")
T_solution_old_iter = T_solution.copy()
while not converged and iteration < max_iterations:
iteration += 1
q_source_map_for_heatsink = np.zeros((nx, ny))
for module_item in modules_data_sim:
nx_mod = module_item['nx_mod_local'];
ny_mod = module_item['ny_mod_local']
N_nodes_mod = nx_mod * ny_mod
if N_nodes_mod == 0: continue
q_chip_source_map_module = np.zeros((nx_mod, ny_mod))
for chip_item in module_item['chips']:
if chip_item['power'] <= 0 or chip_item['center_x_idx_local'] < 0: continue
i_min_chip_loc = max(0, chip_item['center_x_idx_local'] - chip_item['nx_half_local'])
i_max_chip_loc = min(nx_mod - 1, chip_item['center_x_idx_local'] + chip_item['nx_half_local'])
j_min_chip_loc = max(0, chip_item['center_y_idx_local'] - chip_item['ny_half_local'])
j_max_chip_loc = min(ny_mod - 1, chip_item['center_y_idx_local'] + chip_item['ny_half_local'])
num_nodes_chip_on_module = (i_max_chip_loc - i_min_chip_loc + 1) * \
(j_max_chip_loc - j_min_chip_loc + 1)
if num_nodes_chip_on_module > 0:
power_per_node_chip = chip_item['power'] / num_nodes_chip_on_module
q_density_on_node_chip = power_per_node_chip / dA
q_chip_source_map_module[i_min_chip_loc:i_max_chip_loc + 1,
j_min_chip_loc:j_max_chip_loc + 1] += q_density_on_node_chip
A_matrix_module = lil_matrix((N_nodes_mod, N_nodes_mod));
b_vector_module = np.zeros(N_nodes_mod)
C_x_cond_mod = K_MODULE_BASEPLATE * T_MODULE_BASEPLATE * dy / dx if dx > 1e-12 else 0
C_y_cond_mod = K_MODULE_BASEPLATE * T_MODULE_BASEPLATE * dx / dy if dy > 1e-12 else 0
for i_mod_node in range(nx_mod):
for j_mod_node in range(ny_mod):
idx_mod = i_mod_node * ny_mod + j_mod_node
global_i = module_item['footprint_i_min_global'] + i_mod_node;
global_j = module_item['footprint_j_min_global'] + j_mod_node
T_heatsink_under_module_node = T_solution[global_i, global_j]
C_h_interface_mod = H_MODULE_TO_HEATSINK_INTERFACE * dA;
coef_diag_mod = -C_h_interface_mod
q_chip_W_node = q_chip_source_map_module[i_mod_node, j_mod_node] * dA
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
LU_mod = splu(A_matrix_module.tocsr());
T_flat_mod = LU_mod.solve(b_vector_module)
module_item['T_module_internal_solution'] = T_flat_mod.reshape((nx_mod, ny_mod))
except (RuntimeError, ValueError) as e_solve_mod:
results['status'] = 'Error';
results[
'error_message'] = f"Error solver módulo {module_item['id']} iter {iteration}: {e_solve_mod}";
return results
for i_mod_node in range(nx_mod):
for j_mod_node in range(ny_mod):
global_i = module_item['footprint_i_min_global'] + i_mod_node;
global_j = module_item['footprint_j_min_global'] + j_mod_node
T_mod_base_node = module_item['T_module_internal_solution'][i_mod_node, j_mod_node]
T_heatsink_node = T_solution[global_i, global_j]
P_node_mod_to_hs = H_MODULE_TO_HEATSINK_INTERFACE * dA * (T_mod_base_node - T_heatsink_node)
q_source_map_for_heatsink[global_i, global_j] += P_node_mod_to_hs / dA
b_vector_heatsink = np.zeros(N_nodes)
for i_hs_node in range(nx):
for j_hs_node in range(ny):
idx_hs = i_hs_node * ny + j_hs_node
q_source_from_modules_W_node = q_source_map_for_heatsink[i_hs_node, j_hs_node] * dA
T_air_local_hs = T_air_solution[i_hs_node, j_hs_node]
b_vector_heatsink[idx_hs] = -q_source_from_modules_W_node - h_eff_FDM_heatsink * dA * T_air_local_hs
try:
T_flat_heatsink = LU_heatsink.solve(b_vector_heatsink);
T_solution_new = T_flat_heatsink.reshape((nx, ny))
except (RuntimeError, ValueError) as e_solve_hs:
results['status'] = 'Error';
results['error_message'] = f"Error solver disipador iter {iteration}: {e_solve_hs}";
return results
T_air_next_iter = np.full_like(T_air_solution, T_ambient_inlet);
delta_T_air_nodes_iter = np.zeros_like(T_solution)
for i_node_tair in range(nx):
for j_node_tair in range(ny):
P_conv_iter = h_eff_FDM_heatsink * dA * \
(T_solution_new[i_node_tair, j_node_tair] - T_air_solution[i_node_tair, j_node_tair])
P_conv_nodes_iter = max(0, P_conv_iter)
if m_dot_cp_column > 1e-12:
delta_T_air_nodes_iter[i_node_tair, j_node_tair] = P_conv_nodes_iter / m_dot_cp_column
else:
delta_T_air_nodes_iter[i_node_tair, j_node_tair] = 0
for i_node_acc in range(nx):
for j_node_acc in range(1, ny):
T_air_next_iter[i_node_acc, j_node_acc] = T_air_next_iter[i_node_acc, j_node_acc - 1] + \
delta_T_air_nodes_iter[i_node_acc, j_node_acc - 1]
if iteration > 1:
max_diff_T_heatsink = np.max(np.abs(T_solution_new - T_solution_old_iter))
converged = max_diff_T_heatsink < convergence_tolerance
T_solution_old_iter = T_solution_new.copy()
T_solution = T_solution_new.copy();
T_air_solution = T_air_next_iter.copy()
if iteration % 10 == 0 or iteration == 1 or (converged and iteration > 1):
max_T_sol_iter = np.max(T_solution) if not np.isnan(T_solution).all() else np.nan
diff_str = f"{max_diff_T_heatsink:.4f}°C" if iteration > 1 else "N/A (1st iter)"
print(
f"[SimCore] Iter {iteration}: Max ΔT_disipador = {diff_str}. T_max_disipador = {max_T_sol_iter:.2f}°C")

print(f"[SimCore] Bucle terminado en {iteration} iteraciones. Converged: {converged}")
results['convergence'] = converged;
results['iterations'] = iteration
# ... (asignación de resultados t_max_base, t_avg_base, t_air_outlet, sin cambios) ...
if np.isnan(T_solution).any():
results['t_max_base'] = np.nan;
results['t_avg_base'] = np.nan
print("[SimCore] ADVERTENCIA: NaN encontrado en la solución del disipador.")
else:
results['t_max_base'] = np.max(T_solution);
results['t_avg_base'] = np.mean(T_solution)
if ny > 0 and not np.isnan(T_air_solution[:, ny - 1]).any():
results['t_air_outlet'] = np.mean(T_air_solution[:, ny - 1])
else:
results['t_air_outlet'] = np.nan

max_tj_overall = -float('inf');
max_tj_chip_label = "";
max_t_ntc_overall = -float('inf');
max_t_module_surface_overall = -float('inf')

# --- FUNCIÓN PARA T_HÍBRIDA (IGBTs) ---
def get_hybrid_temp_in_area_local(T_matrix_func, center_x_idx_func, center_y_idx_func,
nx_half_func, ny_half_func, Max_Nx_func, Max_Ny_func):
# ... (código idéntico al de la respuesta anterior, omitido por brevedad) ...
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
t_max_area = np.nanmax(area_nodes_func)
t_avg_area = np.nanmean(area_nodes_func)
if np.isnan(t_max_area) or np.isnan(t_avg_area):
return np.nan
return (t_max_area + t_avg_area) / 2.0

# --- FUNCIÓN PARA T_AVG (Diodos) ---
def get_avg_temp_in_area_local(T_matrix_func, center_x_idx_func, center_y_idx_func,
nx_half_func, ny_half_func, Max_Nx_func, Max_Ny_func):
# ... (código idéntico a la versión original de get_avg_temp_local, omitido por brevedad) ...
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
module_result = {'id': module_item_post['id'], 'chips': [], 't_ntc': np.nan}
T_module_internal_map = module_item_post['T_module_internal_solution']
if not np.isnan(T_module_internal_map).all():
max_t_module_surface_overall = np.nanmax(
[max_t_module_surface_overall, np.nanmax(T_module_internal_map)])

# ... (cálculo de distribución de potencia normalizada, sin cambios) ...
module_chip_powers_ordered = {}
for chip_item_post_pwr in module_item_post['chips']:
module_chip_powers_ordered[chip_item_post_pwr['suffix']] = chip_item_post_pwr['power']
actual_power_values = [module_chip_powers_ordered.get(s, 0.0) for s in CHIP_ORDER_FOR_RTH_TABLE]
total_module_power = sum(actual_power_values)
actual_power_distribution_normalized = [0.0] * len(CHIP_ORDER_FOR_RTH_TABLE)
if total_module_power > 1e-6:
actual_power_distribution_normalized = [p / total_module_power for p in actual_power_values]
print(f"Module ID: {module_item_post['id']}")
print(f" Actual Powers (Order: {CHIP_ORDER_FOR_RTH_TABLE}): {actual_power_values}")
print(f" Total Module Power: {total_module_power:.2f} W")
print(f" Normalized Power Distribution: {[f'{p:.2f}' for p in actual_power_distribution_normalized]}")
all_ntc_temps_for_module = []

for chip_item_post in module_item_post['chips']:
is_igbt_chip_type = chip_item_post['type'] == "IGBT"

# --- SELECCIÓN DE FUNCIÓN BASADA EN TIPO DE CHIP ---
if is_igbt_chip_type:
T_base_chip_on_heatsink_val = get_hybrid_temp_in_area_local(
T_solution, chip_item_post['center_x_idx_global'], chip_item_post['center_y_idx_global'],
chip_item_post['nx_half_global'], chip_item_post['ny_half_global'], nx, ny)

if chip_item_post['center_x_idx_local'] >= 0: # Solo si el chip está en la huella del módulo
T_base_chip_on_module_surface_val = get_hybrid_temp_in_area_local(
T_module_internal_map, chip_item_post['center_x_idx_local'],
chip_item_post['center_y_idx_local'],
chip_item_post['nx_half_local'], chip_item_post['ny_half_local'],
module_item_post['nx_mod_local'], module_item_post['ny_mod_local'])
else:
T_base_chip_on_module_surface_val = np.nan
else: # Es un Diodo
T_base_chip_on_heatsink_val = get_avg_temp_in_area_local(
T_solution, chip_item_post['center_x_idx_global'], chip_item_post['center_y_idx_global'],
chip_item_post['nx_half_global'], chip_item_post['ny_half_global'], nx, ny)

if chip_item_post['center_x_idx_local'] >= 0:
T_base_chip_on_module_surface_val = get_avg_temp_in_area_local(
T_module_internal_map, chip_item_post['center_x_idx_local'],
chip_item_post['center_y_idx_local'],
chip_item_post['nx_half_local'], chip_item_post['ny_half_local'],
module_item_post['nx_mod_local'], module_item_post['ny_mod_local'])
else:
T_base_chip_on_module_surface_val = np.nan

chip_item_post['T_base_chip_on_heatsink'] = T_base_chip_on_heatsink_val
chip_item_post[
'T_base_chip_on_module_surface'] = T_base_chip_on_module_surface_val # Guardamos la Tbase del módulo también

# ... (cálculo de Tj_from_heatsink y Tj_from_module_surface, sin cambios directos) ...
Tj_from_heatsink_val = np.nan
if not np.isnan(T_base_chip_on_heatsink_val):
if chip_item_post['power'] > 1e-6 and not np.isnan(chip_item_post['Rth_jhs']):
Tj_from_heatsink_val = T_base_chip_on_heatsink_val + chip_item_post['Rth_jhs'] * chip_item_post[
'power']
else:
Tj_from_heatsink_val = T_base_chip_on_heatsink_val
chip_item_post['Tj_from_heatsink'] = Tj_from_heatsink_val

Tj_from_module_surface_val = np.nan
if not np.isnan(T_base_chip_on_module_surface_val): # Usamos la Tbase del módulo calculada arriba
if chip_item_post['power'] > 1e-6 and not np.isnan(chip_item_post['Rth_jhs']):
Tj_from_module_surface_val = T_base_chip_on_module_surface_val + chip_item_post['Rth_jhs'] * \
chip_item_post['power']
else:
Tj_from_module_surface_val = T_base_chip_on_module_surface_val
chip_item_post['Tj_from_module_surface'] = Tj_from_module_surface_val
chip_item_post['Tj'] = Tj_from_heatsink_val

# ... (resto del post-procesamiento del chip: NTC, etc., sin cambios) ...
if not np.isnan(chip_item_post['Tj']):
if chip_item_post['Tj'] > max_tj_overall:
max_tj_overall = chip_item_post['Tj'];
max_tj_chip_label = f"{chip_item_post['label']} (P={chip_item_post['power']:.1f}W)"
chip_suffix = chip_item_post['suffix']
chip_power_actual = chip_item_post['power']
Tj_chip_simulated = chip_item_post['Tj']
if chip_suffix in EXPERIMENTAL_RTH_NTC_DATA and chip_power_actual > 1e-6 and not np.isnan(
Tj_chip_simulated):
experimental_data_for_source_chip = EXPERIMENTAL_RTH_NTC_DATA[chip_suffix]
selected_rth_j_ntc, closest_exp_dist = find_closest_experimental_rth(
actual_power_distribution_normalized, experimental_data_for_source_chip)
if not np.isnan(selected_rth_j_ntc):
T_ntc_estimate_from_chip = Tj_chip_simulated - chip_power_actual * selected_rth_j_ntc
all_ntc_temps_for_module.append(T_ntc_estimate_from_chip)
print(f" Chip {chip_suffix} (Tj={Tj_chip_simulated:.2f}°C, P={chip_power_actual:.1f}W):")
print(
f" Selected Rth_{chip_suffix}-NTC = {selected_rth_j_ntc:.5f} K/W (for exp. dist: {[f'{p:.2f}' for p in closest_exp_dist]})")
print(f" Estimated T_NTC from this chip = {T_ntc_estimate_from_chip:.2f}°C")
else:
print(f" Chip {chip_suffix}: No suitable Rth_j-NTC found or P=0/Tj=NaN.")
else:
if chip_power_actual <= 1e-6:
print(f" Chip {chip_suffix}: P={chip_power_actual:.1f}W <= 0. Skipping NTC contribution.")
elif np.isnan(Tj_chip_simulated):
print(f" Chip {chip_suffix}: Tj is NaN. Skipping NTC contribution.")
else:
print(
f" Chip {chip_suffix}: No experimental Rth_j-NTC data defined. Skipping NTC contribution.")
module_result['chips'].append({
'suffix': chip_item_post['suffix'],
't_base_heatsink': T_base_chip_on_heatsink_val, # Esta es la T_base específica del tipo
't_base_module_surface': T_base_chip_on_module_surface_val, # Esta es la T_base específica del tipo
'tj_from_heatsink': Tj_from_heatsink_val,
'tj_from_module_surface': Tj_from_module_surface_val,
'tj': chip_item_post['Tj']
})

# ... (cálculo de T_ntc_final_experimental_module y asignación a resultados, sin cambios) ...
T_ntc_final_experimental_module = np.nan
if all_ntc_temps_for_module:
T_ntc_final_experimental_module = np.mean(all_ntc_temps_for_module)
min_T_ntc_est = np.min(all_ntc_temps_for_module)
max_T_ntc_est = np.max(all_ntc_temps_for_module)
range_T_ntc_est = max_T_ntc_est - min_T_ntc_est
print(
f" -> Module {module_item_post['id']} T_NTC Estimates: Min={min_T_ntc_est:.2f}, Max={max_T_ntc_est:.2f}, Range={range_T_ntc_est:.2f}°C")
print(
f" ==> Module {module_item_post['id']} Final Avg T_NTC (Experimental) = {T_ntc_final_experimental_module:.2f}°C")
else:
print(f" -> Module {module_item_post['id']}: No T_NTC estimates generated. Final T_NTC is NaN.")
module_item_post['T_ntc_final_experimental'] = T_ntc_final_experimental_module
module_result['t_ntc'] = T_ntc_final_experimental_module
if not np.isnan(T_ntc_final_experimental_module):
max_t_ntc_overall = np.nanmax([max_t_ntc_overall, T_ntc_final_experimental_module])
print("-" * 30)
module_results_list.append(module_result)

print("--- [End NTC Calculation Details] ---\n")
results['t_max_junction'] = max_tj_overall if not np.isinf(max_tj_overall) else np.nan
results['t_max_junction_chip'] = max_tj_chip_label
results['t_max_ntc'] = max_t_ntc_overall if not np.isinf(max_t_ntc_overall) else np.nan
results['module_results'] = module_results_list
if np.isinf(max_t_module_surface_overall): max_t_module_surface_overall = T_ambient_inlet + 10

print("[SimCore] Generando gráficos...")
# ... (sección de gráficos, solo cambios en leyendas si es necesario) ...
# ... (omitido por brevedad, es idéntico al de la respuesta anterior, excepto por la actualización de leyendas) ...
try:
x_coords_gfx_main = np.linspace(0, lx, nx);
y_coords_gfx_main = np.linspace(0, ly, ny)
X_gfx_main, Y_gfx_main = np.meshgrid(x_coords_gfx_main, y_coords_gfx_main, indexing='ij')

def fig_to_data_uri_gfx(fig_obj_gfx):
buf_gfx = io.BytesIO();
fig_obj_gfx.savefig(buf_gfx, format='png', bbox_inches='tight', dpi=90)
buf_gfx.seek(0);
img_base64_gfx = base64.b64encode(buf_gfx.getvalue()).decode('utf-8')
plt.close(fig_obj_gfx);
return f"data:image/png;base64,{img_base64_gfx}"

fig_main_gfx, axes_main_gfx = plt.subplots(1, 2, figsize=(11, 4.5))
ax_base_gfx = axes_main_gfx[0]
min_T_plot_main_gfx = T_ambient_inlet
max_T_plot_main_val_gfx = results['t_max_base'] if not np.isnan(
results['t_max_base']) else T_ambient_inlet + 1
max_T_plot_main_val_gfx = max(max_T_plot_main_val_gfx, T_ambient_inlet + 1.0);
max_T_plot_main_val_gfx += (max_T_plot_main_val_gfx - min_T_plot_main_gfx) * 0.05
levels_main_plot_gfx = np.linspace(min_T_plot_main_gfx, max_T_plot_main_val_gfx, 30) if abs(
max_T_plot_main_val_gfx - min_T_plot_main_gfx) > 1e-3 else np.linspace(min_T_plot_main_gfx - 0.5,
max_T_plot_main_val_gfx + 0.5, 2)
T_solution_plot_gfx_main = np.nan_to_num(T_solution, nan=min_T_plot_main_gfx - 10)
contour_base_main_gfx = ax_base_gfx.contourf(X_gfx_main, Y_gfx_main, T_solution_plot_gfx_main,
levels=levels_main_plot_gfx, cmap="hot", extend='max')
fig_main_gfx.colorbar(contour_base_main_gfx, ax=ax_base_gfx, label="T Disipador (°C)")
tmax_str_main_gfx = f"{results['t_max_base']:.1f}°C" if not np.isnan(results['t_max_base']) else "N/A"
ax_base_gfx.set_title(f"T Disipador Principal (Max={tmax_str_main_gfx})");
ax_base_gfx.set_aspect('equal')
ax_air_main_gfx = axes_main_gfx[1]
T_air_max_plot_main_gfx = np.nanmax(T_air_solution) if not np.isnan(
T_air_solution).all() else T_ambient_inlet
min_T_air_plot_main_gfx = T_ambient_inlet;
max_T_air_plot_main_val_gfx = T_air_max_plot_main_gfx
max_T_air_plot_main_val_gfx = max(max_T_air_plot_main_val_gfx, min_T_air_plot_main_gfx + 0.1);
max_T_air_plot_main_val_gfx += (max_T_air_plot_main_val_gfx - min_T_air_plot_main_gfx) * 0.05
levels_air_main_gfx = np.linspace(min_T_air_plot_main_gfx, max_T_air_plot_main_val_gfx, 30) if abs(
max_T_air_plot_main_val_gfx - min_T_air_plot_main_gfx) > 1e-3 else np.linspace(
min_T_air_plot_main_gfx - 0.5, max_T_air_plot_main_val_gfx + 0.5, 2)
T_air_solution_plot_gfx_main = np.nan_to_num(T_air_solution, nan=min_T_air_plot_main_gfx - 10)
contour_air_main_gfx = ax_air_main_gfx.contourf(X_gfx_main, Y_gfx_main, T_air_solution_plot_gfx_main,
levels=levels_air_main_gfx, cmap="coolwarm", extend='max')
fig_main_gfx.colorbar(contour_air_main_gfx, ax=ax_air_main_gfx, label="T Aire (°C)")
tairout_str_main_gfx = f"{results['t_air_outlet']:.1f}°C" if not np.isnan(
results['t_air_outlet']) else "N/A"
ax_air_main_gfx.set_title(f"T Aire (Salida Prom={tairout_str_main_gfx})");
ax_air_main_gfx.set_aspect('equal')
for ax_item_main_gfx in axes_main_gfx:
for module_item_main_gfx in modules_data_sim:
center_x_mod_main_gfx, center_y_mod_main_gfx = module_item_main_gfx['center_x_m'], \
module_item_main_gfx['center_y_m']
rect_x_mod_main_gfx, rect_y_mod_main_gfx = center_x_mod_main_gfx - w_igbt_footprint / 2, center_y_mod_main_gfx - h_igbt_footprint / 2
rect_mod_main_plot_gfx = plt.Rectangle((rect_x_mod_main_gfx, rect_y_mod_main_gfx), w_igbt_footprint,
h_igbt_footprint,
edgecolor='gray', facecolor='none', lw=0.8, ls='--');
ax_item_main_gfx.add_patch(rect_mod_main_plot_gfx)
if module_item_main_gfx['ntc_abs_x'] is not None and module_item_main_gfx[
'ntc_abs_y'] is not None and \
0 <= module_item_main_gfx['ntc_abs_x'] <= lx and 0 <= module_item_main_gfx[
'ntc_abs_y'] <= ly:
ax_item_main_gfx.plot(module_item_main_gfx['ntc_abs_x'], module_item_main_gfx['ntc_abs_y'],
'wo', markersize=3, markeredgecolor='black')
# --- CAMBIO EN EL TÍTULO DEL GRÁFICO PRINCIPAL ---
fig_main_gfx.suptitle(
f"Simulación ({len(modules_data_sim)} Módulos, P={P_total_fuentes_nominal:.0f}W, Rth_hs={rth_heatsink:.3f}) - Tj(IGBT:híbrido,Diode:avg), T_NTC(exp.)",
fontsize=9) # Ajustar tamaño si es muy largo
fig_main_gfx.tight_layout(rect=[0, 0.03, 1, 0.93]);
results['plot_base_data_uri'] = fig_to_data_uri_gfx(fig_main_gfx)
except Exception as e_gfx_main:
print(f"[SimCore] Error GFX Main: {e_gfx_main}")
results['plot_base_data_uri'] = None
try:
num_modules_plot_zoom = len(modules_data_sim)
if num_modules_plot_zoom == 0: raise ValueError("No modules to plot zoom for.")
ncols_zoom_gfx = min(3, num_modules_plot_zoom);
nrows_zoom_gfx = math.ceil(num_modules_plot_zoom / ncols_zoom_gfx)
fig_zoom_gfx, axes_zoom_list_gfx = plt.subplots(nrows=nrows_zoom_gfx, ncols=ncols_zoom_gfx,
figsize=(ncols_zoom_gfx * 3.8, nrows_zoom_gfx * 3.5),
squeeze=False)
axes_flat_zoom_gfx = axes_zoom_list_gfx.flatten();
contour_zoom_ref_plot_gfx = None
min_T_plot_zoom_mod_gfx = T_ambient_inlet
max_T_plot_zoom_mod_val_gfx = max(max_t_module_surface_overall,
results.get('t_max_junction', T_ambient_inlet + 10))
if np.isnan(max_T_plot_zoom_mod_val_gfx) or np.isinf(max_T_plot_zoom_mod_val_gfx):
max_T_plot_zoom_mod_val_gfx = T_ambient_inlet + 20
max_T_plot_zoom_mod_val_gfx = max(max_T_plot_zoom_mod_val_gfx, min_T_plot_zoom_mod_gfx + 1.0)
levels_zoom_plot_mod_gfx = np.linspace(min_T_plot_zoom_mod_gfx, max_T_plot_zoom_mod_val_gfx, 30) if abs(
max_T_plot_zoom_mod_val_gfx - min_T_plot_zoom_mod_gfx) > 1e-3 else np.linspace(
min_T_plot_zoom_mod_gfx - 0.5, max_T_plot_zoom_mod_val_gfx + 0.5, 2)
for idx_zoom_gfx, module_item_zoom_gfx in enumerate(modules_data_sim):
if idx_zoom_gfx >= len(axes_flat_zoom_gfx): break
ax_z_plot_gfx = axes_flat_zoom_gfx[idx_zoom_gfx]
T_module_internal_plot = module_item_zoom_gfx['T_module_internal_solution']
nx_mod_loc_plot = module_item_zoom_gfx['nx_mod_local'];
ny_mod_loc_plot = module_item_zoom_gfx['ny_mod_local']
mod_origin_x_global = module_item_zoom_gfx['center_x_m'] - w_igbt_footprint / 2
mod_origin_y_global = module_item_zoom_gfx['center_y_m'] - h_igbt_footprint / 2
x_coords_mod_local_gfx = np.linspace(mod_origin_x_global, mod_origin_x_global + w_igbt_footprint,
nx_mod_loc_plot)
y_coords_mod_local_gfx = np.linspace(mod_origin_y_global, mod_origin_y_global + h_igbt_footprint,
ny_mod_loc_plot)
X_mod_local_gfx, Y_mod_local_gfx = np.meshgrid(x_coords_mod_local_gfx, y_coords_mod_local_gfx,
indexing='ij')
T_module_plot_safe = np.nan_to_num(T_module_internal_plot, nan=min_T_plot_zoom_mod_gfx - 10)
contour_zoom_gfx_detail = ax_z_plot_gfx.contourf(X_mod_local_gfx, Y_mod_local_gfx, T_module_plot_safe,
levels=levels_zoom_plot_mod_gfx, cmap="hot",
extend='max')
if idx_zoom_gfx == 0: contour_zoom_ref_plot_gfx = contour_zoom_gfx_detail
center_x_mod_zoom_gfx, center_y_mod_zoom_gfx = module_item_zoom_gfx['center_x_m'], module_item_zoom_gfx[
'center_y_m']
rect_x_mod_zoom_gfx, rect_y_mod_zoom_gfx = center_x_mod_zoom_gfx - w_igbt_footprint / 2, center_y_mod_zoom_gfx - h_igbt_footprint / 2
rect_mod_zoom_plot_gfx_obj = plt.Rectangle((rect_x_mod_zoom_gfx, rect_y_mod_zoom_gfx), w_igbt_footprint,
h_igbt_footprint,
edgecolor='black', facecolor='none', lw=1.0, ls='--');
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
facecolor='none', lw=1.0, ls=':');
ax_z_plot_gfx.add_patch(rect_chip_zoom_plot_gfx_obj)
tj_val = chip_plot_info['tj']
tbase_hs_val = chip_plot_info['t_base_heatsink']
tj_str_zoom_gfx = f"Tj={tj_val:.1f}" if not np.isnan(tj_val) else "Tj=N/A"
# --- CAMBIO EN LA LEYENDA DEL GRÁFICO ---
tbase_label = "Tb_hs_híbrida" if is_igbt_zoom_plot_gfx else "Tb_hs_avg"
tbase_str_zoom_gfx = f"{tbase_label}={tbase_hs_val:.1f}" if not np.isnan(
tbase_hs_val) else f"{tbase_label}=N/A"
ax_z_plot_gfx.text(center_x_phys_zoom_gfx, center_y_phys_zoom_gfx,
f"{original_chip_item['suffix']}\n{tj_str_zoom_gfx}\n({tbase_str_zoom_gfx})",
color='white', ha='center', va='center', fontsize=5.5,
bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.6))
if module_item_zoom_gfx['ntc_abs_x'] is not None and module_item_zoom_gfx['ntc_abs_y'] is not None:
ntc_x_plot_zoom_gfx, ntc_y_plot_zoom_gfx = module_item_zoom_gfx['ntc_abs_x'], module_item_zoom_gfx[
'ntc_abs_y']
margin_factor_zoom_gfx = 0.3
zoom_x_min_plot_gfx = module_item_zoom_gfx['center_x_m'] - w_igbt_footprint / 2 * (
1 + margin_factor_zoom_gfx);
zoom_x_max_plot_gfx = module_item_zoom_gfx['center_x_m'] + w_igbt_footprint / 2 * (
1 + margin_factor_zoom_gfx)
zoom_y_min_plot_gfx = module_item_zoom_gfx['center_y_m'] - h_igbt_footprint / 2 * (
1 + margin_factor_zoom_gfx);
zoom_y_max_plot_gfx = module_item_zoom_gfx['center_y_m'] + h_igbt_footprint / 2 * (
1 + margin_factor_zoom_gfx)
if zoom_x_min_plot_gfx <= ntc_x_plot_zoom_gfx <= zoom_x_max_plot_gfx and \
zoom_y_min_plot_gfx <= ntc_y_plot_zoom_gfx <= zoom_y_max_plot_gfx:
ax_z_plot_gfx.plot(ntc_x_plot_zoom_gfx, ntc_y_plot_zoom_gfx, 'ro', markersize=4,
markeredgecolor='white')
t_ntc_val_plot_zoom_gfx = module_item_zoom_gfx['T_ntc_final_experimental']
t_ntc_str_plot_zoom_gfx = f"NTC≈{t_ntc_val_plot_zoom_gfx:.1f}" if not np.isnan(
t_ntc_val_plot_zoom_gfx) else "NTC=N/A"
ax_z_plot_gfx.text(ntc_x_plot_zoom_gfx + (zoom_x_max_plot_gfx - zoom_x_min_plot_gfx) * 0.03,
ntc_y_plot_zoom_gfx, t_ntc_str_plot_zoom_gfx, color='red', ha='left',
va='center', fontsize=6,
bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.7))
ax_z_plot_gfx.set_xlim(zoom_x_min_plot_gfx, zoom_x_max_plot_gfx);
ax_z_plot_gfx.set_ylim(zoom_y_min_plot_gfx, zoom_y_max_plot_gfx)
ax_z_plot_gfx.set_title(f"{module_item_zoom_gfx['id']} (T Módulo Sup.)", fontsize=8)
ax_z_plot_gfx.set_xlabel("x (m)", fontsize=7);
ax_z_plot_gfx.set_ylabel("y (m)", fontsize=7)
ax_z_plot_gfx.tick_params(axis='both', which='major', labelsize=6);
ax_z_plot_gfx.set_aspect('equal', adjustable='box');
ax_z_plot_gfx.grid(True, linestyle=':', alpha=0.3)
for i_ax_flat_zoom_gfx in range(idx_zoom_gfx + 1, len(axes_flat_zoom_gfx)):
axes_flat_zoom_gfx[i_ax_flat_zoom_gfx].axis('off')
if contour_zoom_ref_plot_gfx is not None and num_modules_plot_zoom > 0:
fig_zoom_gfx.subplots_adjust(right=0.85, top=0.90, bottom=0.1);
cbar_ax_zoom_gfx = fig_zoom_gfx.add_axes([0.88, 0.15, 0.03, 0.7])
cbar_zoom_gfx = fig_zoom_gfx.colorbar(contour_zoom_ref_plot_gfx, cax=cbar_ax_zoom_gfx,
label="T Módulo Sup. (°C)");
cbar_zoom_gfx.ax.tick_params(labelsize=7);
cbar_zoom_gfx.set_label("T Módulo Sup. (°C)", size=8)
fig_zoom_gfx.suptitle(f"Detalle Módulos (Temperatura Superficie Módulo)", fontsize=10)
results['plot_zoom_data_uri'] = fig_to_data_uri_gfx(fig_zoom_gfx)
except Exception as e_gfx_zoom:
print(f"[SimCore] Error GFX Zoom: {e_gfx_zoom}")
results['plot_zoom_data_uri'] = None

results['status'] = 'Success'
print("[SimCore] Generación de gráficos completada.")

except Exception as e_general:
import traceback
print(f"[SimCore] Error general en simulación: {e_general}");
traceback.print_exc()
results['status'] = 'Error'
results['error_message'] = f"Error general: {str(e_general)}"

print(f"[SimCore] Simulación completada en {time.time() - start_time_sim:.2f}s. Estado: {results['status']}")
return results


# --- Bloque para pruebas locales ---
if __name__ == '__main__':
print("Ejecutando prueba local de simulador_core.py (Tj IGBT:híbrida, Tj Diode:avg, T_NTC EXPERIMENTAL)...")
test_module_defs = [
{'id': 'Mod_A', 'center_x': 0.1, 'center_y': 0.20},
{'id': 'Mod_B', 'center_x': 0.1, 'center_y': 0.08},
]
test_chip_powers_dist = {
'Mod_A_IGBT1': 100.0, 'Mod_A_Diode1': 25.0, 'Mod_A_IGBT2': 100.0, 'Mod_A_Diode2': 25.0,
'Mod_B_IGBT1': 50.0, 'Mod_B_Diode1': 50.0, 'Mod_B_IGBT2': 0.0, 'Mod_B_Diode2': 0.0,
}
test_chip_powers = test_chip_powers_dist
test_lx = 0.2;
test_ly = 0.3
test_t_heatsink = 0.005;
test_rth_heatsink = 0.020
test_nx = 60; # Para pruebas rápidas
test_ny = 90

print(f"\n--- PARÁMETROS DE PRUEBA (Tj IGBT:híbrida, Tj Diode:avg, T_NTC EXPERIMENTAL) ---")
# ... (impresión de parámetros igual, omitido por brevedad) ...
print(f"Lx={test_lx}m, Ly={test_ly}m, t_hs={test_t_heatsink}m")
print(f"Rth_heatsink (global) = {test_rth_heatsink} K/W")
print(f"Nx={test_nx}, Ny={test_ny}")
print(f"K_MAIN_HEATSINK_BASE = {K_MAIN_HEATSINK_BASE} W/mK")
print(f"K_MODULE_BASEPLATE = {K_MODULE_BASEPLATE} W/mK, T_MODULE_BASEPLATE = {T_MODULE_BASEPLATE}m")
print(
f"K_TIM_INTERFACE = {K_TIM_INTERFACE} W/mK, T_TIM_INTERFACE = {T_TIM_INTERFACE}m => H_TIM = {H_MODULE_TO_HEATSINK_INTERFACE:.1f} W/m^2K")
print(f"T_ambient_inlet = {T_ambient_inlet}°C")
print(f"Rth_jhs_igbt = {Rth_jhs_igbt} K/W, Rth_jhs_diode = {Rth_jhs_diode} K/W")
print(f"A_conv_disipador (usada para calcular h_avg) = {0.7716} m^2 (¡REVISAR ESTE VALOR!)")
print(f"--- Potencias de prueba ({test_chip_powers}): ---")
for k, v in test_chip_powers.items(): print(f" {k}: {v}W")
print(f"---------------------------\n")

results_test = run_thermal_simulation(test_chip_powers, lx=test_lx, ly=test_ly, t=test_t_heatsink,
rth_heatsink=test_rth_heatsink, module_definitions=test_module_defs,
nx=test_nx, ny=test_ny)

print("\nResultados de la prueba (Tj IGBT:híbrida, Tj Diode:avg, T_NTC EXPERIMENTAL):");
print(f"Status: {results_test['status']}")
if results_test['status'] == 'Success':
print(f"Simulación con {len(test_module_defs)} módulos.")
print(f"Convergencia: {results_test['convergence']} en {results_test['iterations']} iteraciones")
print(f"T Max Disipador Principal: {results_test.get('t_max_base', np.nan):.2f} °C")
# --- CAMBIO EN LA LEYENDA DE RESULTADOS ---
print(
f"T Max Juntura (Oficial, IGBT:híbrido, Diode:avg): {results_test.get('t_max_junction', np.nan):.2f} °C ({results_test.get('t_max_junction_chip', '')})")
print(f"T Max NTC (EXPERIMENTAL): {results_test.get('t_max_ntc', np.nan):.2f} °C")
print(f"T Aire Salida Promedio: {results_test.get('t_air_outlet', np.nan):.2f} °C")
print("Resultados por Módulo:")
for mod_res_test in results_test.get('module_results', []):
print(f" - {mod_res_test.get('id', 'N/A')}: T_NTC_exp_avg={mod_res_test.get('t_ntc', np.nan):.2f}°C")
for chip_res in mod_res_test.get('chips', []):
print(f" Chip {chip_res.get('suffix')}:")
# --- CAMBIO EN LA LEYENDA DE RESULTADOS ---
base_label_hs = "Tb_hs_HÍBRIDA" if "IGBT" in chip_res.get('suffix') else "Tb_hs_AVG"
base_label_mod = "Tb_mod_HÍBRIDA" if "IGBT" in chip_res.get('suffix') else "Tb_mod_AVG"
print(
f" {base_label_hs} = {chip_res.get('t_base_heatsink', np.nan):.2f}°C -> Tj_hs = {chip_res.get('tj_from_heatsink', np.nan):.2f}°C (Oficial)")
print(
f" {base_label_mod}= {chip_res.get('t_base_module_surface', np.nan):.2f}°C -> Tj_mod = {chip_res.get('tj_from_module_surface', np.nan):.2f}°C (Verif.)")

print(f"Plot Base (Disipador): {'OK' if results_test.get('plot_base_data_uri') else 'FAIL'}")
print(f"Plot Zoom (T Módulo Sup.): {'OK' if results_test.get('plot_zoom_data_uri') else 'FAIL'}")

plot_dir = "test_plots_Tj_diff_NTC_exp" # CAMBIO DE NOMBRE DE CARPETA
os.makedirs(plot_dir, exist_ok=True)
if results_test.get('plot_base_data_uri'):
img_data_test = results_test['plot_base_data_uri'].split(',')[1]
with open(os.path.join(plot_dir, "test_plot_base_Tj_diff_NTC_exp.png"), "wb") as fh_test: fh_test.write(
base64.b64decode(img_data_test))
print(f" -> {os.path.join(plot_dir, 'test_plot_base_Tj_diff_NTC_exp.png')} guardado.")
if results_test.get('plot_zoom_data_uri'):
img_data_test_zoom = results_test['plot_zoom_data_uri'].split(',')[1]
with open(os.path.join(plot_dir, "test_plot_zoom_module_Tj_diff_NTC_exp.png"),
"wb") as fh_test_zoom: fh_test_zoom.write(base64.b64decode(img_data_test_zoom))
print(f" -> {os.path.join(plot_dir, 'test_plot_zoom_module_Tj_diff_NTC_exp.png')} guardado.")
else:
print(f"Error: {results_test.get('error_message', 'Desconocido')}")

print("Prueba local (Tj IGBT:híbrida, Tj Diode:avg, T_NTC EXPERIMENTAL) finalizada.")

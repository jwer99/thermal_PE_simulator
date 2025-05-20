# simulador_core.py

import matplotlib
matplotlib.use('Agg') # Usar backend no interactivo para Flask
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu
import time
import math
import os
import io
import base64
import traceback # Importar para trazas de error detalladas

# --- Constantes ---
NTC_REL_X = +0.021
NTC_REL_Y = -0.036
NTC_LAYER_LEVEL = 2 # Asumiendo que T_NTC se calcula en la capa 2 (Base Cu Módulo)
RTH_CHIP_NTC = {"IGBT1": 0.09, "IGBT2": 0.06, "Diode1": 0.08, "Diode2": 0.07} # Resistencia Chip Base -> Punto NTC
cp_aire = 1005.0 # J/kgK
rho_aire = 1.109 # kg/m³ (aproximado a 50°C)
w_igbt_footprint, h_igbt_footprint = 0.062, 0.122 # Dimensiones físicas módulo (m)
layers = [
    {'name': 'TIM', 't': 0.000125, 'k': 5.0},
    {'name': 'Base Cu Modulo', 't': 0.005, 'k': 390.0},
    {'name': 'Al (DBC) Modulo', 't': 0.003, 'k': 218.0}, # Considerado parte del módulo, k fijo aquí
]
# Dimensiones y Rth junction-heatsink de los chips (aproximados)
h_chip_igbt_m, w_chip_igbt_m = 0.036, 0.022
A_chip_igbt = w_chip_igbt_m * h_chip_igbt_m
h_chip_diode_m, w_chip_diode_m = 0.036, 0.011
A_chip_diode = w_chip_diode_m * h_chip_diode_m
Rth_jhs_igbt = 0.0721 # °C/W (Junction to Heatsink Surface under chip)
Rth_jhs_diode = 0.131 # °C/W
Nx_base_default, Ny_base_default = 250, 350 # Resolución FDM por defecto

# Posiciones relativas de los centros de los chips respecto al centro del módulo
chip_rel_positions = {
    "IGBT1": (-0.006, +0.023), "Diode1": (+0.012, +0.023),
    "IGBT2": (+0.006, -0.023), "Diode2": (-0.012, -0.023),
}
# --- FIN CONSTANTES ---

def fig_to_data_uri(fig, tight=True, pad_inches=0.1, transparent=False):
    """Convierte una figura de Matplotlib a Data URI PNG."""
    buf = io.BytesIO()
    save_kwargs = {'format': 'png', 'dpi': 90, 'transparent': transparent}
    if tight:
        # Usar bbox_inches='tight' solo si no necesitamos control exacto del borde
        # Para la imagen RAW, es mejor no usarlo y ajustar márgenes manualmente
        save_kwargs['bbox_inches'] = 'tight'
        save_kwargs['pad_inches'] = pad_inches
    fig.savefig(buf, **save_kwargs)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig) # Importante cerrar la figura
    return f"data:image/png;base64,{img_base64}"


def run_thermal_simulation(specific_chip_powers,
                           lx, ly, t, k_base, rth_heatsink,
                           t_ambient_inlet, Q_total_m3_h,
                           module_definitions,
                           nx=Nx_base_default, ny=Ny_base_default):
    """
    Ejecuta simulación FDM, genera URIs para plots separados y añade datos numéricos.
    """
    results = {
        'status': 'Processing', 'convergence': False, 'iterations': 0,
        't_max_base': np.nan, 't_avg_base': np.nan, 't_air_outlet': np.nan,
        't_max_junction': np.nan, 't_max_junction_chip': '', 't_max_ntc': np.nan,
        'module_results': [],
        'plot_base_data_uri': None,          # Combinado T_Base + T_Aire (Estático)
        'plot_zoom_data_uri': None,          # Zoom Módulos (Estático)
        'plot_interactive_raw_uri': None,    # SOLO contourf T_Base (Interactivo)
        'error_message': None,
        'temperature_matrix': None, 'x_coordinates': None, 'y_coordinates': None,
        'sim_lx': None, 'sim_ly': None, 'sim_nx': None, 'sim_ny': None,
    }
    start_time_sim = time.time()
    print(f"[SimCore] Iniciando con {len(module_definitions)} módulos.")
    print(f"[SimCore] Disipador: Lx={lx:.3f}, Ly={ly:.3f}, t={t:.4f}, k={k_base:.1f}, Rth={rth_heatsink:.4f}")
    print(f"[SimCore] Ambiente: T_in={t_ambient_inlet:.1f}°C, Q={Q_total_m3_h:.1f} m³/h")
    print(f"[SimCore] FDM: Nx={nx}, Ny={ny}")

    # --- Validaciones ---
    if not isinstance(specific_chip_powers, dict): results['status']='Error'; results['error_message']='Powers dict invalido.'; return results
    if not all(isinstance(d, (int, float)) and d > 1e-9 for d in [lx, ly, t, k_base, rth_heatsink, Q_total_m3_h]): results['status']='Error'; results['error_message']='Dimensiones, k_base, Rth y Caudal (Q) deben ser números > 0.'; return results
    if not isinstance(t_ambient_inlet, (int, float)): results['status']='Error'; results['error_message']='T_ambient_inlet debe ser un número.'; return results
    if not isinstance(module_definitions, list): results['status']='Error'; results['error_message']='module_definitions debe ser lista.'; return results
    if not all(isinstance(n, int) and n > 1 for n in [nx, ny]): results['status'] = 'Error'; results['error_message'] = 'Nx/Ny deben ser > 1.'; return results

    # --- Manejar caso sin módulos ---
    if not module_definitions:
        print("[SimCore] No se proporcionaron módulos. Simulando solo convección base.")
        try:
            # ... (cálculos h_eff_FDM, C_x, C_y igual que antes) ...
            A_conv_disipador = lx * ly * 2 + (2 * lx * t) + (2 * ly * t)
            if rth_heatsink > 1e-9 and A_conv_disipador > 1e-9: h_avg_disipador = 1.0 / (rth_heatsink * A_conv_disipador)
            else: h_avg_disipador = 85.0
            A_placa_base_FDM = lx * ly
            if A_placa_base_FDM > 1e-9: h_eff_FDM = h_avg_disipador * (A_conv_disipador / A_placa_base_FDM)
            else: h_eff_FDM = 10.0
            dx = lx / (nx - 1) if nx > 1 else lx; dy = ly / (ny - 1) if ny > 1 else ly
            N_nodes = nx * ny; dA = dx * dy
            C_x = k_base * t * dy / dx if dx > 1e-12 else 0
            C_y = k_base * t * dx / dy if dy > 1e-12 else 0
            A = lil_matrix((N_nodes, N_nodes)); b = np.zeros(N_nodes)
            for i in range(nx):
                for j in range(ny):
                    idx = i * ny + j; h_local = h_eff_FDM; C_h_local = h_local * dA; coef_diag = -C_h_local
                    b[idx] = - h_local * dA * t_ambient_inlet
                    if i > 0: A[idx, idx - ny] = C_x; coef_diag -= C_x
                    if i < nx - 1: A[idx, idx + ny] = C_x; coef_diag -= C_x
                    if j > 0: A[idx, idx - 1]  = C_y; coef_diag -= C_y
                    if j < ny - 1: A[idx, idx + 1]  = C_y; coef_diag -= C_y
                    A[idx, idx] = coef_diag

            try:
                LU = splu(A.tocsr()); T_flat = LU.solve(b); T = T_flat.reshape((nx, ny))
                T_air = np.full((nx, ny), t_ambient_inlet) # Aire no se calienta
                results['t_max_base'] = np.max(T); results['t_avg_base'] = np.mean(T)
                results['t_air_outlet'] = t_ambient_inlet
                results['convergence'] = True; results['iterations'] = 0
                results['status'] = 'Success'

                # --- Generar gráficos (caso sin módulos) ---
                try:
                    x_coords=np.linspace(0,lx,nx); y_coords=np.linspace(0,ly,ny); X,Y=np.meshgrid(x_coords,y_coords,indexing='ij')
                    min_T_plot=t_ambient_inlet; max_T_plot=np.max(T) if not np.isnan(T).all() else t_ambient_inlet+1
                    max_T_plot = max(min_T_plot + 0.1, max_T_plot)
                    levels_main=np.linspace(min_T_plot, max_T_plot, 30) if abs(max_T_plot-min_T_plot)>1e-3 else np.linspace(min_T_plot-0.5,max_T_plot+0.5,2)
                    T_plot_base = np.nan_to_num(T, nan=min_T_plot-10)

                    # --- Gráfico Combinado Base/Aire (Estático) ---
                    fig_main, axes = plt.subplots(1, 2, figsize=(11, 4.5)) # Tamaño normal aquí
                    title_fontsize=12; label_fontsize=10; tick_fontsize=9; suptitle_fontsize=14
                    cbar_base = fig_main.colorbar(axes[0].contourf(X,Y,T_plot_base,levels=levels_main,cmap="hot",extend='max'),ax=axes[0])
                    cbar_base.set_label("T Base (°C)", size=label_fontsize); cbar_base.ax.tick_params(labelsize=tick_fontsize)
                    axes[0].set_title("T Base (Sin Módulos)", fontsize=title_fontsize); axes[0].set_aspect('equal')
                    axes[0].set_xlabel("x (m)", fontsize=label_fontsize); axes[0].set_ylabel("y (m)", fontsize=label_fontsize)
                    axes[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
                    levels_air = np.array([t_ambient_inlet-0.5, t_ambient_inlet+0.5])
                    cbar_air = fig_main.colorbar(axes[1].contourf(X,Y,T_air,levels=levels_air, cmap="coolwarm"),ax=axes[1])
                    cbar_air.set_label("T Aire (°C)", size=label_fontsize, ticks=[t_ambient_inlet]); cbar_air.ax.tick_params(labelsize=tick_fontsize)
                    axes[1].set_title("T Aire (Entrada)", fontsize=title_fontsize); axes[1].set_aspect('equal')
                    axes[1].set_xlabel("x (m)", fontsize=label_fontsize); axes[1].set_ylabel("y (m)", fontsize=label_fontsize)
                    axes[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
                    fig_main.suptitle(f"Visión General (Sin Módulos, T_in={t_ambient_inlet:.1f}°C)", fontsize=suptitle_fontsize)
                    fig_main.tight_layout(rect=[0, 0.03, 1, 0.93])
                    results['plot_base_data_uri'] = fig_to_data_uri(fig_main, tight=True, pad_inches=0.1)

                    # --- Gráfico RAW SOLO T_Base (Interactivo) ---
                    fig_raw, ax_raw = plt.subplots(figsize=(lx*100/2.54/90*2.5, ly*100/2.54/90*2.5), dpi=90) # Ajustar figsize para aspecto correcto
                    ax_raw.contourf(X, Y, T_plot_base, levels=levels_main, cmap="hot", extend='neither')
                    ax_raw.set_xlim(0, lx); ax_raw.set_ylim(0, ly)
                    ax_raw.set_aspect('equal')
                    ax_raw.axis('off') # Sin ejes
                    fig_raw.subplots_adjust(left=0, right=1, top=1, bottom=0) # Sin márgenes internos
                    results['plot_interactive_raw_uri'] = fig_to_data_uri(fig_raw, tight=False, transparent=True) # Guardar sin bbox_inches=tight

                    # Añadir datos numéricos
                    results['temperature_matrix'] = T; results['x_coordinates'] = x_coords; results['y_coordinates'] = y_coords
                    results['sim_lx'] = lx; results['sim_ly'] = ly; results['sim_nx'] = nx; results['sim_ny'] = ny
                    print("[SimCore] Gráficos (sin módulos) y datos interactivos generados.")

                except Exception as e_gfx:
                    print(f"[SimCore] Error GFX (sin módulos): {e_gfx}")
                    traceback.print_exc()
                    results['error_message'] = (results.get('error_message') or "") + f"; GFX Err (no mod): {e_gfx}"


            except Exception as e_solve:
                 results['status'] = 'Error'; results['error_message'] = f"Error resolviendo sistema base: {e_solve}"
                 traceback.print_exc()
        except Exception as e_base:
             results['status'] = 'Error'; results['error_message'] = f"Error en setup sin módulos: {e_base}"
             traceback.print_exc()

        print(f"[SimCore] Simulación (sin módulos) completada en {time.time() - start_time_sim:.2f}s.")
        return results # Devuelve los resultados del caso sin módulos

    # --- Ejecución con módulos ---
    try:
        # --- Cálculos iniciales, Config FDM, Procesar geometría, Solver FDM ---
        # ... (Código idéntico a la versión anterior hasta el final del bucle while de convergencia) ...
        Q_total_m3_s = Q_total_m3_h / 3600.0; m_dot_total_kgs = Q_total_m3_s * rho_aire
        A_conv_disipador = lx * ly * 2 + (2 * lx * t) + (2 * ly * t)
        if rth_heatsink > 1e-9 and A_conv_disipador > 1e-9: h_avg_disipador = 1.0 / (rth_heatsink * A_conv_disipador)
        else: h_avg_disipador = 85.0
        A_placa_base_FDM = lx * ly
        if A_placa_base_FDM > 1e-9: h_eff_FDM = h_avg_disipador * (A_conv_disipador / A_placa_base_FDM)
        else: h_eff_FDM = 10.0
        Rth_1D_stack_pp = sum(layer['t'] / layer['k'] for layer in layers if layer['k'] > 1e-9)
        t_stack = sum(layer['t'] for layer in layers)
        k_eff_stack = t_stack / Rth_1D_stack_pp if Rth_1D_stack_pp > 1e-12 else 0.0
        A_igbt_footprint = w_igbt_footprint * h_igbt_footprint
        sqrt_A_footprint = math.sqrt(A_igbt_footprint) if A_igbt_footprint > 1e-12 else 0
        Psi_estimado = 0.25
        R_sp_total = Psi_estimado / (k_eff_stack * sqrt_A_footprint) if k_eff_stack > 1e-9 and sqrt_A_footprint > 1e-9 else float('inf')
        R_sp_stack_pp = R_sp_total * A_igbt_footprint if A_igbt_footprint > 1e-12 else float('inf')
        Rth_convection_pp = 1.0 / h_eff_FDM if h_eff_FDM > 1e-9 else float('inf')
        Rth_total_igbt_pp = Rth_1D_stack_pp + R_sp_stack_pp + Rth_convection_pp
        h_eff_IGBT = 1.0 / Rth_total_igbt_pp if Rth_total_igbt_pp > 1e-12 else 0.0
        dx = lx / (nx - 1) if nx > 1 else lx; dy = ly / (ny - 1) if ny > 1 else ly
        if lx > 1e-9 and nx > 0 and dx > 1e-12: m_dot_per_meter_width = m_dot_total_kgs / lx; m_dot_column_kgs = m_dot_per_meter_width * dx; m_dot_cp_column = m_dot_column_kgs * cp_aire
        else: m_dot_cp_column = 1e-12
        if m_dot_cp_column < 1e-12: m_dot_cp_column = 1e-12
        N_nodes = nx * ny; T_air = np.full((nx, ny), t_ambient_inlet); T = np.full((nx, ny), t_ambient_inlet + 10.0)
        P_total_fuentes_nominal = sum(p for p in specific_chip_powers.values() if p is not None and p > 0)
        print(f"[SimCore] Potencia total nominal activa: {P_total_fuentes_nominal:.1f} W")
        modules_data_sim = []; all_chips_data_sim = []
        Nx_module_footprint = max(1, int(round(w_igbt_footprint / dx))) if dx > 1e-12 else 1
        Ny_module_footprint = max(1, int(round(h_igbt_footprint / dy))) if dy > 1e-12 else 1
        nx_module_half = Nx_module_footprint // 2; ny_module_half = Ny_module_footprint // 2
        for mod_def in module_definitions:
            module_id_base = mod_def['id']; module_center_x = mod_def['center_x']; module_center_y = mod_def['center_y']
            half_w_footprint = w_igbt_footprint / 2; half_h_footprint = h_igbt_footprint / 2
            module_center_x = max(half_w_footprint, min(lx - half_w_footprint, module_center_x))
            module_center_y = max(half_h_footprint, min(ly - half_h_footprint, module_center_y))
            module_center_x_idx = int(round(module_center_x / dx)) if dx > 1e-12 else 0
            module_center_y_idx = int(round(module_center_y / dy)) if dy > 1e-12 else 0
            module_center_x_idx = max(0, min(nx - 1, module_center_x_idx))
            module_center_y_idx = max(0, min(ny - 1, module_center_y_idx))
            footprint_i_min = max(0, module_center_x_idx - nx_module_half); footprint_i_max = min(nx - 1, module_center_x_idx + nx_module_half)
            footprint_j_min = max(0, module_center_y_idx - ny_module_half); footprint_j_max = min(ny - 1, module_center_y_idx + ny_module_half)
            module_info = { 'id': module_id_base, 'center_x_m': module_center_x, 'center_y_m': module_center_y, 'center_x_idx': module_center_x_idx, 'center_y_idx': module_center_y_idx, 'footprint_i_min': footprint_i_min, 'footprint_i_max': footprint_i_max, 'footprint_j_min': footprint_j_min, 'footprint_j_max': footprint_j_max, 'chips': [], 'ntc_abs_x': None, 'ntc_abs_y': None, 'ntc_i_idx': None, 'ntc_j_idx': None, 'T_ntc_matrix_calc': np.nan }
            ntc_abs_x = module_center_x + NTC_REL_X; ntc_abs_y = module_center_y + NTC_REL_Y
            module_info['ntc_abs_x'] = ntc_abs_x; module_info['ntc_abs_y'] = ntc_abs_y
            if 0 <= ntc_abs_x <= lx and 0 <= ntc_abs_y <= ly:
                ntc_i_idx = int(round(ntc_abs_x / dx)) if dx > 1e-12 else 0; ntc_j_idx = int(round(ntc_abs_y / dy)) if dy > 1e-12 else 0
                ntc_i_idx = max(0, min(nx - 1, ntc_i_idx)); ntc_j_idx = max(0, min(ny - 1, ntc_j_idx))
                module_info['ntc_i_idx'] = ntc_i_idx; module_info['ntc_j_idx'] = ntc_j_idx
            for chip_label_suffix, (rel_x, rel_y) in chip_rel_positions.items():
                chip_full_id = f"{module_id_base}_{chip_label_suffix}"; chip_center_x_m = module_center_x + rel_x; chip_center_y_m = module_center_y + rel_y
                chip_center_x_m = max(0, min(lx, chip_center_x_m)); chip_center_y_m = max(0, min(ly, chip_center_y_m))
                chip_x_idx = int(round(chip_center_x_m / dx)) if dx > 1e-12 else 0; chip_y_idx = int(round(chip_center_y_m / dy)) if dy > 1e-12 else 0
                chip_x_idx = max(0, min(nx-1, chip_x_idx)); chip_y_idx = max(0, min(ny-1, chip_y_idx))
                is_igbt="IGBT" in chip_label_suffix; chip_type="IGBT" if is_igbt else "Diode"
                if is_igbt: chip_w_m,chip_h_m,chip_A,chip_Rth = w_chip_igbt_m,h_chip_igbt_m,A_chip_igbt,Rth_jhs_igbt
                else: chip_w_m,chip_h_m,chip_A,chip_Rth = w_chip_diode_m,h_chip_diode_m,A_chip_diode,Rth_jhs_diode
                chip_Nx_cells = max(1,int(round(chip_w_m/dx))) if dx>1e-12 else 1; chip_Ny_cells = max(1,int(round(chip_h_m/dy))) if dy>1e-12 else 1
                chip_nx_half,chip_ny_half = chip_Nx_cells//2, chip_Ny_cells//2; chip_power = specific_chip_powers.get(chip_full_id, 0.0); chip_q = chip_power / chip_A if chip_A > 1e-12 else 0.0
                chip_data = {'label':chip_full_id, 'suffix':chip_label_suffix, 'type':chip_type, 'center_x_m':chip_center_x_m,'center_y_m':chip_center_y_m, 'center_x_idx':chip_x_idx,'center_y_idx':chip_y_idx, 'power':chip_power,'q':chip_q, 'Rth_jhs':chip_Rth, 'T_base_chip':np.nan,'Tj':np.nan, 'w_m':chip_w_m,'h_m':chip_h_m,'Area_m2':chip_A, 'Nx_cells':chip_Nx_cells,'Ny_cells':chip_Ny_cells, 'nx_half':chip_nx_half,'ny_half':chip_ny_half}
                module_info['chips'].append(chip_data); all_chips_data_sim.append(chip_data)
            modules_data_sim.append(module_info)
        dA = dx * dy; C_x = k_base * t * dy / dx if dx > 1e-12 else 0; C_y = k_base * t * dx / dy if dy > 1e-12 else 0
        max_iterations = 100; convergence_tolerance = 0.01; iteration = 0; converged = False
        h_local_map = np.full((nx, ny), h_eff_FDM); q_fuente_map = np.zeros((nx, ny))
        for module in modules_data_sim:
            imin, imax = module['footprint_i_min'], module['footprint_i_max']
            jmin, jmax = module['footprint_j_min'], module['footprint_j_max']
            if imin <= imax and jmin <= jmax: h_local_map[imin:imax+1, jmin:jmax+1] = h_eff_IGBT
        for chip in all_chips_data_sim:
             i_min_chip=max(0,chip['center_x_idx']-chip['nx_half']); i_max_chip=min(nx-1,chip['center_x_idx']+chip['nx_half'])
             j_min_chip=max(0,chip['center_y_idx']-chip['ny_half']); j_max_chip=min(ny-1,chip['center_y_idx']+chip['ny_half'])
             num_nodes_chip=(i_max_chip-i_min_chip+1)*(j_max_chip-j_min_chip+1)
             if num_nodes_chip>0 and dA>1e-12 and chip['power']>0:
                 power_per_node_dist=chip['power']/num_nodes_chip; q_W_per_m2_dist=power_per_node_dist/dA
                 if i_min_chip<=i_max_chip and j_min_chip<=j_max_chip: q_fuente_map[i_min_chip:i_max_chip+1, j_min_chip:j_max_chip+1] += q_W_per_m2_dist
        A = lil_matrix((N_nodes, N_nodes)); b = np.zeros(N_nodes)
        for i in range(nx):
            for j in range(ny):
                idx=i*ny+j; h_local=h_local_map[i,j]; C_h_local=h_local*dA; coef_diag=-C_h_local
                if i > 0: A[idx, idx-ny]=C_x; coef_diag-=C_x
                if i < nx-1: A[idx, idx+ny]=C_x; coef_diag-=C_x
                if j > 0: A[idx, idx-1]=C_y; coef_diag-=C_y
                if j < ny-1: A[idx, idx+1]=C_y; coef_diag-=C_y
                A[idx, idx]=coef_diag
        A_csr = A.tocsr()
        try: LU = splu(A_csr)
        except RuntimeError as e: results['status']='Error'; results['error_message']=f"Error LU: {e}"; traceback.print_exc(); return results
        print("[SimCore] Iniciando bucle iterativo T_aire / T_base...")
        while not converged and iteration < max_iterations:
            iter_start_time = time.time(); iteration += 1; b.fill(0.0)
            for i in range(nx):
                for j in range(ny):
                    idx=i*ny+j; h_local=h_local_map[i,j]; q_fuente_W_m2=q_fuente_map[i,j]
                    q_fuente_W_nodo=q_fuente_W_m2*dA; T_ambient_local=T_air[i,j]
                    b[idx] = -q_fuente_W_nodo - h_local*dA*T_ambient_local
            try: T_flat = LU.solve(b); T_new = T_flat.reshape((nx, ny))
            except (RuntimeError,ValueError) as e: results['status']='Error'; results['error_message']=f"Error solver iter {iteration}: {e}"; traceback.print_exc(); return results
            T_air_next=np.full_like(T_air, t_ambient_inlet); delta_T_air_nodes=np.zeros_like(T)
            h_surface_to_air = h_eff_FDM
            for i in range(nx):
                for j in range(ny):
                    P_conv = h_surface_to_air * dA * (T_new[i,j] - T_air[i,j]); P_conv_nodes = max(0, P_conv)
                    if m_dot_cp_column > 1e-12: delta_T_air_nodes[i,j] = P_conv_nodes / m_dot_cp_column
                    else: delta_T_air_nodes[i,j] = 0
            for i in range(nx):
                for j in range(1, ny): T_air_next[i, j] = T_air_next[i, j-1] + delta_T_air_nodes[i, j-1]
            max_diff_T_air = np.max(np.abs(T_air_next - T_air)); converged = max_diff_T_air < convergence_tolerance
            T = T_new.copy(); T_air = T_air_next.copy()
        print(f"[SimCore] Bucle terminado en {iteration} iteraciones. Converged: {converged}")
        results['convergence'] = converged; results['iterations'] = iteration

        # --- Post-procesamiento (Cálculo de métricas, Tj, T_ntc) ---
        # ... (Código idéntico a la versión anterior) ...
        if np.isnan(T).any(): results['t_max_base']=np.nan; results['t_avg_base']=np.nan
        else: results['t_max_base']=np.max(T); results['t_avg_base']=np.mean(T)
        if ny>0 and not np.isnan(T_air[:,ny-1]).any(): results['t_air_outlet']=np.mean(T_air[:,ny-1])
        else: results['t_air_outlet']=np.nan
        max_tj_overall=-float('inf'); max_tj_chip_label=""; max_t_ntc_overall=-float('inf')
        def get_avg_chip_base_temp_local(T_matrix, chip_center_x_idx, chip_center_y_idx, chip_nx_half, chip_ny_half, Nx_global, Ny_global):
            i_min=max(0, chip_center_x_idx-chip_nx_half); i_max=min(Nx_global-1, chip_center_x_idx+chip_nx_half)
            j_min=max(0, chip_center_y_idx-chip_ny_half); j_max=min(Ny_global-1, chip_center_y_idx+chip_ny_half)
            if i_min > i_max or j_min > j_max: return np.nan
            i_max = min(i_max, T_matrix.shape[0] - 1); j_max = min(j_max, T_matrix.shape[1] - 1)
            area_nodes=T_matrix[i_min:i_max+1, j_min:j_max+1];
            if area_nodes.size == 0: return np.nan
            return np.nanmean(area_nodes)
        module_results_list = []
        for module in modules_data_sim:
            module_result = {'id': module['id'], 'chips': [], 't_ntc': np.nan}
            for chip in module['chips']:
                T_base_chip = get_avg_chip_base_temp_local(T, chip['center_x_idx'], chip['center_y_idx'], chip['nx_half'], chip['ny_half'], nx, ny)
                chip['T_base_chip'] = T_base_chip; Tj = np.nan
                if not np.isnan(T_base_chip):
                    if chip['power'] > 1e-6 and not np.isnan(chip['Rth_jhs']): Tj = T_base_chip + chip['Rth_jhs'] * chip['power']
                    else: Tj = T_base_chip
                    if not np.isnan(Tj):
                        if Tj > max_tj_overall: max_tj_overall = Tj; max_tj_chip_label = f"{chip['label']} (P={chip['power']:.1f}W)"
                chip['Tj'] = Tj
                module_result['chips'].append({'suffix': chip['suffix'], 't_base': T_base_chip, 'tj': Tj})
            numerator = 0.0; denominator = 0.0; valid_chip_contribution = False
            for chip in module['chips']:
                chip_suffix=chip['suffix']
                if chip_suffix in RTH_CHIP_NTC:
                    rth_chip_ntc=RTH_CHIP_NTC[chip_suffix]; t_base_chip=chip['T_base_chip']
                    if not np.isnan(t_base_chip) and rth_chip_ntc > 1e-9 and np.isfinite(rth_chip_ntc):
                        conductance = 1.0 / rth_chip_ntc; numerator += conductance * t_base_chip; denominator += conductance; valid_chip_contribution = True
            t_ntc_matrix = np.nan
            if valid_chip_contribution and denominator > 1e-9:
                t_ntc_matrix = numerator / denominator; module['T_ntc_matrix_calc'] = t_ntc_matrix
                if not np.isnan(t_ntc_matrix): max_t_ntc_overall = np.nanmax([max_t_ntc_overall, t_ntc_matrix])
            module_result['t_ntc'] = t_ntc_matrix
            module_results_list.append(module_result)
        results['t_max_junction'] = max_tj_overall if not np.isinf(max_tj_overall) else np.nan
        results['t_max_junction_chip'] = max_tj_chip_label
        results['t_max_ntc'] = max_t_ntc_overall if not np.isinf(max_t_ntc_overall) else np.nan
        results['module_results'] = module_results_list

        # --- Generación de Gráficos ---
        print("[SimCore] Generating Graphics...")
        x_coords=np.linspace(0,lx,nx); y_coords=np.linspace(0,ly,ny); X,Y=np.meshgrid(x_coords,y_coords,indexing='ij')

        min_T_plot=t_ambient_inlet
        max_T_plot=results['t_max_base'] if not np.isnan(results['t_max_base']) else t_ambient_inlet+1
        max_T_plot=max(max_T_plot, min_T_plot+1); max_T_plot+=(max_T_plot-min_T_plot)*0.05
        levels_main=np.linspace(min_T_plot,max_T_plot,30) if abs(max_T_plot-min_T_plot)>1e-3 else np.linspace(min_T_plot-0.5,max_T_plot+0.5,2)
        T_plot_base=np.nan_to_num(T, nan=min_T_plot-10)

        # --- 1. Gráfico Combinado Base/Aire (Estático) ---
        try:
            fig_main, axes = plt.subplots(1, 2, figsize=(22, 9)) # Tamaño grande
            title_fontsize = 18; label_fontsize = 16; tick_fontsize = 14; suptitle_fontsize = 20

            # Subplot T Base
            cbar_base = fig_main.colorbar(axes[0].contourf(X,Y,T_plot_base,levels=levels_main,cmap="hot",extend='max'), ax=axes[0])
            cbar_base.set_label("T Base (°C)", size=label_fontsize); cbar_base.ax.tick_params(labelsize=tick_fontsize)
            tmax_str=f"{results['t_max_base']:.1f}°C" if not np.isnan(results['t_max_base']) else "N/A"
            axes[0].set_title(f"T Base (Max={tmax_str})", fontsize=title_fontsize); axes[0].set_aspect('equal')
            axes[0].set_xlabel("x (m)", fontsize=label_fontsize); axes[0].set_ylabel("y (m)", fontsize=label_fontsize)
            axes[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)

            # Subplot T Aire
            min_T_air_plot=t_ambient_inlet; T_air_max_plot=np.nanmax(T_air) if not np.isnan(T_air).all() else t_ambient_inlet; max_T_air_plot=T_air_max_plot
            max_T_air_plot=max(max_T_air_plot,min_T_air_plot+0.1); max_T_air_plot+=(max_T_air_plot-min_T_air_plot)*0.05
            levels_air=np.linspace(min_T_air_plot,max_T_air_plot,30) if abs(max_T_air_plot-min_T_air_plot)>1e-3 else np.linspace(min_T_air_plot-0.5,max_T_air_plot+0.5,2)
            T_air_plot=np.nan_to_num(T_air, nan=min_T_air_plot-10)
            cbar_air = fig_main.colorbar(axes[1].contourf(X,Y,T_air_plot,levels=levels_air,cmap="coolwarm",extend='max'), ax=axes[1])
            cbar_air.set_label("T Air (°C)", size=label_fontsize); cbar_air.ax.tick_params(labelsize=tick_fontsize)
            tairout_str=f"{results['t_air_outlet']:.1f}°C" if not np.isnan(results['t_air_outlet']) else "N/A"
            axes[1].set_title(f"Mean Air outlet temperature={tairout_str}", fontsize=title_fontsize); axes[1].set_aspect('equal')
            axes[1].set_xlabel("x (m)", fontsize=label_fontsize); axes[1].set_ylabel("y (m)", fontsize=label_fontsize)
            axes[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)

            # Overlays
            for ax in axes:
                 for module in modules_data_sim:
                     center_x_mod,center_y_mod=module['center_x_m'],module['center_y_m']; rect_x_mod,rect_y_mod=center_x_mod-w_igbt_footprint/2,center_y_mod-h_igbt_footprint/2
                     rect_mod=plt.Rectangle((rect_x_mod,rect_y_mod),w_igbt_footprint,h_igbt_footprint,edgecolor='gray',facecolor='none',lw=0.8,ls='--'); ax.add_patch(rect_mod)
                     if module['ntc_abs_x'] is not None and module['ntc_abs_y'] is not None and 0<=module['ntc_abs_x']<=lx and 0<=module['ntc_abs_y']<=ly: ax.plot(module['ntc_abs_x'], module['ntc_abs_y'], 'wo', markersize=3, markeredgecolor='black')

            fig_main.suptitle(f"Static General Vision ({len(modules_data_sim)} Module, P={P_total_fuentes_nominal:.0f}W, T_in={t_ambient_inlet:.1f}°C, Q={Q_total_m3_h:.0f}m³/h)", fontsize=suptitle_fontsize)
            fig_main.tight_layout(rect=[0, 0.03, 1, 0.93])
            results['plot_base_data_uri'] = fig_to_data_uri(fig_main, tight=True, pad_inches=0.1) # Guarda URI combinado
        except Exception as e:
            print(f"[SimCore] Error GFX Main (Combinado): {e}")
            traceback.print_exc()
            results['error_message'] = (results.get('error_message') or "") + f"; GFX Main Err: {e}"

        # --- 2. Gráfico RAW SOLO T_Base (Interactivo) ---
        try:
            # Ajustar figsize basado en lx/ly para mantener aspecto
            # dpi=90 es el valor por defecto usado en fig_to_data_uri
            fig_width_inches = lx * (90 / 2.54) / 2 # Ajustar multiplicador si es necesario
            fig_height_inches = ly * (90 / 2.54) / 2
            fig_raw, ax_raw = plt.subplots(figsize=(fig_width_inches, fig_height_inches), dpi=90)

            ax_raw.contourf(X, Y, T_plot_base, levels=levels_main, cmap="hot", extend='neither')
            ax_raw.set_xlim(0, lx); ax_raw.set_ylim(0, ly)
            ax_raw.set_aspect('equal')
            ax_raw.axis('off')
            fig_raw.subplots_adjust(left=0, right=1, top=1, bottom=0) # Quitar márgenes

            # Guardar sin bbox_inches='tight' para evitar márgenes blancos inesperados
            results['plot_interactive_raw_uri'] = fig_to_data_uri(fig_raw, tight=False, transparent=True)
            print("[SimCore] Gráfico RAW interactivo generado.")
        except Exception as e:
            print(f"[SimCore] Error GFX RAW Interactivo: {e}")
            traceback.print_exc()
            results['error_message'] = (results.get('error_message') or "") + f"; GFX Interactive RAW Err: {e}"

        # --- 3. Gráfico Zoom por Módulo (Estático) ---
        try:
             zoom_title_fontsize=10; zoom_label_fontsize=8; zoom_tick_fontsize=7; zoom_suptitle_fontsize=12; chip_text_fontsize=7; ntc_text_fontsize=8; ntc_markersize=5
             num_modules=len(modules_data_sim);
             if num_modules == 0: raise ValueError("No modules to plot zoom for.")
             ncols=min(3,num_modules); nrows=math.ceil(num_modules/ncols); fig_zoom,axes_zoom=plt.subplots(nrows=nrows,ncols=ncols,figsize=(ncols*4,nrows*3.8),squeeze=False)
             axes_flat=axes_zoom.flatten(); contour_zoom_ref=None
             # Usar los mismos niveles que el gráfico base interactivo para consistencia
             levels_zoom = levels_main
             T_plot_zoom_src = T_plot_base # Usar la misma matriz preprocesada

             for idx, module in enumerate(modules_data_sim):
                 if idx >= len(axes_flat): break
                 ax=axes_flat[idx]; margin_factor=0.3
                 zoom_x_min=module['center_x_m']-w_igbt_footprint/2*(1+margin_factor); zoom_x_max=module['center_x_m']+w_igbt_footprint/2*(1+margin_factor)
                 zoom_y_min=module['center_y_m']-h_igbt_footprint/2*(1+margin_factor); zoom_y_max=module['center_y_m']+h_igbt_footprint/2*(1+margin_factor)
                 zoom_x_min=max(0, zoom_x_min); zoom_x_max=min(lx, zoom_x_max); zoom_y_min=max(0, zoom_y_min); zoom_y_max=min(ly, zoom_y_max)
                 contour_zoom=ax.contourf(X, Y, T_plot_zoom_src, levels=levels_zoom, cmap="hot", extend='max'); # Extender max aquí sí
                 if idx==0: contour_zoom_ref=contour_zoom
                 center_x_mod,center_y_mod=module['center_x_m'],module['center_y_m']; rect_x_mod,rect_y_mod=center_x_mod-w_igbt_footprint/2,center_y_mod-h_igbt_footprint/2
                 rect_mod=plt.Rectangle((rect_x_mod,rect_y_mod),w_igbt_footprint,h_igbt_footprint,edgecolor='black',facecolor='none',lw=1.0,ls='--'); ax.add_patch(rect_mod)
                 for chip in module['chips']:
                       center_x_phys,center_y_phys=chip['center_x_m'],chip['center_y_m']; chip_w_plot,chip_h_plot=chip['w_m'],chip['h_m']
                       rect_x_phys,rect_y_phys=center_x_phys-chip_w_plot/2,center_y_phys-chip_h_plot/2; is_igbt=chip['type']=="IGBT"
                       rect=plt.Rectangle((rect_x_phys,rect_y_phys),chip_w_plot,chip_h_plot,edgecolor='cyan' if is_igbt else 'lime',facecolor='none',lw=1.0,ls=':'); ax.add_patch(rect)
                       tj_str=f"Tj={chip['Tj']:.1f}" if chip['Tj'] is not None and not np.isnan(chip['Tj']) else "Tj=N/A"
                       tbase_str=f"Tb≈{chip['T_base_chip']:.1f}" if chip['T_base_chip'] is not None and not np.isnan(chip['T_base_chip']) else "Tb=N/A"
                       ax.text(center_x_phys,center_y_phys,f"{chip['suffix']}\n{tj_str}\n({tbase_str})",color='white',ha='center',va='center',fontsize=chip_text_fontsize,bbox=dict(boxstyle='round,pad=0.1',fc='black',alpha=0.6))
                 if module['ntc_abs_x'] is not None and module['ntc_abs_y'] is not None:
                       ntc_x_plot,ntc_y_plot=module['ntc_abs_x'],module['ntc_abs_y']
                       if zoom_x_min<=ntc_x_plot<=zoom_x_max and zoom_y_min<=ntc_y_plot<=zoom_y_max:
                           ax.plot(ntc_x_plot,ntc_y_plot,'ro',markersize=ntc_markersize,markeredgecolor='white')
                           t_ntc_val=module['T_ntc_matrix_calc']; t_ntc_str=f"NTC≈{t_ntc_val:.1f}" if not np.isnan(t_ntc_val) else "NTC=N/A"
                           ax.text(ntc_x_plot+(zoom_x_max-zoom_x_min)*0.03,ntc_y_plot,t_ntc_str,color='red',ha='left',va='center',fontsize=ntc_text_fontsize,bbox=dict(boxstyle='round,pad=0.1',fc='white',alpha=0.7))
                 ax.set_xlim(zoom_x_min,zoom_x_max); ax.set_ylim(zoom_y_min,zoom_y_max); ax.set_title(f"{module['id']}",fontsize=zoom_title_fontsize); ax.set_xlabel("x (m)",fontsize=zoom_label_fontsize); ax.set_ylabel("y (m)",fontsize=zoom_label_fontsize); ax.tick_params(axis='both',which='major',labelsize=zoom_tick_fontsize); ax.set_aspect('equal',adjustable='box'); ax.grid(True,linestyle=':',alpha=0.3)
             for i in range(idx+1, len(axes_flat)): axes_flat[i].axis('off')
             if contour_zoom_ref is not None and num_modules > 0:
                  fig_zoom.subplots_adjust(right=0.85, top=0.90, bottom=0.1); cbar_ax=fig_zoom.add_axes([0.88,0.15,0.03,0.7])
                  cbar=fig_zoom.colorbar(contour_zoom_ref,cax=cbar_ax);
                  cbar.set_label("T Base (°C)",size=zoom_label_fontsize)
                  cbar.ax.tick_params(labelsize=zoom_tick_fontsize)
             fig_zoom.suptitle(f"Detailed Módules (Estátic)", fontsize=zoom_suptitle_fontsize)
             results['plot_zoom_data_uri'] = fig_to_data_uri(fig_zoom, tight=True, pad_inches=0.1) # Guarda URI zoom
        except Exception as e:
            print(f"[SimCore] Error GFX Zoom: {e}")
            traceback.print_exc()
            results['error_message'] = (results.get('error_message') or "") + f"; GFX Zoom Err: {e}"

        results['status'] = 'Success'
        print("[SimCore] Generación de gráficos completada.")

        # --- Añadir datos numéricos ---
        if results['status'] == 'Success':
             try:
                 results['temperature_matrix'] = T; results['x_coordinates'] = x_coords; results['y_coordinates'] = y_coords
                 results['sim_lx'] = lx; results['sim_ly'] = ly; results['sim_nx'] = nx; results['sim_ny'] = ny
                 print("[SimCore] Datos numéricos añadidos a los resultados para interactividad.")
             except NameError as e:
                 print(f"[SimCore] Advertencia: No se pudieron añadir datos numéricos. Error: {e}")
                 results['status'] = 'Success_NoInteractiveData'
                 if results.get('error_message') is None: results['error_message'] = ''
                 results['error_message'] += "; Warn: Datos interactivos no generados"

    except Exception as e:
        print(f"[SimCore] Error general en simulación: {e}")
        traceback.print_exc()
        results['status'] = 'Error'
        if results.get('error_message') is None: results['error_message'] = f"Error inesperado en simulador: {e}"

    print(f"[SimCore] Simulación completada en {time.time() - start_time_sim:.2f}s. Estado: {results['status']}")
    return results

# --- Bloque para pruebas locales ---
if __name__ == '__main__':
     print("Ejecutando prueba local de simulador_core.py (Modo Dinámico)...")
     test_module_defs = [ {'id': 'Mod_A', 'center_x': 0.1, 'center_y': 0.25}, {'id': 'Mod_B', 'center_x': 0.2, 'center_y': 0.15}, ]
     test_chip_powers = { 'Mod_A_IGBT1': 150.0, 'Mod_A_Diode1': 50.0, 'Mod_A_IGBT2': 150.0, 'Mod_A_Diode2': 50.0, 'Mod_B_IGBT1': 200.0, 'Mod_B_Diode1': 80.0, 'Mod_B_IGBT2': 200.0, 'Mod_B_Diode2': 80.0, }
     test_lx = 0.25; test_ly = 0.35; test_t = 0.02; test_rth = 0.015
     test_k_base = 218.0
     test_t_ambient_inlet = 45.0
     test_Q_total_m3_h = 800.0
     test_nx = 50; test_ny = 70 # Bajar resolución para prueba rápida
     results = run_thermal_simulation(
         specific_chip_powers=test_chip_powers,
         lx=test_lx, ly=test_ly, t=test_t, k_base=test_k_base, rth_heatsink=test_rth,
         t_ambient_inlet=test_t_ambient_inlet, Q_total_m3_h=test_Q_total_m3_h,
         module_definitions=test_module_defs,
         nx=test_nx, ny=test_ny
     )
     print("\nResultados de la prueba:"); print(f"Status: {results['status']}")
     if results['status'].startswith('Success'):
         print(f"  Simulación con {len(test_module_defs)} módulos.")
         print(f"  Plot Combinado URI: {'Presente' if results.get('plot_base_data_uri') else 'Ausente'}")
         print(f"  Plot Interactivo RAW URI: {'Presente' if results.get('plot_interactive_raw_uri') else 'Ausente'}")
         print(f"  Plot Zoom URI: {'Presente' if results.get('plot_zoom_data_uri') else 'Ausente'}")
         print(f"  Datos Interactivos (Matriz T): {'Presente' if results.get('temperature_matrix') is not None else 'Ausente'}")
     else: print(f"  Error: {results.get('error_message', 'Desconocido')}")
     if 'Warn:' in results.get('error_message', ''): print(f"  Advertencia: {results.get('error_message')}")
     print("\nPrueba local finalizada.")
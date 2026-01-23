import numpy as np

def solve_channel_flow_2d(L, H, U_in, T_in, T_wall, rho, mu, cp, k_fluid, nx=30, ny=15, max_iter=2000):
    """
    Solves 2D laminar Navier-Stokes (Vorticity-Streamfunction) and Energy equation
    for flow between parallel plates (channel flow).

    Parameters:
    - L: Length of the channel (m) [Flow direction]
    - H: Gap between plates (m) [Transverse direction, total width]
    - U_in: Inlet velocity (m/s)
    - T_in: Inlet temperature (degC)
    - T_wall: Wall temperature (degC)
    - rho: Density (kg/m3)
    - mu: Dynamic viscosity (Pa.s)
    - cp: Specific heat (J/kgK)
    - k_fluid: Thermal conductivity (W/mK)
    - nx: Number of grid points in flow direction (Length)
    - ny: Number of grid points in transverse direction (Half-Gap) - We simulate half due to symmetry.
    - max_iter: Maximum iterations for the explicit solver

    Returns:
    - h_array: 1D array of heat transfer coefficient along the length (size nx).
    """

    if L <= 0 or H <= 0 or U_in <= 0:
        print("[NS Solver] Invalid geometry or velocity parameters.")
        return None

    # Kinematic viscosity and Thermal diffusivity
    nu = mu / rho
    alpha = k_fluid / (rho * cp)

    # Grid generation (Half channel due to symmetry)
    # x: flow direction (0 to L)
    # y: transverse direction (0 to H/2)
    H_half = H / 2.0
    dx = L / (nx - 1)
    dy = H_half / (ny - 1)

    # Initialize fields
    psi = np.zeros((nx, ny))
    omega = np.zeros((nx, ny))
    T = np.full((nx, ny), T_in)

    # Initial Guess for psi (Linear profile - plug flow / developed flow hybrid)
    # psi = U * y
    y_coords = np.linspace(0, H_half, ny)
    for i in range(nx):
        psi[i, :] = U_in * y_coords

    # Stability condition check (CFL) roughly
    dt = 0.5 * min(dx**2, dy**2) / max(nu, alpha)
    # Limit dt if velocity is high (CFL < 1 for advection)
    if U_in > 1e-9:
        dt_adv = min(dx, dy) / U_in
        dt = min(dt, dt_adv * 0.5)

    # Iteration loop
    for it in range(max_iter):
        wn = omega.copy()
        pn = psi.copy()
        Tn = T.copy()

        # 1. Solve Vorticity Transport Equation (Explicit)
        # u domega/dx + v domega/dy = nu * laplacian(omega)
        # u = dpsi/dy, v = -dpsi/dx

        # Compute velocities (Central difference interior)
        u = np.zeros((nx, ny))
        v = np.zeros((nx, ny))

        # Simple finite difference for u, v
        # u[i,j] = (psi[i, j+1] - psi[i, j-1]) / (2*dy)
        # v[i,j] = -(psi[i+1, j] - psi[i-1, j]) / (2*dx)
        # Using forward/backward at boundaries where needed, but mostly interior

        # Vectorized interior calculation
        u[1:-1, 1:-1] = (pn[1:-1, 2:] - pn[1:-1, :-2]) / (2 * dy)
        v[1:-1, 1:-1] = -(pn[2:, 1:-1] - pn[:-2, 1:-1]) / (2 * dx)

        # Laplacian of Omega
        d2w_dx2 = (wn[2:, 1:-1] - 2*wn[1:-1, 1:-1] + wn[:-2, 1:-1]) / dx**2
        d2w_dy2 = (wn[1:-1, 2:] - 2*wn[1:-1, 1:-1] + wn[1:-1, :-2]) / dy**2

        # Advection terms (Upwind or Central? Central is oscillatory for high Re. Let's use Upwind)
        # u dw/dx
        dw_dx = np.zeros_like(d2w_dx2)
        u_core = u[1:-1, 1:-1]
        v_core = v[1:-1, 1:-1]

        # Upwinding for dw/dx
        dw_dx = np.where(u_core > 0,
                         (wn[1:-1, 1:-1] - wn[:-2, 1:-1]) / dx,
                         (wn[2:, 1:-1] - wn[1:-1, 1:-1]) / dx)

        # Upwinding for dw/dy
        dw_dy = np.where(v_core > 0,
                         (wn[1:-1, 1:-1] - wn[1:-1, :-2]) / dy,
                         (wn[1:-1, 2:] - wn[1:-1, 1:-1]) / dy)

        # Update Omega (Interior)
        omega[1:-1, 1:-1] = wn[1:-1, 1:-1] + dt * (
            - (u_core * dw_dx + v_core * dw_dy) + nu * (d2w_dx2 + d2w_dy2)
        )

        # 2. Solve Streamfunction Equation (Poisson) -> SOR or Relaxation
        # Laplacian(psi) = -omega
        # Iterative solver for Poisson equation within the time step (or just one step of SOR)
        # Using a few SOR steps per time step to update psi
        for _ in range(5): # Small number of sub-iterations
            psi[1:-1, 1:-1] = 0.25 * (psi[2:, 1:-1] + psi[:-2, 1:-1] +
                                     psi[1:-1, 2:] + psi[1:-1, :-2] +
                                     dx*dy*omega[1:-1, 1:-1]) # Assuming dx approx dy for coeff, actually:

            # More accurate Poisson update for non-uniform grid aspect ratio:
            # (psi_xx + psi_yy = -omega)
            # psi_i,j = ( (psi_i+1 + psi_i-1)/dx^2 + (psi_j+1 + psi_j-1)/dy^2 + omega ) / (2/dx^2 + 2/dy^2)
            denom = 2/dx**2 + 2/dy**2
            psi[1:-1, 1:-1] = ( (psi[2:, 1:-1] + psi[:-2, 1:-1])/dx**2 +
                                (psi[1:-1, 2:] + psi[1:-1, :-2])/dy**2 +
                                omega[1:-1, 1:-1] ) / denom

        # 3. Solve Energy Equation
        # dT/dt + u dT/dx + v dT/dy = alpha * laplacian(T)
        d2T_dx2 = (Tn[2:, 1:-1] - 2*Tn[1:-1, 1:-1] + Tn[:-2, 1:-1]) / dx**2
        d2T_dy2 = (Tn[1:-1, 2:] - 2*Tn[1:-1, 1:-1] + Tn[1:-1, :-2]) / dy**2

        dT_dx = np.where(u_core > 0, (Tn[1:-1, 1:-1] - Tn[:-2, 1:-1]) / dx, (Tn[2:, 1:-1] - Tn[1:-1, 1:-1]) / dx)
        dT_dy = np.where(v_core > 0, (Tn[1:-1, 1:-1] - Tn[1:-1, :-2]) / dy, (Tn[1:-1, 2:] - Tn[1:-1, 1:-1]) / dy)

        T[1:-1, 1:-1] = Tn[1:-1, 1:-1] + dt * (
            - (u_core * dT_dx + v_core * dT_dy) + alpha * (d2T_dx2 + d2T_dy2)
        )

        # 4. Boundary Conditions

        # Inlet (x=0): u=U_in, v=0, T=T_in
        # psi = U_in * y
        psi[0, :] = U_in * y_coords
        omega[0, :] = 0 # Uniform inlet profile has zero vorticity
        T[0, :] = T_in

        # Outlet (x=L): Neumann zero for all
        psi[-1, :] = psi[-2, :]
        omega[-1, :] = omega[-2, :]
        T[-1, :] = T[-2, :]

        # Symmetry (y=0, index j=0): v=0, du/dy=0 -> psi=0, omega=0, dT/dy=0
        psi[:, 0] = 0
        omega[:, 0] = 0
        T[:, 0] = T[:, 1] # Adiabatic / Symmetry

        # Wall (y=H/2, index j=ny-1): u=0, v=0, T=T_wall
        # psi = Constant = U_in * H/2
        psi[:, -1] = U_in * H_half
        T[:, -1] = T_wall
        # Wall vorticity: omega_wall = -(d2psi/dy2) = -2*(psi_ny-1 - psi_ny-2) / dy^2  (using Taylor expansion with u_wall=0)
        omega[:, -1] = -2 * (psi[:, -1] - psi[:, -2]) / dy**2

        # Convergence check (simple)
        if it % 100 == 0:
            diff = np.max(np.abs(omega - wn))
            if diff < 1e-5:
                # print(f"[NS Solver] Converged at iteration {it}")
                break

    # Calculate Heat Transfer Coefficient h(x)
    # h = -k * (dT/dy)_wall / (T_wall - T_bulk)
    # T_bulk is approx T_in for this simple solver or simplified as the reference temp difference
    # Here we use the local definition often used in correlations: h based on (T_wall - T_fluid_local_bulk)
    # But for simplicity and matching the inputs of the main sim, we might use (T_wall - T_in) or similar.
    # Let's calculate local heat flux q" = -k * dT/dy

    # Gradient at wall (one-sided)
    # dT/dy at y=H/2 (j=-1)
    # T_wall is T[:, -1]
    # T_fluid_near_wall is T[:, -2]
    dT_dy_wall = (T[:, -1] - T[:, -2]) / dy
    q_flux = k_fluid * dT_dy_wall # flux into fluid (k * (Tw - Tfw)/dy)

    # h = q_flux / (T_wall - T_ref)
    # T_ref typically T_inlet or T_bulk. The main simulation usually expects h referenced to T_air (local or inlet).
    # Since the main simulation calculates T_air separately in 2.5D, we should probably provide h referenced to T_air_local.
    # In this solver, T represents the fluid temp.
    # T_bulk(x) calculation: Velocity-weighted average
    # T_bulk = int(u*T dy) / int(u dy)
    # u is (nx, ny), T is (nx, ny). Sum over axis 1 (y-direction)
    # Note: we should strictly integrate, but sum is fine for uniform grid

    numerator = np.sum(u * T, axis=1)
    denominator = np.sum(u, axis=1)

    # Handle zero velocity case (avoid division by zero)
    denominator = np.where(denominator < 1e-9, 1.0, denominator)
    T_bulk = numerator / denominator

    # Fallback if velocity was zero (use simple mean)
    T_bulk_simple = np.mean(T, axis=1)
    T_bulk = np.where(np.sum(u, axis=1) < 1e-9, T_bulk_simple, T_bulk)

    # Avoid division by zero
    delta_T = T_wall - T_bulk
    delta_T = np.where(np.abs(delta_T) < 1e-3, 1.0, delta_T)

    h_local = q_flux / delta_T

    # Clip values to reasonable range
    h_local = np.clip(h_local, 2.0, 500.0)

    return h_local

if __name__ == "__main__":
    # Simple test
    print("Testing NS solver...")
    h_res = solve_channel_flow_2d(L=0.2, H=0.01, U_in=2.0, T_in=25, T_wall=60,
                                  rho=1.16, mu=1.85e-5, cp=1007, k_fluid=0.026)
    print("Result shape:", h_res.shape if h_res is not None else "None")
    print("Avg h:", np.mean(h_res) if h_res is not None else "None")

"""

        u = -K (z - z_eq)      with     z = [x, th1, th2, th3, xd, th1d, th2d, th3d]^T

where K is computed by solving the continuous-time LQR problem for the
system linearized about the (unstable) upright equilibrium.

Everything a user typically wants to change lives in the PARAMETERS block
below: masses, lengths, LQR weights, actuator saturation and an external
disturbance force (a "kick" or a sustained push) applied to the cart.

Sections:
  1. Symbolic model  (derive.py)              -> mass matrix M(q), forcing F
  2. Numeric nonlinear dynamics                -> qddot = M(q)^-1 F(q,qdot,u,Fd)
  3. Linearization about the upright position  -> A, B  (finite differences)
  4. LQR gain                                  -> K = R^-1 B^T P
  5. Closed-loop nonlinear simulation          -> scipy.integrate.solve_ivp
  6. Plots: cart position, angles, control effort, and an animation (gif)
=============================================================================
"""

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

from derive import derive

# Output files are written next to this script, in an "outputs" folder that
# is created automatically if it doesn't exist yet.
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTDIR, exist_ok=True)

# =============================================================================
# 1. PARAMETERS  -- edit these to explore different physical configurations
# =============================================================================
PARAMS = dict(
    M=2.0,     # cart mass                 [kg]
    m1=0.3,    # pendulum-1 (bottom) mass  [kg]
    m2=0.2,    # pendulum-2 (middle) mass  [kg]
    m3=0.1,    # pendulum-3 (top) mass     [kg]
    l1=0.5,    # pendulum-1 length         [m]
    l2=0.4,    # pendulum-2 length         [m]
    l3=0.3,    # pendulum-3 length         [m]
    g=9.81,    # gravitational acceleration [m/s^2]
    b0=0.5,    # cart viscous friction      [N.s/m]
    b1=0.02,   # joint-1 viscous friction   [N.m.s]
    b2=0.02,   # joint-2 viscous friction   [N.m.s]
    b3=0.02,   # joint-3 viscous friction   [N.m.s]
)

# LQR weights: state = [x, th1, th2, th3, xd, th1d, th2d, th3d]
# NOTE ON TUNING A TRIPLE INVERTED PENDULUM: the open-loop upright equilibrium
# is *very* unstable (growth rate of several rad/s for these dimensions), so
# even a well-designed LQR needs a large, fast corrective force during the
# first ~0.3-0.5 s -- a few hundred Newtons is normal for this benchmark, not
# a sign of a bad design. If U_MAX is set too low, the controller saturates
# for too long, the state leaves the region where the linearization is valid,
# and the pendulums can fall over instead of recovering (this is exactly what
# was happening in the first version of this script -- always sanity check
# your U_MAX against the peak force the closed loop actually demands, e.g.
# with the linear model: u(t) = -K @ expm((A-B@K)*t) @ z0).
Q_LQR = np.diag([2.0, 150.0, 300.0, 450.0, 1.0, 80.0, 150.0, 220.0])
R_LQR = np.array([[0.02]])           # control-effort penalty (smaller -> faster, more aggressive control)

# Actuator saturation (set to None to disable). Keep this comfortably above
# the peak force the gain K actually demands for your chosen initial
# condition / disturbance (the script prints a warning if it saturates for
# more than a brief instant).
U_MAX = 650.0          # [N]

# Initial condition: small angular offsets from upright (radians), everything else at rest
INITIAL_STATE = np.array([0.0,               # x
                           np.deg2rad(2.0),   # theta1
                           np.deg2rad(-1.5),  # theta2
                           np.deg2rad(2.0),   # theta3
                           0.0, 0.0, 0.0, 0.0])

# External disturbance: a horizontal "kick" force applied to the CART
DIST_MAGNITUDE = 6.0    # [N]  (set to 0.0 to disable)
DIST_START = 2.0        # [s]
DIST_DURATION = 0.15    # [s]  (short pulse = an impulsive kick)

T_FINAL = 8.0            # simulation duration [s]
N_POINTS = 1600          # number of time samples for plotting/animation


# =============================================================================
# 2. BUILD THE NUMERIC NONLINEAR MODEL FROM THE SYMBOLIC DERIVATION
# =============================================================================
def build_model(print_equations=True):
    eqs, A_mat, b_vec, s = derive()

    if print_equations:
        print("=" * 78)
        print(" NONLINEAR EQUATIONS OF MOTION (Euler-Lagrange)")
        print("=" * 78)
        names = ["d/dt(dL/dxdot)   - dL/dx      = u + Fd - b0*xdot",
                 "d/dt(dL/dth1dot) - dL/dtheta1  = -b1*theta1dot",
                 "d/dt(dL/dth2dot) - dL/dtheta2  = -b2*theta2dot",
                 "d/dt(dL/dth3dot) - dL/dtheta3  = -b3*theta3dot"]
        for n, e in zip(names, eqs):
            print("\n  " + n + "  ->")
            sp.pprint(sp.simplify(e.lhs))
        print("\n" + "-" * 78)
        print(" Manipulator (mass-matrix) form:   M(q) * qddot = F(q, qdot, u, Fd)")
        print("-" * 78)
        print("\n M(q) =")
        sp.pprint(A_mat)
        print("\n F(q,qdot,u,Fd) =")
        sp.pprint(b_vec)
        print("=" * 78 + "\n")

    arg_order_M = (s['xs'], s['th1s'], s['th2s'], s['th3s'],
                    s['M'], s['m1'], s['m2'], s['m3'], s['l1'], s['l2'], s['l3'], s['g'])
    arg_order_F = (s['xs'], s['th1s'], s['th2s'], s['th3s'],
                    s['xds'], s['th1ds'], s['th2ds'], s['th3ds'],
                    s['us'], s['fds'],
                    s['M'], s['m1'], s['m2'], s['m3'], s['l1'], s['l2'], s['l3'], s['g'],
                    s['b0'], s['b1'], s['b2'], s['b3'])

    M_func = sp.lambdify(arg_order_M, A_mat, modules='numpy')
    F_func = sp.lambdify(arg_order_F, b_vec, modules='numpy')
    return M_func, F_func


def make_dynamics(M_func, F_func, params):
    p = (params['M'], params['m1'], params['m2'], params['m3'],
         params['l1'], params['l2'], params['l3'], params['g'])
    pb = (params['b0'], params['b1'], params['b2'], params['b3'])

    def qddot(q, qd, u, Fd):
        Mnum = np.array(M_func(*q, *p), dtype=float)
        Fnum = np.array(F_func(*q, *qd, u, Fd, *p, *pb), dtype=float).flatten()
        return np.linalg.solve(Mnum, Fnum)

    def f(z, u, Fd):
        """state derivative for state z=[x,th1,th2,th3,xd,th1d,th2d,th3d]"""
        q, qd = z[:4], z[4:]
        acc = qddot(q, qd, u, Fd)
        return np.concatenate([qd, acc])

    return f


# =============================================================================
# 3. LINEARIZATION ABOUT THE UPRIGHT EQUILIBRIUM (central finite differences)
# =============================================================================
def linearize(f, z_eq, u_eq=0.0, eps=1e-6):
    n = len(z_eq)
    A = np.zeros((n, n))
    B = np.zeros((n, 1))
    for i in range(n):
        dz = np.zeros(n); dz[i] = eps
        A[:, i] = (f(z_eq + dz, u_eq, 0.0) - f(z_eq - dz, u_eq, 0.0)) / (2 * eps)
    B[:, 0] = (f(z_eq, u_eq + eps, 0.0) - f(z_eq, u_eq - eps, 0.0)) / (2 * eps)
    return A, B


def lqr(A, B, Qw, Rw):
    P = solve_continuous_are(A, B, Qw, Rw)
    K = np.linalg.solve(Rw, B.T @ P)
    return K, P


# =============================================================================
# 4. DISTURBANCE FORCE PROFILE
# =============================================================================
def disturbance(t, magnitude=DIST_MAGNITUDE, start=DIST_START, duration=DIST_DURATION):
    if magnitude != 0.0 and start <= t <= start + duration:
        return magnitude
    return 0.0


# =============================================================================
# 5. SIMULATION
# =============================================================================
def simulate(f, K, z_eq, u_max=U_MAX):
    def controller(z):
        u = float((-K @ (z - z_eq)).item())
        if u_max is not None:
            u = np.clip(u, -u_max, u_max)
        return u

    u_history = {'t': [], 'u': []}

    def rhs(t, z):
        u = controller(z)
        Fd = disturbance(t)
        u_history['t'].append(t)
        u_history['u'].append(u)
        return f(z, u, Fd)

    t_eval = np.linspace(0, T_FINAL, N_POINTS)
    sol = solve_ivp(rhs, [0, T_FINAL], INITIAL_STATE, t_eval=t_eval,
                     method='RK45', max_step=0.01, rtol=1e-8, atol=1e-9)

    if u_max is not None:
        u_hist = np.array([controller(sol.y[:, k]) for k in range(len(sol.t))])
        sat_fraction = np.mean(np.isclose(np.abs(u_hist), u_max))
        if sat_fraction > 0.05:
            print(f"\n*** WARNING: controller saturated at +/-{u_max} N for "
                  f"{sat_fraction*100:.1f}% of the simulated samples. "
                  f"Consider raising U_MAX, softening Q_LQR/R_LQR, or reducing "
                  f"the initial tilt / disturbance magnitude.")

    return sol, controller


# =============================================================================
# 6. PLOTS
# =============================================================================
def plot_results(sol, params, controller):
    t = sol.t
    x, th1, th2, th3 = sol.y[0], sol.y[1], sol.y[2], sol.y[3]
    xd = sol.y[4]
    u_vals = np.array([controller(sol.y[:, k]) for k in range(len(t))])
    Fd_vals = np.array([disturbance(tt) for tt in t])

    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)

    axes[0].plot(t, x, label='cart position x(t)', color='tab:blue')
    axes[0].plot(t, xd, label='cart velocity xdot(t)', color='tab:cyan', alpha=0.7)
    axes[0].set_ylabel('x [m], xdot [m/s]')
    axes[0].set_title('Cart position / velocity')
    axes[0].legend(loc='upper right')
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, np.rad2deg(th1), label=r'$\theta_1$ (bottom)')
    axes[1].plot(t, np.rad2deg(th2), label=r'$\theta_2$ (middle)')
    axes[1].plot(t, np.rad2deg(th3), label=r'$\theta_3$ (top)')
    axes[1].axhline(0, color='k', lw=0.5)
    axes[1].set_ylabel('angle [deg]')
    axes[1].set_title('Pendulum angles vs time (0 deg = upright)')
    axes[1].legend(loc='upper right')
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, u_vals, color='tab:red', label='control force u(t)')
    axes[2].plot(t, Fd_vals, color='tab:orange', label='disturbance Fd(t)', linestyle='--')
    axes[2].set_ylabel('Force [N]')
    axes[2].set_title('Control effort and disturbance')
    axes[2].legend(loc='upper right')
    axes[2].grid(alpha=0.3)

    axes[3].plot(t, sol.y[5], label=r'$\dot\theta_1$')
    axes[3].plot(t, sol.y[6], label=r'$\dot\theta_2$')
    axes[3].plot(t, sol.y[7], label=r'$\dot\theta_3$')
    axes[3].set_ylabel('angular rate [rad/s]')
    axes[3].set_xlabel('time [s]')
    axes[3].set_title('Angular velocities')
    axes[3].legend(loc='upper right')
    axes[3].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'pendulum_timeseries.png'), dpi=140)
    plt.close(fig)


def animate_cart(sol, params, filename=None,
                  n_frames=200, fps=25):
    if filename is None:
        filename = os.path.join(OUTDIR, 'pendulum_animation.gif')
    t = sol.t
    idx = np.linspace(0, len(t) - 1, n_frames).astype(int)
    x = sol.y[0][idx]
    th1 = sol.y[1][idx]
    th2 = sol.y[2][idx]
    th3 = sol.y[3][idx]
    tt = t[idx]

    l1, l2, l3 = params['l1'], params['l2'], params['l3']

    def joints(xc, a1, a2, a3):
        p0 = np.array([xc, 0.0])
        p1 = p0 + np.array([l1 * np.sin(a1), l1 * np.cos(a1)])
        p2 = p1 + np.array([l2 * np.sin(a2), l2 * np.cos(a2)])
        p3 = p2 + np.array([l3 * np.sin(a3), l3 * np.cos(a3)])
        return p0, p1, p2, p3

    L = l1 + l2 + l3
    xmin, xmax = x.min() - L - 0.3, x.max() + L + 0.3

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.2, L + 0.3)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Triple Inverted Pendulum Simulation')

    rail, = ax.plot([xmin, xmax], [0, 0], color='gray', lw=1)
    cart_w, cart_h = 0.3, 0.15
    cart_patch = plt.Rectangle((x[0] - cart_w / 2, -cart_h / 2), cart_w, cart_h,
                                color='tab:blue')
    ax.add_patch(cart_patch)
    line, = ax.plot([], [], 'o-', lw=2.5, color='tab:red', markersize=6)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, cart_patch, time_text

    def update(frame):
        p0, p1, p2, p3 = joints(x[frame], th1[frame], th2[frame], th3[frame])
        xs = [p0[0], p1[0], p2[0], p3[0]]
        ys = [p0[1], p1[1], p2[1], p3[1]]
        line.set_data(xs, ys)
        cart_patch.set_xy((x[frame] - cart_w / 2, -cart_h / 2))
        time_text.set_text(f't = {tt[frame]:.2f} s')
        return line, cart_patch, time_text

    anim = animation.FuncAnimation(fig, update, frames=len(idx), init_func=init,
                                    interval=1000 / fps, blit=True)
    anim.save(filename, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def main():
    M_func, F_func = build_model(print_equations=True)
    f = make_dynamics(M_func, F_func, PARAMS)

    z_eq = np.zeros(8)   # upright, at rest, cart at x=0
    A, B = linearize(f, z_eq)

    print("Linearized state matrix A (about the upright equilibrium):")
    print(np.round(A, 3))
    print("\nLinearized input matrix B:")
    print(np.round(B, 3))

    # controllability check
    n = A.shape[0]
    ctrb = np.hstack([np.linalg.matrix_power(A, i) @ B for i in range(n)])
    rank = np.linalg.matrix_rank(ctrb)
    print(f"\nControllability matrix rank: {rank} / {n} "
          f"({'controllable' if rank == n else 'NOT fully controllable!'})")

    K, P = lqr(A, B, Q_LQR, R_LQR)
    print("\nLQR gain matrix K:")
    print(np.round(K, 3))

    print("\nClosed-loop eigenvalues (A - B K):")
    print(np.round(np.linalg.eigvals(A - B @ K), 3))

    sol, controller = simulate(f, K, z_eq)

    if not sol.success:
        print("\n*** Integration did not fully succeed:", sol.message)

    plot_results(sol, PARAMS, controller)
    animate_cart(sol, PARAMS)

    print("\nDone. Plots written to:")
    print(f"  {os.path.join(OUTDIR, 'pendulum_timeseries.png')}")
    print(f"  {os.path.join(OUTDIR, 'pendulum_animation.gif')}")


if __name__ == "__main__":
    main()


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# --- this is what "runs the other files" -----------------------------------
import triple_pendulum_lqr as tp   # runs derive.py internally via build_model()
OUTDIR = tp.OUTDIR   # same "outputs" folder next to the scripts (auto-created)
 

def run_simulations():
    print("Building symbolic model and lambdified dynamics ...")
    M_func, F_func = tp.build_model(print_equations=False)
    f = tp.make_dynamics(M_func, F_func, tp.PARAMS)

    z_eq = np.zeros(8)
    A, B = tp.linearize(f, z_eq)
    K, _ = tp.lqr(A, B, tp.Q_LQR, tp.R_LQR)
    print("LQR gain K =", np.round(K, 3))

    print("Running CONTROLLED simulation ...")
    sol_ctrl, controller = tp.simulate(f, K, z_eq)

    print("Running UNCONTROLLED simulation (u = 0) ...")
    def rhs_open_loop(t, z):
        return f(z, 0.0, tp.disturbance(t))

    t_eval = np.linspace(0, tp.T_FINAL, tp.N_POINTS)
    sol_free = solve_ivp(rhs_open_loop, [0, tp.T_FINAL], tp.INITIAL_STATE,
                          t_eval=t_eval, method='RK45', max_step=0.01,
                          rtol=1e-8, atol=1e-9)

    return sol_ctrl, sol_free


# =============================================================================
# 2. Animation helpers
# =============================================================================
def _joints(xc, a1, a2, a3, l1, l2, l3):
    p0 = np.array([xc, 0.0])
    p1 = p0 + np.array([l1 * np.sin(a1), l1 * np.cos(a1)])
    p2 = p1 + np.array([l2 * np.sin(a2), l2 * np.cos(a2)])
    p3 = p2 + np.array([l3 * np.sin(a3), l3 * np.cos(a3)])
    return p0, p1, p2, p3


def _sample(sol, n_frames):
    idx = np.linspace(0, len(sol.t) - 1, n_frames).astype(int)
    return (sol.t[idx], sol.y[0][idx], sol.y[1][idx], sol.y[2][idx], sol.y[3][idx])


def animate_single(sol, params, title, filename, n_frames=200, fps=25):
    t, x, th1, th2, th3 = _sample(sol, n_frames)
    l1, l2, l3 = params['l1'], params['l2'], params['l3']
    L = l1 + l2 + l3
    xmin, xmax = x.min() - L - 0.3, x.max() + L + 0.3

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-(L + 0.3), L + 0.3)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_title(title)
    ax.plot([xmin, xmax], [0, 0], color='gray', lw=1)

    cart_w, cart_h = 0.3, 0.15
    cart_patch = plt.Rectangle((x[0] - cart_w / 2, -cart_h / 2), cart_w, cart_h, color='tab:blue')
    ax.add_patch(cart_patch)
    line, = ax.plot([], [], 'o-', lw=2.5, color='tab:red', markersize=6)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, cart_patch, time_text

    def update(frame):
        p0, p1, p2, p3 = _joints(x[frame], th1[frame], th2[frame], th3[frame], l1, l2, l3)
        line.set_data([p0[0], p1[0], p2[0], p3[0]], [p0[1], p1[1], p2[1], p3[1]])
        cart_patch.set_xy((x[frame] - cart_w / 2, -cart_h / 2))
        time_text.set_text(f't = {t[frame]:.2f} s')
        return line, cart_patch, time_text

    anim = animation.FuncAnimation(fig, update, frames=len(t), init_func=init,
                                    interval=1000 / fps, blit=True)
    anim.save(filename, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  saved {filename}")


def animate_comparison(sol_ctrl, sol_free, params, filename, n_frames=200, fps=25):
    tC, xC, th1C, th2C, th3C = _sample(sol_ctrl, n_frames)
    tF, xF, th1F, th2F, th3F = _sample(sol_free, n_frames)
    l1, l2, l3 = params['l1'], params['l2'], params['l3']
    L = l1 + l2 + l3

    xmin_c, xmax_c = xC.min() - L - 0.3, xC.max() + L + 0.3
    xmin_f, xmax_f = xF.min() - L - 0.3, xF.max() + L + 0.3

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6))
    for ax, ttl, xmin, xmax in [(axL, 'LQR CONTROLLED', xmin_c, xmax_c),
                                 (axR, 'NO CONTROL (u = 0)', xmin_f, xmax_f)]:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-(L + 0.3), L + 0.3)
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        ax.set_xlabel('x [m]')
        ax.set_title(ttl)
        ax.plot([xmin, xmax], [0, 0], color='gray', lw=1)
    axL.set_ylabel('y [m]')
    fig.suptitle('Triple Inverted Pendulum on a Cart: controlled vs. uncontrolled')

    cart_w, cart_h = 0.3, 0.15
    cartL = plt.Rectangle((xC[0] - cart_w / 2, -cart_h / 2), cart_w, cart_h, color='tab:blue')
    cartR = plt.Rectangle((xF[0] - cart_w / 2, -cart_h / 2), cart_w, cart_h, color='tab:blue')
    axL.add_patch(cartL); axR.add_patch(cartR)
    lineL, = axL.plot([], [], 'o-', lw=2.5, color='tab:green', markersize=6)
    lineR, = axR.plot([], [], 'o-', lw=2.5, color='tab:red', markersize=6)
    timeL = axL.text(0.02, 0.95, '', transform=axL.transAxes)
    timeR = axR.text(0.02, 0.95, '', transform=axR.transAxes)

    def init():
        lineL.set_data([], []); lineR.set_data([], [])
        timeL.set_text(''); timeR.set_text('')
        return lineL, lineR, cartL, cartR, timeL, timeR

    def update(frame):
        p0, p1, p2, p3 = _joints(xC[frame], th1C[frame], th2C[frame], th3C[frame], l1, l2, l3)
        lineL.set_data([p0[0], p1[0], p2[0], p3[0]], [p0[1], p1[1], p2[1], p3[1]])
        cartL.set_xy((xC[frame] - cart_w / 2, -cart_h / 2))
        timeL.set_text(f't = {tC[frame]:.2f} s')

        q0, q1, q2, q3 = _joints(xF[frame], th1F[frame], th2F[frame], th3F[frame], l1, l2, l3)
        lineR.set_data([q0[0], q1[0], q2[0], q3[0]], [q0[1], q1[1], q2[1], q3[1]])
        cartR.set_xy((xF[frame] - cart_w / 2, -cart_h / 2))
        timeR.set_text(f't = {tF[frame]:.2f} s')
        return lineL, lineR, cartL, cartR, timeL, timeR

    anim = animation.FuncAnimation(fig, update, frames=n_frames, init_func=init,
                                    interval=1000 / fps, blit=True)
    fig.tight_layout()
    anim.save(filename, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  saved {filename}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    sol_ctrl, sol_free = run_simulations()

    print("\nGenerating animations ...")
    animate_single(sol_ctrl, tp.PARAMS, 'Triple Inverted Pendulum - LQR CONTROLLED',
                    os.path.join(OUTDIR, 'pendulum_controlled_animation.gif'))
    animate_single(sol_free, tp.PARAMS, 'Triple Inverted Pendulum - NO CONTROL',
                    os.path.join(OUTDIR, 'pendulum_uncontrolled_animation.gif'))
    animate_comparison(sol_ctrl, sol_free, tp.PARAMS,
                        os.path.join(OUTDIR, 'pendulum_comparison_animation.gif'))

    print("\nDone.")


if __name__ == "__main__":
    main()

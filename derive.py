"""
Symbolic derivation of the equations of motion of a triple inverted pendulum
mounted on a cart (single actuator: horizontal force u on the cart).

Generalized coordinates:  q = [x, theta1, theta2, theta3]
    x       : cart position along the rail
    theta_i : ABSOLUTE angle of pendulum i measured from the upward vertical
              (theta_i = 0  ->  pendulum i points straight up)

The model treats each pendulum as a point mass m_i at the tip of a massless
rigid rod of length l_i, hinged in series (theta_i is an absolute angle, not
relative to the previous link).

Equations are derived with the Euler-Lagrange method and returned in the
"manipulator" (mass-matrix) form:

        M(q) * qddot = F(q, qdot) + [u + Fd, 0, 0, 0]^T

where u is the control force applied to the cart and Fd an external
disturbance force applied to the cart (both along x).
"""

import sympy as sp

def derive():
    t = sp.symbols('t')
    g = sp.symbols('g', positive=True)
    M, m1, m2, m3 = sp.symbols('M m1 m2 m3', positive=True)
    l1, l2, l3 = sp.symbols('l1 l2 l3', positive=True)
    b0, b1, b2, b3 = sp.symbols('b0 b1 b2 b3', nonnegative=True)

    x = sp.Function('x')(t)
    th1 = sp.Function('theta1')(t)
    th2 = sp.Function('theta2')(t)
    th3 = sp.Function('theta3')(t)
    u = sp.Function('u')(t)
    Fd = sp.Function('Fd')(t)

    xdot, th1dot, th2dot, th3dot = (sp.diff(q, t) for q in (x, th1, th2, th3))

    # --- Cartesian positions of the three point masses -------------------
    x1 = x + l1 * sp.sin(th1)
    y1 = l1 * sp.cos(th1)
    x2 = x1 + l2 * sp.sin(th2)
    y2 = y1 + l2 * sp.cos(th2)
    x3 = x2 + l3 * sp.sin(th3)
    y3 = y2 + l3 * sp.cos(th3)

    def vel(expr):
        return sp.diff(expr, t)

    x1d, y1d = vel(x1), vel(y1)
    x2d, y2d = vel(x2), vel(y2)
    x3d, y3d = vel(x3), vel(y3)

    # --- Kinetic / potential energy ---------------------------------------
    T = (sp.Rational(1, 2) * M * xdot**2
         + sp.Rational(1, 2) * m1 * (x1d**2 + y1d**2)
         + sp.Rational(1, 2) * m2 * (x2d**2 + y2d**2)
         + sp.Rational(1, 2) * m3 * (x3d**2 + y3d**2))

    V = m1 * g * y1 + m2 * g * y2 + m3 * g * y3

    L = sp.expand_trig(sp.simplify(T - V))

    qs = [x, th1, th2, th3]
    qdots = [xdot, th1dot, th2dot, th3dot]
    # generalized forces: control + disturbance act on the cart DOF only,
    # each DOF also has a (small) linear viscous damping term
    Q = [u + Fd - b0 * xdot, -b1 * th1dot, -b2 * th2dot, -b3 * th3dot]

    eqs = []
    for qi, qidot, Qi in zip(qs, qdots, Q):
        dL_dqdot = sp.diff(L, qidot)
        ddt = sp.diff(dL_dqdot, t)
        dL_dq = sp.diff(L, qi)
        eqs.append(sp.Eq(ddt - dL_dq, Qi))

    # --- Replace time-functions/derivatives with plain algebraic symbols --
    xs, th1s, th2s, th3s = sp.symbols('x theta1 theta2 theta3')
    xds, th1ds, th2ds, th3ds = sp.symbols('xdot theta1dot theta2dot theta3dot')
    xdds, th1dds, th2dds, th3dds = sp.symbols('xddot theta1ddot theta2ddot theta3ddot')
    us, fds = sp.symbols('u Fd')

    subs_dd = {sp.Derivative(x, (t, 2)): xdds, sp.Derivative(th1, (t, 2)): th1dds,
               sp.Derivative(th2, (t, 2)): th2dds, sp.Derivative(th3, (t, 2)): th3dds}
    subs_d = {sp.Derivative(x, t): xds, sp.Derivative(th1, t): th1ds,
              sp.Derivative(th2, t): th2ds, sp.Derivative(th3, t): th3ds}
    subs_0 = {x: xs, th1: th1s, th2: th2s, th3: th3s, u: us, Fd: fds}

    exprs = []
    for eq in eqs:
        e = (eq.lhs - eq.rhs).subs(subs_dd).subs(subs_d).subs(subs_0)
        exprs.append(sp.expand(e))

    accvec = [xdds, th1dds, th2dds, th3dds]
    A_mat, b_vec = sp.linear_eq_to_matrix(exprs, accvec)
    # exprs == A_mat*accvec - b_vec  =>  M(q) qddot = b_vec
    A_mat = sp.simplify(A_mat)
    b_vec = sp.simplify(b_vec)

    syms = dict(xs=xs, th1s=th1s, th2s=th2s, th3s=th3s,
                xds=xds, th1ds=th1ds, th2ds=th2ds, th3ds=th3ds,
                us=us, fds=fds,
                M=M, m1=m1, m2=m2, m3=m3, l1=l1, l2=l2, l3=l3, g=g,
                b0=b0, b1=b1, b2=b2, b3=b3)

    return eqs, A_mat, b_vec, syms


if __name__ == "__main__":
    eqs, A_mat, b_vec, syms = derive()
    print("Euler-Lagrange equations (nonlinear):")
    for e in eqs:
        sp.pprint(e)
        print()
    print("\nMass matrix M(q):")
    sp.pprint(A_mat)
    print("\nRHS vector F(q,qdot,u,Fd):")
    sp.pprint(b_vec)

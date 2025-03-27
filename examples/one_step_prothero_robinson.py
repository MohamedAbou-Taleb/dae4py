import numpy as np
from dae4py.irk import solve_dae_IRK_generic
from dae4py.butcher_tableau import radau_tableau, gauss_legendre_tableau


"""
References
----------
Hairer2015: https://doi.org/10.1007/978-3-540-70529-1_139
"""
la = -20


def g(t):
    return np.arctan(2 * t)


def g_dot(t):
    return 2 / (1 + (2 * t) ** 2)


def F(t, y, yp):
    return yp - (la * (y - g(t)) + g_dot(t))


def true_sol(t):
    return g(t), g_dot(t)


if __name__ == "__main__":
    # time span
    t0 = -1.9
    t1 = 1.5
    t_span = (t0, t1)

    # initial conditions
    y0, yp0 = true_sol(t0)

    # solver options
    h = 3.4
    atol = rtol = 1e-14

    # Radau method
    s = 2
    atol = rtol = 1e-14

    # single step of RadauIIA(2)
    A, b, c, p, q = radau_tableau(s)
    sol_RadauIIA2 = solve_dae_IRK_generic(
        F, y0, yp0, t_span, h, A, b, c, atol=atol, rtol=rtol
    )

    np.savetxt(
        "RadauIIA(2)_prothero_robinson.txt",
        np.array(
            [
                [t0, t1],
                [y0, sol_RadauIIA2.y[-1][0]],
                [t0, t0 + c[0] * h],
                [t0, t0 + c[1] * h],
                [y0, sol_RadauIIA2.Y[-1][0][0]],
                [y0, sol_RadauIIA2.Y[-1][1][0]],
                [yp0, sol_RadauIIA2.Yp[-1][0][0]],
                [yp0, sol_RadauIIA2.Yp[-1][1][0]],
                [h, h],
            ]
        ).T,
        header="t, y, tau1, tau2, Y1, Y2, Y_dot1, Y_dot2, h",
        delimiter=", ",
        comments="",
    )

    # single step of GaussLegendre(2)
    A, b, c, p, q = gauss_legendre_tableau(s)
    sol_GaussLegendre2 = solve_dae_IRK_generic(
        F, y0, yp0, t_span, h, A, b, c, atol=atol, rtol=rtol
    )

    np.savetxt(
        "GaussLegendre(2)_prothero_robinson.txt",
        np.array(
            [
                [t0, t1],
                [y0, sol_GaussLegendre2.y[-1][0]],
                [t0, t0 + c[0] * h],
                [t0, t0 + c[1] * h],
                [y0, sol_GaussLegendre2.Y[-1][0][0]],
                [y0, sol_GaussLegendre2.Y[-1][1][0]],
                [yp0, sol_GaussLegendre2.Yp[-1][0][0]],
                [yp0, sol_GaussLegendre2.Yp[-1][1][0]],
                [h, h],
            ]
        ).T,
        header="t, y, tau1, tau2, Y1, Y2, Y_dot1, Y_dot2, h",
        delimiter=", ",
        comments="",
    )

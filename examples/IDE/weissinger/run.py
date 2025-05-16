import numpy as np
import matplotlib.pyplot as plt
from dae4py.irk import solve_dae_IRK
from dae4py.bdf import solve_dae_BDF
from dae4py.butcher_tableau import radau_tableau, gauss_legendre_tableau
from dae4py.radau import solve_dae_radau
from weissinger import problem


def trajectory(s=None, tableau=None):
    F = problem.F
    t_span = problem.t_span
    y0 = problem.y0
    yp0 = problem.yp0

    # solver options
    h = 1e-1
    atol = rtol = 1e-6
    if s is None or tableau is None:
        sol = solve_dae_BDF(F, y0, yp0, t_span, h, atol=atol, rtol=rtol)
    else:
        sol = solve_dae_IRK(F, y0, yp0, t_span, h, tableau(s), atol=atol, rtol=rtol)
    t = sol.t
    y = sol.y
    yp = sol.yp

    # visualization
    fig, ax = plt.subplots(2)
    y_true, yp_true = problem.true_sol(t)

    ax[0].plot(t, y[:, 0], "-k", label=f"y")
    ax[0].plot(t, y_true, "rx", label=f"y true")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(t, yp[:, 0], "-k", label=f"yp")
    ax[1].plot(t, yp_true, "rx", label=f"yp true")
    ax[1].grid()
    ax[1].legend()

    plt.show()


def adaptive_radau_IIA(s=3):
    F = problem.F
    t_span = problem.t_span
    y0 = problem.y0
    yp0 = problem.yp0

    # solver options
    h0 = 1e-2
    atol = rtol = 1e-6
    sol = solve_dae_radau(F, y0, yp0, t_span, h0, s=s, atol=atol, rtol=rtol)
    print(sol)
    t = sol.t
    y = sol.y
    yp = sol.yp

    # visualization
    fig, ax = plt.subplots(3)
    y_true, yp_true = problem.true_sol(t)

    ax[0].plot(t, y[:, 0], "-k", label=f"y")
    ax[0].plot(t, y_true, "rx", label=f"y true")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(t, yp[:, 0], "-k", label=f"yp")
    ax[1].plot(t, yp_true, "rx", label=f"yp true")
    ax[1].grid()
    ax[1].legend()

    ax[2].plot(t[1:], np.diff(t), "-k", label=f"h")
    ax[2].grid()
    ax[2].legend()
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")

    plt.show()


if __name__ == "__main__":
    trajectory()  # BDF case
    trajectory(s=2, tableau=gauss_legendre_tableau)
    trajectory(s=2, tableau=radau_tableau)

    adaptive_radau_IIA(s=3)

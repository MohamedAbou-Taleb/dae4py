import numpy as np
import matplotlib.pyplot as plt
from dae4py.irk import solve_dae_IRK
from dae4py.bdf import solve_dae_BDF
from dae4py.butcher_tableau import radau_tableau, gauss_legendre_tableau
from dae4py.radau import solve_dae_radau
from robertson import problem


def trajectory(s=None, tableau=None):
    F = problem.F
    t_span = problem.t_span
    y0 = problem.y0
    yp0 = problem.yp0

    # solver options
    h = 1e-4
    atol = rtol = 1e-6
    if s is None or tableau is None:
        sol = solve_dae_BDF(F, y0, yp0, t_span, h, atol=atol, rtol=rtol)
    else:
        sol = solve_dae_IRK(F, y0, yp0, t_span, h, tableau(s), atol=atol, rtol=rtol)
    t = sol.t
    y = sol.y

    # visualization
    n = len(y0)
    fig, ax = plt.subplots(2)

    for i in range(n):
        if i == 1:
            ax[0].plot(t, y[:, i] * 1e4, "-", label=f"y{i+1} * 1e4")
        else:
            ax[0].plot(t, y[:, i], "-", label=f"y{i+1}")

    ax[0].grid()
    ax[0].legend()
    ax[0].set_xscale("log")

    plt.show()


def adaptive_radau_IIA(s=3):
    F = problem.F
    t_span = problem.t_span
    y0 = problem.y0
    yp0 = problem.yp0

    # solver options
    t_eval = np.logspace(-10, 6, num=200)
    h0 = 1e-8
    atol = 1e-8
    rtol = 1e-8
    sol = solve_dae_radau(
        F, y0, yp0, t_span, h0, s=s, atol=atol, rtol=rtol, t_eval=t_eval
    )
    print(sol)
    t = sol.t
    y = sol.y
    yp = sol.yp

    # visualization
    n = len(y0)
    fig, ax = plt.subplots(2, 2)

    for i in range(n):
        ax[0, 1].plot(t, yp[:, i], "-", label=f"yp{i+1}")
        ax[0, 1].plot(sol.t_eval, sol.yp_eval[:, i], "-x", label=f"yp{i+1} eval")
        if i == 1:
            ax[0, 0].plot(t, y[:, i] * 1e4, "-", label=f"y{i+1} * 1e4")
            ax[0, 0].plot(
                sol.t_eval, sol.y_eval[:, i] * 1e4, "--x", label=f"y{i+1} * 1e4 eval"
            )
        else:
            ax[0, 0].plot(t, y[:, i], "-", label=f"y{i+1}")
            ax[0, 0].plot(sol.t_eval, sol.y_eval[:, i], "-x", label=f"y{i+1} eval")

    ax[0, 0].grid()
    ax[0, 0].legend()
    ax[0, 0].set_xscale("log")

    ax[0, 1].grid()
    ax[0, 1].legend()
    ax[0, 1].set_xscale("log")

    ax[1, 0].plot(t[1:], np.diff(t), "-k", label=f"h")
    ax[1, 0].grid()
    ax[1, 0].legend()
    ax[1, 0].set_xscale("log")
    ax[1, 0].set_yscale("log")

    plt.show()


if __name__ == "__main__":
    # trajectory()  # BDF case
    # trajectory(s=2, tableau=gauss_legendre_tableau)
    # trajectory(s=2, tableau=radau_tableau)

    adaptive_radau_IIA(s=3)

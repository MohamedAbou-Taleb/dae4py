import numpy as np
import matplotlib.pyplot as plt
from dae4py.bdf import solve_dae_BDF
from dae4py.irk import solve_dae_IRK
from dae4py.butcher_tableau import radau_tableau
from clairaut import ClairautDAEProblem, C_SPAN

INDEX = 0


def trajectory(C, index=0, singular_sol=False, tableau=None, axs=None):
    problem = ClairautDAEProblem(C, index, singular_sol)
    F = problem.F
    t_span = problem.t_span
    y0 = problem.y0
    yp0 = problem.yp0

    # solver options
    atol = rtol = 1e-6
    h = 5e-2
    if tableau is None:
        sol = solve_dae_BDF(F, y0, yp0, t_span, h, atol=atol, rtol=rtol)
    else:
        sol = solve_dae_IRK(
            F, y0, yp0, t_span, h, atol=atol, rtol=rtol, tableau=tableau
        )
    t = sol.t
    y = sol.y

    # visualization
    y_true, _ = problem.true_sol(t)

    if axs is None:
        fig, axs = plt.subplots(1)
        axs = [axs]  # brackets are only needed if argument of plt.subplots is 1

    if C is None:
        col_true = "green"
        col_int = "green"
        width = 3
    else:
        col_true = "r"
        col_int = "k"
        width = 1

    if len(y.shape) > 1:
        y = y[:, 0]

    axs[0].plot(t, y, "-", label=f"y", color=col_int, linewidth=width)
    axs[0].plot(t, y_true, "--", label=f"y true", linewidth=width, color=col_true)

    # axs[1].plot(t, yp, "-k", label=f"yp")
    # axs[1].plot(t, yp_true, "rx", label=f"yp true")
    # axs[1].grid()
    # axs[1].legend()

    return axs, sol


if __name__ == "__main__":
    axs = None

    # general solutions
    nC = 20
    header = (
        "t, "
        + ", ".join(f"y_C{i + 1}" for i in range(nC))
        + ", y_singular, "
        + ", ".join(f"yp_C{i + 1}" for i in range(nC))
        + ", yp_singular"
    )
    ys = []
    yps = []
    for C in np.linspace(*C_SPAN, nC):
        print(f"C: {C}")
        if INDEX > 0:
            axs, sol = trajectory(C, index=1, axs=axs)  #  BDF version
            axs, sol = trajectory(C, index=1, tableau=radau_tableau(3), axs=axs)
        else:
            # axs, sol = trajectory(C, axs=axs) #  BDF version
            axs, sol = trajectory(C, tableau=radau_tableau(3), axs=axs)
            ys.append(sol.y)
            yps.append(sol.yp)

    # singular solution
    if INDEX > 0:
        axs, sol = trajectory(None, index=1, singular_sol=True, axs=axs)
        axs, sol = trajectory(
            None, index=1, singular_sol=True, tableau=radau_tableau(3), axs=axs
        )
    else:
        # axs, sol = trajectory(None, singular_sol=True, axs=axs)
        axs, sol = trajectory(
            None, singular_sol=True, tableau=radau_tableau(3), axs=axs
        )
        ys.append(sol.y)
        yps.append(sol.yp)

        # export solution
        np.savetxt(
            f"clairaut.txt",
            np.hstack(
                [sol.t[:, None], np.array(ys)[:, :, 0].T, np.array(yps)[:, :, 0].T]
            ),
            header=header,
            delimiter=", ",
            comments="",
        )

    plt.show()

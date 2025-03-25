import numpy as np
from tqdm import tqdm
from scipy._lib._util import _RichResult
from dae4py.math import newton

MAX_ORDER = 6

# BDF coefficients
BDF_COEFFICIENTS = [
    np.flip(np.array([1, -1])),
    np.flip(np.array([3 / 2, -2, 1 / 2])),
    np.flip(np.array([11 / 6, -3, 3 / 2, -1 / 3])),
    np.flip(np.array([25 / 12, -4, 3, -4 / 3, 1 / 4])),
    np.flip(np.array([137 / 60, -5, 5, -10 / 3, 5 / 4, -1 / 5])),
    np.flip(np.array([147 / 60, -6, 15 / 2, -20 / 3, 15 / 4, -6 / 5, 1 / 6])),
]


def solve_dae_BDF(F, y0, yp0, t_span, h, atol=1e-6, rtol=1e-6):
    """
    Solves a system of DAEs using BDF methods.

    Parameters
    ----------
    F: callable
        Function defining the DAE system, F(t, y, yp) = 0.
    y0: array-like
        Initial condition for y.
    yp0: array-like
        Initial condition for y'.
    t_span: Tuple
        (t0, t1) defining the time span.
    h: float
        Step-size.
    atol: float, defaul: 1e-6
        Absolute tolerance for the Newton solver.
    rtol: float, default: 1e-6
        Relative tolerance for the Newton solver.

    Returns
    -------
    solution: _RichResult
        Container that stores
        - t (array-like): Time grid.
        - y (array-like): State at the time grid.
        - yp (array-like): Derivative at the time grid.
        - Y (array-like): Stage values at the time grid.
        - Yp (array-like): Stage derivative at the time grid.
    """
    t0, t1 = t_span
    if t1 <= t0:
        raise ValueError("t1 must be greater than t0")

    y0, yp0 = np.atleast_1d(y0), np.atleast_1d(yp0)
    m = len(y0)

    # initialize solution arrays
    t = [t0]
    y = [y0]
    yp = [yp0]

    # history array and current derivative
    history = np.zeros((MAX_ORDER + 1, m))
    history[0] = y0
    yp1 = yp0

    # progress bar for tracking
    steps = int(np.ceil((t1 - t0) / h))
    with tqdm(total=steps, desc="BDF integration") as pbar:
        order = 1
        while t0 < t1:
            # get BDF coefficients of the current order
            coeffs = BDF_COEFFICIENTS[order - 1]

            def residual(yp1):
                y1 = (h * yp1 - np.dot(coeffs[:-1], history[:order])) / coeffs[-1]
                return np.atleast_1d(F(t0 + h, y1, yp1))

            # solve the nonlinear system
            sol = newton(residual, yp1, atol=atol, rtol=rtol)
            if not sol.success:
                raise RuntimeError(
                    f"Newton solver failed at t={t0 + h} with error={sol.error:.2e}"
                )

            # extract the solution
            yp1 = sol.x
            y1 = (h * yp1 - np.dot(coeffs[:-1], history[:order])) / coeffs[-1]
            history[order] = y1

            # advance time, append to solution arrays and update progress bar
            t0 += h
            pbar.update(1)
            t.append(t0)
            y.append(y1)
            yp.append(yp1)

            # shift history and increase order
            if order < MAX_ORDER:
                order += 1
            else:
                history[:-1] = history[1:]  # shift history for full order

    return _RichResult(
        t=np.array(t),
        y=np.array(y),
        yp=np.array(yp),
    )

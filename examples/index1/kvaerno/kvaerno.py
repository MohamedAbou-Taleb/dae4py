import numpy as np
from dae4py.dae_problem import DAEProblem


# def rhs(t, y):
#     y1, y2 = y

#     yp = np.zeros(2, dtype=y.dtype)
#     yp[0] = -0.04 * y1 + 1e4 * y2 * y3
#     yp[1] = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2

#     return yp


def F(t, y, yp):
    y1, y2 = y
    y1p, y2p = yp

    F = np.zeros(2, dtype=np.common_type(y, yp))
    F[0] = (np.sin(y1p)**2 + np.cos(y2)**2) * y2p**2 - ((t-6)**2) * ((t-2)**2) * y1*np.exp(-t)
    F[1] = (4-t)*((y1 + y2)**3) - 64*t**2*np.exp(-t)*y1*y2

    return F


t0 = 1e-10
t1 = 5e1
t1 = 1e6
# y0 = np.array([1, 0, 0], dtype=float)
# yp0 = rhs(t0, y0)
y0 = np.array([0, 0], dtype=float)
yp0 = np.array([0, 0], dtype=float)

problem = DAEProblem(
    name="Kvaerno",
    F=F,
    t_span=(t0, t1),
    index=1,
    y0=y0,
    yp0=yp0,
)

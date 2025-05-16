import numpy as np
from dae4py.dae_problem import DAEProblem


class KamkeDAEProblem(DAEProblem):
    def __init__(self, C):
        self.C = C
        super().__init__(
            name="Clairaut",
            F=self.F,
            t_span=(-1.999, 10),
            index=0,
            true_sol=self.true_sol,
        )

    def F(self, t, y, yp):
        return 16 * y**2 * yp**3 + 2 * t * yp - y

    def true_sol(self, t):
        return (
            np.atleast_1d(np.sqrt(self.C * t + 2 * self.C**3)),
            np.atleast_1d(self.C / np.sqrt(self.C * t + 2 * self.C**3)),
        )

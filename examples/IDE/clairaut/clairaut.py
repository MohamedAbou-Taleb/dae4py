import numpy as np
from dae4py.dae_problem import DAEProblem

CASES = [
    "quadratic_neg",
    "quadratic_pos",
    "cubic_neg",
    "cubic_pos",
    "ln",
    "sqrt",
    "Kamke",
]
CASE = CASES[0]


# function and its derivative
match CASE:
    case "quadratic_neg":
        T_SPAN = [-10, 10]
        C_SPAN = T_SPAN

        def f(yp):
            return -(yp**2)

        def f_prime(yp):
            return -2 * yp

        def f_prime_prime(yp):
            return -2

        def f_prime_inv(s):
            return -0.5 * s

    case "quadratic_pos":
        T_SPAN = [-10, 10]
        C_SPAN = T_SPAN

        def f(yp):
            return yp**2

        def f_prime(yp):
            return 2 * yp

        def f_prime_prime(yp):
            return 2

        def f_prime_inv(s):
            return 0.5 * s

    case "cubic_neg":
        T_SPAN = [1, 300]
        C_SPAN = [-10, 10]

        def f(yp):
            return -(yp**3)

        def f_prime(yp):
            return -3 * yp**2

        def f_prime_prime(yp):
            return -6 * yp

        def f_prime_inv(s):
            if np.all(s <= 0):
                return np.sqrt(-(1 / 3) * s)
            else:
                raise ValueError(
                    f"case '{CASE}': {s} not in the image of f_prime. Adjust T_SPAN."
                )

    case "cubic_pos":
        T_SPAN = [-300, -1]
        C_SPAN = [-10, 10]

        def f(yp):
            return 2 * yp**3

        def f_prime(yp):
            return 6 * yp**2

        def f_prime_prime(yp):
            return 12 * yp

        def f_prime_inv(s):
            if np.all(s >= 0):
                return np.sqrt((1 / 6) * s)
            else:
                raise ValueError(
                    f"case '{CASE}': {s} not in the image of f_prime. Adjust T_SPAN."
                )

    case "ln":
        T_SPAN = [-10, -1]
        C_SPAN = [-20000, 20000]

        def f(yp):
            return (np.log(np.abs(yp)) - 1) * yp

        def f_prime(yp):
            return np.log(np.abs(yp))

        def f_prime_prime(yp):
            return 1 / yp

        def f_prime_inv(s):
            # Positive branch.
            return np.exp(s)

    case "sqrt":
        T_SPAN = [-10, -1]
        C_SPAN = [-100, 100]

        def f(yp):
            return (2 / 3) * yp * np.sqrt(np.abs(yp))

        def f_prime(yp):
            return np.sqrt(np.abs(yp))

        def f_prime_prime(yp):
            return 0.5 * np.sign(yp) / np.sqrt(np.abs(yp))

        def f_prime_inv(s):
            # Positive branch.
            return s**2

    case "Kamke":
        # T_SPAN = [-1000, -1e-1]
        T_SPAN = [-1e2, -1e-1]
        C_SPAN = [-10, 10]

        def f(yp):
            return 2 * yp**3

        def f_prime(yp):
            return 6 * yp**2

        def f_prime_prime(yp):
            return 12 * yp

        def f_prime_inv(s):
            if np.all(s >= 0):
                # return np.sqrt((1/6)*s)
                return -np.sqrt((1 / 6) * s)
            else:
                raise ValueError(
                    f"case '{CASE}': {s} not in the image of f_prime. Adjust T_SPAN."
                )

    case _:
        raise ValueError(f"Clairaut case {CASE} unknown. Allowed cases: {CASES}")


class ClairautDAEProblem(DAEProblem):
    def __init__(self, C=None, index=0, singular_sol=False):
        t0 = T_SPAN[0]
        self.C = C
        self.singular_sol = singular_sol

        if singular_sol:
            # singular solution (envelope):
            yp0 = np.atleast_1d(f_prime_inv(-t0)).astype(float)
            self.C = yp0[0]
        else:
            if C is None:
                raise RuntimeError("C must be given for general solution.")
            # general solution (straight line):
            yp0 = np.atleast_1d(C)

        y0 = t0 * yp0 + f(yp0)

        if index > 0:
            y0 = np.append(y0, 0)
            yp0 = np.append(yp0, 0)

        super().__init__(
            name="Clairaut",
            F=self.F,
            t_span=T_SPAN,
            index=index,
            y0=y0,
            yp0=yp0,
            true_sol=self.true_sol,
        )

    def F(self, t, y, yp):
        if self.index > 0:
            y, mu = y
            yp, mup = yp

            # constraint equation and its derivative
            if self.singular_sol:
                constr = t + f_prime(yp)
                dconstr_dyp = f_prime_prime(yp)
            else:
                constr = yp - self.C
                dconstr_dyp = np.ones_like(yp)

            # differential equation
            ode = np.atleast_1d(t * yp + f(yp) - y)

            # index 1 DAE
            r = np.hstack((ode + dconstr_dyp * mup, constr))
        else:
            if self.singular_sol:
                r = t + f_prime(yp)
            else:
                r = yp - self.C

        return r

    def true_sol(self, t):
        if self.singular_sol:
            # singular solution (envelope)
            yp = f_prime_inv(-t)
            y = t * yp + f(yp)
        else:
            # general solution (straight lines)
            yp = self.C * np.ones_like(np.atleast_1d(t))
            y = self.C * t + f(self.C)

        return y, yp


class ClairautYDAEProblem(ClairautDAEProblem):
    def __init__(self, C=None):

        if CASE != "cubic_pos":
            raise ValueError("YDAE only works for case cubic_pos!")

        super().__init__(C)

        self.u0 = self.y0
        self.up0 = self.yp0

    def F(self, t, y, yp):
        u = y**2
        up = 2 * y * yp
        return super().F(t, u, up)

    def true_sol(self, t):
        u, up = super().true_sol(t)
        y = np.sqrt(u)
        yp = 0.5 * up / y
        return y, yp

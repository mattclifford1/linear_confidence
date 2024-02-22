import cvxpy as cp
import deltas.utils.equations as equations


def loss_one_delta(delta1, c1, c2, N1, N2, M_emp, R):
    # eq. 6 with delta2 calced from delta1
    delta2 = delta2_given_delta1(N1, N2, M_emp, delta1, R)
    return equations.loss(c1, c2, delta1, delta2, N1, N2)

def delta2_given_delta1(N1, N2, M_emp, delta1, R):
    # eq.9
    return cp.exp((-N2/2) * cp.square(delta2_inside_bracket(N1, N2, M_emp, delta1, R)))


def delta2_inside_bracket(N1, N2, M_emp, delta1, R, two=False):
    # in eq. 9 and 10
    inner = cp.sqrt((1/N1)+(1/N2))
    right = cp.sqrt((2*cp.log(1/delta1)) / N1)
    if two == True:
        both = (M_emp/(2*R)) - 2*inner - right
    else:
        both = (M_emp/(2*R)) - inner - right
    return both


if __name__ == '__main__':
    ### doesn't work as non convex
    c1, c2, N1, N2, M_emp, R_sup = 1, 1, 100, 150, 0.2, 1

    # Optimise using cvxpy
    d1 = cp.Variable()

    # Create two constraints.
    constraints = [d1 <= 1,
                d1 >= 0]

    # Form objective.
    # obj = cp.Minimize(deltas.loss(c1, c2, d1, d2, N1, N2))
    obj = cp.Minimize(loss_one_delta(
        d1, c1, c2, N1, N2, M_emp, R_sup))

    # Form and solve problem.
    prob = cp.Problem(obj, constraints)
    prob.solve()  # Returns the optimal value.
    print("status:", prob.status)
    print("optimal value", prob.value)
    print("optimal var", d1.value)
import numpy as np
import torch


def class_cost(c=1, delta=0.5, N=100):
    # half of eq. 6
    return c*((1-delta)*(1/(N+1)) + delta)

def loss(c1, c2, delta1, delta2, N1, N2):
    # eq. 6
    J = class_cost(c1, delta1, N1) + \
        class_cost(c2, delta2, N2)
    return J

def loss_one_delta(delta1, c1, c2, N1, N2, M_emp, R):
    # eq. 6 with delta2 calced from delta1
    delta2 = delta2_given_delta1(N1, N2, M_emp, delta1, R)
    return loss(c1, c2, delta1, delta2, N1, N2)

def delta2_inside_bracket(N1, N2, M_emp, delta1, R):
    # in eq. 9 and 10
    inner = np.sqrt((1/N1)+(1/N2))
    right = np.sqrt((2*np.log(1/delta1)) / N1)
    both = (M_emp/(2*R)) - 2*inner - right
    # both = (M_emp/(2*R)) - inner - right
    return both

def delta2_given_delta1(N1, N2, M_emp, delta1, R):
    # eq.9
    return np.exp((-N2/2) * np.square(delta2_inside_bracket(N1, N2, M_emp, delta1, R)))

def dd2_dd1(N1, N2, M_emp, delta1, R):
    # above eq.10
    left = delta2_given_delta1(N1, N2, M_emp, delta1, R)
    right = (-N1*delta2_inside_bracket(N1, N2, M_emp, delta1, R)) * \
            ((1/(N1*delta1)) * (1/(np.sqrt((2*np.log(1/delta1))/N1))))
    return left * right

def J_derivative(delta1, c1, c2, N1, N2, M_emp, R):
    # eq. 10
    left = c1*(N1/(N1+1))
    right = dd2_dd1(N1, N2, M_emp, delta1, R) * (c2*(N2/(N2+1)))
    return left + right

def scipy_optim_func(delta1, c1, c2, N1, N2, M_emp, R):
    # return cost function value and it's derivative for scipy to minimise
    return loss_one_delta(delta1, c1, c2, N1, N2, M_emp, R), J_derivative(delta1, c1, c2, N1, N2, M_emp, R)

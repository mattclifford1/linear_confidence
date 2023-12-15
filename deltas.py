'''
functions for cost, derivative and finding deltas from one another
'''
import numpy as np
import radius
from misc import USE_TWO


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


def loss_one_delta_matt(delta1, c1, c2, N1, N2, M_emp, R):
    # eq. 6 with delta2 calced from delta1
    delta2 = delta2_given_delta1_wolf(N1, N2, M_emp, delta1, R)
    return loss(c1, c2, delta1, delta2, N1, N2)


def contraint_eq7(N1, N2, R1_emp, R2_emp, delta1, R, M_emp, D_emp):
    # eq. 8 in scipy contraint form that it equals 0
    delta2 = delta2_given_delta1(N1, N2, M_emp, delta1, R)
    R1_est = radius.R_upper_bound(R1_emp, R, N1, delta1)
    R2_est = radius.R_upper_bound(R2_emp, R, N2, delta2)
    equal_to_0 = R1_est + R2_est - D_emp
    return equal_to_0


def contraint_eq7_matt(N1, N2, R1_emp, R2_emp, delta1, R, M_emp, D_emp):
    # eq. 8 in scipy contraint form that it equals 0
    delta2 = delta2_given_delta1_matt(N1, N2, M_emp, delta1, R)
    R1_est = radius.R_upper_bound(R1_emp, R, N1, delta1)
    R2_est = radius.R_upper_bound(R2_emp, R, N2, delta2)
    equal_to_0 = R1_est + R2_est - D_emp
    return equal_to_0


def contraint_eq8(N1, N2, delta1, R, M_emp):
    # eq. 8 in scipy contraint form that it equals 0
    delta2 = delta2_given_delta1(N1, N2, M_emp, delta1, R)
    equal_to_0 = radius.error_upper_bound(
        R, N1, delta1) + radius.error_upper_bound(R, N2, delta2) - M_emp
    return equal_to_0

def inner_margin(N, delta):
    # inside of eq. 8
    return (np.sqrt(1/N)) * (2 + (np.sqrt(2*np.log(1/delta))))

def delta2_inside_bracket(N1, N2, M_emp, delta1, R, two=USE_TWO):
    # in eq. 9 and 10
    inner = np.sqrt((1/N1)+(1/N2))
    right = np.sqrt((2*np.log(1/delta1)) / N1)
    if two == True:
        both = (M_emp/(2*R)) - 2*inner - right
    else:
        both = (M_emp/(R)) - 2*inner - right
    return both

def delta2_given_delta1(N1, N2, M_emp, delta1, R):
    # eq.9
    return np.exp((-N2/2) * np.square(delta2_inside_bracket(N1, N2, M_emp, delta1, R)))


def delta2_given_delta1_matt(N1, N2, M_emp, delta1, R):
    # eq.9 but re doing the maths
    x = (1/np.sqrt(N1))*(2 + np.sqrt(2*np.log(1/delta1))) - (M_emp/(1*R))
    y = ( (x*np.sqrt(N2) - 2)**2 )/2
    return 1/(np.exp(y))

def delta2_given_delta1_wolf(N1, N2, M, delta1, R):
    l = (1/delta1)**(-N2/N1)
    ll_brac = -(2*N2/N1) -((4*np.sqrt(N2))/np.sqrt(N1)) -((N2*(M**2))/(8*(R**2)))
    l_brac = ((N2*M)/(np.sqrt(N1)*R)) + ((np.sqrt(N2)*M)/R)
    m_brac = ((N2*M*np.sqrt(np.log(1/delta1))) / (np.sqrt(2)*np.sqrt(N1)*R))
    r_brac = -((2*np.sqrt(2)*N2*np.sqrt(np.log(1/delta1))) / (N1))
    rr_brac = -((2*np.sqrt(2)*np.sqrt(N2)*np.sqrt(np.log(1/delta1))) / (np.sqrt(N1))) - 2
    brac = ll_brac + l_brac + m_brac + r_brac + rr_brac
    return l * np.exp(brac)


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


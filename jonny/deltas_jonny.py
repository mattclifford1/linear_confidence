'''
functions for cost, derivative and finding deltas from one another
'''
import numpy as np
import deltas.utils.radius as radius
from deltas.misc.use_two import USE_TWO


def class_cost(c=1, delta=0.5, N=100):
    # half of eq. 6
    return c*((1-delta)*(1/(N+1)) + delta)


def loss(delta1, delta2, beta1, beta2, data_info):
    # eq. 6
    N1 = data_info['N1']
    N2 = data_info['N2']
    c1 = data_info['c1']
    c2 = data_info['c2']
    # slacks_1 = data_info['slacks1']
    # slacks_2 = data_info['slacks2']

    J = class_cost(c1, delta1, N1) + \
        class_cost(c2, delta2, N2) + 2-(beta1+beta2)
    return J


def loss_one_delta(delta1, data_info):
    # eq. 6 with delta2 calced from delta1
    
    delta2_given_delta1_func = data_info['delta2_given_delta1_func']

    delta2 = delta2_given_delta1_func(delta1, data_info)
    return loss(delta1, delta2, data_info)


def loss_one_delta_matt(delta1, c1, c2, N1, N2, M_emp, R):
    # eq. 6 with delta2 calced from delta1
    delta2 = delta2_given_delta1_matt(N1, N2, M_emp, delta1, R)
    return loss(c1, c2, delta1, delta2, N1, N2)


def contraint_eq7(delta1, delta2, beta1, beta2, data_info):

    # eq. 8 in scipy contraint form that it equals 0
    N1 = data_info['N1']
    N2 = data_info['N2']
    R1_emp = data_info['empirical R1']*beta1
    R2_emp = data_info['empirical R2']*beta2
    R = data_info['R all data']
    D_emp = data_info['empirical D']

    R1_est = radius.R_upper_bound(R1_emp, R, N1, delta1)
    R2_est = radius.R_upper_bound(R2_emp, R, N2, delta2)

    # should now be equal to zero (ideally)
    equal_to_0 = R1_est + R2_est - D_emp

    #for each point

    return equal_to_0



def contraint_eq8(delta1, delta2, data_info):
    # eq. 8 in scipy contraint form that it equals 0
    N1 = data_info['N1']
    N2 = data_info['N2']
    R = data_info['R all data']
    M_emp = data_info['empirical margin']

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

def delta2_given_delta1(delta1, data_info):
    # eq.9
    N1 = data_info['N1']
    N2 = data_info['N2']
    R = data_info['R all data']
    M_emp = data_info['empirical margin']

    return np.exp((-N2/2) * np.square(delta2_inside_bracket(N1, N2, M_emp, delta1, R)))


def delta2_given_delta1_matt(delta1, data_info):
    # eq.9 but re doing the maths - refer to notes for letters
    N1 = data_info['N1']
    N2 = data_info['N2']
    R = data_info['R all data']
    R1_emp = data_info['empirical R1']
    R2_emp = data_info['empirical R2']
    D_emp = data_info['empirical D']
    
    if USE_TWO == True:
        factor = 2
    else:
        factor = 1

    def error(R, N, d, f):
        return f*(R/np.sqrt(N)) * (2 + np.sqrt(2*np.log(1/d)))

    B = D_emp - R2_emp - R1_emp - error(R, N1, delta1, factor)
    delta2 = 1/np.exp(0.5*(np.square(((B*np.sqrt(N2))/(factor*R)) - 2)))
    return delta2

def eq7_matt(delta1, delta2, data_info):
    N1 = data_info['N1']
    N2 = data_info['N2']
    R = data_info['R all data']
    R1_emp = data_info['empirical R1']
    R2_emp = data_info['empirical R2']
    D_emp = data_info['empirical D']

    if USE_TWO == True:
        factor = 2
    else:
        factor = 1
        
    R1_est = R1_emp + factor*(R/np.sqrt(N1))*(2 + np.sqrt(2*np.log(1/delta1)))
    R2_est = R2_emp + factor*(R/np.sqrt(N2))*(2 + np.sqrt(2*np.log(1/delta2)))
    return R1_est + R2_est - D_emp


def delta2_given_delta1_jonny(N1, N2, M_emp, delta1, R):
    # eq.9 but re doing the maths - jonny
    left = (np.sqrt(N2)*M_emp) / (2*R)
    right = np.sqrt(N2/N1)*(2*np.sqrt(2*np.log(1/delta1)))
    inner = left - right - 2
    return np.exp(-(inner**2))

def delta2_given_delta1_wolf(N1, N2, M, delta1, R):
    l = (1/delta1)**(-N2/N1)
    ll_brac = -(2*N2/N1) -((4*np.sqrt(N2))/np.sqrt(N1)) -((N2*(M**2))/(8*(R**2)))
    l_brac = ((N2*M)/(np.sqrt(N1)*R)) + ((np.sqrt(N2)*M)/R)
    m_brac = ((N2*M*np.sqrt(np.log(1/delta1))) / (np.sqrt(2)*np.sqrt(N1)*R))
    r_brac = -((2*np.sqrt(2)*N2*np.sqrt(np.log(1/delta1))) / (N1))
    rr_brac = -((2*np.sqrt(2)*np.sqrt(N2)*np.sqrt(np.log(1/delta1))) / (np.sqrt(N1))) - 2
    brac = ll_brac + l_brac + m_brac + r_brac + rr_brac
    return l * np.exp(brac)


def dd2_dd1(N1, N2, M_emp, delta1, R, delta2):
    # above eq.10
    # left = delta2_given_delta1(N1, N2, M_emp, delta1, R)
    left = delta2
    right = (-N1*delta2_inside_bracket(N1, N2, M_emp, delta1, R)) * \
            ((1/(N1*delta1)) * (1/(np.sqrt((2*np.log(1/delta1))/N1))))
    return left * right

def J_derivative(delta1, data_info):
    # eq. 10
    N1 = data_info['N1']
    N2 = data_info['N2']
    c1 = data_info['c1']
    c2 = data_info['c2']
    R = data_info['R all data']
    M_emp = data_info['empirical margin']
    delta2_given_delta1_func = data_info['delta2_given_delta1_func']

    delta2 = delta2_given_delta1_func(delta1, data_info)

    left = c1*(N1/(N1+1))
    right = dd2_dd1(N1, N2, M_emp, delta1, R, delta2) * (c2*(N2/(N2+1)))
    return left + right


def scipy_optim_func(delta1, data_info):
    # return cost function value and it's derivative for scipy to minimise
    return loss_one_delta(delta1, data_info), J_derivative(delta1, data_info)


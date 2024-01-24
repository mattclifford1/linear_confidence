'''
functions for cost, derivative and finding deltas from one another

Jonny's workspace
'''
import numpy as np


def delta2_given_delta1(delta1, data_info):
    # eq.9 but re doing the maths - jonny
    N1 = data_info['N1']
    N2 = data_info['N2']
    R = data_info['R all data']
    M_emp = data_info['empirical margin']
    
    left = (np.sqrt(N2)*M_emp) / (2*R)
    right = np.sqrt(N2/N1)*(2*np.sqrt(2*np.log(1/delta1)))
    inner = left - right - 2
    return np.exp(-(inner**2))

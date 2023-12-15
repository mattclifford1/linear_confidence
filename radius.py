import numpy as np
from misc import USE_TWO


def R_upper_bound(R_emp, R_sup, N, delta, two=USE_TWO):
    # eq. 5
    return R_emp + error_upper_bound(R_sup, N, delta, two)


def error_upper_bound(R_sup, N, delta, two=USE_TWO):
    error = (R_sup/np.sqrt(N)) * (2 + (np.sqrt( 2*np.log(1/delta) )))
    if two == True:
        error *= 2
    return error

def calc_emp_R(X_proj, mean_proj):
    euclid = np.sqrt(np.sum(np.square(X_proj - mean_proj), axis=1))
    print(X_proj)
    print(mean_proj)


def supremum(X, x0=0):
    
    euclid = np.sqrt(np.sum(np.square(X - x0), axis=1))
    return np.max(euclid)



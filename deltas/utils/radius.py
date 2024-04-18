import numpy as np
from deltas.misc.use_two import USE_TWO, USE_GLOBAL_R
import warnings


def R_upper_bound(R_emp, R_sup, N, delta, two=USE_TWO):
    # eq. 5
    if USE_GLOBAL_R == True:
        R = R_sup
    else:
        R = R_emp
    return R_emp + error_upper_bound(R, N, delta, two)


def error_upper_bound(R_sup, N, delta, two=USE_TWO):
    with warnings.catch_warnings(record=True) as w:
        error = (R_sup/np.sqrt(N)) * (2 + (np.sqrt( 2*np.log(1/delta) )))
        if len(w) > 0:
            raise ValueError
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



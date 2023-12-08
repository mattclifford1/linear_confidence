import numpy as np


def R_upper_bound(R_emp, R_sup, N, delta, two=False):
    # eq. 5
    if two == True:
        R_expt = R_emp + (2*(R_sup/np.sqrt(N))) * (2+(np.sqrt(2*np.log(1/delta))))
    else:
        R_expt = R_emp + ((R_sup/np.sqrt(N))) * (2+(np.sqrt(2*np.log(1/delta))))
    return R_expt

def calc_emp_R(X_proj, mean_proj):
    euclid = np.sqrt(np.sum(np.square(X_proj - mean_proj), axis=1))
    print(X_proj)
    print(mean_proj)


def supremum(X, x0=0):
    
    euclid = np.sqrt(np.sum(np.square(X - x0), axis=1))
    return np.max(euclid)

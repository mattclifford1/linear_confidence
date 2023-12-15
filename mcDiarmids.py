import numpy as np

# upper bound of eq. 2 distance
def upper_bound(R, N, delta):
    return (R/np.sqrt(N)) * (2+(np.sqrt(2*np.log(1/delta))))
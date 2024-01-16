'''
simple grid search to find deltas wrt a contraint tolerance
'''

import numpy as np

def calc(loss_func, data_info, contraint, bounds=[[0, 0], [1, 1]]):
    # get grid of deltas
    delta_linspace = 10
    delta1s = np.linspace(bounds[0][0], bounds[1][0], delta_linspace)
    delta2s = np.linspace(bounds[0][1], bounds[1][1], delta_linspace)

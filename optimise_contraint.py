from scipy.optimize import minimize, Bounds
import numpy as np
'''
get delta2 from delta1 by minimising the contraint
'''


def get_init_deltas(contraint_func, data_info):
    '''
    contraint_func: to equal 0 with args: delta1, delta2, data_info
    delta1: fixed delta1 value
    '''
    deltas_init = (np.random.uniform(), np.random.uniform())

    def loss_wrapper(deltas_init):
        return np.abs(contraint_func(deltas_init[0], deltas_init[1], data_info))

    res = minimize(loss_wrapper,
                   deltas_init,
                #    (data_info),
                   #    method='SLSQP',
                   bounds=Bounds([0, 0], [1, 1]),
                #    jac=use_grad,  # use gradient
                #    constraints=contrs
                   )

    # print(res.x[0])
    # print(res.x[1])
    # print(contraint_func(res.x[0], res.x[1], data_info))

    return res.x[0], res.x[1]

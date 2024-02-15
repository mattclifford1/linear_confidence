'''
simple grid search to find deltas wrt a contraint tolerance
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
import optimise_contraint
import deltas as ds
import radius
import plots


def optimise(data_info, loss_func, contraint_func, delta1_from_delta2=None, num_deltas=1, grid_search=False, _print=True, _plot=True):
    # get initial deltas
    delta1 = np.random.uniform()
    delta1 = 1   # use min error distance to give it the best chance to optimise correctly
    if num_deltas == 2:
        bounds = Bounds([0, 0], [1, 1])
        if delta1_from_delta2 != None:
            delta2 = delta1_from_delta2(delta1, data_info)
        else:
            # get from the contraint function
            delta1, delta2 = optimise_contraint.get_init_deltas(
                contraint_func, data_info)
        deltas_init = [delta1, delta2]
        if _print == True:
            print(f'deltas init: {deltas_init}')
    else:
        bounds = Bounds([0], [1])
        deltas_init = [delta1]
        if _print == True:
            print(
                f'deltas init: {[deltas_init[0], delta1_from_delta2(deltas_init[0], data_info)]}')

    if isinstance(loss_func, tuple) or isinstance(loss_func, list):
        use_grad = True
    else:
        use_grad = False

    # set up contraints
    data_info['delta2_given_delta1_func'] = delta1_from_delta2
    if num_deltas == 2:
        def contraint_wrapper(deltas):
            return contraint_func(deltas[0], deltas[1], data_info)

    else:
        def contraint_wrapper(deltas):
            delta2 = delta1_from_delta2(deltas[0], data_info)
            return contraint_func(deltas[0], delta2, data_info)

    solution_possible = True if contraint_wrapper(deltas_init) <= 0 else False
    if _print == True:
        print(f'eq. 7 can be satisfied: {ds.contraint_eq7(1, 1, data_info) <= 0}')
        print(f'constraint init: {solution_possible}')

    # return early if optimisation not possible
    if solution_possible == False:
        return 1, 1, solution_possible, False

    def contraint_real(deltas):
        return np.sum(np.iscomplex(deltas))

    contrs = [
        {'type': 'eq', 'fun': contraint_wrapper},
        # {'type':'eq', 'fun': contraint_real},  # more equality contraints that independaent variables
    ]
    if grid_search == True:
        # line search for optimal value - only works for one delta atm
        resolution = 1000
        delta1s = np.linspace(0.000000000000001, 1, resolution)
        J = loss_func(delta1s, data_info)
        # eliminate any deltas which don't satisfy the constraint
        tol = 1e-5
        constraints = np.array([contraint_wrapper([d]) for d in delta1s])
        J[constraints > tol] = np.max(J)
        J[constraints < -tol] = np.max(J)
        deltas = [delta1s[np.argmin(J)]]
        optim_msg = 'Grid Search Optimisation Complete'
    else:
        res = minimize(loss_func,
                       deltas_init,
                       (data_info),
                       #    method='SLSQP',
                       bounds=bounds,
                       jac=use_grad,  # use gradient
                       constraints=contrs
                       )
        deltas = res.x
        optim_msg = res.message

    if len(deltas) == 1:
        delta1 = deltas[0]
        delta2 = delta1_from_delta2(delta1, data_info)
    else:
        delta1 = deltas[0]
        delta2 = deltas[1]

    solution_found = True if contraint_wrapper(deltas) == 0 else False
    if _print == True:
        print(optim_msg)
        print(f'    delta1 : {delta1} \n    delta2: {delta2}')
        print(f'    constraint satisfied: {solution_found}')

    if _plot == True:
        # plot loss function
        if num_deltas == 1:
            if grid_search != True:
                delta1s = np.linspace(0.000000000001, 1, 1000)
                J = loss_func(delta1s, data_info)
                constraints = [contraint_wrapper([d]) for d in delta1s]
            _, ax = plt.subplots(1, 1)
            ax.plot(delta1s, J, label='Loss')
            ax.plot(delta1s, constraints, label='constraint')
            ax.set_xlabel('delta1')
            # ax.set_ylabel('Loss')
            ax.legend()

        # calculate each R upper bound
        R1_est = radius.R_upper_bound(
            data_info['empirical R1'], data_info['R all data'], data_info['N1'], delta1)
        R2_est = radius.R_upper_bound(
            data_info['empirical R2'], data_info['R all data'], data_info['N2'], delta2)
        _, ax = plt.subplots(1, 1)
        _ = plots.plot_projection(
            data_info['projected_data'], None, R1_est, R2_est, R_est=True, ax=ax)

    return delta1, delta2, solution_possible, solution_found

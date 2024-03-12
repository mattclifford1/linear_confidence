'''
simple grid search to find deltas wrt a contraint tolerance
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
import deltas.optimisation.optimise_contraint as optimise_contraint
import deltas.utils.equations as ds
import deltas.utils.radius as radius
import deltas.plotting.plots as plots


def optimise(data_info, loss_func, contraint_func, delta2_from_delta1=None, num_deltas=1, _print=True, _plot=True, grid_search=False, grid_2D=False):
    # get initial deltas
    delta1 = np.random.uniform()
    delta1 = 1   # use min error distance to give it the best chance to optimise correctly
    if num_deltas == 2:
        bounds = Bounds([0, 0], [1, 1])
        if delta2_from_delta1 != None:
            delta2 = delta2_from_delta1(delta1, data_info)
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
                f'deltas init: {[deltas_init[0], delta2_from_delta1(deltas_init[0], data_info)]}')

    if isinstance(loss_func, tuple) or isinstance(loss_func, list):
        use_grad = True
    else:
        use_grad = False

    # set up contraints
    data_info['delta2_given_delta1_func'] = delta2_from_delta1
    if num_deltas == 2:
        def contraint_wrapper(deltas):
            return contraint_func(deltas[0], deltas[1], data_info)

    else:
        def contraint_wrapper(deltas):
            delta2 = delta2_from_delta1(deltas[0], data_info)
            return contraint_func(deltas[0], delta2, data_info)

    solution_possible = True if contraint_func(1, 1, data_info) <= 0 else False
    if _print == True:
        print(
            f'eq. 7 can be satisfied: {solution_possible}')
        print(f'constraint init: {solution_possible}')

    def contraint_real(deltas):
        return np.sum(np.iscomplex(deltas))

    contrs = [
        {'type': 'eq', 'fun': contraint_wrapper},
        # {'type':'eq', 'fun': contraint_real},  # more equality contraints that independaent variables
    ]
    resolution = 10000
    tol_constraint = 1e-6
    if grid_search == True and solution_possible == True:
        # line search for optimal value - only works for one delta atm
        delta1s = np.linspace(1/resolution, 1, resolution)
        J = loss_func(delta1s, data_info)
        # eliminate any deltas which don't satisfy the constraint
        constraints = np.array([contraint_wrapper([d]) for d in delta1s])
        J[constraints > tol_constraint] = np.max(J)
        J[constraints < -tol_constraint] = np.max(J)
        deltas = [delta1s[np.argmin(J)]]
        optim_msg = 'Grid Search Optimisation Complete'
    elif grid_search == True and solution_possible == False and grid_2D == True:
        # be careful as this method can use a lot of RAM!
        if _print == True:
            print('Solution not possible so ignoring constraint and using decoupled loss function for each delta')
        # do a 2D grid search without the contraint as it's impossible to satisfy
        delta1s = np.linspace(1/resolution, 1, resolution)
        delta2s = np.linspace(1/resolution, 1, resolution)
        delta1s_grid, delta2s_grid = np.meshgrid(delta1s, delta2s)
        J = ds.loss(delta1s_grid, delta2s_grid, data_info)
        deltas = [delta1s[np.argmin(J)], delta2s[np.argmin(J)]]
        optim_msg = 'Contraint not possible so used uncoupled delta grid Search Optimisation'
        num_deltas = 2
    elif grid_search == True and solution_possible == False and grid_2D == False:
        if _print == True:
            print('solution not possible with 1D grid search, returning None')
        return None
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
        delta2 = delta2_from_delta1(delta1, data_info)
    else:
        delta1 = deltas[0]
        delta2 = deltas[1]

    solution_found = True if contraint_wrapper(deltas) == 0 else False
    if _print == True:
        print(optim_msg)
        print(f'    delta1 : {delta1} \n    delta2: {delta2}')
        print(f'    constraint satisfied: {solution_found}')

    if _plot == True and grid_search == True:
        # plot loss function
        if num_deltas == 1:
            _, ax = plt.subplots(1, 1)
            ax.plot(delta1s, J, label='Loss')
            ax.plot(delta1s, constraints, label='constraint')
            ax.set_xlabel('delta1')
            ax.legend()
            plt.show()
        else:
            _, ax = plt.subplots(1, 1)
            c = ax.contourf(delta1s_grid, delta2s_grid, J)
            ax.set_xlabel('delta1')
            ax.set_ylabel('delta2')
            cbar = plt.colorbar(c)
            cbar.ax.set_ylabel('Loss')
            plt.show()
        plots.deltas_projected_boundary(delta1, delta2, data_info)

    return {'delta1': delta1, 
            'delta2': delta2, 
            'solution_possible': solution_possible, 
            'solution_found': solution_found,
            'loss': loss_func(delta1, data_info)}

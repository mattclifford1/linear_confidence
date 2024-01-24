import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint, Bounds
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import data_utils
import plots
import normal
import projection
import radius
import deltas as ds
import optimise_contraint
import models


def get_data_and_classifier(m1 = [1, 1],
                            m2 = [10, 10],
                            cov1 = [[1, 0], [0, 1]],
                            cov2 = [[1, 0], [0, 1]],
                            N1 = 10000,
                            N2 = 10000,
                            scale = True,
                            model='Linear',
                            balance_clf=False,
                            test_nums=[10000, 10000],
                            _plot=True):
    data = normal.get_two_classes(means=[m1, m2],
                                  covs=[cov1, cov2],
                                  num_samples=[N1, N2])
    data_test = normal.get_two_classes(means=[m1, m2],
                                       covs=[cov1, cov2],
                                       num_samples=[test_nums[0], test_nums[1]])


    scaler = data_utils.normaliser(data)
    if scale == True:
        data = scaler(data)
        data_test = scaler(data_test)
        m1 = scaler.transform_instance(m1)
        m2 = scaler.transform_instance(m2)

    if balance_clf == True:
        weights = 'balanced'
    else:
        weights = None
    if model == 'SVM':
        clf = models.SVM(class_weight=weights).fit(data['X'], data['y'])
    elif model == 'Linear':
        clf = models.linear(class_weight=weights).fit(data['X'], data['y'])

    if _plot == True:
        ax, _ = plots._get_axes(None)
        plots.plot_classes(data, ax=ax)
        plots.plot_decision_boundary(clf, data, ax=ax)
        plots.plt.show()
    return {'data': data, 'clf': clf, 'mean1': m1, 'mean2': m2, 'data_test': data_test}


def data_project_and_info(data, m1, m2, clf, data_test=None, _plot=True, _print=True):
    ''' calcualte info we need from data: R, M, D etc. 
    also project the data from the classfiier'''
    # get projections
    proj_data = projection.from_clf(data, clf, supports=True)
    if data_test != None:
        proj_data_test = projection.from_clf(data_test, clf, supports=True)
        proj_data_test['y'] = data_test['y']   # DON'T USE THE CLF PREDS!!!
    else:
        proj_data_test = None
    proj_means = projection.from_clf({'X': np.array([m1, m2]), 'y': [0, 1]}, clf)

    # Empircal M
    M_emp = np.abs(proj_data['supports'][1]-proj_data['supports'][0]).squeeze()

    # get Rs
    # R_sup = radius.supremum(data['X'])
    R_sup = radius.supremum(proj_data['X'])
    # empirical means
    xp1, xp2 = projection.get_classes(proj_data)
    emp_xp1, emp_xp2 = projection.get_emp_means(proj_data)
    R1_emp = radius.supremum(proj_data['X1'], emp_xp1)
    R2_emp = radius.supremum(proj_data['X2'], emp_xp2)

    # Empirical D
    D_emp = np.abs(emp_xp1 - emp_xp2)

    data_info = {'projected_data': proj_data,
                 'projected_data_test': proj_data_test,
                 'projected_means': proj_means,
                 'empirical margin': M_emp,
                 'R all data': R_sup,
                 'projected_data 1': xp1,
                 'projected_data 2': xp2,
                 'empirical_projected_mean 1': emp_xp1,
                 'empirical_projected_mean 2': emp_xp2,
                 'empirical R1': R1_emp,
                 'empirical R2': R2_emp,
                 'empirical D': D_emp,
                 'N1': (data['y']==0).sum(),
                 'N2': (data['y']==1).sum(),
                 }
    # plot
    if _plot == True:
        _ = plots.plot_projection(proj_data, proj_means, R1_emp, R2_emp, data_info=data_info)
    
    if _print == True:
        print(f'R1 empirical: {R1_emp}\nR2 empirical: {R2_emp}')

    return data_info

def print_params(data_info):
    print(
        f"""Parameters
        R:  {data_info['R all data']}
        N1: {data_info['N1']}
        N2: {data_info['N1']}
        R1: {data_info['empirical R1']}
        R2: {data_info['empirical R2']}
        M:  {data_info['empirical margin']}
        D:  {data_info['empirical D']}
        C1: {data_info['c1']}
        C2: {data_info['c2']}""")


def optimise(data_info, loss_func, contraint_func, delta1_from_delta2=None, num_deltas=1, grid_search=False,_print=True, _plot=True):
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
    
    if _print == True:
        print(f'eq. 7 can be satisfied: {ds.contraint_eq7(1, 1, data_info) <= 0}')
        print(f'constraint init: {contraint_wrapper(deltas_init) <= 0}')

    def contraint_real(deltas):
        return np.sum(np.iscomplex(deltas))

    contrs = [
        {'type':'eq', 'fun': contraint_wrapper},
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
        J[constraints>tol] = np.max(J)
        J[constraints<-tol] = np.max(J)
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

    if _print == True:
        print(optim_msg)
        print(f'    delta1 : {delta1} \n    delta2: {delta2}')
        print(f'    constraint satisfied: {contraint_wrapper(deltas)==0}')
    
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
        R1_est = radius.R_upper_bound(data_info['empirical R1'], data_info['R all data'], data_info['N1'], delta1)
        R2_est = radius.R_upper_bound(data_info['empirical R2'], data_info['R all data'], data_info['N2'], delta2)
        _, ax = plt.subplots(1, 1)
        _ = plots.plot_projection(data_info['projected_data'], data_info['projected_means'], R1_est, R2_est, R_est=True, ax=ax)
    return delta1, delta2


def eval_test(data_clf, data_info, delta1, delta2, _print=True, _plot=True):
    # calculate each R upper bound
    R1_est = radius.R_upper_bound(
        data_info['empirical R1'], data_info['R all data'], data_info['N1'], delta1)
    R2_est = radius.R_upper_bound(
        data_info['empirical R2'], data_info['R all data'], data_info['N2'], delta2)
    Rs = {'R1': R1_est, 'R2': R2_est}
    
    # est data
    if data_clf['data_test'] == None:
        return
    # projected test data
    if data_info['projected_data_test'] == None:
        return
    
    # add error to min class and minus from max class
    min_mean = np.argmin(data_info['projected_means']['X'])
    max_mean = np.argmax(data_info['projected_means']['X'])
    class_names = ['1', '2']
    upper_min_class = data_info['projected_means'][f'X{class_names[min_mean]}'] + \
        Rs[f'R{class_names[min_mean]}']
    lower_max_class = data_info['projected_means'][f'X{class_names[max_mean]}'] - \
        Rs[f'R{class_names[max_mean]}']
    
    # get average as the boundary
    boundary = (upper_min_class + lower_max_class)/2
    class_nums = [data_info['projected_means']['y'][min_mean], data_info['projected_means']['y'][max_mean]]
    delta_clf = delta_adjusted_clf(boundary, class_nums)

    # predict on both classifiers (original and delta adjusted)
    y_clf = data_clf['clf'].predict(data_clf['data_test']['X'])
    y_deltas = delta_clf.predict(data_info['projected_data_test']['X'])

    if _print == True:
        print(f"original accuracy: {accuracy_score(data_clf['data_test']['y'], y_clf)}")
        print(
            f"deltas   accuracy: {accuracy_score(data_info['projected_data_test']['y'], y_deltas)}")

    if _plot == True:
        # TODO: make this shode a loop
        _, ax = plt.subplots(1, 1)

        # clf points
        # data = data_clf['data_test']
        # data_projected = projection.from_clf(
        #     data_clf['data_test'], data_clf['clf'])
        # data_projected['y'] = data_clf['data_test']['y']
        # xp1, xp2 = projection.get_classes(data_projected)
        # y = 0
        # ax.scatter(xp1, np.ones_like(xp1)*y, c='b', s=10,
        #            label='proj 1', marker='o')
        # ax.scatter(xp2, np.ones_like(xp2)*y, c='r', s=10, 
        #            label='proj 2', marker='o')
        
        proj_data = {'X': data_info['projected_data_test']['X'], 'y': y_clf}

        X = data_clf['data_test']['X']
        proj_data['y'] = y_clf
        xp1, xp2 = projection.get_classes(proj_data)
        y = -0.1
        ax.scatter(xp1, np.ones_like(xp1)*y, c='b', s=10,
                   label='Clf pred 1', marker='o')
        ax.scatter(xp2, np.ones_like(xp2)*y, c='r', s=10, 
                   label='Clf pred 2', marker='o')
        # clf points - linspace
        X, y = plots.get_grid_pred(
            data_clf['clf'], data_clf['data_test'], probs=False, flat=True)
        data = {'X': X, 'y': y}
        data_projected = projection.from_clf(data, data_clf['clf'])
        xp1, xp2 = projection.get_classes(data_projected)
        y = -0.2
        ax.scatter(xp1, np.ones_like(xp1)*y, c='b', s=10, label='Original clf 1', marker='x')
        ax.scatter(xp2, np.ones_like(xp2)*y, c='r', s=10, marker='x')

        # deltas points
        # data = data_info['projected_data_test']
        # xp1, xp2 = projection.get_classes(data)
        # y = -0.4
        # ax.scatter(xp1, np.ones_like(xp1)*y, c='b', s=10,
        #            label='proj 1', marker='o')
        # ax.scatter(xp2, np.ones_like(xp2)*y, c='r', s=10,
        #            label='proj 2', marker='o')
        
        proj_data = {'X': data_clf['data_test'], 'y': y_deltas}
        proj_data = {'X': data_info['projected_data_test']['X'], 'y': y_deltas}
        xp1, xp2 = projection.get_classes(proj_data)
        y = -0.5
        ax.scatter(xp1, np.ones_like(xp1)*y, c='b', s=10,
                   label='Deltas pred 1', marker='o')
        ax.scatter(xp2, np.ones_like(xp2)*y, c='r', s=10,
                   label='Deltas pred 2', marker='o')
        # deltas points - linspace
        deltas_preds = delta_clf.predict(data_projected['X'])
        xp1, xp2 = projection.get_classes({'X': data_projected['X'], 'y': deltas_preds})
        y = -0.6
        ax.scatter(xp1, np.ones_like(xp1)*y, c='b', s=20,
                   label='Deltas clf 1', marker='+')
        ax.scatter(xp2, np.ones_like(xp2)*y, c='r', s=20, marker='+')


        ax.legend()
        ax.plot([0], [-1.5], c='w')
        ax.set_title('original vs deltas on test dataset')

        # plot in original space
        _, ax2 = plt.subplots(1, 1)
        data = {'X': data_clf['data_test']['X'], 'y': data_clf['clf'].predict(
            data_clf['data_test']['X'])}
        # data = data_clf['data_test']
        plots.plot_classes(data, ax=ax2)
        plots.plot_decision_boundary(
            data_clf['clf'], data_clf['data_test'], ax=ax2, probs=False)
        plots.plt.show()



class delta_adjusted_clf:
    ''' boundary to make decision in projected space '''
    def __init__(self, boundary, class_nums):
        self.boundary = boundary
        self.class_nums = class_nums

    def predict(self, X):
        preds = np.zeros(X.shape)
        preds[X <= self.boundary] = self.class_nums[0]
        preds[X > self.boundary] = self.class_nums[1]
        return preds
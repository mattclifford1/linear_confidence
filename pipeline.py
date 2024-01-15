from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, LinearConstraint, Bounds
from sklearn.metrics import accuracy_score

import data_utils
import plots
import normal
import projection
import radius
import deltas
import optimise_contraint


def get_data_and_classifier(m1 = [1, 1],
                            m2 = [10, 10],
                            cov1 = [[1, 0], [0, 1]],
                            cov2 = [[1, 0], [0, 1]],
                            N1 = 10000,
                            N2 = 10000,
                            scale = True,
                            model='Linear',
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

    if model == 'SVM':
        clf = SVC(random_state=0, probability=True,
                kernel='linear').fit(data['X'], data['y'])
    elif model == 'Linear':
        clf = LogisticRegression().fit(data['X'], data['y'])

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

    # plot
    if _plot == True:
        ax = plots.plot_projection(proj_data, proj_means, R1_emp, R2_emp)

    if _print == True:
        print(f'R1 empirical: {R1_emp}\nR2 empirical: {R2_emp}')

    return {'projected_data': proj_data,
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
            }

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


def optimise(data_info, loss_func, contraint_func, delta1_from_delta2=None, num_deltas=1, _print=True, _plot=True):
    # get initial deltas
    delta1 = np.random.uniform()
    if num_deltas == 2:
        if delta1_from_delta2 != None:
            delta2 = delta1_from_delta2(delta1, data_info)
        else:
            # get from the contraint function
            delta1, delta2 = optimise_contraint.get_init_deltas(
                contraint_func, data_info)
        deltas_init = (delta1, delta2)
    else:
        deltas_init = (delta1) 

    if isinstance(loss_func, tuple) or isinstance(loss_func, list):
        use_grad = True
    else:
        use_grad = False

    # set up contraints
    data_info['delta2_given_delta1_func'] = delta1_from_delta2
    if num_deltas == 2:
        def contraint_wrapper(deltas):
            return contraint_func(deltas[0], deltas[1], data_info)
        print(
            f'constraint init: {contraint_wrapper(deltas_init)} should equal 0')
    else:
        def contraint_wrapper(deltas):
            delta2 = delta1_from_delta2(deltas[0], data_info)
            return contraint_func(deltas[0], delta2, data_info)

    def contraint_real(deltas):
        return np.sum(np.iscomplex(deltas))

    contrs = [
        {'type':'eq', 'fun': contraint_wrapper},
        {'type':'eq', 'fun': contraint_real},
            ]

    res = minimize(loss_func,
                   deltas_init,
                   (data_info), 
                   #    method='SLSQP',
                   bounds=Bounds([0], [1]),
                   jac=use_grad,  # use gradient
                   constraints=contrs
                   )

    deltas = res.x
    if len(deltas) == 1:
        delta1 = deltas[0]
        delta2 = delta1_from_delta2(delta1, data_info)
    else:
        delta1 = deltas[0]
        delta2 = deltas[1]

    if _print == True:
        print(f'delta1 : {delta1} \ndelta2: {delta2}')
        print(f'constraint: {contraint_wrapper(deltas)} should equal 0')
    
    if _plot == True:
        # calculate each R upper bound
        R1_est = radius.R_upper_bound(data_info['empirical R1'], data_info['R all data'], data_info['N1'], delta1)
        R2_est = radius.R_upper_bound(data_info['empirical R2'], data_info['R all data'], data_info['N2'], delta1)
        print(f'R1_est : {R1_est} \nR2_est: {R2_est}')
        ax = plots.plot_projection(data_info['projected_data'], data_info['projected_means'], R1_est, R2_est, R_est=True)
    return delta1, delta2


def eval_test(data_clf, data_info, delta1, delta2, _print=True, _plot=True):
    # calculate each R upper bound
    R1_est = radius.R_upper_bound(
        data_info['empirical R1'], data_info['R all data'], data_info['N1'], delta1)
    R2_est = radius.R_upper_bound(
        data_info['empirical R2'], data_info['R all data'], data_info['N2'], delta1)
    Rs = {'R1': R1_est, 'R2': R2_est}
    
    # est data
    if data_clf['data_test'] == None:
        print('No test data found in data_clf!')
        return
    # projected test data
    if data_info['projected_data_test'] == None:
        print('No test data found in data_info!')
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

    print(f"original accuracy: {accuracy_score(data_clf['data_test']['y'], y_clf)}")
    print(f"deltas   accuracy: {accuracy_score(data_info['projected_data_test']['y'], y_deltas)}")


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
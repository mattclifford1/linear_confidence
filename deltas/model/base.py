'''
scikit-learn style class to fit deltas
'''
import numpy as np
import os

import deltas.plotting.plots as plots
import deltas.utils.projection as projection
import deltas.utils.radius as radius
import deltas.utils.equations as ds
import deltas.optimisation.optimise_deltas as optimise_deltas


class base_deltas:
    def __init__(self, clf=None, dim_reducer=None):
        self.clf = clf
        self.dim_reducer = dim_reducer
        self._setup()

    def get_params(self, deep=True):
        # return {"clf": self.clf, "dim_reducer": self.dim_reducer}
        return {}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _setup(self):
        self.data_info_made = False
        self.is_fit = False
        # deltas optimisation functions
        self.loss_func = ds.loss_one_delta
        # for scipy minimize
        self.loss_func = (ds.loss_one_delta, ds.J_derivative)

        self.contraint_func = ds.contraint_eq7
        self.delta2_from_delta1 = ds.delta2_given_delta1_matt
        # self.delta1_from_delta2 = ds.delta1_given_delta2_matt

    def fit(self, X, y, costs=(1, 1), clf=None, _plot=False, _print=False, grid_search=True, **kwargs):
        if not isinstance(clf, type(None)):
            if not hasattr(clf, 'get_projection'):
                raise AttributeError(
                    f"Classifier {clf} needs 'get_projection' method")
            self.clf = clf
        # Make data_info - R_ests, D, etc.
        self.data_info = self.get_data_info(X, y, self.clf, costs, _print=_print)
        self.data_info_made = True

        # optimise for the deltas
        res = self._optimise(self.data_info, 
                             self.loss_func,
                             self.contraint_func,
                             self.delta2_from_delta1,
                             grid_search=grid_search,
                             _plot=_plot, 
                             _print=_print)
        self.delta1 = res['delta1']
        self.delta2 = res['delta2']
        self.solution_possible = res['solution_possible']
        self.solution_found = res['solution_found']

        # make boundary 
        self.boundary, self.class_nums = self._make_boundary(
            self.delta1, self.delta2)
        self.is_fit = True

        return self
    
    def _make_boundary(self, delta1, delta2):
        # calculate each R upper bound
        R1_est = radius.R_upper_bound(
            self.data_info['empirical R1'], self.data_info['R all data'], self.data_info['N1'], delta1)
        R2_est = radius.R_upper_bound(
            self.data_info['empirical R2'], self.data_info['R all data'], self.data_info['N2'], delta2)
        Rs = {'R1': R1_est, 'R2': R2_est}

        # add error to min class and minus from max class
        means = [self.data_info['empirical_projected_mean 1'], self.data_info['empirical_projected_mean 2']]
        min_mean = np.argmin(means)
        max_mean = np.argmax(means)
        class_names = ['1', '2']
        upper_min_class = self.data_info[f'empirical_projected_mean {class_names[min_mean]}'] + \
            Rs[f'R{class_names[min_mean]}']
        lower_max_class = self.data_info[f'empirical_projected_mean {class_names[max_mean]}'] - \
            Rs[f'R{class_names[max_mean]}']
    
        # get average as the boundary
        boundary = (upper_min_class + lower_max_class)/2
        class_nums = [0, 1]
        class_nums = [class_nums[min_mean], 
                      class_nums[max_mean]]
        return boundary, class_nums
    
    def predict(self, X):
        if self.is_fit == False:
            print("Not fit to any data yet, call 'fit(X, y)' or  method first")
            return self.clf.predict(X)
            # raise AttributeError("Not fit to any data yet, call 'fit(X, y)' or  method first")
        return self._predict(X, self.boundary, self.class_nums)
    
    def predict_proba(self, X):
        return self.clf.predict_proba(X)
        
    def _predict(self, X, boundary, class_nums):
        # project data if not already
        if X.shape[1] != 1:
            if self.clf != None:
                X = self.clf.get_projection(X)
            else:
                raise AttributeError(
                    f"Deltas classifier needs original classifier to project feature space onto 1D classification space")
        preds = np.zeros(X.shape)
        preds[X <= boundary] = class_nums[0]
        preds[X > boundary] = class_nums[1]
        return preds.squeeze()

    def _predict_given_delta1(self, X, delta1):
        delta2 = self.delta2_from_delta1(delta1, self.data_info)
        boundary, class_nums = self._make_boundary(delta1, delta2)
        return self._predict(X, boundary, class_nums)
    
    def _predict_given_delta2(self, X, delta2):
        delta1 = self.delta1_from_delta2(delta2, self.data_info)
        boundary, class_nums = self._make_boundary(delta1, delta2)
        return self._predict(X, boundary, class_nums)

    @staticmethod
    def get_data_info(X, y, clf=None, costs=(1, 1), _print=False, supports=True):
        # get projection information
        data = {'X': X, 'y': y}
        if clf != None:
            # project data according to classifier and calculate data attributes needed
            proj_data = projection.from_clf(data, clf, supports=supports)
        elif X.shape[1] == 1:
            proj_data = projection.make_calcs(data, supports=supports)
        else:
            raise AttributeError('Provide classifier to project or already projected data X (one dimensional)')
        
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

        # Empircal M
        if 'supports' in proj_data.keys():
            M_emp = np.abs(proj_data['supports'][1]-proj_data['supports'][0]).squeeze()
        else:
            M_emp = D_emp - R1_emp - R2_emp

        data_info = {'projected_data': proj_data,
                     'empirical margin': M_emp,
                     'R all data': R_sup,
                     'projected_data 1': xp1,
                     'projected_data 2': xp2,
                     'empirical_projected_mean 1': emp_xp1,
                     'empirical_projected_mean 2': emp_xp2,
                     'empirical R1': R1_emp,
                     'empirical R2': R2_emp,
                     'empirical D': D_emp,
                     'N1': (data['y'] == 0).sum(),
                     'N2': (data['y'] == 1).sum(),
                     'c1': costs[0],
                     'c2': costs[1],
                    }
        # if _print == True:
        #     print(f'R1 empirical: {R1_emp}\nR2 empirical: {R2_emp}')
        return data_info
    
    @staticmethod
    def _optimise(data_info, 
                  loss_func, 
                  contraint_func,
                  delta2_from_delta1=None,
                  grid_search=True,
                  _plot=False, 
                  _print=False):
        # optimise for the deltas.
        res = optimise_deltas.optimise(
            data_info=data_info,
            loss_func=loss_func,
            contraint_func=contraint_func,
            delta2_from_delta1=delta2_from_delta1, 
            grid_search=grid_search, 
            _print=_print, 
            _plot=_plot)
        return res

    @staticmethod
    def _plot_projection(X, y, clf):
        # use to plot without fitting
        data_info = base_deltas.get_data_info(X, y, clf)
        base_deltas._plot_data(data_info, clf)

    def plot_data(self, data=None, data_clf=None, m1=None, m2=None):
        if self.data_info_made == True:
            self._plot_data(self.data_info, self.clf, data=data, data_clf=data_clf, m1=m1, m2=m2)
        else:
            print("Not fit to any data yet, call 'fit(X, y)'  method first")

    @staticmethod
    def _plot_data(data_info, clf, data=None, data_clf=None, m1=None, m2=None, save_file=None, diagram=False):
        # project means if we have them
        if isinstance(save_file, type(None)):
            ax = plots._get_axes()
            ds = [0.9999999999999999999]
        else:
            fig, ax = plots.plt.subplots(figsize=(5.5, 2.5))
            ds = []
            # ds = [0.9999999999999999999]
        proj_means = None
        if isinstance(data_clf, dict):
            if 'mean1' in data_clf.keys() and 'mean2' in data_clf.keys():
                m1 = data_clf['mean1']
                m2 = data_clf['mean2']
                proj_means = projection.from_clf({'X': np.array([m1, m2]), 'y': [0, 1]}, clf)
        if isinstance(data, dict):
            proj_data = projection.from_clf(data, clf)
        else:
            print('plotting training data as data input')
            proj_data = data_info['projected_data']
        # plot the data
        ax, fig = plots.plot_projection(
            proj_data, 
            proj_means,
            data_info['empirical R1'],
            data_info['empirical R2'],
            deltas_to_plot=ds,
            data_info=data_info,
            D=diagram,
            )
        if isinstance(save_file, type(None)):
            plots.plt.show()
        else:
            fig.tight_layout()
            fig.savefig(save_file +'-training.png', dpi=500)




    def print_params(self):
        if self.data_info_made == True:
            print(
                f"""Parameters
                R:  {self.data_info['R all data']}
                N1: {self.data_info['N1']}
                N2: {self.data_info['N2']}
                R1: {self.data_info['empirical R1']}
                R2: {self.data_info['empirical R2']}
                M:  {self.data_info['empirical margin']}
                D:  {self.data_info['empirical D']}
                C1: {self.data_info['c1']}
                C2: {self.data_info['c2']}""")
        else:
            print("Not fit to any data yet, call 'fit(X, y)' or  method first")

    def print_deltas(self):
        if self.is_fit == True:
            print(f""""
                  delta1: {self.delta1} 
                  delta2: {self.delta2}
                  constraint: {self.contraint_func(self.delta1, self.delta2, self.data_info)}
                  """)

    def get_bias(self):
        if self.is_fit == True:
            return -self.boundary
        else:
            if hasattr(self.clf, 'get_bias'):
                print('Giving bias from original classifier')
                return self.clf.get_bias()
            else:
                print('Not fit to give bias')

    def get_projection(self, X):
        return self.clf.get_projection(X)

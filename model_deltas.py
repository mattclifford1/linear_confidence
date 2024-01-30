'''
scikit-learn style class to fit deltas
'''
import numpy as np
import data_utils
import plots
import normal
import projection
import radius
import deltas as ds
import optimise_contraint
import optimise_deltas


class base_deltas:
    def __init__(self, clf):
        if not hasattr(clf, 'get_projection'):
            raise AttributeError(f"Classifier {clf} needs 'get_projection' method")
        self.clf = clf
        self.data_info_made = False
        self.is_fit = False
        # deltas optimisation functions
        self.loss_func = ds.loss_one_delta
        self.contraint_func = ds.contraint_eq7
        self.delta1_from_delta2 = ds.delta2_given_delta1_matt


    def fit(self, X, y, costs=(1, 1), _plot=False, _print=False):
        # Make data_info - R_ests, D, etc.
        self.data_info = self.get_data_info(X, y, costs, _print=_print)

        # optimise for the deltas
        res = self._optimise(self.data_info, 
                             self.loss_func,
                             self.contraint_func,
                             self.delta1_from_delta2,
                             _plot=_plot, 
                             _print=_print)
        self.delta1, self.delta2, self.solution_possible, self.solution_found = res
        self.is_fit = True

        return self



    def get_data_info(self, X, y, costs=(1, 1), _print=False):
        # project data according to classifier and calculate data attributes needed
        data = {'X': X, 'y': y}
        proj_data = projection.from_clf(data, self.clf, supports=True)
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
        self.data_info_made = True
        return data_info
    
    def _optimise(self, 
                  data_info, 
                  loss_func, 
                  contraint_func,
                  delta1_from_delta2=None,
                  num_deltas=1, 
                  grid_search=True,
                  _plot=False, 
                  _print=False):
        # optimise for the deltas. N.B. keep data_info as arg for flexibility
        res = optimise_deltas.optimise(
            data_info, 
            loss_func, 
            contraint_func, 
            delta1_from_delta2, 
            num_deltas, 
            grid_search, 
            _print, 
            _plot)

        return res


    def plot_data(self, m1, m2):
        if self.data_info_made == True:
            proj_means = projection.from_clf({'X': np.array([m1, m2]), 'y': [0, 1]}, self.clf)
            _ = plots.plot_projection(
                self.data_info['projected_data'], 
                proj_means,
                self.data_info['empirical R1'],
                self.data_info['empirical R1'],
                data_info=self.data_info)
        else:
            print("Not fit to any data yet, call 'fit(X, y)' or  method first")


    def print_params(self):
        if self.data_info_made == True:
            print(
                f"""Parameters
                R:  {self.data_info['R all data']}
                N1: {self.data_info['N1']}
                N2: {self.data_info['N1']}
                R1: {self.data_info['empirical R1']}
                R2: {self.data_info['empirical R2']}
                M:  {self.data_info['empirical margin']}
                D:  {self.data_info['empirical D']}
                C1: {self.data_info['c1']}
                C2: {self.data_info['c2']}""")
        else:
            print("Not fit to any data yet, call 'fit(X, y)' or  method first")

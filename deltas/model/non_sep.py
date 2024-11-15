'''
scikit-learn style class to fit deltas in the overlapping case
'''
import numpy as np
import os

import deltas.plotting.plots as plots
import deltas.utils.projection as projection
import deltas.utils.radius as radius
import deltas.utils.equations as ds
import deltas.optimisation.optimise_deltas as optimise_deltas


class deltas:
    def __init__(self, clf=None, dim_reducer=None):
        if not hasattr(clf, 'get_projection'):
            raise AttributeError(
                f"Classifier {clf} needs 'get_projection' method")
        self.clf = clf
        self.dim_reducer = dim_reducer
        self._setup()
        

    def _setup(self):
        self.data_info_made = False
        self.is_fit = False

    def fit(self, X, y, costs=(1, 1), **kwargs):
        self.data_info = data_info(X, y, self.clf)
        self.costs = costs
        self.is_fit = True

        return self

    def get_loss(self, bias):
        ''' get the loss as a specific bias point'''

        pass

    def get_generalisation_error(self, X_t, cls=1):
        '''X_t is the test data points or current bias term
        and we want to give the error term from the empirical mean before
        taking into account the error around the empirical mean vs expectation'''
        if not self.is_fit:
            raise AttributeError("Call .fit(X, y) first")
        # get vars we need
        mean = self.data_info(f'emp_xp{cls}')
        # add on the min concentration inequality error onto the dists as thats the dist we care about
        d_train = self.data_info(f'd_{cls}') + self.data_info(f'min_conc_{cls}')
        N = self.data_info(f'N{cls}')
        # get the distance of test point to mean
        d_test = np.abs(X_t - mean)
        # where the test point vs train points
        d_comp = d_test > d_train

        # if the test point is further away than the furthest train point
        if d_comp.all() == True:
            # old error term -- is the beyond all train points
            k_furthest = 1
        else:
            # argmin will give the index of the first False value
            k_furthest = N + 1 - np.argmin(d_comp)
        # equation 1 (but new version)
        error = k_furthest/(N+1)

        # get the training point we are calculating the error from
        ind = N - k_furthest
        if ind == -1: # point closer than all training points
            dist_add = 0
        else:
            dist_add = d_train[ind]
        # get the point we are using
        if mean < X_t:
            point = mean + dist_add
        else:
            point = mean - dist_add
        print(f'k: {k_furthest}, point: {point}')
        return error, point


class data_info:
    def __init__(self, X, y, clf):
        '''
        Calculate the variables needed for the optimisation
         - R_ests, D, etc.
        '''
        data = {'X': X, 'y': y}
        # get projection
        self.proj_data = projection.from_clf(data, clf, supports=False)
        # split into classes
        self.xp1, self.xp2 = self.proj_data['X1'], self.proj_data['X2']
        # calc empirical means
        self.emp_xp1 = np.mean(self.xp1)
        self.emp_xp2 = np.mean(self.xp2)
        # distances from means
        self.d_1 = self.order_distances_from_mean(
            self.emp_xp1, self.xp1)
        self.d_2 = self.order_distances_from_mean(
            self.emp_xp2, self.xp2)
        # class Ns
        self.N1, self.N2 = len(self.xp1), len(self.xp2)
        # calc radius estimates
        self.R1_emp = radius.supremum(self.xp1, self.emp_xp1)
        self.R2_emp = radius.supremum(self.xp2, self.emp_xp2)
        # Empirical D
        self.D_emp = np.abs(self.emp_xp1 - self.emp_xp2)
        # stability numbers -- min error on empirical mean
        self.min_conc_1 = radius.error_upper_bound(self.R1_emp, self.N1, 0.9999999999999999)
        self.min_conc_2 = radius.error_upper_bound(self.R2_emp, self.N2, 0.9999999999999999)

    def order_distances_from_mean(self, mean, X):
        dists = np.sqrt(np.sum(np.square(X - mean), axis=1))
        return np.sort(dists)


    def __repr__(self):
        print_info = "Data Info contains the following attributes:\n"
        for key in self.__dict__:
            print_info += f"    {key}\n"
        return print_info

    def __call__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        else:
            raise AttributeError(f"{attr} not in data_info.\n {self}")
        
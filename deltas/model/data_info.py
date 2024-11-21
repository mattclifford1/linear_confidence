'''
make the info about the training data to calculate concentration inequalities and generalisation error
'''
import numpy as np

import deltas.utils.projection as projection
import deltas.utils.radius as radius
from deltas.misc.use_two import USE_TWO

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
        if USE_TWO == True:
            factor = 2
        else:   
            factor = 1
        self.min_conc_1 = factor*(self.R1_emp/np.sqrt(self.N1))
        self.min_conc_2 = factor*(self.R2_emp/np.sqrt(self.N2))
        # self.min_conc_1 = radius.error_upper_bound(
        #     self.R1_emp, self.N1, 0.999999999999999999999999999999999999)
        # self.min_conc_2 = radius.error_upper_bound(
        #     self.R2_emp, self.N2, 0.999999999999999999999999999999999999)

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

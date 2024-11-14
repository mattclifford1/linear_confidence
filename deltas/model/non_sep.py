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
        self._calc_vars(X, y)
        self.costs = costs

    def _calc_vars(self, X, y):
        '''
        Calculate the variables needed for the optimisation
         - R_ests, D, etc.
        '''
        data = {'X': X, 'y': y}
        # get projection
        proj_data = projection.from_clf(data, self.clf, supports=False)
        # split into classes
        self.xp1, self.xp2 = proj_data['X1'], proj_data['X2']
        # calc empirical means
        self.emp_xp1 = np.mean(self.xp1)
        self.emp_xp2 = np.mean(self.xp2)
        # calc radius estimates
        self.R1_emp = radius.supremum(self.xp1, self.emp_xp1)
        self.R2_emp = radius.supremum(self.xp2, self.emp_xp2)
        # Empirical D
        self.D_emp = np.abs(self.emp_xp1 - self.emp_xp2)

    def get_loss(self):
        pass


# Wrappers for costlca to make it conform to sklearn styling
# Author: Matt Clifford
import numpy as np
from deltas.costcla_local.directcost import BayesMinimumRiskClassifier, ThresholdingOptimization

''' Methods from costcla to compare against 
******************************
cost_mat: array-like of shape = [n_samples, 4]
                Cost matrix of the classification problem
                Where the columns represents the costs of: false positives, false negatives,
                true positives and true negatives, for each example.

                -> usually cost of true pos and true neg is 0
                -> see minute 12 onwards of https://www.youtube.com/watch?v=UUVRdRpPhJU&ab_channel=PyData for more info
'''

class BMR(BayesMinimumRiskClassifier):
    ''' docs say to use test set for fitting ... '''

    def __init__(self, clf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clf = clf

    def get_projection(self, *args, **kwargs):
        return self.clf.get_projection(*args, **kwargs)

    def get_bias(self, *args, **kwargs):
        return self.clf.get_bias(*args, **kwargs)

    def predict(self, X, cost_mat=None):
        y_prob_pred = self.clf.predict_proba(X)
        if isinstance(cost_mat, type(None)):
            cost_mat = np.ones([y_prob_pred.shape[0], 4])
            cost_mat[:, 2] = 0.0
            cost_mat[:, 3] = 0.0
        return super().predict(y_prob_pred, cost_mat)

    def fit(self, X, y):
        y_prob_train = self.clf.predict_proba(X)
        super().fit(y, y_prob_train)
        return self


class Thresholding(ThresholdingOptimization):
    def __init__(self, clf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clf = clf

    def get_projection(self, *args, **kwargs):
        return self.clf.get_projection(*args, **kwargs)

    def get_bias(self, *args, **kwargs):
        return self.clf.get_bias(*args, **kwargs)

    def predict(self, X):
        y_prob_pred = self.clf.predict_proba(X)
        return super().predict(y_prob_pred)

    def fit(self, X, y, cost_train=None):
        ''' use train for X and y '''
        y_prob_pred = self.clf.predict_proba(X)
        if isinstance(cost_train, type(None)):
            cost_train = np.ones([y_prob_pred.shape[0], 4])
            cost_train[:, 2] = 0.0
            cost_train[:, 3] = 0.0

        return super().fit(y_prob=y_prob_pred, cost_mat=cost_train, y_true=y)

'''
selection of models that follow sklearn schematic but are extended to
have get_projection method to use for deltas
'''
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

class SVM(SVC):
    def __init__(self, kernel='linear', **kwargs):
        super().__init__(random_state=0, probability=True,
                       kernel=kernel, **kwargs)
        
    def get_projection(self, X):
        projected = np.dot(X, self.coef_.T)/np.linalg.norm(self.coef_.T)
        return projected
        
class linear(LogisticRegression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_projection(self, X):
        projected = np.dot(X, self.coef_.T)/np.linalg.norm(self.coef_.T)
        return projected

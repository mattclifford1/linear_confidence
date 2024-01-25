'''
selection of models that follow sklearn schematic but are extended to
have get_projection method to use for deltas
'''
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network._multilayer_perceptron import safe_sparse_dot, ACTIVATIONS

class SVM(SVC):
    def __init__(self, kernel='linear', **kwargs):
        super().__init__(random_state=0, probability=True,
                       kernel=kernel, **kwargs)
        
    def get_projection(self, X):
        # TODO: we need to also use the kernel if not linear kernel?
        projected = np.dot(X, self.coef_.T)/np.linalg.norm(self.coef_.T)
        return projected
        

class linear(LogisticRegression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_projection(self, X):
        # normalised projection
        projected = np.dot(X, self.coef_.T)/np.linalg.norm(self.coef_.T)
        return projected
    

class NN(MLPClassifier):
    def __init__(self, class_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weight = class_weight #TODO: implement class balancing for NN (over samplling on fit method?)

    def get_projection(self, X, check_input=True):
        # adapted from https://github.com/scikit-learn/scikit-learn/blob/9e38cd00d032f777312e639477f1f52f3ea4b3b7/sklearn/neural_network/_multilayer_perceptron.py#L187
        if check_input:
            X = self._validate_data(
                X, accept_sparse=["csr", "csc"], reset=False)

        # Initialize first layer
        activation = X

        # Forward propagate
        hidden_activation = ACTIVATIONS[self.activation]
        # compute all until the last layer
        for i in range(self.n_layers_ - 2): 
            activation = safe_sparse_dot(activation, self.coefs_[i])
            activation += self.intercepts_[i]
            hidden_activation(activation)

        # get projection from last layer
        projected = safe_sparse_dot(activation, self.coefs_[-1])/np.linalg.norm(self.coefs_[-1].T)
        return projected

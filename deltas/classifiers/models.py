'''
selection of models that follow sklearn schematic but are extended to
have get_projection method to use for deltas
'''
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network._multilayer_perceptron import safe_sparse_dot, ACTIVATIONS
from imblearn.over_sampling import RandomOverSampler


class SVM(SVC):
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', ** kwargs):
        self.kernel = kernel
        super().__init__(random_state=0, probability=True,
                         kernel=self.kernel, 
                         C=C,
                         gamma=gamma,
                         **kwargs)
        
    def get_projection(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if self.kernel == 'linear':
            X_kern = self._compute_kernel(X)  # dont actually needed (is identity for linear)
            projected = np.dot(X_kern, self.coef_.T) / np.linalg.norm(self.coef_.T)
        else:
            # use self.dual_coef_ in the kernel form

            # simpler format below as otherwise can do dig around in libSVM
            # can use this for linear too in the future
            projected = self.decision_function(X) - self.intercept_
            projected = np.expand_dims(projected, axis=1)
        return projected
    
    def get_bias(self):
        return self.intercept_


class linear(LogisticRegression):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_projection(self, X):
        # normalised projection
        projected = np.dot(X, self.coef_.T)/np.linalg.norm(self.coef_.T)
        return projected
    
    def get_bias(self):
        return self.intercept_
    

class NN(MLPClassifier):
    def __init__(self, hidden_layer_sizes=(20, 50, 100,), 
                 class_weight=None, 
                 max_iter=500, 
                 solver='adam',
                 activation='relu',
                 learning_rate_init=0.0001,
                 **kwargs):
        super().__init__(hidden_layer_sizes=hidden_layer_sizes, 
                         max_iter=max_iter, 
                         solver=solver,
                         activation=activation,
                         learning_rate_init=learning_rate_init,
                         **kwargs)
        self.class_weight = class_weight

    def fit(self, X, y, *args, **kwargs):
        if self.class_weight == 'balanced':
            ROS = RandomOverSampler(random_state=0)
            X, y = ROS.fit_resample(X, y)
        return super().fit(X, y, *args, **kwargs)

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
    
    def get_bias(self):
        return self.intercepts_[-1]


class delta_adjusted_clf:
    ''' boundary to make decision in projected space '''

    def __init__(self, boundary, class_nums, clf=None):
        self.boundary = boundary
        self.class_nums = class_nums
        self.clf = clf

    def predict(self, X):
        # project data if not already
        if X.shape[1] != 1:
            if self.clf != None:
                X = self.clf.get_projection(X)
            else:
                raise AttributeError(f"Deltas classifier needs original classifier to project feature space onto 1D classification space")
        preds = np.zeros(X.shape)
        preds[X <= self.boundary] = self.class_nums[0]
        preds[X > self.boundary] = self.class_nums[1]
        return preds
    
    def get_bias(self):
        return self.boundary

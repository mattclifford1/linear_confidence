'''
selection of models that follow sklearn schematic but are extended to
have get_projection method to use for deltas

History worth knowing: this module used to carry ~350 lines of vendored
sklearn 1.3.x MLPClassifier internals (`_fit_weighted`,
`_fit_stochastic_weighted`, `_backprop_weighted`, plus weighted loss
functions), because upstream MLPClassifier.fit did not accept sample_weight
and the MIMIC "Balanced Weights" baseline needs it. That pinned the whole repo
to sklearn 1.3.2. scikit-learn#25646 landed the feature upstream, so the copy
is gone and `class_weight='balanced'` is now an ordinary weighted fit. The
algorithm is unchanged - it is the same weighted backprop, just maintained by
sklearn rather than here.

For new work prefer the sibling `projection_models` package, which provides the
same `get_projection` interface over a much wider set of model families. These
classes remain because the ECAI pipeline and its cached artefacts refer to
them.
'''
import warnings

import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import validate_data


# sklearn keeps its activations in the private sklearn.neural_network._base.
# They are four lines of arithmetic and the whole point of this file is to stop
# depending on sklearn internals, so they live here. In-place to match
# sklearn's own convention, which the forward pass in NN.get_projection relies
# on.
def _identity(X):
    return X


def _logistic(X):
    np.negative(X, out=X)
    np.exp(X, out=X)
    np.add(X, 1, out=X)
    np.reciprocal(X, out=X)
    return X


def _tanh(X):
    np.tanh(X, out=X)
    return X


def _relu(X):
    np.maximum(X, 0, out=X)
    return X


ACTIVATIONS = {'identity': _identity, 'logistic': _logistic, 'tanh': _tanh,
               'relu': _relu}


class SVM(SVC):
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', random_state=0, class_weight=None, **kwargs):
        self.kernel = kernel
        super().__init__(probability=True,
                         kernel=self.kernel, 
                         C=C,
                         gamma=gamma,
                         random_state=random_state,
                         class_weight=class_weight,
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
                raise AttributeError(
                    f"Deltas classifier needs original classifier to project feature space onto 1D classification space")
        preds = np.zeros(X.shape)
        preds[X <= self.boundary] = self.class_nums[0]
        preds[X > self.boundary] = self.class_nums[1]
        return preds

    def get_bias(self):
        return self.boundary
    

class NN(MLPClassifier):
    def __init__(self, 
                #  hidden_layer_sizes=(20, 50, 100,), 
                 hidden_layer_sizes=(20, 50,), 
                #  hidden_layer_sizes=(10, 20, 50, 20, 10), 
                 class_weight=None, 
                 max_iter=500, 
                 solver='adam',
                 learning_rate='constant',
                 activation='relu',
                 learning_rate_init=0.0001,
                 random_state=42,
                 **kwargs):
        super().__init__(hidden_layer_sizes=hidden_layer_sizes, 
                         max_iter=max_iter, 
                         solver=solver,
                         activation=activation,
                         learning_rate=learning_rate,
                         learning_rate_init=learning_rate_init,
                         random_state=random_state,
                        #  early_stopping=True,
                         **kwargs)
        self.class_weight = class_weight


    def fit(self, X, y, *args, **kwargs):
        # sklearn >= 1.7 supports sample_weight in MLPClassifier.fit
        # (scikit-learn#25646), so class_weight='balanced' is just a weighted
        # fit. This used to require ~350 lines of vendored MLP internals; see
        # the note in the module docstring.
        if self.class_weight == 'balanced':
            kwargs['sample_weight'] = compute_sample_weight('balanced', y)
        return super().fit(X, y, *args, **kwargs)

    def get_projection(self, X, check_input=True):
        # adapted from https://github.com/scikit-learn/scikit-learn/blob/9e38cd00d032f777312e639477f1f52f3ea4b3b7/sklearn/neural_network/_multilayer_perceptron.py#L187
        if check_input:
            X = validate_data(
                self, X, accept_sparse=["csr", "csc"], reset=False)

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

'''
selection of models that follow sklearn schematic but are extended to
have get_projection method to use for deltas
'''
import warnings

import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.neural_network._multilayer_perceptron import safe_sparse_dot, ACTIVATIONS, DERIVATIVES, _STOCHASTIC_SOLVERS
from sklearn.neural_network._stochastic_optimizers import AdamOptimizer, SGDOptimizer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import _check_sample_weight, check_random_state
from sklearn.base import is_classifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle, gen_batches, _safe_indexing

from scipy.special import xlogy
from itertools import chain

from imblearn.over_sampling import RandomOverSampler


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
        # need to handle balanced class_weights as current sklearn doesn't support it but will do in the future see: https://github.com/scikit-learn/scikit-learn/pull/25326 
        if self.class_weight == 'balanced':
            sample_weights = compute_sample_weight('balanced', y)
            return self._fit_weighted(X, y, sample_weight=sample_weights)
            # ROS = RandomOverSampler(random_state=0)
            # X, y = ROS.fit_resample(X, y)
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
    
    # CLASS WEIGHT ADJUSTMENT TO THE LOSS FUNCTION
    # need to handle balanced class_weights as current sklearn doesn't support it but will do in the future see:
    # https://github.com/scikit-learn/scikit-learn/pull/25326 (class) and
    # https://github.com/scikit-learn/scikit-learn/pull/25646 (sample)
    # change the backprop call to add sample weights locally
    # N.B. we only care about log_loss for classification
    def _fit_weighted(self, X, y, incremental=False, sample_weight=None):
            # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        if np.any(np.array(hidden_layer_sizes) <= 0):
            raise ValueError(
                "hidden_layer_sizes must be > 0, got %s." % hidden_layer_sizes
            )
        first_pass = not hasattr(self, "coefs_") or (
            not self.warm_start and not incremental
        )

        X, y = self._validate_input(X, y, incremental, reset=first_pass)
        # Handle sample_weight
        if sample_weight is not None:
            sample_weight = _check_sample_weight(
                sample_weight, X, dtype=X.dtype)
            if sum(sample_weight != 0) == 0:
                raise ValueError("sample_weight must not be all zeros")

        _, n_features = X.shape

        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape((-1, 1))

        self.n_outputs_ = y.shape[1]

        layer_units = [n_features] + hidden_layer_sizes + [self.n_outputs_]

        # check random state
        self._random_state = check_random_state(self.random_state)

        if first_pass:
            # First time training the model
            self._initialize(y, layer_units, X.dtype)

        # Initialize lists
        activations = [X] + [None] * (len(layer_units) - 1)
        deltas = [None] * (len(activations) - 1)

        coef_grads = [
            np.empty((n_fan_in_, n_fan_out_), dtype=X.dtype)
            for n_fan_in_, n_fan_out_ in zip(layer_units[:-1], layer_units[1:])
        ]

        intercept_grads = [
            np.empty(n_fan_out_, dtype=X.dtype) for n_fan_out_ in layer_units[1:]
        ]

        # Run the Stochastic optimization solver
        if self.solver in _STOCHASTIC_SOLVERS:
            self._fit_stochastic_weighted(
                X,
                y,
                activations,
                deltas,
                coef_grads,
                intercept_grads,
                layer_units,
                incremental,
                sample_weight,
            )

        # Run the LBFGS solver
        elif self.solver == "lbfgs":
            self._fit_lbfgs(
                X,
                y,
                activations,
                deltas,
                coef_grads,
                intercept_grads,
                layer_units,
                sample_weight,
            )

        # validate parameter weights
        weights = chain(self.coefs_, self.intercepts_)
        if not all(np.isfinite(w).all() for w in weights):
            raise ValueError(
                "Solver produced non-finite parameter weights. The input data may"
                " contain large values and need to be preprocessed."
            )

        return self
    
    def _fit_stochastic_weighted(
        self,
        X,
        y,
        activations,
        deltas,
        coef_grads,
        intercept_grads,
        layer_units,
        incremental,
        sample_weight,
    ):
        params = self.coefs_ + self.intercepts_
        if not incremental or not hasattr(self, "_optimizer"):
            if self.solver == "sgd":
                self._optimizer = SGDOptimizer(
                    params,
                    self.learning_rate_init,
                    self.learning_rate,
                    self.momentum,
                    self.nesterovs_momentum,
                    self.power_t,
                )
            elif self.solver == "adam":
                self._optimizer = AdamOptimizer(
                    params,
                    self.learning_rate_init,
                    self.beta_1,
                    self.beta_2,
                    self.epsilon,
                )

        # early_stopping in partial_fit doesn't make sense
        if self.early_stopping and incremental:
            raise ValueError(
                "partial_fit does not support early_stopping=True")
        early_stopping = self.early_stopping
        sample_weight_val = None
        if early_stopping:
            # don't stratify in multilabel classification
            should_stratify = is_classifier(self) and self.n_outputs_ == 1
            stratify = y if should_stratify else None
            if sample_weight is None:
                X, X_val, y, y_val = train_test_split(
                    X,
                    y,
                    random_state=self._random_state,
                    test_size=self.validation_fraction,
                    stratify=stratify,
                )
            else:
                X, X_val, y, y_val, sample_weight, sample_weight_val = train_test_split(
                    X,
                    y,
                    sample_weight,
                    random_state=self._random_state,
                    test_size=self.validation_fraction,
                    stratify=stratify,
                )
            if is_classifier(self):
                y_val = self._label_binarizer.inverse_transform(y_val)
        else:
            X_val = None
            y_val = None

        n_samples = X.shape[0]
        sample_idx = np.arange(n_samples, dtype=int)

        if self.batch_size == "auto":
            batch_size = min(200, n_samples)
        else:
            if self.batch_size > n_samples:
                warnings.warn(
                    "Got `batch_size` less than 1 or larger than "
                    "sample size. It is going to be clipped"
                )
            batch_size = np.clip(self.batch_size, 1, n_samples)

        try:
            self.n_iter_ = 0
            for it in range(self.max_iter):
                if self.shuffle:
                    # Only shuffle the sample indices instead of X and y to
                    # reduce the memory footprint. These indices will be used
                    # to slice the X and y.
                    sample_idx = shuffle(
                        sample_idx, random_state=self._random_state)

                accumulated_loss = 0.0
                for batch_slice in gen_batches(n_samples, batch_size):
                    batch_idx = sample_idx[batch_slice] if self.shuffle else batch_slice
                    X_batch = _safe_indexing(X, batch_idx)
                    y_batch = y[batch_idx]
                    sw_batch = (
                        None if sample_weight is None else sample_weight[batch_idx]
                    )

                    activations[0] = X_batch
                    batch_loss, coef_grads, intercept_grads = self._backprop_weighted(
                        X_batch,
                        y_batch,
                        activations,
                        deltas,
                        coef_grads,
                        intercept_grads,
                        sw_batch,
                    )
                    accumulated_loss += batch_loss * (
                        batch_slice.stop - batch_slice.start
                    )

                    # update weights
                    grads = coef_grads + intercept_grads
                    self._optimizer.update_params(params, grads)

                self.n_iter_ += 1
                self.loss_ = accumulated_loss / X.shape[0]

                self.t_ += n_samples
                self.loss_curve_.append(self.loss_)
                if self.verbose:
                    print("Iteration %d, loss = %.8f" %
                          (self.n_iter_, self.loss_))

                # update no_improvement_count based on training loss or
                # validation score according to early_stopping
                self._update_no_improvement_count(
                    early_stopping,
                    X_val,
                    y_val,
                    # sample_weight_val,
                )

                # for learning rate that needs to be updated at iteration end
                self._optimizer.iteration_ends(self.t_)

                if self._no_improvement_count > self.n_iter_no_change:
                    # not better than last `n_iter_no_change` iterations by tol
                    # stop or decrease learning rate
                    if early_stopping:
                        msg = (
                            "Validation score did not improve more than "
                            "tol=%f for %d consecutive epochs."
                            % (self.tol, self.n_iter_no_change)
                        )
                    else:
                        msg = (
                            "Training loss did not improve more than tol=%f"
                            " for %d consecutive epochs."
                            % (self.tol, self.n_iter_no_change)
                        )

                    is_stopping = self._optimizer.trigger_stopping(
                        msg, self.verbose)
                    if is_stopping:
                        break
                    else:
                        self._no_improvement_count = 0

                if incremental:
                    break
        except KeyboardInterrupt:
            warnings.warn("Training interrupted by user.")

        if early_stopping:
            # restore best weights
            self.coefs_ = self._best_coefs
            self.intercepts_ = self._best_intercepts

    
    def _backprop_weighted(
        self, X, y, activations, deltas, coef_grads, intercept_grads, sample_weight=None
    ):
        n_samples = X.shape[0]

        # Forward propagate
        activations = self._forward_pass(activations)

        # Get loss
        loss_func_name = self.loss
        if loss_func_name == "log_loss" and self.out_activation_ == "logistic":
            loss_func_name = "binary_log_loss"

        loss = LOSS_FUNCTIONS_WEIGHTED[loss_func_name](
            y, activations[-1], sample_weight=sample_weight
        )
        # Add L2 regularization term to loss
        values = 0
        for s in self.coefs_:
            s = s.ravel()
            values += np.dot(s, s)
        loss += (0.5 * self.alpha) * values / n_samples

        # Backward propagate
        last = self.n_layers_ - 2

        # The calculation of delta[last] here works with following
        # combinations of output activation and loss function:
        # sigmoid and binary cross entropy, softmax and categorical cross
        # entropy, and identity with squared loss
        deltas[last] = activations[-1] - y
        if sample_weight is not None:
            if len(sample_weight.shape) == 1:
                sample_weight = np.expand_dims(sample_weight, axis=1)
            deltas[last] *= sample_weight

        # Compute gradient for the last layer
        self._compute_loss_grad(
            last, n_samples, activations, deltas, coef_grads, intercept_grads
        )

        inplace_derivative = DERIVATIVES[self.activation]
        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 2, 0, -1):
            deltas[i - 1] = safe_sparse_dot(deltas[i], self.coefs_[i].T)
            inplace_derivative(activations[i], deltas[i - 1])

            self._compute_loss_grad(
                i - 1, n_samples, activations, deltas, coef_grads, intercept_grads
            )

        return loss, coef_grads, intercept_grads


# https://github.com/scikit-learn/scikit-learn/blob/676eb48604b188c4f8d835c24cd398a5aa38c9aa/sklearn/neural_network/_base.py 
def log_loss_weighted(y_true, y_prob, sample_weight=None):
    eps = np.finfo(y_prob.dtype).eps
    y_prob = np.clip(y_prob, eps, 1 - eps)
    if y_prob.shape[1] == 1:
        y_prob = np.append(1 - y_prob, y_prob, axis=1)

    if y_true.shape[1] == 1:
        y_true = np.append(1 - y_true, y_true, axis=1)

    temp = xlogy(y_true, y_prob)

    return -np.average(temp, weights=sample_weight, axis=0).sum()

# https://github.com/scikit-learn/scikit-learn/blob/676eb48604b188c4f8d835c24cd398a5aa38c9aa/sklearn/neural_network/_base.py
def binary_log_loss_weighted(y_true, y_prob, sample_weight=None):
    eps = np.finfo(y_prob.dtype).eps
    y_prob = np.clip(y_prob, eps, 1 - eps)

    temp = xlogy(y_true, y_prob) + xlogy(1 - y_true, 1 - y_prob)

    return -np.average(temp, weights=sample_weight, axis=0).sum()


LOSS_FUNCTIONS_WEIGHTED = {
    "log_loss": log_loss_weighted,
    "binary_log_loss": binary_log_loss_weighted,
}

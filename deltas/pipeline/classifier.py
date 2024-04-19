import numpy as np
# from costcla.models.directcost import BayesMinimumRiskClassifier
from deltas.costcla_local.directcost import BayesMinimumRiskClassifier, ThresholdingOptimization
import deltas.plotting.plots as plots
import deltas.classifiers.models as models
import deltas.pipeline.data as pipe_data


def get_classifier(data_clf, model='Linear', balance_clf=False, costcla_methods=True, _plot=True, _plot_data=False):
    data = data_clf['data']

    # dim reducer (PCA) for plotting in higher dims
    if data['X'].shape[1] > 2:
        if 'dim_reducer' in data_clf.keys():
            dim_reducer = data_clf['dim_reducer']
        else:
            dim_reducer = pipe_data.get_dim_reducer(data)
    else:
        dim_reducer = None

    # SMOTE ==================================================================
    SMOTE_data = pipe_data.get_SMOTE_data(data)

    # Weighted Classifier ====================================================
    if balance_clf == True:
        weights = 'balanced'
    else:
        weights = None

    # Train Model ============================================================
    if model in ['SVM', 'SVM-linear']:
        clf = models.SVM(kernel='linear', class_weight=weights).fit(
            data['X'], data['y'])
        clf_SMOTE = models.SVM(kernel='linear', class_weight=weights).fit(
            SMOTE_data['X'], SMOTE_data['y'])
    elif model == 'SVM-rbf':
        clf = models.SVM(kernel='rbf', class_weight=weights).fit(
            data['X'], data['y'])
        clf_SMOTE = models.SVM(kernel='rbf', class_weight=weights).fit(
            SMOTE_data['X'], SMOTE_data['y'])
    elif model == 'Linear':
        clf = models.linear(class_weight=weights).fit(data['X'], data['y'])
        clf_SMOTE = models.linear(class_weight=weights).fit(
            SMOTE_data['X'], SMOTE_data['y'])
    elif model == 'MLP':
        clf = models.NN(class_weight=weights).fit(data['X'], data['y'])
        clf_SMOTE = models.NN(class_weight=weights).fit(
            SMOTE_data['X'], SMOTE_data['y'])
    else:
        raise ValueError(f"model: {model} not in list of available models")
    
    # Model adjsutment methods from the literature ===========================
    clfs = {'original': clf, 'SMOTE': clf_SMOTE}
    if costcla_methods == True:
        # Bayes Minimum Risk 
        # docs say to use test data but that is cheating...?
        # maybe use the training data as that is fair??? - performs poorly on without test ...
        data_test = data_clf['data_test']
        X = data_test['X']
        y = data_test['y']
        # X = data['X']
        # y = data['y']
        clf_bmr = BMR(clf).fit(X, y)

        # Thresholding
        clf_tresh = Thresh(clf).fit(data['X'], data['y'])

        clfs['Bayes Minimum Risk'] = clf_bmr
        clfs['Thresholding'] = clf_tresh
    # PLOT ===================================================================
    # data for plotting purposes only
    train_data = {'original': data, 'SMOTE': SMOTE_data}
    for name, classif in clfs.items():
        if name in train_data.keys():
            data_plot = train_data[name]
        else:
            data_plot = train_data['original']
        if _plot_data == True:
            print(name)
            ax, _ = plots._get_axes(None)
            plots.plot_classes(data_plot, ax=ax, dim_reducer=dim_reducer)
            ax.set_title(name)
            plots.plt.show()
        if _plot == True:
            print(name)
            ax, _ = plots._get_axes(None)
            plots.plot_classes(data_plot, ax=ax, dim_reducer=dim_reducer)
            plots.plot_decision_boundary(
                classif, data_plot, ax=ax, probs=False, dim_reducer=dim_reducer)
            ax.set_title(name)
            plots.plt.show()

    return clfs



''' Methods from costcla to compare against '''
class BMR(BayesMinimumRiskClassifier):
    ''' docs say to use test set for fitting ... '''
    def __init__(self, clf):
        super().__init__()
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
    

class Thresh(ThresholdingOptimization):
    def __init__(self, clf):
        super().__init__()
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
    


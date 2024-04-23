import numpy as np
from sklearn.model_selection import GridSearchCV


# from costcla.models.directcost import BayesMinimumRiskClassifier
from deltas.costcla_local.models import BMR, Thresholding
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


    # Train Model ============================================================
    if model in ['SVM', 'SVM-linear']:
        clf = models.SVM(kernel='linear').fit(data['X'], data['y'])
        clf_weighted = models.SVM(class_weight='balanced', kernel='linear').fit(data['X'], data['y'])
        clf_SMOTE = models.SVM(kernel='linear').fit(SMOTE_data['X'], SMOTE_data['y'])
    elif model == 'SVM-rbf':
        # defining parameter range
        param_grid = {'C': [0.1, 1, 10, 100, 1000, 10000],
                    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                    'kernel': ['rbf']}

        grid = GridSearchCV(models.SVM(), param_grid, refit=True, n_jobs=-1)
        # fitting the model for grid search
        print('Tuning SVM params with 5 fold CV')
        grid.fit(data['X'], data['y'])
        clf = grid.best_estimator_
        print(f'Best SVM params: {grid.best_params_}')
        # clf = models.SVM(kernel='rbf').fit( data['X'], data['y'])
        clf_weighted = models.SVM(class_weight='balanced', kernel='rbf').fit( data['X'], data['y'])
        clf_SMOTE = models.SVM(kernel='rbf').fit(SMOTE_data['X'], SMOTE_data['y'])
    elif model == 'Linear':
        clf = models.linear().fit(data['X'], data['y'])
        clf_weighted = models.linear(class_weight='balanced').fit(data['X'], data['y'])
        clf_SMOTE = models.linear().fit( SMOTE_data['X'], SMOTE_data['y'])
    elif model == 'MLP':
        clf = models.NN().fit(data['X'], data['y'])
        clf_weighted = models.NN(class_weight='balanced').fit(data['X'], data['y'])
        clf_SMOTE = models.NN().fit(SMOTE_data['X'], SMOTE_data['y'])
    elif model == 'MLP-deep':
        clf = models.NN(hidden_layer_sizes=(10, 20, 50, 50, 50, 100, 200, 300,), activation='relu', solver='adam'
                        ).fit(data['X'], data['y'])
        clf_weighted = models.NN(class_weight='balanced').fit(data['X'], data['y'])
        clf_SMOTE = models.NN().fit(SMOTE_data['X'], SMOTE_data['y'])
    else:
        raise ValueError(f"model: {model} not in list of available models")
    
    # Model adjsutment methods from the literature ===========================
    clfs = {'Original': clf, 'SMOTE': clf_SMOTE,
            'Balanced Weights': clf_weighted}
    if costcla_methods == True:
        # Bayes Minimum Risk 
        X = data['X']
        y = data['y']
        clf_bmr = BMR(clf).fit(X, y)
        clf_bmr_non_cal = BMR(clf, calibration=False).fit(X, y)

        # Thresholding
        clf_tresh = Thresholding(clf, calibration=False).fit(data['X'], data['y'])

        clfs['Bayes Minimum Risk (calibrated)'] = clf_bmr
        clfs['Bayes Minimum Risk'] = clf_bmr_non_cal
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
   


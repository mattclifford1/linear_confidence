import numpy as np
from sklearn.model_selection import GridSearchCV


# from costcla.models.directcost import BayesMinimumRiskClassifier
from deltas.costcla_local.models import BMR, Thresholding
from deltas.classifiers.large_margin_train import LargeMarginClassifier
import deltas.plotting.plots as plots
import deltas.classifiers.models as models
import deltas.pipeline.data as pipe_data


def get_classifier(data_clf, model='Linear', balance_clf=False, costcla_methods=True, binary=True, epochs=2, _plot=True, _plot_data=False):
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


    weighted = True
    # Train Model ============================================================
    if model in ['SVM', 'SVM-linear']:
        clf = models.SVM(kernel='linear').fit(data['X'], data['y'])
        clf_weighted = models.SVM(class_weight='balanced', kernel='linear').fit(data['X'], data['y'])
        clf_SMOTE = models.SVM(kernel='linear').fit(SMOTE_data['X'], SMOTE_data['y'])
    elif model == 'SVM-rbf':
        # defining parameter range
        param_grid = {'C': [0.1, 1, 10, 100, 500, 2000, 10000],
                      'gamma': ['scale', 'auto', 1, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001],
                      'kernel': ['rbf']}

        print('Tuning SVM params with 5 fold CV')
        # original
        grid_original = GridSearchCV(models.SVM(), param_grid, refit=True, n_jobs=-1)
        grid_original.fit(data['X'], data['y'])
        clf = grid_original.best_estimator_
        print(f'Best SVM params: {grid_original.best_params_}')
        # weighted
        grid_weighted = GridSearchCV(models.SVM(
            class_weight='balanced'), param_grid, refit=True, n_jobs=-1)
        grid_weighted.fit(data['X'], data['y'])
        clf_weighted = grid_weighted.best_estimator_
        # SMOTE
        grid_SMOTE = GridSearchCV(models.SVM(), param_grid, refit=True, n_jobs=-1)
        grid_SMOTE.fit(SMOTE_data['X'], SMOTE_data['y'])
        clf_SMOTE = grid_SMOTE.best_estimator_
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
    elif model == 'MNIST':
        clf = LargeMarginClassifier(binary=binary).fit(
            data['X'], data['y'], epochs=epochs)
        weighted = False
        clf_SMOTE = LargeMarginClassifier(binary=binary).fit(
            SMOTE_data['X'], SMOTE_data['y'], epochs=epochs)
    else:
        raise ValueError(f"model: {model} not in list of available models")
    
    # Model adjsutment methods from the literature ===========================
    clfs = {'Baseline': clf, 'SMOTE': clf_SMOTE}
    if weighted == True:
        clfs['Balanced Weights'] = clf_weighted
    if costcla_methods == True:
        # Bayes Minimum Risk 
        X = data['X']
        y = data['y']
        clf_bmr = BMR(clf).fit(X, y)
        clf_bmr_non_cal = BMR(clf, calibration=False).fit(X, y)

        # Thresholding
        clf_tresh = Thresholding(clf, calibration=False).fit(data['X'], data['y'])

        clfs['BMR'] = clf_bmr
        # clfs['BMR (uncalibrated)'] = clf_bmr_non_cal
        clfs['Threshold'] = clf_tresh
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
   


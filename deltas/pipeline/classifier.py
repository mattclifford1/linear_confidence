import numpy as np
from sklearn.model_selection import GridSearchCV
import os
import matplotlib.pyplot as plt

# from costcla.models.directcost import BayesMinimumRiskClassifier
from deltas.costcla_local.models import BMR, Thresholding
from deltas.classifiers.mnist_train import MNIST_torch, LargeMarginClassifier
import deltas.plotting.plots as plots
import deltas.classifiers.models as models
import deltas.pipeline.data as pipe_data


def get_classifier(data_clf, model='Linear', balance_clf=False, costcla_methods=True, binary=True, epochs=2, _plot=True, _print=False, _plot_data=False, save_file=None, diagram=False):
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
        if _print == True:
            print('Tuning SVM params with 5 fold CV')
        # original
        grid_original = GridSearchCV(models.SVM(), param_grid, refit=True, n_jobs=-1)
        grid_original.fit(data['X'], data['y'])
        clf = grid_original.best_estimator_
        if _print == True:
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
        model = MNIST_torch
        # model = LargeMarginClassifier
        clf = model(hots=1, lr=0.01, cuda=True).fit(
            data['X'], data['y'], epochs=epochs)
        weighted = False
        clf_SMOTE = model(hots=2).fit(
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

    if save_file != None:
        # data
        fig, axs = plt.subplots(figsize=(8,8))
        # fig, axs = plt.subplots()
        plots.plot_classes(data, ax=axs, dim_reducer=dim_reducer)
        if diagram == True:
            axs.plot([-2, 6], [6, -2], c='k', linewidth=5)
            axs.plot([-6, 4], [4, -6], c='k', linestyle='dashed', linewidth=5)
            # axs.annotate("", xy=(2, 2), dx=-2, xytext=(0, 0),
            #             arrowprops=dict(arrowstyle="->"))
            axs.arrow(1, 1, -1.2, -1.2, color='k', width=0.2)
            axs.text(-0.8, -0.8, r'$< \phi(x), w > + b = 0$',
                    fontsize=18, rotation=-45)
            axs.text(-3.9, -3.9,
                    r"$< \phi(x), w > + b^{'} = 0$", fontsize=18, rotation=-45)
            fig.savefig(save_file+'_data_original.png')
        else:
            axs.text(-8, -2, r'$S_1$',
                     fontsize=30)
            axs.text(2, 6, r'$S_2$',
                     fontsize=30)
            fig.savefig(save_file+'_data_original_S.png')
        # clfs
        fig, axs = plt.subplots(1, 2, figsize=(
            16, 8), sharey=True)  # width_ratios=[1, 1.5]
        plots.plot_classes(data, ax=axs[0], dim_reducer=dim_reducer)
        plots.plot_decision_boundary(
            clf, data, ax=axs[0], probs=False, dim_reducer=dim_reducer, colourbar=True)
        axs[0].set_title('Baseline', fontsize=28)

        plots.plot_classes(SMOTE_data, ax=axs[1], dim_reducer=dim_reducer)
        plots.plot_decision_boundary(
            clf_SMOTE, SMOTE_data, ax=axs[1], probs=False, dim_reducer=dim_reducer)
        axs[1].set_title('SMOTE', fontsize=28)

        fig.tight_layout()
        fig.savefig(save_file+'_data.png')
        # plt.show()
    return clfs
   


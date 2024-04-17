import deltas.plotting.plots as plots
import deltas.classifiers.models as models
import deltas.pipeline.data as pipe_data


def get_classifier(data_clf, model='Linear', balance_clf=False, _plot=True, _plot_data=False):
    data = data_clf['data']
    # dim reducer (PCA) for plotting in higher dims
    if data['X'].shape[1] > 2:
        if 'dim_reducer' in data_clf.keys():
            dim_reducer = data_clf['dim_reducer']
        else:
            dim_reducer = pipe_data.get_dim_reducer(data)
    else:
        dim_reducer = None
    # SMOTE
    SMOTE_data = pipe_data.get_SMOTE_data(data)
    # Weighted Classifier
    if balance_clf == True:
        weights = 'balanced'
    else:
        weights = None
    # Train Model
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

    # PLOT
    if _plot_data == True:
        for name, classif, t_data in zip(['clf', 'SMOTE'], [clf, clf_SMOTE], [data, SMOTE_data]):
            print(name)
            ax, _ = plots._get_axes(None)
            plots.plot_classes(t_data, ax=ax, dim_reducer=dim_reducer)
            ax.set_title(name)
            plots.plt.show()
    if _plot == True:
        for name, classif, t_data in zip(['clf', 'SMOTE'], [clf, clf_SMOTE], [data, SMOTE_data]):
            print(name)
            ax, _ = plots._get_axes(None)
            plots.plot_classes(t_data, ax=ax, dim_reducer=dim_reducer)
            plots.plot_decision_boundary(
                classif, t_data, ax=ax, probs=False, dim_reducer=dim_reducer)
            ax.set_title(name)
            plots.plt.show()

    return clf, clf_SMOTE

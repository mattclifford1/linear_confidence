from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, auc, roc_auc_score, precision_recall_fscore_support
from imblearn.metrics import geometric_mean_score
import matplotlib.pyplot as plt
import umap

import deltas.plotting.plots as plots
import deltas.utils.projection as projection


# make precision for each class
def precision0(*args, **kwargs):
    return precision_score(*args, **kwargs, pos_label=0)


def precision1(*args, **kwargs):
    return precision_score(*args, **kwargs, pos_label=1)

def fscore(*args, **kwargs):
    prec, recal, fscor, sup = precision_recall_fscore_support(*args, **kwargs)
    return fscor

def precision(*args, **kwargs):
    prec, recal, fscor, sup = precision_recall_fscore_support(*args, **kwargs)
    return prec[0]

def recall(*args, **kwargs):
    prec, recal, fscor, sup = precision_recall_fscore_support(*args, **kwargs)
    return recal[0]


def minority_accuracy(*args, **kwargs):
    min_acc = confusion_matrix(*args, **kwargs, normalize="true").diagonal()[1]
    return min_acc


def eval_test(clfs, test_data, _print=True, _plot=True, dim_reducer=None, save_file=None, bayes_optimal=False):
    # using new class for deltas format

    # predict on both classifiers (original and delta adjusted)
    preds = {}
    probs = {}
    for name, clf in clfs.items():
        preds[name] = clf.predict(test_data['X'])
        probs[name] = clf.predict_proba(test_data['X'])[:, 1]


    metrics = {
        'Accuracy': accuracy_score,
        # 'Minority Accuracy': minority_accuracy,
        'G-Mean': geometric_mean_score,
        # 'ROC-AUC': roc_auc_score,
        # 'Precision1 (red)': precision0,
        # 'Precision2 (blue)' : precision1,
        # 'Precision': precision,
        # 'Recall': recall,
        'F1': f1_score,
        # 'F-score-1': fscore,
                }
    
    index_name = 'Method'
    d = {index_name: list(preds.keys())}
    for metric, func in metrics.items():
        _scores = []

        # make sure to use probabilities for ROC!!!
        scoring = preds
        if metric == 'ROC-AUC':
            scoring = probs

        # get metric for each classifier's scores
        for name, y_preds in scoring.items():
            _scores.append(func(test_data['y'], y_preds))
        d[metric] = _scores

    scores_df = pd.DataFrame(d).set_index(index_name)

    if _print == True:
        print(scores_df, '\n\n')

    if _plot == True:
        # plot in original space
        for name, clf in clfs.items():
            print(name)
            ax, _ = plots._get_axes(None)
            # data = {'X': test_data['X'], 'y': preds[name]}
            data = test_data
            # data = test_data
            plots.plot_classes(data, ax=ax, dim_reducer=dim_reducer)
            plots.plot_decision_boundary(
                clf, test_data, ax=ax, probs=False, dim_reducer=dim_reducer)
            ax.set_title(name)
            plots.plt.show()

        # plot in projected space
        def _plot_projection_test_and_grid(X, clf, clf_projecter, y_plot, name, grid=False, ax=None):
            proj_data = {'X': clf_projecter.get_projection(X),
                         'y': clf.predict(X)}

            xp1, xp2 = projection.get_classes(proj_data)
            y_plot -= 0.1

            if grid == True:
                m = 'x'
            else:
                m = 'o'
            ax.scatter(xp1, np.ones_like(xp1)*y_plot, c='b', s=10,
                       marker=m)
            ax.scatter(xp2, np.ones_like(xp2)*y_plot, c='r', s=10,
                       marker=m)
            return y_plot

        def _plot_projection_test_data(test_data, clf_projecter, y_plot, ax=None):
            # if not hasattr(clf_projecter, 'get_projection'):
            #     return y_plot
            
            proj_data = {'X': clf_projecter.get_projection(test_data['X']),
                         'y': test_data['y']}

            xp1, xp2 = projection.get_classes(proj_data)
            if xp1.shape[0] > xp2.shape[1]:
                colours = ['r', 'b']
                plot_order = [xp2, xp1]
            else:
                colours = ['b', 'r']
                plot_order = [xp1, xp2]

            y_plot -= 0.1

            ax.scatter(plot_order[0], np.ones_like(plot_order[0])*y_plot, c=colours[0], s=25,
                       marker='o')
            ax.scatter(plot_order[1], np.ones_like(plot_order[1])*y_plot, c=colours[1], s=10,
                       marker='x')
            return y_plot

        _, ax = plt.subplots(1, 1)
        y_plot = 0
        for name, clf in clfs.items():
            # plot test data
            y_plot = _plot_projection_test_data(test_data, clf, y_plot, ax)
            if hasattr(clf, 'get_bias'):
                bias = clf.get_bias()
                if bias != None:
                    ax.scatter([-bias], [y_plot], marker='|', c='k', s=200,
                            label=f'{name} Boundary')
                else:
                    continue
            elif test_data['X'].shape[1] < 32:  # too big for meshgrid otherwise
                # plot linspace/grid
                y_plot = _plot_projection_test_and_grid(
                    test_data['X'], clf, clfs['original'], y_plot, name, True, ax)
            else:
                print(f'Classifier {name} has no get_bias method and the feature space it too big to show boundary via meshgrid')

            y_plot -= 0.2

        ax.legend()
        ax.plot([0], [-1.5], c='w')
        ax.set_title(
            'Boundaries on test dataset in projected space')
        plots.plt.show()
    
    if save_file != None:
        fig, axs = plt.subplots(3, 2, figsize=(16, 7*3), 
                                sharey=True, 
                                # width_ratios=[4, 0.2, 4, 0.2], 
                                # width_ratios=[4, 4], 
                                # height_ratios=[3.5, 3.5, 3.5]
                                )
        x_count = 0
        y_count = 0
        for name, clf in clfs.items():
            # print(name)
            # data = {'X': test_data['X'], 'y': preds[name]}
            data = test_data
            # data = test_data
            plots.plot_classes(
                data, ax=axs[y_count, x_count], dim_reducer=dim_reducer, bayes_optimal=bayes_optimal)
            c = plots.plot_decision_boundary(
                clf, test_data, ax=axs[y_count, x_count], probs=False, dim_reducer=dim_reducer, colourbar=True)
            axs[y_count, x_count].set_title(name, fontsize=28)
            # x_count += 1
            # # cbar_label = 'Predicted Class'
            # # ticks = [0, 1]
            # # cbar = plt.colorbar(c, ticks=ticks)
            # # cbar.ax.tick_params(labelsize=24)
            # # cbar.ax.set_ylabel(cbar_label, size=24)
            # plt.colorbar(c, cax=axs[y_count, x_count])


            x_count += 1
            if x_count == 2:
                y_count += 1
                x_count = 0
            if y_count == 3: 
                break
        plt.tight_layout()
        plt.savefig(save_file+'_eval.png', dpi=300, bbox_inches='tight')


    # if _print == True:
    #     df = scores_df.style.format(precision=4)
    #     print('LATEX table format\n\n')
    #     print(df.to_latex())
    return scores_df

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score
import matplotlib.pyplot as plt
import umap

import deltas.plotting.plots as plots
import deltas.utils.projection as projection


# make precision for each class
def precision0(*args, **kwargs):
    return precision_score(*args, **kwargs, pos_label=0)


def precision1(*args, **kwargs):
    return precision_score(*args, **kwargs, pos_label=1)


def eval_test(clfs, test_data, _print=True, _plot=True, dim_reducer=None):
    # using new class for deltas format

    # predict on both classifiers (original and delta adjusted)
    preds = {}
    for name, clf in clfs.items():
        preds[name] = clf.predict(test_data['X'])

    if _print == True:
        metrics = {'accuracy': accuracy_score,
                   'F1': f1_score,
                   'precision0': precision0,
                   'precision1': precision1,
                   }
        for metric, func in metrics.items():
            for name, y_preds in preds.items():
                print(f"{name} {metric}: {func(test_data['y'], y_preds)}")
            print('')

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
            proj_data = {'X': clf_projecter.get_projection(test_data['X']),
                         'y': test_data['y']}

            xp1, xp2 = projection.get_classes(proj_data)
            y_plot -= 0.1

            ax.scatter(xp1, np.ones_like(xp1)*y_plot, c='b', s=10,
                       marker='o')
            ax.scatter(xp2, np.ones_like(xp2)*y_plot, c='r', s=10,
                       marker='o')
            return y_plot

        _, ax = plt.subplots(1, 1)
        y_plot = 0
        for name, clf in clfs.items():
            # plot test data
            y_plot = _plot_projection_test_data(test_data, clf, y_plot, ax)
            if hasattr(clf, 'get_bias'):
                bias = clf.get_bias()
                if bias != None:
                    ax.scatter([bias], [y_plot], marker='|', c='k', s=200,
                            label=f'{name} Boundary')
                else:
                    continue
            elif test_data['X'].shape[1] < 32:  # too big for meshgrid otherwise
                # plot linspace/grid
                X, _ = plots.get_grid_pred( clf, test_data, probs=False, flat=True, res=25)
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

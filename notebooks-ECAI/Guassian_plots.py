import numpy as np
import deltas.plotting.plots as plots
from deltas.model import base, downsample
from deltas.pipeline import data, classifier, evaluation
import os
# save_file = os.path.join(os.path.expanduser(
#     '~'), 'Downloads', 'Gaussian')
save_file = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'plots', 'Gaussian')
N1 = 1000
N2 = 10
m = 1
# Gaussian (not always seperable)
data_clf = data.get_data(
    m1=[-m, -m],
    m2=[m, m],
    cov1=[[1, 0], [0, 1]],
    cov2=[[1, 0], [0, 1]],
    N1=N1,
    N2=N2,
    scale=False,
    seed=0
    # seed=55
    # seed=20
)
model = 'SVM-linear'
model = 'SVM-rbf'
model = 'Linear'
# model = 'MLP'
# model = 'MLP-Gaussian'

classifiers_dict = classifier.get_classifier(
    data_clf=data_clf,
    model=model,
    _plot=False,
    save_file=save_file,
    bayes_optimal=True)
data_clf['clf'] = classifiers_dict['Baseline']

X = data_clf['data']['X']
y = data_clf['data']['y']
clf = data_clf['clf']
# deltas_model = downsample.downsample_deltas(
#     clf).fit(X, y, _print=True, _plot=True, max_trials=10000)
# deltas_model = base.base_deltas(
#     clf).fit(X, y, grid_search=True, _print=True, _plot=True)
deltas_model = downsample.downsample_deltas(
    clf).fit(X, y, 
             alpha=1,
             method='supports-prop-update_mean',
             _print=True,
             _plot=True,
             save_file=save_file)



classifiers_dict['Our Method'] = deltas_model
evaluation.eval_test(classifiers_dict,
                    #  data_clf['data'], 
                     data_clf['data_test'], 
                     _print=True, 
                     _plot=False, 
                     save_file=save_file,
                     bayes_optimal=True)

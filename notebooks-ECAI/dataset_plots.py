import numpy as np
import deltas.plotting.plots as plots
from deltas.model import base, downsample
from deltas.pipeline import data, classifier, evaluation
import os
save_file = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'plots', 'dataset')
N1 = 1000
N2 = 10
m = 1
costs = (1, 1)  # change for (1, 10) to increase results
# Gaussian (not always seperable)
datasets = {0: 'Breast Cancer', 2: 'Iris', 3: 'Wine', 4: 'Pima Indian Diabetes',
            5: 'Sonar Rocks vs Mines', 6: 'Banknote Authentication',
            7: 'Abalone Gender', 8: 'Ionosphere', 9: 'Wheat Seeds',
            10: 'Credit Scoring 1', 11: 'Credit Scoring 2',
            12: 'Direct Marketing', 13: 'Habermans breast cancer',
            14: 'Wisconsin Breast Cancer', 15: 'Hepatitis',
            16: 'Heart Disease'}

dataset = datasets[4]  # change ind to select dataset to use
data_clf = data.get_real_dataset(dataset, _print=False, seed=0, scale=True)


model = 'SVM-linear'
model = 'SVM-rbf'
# model = 'Linear'
# model = 'MLP'
# model = 'MLP-Gaussian'

classifiers_dict = classifier.get_classifier(
    data_clf=data_clf,
    model=model,
    _plot=False,
    save_file=save_file,
    )
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
             grid_search=False, 
             alpha=1,
             method='supports-prop-update_mean',
             _print=True,
             _plot=True,
             save_file=save_file)



classifiers_dict['Our Method'] = deltas_model
evaluation.eval_test(classifiers_dict,
                     data_clf['data_test'], 
                     _print=True, 
                     _plot=False, 
                     save_file=save_file)

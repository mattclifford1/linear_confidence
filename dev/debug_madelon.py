from deltas.pipeline import data, classifier, evaluation
from deltas.model import downsample
import numpy as np
# np.seterr(all='warn')

N1 = 10
N2 = 1000
data_clf = data.get_non_sep_data_high_dim(
    N1=N1,
    N2=N2,
    scale=False)

model = 'SVM-linear'
model = 'SVM-rbf'
# model = 'Linear'
# model = 'MLP'

balance_clf = True
balance_clf = False

classifiers_dict = classifier.get_classifier(
    data_clf=data_clf,
    model=model,
    balance_clf=balance_clf,
    _plot=False)
data_clf['clf'] = classifiers_dict['original']

X = data_clf['data']['X']
y = data_clf['data']['y']
clf = data_clf['clf']
# deltas_model = model_deltas.reprojection_deltas(
deltas_model = downsample.downsample_deltas(clf
                                            ).fit(X, y, alpha=1000, _print=True, _plot=False, max_trials=100000, parallel=False)

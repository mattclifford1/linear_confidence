from deltas.model import downsample
import deltas.pipeline.run as run
from deltas.model import SSL
N1 = 1000
N2 = 10
data_clf = run.get_non_sep_data(
    N1=N1,
    N2=N2,
    scale=True)
model = 'SVM-linear'
model = 'SVM-rbf'
# model = 'Linear'
# model = 'MLP'

balance_clf = True
balance_clf = False

data_clf['clf'], clf_SMOTE = run.get_classifier(
    data_clf=data_clf,
    model=model,
    balance_clf=balance_clf,
    _plot=False
    )
X = data_clf['data']['X']
y = data_clf['data']['y']
clf = data_clf['clf']
# deltas_model = as.reprojection_deltas(
deltas_model = SSL.SSL_deltas(
    # deltas_model = downsample.downsample_deltas(
    clf,
).fit(X, y, alpha=1000, _print=True, _plot=False, max_trials=100000, parallel=False)

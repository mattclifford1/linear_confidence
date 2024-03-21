from deltas.pipeline import data, classifier, evaluation
from deltas.model import SSL
N1 = 1000
N2 = 10
data_clf = data.get_non_sep_data(
    N1=N1,
    N2=N2,
    scale=True)
model = 'SVM-rbf'

balance_clf = True
balance_clf = False

data_clf['clf'], clf_SMOTE = classifier.get_classifier(
    data_clf=data_clf,
    model=model,
    balance_clf=balance_clf,
    _plot=False
    )
X = data_clf['data']['X']
y = data_clf['data']['y']
clf = data_clf['clf']
deltas_model = SSL.SSL_deltas(
    clf,
).fit(X, y, alpha=1000, _print=True, _plot=False, max_trials=100000, parallel=False)

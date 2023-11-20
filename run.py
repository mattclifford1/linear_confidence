from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import data_utils
import plot
import truncated_normal
import train

if __name__ == '__main__':
    data = truncated_normal.get_two_classes(num_samples=[100, 100])
    scaler = data_utils.normaliser(data)
    data = scaler(data)
    
    # model
    # clf = LogisticRegression(random_state=0).fit(data['X'], data['y'])
    clf = SVC(random_state=0, probability=True, kernel='linear').fit(data['X'], data['y'])
    # clf = train.get_model(data)


    ax, _ = plot._get_axes(None)
    plot.plot_classes(data, ax=ax)
    plot.plot_decision_boundary(clf, data, ax=ax)
    plot.plt.show()


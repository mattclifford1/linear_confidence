import numpy as np
from scipy.spatial.distance import cdist

def from_clf(data, clf, supports=False):
    projected = np.dot(data['X'], clf.coef_.T)/np.linalg.norm(clf.coef_.T)
    # get supports
    xp1 = np.array([p for i, p in enumerate(projected) if data['y'][i] == 0])
    xp2 = np.array([p for i, p in enumerate(projected) if data['y'][i] == 1])
    projected_data = {'X': projected, 'y': data['y']}
    if supports == True:
        Y = cdist(xp1, xp2, 'euclidean')
        projected_data['supports'] = np.array(
            [xp1[np.argmin(Y, axis=0)[0]],
             xp2[np.argmin(Y, axis=1)[0]]
            ])
    xp1 = np.array([p for i, p in enumerate(projected_data['X']) if projected_data['y'][i] == 0])
    xp2 = np.array([p for i, p in enumerate(projected_data['X']) if projected_data['y'][i] == 1])
    projected_data['X1'] = xp1
    projected_data['X2'] = xp2
    return projected_data


# get supports
def closest_node(node, nodes):
    return nodes[cdist([node], nodes).argmin()]

def get_classes(data):
    # data points
    xp1 = np.array([p for i, p in enumerate(data['X']) if data['y'][i] == 0])
    xp2 = np.array([p for i, p in enumerate(data['X']) if data['y'][i] == 1])
    return xp1, xp2

def get_emp_means(data):
    xp1, xp2 = get_classes(data)
    emp_xp1 = np.mean(xp1)
    emp_xp2 = np.mean(xp2)
    return emp_xp1, emp_xp2

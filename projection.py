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

    return projected_data


# get supports
def closest_node(node, nodes):
    return nodes[cdist([node], nodes).argmin()]

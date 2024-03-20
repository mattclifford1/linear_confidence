import numpy as np
from scipy.spatial.distance import cdist

def from_clf(data, clf, supports=False):
    if hasattr(clf, 'get_projection'):
        projected = clf.get_projection(data['X'])
    else:
        raise AttributeError(f"Classifier {clf} needs 'get_projection' method")
    data = {'X': projected, 'y': data['y']}

    return make_calcs(data, supports=supports)


def make_calcs(projected_data, supports=False):
    # get supports
    xp1 = projected_data['X'][projected_data['y'] == 0, :]
    xp2 = projected_data['X'][projected_data['y'] == 1, :]

    if supports == True:
        try:
            Y = cdist(xp1, xp2, 'euclidean')
            projected_data['supports'] = np.array(
                [xp1[np.argmin(Y, axis=0)[0]],
                xp2[np.argmin(Y, axis=1)[0]]
                ])
        except ValueError:
            pass
        except IndexError:
            pass
        
    projected_data['X1'] = xp1
    projected_data['X2'] = xp2
    return projected_data


# get supports
def closest_node(node, nodes):
    return nodes[cdist([node], nodes).argmin()]

def get_classes(data):
    # data points
    if len(data['X'].shape) == 1:
        xp1 = data['X'][data['y'] == 0]
        xp2 = data['X'][data['y'] == 1]
    else:
        xp1 = data['X'][data['y'] == 0, :]
        xp2 = data['X'][data['y'] == 1, :]
    return xp1, xp2

def get_emp_means(data):
    xp1, xp2 = get_classes(data)
    emp_xp1 = np.mean(xp1)
    emp_xp2 = np.mean(xp2)
    return emp_xp1, emp_xp2

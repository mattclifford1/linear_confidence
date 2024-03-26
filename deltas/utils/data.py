'''
misc utils for data processing
'''
import numpy as np
from scipy.spatial.distance import cdist


def split_classes(X, y):
    X1 = X[y == 0, :]
    X2 = X[y == 1, :]
    y1 = y[y == 0]
    y2 = y[y == 1]
    return X1, X2, y1, y2

def get_support_of_class_ind(X, mean=None):
    '''
    return index of the point furthest from the mean of the data
    '''
    # calculate the mean of all data points, if not provided
    if isinstance(mean, type(None)):
        mean = np.mean(X, axis=0)
    
    if len(mean.shape) == 1:
        mean = np.expand_dims(mean, axis=1)

    # calculate the supports
    dists = cdist(X, mean, 'euclidean')
    
    # furthest away from the mean
    return np.argmax(dists)


def get_support_of_class_ind_margin_only(X, X_other, mean_other=None):
    '''
    return index of the point furthest from the mean of the data but only
    on the side of the margin - i.e. support closest to the mean of the other class
    '''
    # calculate the mean of all data points, if not provided
    if isinstance(mean_other, type(None)):
        mean_other = np.mean(X_other, axis=0)

    if len(mean_other.shape) == 1:
        mean_other = np.expand_dims(mean_other, axis=1)

    # calculate the supports
    dists = cdist(X, mean_other, 'euclidean')

    # closest to other class' mean
    return np.argmin(dists)

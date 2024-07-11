'''
Generate toy data from the breast cancer dataset
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
import deltas


def get_breast_cancer(seed=True, **kwargs):
    '''
    breast cancer dataset
    returns:
        - data: dict containing 'X', 'y'
    '''
    # size = 0.453 # for even test 
    # get dataset
    data = load_breast_cancer()
    data = {'X': data.data, 'y': data.target}
    # swap labels for minority convention
    data['y'][data['y'] == 1] = 2
    data['y'][data['y'] == 0] = 1
    data['y'][data['y'] == 2] = 0
    # shuffle the dataset
    data = deltas.data.utils.shuffle_data(data, seed=seed)
    # reduce the size of the dataset
    # data = deltas.data.utils.proportional_downsample(data, **kwargs)
    # split into train, test
    train_data, test_data = deltas.data.utils.proportional_split(
        data, 
        # size=0.701, # for equal eval with 5 ratio
        # size=0.453, # for equal eval with 10 ratio
        size=0.5,
        ratio=10,
        seed=seed)
    return train_data, test_data

def get_wine(**kwargs):
    '''
    wine dataset (0 vs 1,2)
    returns:
        - data: dict containing 'X', 'y'
    '''
    # get dataset
    data = load_wine()
    # convert to binary datatset (0 vs 1,2)
    y = data.target
    y[np.where(y>1)] = 1
    data = {'X': data.data, 'y': y}
    # shuffle the dataset
    data = deltas.data.utils.shuffle_data(data)
    # reduce the size of the dataset
    # data = deltas.data.utils.proportional_downsample(data, **kwargs)
    # split into train, test
    train_data, test_data = deltas.data.utils.proportional_split(data, size=0.8)
    return train_data, test_data

def get_iris(**kwargs):
    '''
    iris dataset (0,2 vs 1)
    returns:
        - data: dict containing 'X', 'y'
    '''
    # get dataset
    data = load_iris()
    # convert to binary datatset (0 vs 1,2)
    y = data.target
    y[np.where(y == 2)] = 0
    data = {'X': data.data, 'y': y}
    # shuffle the dataset
    data = deltas.data.utils.shuffle_data(data)
    # add the feature names
    data['feature_names'] = ['Sepal length',
                             'Sepal width',
                             'Petal length',
                             'Petal width']
    # reduce the size of the dataset
    # data = deltas.data.utils.proportional_downsample(data, **kwargs)
    # split into train, test
    train_data, test_data = deltas.data.utils.proportional_split(data, size=0.8)
    return train_data, test_data


if __name__ == '__main__':
    tr, te = get_breast_cancer()
    print(np.unique(tr['y'], return_counts=True))
    print(np.unique(te['y'], return_counts=True))

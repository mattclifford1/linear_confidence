'''
Generate data from https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html#:~:text=Generate%20a%20random%20n%2Dclass,of%20clusters%20to%20each%20class.
I. Guyon, “Design of experiments for the NIPS 2003 variable selection benchmark”, 2003.
'''
from sklearn.datasets import make_classification
from sklearn.utils import shuffle
import numpy as np

import deltas.data.utils as utils


def get_separable(N1=10000,
                  N2=10000,
                  scale=True,
                  test_nums=[10000, 10000]):
    
    return _get_data(N1=N1, N2=N2, scale=scale, test_nums=test_nums, gen_num=5)

def get_non_separable(N1=10000,
                  N2=10000,
                  scale=True,
                  test_nums=[10000, 10000],
                  dims=2,
                  gen_num=9):
    
    return _get_data(N1=N1, N2=N2, scale=scale, test_nums=test_nums, gen_num=gen_num, dims=dims)

def _get_data(N1=10000,
              N2=10000,
              scale=True,
              test_nums=[10000, 10000],
              gen_num=0,
              dims=2):
    class1_num = N1 + test_nums[0]
    class2_num = N2 + test_nums[1]

    # get samples nums and proportions
    n_samples = class1_num + class2_num + 10 # add samples as sampling isn't always accurate using weights
    weights = [class1_num/n_samples, class2_num/n_samples]
    # sample data
    # 5 = good seperable dataset
    # 9 =  non seperable
    X, y = make_classification(n_samples=n_samples, n_features=dims, n_redundant=0, shuffle=False,
                               n_clusters_per_class=1, weights=weights, flip_y=0, random_state=gen_num)

    # split into classes for manipulation
    class1 = X[y == 0, :]
    class2 = X[y == 1, :]

    # TRAINING DATA
    X1_train = class1[:N1, :]
    X2_train = class2[:N2, :]
    y1_train = np.zeros(N1)
    y2_train = np.ones(N2)
    X_train = np.concatenate([X1_train, X2_train], axis=0)
    y_train = np.concatenate([y1_train, y2_train], axis=0)

    # TESTING DATA
    X1_test = class1[N1:N1+test_nums[0], :]
    X2_test = class2[N2:N2+test_nums[1], :]
    y1_test = np.zeros(test_nums[0])
    y2_test = np.ones(test_nums[1])
    X_test = np.concatenate([X1_test, X2_test], axis=0)
    y_test = np.concatenate([y1_test, y2_test], axis=0)

    # shuffle data up
    X_train, y_train = shuffle(X_train, y_train, random_state=1)
    X_test, y_test = shuffle(X_test, y_test, random_state=1)

    # put in dictionary
    data = {'X': np.array(X_train), 'y': np.array(y_train)}
    data_test = {'X': np.array(X_test), 'y': np.array(y_test)}

    # scale data
    scaler = utils.normaliser(data)
    if scale == True:
        data = scaler(data)
        data_test = scaler(data_test)
    return data, data_test
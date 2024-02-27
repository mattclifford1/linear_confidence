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
                  test_nums=[10000, 10000]):
    
    return _get_data(N1=N1, N2=N2, scale=scale, test_nums=test_nums, gen_num=9)

def _get_data(N1=10000,
              N2=10000,
              scale=True,
              test_nums=[10000, 10000],
              gen_num=0):
    class1_num = N1 + test_nums[0]
    class2_num = N2 + test_nums[1]

    # get samples nums and proportions
    n_samples = class1_num + class2_num
    weights = [class1_num/n_samples, class2_num/n_samples]
    # sample data
    # 5 = good seperable dataset
    # 9 =  non seperable
    X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, shuffle=False,
                               n_clusters_per_class=1, weights=weights, flip_y=0, random_state=gen_num)
    # split into train and test
    class1 = [x for i, x in enumerate(X) if y[i] == 0]
    class2 = [x for i, x in enumerate(X) if y[i] == 1]

    X_train = []
    y_train = []
    for i in range(N1):
        X_train.append(class1[i])
        y_train.append(0)
    for j in range(N2):
        X_train.append(class2[j])
        y_train.append(1)
    X_test = []
    y_test = []
    for i in range(test_nums[0]):
        X_test.append(class1[i+N1])
        y_test.append(0)
    for j in range(test_nums[1]):
        X_test.append(class2[j+N2])
        y_test.append(1)

    X_train, y_train = shuffle(X_train, y_train, random_state=1)
    X_test, y_test = shuffle(X_test, y_test, random_state=1)

    data = {'X': np.array(X_train), 'y': np.array(y_train)}
    data_test = {'X': np.array(X_test), 'y': np.array(y_test)}

    scaler = utils.normaliser(data)
    if scale == True:
        data = scaler(data)
        data_test = scaler(data_test)
    return data, data_test
import numpy as np
import sklearn.utils


def get_two_classes(means=[[0, 0], [10, 10]], 
                    covs=[[[1, 0], [0, 1]],
                         [[1, 1], [1, 1]]], 
                    num_samples=[3, 2]):
    labels = [0, 1]
    X = []
    y = []
    for mean, cov, num_sample, label in zip(means, covs, num_samples, labels):
        X.append(np.random.multivariate_normal(mean, cov, size=num_sample))
        y.append(np.ones(num_sample)*label)
    X = np.vstack(X)
    y = np.hstack(y)
    X, y = sklearn.utils.shuffle(X, y)  # , random_state=seed)
    return {'X': X, 'y': y}


if __name__ == '__main__':
    data = get_two_classes(num_samples=[50, 100])
    print(data)
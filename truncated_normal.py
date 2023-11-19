import numpy as np
import scipy
import sklearn.utils


def truncated_normal(mean, std, bounds, num_samples): # https://stackoverflow.com/questions/64601585/how-to-best-get-a-sample-from-a-truncated-normal-distribution
    # set up number of samples
    means = np.empty(num_samples)
    stds = np.empty(num_samples)
    boundss = np.empty([num_samples, 2])
    # apply deets
    means[:] = mean
    stds[:] = std
    boundss[:, 0] = bounds[0]
    boundss[:, 1] = bounds[1]
    # sample
    samples = scipy.stats.truncnorm.rvs((boundss[:, 0] - means) / stds, (boundss[:, 1] - means) / stds, loc=means, scale=stds)
    return np.expand_dims(samples, axis=1)

def multivar_truncated(means=[0, 0], stds=[1, 1], bounds=[-10, 10], num_samples=100):
    # only does non correlated atm
    X = []
    for mean, std in zip(means, stds):
        X.append(truncated_normal(mean, std, bounds, num_samples))
    return np.hstack(X)

def get_two_classes(means=[[0,0], [10,10]], stds=[[1,1], [1,1]], bounds=[[-2,2], [8,12]], num_samples=[3, 2]):
    labels = [0, 1]
    X = []
    y = []
    for mean, std, bound, num_sample, label in zip(means, stds, bounds, num_samples, labels):
        X.append(multivar_truncated(means=mean, stds=std, bounds=bound, num_samples=num_sample))
        y.append(np.ones(num_sample)*label)
    X = np.vstack(X)
    y = np.hstack(y)
    X, y = sklearn.utils.shuffle(X, y)  # , random_state=seed)
    return {'X': X, 'y':y}

def get_two_classes_R(means=[[0,0], [10,10]], stds=[[1,1], [1,1]], Rs=[2, 2], num_samples=[3, 2]):
    bounds = 1 # write this
    get_two_classes(means=[[0,0], [10,10]], stds=[[1,1], [1,1]], bounds=[[-2,2], [8,12]], num_samples=[3, 2])


if __name__ == '__main__':
    data = get_two_classes(num_samples=[50, 100])
    print(data)



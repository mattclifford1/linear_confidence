import numpy as np

import deltas.data.normal as normal


def get_XOR(num_samples=[100, 100]):
    mu = 5
    cov = [[1, 0], [0, 1]]
    covs = [cov, cov]
    top_data = normal.get_two_classes(means=[[-mu, -mu], [mu, -mu]],
                                      covs=covs,
                                      num_samples=[num_samples[0]//2, num_samples[1]//2])
    bot_data = normal.get_two_classes(means=[[mu, mu], [-mu, mu]],
                                      covs=covs,
                                      num_samples=[num_samples[0]//2, num_samples[1]//2])

    X = np.vstack([top_data['X'], bot_data['X']])
    y = np.hstack([top_data['y'], bot_data['y']])

    return {'X': X, 'y': y}

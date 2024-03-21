import functools

from sklearn.decomposition import PCA, KernelPCA
from imblearn.over_sampling import SMOTE
import umap

import deltas.data.utils as utils
import deltas.data.normal as normal
import deltas.data.madelon as madelon
import deltas.data.sklearn_synthetic as synthetic


def make_data_dim_reducer(data_getter):
    @functools.wraps(data_getter)
    def _wrapper(*args, **kwargs):
        data_dict = data_getter(*args, **kwargs)
        if data_dict['data']['X'].shape[1] > 2:
            dim_reducer = get_dim_reducer(data_dict['data'])
        else:
            dim_reducer = None
        data_dict['dim_reducer'] = dim_reducer
        return data_dict
    return _wrapper


def get_data(m1=[1, 1],
             m2=[10, 10],
             cov1=[[1, 0], [0, 1]],
             cov2=[[1, 0], [0, 1]],
             N1=10000,
             N2=10000,
             scale=True,
             test_nums=[10000, 10000]):
    data = normal.get_two_classes(means=[m1, m2],
                                  covs=[cov1, cov2],
                                  num_samples=[N1, N2])
    data_test = normal.get_two_classes(means=[m1, m2],
                                       covs=[cov1, cov2],
                                       num_samples=[test_nums[0], test_nums[1]])

    scaler = utils.normaliser(data)
    if scale == True:
        data = scaler(data)
        data_test = scaler(data_test)
        m1 = scaler.transform_instance(m1)
        m2 = scaler.transform_instance(m2)

    return {'data': data, 'mean1': m1, 'mean2': m2, 'data_test': data_test}


def get_non_sep_data(N1=10000,
                     N2=10000,
                     scale=True,
                     test_nums=[10000, 10000]):
    data, data_test = madelon.get_non_separable(
        N1=N1, N2=N2, scale=scale, test_nums=test_nums)

    return {'data': data, 'data_test': data_test}


@make_data_dim_reducer
def get_non_sep_data_high_dim(N1=10000,
                              N2=10000,
                              scale=True,
                              test_nums=[10000, 10000]):
    data, data_test = madelon.get_non_separable(
        N1=N1, N2=N2, scale=scale, test_nums=test_nums, dims=100, gen_num=3)

    return {'data': data, 'data_test': data_test}


def get_sep_data(N1=10000,
                 N2=10000,
                 scale=True,
                 test_nums=[10000, 10000]):
    data, data_test = madelon.get_separable(
        N1=N1, N2=N2, scale=scale, test_nums=test_nums)

    return {'data': data, 'data_test': data_test}


def get_synthetic_sep_data(N1=10000,
                           N2=10000,
                           scale=True,
                           test_nums=(10000, 10000)):

    data = synthetic.get_moons((N1, N2))
    data_test = synthetic.get_moons(test_nums)
    return {'data': data, 'data_test': data_test}


def get_SMOTE_data(data):
    oversample = SMOTE()
    X, y = oversample.fit_resample(data['X'], data['y'])
    return {'X': X, 'y': y}


def get_dim_reducer(data, reducer='PCA'):
    X = data['X']
    y = data['y']
    if reducer == 'PCA':
        reducer_model = PCA(n_components=2, svd_solver='full').fit(X)
    elif reducer == 'kernelPCA':
        reducer_model = KernelPCA(
            n_components=2, kernel='rbf', n_jobs=-1).fit(X)
    elif reducer == 'UMAP':
        reducer_model = umap.UMAP().fit(X)
    elif reducer == 'UMAP_supervised':
        reducer_model = umap.UMAP().fit(X, y=y)
    else:
        reducer = None
    return reducer_model

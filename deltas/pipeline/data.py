import functools

from sklearn.decomposition import PCA, KernelPCA
from imblearn.over_sampling import SMOTE
import umap

import deltas.data.utils as utils
import deltas.data.XOR as XOR
import deltas.data.normal as normal
import deltas.data.madelon as madelon
import deltas.data.sklearn_synthetic as synthetic
from deltas.data.loaders import (sklearn_toy, 
                                 diabetes, 
                                 Habermans_breast_cancer, 
                                 sonar_rocks, 
                                 banknote, 
                                 abalone_gender, 
                                 ionosphere, 
                                 wheat_seeds, 
                                 costcla,
                                 mnist,
                                 breast_cancer_W,
                                 hepititus,
                                 heart_disease,
                                 MIMIC_III,
                                 MIMIC_IV)


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

    if scale == True:
        scaler = utils.normaliser(data)
        data = scaler(data)
        data_test = scaler(data_test)
        m1 = scaler.transform_instance(m1)
        m2 = scaler.transform_instance(m2)

    return {'data': data, 'mean1': m1, 'mean2': m2, 'data_test': data_test}


def get_XOR(N1, 
            N2,
            scale=True,
            test_nums=[10000, 10000]):
    # sample data
    data = XOR.get_XOR(num_samples=[N1, N2])
    data_test = XOR.get_XOR(num_samples=test_nums)
    
    #scale
    if scale == True:
        scaler = utils.normaliser(data)
        data = scaler(data)
        data_test = scaler(data_test)

    return {'data': data, 'data_test': data_test}


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


@make_data_dim_reducer
def get_MNIST(scale=False):
    mnist.get_mnist()


@make_data_dim_reducer
def get_real_dataset(dataset='Breast Cancer', _print=True, scale=False, **kwargs):
    AVAILABLE_DATASETS = {
        # 'Gaussian': sample_dataset_to_proportions(get_gaussian),
        # 'Moons': sample_dataset_to_proportions(get_moons),
        'Breast Cancer': sklearn_toy.get_breast_cancer,
        'Iris': sklearn_toy.get_iris,
        'Wine': sklearn_toy.get_wine,
        'Pima Indian Diabetes': diabetes.get_diabetes_indian,
        'Habermans breast cancer': Habermans_breast_cancer.get_Habermans_breast_cancer,
        'Sonar Rocks vs Mines': sonar_rocks.get_sonar,
        'Banknote Authentication': banknote.get_banknote,
        'Abalone Gender': abalone_gender.get_abalone,
        'Ionosphere': ionosphere.get_ionosphere,
        'Wheat Seeds': wheat_seeds.get_wheat_seeds,
        'Credit Scoring 1': costcla.costcla_dataset('CreditScoring_Kaggle2011_costcla'),
        'Credit Scoring 2': costcla.costcla_dataset('CreditScoring_PAKDD2009_costcla'),
        'Direct Marketing': costcla.costcla_dataset('DirectMarketing_costcla'),
        'MNIST': mnist.get_mnist,
        'Wisconsin Breast Cancer':  breast_cancer_W.get_Wisconsin_breast_cancer,
        'Hepatitis': hepititus.get_hepatitis,
        'Heart Disease': heart_disease.get_HD,
        'MIMIC-III': MIMIC_III.get_mortality,
        'MIMIC-III-mortality': MIMIC_III.get_mortality,
        'MIMIC-III-sepsis': MIMIC_III.get_sepsis,
        'MIMIC-IV': MIMIC_IV.get_ready_for_discharge,
        # 'Circles': sample_dataset_to_proportions(get_circles),
        # 'Blobs': sample_dataset_to_proportions(get_blobs),
    }

    # check input correct dataset name
    if dataset not in AVAILABLE_DATASETS.keys():
        raise ValueError(f'dataset needs to be one of:{AVAILABLE_DATASETS.keys()}')

    # load dataset
    train_data, test_data = AVAILABLE_DATASETS[dataset](**kwargs)

    # scale
    scaler = utils.normaliser(train_data)
    if scale == True:
        train_data = scaler(train_data)
        test_data = scaler(test_data)

    train0 = len(train_data['y'])-sum(train_data['y'])
    train1 = sum(train_data['y'])

    test0 = len(test_data['y'])-sum(test_data['y'])
    test1 = sum(test_data['y'])
    if _print == True:
        print(f"{dataset}: {test0+train0+test1+train1}")
        print(f"Number of attribues: {train_data['X'].shape[1]}")
        print( f"Classes total: {test0+train0} - {test1+train1}\n")
        print(f"Classes train: {train0} - {train1}")
        print(f"Classes test:  {test0} - {test1}")
    
    # return in dict format needed 
    return {'data': train_data, 'data_test': test_data}


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

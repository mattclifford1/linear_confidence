from sklearn import preprocessing
from sklearn.utils import shuffle
import numpy as np
from deltas.misc.use_two import RANDOM_STATE


class normaliser:
    def __init__(self, train_data):
        self.scaler = preprocessing.MinMaxScaler(
            feature_range=(-1,1)).fit(train_data['X'])

    def __call__(self, data):
        '''expect data as a dict with 'X', 'y' keys'''
        data['X'] = self.scaler.transform(data['X'])
        return data
    
    def transform_instance(self, X):
        return self.scaler.transform([X])[0]


def set_seed(seed):
    if seed == True:
        np.random.seed(seed=RANDOM_STATE)
    elif type(seed) == int:
        np.random.seed(seed=seed)
    elif seed == False:
        np.random.seed(seed=None)


def shuffle_data(data, seed=True):
    if seed == True:
        seed = RANDOM_STATE
    data['X'], data['y'] = shuffle(
        data['X'], data['y'], random_state=seed)
    return data


def shuffle_dataset(data, seed=True):
    '''
    shuffle numpy arrays (data) in data dict
    '''
    instances = data['X'].shape[0]
    # get random order
    set_seed(seed)
    p = np.random.permutation(instances)
    for key in data.keys():
        # apply to all numpy arrays that are data rows
        if type(data[key]) == np.ndarray and data[key].shape[0] == instances:
            # apply the shuffle
            data[key] = data[key][p]
    return data


def proportional_split(data, size=0.8, seed=True, ratio=None, equal_test=False):
    '''
    create a train, test split that preserves the class distributions
        data: data dict holder
        size: size of the train set (0.5 means equal train, test size)
        ratio: make imbalance ratio in the train set - assumes class 1 is minority
    '''
    if size <= 0 or size > 1:
        raise ValueError(
            f'size needs to be between 0 and 1 instead of :{size}')
    set_seed(seed)
    # get current class proportions
    classes, counts = np.unique(data['y'], return_counts=True)
    classes = sorted(classes)
    test_inds = []
    train_inds = []
    for i, cls in enumerate(classes):
        # get all the inds of current class
        cls_inds = np.where(data['y'] == cls)[0]
        # shuffle all the inds to get a random selection
        set_seed(seed)
        np.random.shuffle(cls_inds)
        set_seed(seed)
        # now split the data inds into train/test
        split_point = int(counts[i]*size)
        if cls == 1: # minority class
            if not isinstance(ratio, type(None)) and ratio != False:
                split_point = max(int(len(train_inds[0])/ratio), 1)
        train_inds.append(list(cls_inds[:split_point]))
        test_inds.append(list(cls_inds[split_point:]))
    # concat all the inds from each class
    train_inds = np.concatenate(train_inds)
    test_inds = np.concatenate(test_inds)
    # now apply the split to all data arrays
    test_split = {}
    instances = data['X'].shape[0]
    for key, val in data.items():
        # apply to all numpy arrays that are data rows
        if isinstance(val, np.ndarray) and data[key].shape[0] == instances:
            # extract and store the splits
            test_split[key] = val[test_inds]  # important to do this one first!
            data[key] = val[train_inds]       # as test data is now deleted
    if equal_test == True:
        classes, counts = np.unique(test_split['y'], return_counts=True)
        max_inst = min(counts)
        for cls in classes:
            inds = np.arange(len(test_split['y']))
            inds = inds[test_split['y']==cls]
            inds_drop = inds[max_inst:]
            test_split['y'] = np.delete(test_split['y'], inds_drop)
            test_split['X'] = np.delete(test_split['X'], inds_drop, axis=0)
    
    return data, test_split


def proportional_downsample(data, percent_of_data=1, seed=True, **kwargs):
    '''
    downsample data whilst keep the represenetaed class proportion distribution
    the same
        data: data dict holder
        percent_of_data: % of the dataset to downsample to
        seed: True, False, or random seed number
    '''
    if percent_of_data <= 0 or percent_of_data > 100:
        raise ValueError(
            f'percent_of_data needs to be between 0 and 100 instead of :{percent_of_data}')
    set_seed(seed)
    # get current class proportions
    classes, counts = np.unique(data['y'], return_counts=True)
    # now downsample
    new_data_counts = (counts*(percent_of_data/100)).astype(np.uint64)
    # make sure we have at least a sample for train/test splits
    new_data_counts[new_data_counts < 2] = 2
    new_inds = []
    for i, cls in enumerate(classes):
        # get all the inds of current class
        cls_inds = np.where(data['y'] == cls)[0]
        # shuffle all the inds to get a random selection
        np.random.shuffle(cls_inds)
        # now store a subsample of class inds
        sub_sample_of_inds = cls_inds[:new_data_counts[i]]
        new_inds.append(list(sub_sample_of_inds))
    # concat all the inds from each class
    new_inds = np.concatenate(new_inds)
    # now only take new_inds from all data arrays
    instances = data['X'].shape[0]
    for key, val in data.items():
        # apply to all numpy arrays that are data rows
        if isinstance(val, np.ndarray) and data[key].shape[0] == instances:
            # apply the shuffle
            data[key] = val[new_inds]
    return data

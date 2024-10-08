'''
Get datasets from the costcla package
    - credit scoring and direct marketing
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import os
import pandas as pd
import deltas

CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))

class costcla_dataset:
    def __init__(self, dataset):
        available_datasets = ['CreditScoring_Kaggle2011_costcla',
                              'CreditScoring_PAKDD2009_costcla',
                              'DirectMarketing_costcla']
        if dataset not in available_datasets:
            raise Exception(
                f'dataset: {dataset} not in costcla available datasets: {available_datasets}')
        self.dataset = dataset

    def __call__(self, percent_of_data=5, **kwargs):
        # get dataset
        data = _get_costcla_dataset(self.dataset)
        # shuffle the dataset
        data = deltas.data.utils.shuffle_data(data) # type: ignore
        # reduce the size of the dataset
        data = deltas.data.utils.proportional_downsample(
            data, percent_of_data=percent_of_data, **kwargs)  # type: ignore
        # split into train, test
        train_data, test_data = deltas.data.utils.proportional_split(data, size=0.7, ratio=10) # type: ignore
        return train_data, test_data



def _get_costcla_dataset(dataset="CreditScoring_Kaggle2011_costcla", normalise=False):
    '''
    load the costcla csv dataset files
    available datasets:
        - CreditScoring_Kaggle2011_costcla
        - CreditScoring_PAKDD2009_costcla
        - DirectMarketing_costcla
    '''
    data = {}
    csvs = ['X', 'y', 'cost_matrix']
    # read and store all csv data
    for csv in csvs:
        df = pd.read_csv(os.path.join(CURRENT_FILE, '..', 'datasets', dataset, f'{csv}.csv'))
        # split into train and test
        data[csv] = df.to_numpy()
        if data[csv].shape[1] == 1:
            data[csv] = data[csv].ravel()
        # get feature names
        if csv == 'X':
            data['feature_names'] = df.columns.to_list()

    # normalise X data
    if normalise == True:
        data['X'] = data['X'] / data['X'].max(axis=0)

    # add  description
    with open(os.path.join(CURRENT_FILE, '..', 'datasets', dataset, 'description.txt'), 'r') as f:
        data['description'] = f.read()
    return data


if __name__ == '__main__':
    train_data, test_data = _get_costcla_dataset()
    print(train_data['X'].shape)
    print(train_data['y'].shape)

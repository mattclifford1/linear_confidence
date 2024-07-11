'''
loader for hepititus: https://archive.ics.uci.edu/dataset/46/hepatitis
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import os
import pandas as pd
import numpy as np
import deltas
from ucimlrepo import fetch_ucirepo

CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


def get_HD(seed=True, **kwargs):
    data = {}
    # df = pd.read_csv(os.path.join(CURRENT_FILE, '..',
    #                  'datasets', 'hepititus', 'data.csv'))

    # fetch dataset
    heart_disease = fetch_ucirepo(id=45)

    # data (as pandas dataframes)
    X = heart_disease.data.features
    y = heart_disease.data.targets

    X.pop('ca')
    X.pop('thal')

    data['y'] = y.to_numpy().squeeze(axis=1)
    data['X'] = X.to_numpy()

    for i in [1,2,]:
        inds = np.where(data['y']==i)
        data['y'] = np.delete(data['y'], inds)
        data['X'] = np.delete(data['X'], inds, axis=0)
    data['y'][data['y']==2] = 1
    data['y'][data['y']==3] = 1
    data['y'][data['y']==4] = 1

    data['feature_names'] = heart_disease.metadata
    # add name and description
    data['description'] = heart_disease.metadata
    # shuffle the dataset
    # print(seed)
    data = deltas.data.utils.shuffle_data(data, seed=seed)  # type: ignore
    # split into train, test
    train_data, test_data = deltas.data.utils.proportional_split( # type: ignore
        data, 
        size=0.5, 
        seed=seed)#, ratio=10)  # type: ignore
    return train_data, test_data

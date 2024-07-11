'''
loader for hepititus: https://archive.ics.uci.edu/dataset/46/hepatitis
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import os
import pandas as pd
import numpy as np
import deltas

CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


def get_hepatitis(seed=True, **kwargs):
    data = {}
    df = pd.read_csv(os.path.join(CURRENT_FILE, '..',
                     'datasets', 'hepititus', 'data.csv'))
    df.pop('PROTIME')
    df.pop('ALKPHOSPHATE')
    df.pop('ALBUMIN')
    # df.pop('LIVERBIG')
    # df.pop('LIVERFIRM')
    df = df[~df.isin(['?']).any(axis=1)]
    data['y'] = df.pop('Class').to_numpy()
    data['y'][data['y'] == 2] = 0
    data['X'] = df.to_numpy()
    data['feature_names'] = df.columns.to_list()
    # add name and description
    with open(os.path.join(CURRENT_FILE, '..', 'datasets', 'hepititus', 'description.txt'), 'r') as f:
        data['description'] = f.read()
    # shuffle the dataset
    data = deltas.data.utils.shuffle_data(data, seed=seed)  # type: ignore
    # split into train, test
    train_data, test_data = deltas.data.utils.proportional_split( # type: ignore
        data, 
        # size=0.8, 
        size=0.5, 
        seed=seed)  # , ratio=5)  # type: ignore
    return train_data, test_data

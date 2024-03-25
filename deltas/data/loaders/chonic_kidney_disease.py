'''
loader for chronic kidney disease dataset: https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease
This dataset can be used to predict the chronic kidney disease and it can be collected from the hospital nearly 2 months of period.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import os
import pandas as pd
import deltas

CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


def get_Habermans_breast_cancer(**kwargs):
    data = {}
    df = pd.read_csv(os.path.join(CURRENT_FILE, '..',
                     'datasets', 'chronic_kideny_disease', 'data.csv'))
    #TODO: sort out the missing values - remove or imput?

    data['y'] = df.pop('class').to_numpy()
    # classes are ?, notpresent, notckd and ckd (chronic kidney disease) 
    # take as ckd or not ckd for ML algorithms
    #TODO: process 'class'

    data['X'] = df.to_numpy()
    data['feature_names'] = df.columns.to_list()
    # add name and description
    with open(os.path.join(CURRENT_FILE, '..', 
                           'datasets', 'chronic_kideny_disease', 'description.txt'), 'r') as f:
        data['description'] = f.read()
    # shuffle the dataset
        data = deltas.data.utils.shuffle_data(data)  # type: ignore
    # split into train, test
    train_data, test_data = deltas.data.utils.proportional_split( # type: ignore
        data, size=0.7)  # type: ignore
    return train_data, test_data

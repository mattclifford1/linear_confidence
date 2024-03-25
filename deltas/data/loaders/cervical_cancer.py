'''
loader for cervical cancer dataset: https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors
This dataset focuses on the prediction of indicators/diagnosis of cervical cancer. The features cover demographic information, habits, and historic medical records.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import os
import pandas as pd
import deltas

CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


def get_Habermans_breast_cancer(**kwargs):
    data = {}
    df = pd.read_csv(os.path.join(CURRENT_FILE, '..',
                     'datasets', 'cervical_cancer', 'data.csv'))
    #TODO: sort out the missing values - remove or imput?

    #TODO: which col to use as class?
    data['y'] = df.pop('????').to_numpy()

    data['X'] = df.to_numpy()
    #TODO: which cols to use as features? (there might be multiple targets)
    data['feature_names'] = df.columns.to_list()
    # add name and description
    with open(os.path.join(CURRENT_FILE, '..', 
                           'datasets', 'cervical_cancer', 'description.txt'), 'r') as f:
        data['description'] = f.read()
    # shuffle the dataset
        data = deltas.data.utils.shuffle_data(data)  # type: ignore
    # split into train, test
    train_data, test_data = deltas.data.utils.proportional_split( # type: ignore
        data, size=0.7)  # type: ignore
    return train_data, test_data

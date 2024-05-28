'''
loader for Wisconsins breast cancer: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.  
They describe characteristics of the cell nuclei present in the image
A few of the images can be found at http://www.cs.wisc.edu/~street/images/
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import os
import pandas as pd
import numpy as np
import deltas

CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


def get_Wisconsin_breast_cancer(seed=True, **kwargs):
    data = {}
    df = pd.read_csv(os.path.join(CURRENT_FILE, '..',
                     'datasets', 'breast_cancer_Wisconsin', 'data.csv'))
    data['y'] = df.pop('diagnosis')
    data['y'][data['y']=='B'] = 0  
    data['y'][data['y']=='M'] = 1
    data['y'] = data['y'].to_list()
    data['y'] = np.array(data['y'])
    df.pop('ID')
    data['X'] = df.to_numpy()
    data['feature_names'] = df.columns.to_list()
    # add name and description
    with open(os.path.join(CURRENT_FILE, '..', 'datasets', 'breast_cancer_Wisconsin', 'description.txt'), 'r') as f:
        data['description'] = f.read()
    # shuffle the dataset
    data = deltas.data.utils.shuffle_data(data)  # type: ignore
    # split into train, test
    train_data, test_data = deltas.data.utils.proportional_split( # type: ignore
        data, size=0.45, ratio=10, seed=seed)  # type: ignore
    return train_data, test_data

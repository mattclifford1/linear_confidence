'''
loader for Habermans breast cancer: https://archive.ics.uci.edu/dataset/43/haberman+s+survival
The dataset contains cases from a study that was conducted between 1958 and 1970 at the 
University of Chicago's Billings Hospital on the survival of patients who had undergone surgery 
for breast cancer.
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import os
import pandas as pd
import deltas

CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


def get_Habermans_breast_cancer(**kwargs):
    data = {}
    df = pd.read_csv(os.path.join(CURRENT_FILE, '..',
                     'datasets', 'Habermans_breast_cancer', 'data.csv'))
    data['y'] = df.pop('Survived_Longer_5_Years').to_numpy()
    data['y'][data['y']==1] = 0  
    data['y'][data['y']==2] = 1
    data['X'] = df.to_numpy()
    data['feature_names'] = df.columns.to_list()
    # add name and description
    with open(os.path.join(CURRENT_FILE, '..', 'datasets', 'Habermans_breast_cancer', 'description.txt'), 'r') as f:
        data['description'] = f.read()
    # shuffle the dataset
    data = deltas.data.utils.shuffle_data(data)  # type: ignore
    # split into train, test
    train_data, test_data = deltas.data.utils.proportional_split( # type: ignore
        data, size=0.7)  # type: ignore
    return train_data, test_data

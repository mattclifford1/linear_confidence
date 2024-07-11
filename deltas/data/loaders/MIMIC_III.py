'''
loader for MIMIC-IV: ready to discharge from ICU prediction 
    - 1 is negative outcome (death or readdmission) 
    - 0 successful discharge
N.B. MIMIC dataset is not provided due to lisencing - you will need to download and process yourself (or email Matt for help)

read the paper for full details: https://bmjopen.bmj.com/content/bmjopen/9/3/e025925.full.pdf
and the github repo https://github.com/UHBristolDataScience/smartt-algortihm/tree/main
processing https://github.com/UHBristolDataScience/towards-decision-support-icu-discharge


Resampled data we exclude IMPUTED: 
    row 2 to 784 (outcome 0): original NRFD
    row 785 to 6606 (outcome 0): resampled NRFD (again should be in blocks of monotonically increasing ICUSTAY_ID, of which you could take one block).
    row 6607 to 13244 (outcome 1): RFD

Resampled data we exclude COMPLETECASE:
    row 2 to 2508: RFD
    row 2509 to 2837: original NRFD
    row 2838 onwards: resampled NRFD 

'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import os
import pandas as pd
import numpy as np
import deltas

CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


def get_mortality(seed=True, complete=False, **kwargs):
    data = {}
    if complete == False:
        file = 'fm_MIMIC_IMPUTED_extended.csv'
    else:
        file = 'fm_MIMIC_COMPLETECASE_extended.csv'
    df = pd.read_csv(os.path.join(CURRENT_FILE, '..', '..', '..',
                     'data', 'MIMIC-III', file))
    
    # remove resampled data we dont care about for the true dataset
    if complete == False:
        df.drop(df.index[0:2], inplace=True)
        df.drop(df.index[785:6606], inplace=True)
    else:
        df.drop(df.index[0:2], inplace=True)
        df.drop(df.index[2837:], inplace=True)

    # not a feature
    df.pop('cohort')
    # not useful features according to paper
    df.pop('age')
    df.pop('sex')
    df.pop('bmi')
    df.pop('los')

    # label
    data['y'] = df.pop('outcome').to_numpy()
    # swap the inds as use convention of minority being 1
    # data orignally is 0 is negative outcome (death or readdmission) - 1 successful discharge
    # but we change that to be the opposite
    data['y'][data['y'] == 1] = 2
    data['y'][data['y'] == 0] = 1
    data['y'][data['y'] == 2] = 0

    # features
    data['X'] = df.to_numpy()
    data['feature_names'] = df.columns.to_list()
    # shuffle the dataset
    data = deltas.data.utils.shuffle_data(data, seed=seed)  # type: ignore
    # split into train, test
    train_data, test_data = deltas.data.utils.proportional_split(  # type: ignore
        data, 
        size=0.5, 
        # size=0.7, 
        seed=seed)
    return train_data, test_data


def get_sepsis(seed=True, **kwargs):
    # https://www.kaggle.com/datasets/missan/mimic-challenge-2019?select=CleanDataSet.csv
    
    data = {}
    df = pd.read_csv(os.path.join(CURRENT_FILE, '..', '..', '..',
                     'data', 'MIMIC-III', 'mimic_challenge_2019_sepsis.csv'))
    

    # not a feature
    df.pop('ID')

    # label
    data['y'] = df.pop('SepsisLabel').to_numpy()

    # features
    data['X'] = df.to_numpy()
    data['feature_names'] = df.columns.to_list()
    # shuffle the dataset
    data = deltas.data.utils.shuffle_data(data, seed=seed)  # type: ignore
    # split into train, test
    train_data, test_data = deltas.data.utils.proportional_split(  # type: ignore
        data, size=0.5, seed=seed)
    return train_data, test_data

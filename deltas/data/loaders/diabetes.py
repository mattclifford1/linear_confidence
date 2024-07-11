'''
loader for the Pima Indians Diabetes Database: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?resource=download
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import os
import pandas as pd
import deltas

CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))

def get_diabetes_indian(seed=True, **kwargs):
    data = {}
    df = pd.read_csv(os.path.join(CURRENT_FILE, '..',
                     'datasets', 'diabetes_pima_indians', 'data.csv'))
    
    data['y'] = df.pop('Outcome').to_numpy()

    # keep:            glucose, BMI, age, insulin, and skin thickness
    # maybe keep only: glucose, BMI, age
    # according to https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8943493/ 
    remove_cols = [
        # 'Pregnancies',
        # 'DiabetesPedigreeFunction', 
        # 'BloodPressure', 
    #    'Insulin',
    #    'SkinThickness'
                   ]
    for col in remove_cols:
        df.pop(col)
    data['X'] = df.to_numpy()
    data['feature_names'] = df.columns.to_list()
    # add name and description
    with open(os.path.join(CURRENT_FILE, '..', 'datasets', 'diabetes_pima_indians', 'description.txt'), 'r') as f:
        data['description'] = f.read()
    # shuffle the dataset
    data = deltas.data.utils.shuffle_data(data, seed=seed)  # type: ignore
    # split into train, test
    train_data, test_data = deltas.data.utils.proportional_split( # type: ignore
        data, 
        size=0.5, 
        # size=0.2, 
        ratio=10, 
        seed=seed)  # type: ignore
    # size=0.2, ratio=5   works nicely
    return train_data, test_data

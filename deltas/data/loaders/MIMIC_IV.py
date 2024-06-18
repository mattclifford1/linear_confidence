'''
loader for MIMIC-IV: ready to discharge from ICU prediction 
    Key outcome data is the RFD variable:
        0 = Not ready for discharge (ie currently in ICU)    -  1616473   instances
        1 = Successfully discharged (ie went home)           -     7634   instances
        2 = Died                                             -    12243   instances

N.B. MIMIC dataset is not provided due to lisencing - you will need to download and process yourself (or email Matt for help)

# code from https://github.com/jeffnclark/TraCE/blob/main/helpers/funcs_icu_study.py
# paper https://proceedings.mlr.press/v233/clark24a/clark24a.pdf 
'''
# author: Matt Clifford <matt.clifford@bristol.ac.uk>

import os
import pandas as pd
import numpy as np
import deltas

CURRENT_FILE = os.path.dirname(os.path.abspath(__file__))


def get_ready_for_discharge(seed=True, **kwargs):
    desired_variables = ['stay_id',
                         'biocarbonate',
                         'bloodOxygen',
                         'bloodPressure',
                         'bun',
                         'creatinine',
                         'fio2',
                         'haemoglobin',
                         'heartRate',
                         'motorGCS',
                         'eyeGCS',
                         'potassium',
                         'respiratoryRate',
                         'sodium',
                         'Temperature [C]',
                         'verbalGCS',
                         'age',
                         'gender',
                         'hours_since_admission',
                         'RFD']
    filepath = os.path.join(CURRENT_FILE, '..', '..', '..',
                            'data', 'MIMIC-IV', 'full_datatable_timeSeries_Labels.csv')

    df = initial_icu_processing(filepath, desired_variables)

    # drop one of the classes (died)
    df = df[df.RFD != 2]


    data = {}
    data['y'] = df.pop('RFD').to_numpy()#[:6638+954]   # number from paper above


    # potentially try random selection of class 0?
    data['X'] = df.to_numpy()#[:6638+954]
    data['feature_names'] = df.columns.to_list()
    # shuffle the dataset
    data = deltas.data.utils.shuffle_data(data, seed=seed)  # type: ignore
    # split into train, test
    train_data, test_data = deltas.data.utils.proportional_split( # type: ignore
        data, size=0.1, seed=seed)
    return train_data, test_data


def nan_post_processing_data(df_data, columns):
    '''
    Function for post processing the dataframe with all the proceessed values (eg removing nans)
    input: 
    df_data -  dataframe which contain the values of all the processed files
    columns -  column of interest to perform data processing on
    returns:
    df_data -  post-processed dataframe
    '''
    postprocessSettings = 2

    # for column in df_data:
    for column in columns:
        # determine replacement values
        if postprocessSettings == 0:
            # leave empty field empty
            newValue = ''
        elif postprocessSettings == 1:
            # replace empty field by a zero
            newValue = 0
        elif postprocessSettings == 2:
            # replace empty field with
            # the mean value for column containing numerical values,
            # and the most frequent value in columns
            # with non-numerical/categorical values
            if 'GCS' in column:
                newValue = df_data[column].mode()
                # FOR THE CASE OF THE MODE, NEED TO GET THE FIRST INDEX
                df_data[column] = df_data[column].fillna(newValue[0])
            elif 'gender' in column:
                newValue = df_data[column].mode()
                df_data[column] = df_data[column].fillna(newValue[0])
            else:
                newValue = df_data[column].mean()
                df_data[column] = df_data[column].fillna(newValue)
    return df_data


def change_categorical(df, categorical_features):
    '''
    function: Used to change data variables to categorical type (for the case of passing to DiCE)
    '''
    for column in categorical_features:
        df[column] = df[column].astype('category')
    return df


def initial_icu_processing(filepath, features):
    '''
    Function used to perform initial preprocessing of the data, this includes:
    choosing the correct columns of interest,
    changing the label to be negative outocome label from -1 to 2 (for the purpose of DiCE),
    changing the gender label,
    as well as filling in missing values in the data
    input:
        filepath - path of the csv file which cotains the data
        features - which features you want to extract and use

    return:desired_df_data -  dataframe after processing the data 
    '''
    df_data = pd.read_csv(filepath,
                          header=0)
    # Get the variables of interest

    # Obtain only the variables of interes
    desired_df_data = df_data[features]
    # Replace values of -1 with 2 (NEGATIVE LABELLED CLASSES DOES NOT SEEM TO WORK WITH DICE)
    desired_df_data['RFD'].replace(-1, 2, inplace=True)

    # Change gender category from string to float
    desired_df_data['gender'].replace('M', 0, inplace=True)
    desired_df_data['gender'].replace('F', 1, inplace=True)
    desired_df_data['gender'] = desired_df_data['gender'].astype('float')

    # Update the mean and the standard deviation for the data
    columns_to_process = desired_df_data.drop(
        columns=['RFD', 'stay_id', 'hours_since_admission'], axis=1).columns.tolist()
    # columns_to_process=desired_df_data.columns.tolist()

    neutral_data = desired_df_data[desired_df_data['RFD'] == 0]
    positive_data = desired_df_data[desired_df_data['RFD'] == 1]
    negative_data = desired_df_data[desired_df_data['RFD'] == 2]

    # Fill in missing values of the data
    processed_neutral_data = nan_post_processing_data(
        neutral_data, columns_to_process)
    processed_negative_data = nan_post_processing_data(
        negative_data, columns_to_process)
    processed_positive_data = nan_post_processing_data(
        positive_data, columns_to_process)

    desired_df_data.update(processed_neutral_data)
    desired_df_data.update(processed_negative_data)
    desired_df_data.update(processed_positive_data)

    return desired_df_data

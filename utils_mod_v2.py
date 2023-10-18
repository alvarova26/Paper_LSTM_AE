import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve,roc_auc_score
import torch
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers
from keras.callbacks import EarlyStopping
from scipy import stats

import time
from datetime import timedelta

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


########################################################################################################## REMOVE VERY EXTREME OUTLIERS
def replace_outliers_z_score(data, z_threshold=3, rep_by_days_back=1, epochs=2):
    # Z-Score Method
    '''
    The function returns a modified DataFrame (res_df) in which the outliers have been replaced with values from previous
    time points. The structure of the DataFrame remains the same as the input DataFrame.
    The function applies the Z-score method to detect and replace outliers in a DataFrame.
    It iterates over columns and epochs, identifies outliers based on Z-scores, and replaces
    them with values from previous time points if available
    '''
    res_df = data.copy()
    for e in range(epochs):
        for col in res_df.columns:
            z_scores = (res_df[col] - res_df[col].mean()) / res_df[col].std()
            z_outliers_indices = res_df[abs(z_scores) > z_threshold].index
            print('Z-Score => Epoch #' + str(e+1) + ': Replacing ' + str(len(z_outliers_indices)) +' in ' + str(col))
            for i in z_outliers_indices:
                if (i - timedelta(days=rep_by_days_back)) > res_df[col].index[0]:
                    res_df.at[i, col] = res_df.at[i - timedelta(days=rep_by_days_back), col]
                elif (i - timedelta(hours=1)) > res_df[col].index[0]:
                    res_df.at[i, col] = res_df.at[i - timedelta(hours=1), col]
    return res_df #, z_outliers_indices

def replace_outliers_iqr(data, iqr_k=3, rep_by_days_back=1, epochs=2):
    # IQR Method
    '''
    The function returns a modified DataFrame (res_df) in which the outliers have been replaced with values from
    previous time points. The structure of the DataFrame remains the same as the input DataFrame.
    The function applies the IQR method to detect and replace outliers in a DataFrame.
    It iterates over columns and epochs, identifies outliers based on the IQR and specified multiplier, and 
    replaces them with values from previous time points if available
    '''
    res_df = data.copy()
    for e in range(epochs):
        for col in res_df.columns:
            q1 = res_df[col].quantile(0.25)
            q3 = res_df[col].quantile(0.75)
            iqr = q3 - q1
            iqr_lower_bound = q1 - iqr_k * iqr
            iqr_upper_bound = q3 + iqr_k * iqr
            iqr_outliers_indices = res_df[(res_df[col] < iqr_lower_bound) | (res_df[col] > iqr_upper_bound)].index
            print('IQR (Interquartile Range) => Epoch #' + str(e+1) + ': Replacing ' + str(len(iqr_outliers_indices)) +' in ' + str(col))
            for i in iqr_outliers_indices:
                if (i - timedelta(days=rep_by_days_back)) > res_df[col].index[0]:
                    res_df.at[i, col] = res_df.at[i - timedelta(days=rep_by_days_back), col]
                elif (i - timedelta(hours=1)) > res_df[col].index[0]:
                    res_df.at[i, col] = res_df.at[i - timedelta(hours=1), col]
    return res_df #, iqr_outliers_indices

########################################################################################################## SMOOTHING FUNCTIONS
def remove_outliers_rolling_windows_vec(df, rolling_window_size=12, epochs=1, rolling_window_std_threshold=2, center=False):
    '''
    The function returns a modified DataFrame (res_df) in which the outliers have been replaced with rolling mean values.
    The structure of the DataFrame remains the same as the input DataFrame.
    This function utilizes rolling windows to detect and replace outliers in a DataFrame.
    It iterates over epochs and smooths the data by replacing outliers with rolling mean values. 
    '''

    # Variable control
    assert isinstance(rolling_window_size, int), 'Assert: rolling_window_size must be integer'
    assert isinstance(epochs, int), 'Assert: epochs must be integer'
    assert isinstance(rolling_window_std_threshold, (int, float)) and 1 <= rolling_window_std_threshold , "Assert: rolling_window_std_threshold must be either int or float greather than or equal to 1"
    assert epochs > 0, 'Assert: epochs must be at least 1'
    assert rolling_window_size > 0, 'Assert: rolling_window_size size must be at least 1'

    # Code
    res_df = df.copy()

    if isinstance(res_df, pd.DataFrame):
        for e in range(epochs):
            print('Epoch: ' + str(e) + ' Removing outliers from series!')
            rolling_mean = res_df.rolling(rolling_window_size, min_periods=1, center=center).mean()           # Calculate the rolling mean for the series
            rolling_std = res_df.rolling(rolling_window_size, min_periods=1, center=center).std()             # Calculate the standard deviation for the series

            upper_threshold = rolling_mean + (rolling_window_std_threshold * rolling_std)
            lower_threshold = rolling_mean - (rolling_window_std_threshold * rolling_std)                               # Calculate the lower and upper thresholds for outliers

            res_df = np.where((res_df > upper_threshold) | (res_df < lower_threshold), rolling_mean, res_df)            # Replace outliers with the mean of the rolling window
            res_df = pd.DataFrame(res_df, columns=df.columns, index=df.index)                                   # Create a new DataFrame with the smoothed values

    return res_df

########################################################################################################## LSTM AE MODELS
def lstm_ae_model_v1(X):
    '''
    This function provides a Keras model for an LSTM-based autoencoder.
    The encoder reduces the dimensionality of the input data, and the decoder attempts to reconstruct the original data.
    
    Input Layers    = 2 (16/4)
    Output Layers   = 2 (4/16)
    '''
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(16, activation='relu', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)
    model = Model(inputs=inputs, outputs=output)
    return model

def lstm_ae_model_v2(X, anomaly_layer='no', early_stopping=False, monitor='loss', patience=5, min_delta=0.001):
    '''
    This function provides a Keras model for an LSTM-based autoencoder.
    The encoder reduces the dimensionality of the input data, and the decoder attempts to reconstruct the original data.
    Early stopping can be employed during training to prevent overfitting.
    Input Layers    = 2 (16/4)
    Output Layers   = 2 (4/16)
    '''
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(16, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)
    
    if anomaly_layer=='yes':
        anomaly = LSTM(1, activation='sigmoid', return_sequences=True)(L5)
        model = Model(inputs=inputs, outputs=[output, anomaly])
    else:
        model = Model(inputs=inputs, outputs=output)

    if early_stopping:
        early_stopping = EarlyStopping(monitor=monitor, patience=patience, min_delta=min_delta, restore_best_weights=True)
        return model, early_stopping

    return model

def lstm_ae_model_v3(X, anomaly_layer='no', early_stopping=False, monitor='loss', patience=5, min_delta=0.001):
    '''
    This function provides a Keras model for an LSTM-based autoencoder.
    The encoder reduces the dimensionality of the input data, and the decoder attempts to reconstruct the original data.
    Early stopping can be employed during training to prevent overfitting.
    Input Layers    = 2 (64/16)
    Output Layers   = 2 (16/64)
    '''
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(64, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(16, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(16, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(64, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)
    
    if anomaly_layer=='yes':
        anomaly = LSTM(1, activation='sigmoid', return_sequences=True)(L5)
        model = Model(inputs=inputs, outputs=[output, anomaly])
    else:
        model = Model(inputs=inputs, outputs=output)

    if early_stopping:
        early_stopping = EarlyStopping(monitor=monitor, patience=patience, min_delta=min_delta, restore_best_weights=True)
        return model, early_stopping

    return model

def lstm_ae_model_v4(X, anomaly_layer='no', early_stopping=False, monitor='loss', patience=5, min_delta=0.001):
    '''
    This function provides a Keras model for an LSTM-based autoencoder.
    The encoder reduces the dimensionality of the input data, and the decoder attempts to reconstruct the original data.
    Early stopping can be employed during training to prevent overfitting.
    Input Layers    = 2 (128/16)
    Output Layers   = 2 (16/128)
    '''
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(128, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(16, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(16, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(128, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)

    if anomaly_layer=='yes':
        anomaly = LSTM(1, activation='sigmoid', return_sequences=True)(L5)
        model = Model(inputs=inputs, outputs=[output, anomaly])
    else:
        model = Model(inputs=inputs, outputs=output)

    if early_stopping:
        early_stopping = EarlyStopping(monitor=monitor, patience=patience, min_delta=min_delta, restore_best_weights=True)
        return model, early_stopping

    return model

def lstm_ae_model_v5(X, anomaly_layer='no', early_stopping=False, monitor='loss', patience=5, min_delta=0.001):
    '''
    This function provides a Keras model for an LSTM-based autoencoder.
    The encoder reduces the dimensionality of the input data, and the decoder attempts to reconstruct the original data.
    Early stopping can be employed during training to prevent overfitting.
    Input Layers    = 4 (128/64/32/16)
    Output Layers   = 4 (16/32/64/128)
    '''
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(128, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(64, activation='relu', return_sequences=True)(L1)
    L3 = LSTM(32, activation='relu', return_sequences=True)(L2)
    L4 = LSTM(16, activation='relu', return_sequences=False)(L3)
    L5 = RepeatVector(X.shape[1])(L4)
    L6 = LSTM(16, activation='relu', return_sequences=True)(L5)
    L7 = LSTM(32, activation='relu', return_sequences=True)(L6)
    L8 = LSTM(64, activation='relu', return_sequences=True)(L7)
    L9 = LSTM(128, activation='relu', return_sequences=True)(L8)
    output = TimeDistributed(Dense(X.shape[2]))(L9)

    if anomaly_layer=='yes':
        anomaly = LSTM(1, activation='sigmoid', return_sequences=True)(L9)
        model = Model(inputs=inputs, outputs=[output, anomaly])
    else:
        model = Model(inputs=inputs, outputs=output)

    if early_stopping:
        early_stopping = EarlyStopping(monitor=monitor, patience=patience, min_delta=min_delta, restore_best_weights=True)
        return model, early_stopping

    return model

def lstm_ae_model_v6(X, anomaly_layer='no', early_stopping=False, monitor='loss', patience=5, min_delta=0.001):
    '''
    This function provides a Keras model for an LSTM-based autoencoder.
    The encoder reduces the dimensionality of the input data, and the decoder attempts to reconstruct the original data.
    Early stopping can be employed during training to prevent overfitting.
    Input Layers    = 5 (256/128/64/32/16)
    Output Layers   = 5 (16/32/64/128/256)
    '''
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(256, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(128, activation='relu', return_sequences=True)(L1)
    L3 = LSTM(64, activation='relu', return_sequences=True)(L2)
    L4 = LSTM(32, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(16, activation='relu', return_sequences=False)(L4)
    L6 = RepeatVector(X.shape[1])(L5)
    L7 = LSTM(16, activation='relu', return_sequences=True)(L6)
    L8 = LSTM(32, activation='relu', return_sequences=True)(L7)
    L9 = LSTM(64, activation='relu', return_sequences=True)(L8)
    L10 = LSTM(128, activation='relu', return_sequences=True)(L9)
    L11 = LSTM(256, activation='relu', return_sequences=True)(L10)
    output = TimeDistributed(Dense(X.shape[2]))(L11)

    if anomaly_layer=='yes':
        anomaly = LSTM(1, activation='sigmoid', return_sequences=True)(L11)
        model = Model(inputs=inputs, outputs=[output, anomaly])
    else:
        model = Model(inputs=inputs, outputs=output)

    if early_stopping:
        early_stopping = EarlyStopping(monitor=monitor, patience=patience, min_delta=min_delta, restore_best_weights=True)
        return model, early_stopping

    return model

########################################################################################################## LSTM AE FUNCTIONS
def get_lstm_ae_predictions(model, X, x_col, x_idx, anomaly_detect=True):
    '''
    The function returns a DataFrame (X_pred) containing the predictions generated by the model. The structure of the
    DataFrame is consistent with the input data, and it can be used for various purposes, such as evaluating model
    performance or visualizing the reconstructed data.
    This function takes an LSTM-based Autoencoder model, input data, and related metadata (column names and indices) and
    generates predictions. It can handle both standard reconstruction tasks and anomaly detection tasks, depending on the
    model's configuration. The resulting predictions are organized into a DataFrame for further analysis or visualization.
    '''
    X_aux = X.copy()
    if anomaly_detect: X_pred, anomaly = model.predict(X_aux)
    else: X_pred = model.predict(X_aux)
    X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
    X_pred = pd.DataFrame(X_pred, index=x_idx, columns=x_col)
    return X_pred

def get_lstm_ae_avg_threshold_per_of_max(X_scores, sensitivity=0.95):
    '''
    The function returns the calculated threshold value, which can be used for setting an anomaly detection threshold for further
    analysis or visualization.
    This function is useful for determining a threshold for anomaly detection based on a percentage of the maximum anomaly
    score. The calculated threshold provides a flexible way to identify anomalies in a dataset by adjusting the sensitivity.
    '''
    X_aux_scores = X_scores.copy()
    dist_col = [col for col in X_aux_scores.columns if 'dist' in col.lower()]
    dist_col = dist_col[0]
    threshold = np.max(X_aux_scores[dist_col]) * sensitivity
    print('Avg. Threshold (' + str(100 * sensitivity) + '% of max(dist)):\t' + str(threshold))
    return threshold

def get_lstm_ae_avg_threshold_per_of_sum(X_scores, sensitivity=0.99):
    '''
    The function returns the calculated threshold value, which can be used for setting an anomaly detection threshold for further
    analysis or visualization.
    This function provides a way to determine an anomaly detection threshold by considering a percentage of the cumulative
    sum of anomaly scores. The threshold calculation is based on the desired sensitivity level. This can be useful for identifying
    anomalies in a dataset where the distribution of anomaly scores varies.
    '''
    X_aux_scores = X_scores.copy()
    dist_col = [col for col in X_aux_scores.columns if 'dist' in col.lower()]
    dist_col = dist_col[0]
    sorted_values = X_aux_scores[dist_col].sort_values().reset_index().drop(columns=['Timestamp'])
    sorted_values['Acc_Sum'] = sorted_values.cumsum()
    sorted_values['Perc_Total_Sum'] = sorted_values['Acc_Sum'] / sorted_values[dist_col].sum()
    index = (sorted_values['Perc_Total_Sum'] > sensitivity).idxmax()
    threshold = sorted_values.iloc[index]
    print('Avg. Threshold (' + str(100 * sensitivity) + '% of sum(dist)):\t' + str(threshold[0]))
    return threshold[0]

def add_lstm_ae_avg_thresholds(X_scores, avg_threshold_per_of_max, avg_threshold_per_of_sum):
    '''
    The function returns the calculated threshold value, which can be used for setting an anomaly detection threshold
    based on the cumulative sum of anomaly scores. This allows for flexible sensitivity in identifying anomalies
    within a dataset.
    The function returns the modified DataFrame (X_aux_scores) with the added threshold columns and final anomaly
    score columns. These additional columns can be used for further analysis and visualization to identify anomalies
    based on the two different thresholding methods.
    This function allows you to add anomaly detection thresholds and final anomaly scores to a DataFrame of anomaly
    scores. The two thresholding methods, based on the maximum anomaly score and the cumulative sum of scores, are
    provided as additional information for identifying anomalies within the data
    '''
    X_aux_scores = X_scores.copy()
    dist_col = [col for col in X_aux_scores.columns if 'dist' in col.lower()]
    dist_col = dist_col[0]
    X_aux_scores['Avg_Threshold_POM'] = avg_threshold_per_of_max
    X_aux_scores['Avg_Threshold_POS'] = avg_threshold_per_of_sum
    X_aux_scores['Final_Anomaly_Score_POM'] = (X_aux_scores[dist_col] > avg_threshold_per_of_max).astype(int)
    X_aux_scores['Final_Anomaly_Score_POS'] = (X_aux_scores[dist_col] > avg_threshold_per_of_sum).astype(int)
    return X_aux_scores

def get_lstm_ae_ind_thresholds_per_of_max(X_scores, sensitivity=0.95):
    '''
    The function returns the 'thresholds' DataFrame, which contains the individual anomaly detection thresholds for
    each series. These thresholds can be used for setting series-specific anomaly detection criteria and are often 
    useful in cases where different series have varying anomaly score distributions.
    This function provides a way to calculate and obtain individual anomaly detection thresholds for different series
    within a DataFrame of anomaly scores. The individualized thresholds allow for customized anomaly detection for each
    series, which can be valuable in diverse data analysis scenarios.
    '''
    X_aux_scores = X_scores.copy()
    series_col = [col for col in X_aux_scores.columns if 'uuid' in col.lower()]
    thresholds = np.max(X_aux_scores[series_col]) * sensitivity
    for index, value in thresholds.items():
        print('Individual Series Threshold (' + str(index) + ' - '+ str(100 * sensitivity) + '% of max(dist)):\t', value)
    thresholds = pd.DataFrame(thresholds, columns=['Threshold_Value'])
    return thresholds

def get_lstm_ae_ind_thresholds_per_of_sum(X_scores, sensitivity=0.95):
    '''
    The function returns the 'thresholds' DataFrame, which contains the individual anomaly detection thresholds for each
    series. These thresholds are based on the cumulative sum of anomaly scores for each series and can be used for setting
    series-specific anomaly detection criteria.
    This function provides a way to calculate and obtain individual anomaly detection thresholds for different series within
    a DataFrame of anomaly scores. The thresholds are calculated based on the cumulative sum of anomaly scores and are printed
    for each series. This allows for customized anomaly detection for each series based on their specific data distributions.
    '''
    X_aux_scores = X_scores.copy()
    series_col = [col for col in X_aux_scores.columns if 'uuid' in col.lower()]
    if len(series_col) > 0:
        thresholds = {}
        for col in series_col:
            sorted_values = X_aux_scores[col].sort_values().reset_index().drop(columns=['Timestamp'])
            sorted_values['Acc_Sum'] = sorted_values.cumsum()
            sorted_values['Perc_Total_Sum'] = sorted_values['Acc_Sum'] / sorted_values[col].sum()
            index = (sorted_values['Perc_Total_Sum'] > sensitivity).idxmax()
            value = sorted_values.iloc[index][0]
            thresholds[col] = value
    thresholds = pd.DataFrame.from_dict(thresholds, orient='index', columns=['Threshold_Value'])
    for index, row in thresholds.iterrows():
        print('Individual Series Threshold (' + str(index) + ' - '+ str(100 * sensitivity) + '% of sum(dist)):\t', row[0])
    return thresholds

def get_lstm_ae_anomaly_matrix_per_of_max(X_train_scores, X_test_scores, sensitivity=0.95, anomaly_count_threshold=2):
    '''
    The function returns the anomaly_matrix, which is a DataFrame representing anomaly scores and detection results. It contains
    binary columns for each series indicating anomalies, as well as columns for the total anomaly count and the final anomaly score.
    This function is used for anomaly detection in time series data with individual anomaly detection thresholds calculated based
    on the percentage of the maximum anomaly score for each series. The function returns an anomaly matrix that provides a
    comprehensive view of anomalies in the test data. It is useful for identifying anomalies within individual series and making
    overall anomaly detection decisions based on a specified threshold.
    '''
    X_aux_train_scores = X_train_scores.copy()
    X_aux_test_scores = X_test_scores.copy()
    series_col = [col for col in X_aux_train_scores.columns if 'uuid' in col.lower()]
    if len(series_col) > 0:
        thresholds_vector = np.max(X_aux_train_scores[series_col]) * sensitivity
        anomaly_matrix = pd.DataFrame()
        for column in series_col:
            threshold = thresholds_vector[column]
            anomaly_matrix[column + '_anomaly'] = (X_aux_test_scores[column] > threshold).astype(int)
        anomaly_matrix['Total_Anomaly_Count_POM'] = anomaly_matrix.sum(axis=1)
        anomaly_matrix['Final_Anomaly_Score_POM'] = (anomaly_matrix['Total_Anomaly_Count_POM'] >= anomaly_count_threshold).astype(int)
    else: anomaly_matrix = None        
    return anomaly_matrix

def get_lstm_ae_anomaly_matrix_per_of_sum(X_train_scores, X_test_scores, sensitivity=0.99, anomaly_count_threshold=2):
    '''
    The function returns the anomaly_matrix, which is a DataFrame representing anomaly scores and detection results. It contains binary
    columns for each series indicating anomalies, as well as columns for the total anomaly count and the final anomaly score.
    This function is used for anomaly detection in time series data with individual anomaly detection thresholds calculated based on
    the cumulative sum of anomaly scores for each series. The function returns an anomaly matrix that provides a comprehensive view
    of anomalies in the test data, taking into account individual series characteristics. It is useful for identifying anomalies within
    individual series and making overall anomaly detection decisions based on a specified threshold.
    '''
    X_aux_train_scores = X_train_scores.copy()
    X_aux_test_scores = X_test_scores.copy()
    series_col = [col for col in X_aux_train_scores.columns if 'uuid' in col.lower()]
    if len(series_col) > 0:
        thresholds = {}
        for col in series_col:
            sorted_values = X_aux_train_scores[col].sort_values().reset_index().drop(columns=['Timestamp'])
            sorted_values['Acc_Sum'] = sorted_values.cumsum()
            sorted_values['Perc_Total_Sum'] = sorted_values['Acc_Sum'] / sorted_values[col].sum()
            index = (sorted_values['Perc_Total_Sum'] > sensitivity).idxmax()
            value = sorted_values.iloc[index][0]
            thresholds[col] = value
        thresholds = pd.DataFrame.from_dict(thresholds, orient='index', columns=['Threshold_Value'])
        anomaly_matrix = pd.DataFrame()
        for col in series_col:
            threshold = thresholds.loc[col, 'Threshold_Value']
            anomaly_matrix[col + '_anomaly'] = (X_aux_test_scores[col] > threshold).astype(int)
        anomaly_matrix['Total_Anomaly_Count_POS'] = anomaly_matrix.sum(axis=1)
        anomaly_matrix['Final_Anomaly_Score_POS'] = (anomaly_matrix['Total_Anomaly_Count_POS'] >= anomaly_count_threshold).astype(int)
    else: anomaly_matrix = None
    return anomaly_matrix

def get_lstm_ae_anomalous_series_from_matrix(timestamp, anomaly_matrix):
    '''
    The function returns a list of anomalous series/entities at the specified timestamp (anomalous_uuids). If there are no anomalous
    series at that timestamp, it returns None.
    This function is used to determine which series/entities are anomalous at a specific timestamp based on the anomaly matrix.
    It checks the final anomaly score for that timestamp and, if the score is 1 (indicating a global anomaly), identifies and lists
    the individual series/entities that are marked as anomalous. This can be valuable for understanding which series are contributing
    to anomalous behavior at a given moment in time.
    '''
    anomaly_matrix_aux = anomaly_matrix.copy()
    selected_row = anomaly_matrix_aux.loc[anomaly_matrix_aux.index == timestamp]
    final_anomaly_score_col = [col for col in selected_row.columns if 'Final_Anomaly_Score' in col]
    final_anomaly_score_col = final_anomaly_score_col[0]
    if selected_row[final_anomaly_score_col][0] == 1:
        uuid_col = [col for col in selected_row.columns if 'uuid' in col.lower()]
        if len(uuid_col) > 0:
            anomalous_uuids = list()
            for col in uuid_col:
                if selected_row[col][0] == 1: anomalous_uuids.append(col)
            anomalous_uuids = [col.split('_anomaly')[0] for col in anomalous_uuids if '_anomaly' in col]
    else: anomalous_uuids = None
    print('Anomalous Series @' + timestamp + ':\t' + str(anomalous_uuids))
    return anomalous_uuids

def print_lstm_ae_anomalous_series_from_matrix(anomaly_matrix):
    '''
    The function doesn't return any value; it primarily prints the information about anomalous series at specific timestamps.
    This function is useful for summarizing and printing the anomalous series/entities at timestamps where global anomalies are
    detected in the anomaly matrix. It provides insight into which series are contributing to anomalous behavior at various
    points in time.
    '''
    anomaly_matrix_aux = anomaly_matrix.copy()
    final_anomaly_score_col = [col for col in anomaly_matrix_aux.columns if 'Final_Anomaly_Score' in col]
    final_anomaly_score_col = final_anomaly_score_col[0]
    anomaly_detected = anomaly_matrix_aux[anomaly_matrix_aux[final_anomaly_score_col]==1]
    for idx in anomaly_detected.index:
        get_lstm_ae_anomalous_series_from_matrix(timestamp=str(idx), anomaly_matrix=anomaly_matrix_aux)

def get_lstm_ae_anomalous_series_from_score(timestamp, X_scores, final_anomaly_score_col='Final_Anomaly_Score_POS'):
    '''
    The function returns a list of anomalous series/entities at the specified timestamp (anomalous_uuids). If there are no
    anomalous series at that timestamp, it returns None.
    This function is used to determine which series/entities are anomalous at a specific timestamp based on the anomaly scores
    and the specified final anomaly score column. It checks the final anomaly score for that timestamp, compares it to the
    average threshold, and, if the score is 1 (indicating a global anomaly), identifies and lists the individual series/entities
    that are marked as anomalous. This can be valuable for understanding which series contribute to anomalous behavior at a given
    moment in time.
    '''
    X_aux_scores = X_scores.copy()
    selected_row = X_aux_scores.loc[X_aux_scores.index == timestamp]
    if selected_row[final_anomaly_score_col][0] == 1:
        if '_POM' in final_anomaly_score_col: avg_col = 'Avg_Threshold_POM'
        if '_POS' in final_anomaly_score_col: avg_col = 'Avg_Threshold_POS'
        uuid_col = [col for col in selected_row.columns if 'uuid' in col.lower()]
        if len(uuid_col) > 0:
            anomalous_uuids = list()
            for col in uuid_col:
                if selected_row[col][0] > selected_row[avg_col][0]: anomalous_uuids.append(col)
    else: anomalous_uuids = None
    print('Anomalous Series @' + timestamp + ':\t' + str(anomalous_uuids))
    return anomalous_uuids

def print_lstm_ae_anomalous_series_from_score(X_scores, final_anomaly_score_col='Final_Anomaly_Score_POS'):
    '''
    The function does not return any specific output, but it prints messages indicating which series/entities are anomalous at
    specific timestamps based on the provided anomaly scores.
    This function is used to identify and report anomalous series/entities at specific timestamps based on the anomaly scores
    and the specified final anomaly score column. It searches for timestamps where the entire dataset is considered anomalous and
    then calls another function to report which individual series/entities contribute to this anomaly. This can be helpful for
    diagnosing and understanding the specific sources of anomalies in a time series dataset.
    '''
    X_aux_scores = X_scores.copy()
    anomaly_detected = X_aux_scores[X_aux_scores[final_anomaly_score_col]==1]
    for idx in anomaly_detected.index:
        get_lstm_ae_anomalous_series_from_score(timestamp=str(idx), X_scores=X_aux_scores, final_anomaly_score_col=final_anomaly_score_col)

########################################################################################################## LSTM AE FUNCTIONS WINDOWS APPROACH

def transform_to_windows(X, win_size=12, win_dir='backward'):
    '''
    The function returns a numpy array containing the collection of windows. Each row in the array represents a window, and the
    columns within each row contain the data points for that window. The order of windows in the array depends on the specified
    win_dir.
    This function is useful for converting a time series dataset into a format suitable for tasks like time series forecasting
    or sequence modeling. It allows you to create overlapping windows of data, which can capture temporal patterns and dependencies
    in the time series. The direction of the windows (backward or forward) provides flexibility in how you want to structure the
    data for your specific analysis.
    '''
    X_aux = X.copy()
    X_windows = []
    if win_dir == 'backward':
        for i in range(win_size, len(X_aux)+1):
            X_windows.append(X_aux[i-win_size:i][::-1])
    if win_dir == 'forward':
        for i in range(win_size, len(X_aux)+1):
            X_windows.append(X_aux[i-win_size:i])
    return np.array(X_windows)

def get_lstm_ae_predictions_3d(model, X_3d, anomaly_detect=False):
    '''
    The primary output of the function is X_pred_3d, which is a 3D numpy array representing the model's predictions for the
    input data. This array has the same shape as the input data.
    Additionally, if anomaly_detect is set to True, there might be an additional output, the anomaly variable, which typically
    contains anomaly scores or labels generated during the prediction process. The specific details of the anomaly variable would
    depend on how it is defined in your code.
    This function is used to obtain predictions from an LSTM Autoencoder model when working with 3D input data, which is common
    in time series data analysis. It can also perform anomaly detection if the anomaly_detect flag is set to True, but further
    details on how anomalies are detected and reported would depend on the broader context of your code.
    '''
    X_aux = X_3d.copy()
    if anomaly_detect: X_pred_3d, anomaly = model.predict(X_aux)
    else: X_pred_3d = model.predict(X_aux)
    return X_pred_3d

def get_lstm_ae_scores_3d(X_3d, X_pred_3d, dist='sqr'):
    '''
    The primary output of the function is X_scores_3d, which is a 3D numpy array containing the computed scores or distances.
    This array has the same shape as the input data, and each element represents the score or distance between the corresponding
    elements of the original and reconstructed data.
    This function is used to calculate scores or distances between the original 3D input data and the reconstructed data, and
    it provides flexibility in choosing whether to compute absolute differences ('abs') or squared differences ('sqr'). These
    scores can be valuable for assessing the performance of an LSTM Autoencoder model in reproducing the input data.
    '''
    X_aux = X_3d.copy()
    X_aux_pred = X_pred_3d.copy()
    if dist=='abs':
        X_scores_3d = np.abs(X_aux_pred-X_aux)
    if dist=='sqr':
        X_scores_3d = (X_aux_pred-X_aux)**2
    return X_scores_3d

def get_lstm_ae_scores_2d(X_3d, x_2d, win_size=12, win_dir='backward', dist='sqr'):
    '''
    The primary output of the function is X_scores_2d, a pandas DataFrame containing the calculated scores. The DataFrame will
    have a datetime index and columns representing the average score for each timestamp in the original data. The column name
    will include the type of distance used in the score calculation (e.g., 'Avg. Dist (sqr)').
    This function is used to calculate average scores or distances between a 3D data array and a 2D reference dataset over a
    rolling window. The function provides flexibility in the choice of the rolling window direction and the type of distance
    used in the score calculation. This can be useful for assessing the quality of data reconstruction by an LSTM Autoencoder.
    '''
    X_aux = X_3d.copy()
    x_ref = x_2d.copy()
    if win_dir == 'backward':
        X_scores_2d = np.mean(X_aux, axis=1)
        X_scores_2d = pd.DataFrame(X_scores_2d, index=x_ref.index[-(len(x_2d)-win_size+1):], columns=x_ref.columns)
        X_scores_2d ['Avg. Dist (' + dist + ')'] = np.mean(X_scores_2d, axis = 1)
    if win_dir == 'forward':
        X_scores_2d = np.mean(X_aux, axis=1)
        X_scores_2d = pd.DataFrame(X_scores_2d, index=x_ref.index[:(len(x_2d)-win_size+1)], columns=x_ref.columns)
        X_scores_2d ['Avg. Dist (' + dist + ')'] = np.mean(X_scores_2d, axis = 1)
    return X_scores_2d

########################################################################################################## PLOT/VIZ FUNCTIONS
def plot_lstm_ae_train_val_loss(history):
    '''
    The primary output of the function is a visualization, specifically a line plot that shows the trend of training and
    validation loss values over epochs.
    This function is typically used after training a neural network, like an LSTM Autoencoder, to visually assess how well the
    model is learning from the training data and whether it's overfitting or underfitting. The function's output plot provides
    insights into the model's training progress and helps in determining whether additional training epochs are necessary or
    if the model is ready for deployment.
    '''
    # Plot Train and Validation losses
    fig, ax = plt.subplots(figsize=(20, 6), dpi=80)
    ax.plot(history['loss'], 'b', label='Train', linewidth=2)
    ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
    ax.set_title('Train and Validation Loss', fontsize=16)
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')
    plt.show()

def plot_lstm_ae_model_dist_distribution(X_scores, dataset='Default', bins=None, threshold_POM=None, threshold_POS=None, xlim_min=None, xlim_max=None):
    '''
    The primary output of the function is a visualization, specifically a histogram of the distance values from the dataset. This
    histogram can provide insights into the distribution of anomaly scores, making it easier to identify data points that may be
    considered anomalies based on the threshold lines.
    This function is useful for assessing the performance of an anomaly detection model by visualizing how well it separates anomalies
    from normal data based on the distance values. It can help data analysts and model evaluators decide on appropriate threshold 
    values for classification.
    '''
    dist_col = [col for col in X_scores.columns if 'dist' in col.lower()]
    dist_col = dist_col[0]
    if xlim_min is None: xlim_min=np.min(X_scores[dist_col])
    if xlim_max is None: xlim_max=np.max(X_scores[dist_col])+(np.max(X_scores[dist_col])/100)

    plt.figure(figsize=(20,6))
    plt.title(dataset + ' Distance Distribution', fontsize=16)
    plt.xlim([xlim_min,xlim_max])

    if threshold_POM is not None:
        plt.axvline(threshold_POM, color='red', linestyle='dashed', label='Threshold_POM')
    if threshold_POS is not None:
        plt.axvline(threshold_POS, color='green', linestyle='dashed', label='Threshold_POS')
    plt.legend()

    sns.distplot(X_scores[dist_col], bins=bins, kde=True, color='blue');

def plot_lstm_ae_ind_series_dist_threshold_orig_val(X_scores, dataset, serie='default', threshold_POM=None, threshold_POS=None, train_split=None, train_split_idx=None):
    '''
    The function produces a pair of subplots in the same figure. The top subplot displays the individual series of log-scaled distance values
    along with horizontal dashed lines indicating the threshold values. The bottom subplot displays the original time series data for comparison.
    Vertical dashed lines are used to indicate the split between training and test data if provided.
    This function is particularly useful for assessing the performance of an anomaly detection model on a specific time series. It helps to
    understand how well the model's threshold values separate anomalies from normal data in a single series. The log-scale in the top subplot
    can enhance visibility for a wide range of anomaly scores. The function provides a clear way to compare series-specific distance values,
    thresholds, and the original series data. It is a valuable tool for visual anomaly detection and model evaluation.
    '''
    # Create a subplot with 2 rows and 1 column
    fig, axs = plt.subplots(2, 1, figsize=(20, 10))
    
    if train_split_idx is not None:
        index_value = train_split_idx
    elif train_split is not None:
        index_position = int(dataset.shape[0] * train_split)
        index_value = dataset.index[index_position]

    # Plot the first subplot (log plot with threshold line)
    axs[0].set_yscale('log')
    X_scores[serie].plot(ax=axs[0], color='blue', alpha=0.5, label=serie+' Dist')
    if threshold_POM is not None:
        axs[0].axhline(threshold_POM, color='red', label='Threshold_POM')
    if threshold_POS is not None:
        axs[0].axhline(threshold_POS, color='green', label='Threshold_POS')
    axs[0].set_ylim([1e-4, 1e1])
    if (train_split is not None) or (train_split_idx is not None):
        axs[0].axvline(index_value, color='black', linestyle='dashed', label='Train/Test Split')
    axs[0].legend()

    # Plot the second subplot (regular plot)
    dataset[serie].plot(ax=axs[1], color='green', alpha=0.5, label=serie+' Original Values')
    if (train_split is not None) or (train_split_idx is not None):
        axs[1].axvline(index_value, color='black', linestyle='dashed', label='Train/Test Split')
    axs[1].legend()

    # Display the plots
    plt.tight_layout()
    plt.show()

def plot_lstm_ae_ind_series_dist_threshold_orig_val_v2(X_scores, dataset, serie='default', threshold_POM=None, threshold_POS=None, train_split=None, train_split_idx=None):
    '''
    The function generates an interactive Plotly figure with two subplots. The top subplot displays the individual series of log-scaled distance
    values along with horizontal dashed lines indicating the threshold values. The bottom subplot displays the original time series data for
    comparison. Vertical dashed lines are used to indicate the split between training and test data if provided.
    This function is valuable for evaluating and exploring the performance of an anomaly detection model on a specific time series. The interactive
    Plotly plot allows users to zoom in, pan, and explore the relationship between anomaly scores and the original series data. It is particularly
    useful when you want to visualize anomalies within the context of the original data and threshold values.
    '''
    fig = go.Figure()

    if train_split_idx is not None:
        index_value = train_split_idx
    elif train_split is not None:
        index_position = int(dataset.shape[0] * train_split)
        index_value = dataset.index[index_position]

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.065, shared_xaxes=True)

    fig.add_trace(go.Scatter(x=list(X_scores.index), y=list(X_scores[serie]), opacity=0.7, name=serie + ' Dist'), 1, 1)

    if threshold_POM is not None:
        fig.add_shape(go.layout.Shape(type="line", x0=X_scores.index[0], x1=X_scores.index[-1], y0=threshold_POM, y1=threshold_POM, line=dict(color='red'), name='Threshold_POM'), row=1, col=1)
    if threshold_POS is not None:
        fig.add_shape(go.layout.Shape(type="line", x0=X_scores.index[0], x1=X_scores.index[-1], y0=threshold_POS, y1=threshold_POS, line=dict(color='green'), name='Threshold_POS'), row=1, col=1)
    if (train_split is not None) or (train_split_idx is not None):
        fig.add_shape(go.layout.Shape(type="line", x0=index_value, x1=index_value, y0=np.min(X_scores[serie]), y1=np.max(X_scores[serie]), opacity=0.7, line=dict(color='black', dash='dash'), name='Train/Test Split'), row=1, col=1)
        fig.add_shape(go.layout.Shape(type="line", x0=index_value, x1=index_value, y0=np.min(dataset[serie]), y1=np.max(dataset[serie]), opacity=0.7, line=dict(color='black', dash='dash'), name='Train/Test Split'), row=2, col=1)

    fig.add_trace(go.Scatter(x=list(dataset.index), y=list(dataset[serie]), opacity=0.7, name=serie + ' Original Values'), 2, 1)
    fig.update_yaxes(type="log", row=1, col=1)

    fig.update_layout(title_text="Dist vs Thresholds", height=800)
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1,
                        label="1d",
                        step="day",
                        stepmode="backward"),
                    dict(count=7,
                        label="7d",
                        step="day",
                        stepmode="backward"),
                    dict(count=14,
                        label="14d",
                        step="day",
                        stepmode="backward"),
                    dict(count=1,
                        label="1m",
                        step="month",
                        stepmode="backward"),
                    dict(count=3,
                        label="3m",
                        step="month",
                        stepmode="backward"),
                    dict(count=6,
                        label="6m",
                        step="month",
                        stepmode="backward"),
                    dict(count=1,
                        label="1y",
                        step="year",
                        stepmode="backward"),
                    dict(step="all")
                ]),
            type="date"),
        
        xaxis2_rangeslider_visible=True,
        xaxis2_rangeslider_thickness=0.1,
        xaxis2_type="date"
        );
    fig.show()

def plot_lstm_ae_dataset_avg_dist_vs_threshold(X_scores, train_split=None, train_split_idx=None):
    '''
    The function generates a line plot showing the average distance values and threshold values for multiple series in a dataset.
    Each series is assigned a distinct color for easy identification. The plot includes a vertical dashed line to mark the split
    between training and test data if provided.
    This function is valuable when you want to examine how average distance values for various series compare to specific threshold
    values over time. It is particularly useful for assessing the performance of an anomaly detection model across different series
    and for identifying patterns or deviations in average distances with respect to their respective thresholds. The visualization
    helps in understanding how the model's performance varies over time and across different series within a dataset.
    '''

    if train_split_idx is not None:
        index_value = train_split_idx
    elif train_split is not None:
        index_position = int(dataset.shape[0] * train_split)
        index_value = dataset.index[index_position]

    # Plot Train's LSTM_AE_model Avg. Dist against the threshold
    dataset = X_scores.copy()
    dist_col = [col for col in dataset.columns if 'dist' in col.lower()]
    dist_col.append('Avg_Threshold_POM')
    dist_col.append('Avg_Threshold_POS')

    # Separate the colors for the series
    colors = ['blue', 'green', 'red']

    # Plot each series with different alpha values
    for i, col in enumerate(dist_col):
        if 'Dist' in col:
            dataset[col].plot(logy=True, figsize=(20, 6), color=colors[i], alpha=0.5)
        else:
            dataset[col].plot(logy=True, figsize=(20, 6), color=colors[i])

    if train_split is not None: plt.axvline(index_value, color='black', linestyle='dashed', label='Train/Test Split')
    plt.title('Avg. Dist vs Avg. Thresholds', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Avg. Dist')
    plt.grid(True)
    plt.show()

def plot_step_01_serie(df1, df2, df3, serie='default', start=None, end=None):
    '''
    The function generates a multi-subplot visualization, with each subplot representing a different stage of data processing or
    transformation for a specific series. This allows you to visually compare how the data evolves across these stages.
    This function is useful when you want to visually inspect how a series of data changes as it goes through various processing steps.
    For instance, you might want to compare the original data to data with outliers removed and then to smoothed data. This can help
    in assessing the impact of each processing step and identifying how the data is modified during the process. It's a useful tool for
    quality control and understanding data preprocessing effects.
    '''
    if start is None: start=0
    if end is None: end=len(df1)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 8))
    ax1.plot(df1[serie][start:end], label='Original Serie '+serie)
    ax2.plot(df2[serie][start:end], label='Serie '+serie+' without extreme outliers')
    ax3.plot(df3[serie][start:end], label='Smoothed Serie'+serie)

def plot_lstm_ae_all_dataset_stages(X_orig, X_wo_ext_out, X_smoothed, serie='default', train_split=None, train_split_idx=None, limit_y_to_smoothed = 'no'):
    '''
    The function generates a multi-subplot visualization, where each subplot represents a different stage of data processing for a specific time
    series. You can compare how the data evolves across these stages, including the impact of removing extreme outliers and smoothing.
    The training/testing split point is also shown.
    This function is beneficial when you want to assess the effects of data preprocessing, such as removing outliers and smoothing, on a
    time series dataset. It allows you to compare different versions of the data, and it's particularly helpful for understanding how these
    preprocessing steps influence the data's distribution and behavior. The training/testing split line is valuable when you want to align this
    information with a specific time frame.
    '''
    fig = go.Figure()

    if train_split_idx is not None:
        index_value = train_split_idx
    elif train_split is not None:
        index_position = int(X_orig.shape[0] * train_split)
        index_value = X_orig.index[index_position]

    fig = make_subplots(rows=3, cols=1, vertical_spacing=0.065, shared_xaxes=True)

    fig.add_trace(go.Scatter(x=list(X_orig.index), y=list(X_orig[serie]), opacity=0.7, name=serie + ' org'), 1, 1)
    fig.add_trace(go.Scatter(x=list(X_wo_ext_out.index), y=list(X_wo_ext_out[serie]), opacity=0.7, name=serie + ' wo_ext_out'), 2, 1)
    fig.add_trace(go.Scatter(x=list(X_smoothed.index), y=list(X_smoothed[serie]), opacity=0.7, name=serie + ' smoothed'), 3, 1)
    if (train_split is not None) or (train_split_idx is not None):
        fig.add_shape(go.layout.Shape(type="line", x0=index_value, x1=index_value, y0=np.min(X_orig[serie]), y1=np.max(X_orig[serie])*1.2, opacity=0.7, line=dict(color='black', dash='dash'), name='Train/Test Split'), row=1, col=1)
        fig.add_shape(go.layout.Shape(type="line", x0=index_value, x1=index_value, y0=np.min(X_wo_ext_out[serie]), y1=np.max(X_wo_ext_out[serie])*1.2, opacity=0.7, line=dict(color='black', dash='dash'), name='Train/Test Split'), row=2, col=1)
        fig.add_shape(go.layout.Shape(type="line", x0=index_value, x1=index_value, y0=np.min(X_smoothed[serie]), y1=np.max(X_smoothed[serie])*1.2, opacity=0.7, line=dict(color='black', dash='dash'), name='Train/Test Split'), row=3, col=1)

    if limit_y_to_smoothed == 'yes':
        fig.update_yaxes(range=[np.min(X_smoothed[serie]), np.max(X_smoothed[serie])*1.2], row=1, col=1)
        fig.update_yaxes(range=[np.min(X_smoothed[serie]), np.max(X_smoothed[serie])*1.2], row=2, col=1)
        fig.update_yaxes(range=[np.min(X_smoothed[serie]), np.max(X_smoothed[serie])*1.2], row=3, col=1)

    fig.update_layout(title_text="Time Series Comparison", height=800)
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1,
                        label="1d",
                        step="day",
                        stepmode="backward"),
                    dict(count=7,
                        label="7d",
                        step="day",
                        stepmode="backward"),
                    dict(count=14,
                        label="14d",
                        step="day",
                        stepmode="backward"),
                    dict(count=1,
                        label="1m",
                        step="month",
                        stepmode="backward"),
                    dict(count=3,
                        label="3m",
                        step="month",
                        stepmode="backward"),
                    dict(count=6,
                        label="6m",
                        step="month",
                        stepmode="backward"),
                    dict(count=1,
                        label="1y",
                        step="year",
                        stepmode="backward"),
                    dict(step="all")
                ]),
            type="date"),
        
        xaxis3_rangeslider_visible=True,
        xaxis3_rangeslider_thickness=0.1,
        xaxis3_type="date"
        );
    fig.show()

def plot_lstm_ae_all_dataset_up_to_six_series(dataset, list_of_series=['default'], timestamp=None, samples_before=288, samples_after=288, train_split=None, train_split_idx=None):
    '''
    The function generates a multi-subplot visualization, with each subplot representing one of the selected time series. You can compare the
    behavior of these series over time, focusing on a specified timestamp and considering the training/testing split if needed.
    This function is helpful when you want to examine and compare multiple time series within a dataset, particularly when you want to focus on
    specific time points or events. It's a valuable tool for understanding how different series behave in relation to each other and important
    timestamps.
    '''
    fig = go.Figure()

    if len(list_of_series) <=6:
        if timestamp is not None:
            if train_split_idx is not None:
                index_value = train_split_idx
            elif train_split is not None:
                index_position = int(dataset.shape[0] * train_split)
                index_value = dataset.index[index_position]

            # Find the index of the provided timestamp
            index = dataset.index.get_loc(timestamp)
            
            # Calculate the start and end indices for the plot
            start_index = max(0, index - samples_before)
            end_index = min(len(dataset), index + samples_after + 1)
            
            # Extract the data for the plot
            plot_data = dataset.iloc[start_index:end_index]
            
            # Add a line trace for each column
            num_of_fig=len(list_of_series)
            fig = make_subplots(rows=num_of_fig, cols=1, vertical_spacing=0.065, shared_xaxes=True)

            fig_row=0
            for column in list_of_series:
                fig_row+=1
                min_val = plot_data[column].min()
                max_val = plot_data[column].max()
                fig.add_trace(go.Scatter(x=list(plot_data.index), y=list(plot_data[column]), opacity=0.7, name=column), fig_row, 1)
                fig.add_shape(go.layout.Shape(type="line", x0=timestamp, x1=timestamp, y0=min_val, y1=max_val, line=dict(color='black', dash='dash')), row=fig_row, col=1)
                if (train_split is not None) or (train_split_idx is not None):
                    fig.add_shape(go.layout.Shape(type="line", x0=index_value, x1=index_value, y0=min_val, y1=max_val*1.2, opacity=0.7, line=dict(color='black', dash='dash'), name='Train/Test Split'), row=fig_row, col=1)
        else:
            if train_split_idx is not None:
                index_value = train_split_idx
            elif train_split is not None:
                index_position = int(dataset.shape[0] * train_split)
                index_value = dataset.index[index_position]

            num_of_fig=len(list_of_series)
            fig = make_subplots(rows=num_of_fig, cols=1, vertical_spacing=0.065, shared_xaxes=True)

            fig_row=0
            for column in list_of_series:
                fig_row+=1
                min_val = dataset[column].min()
                max_val = dataset[column].max()
                fig.add_trace(go.Scatter(x=list(dataset.index), y=list(dataset[column]), opacity=0.7, name=column), fig_row, 1)
                if (train_split is not None) or (train_split_idx is not None):
                    fig.add_shape(go.layout.Shape(type="line", x0=index_value, x1=index_value, y0=min_val, y1=max_val*1.2, opacity=0.7, line=dict(color='black', dash='dash'), name='Train/Test Split'), row=fig_row, col=1)

        fig.update_layout(title_text="Time Series Comparison", height=800)
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=[
                        dict(count=1,
                            label="1d",
                            step="day",
                            stepmode="backward"),
                        dict(count=7,
                            label="7d",
                            step="day",
                            stepmode="backward"),
                        dict(count=14,
                            label="14d",
                            step="day",
                            stepmode="backward"),
                        dict(count=1,
                            label="1m",
                            step="month",
                            stepmode="backward"),
                        dict(count=3,
                            label="3m",
                            step="month",
                            stepmode="backward"),
                        dict(count=6,
                            label="6m",
                            step="month",
                            stepmode="backward"),
                        dict(count=1,
                            label="1y",
                            step="year",
                            stepmode="backward"),
                        dict(step="all")
                    ]),
                type="date"),
            #xaxis_rangeslider_visible=True,
            #xaxis_rangeslider_thickness=0.05,
            xaxis_type="date"
            );
        fig.show()
    else:
        print('The function only plots up to six series at once')

def plot_lstm_ae_samples_around_timestamp(dataset, timestamp, samples_before=288, samples_after=288, series=None):
    '''
    The function generates an interactive Plotly figure with line plots for the selected series, centered around the specified timestamp.
    You can zoom in and out, pan through the data, and examine the behavior of the selected series around the timestamp.
    This function is useful when you want to visualize and analyze the behavior of specific time series data around a particular
    timestamp. It helps you focus on relevant data points and understand how the series behave in proximity to the chosen timestamp.
    '''
    # Find the index of the provided timestamp
    index = dataset.index.get_loc(timestamp)
    
    # Calculate the start and end indices for the plot
    start_index = max(0, index - samples_before)
    end_index = min(len(dataset), index + samples_after + 1)
    
    # Extract the data for the plot
    plot_data = dataset.iloc[start_index:end_index]
    
    # If columns are not provided, use all columns from the DataFrame
    if series is None:
        series = dataset.columns

    # Create the Plotly figure
    fig = go.Figure()
    
    # Add a line trace for each column
    for column in series:
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data[column], mode='lines', name=column))

    def find_min_max_values(data, series):
        min_values = data[series].min()
        max_values = data[series].max()
        return min_values, max_values
    min_vals, max_vals = find_min_max_values(data=dataset,series=series)
    min_val = np.min(min_vals)
    max_val = np.max(max_vals)

    # Add vertical line at the timestamp
    fig.add_shape(
        go.layout.Shape(
            type="line",
            x0=timestamp,
            x1=timestamp,
            y0=min_val,
            y1=max_val,
            line=dict(color='black', dash='dash')
        )
    )
    
    # Update layout for rangeslider
    fig.update_layout(
        title='Data Around ' + str(timestamp),
        xaxis_title='Timestamp',
        yaxis_title='Value',
        xaxis_rangeslider_visible=True,
        showlegend=True
    )
    
    # Show the plot
    fig.show()



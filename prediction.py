import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sqlite3
import os
import numpy as np
#from models.rf import Rf
#from models.svm import Svm

scaler = MinMaxScaler()

last_prediction = None
last_prediction_float = None

# Define function to make predictions on new candle data
def make_prediction(models, df):
    global last_prediction, last_prediction_float

    # Split the data into training and testing sets
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size].copy()
    test_df = df[train_size:].copy()

    # Normalize the values
    scaler = MinMaxScaler()
    train_df.loc[:, ['open', 'high', 'low', 'close', 'volume']] = scaler.fit_transform(train_df.loc[:, ['open', 'high', 'low', 'close', 'volume']])

    test_df.loc[:, ['open', 'high', 'low', 'close', 'volume']] = scaler.transform(test_df.loc[:, ['open', 'high', 'low', 'close', 'volume']])

    # Define input and output data
    X_train = train_df[['open_time', 'open', 'high', 'low', 'close', 'volume']].values
    y_train = train_df['close'].values

    # print("X_train length", len(X_train))
    # print("y_train length", len(y_train))

    X_test = test_df[['open_time', 'open', 'high', 'low', 'close', 'volume']].values
    y_test = test_df['close'].values

    # Train all models with updated data
    # print("training all models with updated data")
    for i, model in enumerate(models):
        try:
            # print("y_train length", len(y_train))
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"Error in make_prediction(): {e}")

    # Evaluate all models
    for i, model in enumerate(models):
        filename = model.__module__.split('.')[-1] + '.py'
        mse = model.evaluate(X_test, y_test)
        print(f'{filename} mean squared error: {mse}')

    # Make predictions using all models
    signals = []
    # print("making predictions using all models")
    for i, model in enumerate(models):
        # Get the last sequence of data from the DataFrame
        data = model.get_last_sequence(df)

        # Reshape the data to have the shape (1, self.sequence_length, 1)
        data = np.reshape(data, (1, model.n_timesteps, 1))

        # Make the prediction using the model
        prediction = model.predict_point_by_point(data)

        filename = model.__module__.split('.')[-1] + '.py'
        # print(filename, 'Prediction:', prediction[0])
        # threshold = 0.05  # 1% threshold

        if prediction[0] > 0.2:
            signals.append('Buy')
            print('Buy signal', prediction[0], 'last prediction was: ' + str('not available yet' if last_prediction is None else last_prediction + (", " +last_prediction_float)) + ", last close was: " + str(df['close'].iloc[-2]) + ", this close is: " + str(df['close'].iloc[-1]))
            last_prediction = 'Buy'
        elif prediction[0] < -0.2:
            signals.append('Sell')
            print('Sell signal', prediction[0], 'last prediction was: ' + str('not available yet' if last_prediction is None else last_prediction + (", " +last_prediction_float)) + ", last close was: " + str(df['close'].iloc[-2]) + ", this close is: " + str(df['close'].iloc[-1]))
            last_prediction = 'Sell'
        else:
            signals.append('Hold')
            print('Hold signal', prediction[0], 'last prediction was: ' + str('not available yet' if last_prediction is None else last_prediction + (", " +last_prediction_float)) + ", last close was: " + str(df['close'].iloc[-2]) + ", this close is: " + str(df['close'].iloc[-1]))
            last_prediction = 'Hold'
        last_prediction_float = prediction[0]
    '''
    # Print signals for each model
    for i, model in enumerate(models):
        filename = model.__module__.split('.')[-1] + '.py'
        print(filename, 'Signal:', signals[i])

    # Print signals for all models together
    if all(signal == 'Buy' for signal in signals):
        print('All models: Buy')
    elif all(signal == 'Sell' for signal in signals):
        print('All models: Sell')
    else:
        print('All models: Hold')'''
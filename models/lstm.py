import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from models.base_model import BaseModel
from numpy import newaxis

class Lstm(BaseModel):
    def __init__(self, input_shape, n_timesteps, checkpoint_path):
        super().__init__(input_shape, n_timesteps, checkpoint_path)
        self.checkpoint_path = checkpoint_path
        self.model = self.create_model()

    def create_model(self):
        # Create a new instance of the LSTM model
        model = Sequential()
        model.add(LSTM(50, input_shape=self.input_shape))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def get_last_sequence(self, df):
        # Extract the last self.n_timesteps timesteps of data from the 'close' column of the df DataFrame
        data = df['close'].iloc[-self.n_timesteps:].values.reshape(-1, 1)
        # Normalize the data
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        return data

    def fit(self, X_train, y_train):
        # values = values.reshape((len(values), 1))
        # Convert the data to a numpy array
        X_train = np.array(X_train)

        # Reshape the data
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1 ))

        checkpoint = ModelCheckpoint(self.checkpoint_path, monitor='loss', verbose=0, save_best_only=True, mode='min')
        self.model.fit(X_train, y_train, epochs=5, batch_size=32, callbacks=[checkpoint], verbose=0)
        self.save()

    # @tf.function(experimental_reduce_retracing_warnings=True)
    def predict_point_by_point(self, data):
        #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        #print('[Model] Predicting Point-by-Point...')
        predicted = self.model.predict(data, verbose=0)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_sequences_multiple(self, data, window_size, prediction_len):
        #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(len(data)/prediction_len)):
            curr_frame = data[i*prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs

    def predict_sequence_full(self, data, window_size):
        #Shift the window by 1 new prediction each time, re-run predictions on new window
        print('[Model] Predicting Sequences Full...')
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
        return predicted

    def evaluate(self, X_test, y_test):
        if not self.loaded:
            self.load()
        
        # Convert the data to a numpy array
        X_test = np.array(X_test)

        # Reshape the data
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1 ))

        y_pred = self.predict_point_by_point(X_test)
        if len(y_pred) == 0:
            return None
        mse = mean_squared_error(y_test, y_pred)
        return mse
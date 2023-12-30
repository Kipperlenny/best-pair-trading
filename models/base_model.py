import os
import pickle
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import numpy as np

class BaseModel(ABC):
    def __init__(self, input_shape, n_timesteps, checkpoint_path):
        self.model = None
        self.scaler = MinMaxScaler()
        self.loaded = False
        self.input_shape = input_shape
        self.n_timesteps = n_timesteps

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

    @abstractmethod
    def predict_point_by_point(self, X):
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test):
        pass

    def predict_classes(self, X):
        if not self.loaded:
            self.load()
        X = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y_pred = self.model.predict(X)
        y_pred_classes = np.round(y_pred).astype(int)
        return y_pred_classes

    def load(self):
        # Load the trained model from disk
        #filename = 'data/' + self.__class__.__name__.lower() + '.pkl'
        #if os.path.exists(filename):
        #    try:
        #        print('loading model from disk')
        #        with open(filename, 'rb') as f:
        #            self.model = pickle.load(f)
        #        self.loaded = True
        #    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError) as e:
        #        print(f"Error loading model from file: {e}")
        #        self.model = None
        #else:
            self.model = self.create_model()
        #    self.loaded = False

    def save(self):
        # Save the trained model to disk
        # print('saving model to disk')
        filename = 'data/' + self.__class__.__name__.lower() + '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    def create_model(self):
        # Create a new instance of the model
        model = None  # Replace None with code that creates a new instance of the model
        return model
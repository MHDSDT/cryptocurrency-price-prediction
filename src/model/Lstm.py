import tensorflow
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM

from src.model.LstmSpecs import LstmSpecs


class Lstm:
    def __init__(self, df):
        self.df = df
        self.model = None
        self.history = None
        self.targets = None
        self.predictions = None
        self.set_lstm_specs()

    def __setitem__(self, key, value):
        setattr(self, key, value)

    @staticmethod
    def normalize_zero_base(df):
        return df / df.iloc[0] - 1

    @staticmethod
    def normalize_min_max(df):
        return (df - df.min()) / (df.max() - df.min())

    @staticmethod
    def extract_window_data(df, window_len, zero_base):
        window_data = []
        for idx in range(len(df) - window_len):
            window_slice = df[idx: (idx + window_len)].copy()
            if zero_base:
                window_slice = Lstm.normalize_zero_base(window_slice)
            else:
                window_slice = Lstm.normalize_min_max(window_slice)
            window_data.append(window_slice.values)
        return np.array(window_data)

    def set_lstm_specs(self):
        for spec in LstmSpecs:
            self[spec.name] = spec.value

    def train_test_split(self, test_size=0.2):
        split_row = len(self.df) - int(test_size * len(self.df))
        train_data = self.df.iloc[:split_row]
        test_data = self.df.iloc[split_row:]
        return train_data, test_data

    def prepare_data(self):
        train_data, test_data = self.train_test_split(test_size=self.test_size)
        x_train = self.extract_window_data(train_data, self.window_len, self.zero_base)
        x_test = self.extract_window_data(test_data, self.window_len, self.zero_base)
        y_train = train_data[self.target_col][self.window_len:].values
        y_test = test_data[self.target_col][self.window_len:].values
        if self.zero_base:
            y_train = y_train / train_data[self.target_col][:-self.window_len].values - 1
            y_test = y_test / test_data[self.target_col][:-self.window_len].values - 1
        return x_train, x_test, y_train, y_test

    def build_lstm_model(self, input_data, output_size):
        self.model = Sequential()
        self.model.add(LSTM(self.lstm_neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=output_size))
        self.model.add(Activation(self.activ_func))
        self.model.compile(loss=self.loss, optimizer=self.optimizer, run_eagerly=True)

    def train(self):
        x_train, x_test, y_train, y_test = self.prepare_data()
        self.build_lstm_model(input_data=x_train, output_size=1)
        self.history = self.model.fit(
            x_train, y_train, validation_data=(x_test, y_test), epochs=self.epochs, batch_size=self.batch_size,
            verbose=0, shuffle=True)

    def predict(self):
        train, test = self.train_test_split(test_size=self.test_size)
        x_train, x_test, y_train, y_test = self.prepare_data()
        self.targets = test[self.target_col][self.window_len:]
        self.predictions = self.model.predict(x_test).squeeze()
        self.predictions = test[self.target_col].values[:-self.window_len] * (self.predictions + 1)
        self.predictions = pd.Series(index=self.targets.index, data=self.predictions)

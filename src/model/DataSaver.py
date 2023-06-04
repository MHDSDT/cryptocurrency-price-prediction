import os

import pandas as pd
from keras.models import load_model


class DataSaver:
    def __init__(self, market):
        self.market = market.lower()
        self.data_dir = '../data'

    def save_data(self, data):
        data.to_csv(os.path.join(self.data_dir, f'data_{self.market}.csv'), index=False)

    def load_data(self, ):
        return pd.read_csv(os.path.join(self.data_dir, f'data_{self.market}.csv'))

    def save_model(self, model):
        model.save(os.path.join(self.data_dir, f'model_{self.market}.h5'))

    def load_model(self):
        return load_model(os.path.join(self.data_dir, f'model_{self.market}.h5'))

import tensorflow
import time
import math
import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM

WINDOW_LEN = 5
TEST_SIZE = 0.2
ZERO_BASE = True
LSTM_NEURONS = 100
EPOCHS = 20
BATCH_SIZE = 32
LOSS = 'mse'
DROPOUT = 0.2
OPTIMIZER = 'adam'
TARGET_COL = 'close'


def fetch_data_from_api():
    endpoint = 'https://api.tabdeal.ir/r/plots/history'
    market = 'USDT_IRT'
    now_time = math.floor(time.time())
    one_year_duration = 3600 * 24 * 365
    resolution_in_hour = 3
    url = endpoint + f'?symbol={market}&from={now_time - one_year_duration}&to={now_time}&resolution={resolution_in_hour * 60}'
    res = requests.get(url)
    data = pd.DataFrame(json.loads(res.content)['data'])
    data = data.set_index('time')
    data.index = pd.to_datetime(data.index, unit='s')
    return data


def split_train_and_test_data(df, test_size=0.2):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data


# def get_extended_data(data, test_size=0.2):
#     extended_data = data.copy()
#     for i in range(math.floor(len(extended_data.index) / test_size)):
#         last_index = extended_data.index[-1]
#         new_index = last_index + pd.DateOffset(hours=3)
#         extended_data.loc[new_index] = [None, None, None, None, None]
#     return extended_data


def plot_two_lines(line1, line2, label1=None, label2=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(13, 7))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel('price USDT', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc='best', fontsize=16)
    plt.show()


def normalize_zero_base(df):
    return df / df.iloc[0] - 1


def normalize_min_max(df):
    return (df - df.min()) / (df.max() - df.min())


def extract_window_data(df, window_len=5, zero_base=True):
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx: (idx + window_len)].copy()
        if zero_base:
            tmp = normalize_zero_base(tmp)
        else:
            tmp = normalize_min_max(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)


def prepare_data(df, target_col, window_len=10, zero_base=True, test_size=0.2):
    train_data, test_data = split_train_and_test_data(df, test_size=test_size)
    x_train = extract_window_data(train_data, window_len, zero_base)
    x_test = extract_window_data(test_data, window_len, zero_base)
    y_train = train_data[target_col][window_len:].values
    y_test = test_data[target_col][window_len:].values
    if zero_base:
        y_train = y_train / train_data[target_col][:-window_len].values - 1
        y_test = y_test / test_data[target_col][:-window_len].values - 1

    return x_train, x_test, y_train, y_test


def build_lstm_model(input_data, output_size, neurons=100, activ_func='linear',
                     dropout=0.2, loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer, run_eagerly=True)
    return model


def plot_lstm_graph(history):
    plt.plot(history.history['loss'], 'r', linewidth=2, label='Train loss')
    plt.plot(history.history['val_loss'], 'g', linewidth=2, label='Validation loss')
    plt.title('LSTM')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.show()


def main():
    data = fetch_data_from_api()
    train, test = split_train_and_test_data(data, test_size=TEST_SIZE)
    plot_two_lines(train[TARGET_COL], test[TARGET_COL], 'training', 'test', title='')

    x_train, x_test, y_train, y_test = prepare_data(data, TARGET_COL, window_len=WINDOW_LEN, zero_base=ZERO_BASE,
                                                    test_size=TEST_SIZE)

    model = build_lstm_model(x_train, output_size=1, neurons=LSTM_NEURONS, dropout=DROPOUT, loss=LOSS,
                             optimizer=OPTIMIZER)
    history = model.fit(
        x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
        shuffle=True)

    plot_lstm_graph(history)

    targets = test[TARGET_COL][WINDOW_LEN:]
    predictions = model.predict(x_test).squeeze()
    predictions = test[TARGET_COL].values[:-WINDOW_LEN] * (predictions + 1)
    predictions = pd.Series(index=targets.index, data=predictions)
    plot_two_lines(targets, predictions, 'actual', 'prediction', lw=3)


if __name__ == "__main__":
    main()

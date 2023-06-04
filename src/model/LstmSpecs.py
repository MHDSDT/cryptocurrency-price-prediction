from enum import Enum


class LstmSpecs(Enum):
    zero_base = True
    loss = 'mse'
    optimizer = 'adam'
    target_col = 'close'
    activ_func = 'linear'
    test_size = 0.2
    window_len = 5
    lstm_neurons = 100
    epochs = 20
    batch_size = 32
    dropout = 0.2

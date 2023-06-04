from src.model.DataSaver import DataSaver
from src.model.Lstm import Lstm
from src.model.Visualizer import Visualizer


class CryptoPricePredictor:
    def __init__(self, exchange, market: str):
        self.exchange = exchange
        self.market = market
        self.lstm = None
        self.data_saver = DataSaver(market)
        self.visualizer = Visualizer()

    def predict_crypto_price(self):
        df = self.exchange.fetch_data(self.market)
        self.lstm = Lstm(df)
        self.lstm.train()
        self.lstm.predict()

        self.visualizer.plot_train_val_loss(self.lstm.history)
        self.visualizer.plot_two_lines(self.lstm.targets, self.lstm.predictions, 'actual', 'prediction')



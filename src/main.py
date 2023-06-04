from src.controller.CryptoPricePredictor import CryptoPricePredictor
from src.model.exchange.Tabdeal import Tabdeal


class Main:
    @staticmethod
    def main():
        exchange = Tabdeal(url='https://api.tabdeal.org', endpoint='/r/plots/history')
        market = 'BTC_USDT'
        crypto_price_predictor = CryptoPricePredictor(exchange, market)
        crypto_price_predictor.predict_crypto_price()


if __name__ == "__main__":
    Main.main()

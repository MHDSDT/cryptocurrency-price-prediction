import pandas as pd
import math
import time
import json
import requests
from src.model.exchange.Exchange import Exchange


class Tabdeal(Exchange):
    ONE_YEAR_DURATION = 3600 * 24 * 365

    def __init__(self, url: str, endpoint: str, market_separator='_', resolution_in_hour=3):
        super().__init__(url, endpoint, market_separator)
        self.params = dict()
        self.resolution_in_hour = resolution_in_hour

    @staticmethod
    def get_now_time():
        return math.floor(time.time())

    @staticmethod
    def get_dataframe(data):
        df = pd.DataFrame(data)
        df = df.set_index('time')
        df.index = pd.to_datetime(df.index, unit='s')
        return df

    def get_one_year_ago_time(self):
        return self.params['to'] - self.ONE_YEAR_DURATION

    def set_params_value(self, market):
        self.params['symbol'] = market.replace('_', self.market_separator)
        self.params['to'] = self.get_now_time()
        self.params['from'] = self.get_one_year_ago_time()
        self.params['resolution'] = self.resolution_in_hour * 60

    def fetch_data(self, market):
        self.set_params_value(market)
        res = requests.get(self.url + self.endpoint, params=self.params)
        if res.status_code == 200:
            data = json.loads(res.content)['data']
            return self.get_dataframe(data)
        else:
            print(f"Request failed with status code: {res.status_code}")

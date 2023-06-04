class Exchange:
    def __init__(self, url: str, endpoint: str, market_separator='_'):
        self.url = url.rstrip('/') if url.endswith('/') else url
        self.endpoint = endpoint.rstrip('/') if endpoint.endswith('/') else endpoint
        self.market_separator = market_separator

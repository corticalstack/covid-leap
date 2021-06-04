from kaggle.api.kaggle_api_extended import KaggleApi


class HelperKaggle:
    def __init__(self):
        self.api = KaggleApi()
        self.authenticate()

    def authenticate(self):
        self.api.authenticate()

import time
from datetime import datetime
import hmac
import hashlib
import requests
import json
import configparser

config = configparser.ConfigParser()
config.read('config.ini', encoding='utf-8')

class GMOHandler:
    def __init__(self, config):
        self.apiKey = config.get('Key', 'apiKey')
        self.secretKey = config.get('Key', 'secretKey')
        self.symbol = config.get('Trade', 'symbol')
        self.position_size = float(config.get('Trade', 'position_size'))

    def get_ticker(self):
        endPoint = 'https://api.coin.z.com/public'
        path = '/v1/ticker?symbol=' + self.symbol
        response = requests.get(endPoint + path)

        if response.status_code == 200:
            out = response.json()['data']
        else:
            out = None
        return out

    def get_market_status(self):
        endPoint = 'https://api.coin.z.com/public'
        path = '/v1/status'

        response = requests.get(endPoint + path)
        if response.status_code == 200:
            out = response.json()['data']['status']
        else:
            out = "CLOSE"
        return out

    def get_account_margin(self):
        apiKey = self.apiKey
        secretKey = self.secretKey
        timestamp = '{0}000'.format(int(time.mktime(datetime.now().timetuple())))
        method = 'GET'
        endPoint = 'https://api.coin.z.com/private'
        path = '/v1/account/margin'

        text = timestamp + method + path
        sign = hmac.new(bytes(secretKey.encode('ascii')), bytes(text.encode('ascii')), hashlib.sha256).hexdigest()

        headers = {
            "API-KEY": apiKey,
            "API-TIMESTAMP": timestamp,
            "API-SIGN": sign
        }

        res = requests.get(endPoint + path, headers=headers)
        if res.status_code == 200:
            out = res.json()['data']
        else:
            out = None
        return out

    def get_open_positions(self):
        apiKey = self.apiKey
        secretKey = self.secretKey
        timestamp = '{0}000'.format(int(time.mktime(datetime.now().timetuple())))
        method    = 'GET'
        endPoint  = 'https://api.coin.z.com/private'
        path      = '/v1/openPositions'

        text = timestamp + method + path
        sign = hmac.new(bytes(secretKey.encode('ascii')), bytes(text.encode('ascii')), hashlib.sha256).hexdigest()
        parameters = {
            "symbol": self.symbol,
            "page": 1,
            "count": 100
        }

        headers = {
            "API-KEY": apiKey,
            "API-TIMESTAMP": timestamp,
            "API-SIGN": sign
        }

        res = requests.get(endPoint + path, headers=headers, params=parameters)
        if res.status_code == 200:
            out = res.json()['data']
        else:
            out = None
        return out

    def run_order(self, side, execution_type="MARKET", in_force="FAK"):
        apiKey = self.apiKey
        secretKey = self.secretKey
        timestamp = '{0}000'.format(int(time.mktime(datetime.now().timetuple())))
        method = 'POST'
        endPoint = 'https://api.coin.z.com/private'
        path = '/v1/order'
        reqBody = {
            "symbol": self.symbol,
            "side": side,
            "executionType": execution_type,
            "timeInForce": in_force,
            "size": str(self.position_size)
        }

        text = timestamp + method + path + json.dumps(reqBody)
        sign = hmac.new(bytes(secretKey.encode('ascii')), bytes(text.encode('ascii')), hashlib.sha256).hexdigest()

        headers = {
            "API-KEY": apiKey,
            "API-TIMESTAMP": timestamp,
            "API-SIGN": sign
        }

        res = requests.post(endPoint + path, headers=headers, data=json.dumps(reqBody))

        return res.json()

    def run_close_order(self, side, position_id, execution_type="MARKET", in_force="FAK"):
        apiKey = self.apiKey
        secretKey = self.secretKey
        timestamp = '{0}000'.format(int(time.mktime(datetime.now().timetuple())))
        method = 'POST'
        endPoint = 'https://api.coin.z.com/private'
        path = '/v1/closeOrder'
        reqBody = {
            "symbol": self.symbol,
            "side": side,
            "executionType": execution_type,
            "timeInForce": in_force,
            "settlePosition": [
                {
                    "positionId": position_id,
                    "size": str(self.position_size)
                }
            ]
        }

        text = timestamp + method + path + json.dumps(reqBody)
        sign = hmac.new(bytes(secretKey.encode('ascii')), bytes(text.encode('ascii')), hashlib.sha256).hexdigest()

        headers = {
            "API-KEY": apiKey,
            "API-TIMESTAMP": timestamp,
            "API-SIGN": sign
        }

        res = requests.post(endPoint + path, headers=headers, data=json.dumps(reqBody))

        return res.json()
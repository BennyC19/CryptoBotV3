"""
"""
import time
import requests
import hmac
import hashlib
import json

API_KEY = "DDoUa7VjmJqtQr1rzH6Nte"
SECRET_KEY = "Li1b1VtWjZ5iZZxD5a4q9t"

coinList = ["BTC", "ETH"]
currentPrices = {}
smallestPrecision = {}
precisionDict = {}

"""
def retrieveCurrentPrice():
    for coin in coinList:
        request = requests.get(f'https://api.crypto.com/v2/public/get-ticker?instrument_name={coin}_USDT')
        request = json.loads(request.text)
        currentPrices[coin] = float(request['result']['data'][-1]['a'])

def getCurrentPrice(coinName):
    return currentPrices[coinName]

def getCoinBalance(coinName):
    
    req = {
        "id": 11,
        "method": "private/get-account-summary",
        "api_key": API_KEY,
        "params": {
            "currency": coinName
        },
        "nonce": int(time.time() * 1000)
    }

    paramString = ""
    if "params" in req:
        for key in req['params']:
            paramString += key
            paramString += str(req['params'][key])
    
    sigPayload = req['method'] + str(req['id']) + req['api_key'] + paramString + str(req['nonce'])
    print(sigPayload)
    req['sig'] = hmac.new(
        bytes(SECRET_KEY, 'utf-8'),
        msg=bytes(sigPayload, 'utf-8'),
        digestmod=hashlib.sha256
    ).hexdigest()
    
    coinBalance = requests.post("https://api.crypto.com/v2/private/get-account-summary", json=req, headers={'Content-Type':'application/json'})
    accounts = json.loads(coinBalance.text)['result']['accounts']
    if len(accounts) == 0:
        return 0
    else:
        amount = json.loads(coinBalance.text)['result']['accounts'][0]['available']
        return amount

print(getCoinBalance('BTC'))

def getTotalBalance():
    totalAmount = 0
    for coin in coinList:
        totalAmount += getCoinBalance(coin) * getCurrentPrice(coin)
    return (totalAmount + getCoinBalance("USDT"))


def Order(coin, orderType, amount):
    req = {
        "id": 12,
        "method" : "private/create-order",
        "api_key": API_KEY,
        "params": {
            "instrument_name": f"{coin}_USDT",
            "side": orderType,
            "type": "MARKET",
            "notional": str(amount),
        },
        "nonce": int(time.time() * 1000)
    }
    
    paramString = ""
    if "params" in req:
        for key in req['params']:
            paramString += key
            paramString += str(req['params'][key])
    
    sigPayload = req['method'] + str(req['id']) + req['api_key'] + paramString + str(req['nonce'])

    print(sigPayload)
    req['sig'] = hmac.new(
        bytes(str(SECRET_KEY), 'utf-8'),
        msg=bytes(sigPayload, 'utf-8'),
        digestmod=hashlib.sha256
    ).hexdigest()

    amount = requests.post("https://api.crypto.com/v2/private/create-order", json=req, headers={'Content-Type':'application/json'})
    accounts = json.loads(amount.text)
    print(accounts)    

Order("BTC", "BUY", 0.0002)

import hmac
import hashlib
import time

req = {
    "id": 14,
    "method": "private/create-order-list",
    "api_key": API_KEY,
    "params": {
        "contingency_type": "LIST",
        "order_list": [
            {
                "instrument_name": "BTC_USDT",
                "side": "BUY",
                "type": "MARKET",
                "quantity": "0.00005"
            }
        ]
    },
    "nonce": int(time.time() * 1000)
}

# First ensure the params are alphabetically sorted by key
param_str = ""

MAX_LEVEL = 3

def params_to_str(obj, level):
    if level >= MAX_LEVEL:
        return str(obj)

    return_str = ""
    for key in sorted(obj):
        return_str += key
        if obj[key] is None:
            return_str += 'null'
        elif isinstance(obj[key], list):
            for subObj in obj[key]:
                return_str += params_to_str(subObj, ++level)
        else:
            return_str += str(obj[key])
    return return_str

if "params" in req:
    param_str = params_to_str(req['params'], 0)

payload_str = req['method'] + str(req['id']) + req['api_key'] + param_str + str(req['nonce'])

req['sig'] = hmac.new(
    bytes(str(SECRET_KEY), 'utf-8'),
    msg=bytes(payload_str, 'utf-8'),
    digestmod=hashlib.sha256
).hexdigest()

amount = requests.post("https://api.crypto.com/v2/private/create-order-list", json=req, headers={'Content-Type':'application/json'})
accounts = json.loads(amount.text) 
print(accounts)

import decimal
import math

def precision():
    for coin in coinList:
        request = requests.get(f'https://api.crypto.com/v2/public/get-trades?instrument_name={coin}_USDT')
        request = json.loads(request.text)

        d = decimal.Decimal(str(request['result']['data'][0]['q']))
        precisionDict[coin] =  ((1 / (10 ** (d.as_tuple().exponent * -1))))

precision()

print(getCoinBalance("BTC"))

amount = getCoinBalance("BTC")

print(precisionDict['BTC'])

print(str(getCoinBalance("BTC"))[:int(math.log10(precisionDict['BTC']) * -1) + 2])


print(format(0.00001357, '.8f'))
"""

request = requests.get(f'https://api.crypto.com/v2/public/get-instruments')
request = json.loads(request.text)

for instrument in request['result']['instruments']:
    for coin in coinList:
        if instrument['instrument_name'] == f'{coin}_USDT':
            print(type(instrument['min_quantity']))

import time
from time import sleep
import pandas as pd
import requests
import threading
from datetime import datetime, timedelta
import json
import csv
from csv import writer
import os
import sys
import numpy as numpy
from collections import deque
from PriceHistoryRetrieval import getPricesForTraining, getPricesForTrainingNewCoin, updatePriceList
from NeuralNetwork import winningAgents
import torch
import hmac
import hashlib
import math

sys.setrecursionlimit(10**9)
threading.stack_size(10**8)

# Global Variables
coinList = []

currentPrices = {}

queuesFor14mRSI = {}
queuesFor14hRSI = {}
queuesFor14dRSI = {}

price12dQueue = {}
price26dQueue = {}
currentPriceVSCurrentEMA12d = {}
currentPriceVSCurrentEMA26d = {}

currentRSIDictionary = {}

availableTetherForCoin = {}
smallestPrecision = {}
orderList = []

api_key = ""
api_secret = ""

def retrieveCurrentPrice():
    for coin in coinList:
        request = requests.get(f'https://api.crypto.com/v2/public/get-ticker?instrument_name={coin}_USDT')
        request = json.loads(request.text)
        currentPrices[coin] = float(request['result']['data'][-1]['a'])

    while True:
        for coin in coinList:
            while True:
                try:
                    request = requests.get(f'https://api.crypto.com/v2/public/get-ticker?instrument_name={coin}_USDT')
                    request = json.loads(request.text)
                    currentPrices[coin] = float(request['result']['data'][-1]['a'])
                except:
                    sleep(1)
                    continue
                break
        sleep(1)

def getCurrentPrice(coinName):
    return currentPrices[coinName]

def addCoinPrice(coin):
    request = requests.get(f'https://api.crypto.com/v2/public/get-ticker?instrument_name={coin}_USDT')
    request = json.loads(request.text)
    currentPrices[coin] = float(request['result']['data'][-1]['a'])

def ReadableTimeToTimeStamp(readableTime):
        s = datetime.strptime(readableTime, "%Y-%m-%d %H:%M:%S.%f")
        timestamp = datetime.timestamp(s)
        timestamp = int(timestamp * 1000)
        return timestamp

def TimeStampToReadableTime(timeStamp):
    readableTime = datetime.fromtimestamp(timeStamp/1000)
    return readableTime

def verifyCoin(coin):
    timeNow = datetime.now()
    startTime = timeNow - timedelta(minutes=1)
    endTime = timeNow
    start = startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
    end = endTime.strftime("%Y-%m-%d %H:%M:%S.%f")
    start = ReadableTimeToTimeStamp(start)
    end = ReadableTimeToTimeStamp(end)
    try:
        binanceUrl = f'https://fapi.binance.com/fapi/v1/klines?symbol={coin}USDT&interval=1m&startTime={start}&endTime={end}'
        data = requests.get(binanceUrl).json()
    except:
        print("Exiting...")
        print("Internet may be unstable")
        os._exit(1) 

    if isinstance(data, dict):
        return False
    else:
        return True

def getCoinPrecision():
    request = requests.get(f'https://api.crypto.com/v2/public/get-instruments')
    request = json.loads(request.text)
    for instrument in request['result']['instruments']:
        for coin in coinList:
            if instrument['instrument_name'] == f'{coin}_USDT':
                smallestPrecision[coin] = float(instrument['min_quantity'])

def getCoinBalance(coinName):
    
    req = {
        "id": 11,
        "method": "private/get-account-summary",
        "api_key": api_key,
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
    
    req['sig'] = hmac.new(
        bytes(api_secret, 'utf-8'),
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

def getAvailableBalance():
    return getCoinBalance("USDT")

def getTotalBalance():
    totalAmount = 0
    for coin in coinList:
        totalAmount += getCoinBalance(coin) * getCurrentPrice(coin)
    return totalAmount

"""
def addCoin(coin):
    if (verifyCoin(coin) == True):
        # sequence to do before adding the coin or else there will be errors
        if coin in coinList:
            print("This coin is already in the list")
        else:
            print("adding coin...")
            addID(coin)
            addCoinPrice(coin)
            addMACDQueue(coin)
            addRSIQueue(coin)
            getPricesForTrainingNewCoin(coin)
            coinList.append(coin)
            print("coin successfully added")
    else:
        print("Invalid Coin")

def deleteCoin(coin):
    try:
        coinList.remove(coin)
    except:
        print("Invalid Coin")
"""

def params_to_str(obj, level):
    MAX_LEVEL = 3
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

def CompleteOrder(orderList):
    req = {
        "id": 14,
        "method": "private/create-order-list",
        "api_key": api_key,
        "params": {
            "contingency_type": "LIST",
            "order_list": orderList
        },
        "nonce": int(time.time() * 1000)    
    }

    param_str = ""
    MAX_LEVEL = 3

    if "params" in req:
        param_str = params_to_str(req['params'], 0)

    payload_str = req['method'] + str(req['id']) + req['api_key'] + param_str + str(req['nonce'])

    req['sig'] = hmac.new(
        bytes(str(api_secret), 'utf-8'),
        msg=bytes(payload_str, 'utf-8'),
        digestmod=hashlib.sha256
    ).hexdigest()

    amount = requests.post("https://api.crypto.com/v2/private/create-order-list", json=req, headers={'Content-Type':'application/json'})

def Commands():
    while True:
        userInput = input()
        if userInput == 'HARDSTOP':
            print("Exiting...")
            os._exit(1) 
        elif userInput == 'total balance':
            print(getTotalBalance() + getAvailableBalance())
        elif userInput == 'delete coin':
            userInput = input("Which Coin?\n")
            deleteCoin(userInput)
        elif userInput == 'add coin':
            userInput = input("Which Coin?\n")
            addCoin(userInput)
        elif userInput == 'all coins':
            print(coinList)
        else:
            print("invalid command")       

def addMACDQueue(coin):
    price12dQueue[coin] = deque()
    price26dQueue[coin] = deque()
    timeNow = datetime.now()
    startTime = timeNow - timedelta(days = 26)
    endTime = timeNow
    start = startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
    end = endTime.strftime("%Y-%m-%d %H:%M:%S.%f")
    start = ReadableTimeToTimeStamp(start)
    end = ReadableTimeToTimeStamp(end)
    binanceUrl = f'https://fapi.binance.com/fapi/v1/klines?symbol={coin}USDT&interval=1d&startTime={start}&endTime={end}'
    data = requests.get(binanceUrl).json()

    priceList = []
    for row in data:
        priceList.append(float(row[4]))
    
    priceArray = numpy.array(priceList)
    
    for price in priceArray[14:]:
        price12dQueue[coin].append(price)

    for price in priceArray:
        price26dQueue[coin].append(price)

def addRSIQueue(coin):
    # For 14m RSI 
    queuesFor14mRSI[coin] = deque()
    timeNow = datetime.now()
    startTime = timeNow - timedelta(minutes = 210)
    endTime = timeNow
    start = startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
    end = endTime.strftime("%Y-%m-%d %H:%M:%S.%f")
    start = ReadableTimeToTimeStamp(start)
    end = ReadableTimeToTimeStamp(end)
    binanceUrl = f'https://fapi.binance.com/fapi/v1/klines?symbol={coin}USDT&interval=15m&startTime={start}&endTime={end}'
    data = requests.get(binanceUrl).json()
    
    for row in data:
        queuesFor14mRSI[coin].append(float(row[4]))

    # For 14h RSI
    queuesFor14hRSI[coin] = deque()
    startTime = timeNow - timedelta(hours = 14)
    endTime = timeNow
    start = startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
    end = endTime.strftime("%Y-%m-%d %H:%M:%S.%f")
    start = ReadableTimeToTimeStamp(start)
    end = ReadableTimeToTimeStamp(end)
    binanceUrl = f'https://fapi.binance.com/fapi/v1/klines?symbol={coin}USDT&interval=1h&startTime={start}&endTime={end}'
    data = requests.get(binanceUrl).json()
    for row in data:
        queuesFor14hRSI[coin].append(float(row[4]))                    
    
    # For 14d RSI
    queuesFor14dRSI[coin] = deque()
    startTime = timeNow - timedelta(days = 14)
    endTime = timeNow
    start = startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
    end = endTime.strftime("%Y-%m-%d %H:%M:%S.%f")
    start = ReadableTimeToTimeStamp(start)
    end = ReadableTimeToTimeStamp(end)
    binanceUrl = f'https://fapi.binance.com/fapi/v1/klines?symbol={coin}USDT&interval=1d&startTime={start}&endTime={end}'
    data = requests.get(binanceUrl).json()
    for row in data:
        queuesFor14dRSI[coin].append(float(row[4]))

def MACDBackgroundLoop():
    for coin in coinList:
        price12dQueue[coin] = deque()
        price26dQueue[coin] = deque()
        timeNow = datetime.now()
        startTime = timeNow - timedelta(days = 26)
        endTime = timeNow
        start = startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        end = endTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        start = ReadableTimeToTimeStamp(start)
        end = ReadableTimeToTimeStamp(end)
        binanceUrl = f'https://fapi.binance.com/fapi/v1/klines?symbol={coin}USDT&interval=1d&startTime={start}&endTime={end}'
        data = requests.get(binanceUrl).json()

        priceList = []
        for row in data:
            priceList.append(float(row[4]))
        
        priceArray = numpy.array(priceList)
        
        for price in priceArray[14:]:
            price12dQueue[coin].append(price)

        for price in priceArray:
            price26dQueue[coin].append(price)

    while True:
        timer = datetime.now()
        try:
            timer = datetime.strptime(str(timer), "%Y-%m-%d %H:%M:%S.%f")
        except:
            pass
        for coin in coinList:
            price = getCurrentPrice(coin)
            if (timer.hour == 19 and timer.minute == 59 and timer.second == 59):
                price12dQueue[coin].popleft()
                price26dQueue[coin].popleft()
                price12dQueue[coin].append(price)
                price26dQueue[coin].append(price)
                
            if ((timer.minute == 59 and timer.second == 59) or (timer.minute == 14 and timer.second == 59)
            or (timer.minute == 29 and timer.second == 59) or (timer.minute == 44 and timer.second == 59)):

                price12dQueue[coin][-1] = price
                price26dQueue[coin][-1] = price

                price12dDF = pd.DataFrame(list(price12dQueue[coin]))
                price26dDF = pd.DataFrame(list(price26dQueue[coin]))

                EMA12d = price12dDF.ewm(span=12, adjust=False).mean()
                EMA26d = price26dDF.ewm(span=26, adjust=False).mean()

                currentEMA12d = EMA12d.iloc[-1] 
                currentEMA26d = EMA26d.iloc[-1]

                EMA12vsPrice = float(((currentEMA12d - price) / price) * 100)
                EMA26vsPrice = float(((currentEMA26d - price) / price) * 100)

                currentPriceVSCurrentEMA12d[coin] = float(EMA12vsPrice)
                currentPriceVSCurrentEMA26d[coin] = float(EMA26vsPrice)
                
                if (coin == coinList[-1]):
                    sleep(1)

def rsiBackgroundLoop():

    for coin in coinList:
        # For 14m RSI 
        queuesFor14mRSI[coin] = deque()
        timeNow = datetime.now()
        startTime = timeNow - timedelta(minutes = 210)
        endTime = timeNow
        start = startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        end = endTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        start = ReadableTimeToTimeStamp(start)
        end = ReadableTimeToTimeStamp(end)
        binanceUrl = f'https://fapi.binance.com/fapi/v1/klines?symbol={coin}USDT&interval=15m&startTime={start}&endTime={end}'
        data = requests.get(binanceUrl).json()
        
        for row in data:
            queuesFor14mRSI[coin].append(float(row[4]))

        # For 14h RSI
        queuesFor14hRSI[coin] = deque()
        startTime = timeNow - timedelta(hours = 14)
        endTime = timeNow
        start = startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        end = endTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        start = ReadableTimeToTimeStamp(start)
        end = ReadableTimeToTimeStamp(end)
        binanceUrl = f'https://fapi.binance.com/fapi/v1/klines?symbol={coin}USDT&interval=1h&startTime={start}&endTime={end}'
        data = requests.get(binanceUrl).json()
        for row in data:
            queuesFor14hRSI[coin].append(float(row[4]))                    

        # For 14d RSI
        queuesFor14dRSI[coin] = deque()
        startTime = timeNow - timedelta(days = 14)
        endTime = timeNow
        start = startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        end = endTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        start = ReadableTimeToTimeStamp(start)
        end = ReadableTimeToTimeStamp(end)
        binanceUrl = f'https://fapi.binance.com/fapi/v1/klines?symbol={coin}USDT&interval=1d&startTime={start}&endTime={end}'
        data = requests.get(binanceUrl).json()
        for row in data:
            queuesFor14dRSI[coin].append(float(row[4]))

    while True:
        timer = datetime.now()
        try:
            timer = datetime.strptime(str(timer), "%Y-%m-%d %H:%M:%S.%f")
        except:
            pass
        for coin in coinList:
            price = getCurrentPrice(coin)
            if (timer.minute == 59 and timer.second == 59):
                queuesFor14hRSI[coin].popleft()
                queuesFor14hRSI[coin].append(price)

            if (timer.hour == 19 and timer.minute == 59 and timer.second == 59):
                queuesFor14dRSI[coin].popleft()
                queuesFor14dRSI[coin].append(price)
                
            if ((timer.minute == 59 and timer.second == 59) or (timer.minute == 14 and timer.second == 59)
            or (timer.minute == 29 and timer.second == 59) or (timer.minute == 44 and timer.second == 59)):
                queuesFor14mRSI[coin].popleft()
                queuesFor14mRSI[coin].append(price)
                queuesFor14hRSI[coin][-1] = price
                queuesFor14dRSI[coin][-1] = price    

                arrayFor14mRSI = numpy.array(queuesFor14mRSI[coin])
                arrayFor14hRSI = numpy.array(queuesFor14hRSI[coin])
                arrayFor14dRSI = numpy.array(queuesFor14dRSI[coin])
                
                # delta
                delta14m = numpy.diff(arrayFor14mRSI)
                delta14h = numpy.diff(arrayFor14hRSI)
                delta14d = numpy.diff(arrayFor14dRSI)

                # Copy
                up14m, down14m = delta14m.copy(), delta14m.copy()
                up14h, down14h = delta14h.copy(), delta14h.copy()
                up14d, down14d = delta14d.copy(), delta14d.copy()

                # Up and Down
                up14m[up14m < 0] = 0
                down14m[down14m > 0] = 0
                up14h[up14h < 0] = 0
                down14h[down14h > 0] = 0
                up14d[up14d < 0] = 0
                down14d[down14d > 0] = 0

                # RS
                RS14m = numpy.average(up14m) / abs(numpy.average(down14m))
                RS14h = numpy.average(up14h) / abs(numpy.average(down14h))
                RS14d = numpy.average(up14d) / abs(numpy.average(down14d))

                # RSI
                RSI14m = (100 - 100 / (1 + RS14m))
                RSI14h = (100 - 100 / (1 + RS14h))
                RSI14d = (100 - 100 / (1 + RS14d))

                currentRSIDictionary[f"{coin}RSI14m"] = RSI14m
                currentRSIDictionary[f"{coin}RSI14h"] = RSI14h
                currentRSIDictionary[f"{coin}RSI14d"] = RSI14d                    

                if (coin == coinList[-1]):
                    sleep(1)

def Main():
    
    background = threading.Thread(name="rsiBackgroundLoop", target=rsiBackgroundLoop)
    background.start()
    background = threading.Thread(name="retrieveCurrentPrice", target=retrieveCurrentPrice)
    background.start()
    sleep(1)
    background = threading.Thread(name="MACDBackgroundLoop", target=MACDBackgroundLoop)
    background.start()
    background = threading.Thread(name="Commands", target=Commands)
    background.start()
    getPricesForTraining(coinList)
    background = threading.Thread(name="updatePriceList", target=updatePriceList, args=(coinList,))
    background.start()
    getCoinPrecision()

    print("Done!")
    print("Making Money...")

    while True:
        timer = datetime.now()
        try:
            timer = datetime.strptime(str(timer), "%Y-%m-%d %H:%M:%S.%f")
        except:
            pass
        if ((timer.minute == 59 and timer.second == 59) or (timer.minute == 14 and timer.second == 59)
            or (timer.minute == 29 and timer.second == 59) or (timer.minute == 44 and timer.second == 59)):
            available = getAvailableBalance()
            for coin in coinList:
                price = getCurrentPrice(coin)
                availableTetherForCoin[coin] = available / len(coinList)
                action = [0,0,0]
                prediction = winningAgents[coin].model(torch.tensor([currentRSIDictionary[f"{coin}RSI14m"],
                                                        currentRSIDictionary[f"{coin}RSI14h"],
                                                        currentRSIDictionary[f"{coin}RSI14d"],
                                                        currentPriceVSCurrentEMA12d[coin],
                                                        currentPriceVSCurrentEMA26d[coin]
                                                        ], dtype=torch.float))

                move = torch.argmax(prediction).item()
                action[move] = 1
                
                if action == [1,0,0]:
                    amount = format((availableTetherForCoin[coin] / 2) / price, '.8f')
                    amount = float(amount[:int(math.log10(smallestPrecision[coin]) * -1) + 2])
                    if amount < smallestPrecision[coin]:
                        print("Not enough to buy with")
                        pass

                    else:
                        orderList.append({
                            "instrument_name": f"{coin}_USDT",
                            "side": "BUY",
                            "type": "MARKET",
                            "quantity": f"{amount}"
                            })

                        print(f"bought ${amount} {coin}")

                elif action == [0,1,0]:
                    amount = format(getCoinBalance(coin), '.8f')
                    amount = float(amount[:int(math.log10(smallestPrecision[coin]) * -1) + 2])
                    if amount < smallestPrecision[coin]:
                        print("Too little to sell")
                        pass

                    else:
                        orderList.append({
                            "instrument_name": f"{coin}_USDT",
                            "side": "SELL",
                            "type": "MARKET",
                            "quantity": f"{amount}"
                            })

                        print(f"sold {amount} {coin}")

                elif action == [0,0,1]:
                    print("Hold")
                    pass
            
            CompleteOrder(orderList)
            orderList.clear()
            sleep(1)
   
# StartUp Sequence
print("What is your api key?")
api_key = input()
print("What is your api secret?")
api_secret = input()
print("What will you be investing in?")
done = False
while done == False:
    coinName = input()
    if (verifyCoin(coinName) == True):
        coinList.append(str(coinName))
    else:
        print(f"{coinName} is not a valid coin")
    print("more? y/n")
    more = input()
    if (more == 'n'):
        break
    elif (more == 'y'):
        continue
    else:
        print("invalid input")
        done = True
coinSet = set(coinList)
newCoinList = list(coinSet)
coinList = newCoinList

print("Commands:")
print("HARDSTOP: stops everything immediately")
print("softstop: stops everything when investments are not at a loss")
print("total balance: total balance of all assets")
print("delete coin: coin entered will no longer be used")
print("add coin: add coin to bot")
print("all coins: shows all coins")

print("Starting Up...")

if __name__ == '__main__':
    Main()

# TODO
# Create new version which uses the Command design pattern
# graduate from using Binance for public price data and just use crypto.com
# Instead of all the data like RSI and EMA, just use the raw price data to determine the same thing
# Use PyGad (genetic training algorithm) instead of whatever I'm doing now which takes forever to train
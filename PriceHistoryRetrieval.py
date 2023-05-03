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
from NeuralNetwork import listsToTorch
from matplotlib import pyplot as plt

coinDayPriceQueues = {}
coinHourPriceQueues = {}
coinQuarterHourPriceQueues = {}

progressPoint = 0

# On Startup -------------------------------------------------------------------#

def progressBar(progress, total):
    percent = 100 * (progress / float(total))
    bar = '█' * int(percent) + '-' * (100 - int(percent))
    print(f"\r|{bar}| {percent:.2f}%", end="\r")

def ReadableTimeToTimeStamp(readableTime):
    s = datetime.strptime(readableTime, "%Y-%m-%d %H:%M:%S.%f")
    timestamp = datetime.timestamp(s)
    timestamp = int(timestamp * 1000)
    return timestamp

def TimeStampToReadableTime(timeStamp):
    readableTime = datetime.fromtimestamp(timeStamp/1000)
    return readableTime

def getPricesForTraining(coinList):
    global progressPoint
    timeNow = datetime.now()
    timeNow = timeNow.replace(microsecond=0,second=0,minute=0,hour=0)

    for coin in coinList:
        timelist = []
        coinDayPriceQueues[coin] = deque()
        coinHourPriceQueues[coin] = deque()
        coinQuarterHourPriceQueues[coin] = deque()

        startTime = timeNow - timedelta(hours = 3000)
        endTime = timeNow - timedelta(hours = 2500)
        
        while startTime < timeNow:
            start = startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
            end = endTime.strftime("%Y-%m-%d %H:%M:%S.%f")
            start = ReadableTimeToTimeStamp(start)
            end = ReadableTimeToTimeStamp(end)
            binanceQuarterlUrl = f'https://fapi.binance.com/fapi/v1/klines?symbol={coin}USDT&interval=15m&startTime={start}&endTime={end}'
            data15m = requests.get(binanceQuarterlUrl).json()

            for row in data15m:
                timelist.append(row[0])
                coinQuarterHourPriceQueues[coin].append(float(row[4]))
                progressPoint += 1
                progressBar(progressPoint, 36000 * len(coinList))

            startTime = startTime + timedelta(hours = 125)
            endTime = endTime + timedelta(hours = 125)

        startTime = timeNow - timedelta(hours = 3000)
        endTime = timeNow - timedelta(hours = 2500)

        while startTime < timeNow:
            
            start = startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
            end = endTime.strftime("%Y-%m-%d %H:%M:%S.%f")
            start = ReadableTimeToTimeStamp(start)
            end = ReadableTimeToTimeStamp(end)
            binanceHourlyUrl = f'https://fapi.binance.com/fapi/v1/klines?symbol={coin}USDT&interval=1h&startTime={start}&endTime={end}'
            data1h = requests.get(binanceHourlyUrl).json()

            for row in data1h:
                for i in range(4):
                    coinHourPriceQueues[coin].append(float(row[4]))
                    progressPoint += 1
                    progressBar(progressPoint, 36000 * len(coinList))
                    
            startTime = startTime + timedelta(hours = 500)
            endTime = endTime + timedelta(hours = 500)

        startTime = timeNow - timedelta(days = 125)
        endTime = timeNow

        start = startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        end = endTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        start = ReadableTimeToTimeStamp(start)
        end = ReadableTimeToTimeStamp(end)
        binanceDailyUrl = f'https://fapi.binance.com/fapi/v1/klines?symbol={coin}USDT&interval=1d&startTime={start}&endTime={end}'
        data1d = requests.get(binanceDailyUrl).json()

        for row in data1d:
            for i in range(96):
                coinDayPriceQueues[coin].append(float(row[4]))
                progressPoint += 1
                progressBar(progressPoint, 36000 * len(coinList))

    for coin in coinList:
        calculateStats(coin,coinDayPriceQueues[coin], coinHourPriceQueues[coin], coinQuarterHourPriceQueues[coin])

def getPricesForTrainingNewCoin(coin):

    timeNow = datetime.now()
    timeNow = timeNow.replace(microsecond=0,second=0,minute=0,hour=0)
    coinDayPriceQueues[coin] = deque()
    coinHourPriceQueues[coin] = deque()
    coinQuarterHourPriceQueues[coin] = deque()

    startTime = timeNow - timedelta(hours = 3000)
    endTime = timeNow - timedelta(hours = 2500)
    
    while startTime < timeNow:
        start = startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        end = endTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        start = ReadableTimeToTimeStamp(start)
        end = ReadableTimeToTimeStamp(end)
        binanceQuarterlUrl = f'https://fapi.binance.com/fapi/v1/klines?symbol={coin}USDT&interval=15m&startTime={start}&endTime={end}'
        data15m = requests.get(binanceQuarterlUrl).json()

        for row in data15m:
            coinQuarterHourPriceQueues[coin].append(float(row[4]))

        startTime = startTime + timedelta(hours = 125)
        endTime = endTime + timedelta(hours = 125)

    startTime = timeNow - timedelta(hours = 3000)
    endTime = timeNow - timedelta(hours = 2500)

    while startTime < timeNow:
        
        start = startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        end = endTime.strftime("%Y-%m-%d %H:%M:%S.%f")
        start = ReadableTimeToTimeStamp(start)
        end = ReadableTimeToTimeStamp(end)
        binanceHourlyUrl = f'https://fapi.binance.com/fapi/v1/klines?symbol={coin}USDT&interval=1h&startTime={start}&endTime={end}'
        data1h = requests.get(binanceHourlyUrl).json()

        for row in data1h:
            for i in range(4):
                coinHourPriceQueues[coin].append(float(row[4]))

        startTime = startTime + timedelta(hours = 500)
        endTime = endTime + timedelta(hours = 500)

    startTime = timeNow - timedelta(days = 125)
    endTime = timeNow

    start = startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
    end = endTime.strftime("%Y-%m-%d %H:%M:%S.%f")
    start = ReadableTimeToTimeStamp(start)
    end = ReadableTimeToTimeStamp(end)
    binanceDailyUrl = f'https://fapi.binance.com/fapi/v1/klines?symbol={coin}USDT&interval=1d&startTime={start}&endTime={end}'
    data1d = requests.get(binanceDailyUrl).json()

    for row in data1d:
        for i in range(96):
            coinDayPriceQueues[coin].append(float(row[4]))

    calculateStats(coin,coinDayPriceQueues[coin], coinHourPriceQueues[coin], coinQuarterHourPriceQueues[coin])

# ------------------------------------------------------------------------------------------------------ #
# Background Price Retrieval Loop
# Everyday, it adds the latest prices to the lists so that the latest training info can be made
# When all the prices are done being retrieved, it calls a function that will calculate all the training info with the latest prices
# Then that function calls a function that creates the loading data for the neural network

def updatePriceList(coinList):
    
    while True:
        timer = datetime.now()
        try:
            timer = datetime.strptime(str(timer), "%Y-%m-%d %H:%M:%S.%f")
        except:
            pass
        if (timer.hour == 0 and timer.minute == 0 and timer.second == 0):
            
            timeNow = datetime.now()
            timeNow = timeNow.replace(microsecond=0,second=0,minute=0,hour=0)

            for coin in coinList:

                for num in range(96):
                    coinQuarterHourPriceQueues[coin].popleft()
                    coinHourPriceQueues[coin].popleft()
                    coinDayPriceQueues[coin].popleft()

                startTime = timeNow - timedelta(hours=24)
                endTime = timeNow - timedelta(minutes=15)
                start = startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
                end = endTime.strftime("%Y-%m-%d %H:%M:%S.%f")
                start = ReadableTimeToTimeStamp(start)
                end = ReadableTimeToTimeStamp(end)
                binanceUrl = f'https://fapi.binance.com/fapi/v1/klines?symbol={coin}USDT&interval=15m&startTime={start}&endTime={end}'
                data = requests.get(binanceUrl).json()
                
                for row in data:
                    coinQuarterHourPriceQueues[coin].append(float(row[4]))
            
                startTime = timeNow - timedelta(hours=24)
                endTime = timeNow - timedelta(hours=1)
                start = startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
                end = endTime.strftime("%Y-%m-%d %H:%M:%S.%f")
                start = ReadableTimeToTimeStamp(start)
                end = ReadableTimeToTimeStamp(end)
                binanceUrl = f'https://fapi.binance.com/fapi/v1/klines?symbol={coin}USDT&interval=1h&startTime={start}&endTime={end}'
                data = requests.get(binanceUrl).json()

                for row in data:
                    for i in range(4):
                        coinHourPriceQueues[coin].append(float(row[4]))
                
                startTime = timeNow - timedelta(hours=24)
                endTime = timeNow
                start = startTime.strftime("%Y-%m-%d %H:%M:%S.%f")
                end = endTime.strftime("%Y-%m-%d %H:%M:%S.%f")
                start = ReadableTimeToTimeStamp(start)
                end = ReadableTimeToTimeStamp(end)
                binanceUrl = f'https://fapi.binance.com/fapi/v1/klines?symbol={coin}USDT&interval=1d&startTime={start}&endTime={end}'
                data = requests.get(binanceUrl).json()
                
                for i in range(96):
                    coinDayPriceQueues[coin].append(float(data[0][4]))
            
            for coin in coinList:
                calculateStats(coin,coinDayPriceQueues[coin], coinHourPriceQueues[coin], coinQuarterHourPriceQueues[coin])

def calculateStats(coin,priceDaysList, priceHoursList, price15MinutesList):
    
    priceDaysList = list(priceDaysList)
    priceHoursList = list(priceHoursList)
    price15MinutesList = list(price15MinutesList)
    RSI14mList = []
    RSI14hList = []
    RSI14dList = []
    currentPriceVSEMA12dList = []
    currentPriceVSEMA26dList = []
    
    # For 14x15 RSI
    startSlice = 0
    endSlice = 14
    price15MinutesArray = numpy.array(price15MinutesList)
    while endSlice < len(price15MinutesList):
        price15MinutesArraySlice = price15MinutesArray[startSlice:endSlice]
        delta14m = numpy.diff(price15MinutesArraySlice)
        up14m, down14m = delta14m.copy(), delta14m.copy()
        up14m[up14m < 0] = 0
        down14m[down14m > 0] = 0
        numpy.seterr(divide='ignore')
        RS14m = numpy.average(up14m) / abs(numpy.average(down14m))
        RSI14m = (100 - 100 / (1 + RS14m))
        RSI14mList.append(RSI14m)
        startSlice += 1
        endSlice += 1
    
    # For 14x56 RSI
    startSlice = 0
    endSlice = 56
    priceHoursArray = numpy.array(priceHoursList)
    while endSlice < len(priceHoursList):
        hours = []
        priceHoursArraySlice = priceHoursArray[startSlice:endSlice]
        firstNumber = priceHoursArraySlice[0]
        hours.append(firstNumber)
        for prices in priceHoursArraySlice:
            if firstNumber != prices:
                hours.append(prices)
                firstNumber = prices
            if (len(hours) > 14):
                hoursArray = numpy.array(hours)
                hours = hoursArray[1:]

        hours[-1] = price15MinutesList[endSlice - 1]
        
        delta14h = numpy.diff(hours)
        up14h, down14h = delta14h.copy(), delta14h.copy()
        up14h[up14h < 0] = 0
        down14h[down14h > 0] = 0
        numpy.seterr(divide='ignore')
        RS14h = numpy.average(up14h) / abs(numpy.average(down14h))
        RSI14h = (100 - 100 / (1 + RS14h))
        RSI14hList.append(RSI14h)
        startSlice += 1
        endSlice += 1

    # For 14x1344 RSI
    startSlice = 0
    endSlice = 1344
    priceDaysArray = numpy.array(priceDaysList)
    while endSlice < len(priceDaysList):
        days = []
        priceDaysArraySlice = priceDaysArray[startSlice:endSlice]
        firstNumber = priceDaysArraySlice[0]
        days.append(firstNumber)
        for prices in priceDaysArraySlice:
            if firstNumber != prices:
                days.append(prices)
                firstNumber = prices
            if (len(days) > 14):
                daysArray = numpy.array(days)
                days = daysArray[1:]
        
        days[-1] = price15MinutesList[endSlice - 1]
        
        delta14d = numpy.diff(days)
        up14d, down14d = delta14d.copy(), delta14d.copy()
        up14d[up14d < 0] = 0
        down14d[down14d > 0] = 0
        numpy.seterr(divide='ignore')
        RS14d = numpy.average(up14d) / abs(numpy.average(down14d))
        RSI14d = (100 - 100 / (1 + RS14d))
        RSI14dList.append(RSI14d)
        startSlice += 1
        endSlice += 1

    # For MACDvsSignalLine
    startSlice = 0
    endSlice = 2496
    priceDaysArray = numpy.array(priceDaysList)
    while endSlice < len(priceDaysList):
        days = []
        priceDaysArraySlice = priceDaysArray[startSlice:endSlice]
        firstNumber = priceDaysArraySlice[0]
        days.append(firstNumber)
        for prices in priceDaysArraySlice:
            if firstNumber != prices:
                days.append(prices)
                firstNumber = prices
            if (len(days) > 26):
                daysArray = numpy.array(days)
                days = daysArray[1:]
        
        days[-1] = price15MinutesList[endSlice - 1]
        
        # find MACD from 'days'
        
        price12dDF = pd.DataFrame(list(days[14:]))
        price26dDF = pd.DataFrame(list(days))

        EMA12d = price12dDF.ewm(span=12, adjust=False).mean()
        EMA26d = price26dDF.ewm(span=26, adjust=False).mean()

        currentEMA12d = EMA12d.iloc[-1] 
        currentEMA26d = EMA26d.iloc[-1]

        EMA12vsPrice = float(((currentEMA12d - days[-1]) / days[-1]) * 100)
        EMA26vsPrice = float(((currentEMA26d - days[-1]) / days[-1]) * 100)

        currentPriceVSEMA12dList.append(EMA12vsPrice)
        currentPriceVSEMA26dList.append(EMA26vsPrice)
        
        startSlice += 1
        endSlice += 1

    print('\n')
    print('training neural network...')

    RSI14mList = RSI14mList[2482:]
    RSI14hList = RSI14hList[2440:]
    RSI14dList = RSI14dList[1152:]
    
    price15MinutesList = price15MinutesList[2496:]
    listsToTorch(coin, RSI14mList,RSI14hList,RSI14dList,currentPriceVSEMA12dList,currentPriceVSEMA26dList, price15MinutesList)
    
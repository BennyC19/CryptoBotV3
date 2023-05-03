import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, OrderedDict
import os
import random
import numpy as numpy
from time import sleep
from matplotlib import pyplot as plt

trainingInformation = {}
winningAgents = {}

def listsToTorch(coin,RSI14mList,RSI14hList,RSI14dList,currentPriceVSEMA12dList,currentPriceVSCurrentEMA26d,price15MinutesList):
    trainingInfoList = [RSI14mList,RSI14hList,RSI14dList,currentPriceVSEMA12dList,currentPriceVSCurrentEMA26d]
    trainingInfo = [list(i) for i in zip(*trainingInfoList)]
    trainingInformation[coin] = trainingInfo  

    train(coin, RSI14dList, price15MinutesList)

def progressBar(progress, total):
    percent = 100 * (progress / float(total))
    bar = '█' * int(percent) + '-' * (100 - int(percent))
    print(f"\r|{bar}| {percent:.2f}%", end="\r")

################################################################################

class NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

#######################################################################

class Agent:
    def __init__(self):
        self.model = NN(5, 3, 3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        self.recordOfActions = []

    def get_action(self, stats):
        action = [0,0,0]
        state0 = torch.tensor(stats, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        action[move] = 1
        return action
    
    def train_step(self, stats, action):
        stats = torch.tensor(stats, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)

        if len(stats.shape) == 1:
            stats = torch.unsqueeze(stats, 0)
            action = torch.unsqueeze(action, 0)

        pred = self.model(stats)
        
        target = pred.clone()
        largest = torch.argmax(action).item()
        target[0][largest] = target[0][largest] * 1.01

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

    def modify_weights(self):
        num = random.randint(0, 3)
        with torch.no_grad():
            if num == 0:
                self.model.linear1.weight[random.randint(0, 2), random.randint(0, 4)] = random.uniform(-1, 1)
            elif num == 1:
                self.model.linear2.weight[random.randint(0, 2), random.randint(0, 2)] = random.uniform(-1, 1)
            elif num == 2:
                self.model.linear1.bias[random.randint(0, 2)] = random.uniform(-1, 1)
            elif num == 3:
                self.model.linear2.bias[random.randint(0, 2)] = random.uniform(-1, 1)

def train(coin, RSI14dList, price15MinutesList):
    progressPoint = 80090
    agent1 = Agent() 
    agent2 = Agent()  
    agent3 = Agent()
    agent4 = Agent()
    agent5 = Agent()
    agent6 = Agent()  
    agent7 = Agent()
    agent8 = Agent()

    recordOfPrices = []

    startSlice = 0
    endSlice = 96 # 12 hours
    price15MinutesArray = numpy.array(price15MinutesList)
    stats = numpy.array(trainingInformation[coin])

    while endSlice < (len(stats) - 400):
        
        price15MinutesArraySlice = price15MinutesArray[startSlice:endSlice]
        statsSlice = stats[startSlice:endSlice]
        
        for index in range(len(statsSlice)):

            currentPrice = price15MinutesArraySlice[index]
            currentStats = statsSlice[index] 
            
            action1 = agent1.get_action(currentStats)
            action2 = agent2.get_action(currentStats)
            action3 = agent3.get_action(currentStats)
            action4 = agent4.get_action(currentStats)
            action5 = agent5.get_action(currentStats)
            action6 = agent2.get_action(currentStats)
            action7 = agent3.get_action(currentStats)
            action8 = agent4.get_action(currentStats)

            recordOfPrices.append(currentPrice)
            agent1.recordOfActions.append(action1)
            agent2.recordOfActions.append(action2)
            agent3.recordOfActions.append(action3)
            agent4.recordOfActions.append(action4)
            agent5.recordOfActions.append(action5)
            agent6.recordOfActions.append(action6)
            agent7.recordOfActions.append(action7)
            agent8.recordOfActions.append(action8)

            progressPoint += 1
            progressBar(progressPoint, 7782912)

        agent1Profit = 0
        buys1 = []
        for price, action in zip(recordOfPrices, agent1.recordOfActions):
            if action == [1,0,0]:
                buys1.append(price)
            elif action == [0,1,0]:
                if len(buys1) != 0:
                    agent1Profit += ((price - (sum(buys1) / len(buys1))) / (sum(buys1) / len(buys1))) * 100
                    buys1.clear()
                else:
                    pass
            elif action == [0,0,1]:
                pass

            progressPoint += 1
            progressBar(progressPoint, 7782912)
        
        agent2Profit = 0
        buys2 = []
        for price, action in zip(recordOfPrices, agent2.recordOfActions):
            if action == [1,0,0]:
                buys2.append(price)
            elif action == [0,1,0]:
                if len(buys2) != 0:
                    agent2Profit += ((price - (sum(buys2) / len(buys2))) / (sum(buys2) / len(buys2))) * 100
                    buys2.clear()
                else:
                    pass
            elif action == [0,0,1]:
                pass

            progressPoint += 1
            progressBar(progressPoint, 7782912)
        
        agent3Profit = 0
        buys3 = []
        for price, action in zip(recordOfPrices, agent3.recordOfActions):
            if action == [1,0,0]:
                buys3.append(price)
            elif action == [0,1,0]:
                if len(buys3) != 0:
                    agent3Profit += ((price - (sum(buys3) / len(buys3))) / (sum(buys3) / len(buys3))) * 100
                    buys3.clear()
                else:
                    pass
            elif action == [0,0,1]:
                pass

            progressPoint += 1
            progressBar(progressPoint, 7782912)

        agent4Profit = 0
        buys4 = []
        for price, action in zip(recordOfPrices, agent4.recordOfActions):
            if action == [1,0,0]:
                buys4.append(price)
            elif action == [0,1,0]:
                if len(buys4) != 0:
                    agent4Profit += ((price - (sum(buys4) / len(buys4))) / (sum(buys4) / len(buys4))) * 100
                    buys4.clear()
                else:
                    pass
            elif action == [0,0,1]:
                pass

            progressPoint += 1
            progressBar(progressPoint, 7782912)
        
        agent5Profit = 0
        buys5 = []
        for price, action in zip(recordOfPrices, agent5.recordOfActions):
            if action == [1,0,0]:
                buys5.append(price)
            elif action == [0,1,0]:
                if len(buys5) != 0:
                    agent5Profit += ((price - (sum(buys5) / len(buys5))) / (sum(buys5) / len(buys5))) * 100
                    buys5.clear()
                else:
                    pass
            elif action == [0,0,1]:
                pass

            progressPoint += 1
            progressBar(progressPoint, 7782912)
        
        agent6Profit = 0
        buys6 = []
        for price, action in zip(recordOfPrices, agent6.recordOfActions):
            if action == [1,0,0]:
                buys6.append(price)
            elif action == [0,1,0]:
                if len(buys6) != 0:
                    agent6Profit += ((price - (sum(buys6) / len(buys6))) / (sum(buys6) / len(buys6))) * 100
                    buys6.clear()
                else:
                    pass
            elif action == [0,0,1]:
                pass

            progressPoint += 1
            progressBar(progressPoint, 7782912)

        agent7Profit = 0
        buys7 = []
        for price, action in zip(recordOfPrices, agent7.recordOfActions):
            if action == [1,0,0]:
                buys7.append(price)
            elif action == [0,1,0]:
                if len(buys7) != 0:
                    agent7Profit += ((price - (sum(buys7) / len(buys7))) / (sum(buys7) / len(buys7))) * 100
                    buys7.clear()
                else:
                    pass
            elif action == [0,0,1]:
                pass

            progressPoint += 1
            progressBar(progressPoint, 7782912)

        agent8Profit = 0
        buys8 = []
        for price, action in zip(recordOfPrices, agent8.recordOfActions):
            if action == [1,0,0]:
                buys8.append(price)
            elif action == [0,1,0]:
                if len(buys8) != 0:
                    agent8Profit += ((price - (sum(buys8) / len(buys8))) / (sum(buys8) / len(buys8))) * 100
                    buys8.clear()
                else:
                    pass
            elif action == [0,0,1]:
                pass

            progressPoint += 1
            progressBar(progressPoint, 7782912)

        #print([agent1Profit, agent2Profit, agent3Profit, agent4Profit, agent5Profit, agent6Profit, agent7Profit, agent8Profit])

        agentProfitDict = { agent1: agent1Profit,
                            agent2: agent2Profit, 
                            agent3: agent3Profit, 
                            agent4: agent4Profit, 
                            agent5: agent5Profit,
                            agent6: agent6Profit,
                            agent7: agent7Profit,
                            agent8: agent8Profit}

        firstPlace = max(agentProfitDict, key=agentProfitDict.get)
        agentProfitDict.pop(firstPlace)
        secondPlace = max(agentProfitDict, key=agentProfitDict.get)
        #agentProfitDict.pop(secondPlace)
        #thirdPlace = max(agentProfitDict, key=agentProfitDict.get)
        #agentProfitDict.pop(thirdPlace)
        #fourthPlace = max(agentProfitDict, key=agentProfitDict.get)
        #agentProfitDict.pop(fourthPlace)
        #fifthPlace = max(agentProfitDict, key=agentProfitDict.get)

        agent1.model = firstPlace.model

        del agent2
        del agent3
        del agent4
        del agent5
        del agent6
        del agent7
        del agent8

        agent2 = Agent()
        agent3 = Agent()
        agent4 = Agent()
        agent5 = Agent()
        agent6 = Agent()
        agent7 = Agent()
        agent8 = Agent()
        
        recordOfPrices.clear()
        agent1.recordOfActions.clear()
        agent2.recordOfActions.clear()
        agent3.recordOfActions.clear()
        agent4.recordOfActions.clear()
        agent5.recordOfActions.clear()
        agent6.recordOfActions.clear()
        agent7.recordOfActions.clear()
        agent8.recordOfActions.clear()

        startSlice += 1
        endSlice += 1

    winningAgents[coin] = agent1

    print("\n")
    
    investment = 100
    volume = 0
    buys = []
    testRecording = []
    for index in range(len(price15MinutesList) - 400, len(price15MinutesList)):
        stats = trainingInformation[coin][index]
        action = agent1.get_action(stats)
        
        if action == [1,0,0]: # Buy
            testRecording.append(investment)
            buys.append(price15MinutesList[index]) 
            volume = volume + investment / (2**len(buys))

        elif action == [0,1,0]: # Sell
            if len(buys) != 0:
                priceBought = sum(buys) / len(buys)
                priceSold = price15MinutesList[index]
                investment = investment + (((priceSold - priceBought) / priceBought)) * investment
                volume += investment
                buys.clear()

            testRecording.append(investment)

        elif action == [0,0,1]: # Hold
            testRecording.append(investment)
            pass

    print(f"Investment return: {investment}")
    print(f"total volume: {volume}")
    plt.plot(testRecording)
    plt.show()  


# TODO
# - try and make the NN more and more accurate (Maybe by adding another agent that will be a copy of firstPlace but it randomizes the weights and biases a bit)
# - put in place some security measures so that it doesnt lose a whole bunch at once
# - convert to crypto.com instead of coinbase
# - make a basic Tkinter GUI
# - put everything in classes and clean up code
# - attach a video to program that guides users how to get the api key's
# - include google trends data

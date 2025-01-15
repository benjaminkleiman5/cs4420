import numpy as np
import json
import kaldiio
import random
import os

class CharacterTokenizer:
    def __init__(self):
        letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        #Generate dictionary of form {A:1, B:2, ... Z:26}
        self.letterToNumber = {letter: index + 1 for index, letter in enumerate(letters)}
        self.letterToNumber["'"] = 27
        self.letterToNumber[" "] = 28
        #Create dictionary of inverse form
        self.numberToLetter = {value: key for key, value in self.letterToNumber.items()}

    def StringToIds(self, string):
        charList = list(string)
        retList = []
        for char in charList:
            retList.append(self.letterToNumber[char])
        return retList
    
    def IdsToString(self, idList):
        retString = ""
        for ID in idList:
            retString = retString + str(self.numberToLetter[ID])
        return retString

def splice_and_subsample(arr, c, r):
    copy = c
    count = 0
    if(c == 0):
        return arr
    leftList = np.concatenate((arr[[0]], arr[0:len(arr)-1]))
    rightList = np.concatenate((arr[1:len(arr)], arr[[len(arr)-1]]))
    retMatrix = np.concatenate((leftList, arr, rightList), axis = 1)
    c -= 1
    while c!= 0:
        arrLeft = leftList
        arrRight = rightList
        leftList = np.concatenate((arrLeft[[0]], arrLeft[0:len(arr)-1]))
        rightList = np.concatenate((arrRight[1:len(arr)], arrRight[[len(arr)-1]]))
        retMatrix = np.concatenate((leftList, retMatrix, rightList), axis = 1)
        c -= 1
    return retMatrix[::r, :]
    
class InputGenerator:
    def __init__(self, jsonFile, batch_size, shuffle, context_length, subsampling_rate):
        self.epoch = 0
        self.total_num_steps = 0
        self.batchSize = batch_size
        self.shuffle = shuffle
        self.indexList = []
        self.flag = True
        self.resetFlag = False
        self.c = context_length
        self.r = subsampling_rate
        self.keys = []
        self.tokenizer = CharacterTokenizer()
        self.dic = {}
        with open(str(jsonFile)) as f:
            self.jsonLoad = json.load(f)
        f.close()

    def next(self):
        retList = []
        if(self.resetFlag):
            self.total_num_steps = 0
            self.resetFlag = False
        if not self.dic:
            self.dic = self.jsonLoad["utts"]
        if not self.keys:
            self.keys = list(self.dic.keys())
        #Construct indexList on first next() of the epoch
        if not self.indexList:
            self.indexList = list(range(1, len(self.keys) + 1))
            if self.shuffle:
                random.shuffle(self.indexList)
        #Pad batches if neccessary on the first next() of the epoch
        if(self.flag):
            #Add extra batches
            if(len(self.indexList) % self.batchSize):
                for i in range((((len(self.keys)//self.batchSize) + 1) * self.batchSize) - len(self.keys)):
                    self.indexList.append(random.choice(self.indexList))
            self.flag = False
        for i in range(self.batchSize):
            utterID = self.keys[self.indexList[self.batchSize * self.total_num_steps + i] - 1]
            splicedAndSubbed = splice_and_subsample(kaldiio.load_mat(self.dic[utterID]["feat"]), self.c, self.r)
            token = self.tokenizer.StringToIds(self.dic[utterID]["text"])
            retList.append((utterID, splicedAndSubbed, token))
        self.total_num_steps += 1
        if(self.total_num_steps == (len(self.indexList) / self.batchSize)):
            print(self.total_num_steps)
            self.epoch += 1
            self.resetFlag = True
            self.flag = True
            self.dic = {}
            self.keys = {}
            self.indexList = []
        return retList

    



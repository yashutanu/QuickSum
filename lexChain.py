#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 08:13:08 2018

@author: chitmi
"""

import math
import os
from nltk.corpus import wordnet as wn
import json
addedToChain=0
lineNumber=-1
summary=[]

class LexicalChain:


    def findScoreofChain(self,chains,sentences,maxSent):
        #calculate score for each chain
        chainScore={}
        for chain in chains:
            length=len(chain)
            temp=[]
            for val in chain:
                temp.append(val.split(",")[0])
            temp=list(set(temp))
            homogenIndex=float(float(length-len(temp))/length)
            scoreofChain=float(length*homogenIndex)
            if (str(scoreofChain) in chainScore):
                chainScore[str(scoreofChain)].append(chain)
            else:
                chainScore[str(scoreofChain)]=[]
                chainScore[str(scoreofChain)].append(chain)
        #print(chains)
        #print(chainScore)
        numerator=0
        sum=0
        #calculate the strong chain score

        for k in chainScore.keys():
            sum+=float(k)
        average=float(sum/len(chainScore.keys()))
        for key in chainScore:
            numerator+=(average-float(key))**2
        standardDev=math.sqrt(numerator/len(chainScore.keys()))
        strongChainScore=average+((0.2)*(standardDev))
        strongChainScore = average
        sentenceScore={}
        #check all the chain that have larger value than strong chain and count the occurences of terms in chain fo every sentence in these strong chains
        for key in chainScore:
            if float(key)>=int(strongChainScore):
                for List in chainScore[key]:
                    for val in List:
                            sentNum=val.split(',')[1]
                            if sentNum in sentenceScore:
                                sentenceScore[sentNum]+=1
                            else:
                                sentenceScore[sentNum]=1

        return (sentenceScore)
        #get maxsent choose the values with max values first and then if maxsent is remain choose based on index


    def addRelated(self,word, chain,dataList):
        global addedToChain
        if(addedToChain==0 and len(dataList)>0):
            for w in dataList:
                for c in chain:
                    ch=c.split(",")[0]
                    if w in ch and addedToChain==0:
                        chain.append(str(word)+","+str(lineNumber))
                        addedToChain=1
                        return


    def addSynonyms(self,word,chain):
        global lineNumber
        global addedToChain
        syn=wn.synsets(word)
        synList=[str(s.name().split(".")[0]) for s in syn]
        synList= (list(set(synList)))
        self.addRelated(word,chain,synList)
        if(addedToChain==0):
            for s in syn:
                hyperList=[str(data.name().split(".")[0]) for data in s.hypernyms()]
                hyperList= (list(set(hyperList)))
                self.addRelated(word, chain,hyperList)
                if (addedToChain == 1):
                    return
                hypoList=[str(data.name().split(".")[0]) for data in s.hyponyms()]
                hypoList=(list(set(hypoList)))
                self.addRelated(word, chain, hypoList)
                if (addedToChain == 1):
                    return
                partHolonymList=[str(data.name().split(".")[0]) for data in s.part_holonyms()]
                partHolonymList = (list(set(partHolonymList)))
                self.addRelated(word, chain,partHolonymList)
                if (addedToChain == 1):
                    return
                subsHolonynmList = [str(data.name().split(".")[0]) for data in s.substance_holonyms()]
                subsHolonynmList = (list(set(subsHolonynmList)))
                self.addRelated(word, chain,subsHolonynmList)
                if (addedToChain == 1):
                    return

    def addFirstWord(self,word,chains):
        global addedToChain
        global lineNumber
        chains.append([str(word)+","+str(lineNumber)])
        addedToChain=1

    def addRepeatedWord(self,word,chain):
        global addedToChain
        global lineNumber
        chain.append(str(word)+","+str(lineNumber))
        addedToChain=1

    def generateSummary(self,dirname,fileName):
        global summary
        #print("fileName: ",fileName)
        os.chdir(dirname)
        if not os.path.exists("lexchain"):
            os.mkdir("lexchain")
        os.chdir("lexchain")   
        filePath="lexicalChain_"+fileName
        #print("filepath in generateSummary:",filePath)
        with open(filePath,"w") as fpSum:
            for line in summary:
                fpSum.write(line+". ")


    def findBestSentences(self,sentences,maxSent,sentenceScore,dirname,file):
        global summary
        summary=[]
        sortedScores=sorted(list(sentenceScore.values()),reverse=True)
        count=0
        #targetVal=sortedScores[0]
        #print ("max Sent :" +str(int(maxSent)-1))
        #print("size of sortedScores:",len(sortedScores))
        #print("maxsent: ",maxSent)
        try:
            targetVal=sortedScores[int(maxSent)-1]
        except:
            targetVal=sortedScores[0]
        sortedKeys=sorted(list(sentenceScore.keys()),key=int)
        for key in sortedKeys:
            if sentenceScore[key] >= targetVal and count<maxSent:
                count+=1
                summary.append(sentences[int(key)])
        #print(summary)
        self.generateSummary(dirname,file)


    def assignChain(self,wordList,chains,sentences,percentage,dirname,file):
        global lineNumber
        global addedToChain
        lineNumber=-1
        addedToChain=0
        for sentence in wordList:
            lineNumber+=1
            for word in sentence:
                word=str(word)
                addedToChain=0
                if len(chains)==0:
                    self.addFirstWord(word,chains)
                else:
                    for chain in chains:
                        if any(word in w for w in chain):
                            self.addRepeatedWord(word,chain)
                        else:
                            self.addSynonyms(word,chain)
                if(addedToChain==0):
                    #did not find a synonym nor a lemma name common add it to new chain
                    chains.append([str(word)+","+str(lineNumber)])

        maxSent = math.ceil(percentage * len(sentences))
        sentenceScore=self.findScoreofChain(chains,sentences,maxSent)
        self.findBestSentences(sentences,maxSent,sentenceScore,dirname,file)

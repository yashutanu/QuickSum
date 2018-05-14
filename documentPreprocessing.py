#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 08:09:19 2018

@author: chitmi
"""

import re
import os
import nltk
from nltk.stem.wordnet import WordNetLemmatizer as wnl

class document_Processing:

    def not_stopword(self,word):
        if word.lower() in nltk.corpus.stopwords.words('english'):
            return False
        else:
            return True

    def pickNounAndLemmatize(self,sentences):
        wordList=[]
        for sentence in sentences:
            temp=[]
            words=sentence.split(" ")
            words=list(filter(None, words))
            tag_tuples = nltk.pos_tag(words)
            for tup in tag_tuples:
                if 'NN' in tup[1]:
                    #regex #lemmatize #stopwordCheck
                    word=re.sub('[^A-Za-z]+', '', tup[0])
                    word=word.lower()
                    lmtzr = wnl()
                    word=lmtzr.lemmatize(word)
                    if(self.not_stopword(word) and len(word)>2):
                        temp.append(word)
            wordList.append(temp)
        return wordList

    def preprocessing(self,file,dirname):
            sentenceList=[]
            os.chdir(dirname)
            fileHandler=open(file,"r",encoding="latin1")
            fileContent=fileHandler.read()
            articleContent=fileContent.split("\n", 1)[1]
            paragraph=articleContent.strip()
            sentenceEnders = re.compile('[.!?]')
            sentList = sentenceEnders.split(paragraph)
            for sentence in sentList:
                if (sentence!=""):
                    sentence=sentence.replace("\n","")
                    sentenceList.append(sentence.strip())
            return sentenceList

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 19:30:05 2018

@author: yashu
"""

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
import pandas as pd
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import global_var as gv

def get_pos(tagname):
	if tagname.startswith('VB'):
		return wordnet.VERB
	elif tagname.startswith('RB'):
		return wordnet.ADV
	elif tagname.startswith('JJ'):
		return wordnet.ADJ
	return wordnet.NOUN

def wordTokenize(sentence):
	return word_tokenize(sentence)

def isValidTerm(term):
	return term.isalnum() and term.lower() not in gv.stopWords

# function to lemmatize the tweets
def lemmatize(sentence):
    #sentence=nlp(sentence)
    words=wordTokenize(sentence)
    pos_words=pos_tag(words)
    global stopWords
    global lemmatizer
    str=""
    for word in pos_words:
        wordpre = word[0].strip()
        if isValidTerm(wordpre):
                wordpre_root = gv.lemmatizer.lemmatize(wordpre,get_pos(word[1]))
                str+=" "+wordpre_root
    return str
#PREPROCESSING
def tokenization(X):
        tokenizer = Tokenizer(num_words=None,
                                           filters='!"#$%&()*+,/:;<=>?@[\\]^_`{|}~\t\n',
                                           lower=False,
                                           split=" ",
                                           char_level=False,
                                           oov_token=None)
        
        
        tokenizer.fit_on_texts(X)
        return tokenizer

#MODEL
print('Building model...')
def build_model(parameters,max_vocab_size,maxlength):
        filters=parameters["filters"]
        embedding_size=parameters["embedding_size"]
        kernel_size=parameters["kernel_size"]
        pool_size=parameters["pool_size"]
        lstm_output_size=parameters["lstm_output_size"]
        model = Sequential()
        model.add(Embedding(max_vocab_size, embedding_size, input_length=maxlength))
        #model.add(Dropout(0.20))
        model.add(Conv1D(filters,
                         kernel_size,
                         padding='valid',
                         activation='relu',
                         strides=1))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(LSTM(lstm_output_size))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

def getTokenizer(X_train):
        X_train=[ lemmatize(k) for k in X_train]
        tokenizer=tokenization(X_train)
        return tokenizer
#TRAINING

def prepareData(datapath):
    train=pd.read_csv(datapath)
    Y_train = list(train[train.columns[2]])
    X_train = list(train[train.columns[1]])
    return X_train,Y_train


def Training(datapath,parameters):
    X_train,Y_train=prepareData(datapath)
    X_train=[ lemmatize(k) for k in X_train]
    train_x, test_x, train_y, test_y = train_test_split(X_train, Y_train, test_size=0.30,random_state=42 )
    tokenizer=tokenization(X_train)
    sequences_train = tokenizer.texts_to_sequences(train_x)
    sequences_test=tokenizer.texts_to_sequences(test_x)
    maxlength=len(max(sequences_train,key=lambda x:len(x)))
    word_index = tokenizer.word_index
    vocabulory_size=len(word_index)
    print("vocabulory:",vocabulory_size)
    max_vocab_size = vocabulory_size+1
    batch_size=parameters["batch_size"]
    epochs=parameters["epochs"]
    x_train =pad_sequences(sequences_train, maxlen=maxlength,padding="post")
    x_test =pad_sequences(sequences_test, maxlen=maxlength,padding="post")
    train_model=build_model(parameters,max_vocab_size,maxlength)
    print(len(train_x),len(train_y),len(test_x),len(test_y))
    print("maxlength:",maxlength)
    #print('x_train shape:', x_train.shape)
    #print('x_test shape:', x_test.shape)
    print("####Training####")
    train_model.fit(x_train,train_y,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test,test_y))
    
    print("####Testing####")
    score, acc=train_model.evaluate(x_test,test_y, batch_size=batch_size)
    plot_model(train_model, to_file='model_plot_v2.png', show_shapes=True, show_layer_names=True)
    print("Summary:",train_model.summary())
    print('Test score:', score)
    print('Test accuracy:', acc)
    # serialize model to JSON
    model_json = train_model.to_json()
    with open("model.json", "w") as json_file:
            json_file.write(model_json)
    # serialize weights to HDF5
    train_model.save_weights("model.h5")
    
    return train_model,tokenizer,maxlength

"""
model,tokenizer,maxlength=Training(gv.sentenceTrainPath,gv.TrainingParameter)

testing_document=pd.read_csv("test.csv")
test_doc=list(testing_document[testing_document.columns[1]])
test_doc1=list(testing_document[testing_document.columns[0]])
X,Y=prepareData(gv.sentenceTrainPath)
test_doc=[ lemmatize(k) for k in test_doc]
tokenizer=getTokenizer(X)
sequences_test_doc=tokenizer.texts_to_sequences(test_doc)
test_docs=pad_sequences(sequences_test_doc,maxlen=maxlength,padding="post")

predicted_output=model.predict(test_docs)
rounded = [int(round(x[0])) for x in predicted_output]
result=list(zip(rounded,test_doc,test_doc1))
#print(result)
for i in result:
    if i[0]==1:
        print("Sentence:",i[0],"====",i[1],"======",i[2])
"""

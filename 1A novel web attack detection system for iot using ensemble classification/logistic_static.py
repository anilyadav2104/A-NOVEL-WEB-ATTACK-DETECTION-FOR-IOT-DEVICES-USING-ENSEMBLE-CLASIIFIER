# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 11:19:15 2018

@author: AVANTIKA GUPTA
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 03:13:21 2018

@author: AVANTIKA GUPTA
"""
# from flask import Flask
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import random
import sys
import json
from sklearn.linear_model import LogisticRegression
import math
from collections import Counter
from sklearn import metrics
import pickle
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras import callbacks
from keras.callbacks import CSVLogger
import keras
import keras.preprocessing.text
import itertools
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import callbacks
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.layers import Convolution1D, MaxPooling1D
from sklearn import preprocessing


# app = Flask(__name__)
def entropy(s):
	p, lns = Counter(s), float(len(s))
	return -sum( count/lns * math.log(count/lns, 2) for count in p.values())

def getTokens(input):
	tokensBySlash = str(input.encode('utf-8')).split('/')	#get tokens after splitting by slash
	allTokens = []
	for i in tokensBySlash:
		tokens = str(i).split('-')	#get tokens after splitting by dash
		tokensByDot = []
		for j in range(0,len(tokens)):
			tempTokens = str(tokens[j]).split('.')	#get tokens after splitting by dot
			tokensByDot = tokensByDot + tempTokens
		allTokens = allTokens + tokens + tokensByDot
	allTokens = list(set(allTokens))	#remove redundant tokens
	if 'com' in allTokens:
		allTokens.remove('com')	#removing .com since it occurs a lot of times and it should not be included in our features
	return allTokens

def TL1():
    allurls = 'major.combined.csv'	#path to our all urls file
    allurlscsv = pd.read_csv(allurls,',',error_bad_lines=False)	#reading file
    allurlsdata = pd.DataFrame(allurlscsv)	#converting to a dataframe
    
    allurlsdata = np.array(allurlsdata)	#converting it into an array
    random.shuffle(allurlsdata)	#shuffling

    y = [d[1] for d in allurlsdata]	#all labels 
    corpus = [d[0] for d in allurlsdata]	#all urls corresponding to a label (either good or bad)
    vectorizer = TfidfVectorizer(tokenizer=getTokens)	#get a vector for each url but use our customized tokenizer
    X = vectorizer.fit_transform(corpus)	#get the X vector

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)	#split into training and testing set 80/20 ratio

    lgs = LogisticRegression()	#using logistic regression
    lgs.fit(X_train, y_train)
    y_pred = lgs.predict(X_test)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(cnf_matrix)
    print(lgs.score(X_test, y_test))	#pring the score. It comes out to be 98%
    return vectorizer, lgs


def TL():
    allurls = 'major.combined.csv'	#path to our all urls file
    allurlscsv = pd.read_csv(allurls,',',error_bad_lines=False)	#reading file
    allurlsdata = pd.DataFrame(allurlscsv)	#converting to a dataframe
    print(allurlsdata.shape)

    allurlsdata = np.array(allurlsdata)	#converting it into an array
    random.shuffle(allurlsdata)	#shuffling
    #arr=np.array(allurlsdata)
    x=allurlsdata[1:,2:]
    y=allurlsdata[1:,1]
    print(x.shape)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)	#split into training and testing set 80/20 ratio
    
    lgs = LogisticRegression()	#using logistic regression
    lgs.fit(X_train, y_train)
    y_pred = lgs.predict(X_test)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(cnf_matrix)
    print(lgs.score(X_test, y_test))	#pring the score. It comes out to be 98%
    return lgs
    
def cnnlstm():
    hidden_dims = 128
    nb_filter = 64
    filter_length = 5 
    embedding_vecor_length = 128
    pool_length = 4
    lstm_output_size = 70
    
    allurls = 'major.combined.csv'	#path to our all urls file
    allurlscsv = pd.read_csv(allurls,',',error_bad_lines=False)	#reading file
    allurlsdata = pd.DataFrame(allurlscsv)	#converting to a dataframe
    
    allurlsdata = np.array(allurlsdata)	#converting it into an array
    random.shuffle(allurlsdata)	#shuffling

    y = [d[1] for d in allurlsdata]	#all labels 
    corpus = [d[0] for d in allurlsdata]	#all urls corresponding to a label (either good or bad)
    vectorizer = TfidfVectorizer(tokenizer=getTokens, ngram_range=(1,2), binary=True, max_features=50)	#get a vector for each url but use our customized tokenizer
    X = vectorizer.fit_transform(corpus)	#get the X vector
    print(X.shape)

    lb = preprocessing.LabelBinarizer()
    y = lb.fit_transform(y)

    maxlen = X.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)	#split into training and testing set 80/20 ratio
    
    model = Sequential()
    model.add(Embedding(500, embedding_vecor_length, input_length=maxlen))
    model.add(Convolution1D(nb_filter=nb_filter,
        filter_length=filter_length,
        border_mode='valid',
        activation='relu',
        subsample_length=1))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    print(model.summary())
    
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    checkpointer = callbacks.ModelCheckpoint(filepath="checkpoint-{epoch:02d}.hdf5", save_best_only=True, monitor='loss')
    csv_logger = CSVLogger('training_set_lstmanalysis.csv',separator=',', append=False)
    model.fit(X_train, y_train, batch_size=128, nb_epoch=10,validation_split=0.33, shuffle=True,callbacks=[checkpointer,csv_logger])
    model.save("coomplemodel.hdf5")
    score, acc = model.evaluate(X_test, y_test, batch_size=32)
    print('Test score:', score)
    print('Test accuracy:', acc)


def main():
    
    lgs=TL()
    filename = 'finalized_model_static.sav'
    pickle.dump(lgs, open(filename, 'wb'))

    vectorizer, lgs1  = TL1()
    
    #checking some random URLs. The results come out to be expected. The first two are okay and the last four are malicious/phishing/bad
    
    X_predict = ['wikipedia.com','google.com/search=faizanahad','pakistanifacebookforever.com/getpassword.php/','www.radsport-voggel.de/wp-admin/includes/log.exe','ahrenhei.without-transfer.ru/nethost.exe','www.itidea.it/centroesteticosothys/img/_notes/gum.exe']
    
    X_predict = vectorizer.transform(X_predict)
    
    y_Predict = lgs1.predict(X_predict)
    
    print(y_Predict)
    
    cnnlstm()

if __name__ == '__main__':
    main()

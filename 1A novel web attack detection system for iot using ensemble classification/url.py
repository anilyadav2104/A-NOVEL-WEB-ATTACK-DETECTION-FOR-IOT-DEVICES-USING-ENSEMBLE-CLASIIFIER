from tkinter import messagebox
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from collections import Counter
import tkinter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import random
import sys
import os
import math
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from xgboost.sklearn import XGBClassifier
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

main = tkinter.Tk()
main.title("A novel web attack detection system for IOT using assemble classification")
main.geometry("1300x1200")
global X_train, X_test, y_train, y_test
global filename
global logit, logit_acc
global xgb, xgb_acc
global model, cnn_acc
global vectorizer

def entropy(s):
    p, lns = Counter(s), float(len(s))
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())

def getTokens(input):
    tokensBySlash = str(input.encode('utf-8')).split('/')  # get tokens after splitting by slash
    allTokens = []
    for i in tokensBySlash:
        tokens = str(i).split('-')  # get tokens after splitting by dash
        tokensByDot = []
        for j in range(0, len(tokens)):
            tempTokens = str(tokens[j]).split('.')  # get tokens after splitting by dot
            tokensByDot = tokensByDot + tempTokens
        allTokens = allTokens + tokens + tokensByDot
    allTokens = list(set(allTokens))  # remove redundant tokens
    if 'com' in allTokens:
        allTokens.remove('com')  # removing .com since it occurs a lot of times and it should not be included in our features
    return allTokens


def runLogit():
    global X_train, X_test, y_train, y_test
    global logit, logit_acc
    data = pd.read_csv(filename, ',', error_bad_lines=False)  # reading file
    data = pd.DataFrame(data)  # converting to a dataframe
    print(data.shape)

    data = np.array(data)  # converting it into an array
    random.shuffle(data)  # shuffling

    x = data[1:, 2:]
    y = data[1:, 1]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)  # split into training and testing set 80/20 ratio

    logit = LogisticRegression()  # using logistic regression
    logit.fit(X_train, y_train)
    y_pred = logit.predict(X_test)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(cnf_matrix)
    text.insert(END,f"Confusion_matrix: {cnf_matrix}\n")

    # Display shapes and accuracy in the text box
    text.insert(END, f"X_train Shape: {X_train.shape}\n")
    text.insert(END, f"X_test Shape: {X_test.shape}\n")
    text.insert(END, f"y_train Shape: {y_train.shape}\n")
    text.insert(END, f"y_test Shape: {y_test.shape}\n")
    logit_acc = logit.score(X_test, y_test) * 100
    text.insert(END, f"Logistic Regression Algorithm Accuracy is: {logit_acc:.2f}%\n")
global allurls
allurls = 'C:\Users\anil2\Downloads\1A novel web attack detection system for iot using ensemble classification\dataset\major.combined.csv'
def logit1():
    allurls = 'C:\Users\anil2\Downloads\1A novel web attack detection system for iot using ensemble classification\dataset\major.combined.csv'  # path to our all urls file
    allurlscsv = pd.read_csv(allurls, ',', error_bad_lines=False)  # reading file
    allurlsdata = pd.DataFrame(allurlscsv)  # converting to a dataframe
    
    allurlsdata = np.array(allurlsdata)  # converting it into an array
    random.shuffle(allurlsdata)  # shuffling

    y = [d[1] for d in allurlsdata]  # all labels 
    corpus = [d[0] for d in allurlsdata]  # all urls corresponding to a label (either good or bad)
    vectorizer = TfidfVectorizer(tokenizer=getTokens, max_features=50)  # get a vector for each url but use our customized tokenizer
    X = vectorizer.fit_transform(corpus)  # get the X vector

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # split into training and testing set 80/20 ratio

    lgs = LogisticRegression()  # using logistic regression
    lgs.fit(X_train, y_train)
    y_pred = lgs.predict(X_test)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(cnf_matrix)
    print(lgs.score(X_test, y_test))  # print the score. It comes out to be 98%
    return vectorizer, lgs

def predict():
    global model, logit, xgb
    global vectorizer
    file = open("./test", 'r')
    pred = [line.strip() for line in file.readlines()]
    print(pred)
    urls = pred[0].split(',')
    data = pd.read_csv(filename, ',', error_bad_lines=False)  # reading file
    data = pd.DataFrame(data)  # converting to a dataframe
    text.insert(END, 'Data Description:' + str(data.info()) + "\n")
    text.insert(END, 'Data Information:' + str(data.head()) + "\n")
    text.insert(END, 'Data shape:' + str(data.shape) + "\n")
    text.insert(END, 'Data describe:' + str(data.describe()) + "\n")
    data = np.array(data)  # converting it into an array
    random.shuffle(data)  # shuffling
    y = [d[1] for d in data]  # all labels 
    corpus = [d[0] for d in data]  # all urls corresponding to a label (either good or bad)
    vectorizer = TfidfVectorizer(tokenizer=getTokens, ngram_range=(1, 2), binary=True, max_features=50)  # get a vector for each url but use our customized tokenizer
    X = vectorizer.fit_transform(corpus)  # get the X vector
    print(X.shape)
    
    predict = vectorizer.transform(urls)
    
    # Instantiate logistic regression model
    vectorizer, lgs = logit1()
    
    # Predict using logistic regression model
    predlgs = lgs.predict(predict)
    
    for url in urls:
        print(url)
        req = Request("http://" + url)
        try:
            response = urlopen(req)
        except HTTPError as e:
            print('The server couldn\'t fulfill the request.')
            print('Error code: ', e.code)
        except URLError as e:
            print('We failed to reach a server.')
            print('Reason: ', e.reason)
        else:
            print('Website is working fine')
            text.insert(END,'Website is working fine')
            
    print(predlgs)
    text.insert(END, predlgs)


def upload():
    global filename
    text.delete('1.0', END)
    filename = askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    text.insert(END, "Dataset loaded\n\n")

def graph():
    height = [logit_acc, xgb_acc]
    bars = ('LogisticRegression', 'XGBoost')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='Ensemble Machine Learning Model for Phishing Intrusion Detection and Classification from URLs')
title.config(bg='light salmon', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0, y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Dataset", command=upload)
upload.place(x=700, y=100)
upload.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='dark orchid', fg='white')  
pathlabel.config(font=font1)
pathlabel.place(x=700, y=150)

lgt = Button(main, text="LogisticRegression", command=runLogit)
lgt.place(x=700, y=200)
lgt.config(font=font1)



graph = Button(main, text="Performance Evaluation", command=graph)
graph.place(x=700, y=300)
graph.config(font=font1)

pred = Button(main, text="URLs Phishing Detection", command=predict)
pred.place(x=700, y=350)
pred.config(font=font1)

font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=80)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=100)
text.config(font=font1)

def runXgboost():
    global filename
    global X_train, X_test, y_train, y_test
    global xgb, xgb_acc

    data = pd.read_csv(filename, ',', error_bad_lines=False)  # reading file
    data = pd.DataFrame(data)  # converting to a dataframe

    data = np.array(data)  # converting it into an array
    random.shuffle(data)  # shuffling

    y = data[1:, 1]
    # Convert 'B' to 0 and 'M' to 1
    y = np.where(y == 'B', 0, 1)

    corpus = [str(doc) for doc in data[1:, 2:]]  # Convert each array element to string
    vectorizer = TfidfVectorizer(tokenizer=getTokens, ngram_range=(1, 2), binary=True, max_features=50)
    X = vectorizer.fit_transform(corpus)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(cnf_matrix)
    xgb_acc = metrics.accuracy_score(y_test, y_pred) * 100
    print("XGBoost Algorithm Accuracy is: ", xgb_acc)

    # Display shapes and accuracy in the text box
    text.insert(END, f"X_train Shape: {X_train.shape}\n")
    text.insert(END, f"X_test Shape: {X_test.shape}\n")
    text.insert(END, f"y_train Shape: {y_train.shape}\n")
    text.insert(END, f"y_test Shape: {y_test.shape}\n")
    text.insert(END, f"XGBoost Algorithm Accuracy is: {xgb_acc:.2f}%\n")

# Add XGBoost button
xgb_btn = Button(main, text="XGBoost", command=runXgboost)
xgb_btn.place(x=700, y=250)
xgb_btn.config(font=font1)


main.config(bg='turquoise')
main.mainloop()

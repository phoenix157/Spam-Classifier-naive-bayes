# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import os
import io
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Creating functions to read data
def readData(path):
    for root, dirsnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)
            
            body = False
            lines = []
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                print(line)
                if body:
                    lines.append(line)
                elif line == '\n':
                    body = True
            print(lines)
            f.close()
            message = '\n'.join(lines)
            yield path, message

def getData(path, classification):
    rows = []
    for filename, message in readData(path):
        rows.append({'message': message, 'class': classification})

    return DataFrame(rows)

def readFromData(data_train):
    rows = []
    data_train_message = data_train['message'].values
    data_train_class = data_train['class'].values
    for i,j in zip(data_train_message, data_train_class):
        rows.append({'message': i, 'class': j})        
    return DataFrame(rows)
    
# data = DataFrame({'message': [], 'class': []})
dir = os.path.dirname(os.path.realpath('__file__'))

data_spam = DataFrame({'message': [], 'class': []})
data_spam = data_spam.append(getData(os.path.join(dir , 'emails','spam'), 'spam'))

data_ham = DataFrame({'message': [], 'class': []})
data_ham = data_ham.append(getData(os.path.join(dir , 'emails','ham'), 'ham'))

data_spam_train, data_spam_test = train_test_split(data_spam, test_size = 0.2)
data_ham_train, data_ham_test = train_test_split(data_ham, test_size = 0.2)

# data = data.append(getData(os.path.join(dir , 'emails','spam'), 'spam'))
# data = data.append(getData(os.path.join(dir , 'emails','ham'), 'ham'))

# Join data
data_train = DataFrame({'message': [], 'class': []})
data_train = data_train.append(readFromData(data_spam_train))
data_train = data_train.append(readFromData(data_ham_train))

data_test = DataFrame({'message': [], 'class': []})
data_test = data_test.append(readFromData(data_spam_test))
data_test = data_test.append(readFromData(data_ham_test))

# Initialize classifier and vectorizer
classifier = MultinomialNB()
vectorizer = CountVectorizer()

# Train data
counts = vectorizer.fit_transform(data_train['message'].values)
targets = data_train['class'].values
classifier.fit(counts, targets)

# Predict Data and confusion matrix
predictions = classifier.predict(vectorizer.transform(data_test['message'].values))
cm = confusion_matrix(data_test['class'].values, predictions)
print("Confusion Matrix:")
print(cm)

accuracy = accuracy_score(data_test['class'].values, predictions)
print("Accuracy:")
print(accuracy)

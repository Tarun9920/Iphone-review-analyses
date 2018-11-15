# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:05:45 2018

@author: tarun
"""

import json
import csv
import nltk
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


df= pd.read_csv('Data.csv')

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Doing cleaning for all the text
corpus = []
for i in range(0, 16):
    review= re.sub('[^a-zA-z]',' ',df['text'][i])
    review= review.lower()
    review= review.split()
    review = [word for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y_in= df['class']
y = y_in[pd.notnull(y_in)]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
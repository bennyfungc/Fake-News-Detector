# Script for fake news detector project

# Project writeup and script found in the .ipynb Jupyter Notebook file.
# Writeup can also be viewed from README.md. 

# Import packages: numpy, pandas, sklearn
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Import data and take a look at first 5 items
dataframe = pd.read_csv('./news.csv')

dataframe.shape
dataframe.head()

# Check labels from the data
labels = dataframe.label
labels.head()

# Separate training data from testing data
x_train, x_test, y_train, y_test = train_test_split(dataframe['text'], labels, test_size=0.2, random_state=7)

# Initialize a TfidfVectorizer and fit and transform the train set and
# transform the test set
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

# Initialize a PassiveAggressiveClassifier and fit the weighted matrix and test matrix
# Predict on test set and calculate accuracy.
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100, 2)}%')

# Show confusion matrix
confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
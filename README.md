# Fake News Detector

The goal of this project is to learn methods of categorizing data by identifying useful attributes from our data in order to fit the data to a model. Specifically, we will be using the following numerical statistic to fit a Passive-Aggressive classification model:

**Term Frequency (TF)** : The number of times a term occurs in a document. A higher value means a term occurs more often than others.

**Inverse Document Frequency (IDF)**: Measure of how irrelevant a term is if it occurs too often across many other documents.


Using TF will allow our Passive-Aggressive Classifier to passively learn to classify articles based on occurrences of specific keywords to determine real news. The IDF will allow the model to make corrections in future predictions with further classification of specific keyterms with negative feedback between different news articles.

1. Import necessary packages from numpy, pandas, and sklearn.

2. Read the data into a Dataframe, get the shape of the data and the first 5 records to get a feel of the data we are working with.

3. Get labels from the data

4. Separate training data from testing data.

5. Initialize a TfidfVectorizer with English stop words with a document frequency of 0.7 to process our data set into a matrix of TF-IDF features. Fit and transform the vectorizer on the train set and transform the test set.

6. Initialize a PassiveAggressiveClassifier and fit this on tfidf_train and y_train. Then, predict on the test set from the TfidfVectorizer and calculate the accuracy using accuracy_score().

7. Print out a confusion matrix to gain insight into number of true and false negatives and positives.

Our model was able to identify 594 true positives, 587 true negatives, 42 false positives, and 44 false negatives.

---

*This is a project intended for educational purposes. Project guided by* [Data Flair](https://data-flair.training/blogs/advanced-python-project-detecting-fake-news/)
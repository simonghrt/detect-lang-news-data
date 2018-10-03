#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
In this file, we'll apply several machine learning techniques to this problem
"""


import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


"""
Train and test classifier and embedding method
Embedding / Word representation : TfIdf
Classifier : Logistic Regression 
"""
def train_test_tfidf_logistic():
    # Training dataset
    data_train = preprocessing.prepare_train_dataset()
    X_train = data_train['content'].values
    Y_train = data_train['lang'].values

    # Fitting TfIdf
    vectorizer = TfidfVectorizer();
    vect_train = vectorizer.fit_transform(X_train)

    # Fitting Logistic Regression Classifier
    clf = LogisticRegression(C = 1)
    clf.fit(vect_train, Y_train)
    
    # Testing dataset
    data_test = preprocessing.prepare_test_dataset()
    X_test = data_test['content'].values
    Y_test = data_test['lang'].values
    vect_test = vectorizer.transform(X_test)

    # Printing scores
    print("Training score :")
    print(clf.score(vect_train, Y_train))
    print("Testing score :")
    print(clf.score(vect_test, Y_test))
    Y_test_pred = clf.predict(vect_test)
    print("Classification report :")
    print(classification_report(Y_test, Y_test_pred))
    # TODO Print confusion matrix

    # Test of one prediction
    print("Sentence to predict :")
    sentence = "Hello my name is Bob and I live in America"
    print(sentence)
    vect = vectorizer.transform([sentence])
    print(clf.predict(vect))


"""
Train and test classifier and embedding method
Embedding / Word representation : TfIdf
Classifier : Logistic Regression with Grid Search 
"""
def train_test_tfidf_logistic_with_gridsearch():
    # Training dataset
    data_train = preprocessing.prepare_train_dataset()
    X_train = data_train['content'].values
    Y_train = data_train['lang'].values

    # Fitting TfIdf
    vectorizer = TfidfVectorizer();
    vect_train = vectorizer.fit_transform(X_train)

    # Fitting Logistic Regression Classifier
    params = {'C': [0.0001, 0.1, 1, 10, 1000, 100000]}
    lg = LogisticRegression()
    clf = GridSearchCV(lg, params)
    clf.fit(vect_train, Y_train)
   
    # Print best param
    print("Best parameters with grid search")
    print(clf.best_params_)
    print("Grid scores on test set")
    means = clf.cv_results_['mean_test_score']
    for mean, param in zip(means, clf.cv_results_['params']):
        print("%0.3f for %r"
              % (mean, param))

    # Testing dataset
    data_test = preprocessing.prepare_test_dataset()
    X_test = data_test['content'].values
    Y_test = data_test['lang'].values
    vect_test = vectorizer.transform(X_test)

    # Printing scores
    print("Training score :")
    print(clf.score(vect_train, Y_train))
    print("Testing score :")
    print(clf.score(vect_test, Y_test))
    Y_test_pred = clf.predict(vect_test)
    print("Classification report :")
    print(classification_report(Y_test, Y_test_pred))
    # TODO Print confusion matrix

    # Test of one prediction
    print("Sentence to predict :")
    sentence = "Hello my name is Bob and I live in America"
    print(sentence)
    vect = vectorizer.transform([sentence])
    print(clf.predict(vect))


train_test_tfidf_logistic_with_gridsearch()

#!/usr/bin/env python
# coding: utf-8

#modules for 
import pandas as pd
import numpy as np

#module for splitiing data set
from sklearn.model_selection import train_test_split

#modules for Regression models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

#models for Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#modules for regression metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

#modules for classification metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

#module for grid search
from sklearn.model_selection import GridSearchCV

#module for writing into csv
import csv

def print_metrics(y, yhat, model, name, first = False):
    
    #printing results of the model
    print(f'Results for {model}')    
    print(f'MAE: {mean_absolute_error(y, yhat)}')
    print(f'MSE: {mean_squared_error(y, yhat)}')
    print(f'RMSE: {np.sqrt(mean_squared_error(y, yhat))}')
    print(f'R2: {r2_score(y, yhat)}')
    
    row = [model, round(r2_score(y, yhat), 2), round(np.sqrt(mean_squared_error(y, yhat)), 2)
                   , round(mean_absolute_error(y, yhat), 2)]
    
    file_name = name + '_metric.csv'
    
    if first:
        mode = 'w'
    else:
        mode = 'a'
    
    with open(file_name, mode, newline = '') as file:
        writer = csv.writer(file)
        if first:
            header = ['model', 'accuracy', 'weightedRecall', 'weightedPrecision']
            writer.writerow(header)
        writer.writerow(row)

def fit_regression_models(train_X, train_y, test_X, test_y, name):
    
    #Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(train_X, train_y)
    predictions = lr_model.predict(test_X)
    #printing metrics and writing to file
    print_metrics(test_y, predictions, 'Linear Regression', name, first = True)
    
    #Decision Tree Regression
    dt_model = DecisionTreeRegressor(random_state = 7)
    dt__model = GridSearchCV(dt_model, param_grid = {'max_depth': [5, 10, 15]}, cv = 3)
    dt__model.fit(train_X, train_y)
    predictions = dt__model.predict(test_X)
    #printing metrics and writing to file
    print_metrics(test_y, predictions, 'Decision Tree', name)
    
    #Random Forest Regression
    rf_model = RandomForestRegressor(random_state = 7)
    rf__model = GridSearchCV(rf_model, param_grid = {'max_depth': [5, 10, 15], 'n_estimators': [10, 15, 20]}, cv = 3)
    rf__model.fit(train_X, train_y)
    predictions = rf__model.predict(test_X)
    #printing metrics and writing to file
    print_metrics(test_y, predictions, 'Random Forest', name)
    
    #Gradient Boosting Regression
    gb_model = GradientBoostingRegressor(random_state = 7)
    gb_model.fit(train_X, train_y)
    predictions = gb_model.predict(test_X)
    #printing metrics and writing to file
    print_metrics(test_y, predictions, 'Gradient Boosted Trees', name)

def get_metrics_classification(y, yhat, model, name, first = False):
    
    #printing metrics of the model
    print(classification_report(y, yhat))
    
    row = [model, round(accuracy_score(y, yhat), 2)
                  , round(recall_score(y, yhat, average = 'weighted'), 2)
                   , round(precision_score(y, yhat, average = 'weighted'), 2)]
    
    file_name = name + '_metric.csv'
    
    if first:
        mode = 'w'
    else:
        mode = 'a'
    
    with open(file_name, mode, newline = '') as file:
        writer = csv.writer(file)
        if first:
            header = ['model', 'accuracy', 'weightedRecall', 'weightedPrecision']
            writer.writerow(header)
        writer.writerow(row)

def fit_classification_models(train_X, train_y, test_X, test_y, name):
    
    #Logistic Regression
    lr_model = LogisticRegression()
    lr_model.fit(train_X, train_y)
    prediction = lr_model.predict(test_X)
    #printing metrics and writing to file
    get_metrics_classification(test_y, prediction, 'Logistic Regression', name, first = True)
    
    #Decision Tree Classification
    dtc_model = DecisionTreeClassifier(random_state = 7)
    dtc__model = GridSearchCV(dtc_model, param_grid = {'max_depth': [5, 10, 15]}, cv = 3)
    dtc__model.fit(train_X, train_y)
    prediction = dtc__model.predict(test_X)
    #printing metrics and writing to file
    get_metrics_classification(test_y, prediction, 'Decision Tree', name)
    
    #Random Forest Classification
    rfc_model = RandomForestClassifier()
    rfc__model = GridSearchCV(rfc_model, param_grid = {'max_depth': [5, 10, 15], 'n_estimators': [10, 15, 20]}, cv = 3)
    rfc__model.fit(train_X, train_y)
    prediction = rfc__model.predict(test_X)
    #printing metrics and writing to file
    get_metrics_classification(test_y, prediction, 'Random Forest', name)
    
    #One vs Rest Classification
    ovr_model = OneVsRestClassifier(SVC())
    ovr_model.fit(train_X, train_y)
    prediction = ovr_model.predict(test_X)
    #printing metrics and writing to file
    get_metrics_classification(test_y, prediction, 'One vs Rest', name)
    
    #Naive Bayes Clasiification
    nbc_model = GaussianNB()
    nbc__model = GridSearchCV(nbc_model, param_grid = {'var_smoothing': [.5, 1, 2]}, cv = 3)
    nbc__model.fit(train_X, train_y)
    prediction = nbc__model.predict(test_X)
    #printing metrics and writing to file
    get_metrics_classification(test_y, prediction, 'Naive Bayes', name)
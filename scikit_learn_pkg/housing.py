#!/usr/bin/env python
# coding: utf-8

#modules to ignore warnings
import sys
import warnings

#Importing modules to handle data
import pandas as pd

#module for splitting data into test and train
from sklearn.model_selection import train_test_split

#regression models 
from utils import fit_regression_models

#module for summarizing data
from utils import summarize_data

#Disabling Warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

df = pd.read_csv('data/housing/Boston.csv', header = None)

#dropping index column
df.drop(0, axis = 1, inplace = True)

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

#Sample of the data
print(df.head())

summarize_data(df, ['CHAS', 'RAD'], ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'TAX', 'PTRATIO', 'B', 'LSTAT'], 'MEDV')

#splitting data into test and train
train, test = train_test_split(df, test_size=0.25, random_state = 7)

train_X = train[train.columns[:-1]]
train_y = train['MEDV']
test_X = test[train.columns[:-1]]
test_y = test['MEDV']

#fitting regression models
fit_regression_models(train_X, train_y, test_X, test_y, 'scikit_learn_pkg/metrics/housing')
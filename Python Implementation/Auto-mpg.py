#!/usr/bin/env python
# coding: utf-8


#modules to ignore warnings
import sys
import warnings

#Importing modules to handle data
import pandas as pd

#module for splitting data into test and train
from sklearn.model_selection import train_test_split

#module for summarizing data
from utils import summarize_data

#regression models 
from utils import fit_regression_models

#Disabling Warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

df = pd.read_csv('auto-mpg.csv', header = None)

df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']

#filling missing values with mean
df['horsepower'].fillna(value = df['horsepower'].mean(), inplace = True)

print(df.head(), end = '\n\n')

summarize_data(df, ['cylinders', 'model_year', 'origin'], ['displacement', 'horsepower', 'weight', 'acceleration'], 'mpg')

#dropping car name column
df.drop('car_name', axis = 1, inplace = True)

#one hot encoding
df = pd.concat([df, pd.get_dummies(df['cylinders']), pd.get_dummies(df['origin']), pd.get_dummies(df['model_year'])], axis = 1)
df.drop(['cylinders', 'origin', 'model_year'], axis = 1, inplace = True)

#splitting data into test and train
train, test = train_test_split(df, test_size=0.25, random_state = 7)

train_X = train[train.columns[1:]]
train_y = train['mpg']
test_X = test[train.columns[1:]]
test_y = test['mpg']

#fitiing regression models
fit_regression_models(train_X, train_y, test_X, test_y, 'Auto-mpg')
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

#Two different datasets for red wine and white wine
red_df = pd.read_csv('data/wine/winequality-red.csv', sep = ';')
white_df = pd.read_csv('data/wine/winequality-white.csv', sep = ';')

red_df['color'] = 'red'
white_df['color'] = 'white'

#combining red wine data and white wine data
df = pd.concat([red_df, white_df], ignore_index=True)

print(df.head())

summarize_data(df, ['color'], ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'], 'quality')

#one hot encoding
df = pd.concat([df, pd.get_dummies(df['color'])], axis = 1)
df.drop('color', axis = 1, inplace = True)

#predictor variables
independent_variables = list(df.columns)

#quality is the predicted variable
independent_variables.remove('quality')

#splitting data into test and train
train, test = train_test_split(df, test_size=0.25, random_state = 7)

train_X = train[independent_variables]
train_y = train['quality']
test_X = test[independent_variables]
test_y = test['quality']

#fitting regression models
fit_regression_models(train_X, train_y, test_X, test_y, 'scikit_learn_pkg/metrics/wine')
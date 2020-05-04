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

#classification models
from utils import fit_classification_models

#Disabling Warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

iris_df = pd.read_csv('data/iris/iris.csv', header = None)

iris_df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

print(iris_df.head())

classes = summarize_data(iris_df, [], ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], 'species', 'classification')

#splitting data into test and train
iris_train, iris_test = train_test_split(iris_df, test_size=0.25, random_state = 7)

iris_train_X = iris_train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
iris_train_y = iris_train['species']
iris_test_X = iris_test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
iris_test_y = iris_test['species']

#fitiing classification models
fit_classification_models(iris_train_X, iris_train_y, iris_test_X, iris_test_y, 'scikit_learn_pkg/metrics/iris', classes)
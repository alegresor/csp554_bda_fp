#!/usr/bin/env python
# coding: utf-8

#Importing modules to handle data
import pandas as pd

#module for splitting data into test and train
from sklearn.model_selection import train_test_split

#classification models
from utils import fit_classification_models

iris_df = pd.read_csv('iris.csv', header = None)

iris_df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

#splitting data into test and train
iris_train, iris_test = train_test_split(iris_df, test_size=0.25, random_state = 7)

iris_train_X = iris_train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
iris_train_y = iris_train['species']
iris_test_X = iris_test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
iris_test_y = iris_test['species']

#fitiing classification models
fit_classification_models(iris_train_X, iris_train_y, iris_test_X, iris_test_y, 'Iris')
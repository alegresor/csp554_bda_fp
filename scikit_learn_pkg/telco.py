#!/usr/bin/env python
# coding: utf-8

#modules to ignore warnings
import sys
import warnings

#Importing modules to handle data
import pandas as pd
import numpy as np

#module for splitting data into test and train
from sklearn.model_selection import train_test_split

#module for summarizing data
from utils import summarize_data

#classification models 
from utils import fit_classification_models

#Disabling Warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

df = pd.read_csv('data/telco/WA_Fn-UseC_-Telco-Customer-Churn.csv')

#Conerting text column to float
df.loc[df['TotalCharges'] == ' ', 'TotalCharges'] = np.nan
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

print(df.head())

classes = summarize_data(df, ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 
	'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV','StreamingMovies', 'Contract', 'PaperlessBilling', 
	'PaymentMethod'], ['tenure', 'MonthlyCharges', 'TotalCharges'], 'Churn', 'classification')

#Converting text to integers in columns
df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
df['PhoneService'] = df['PhoneService'].map({'Yes': 1, 'No': 0})
df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
df['MultipleLines'] = df['MultipleLines'].map({'Yes': 1, 'No': 0, 'No phone service': -1})
df['OnlineSecurity'] = df['OnlineSecurity'].map({'Yes': 1, 'No': 0, 'No internet service': -1})
df['OnlineBackup'] = df['OnlineBackup'].map({'Yes': 1, 'No': 0, 'No internet service': -1})
df['DeviceProtection'] = df['DeviceProtection'].map({'Yes': 1, 'No': 0, 'No internet service': -1})
df['TechSupport'] = df['TechSupport'].map({'Yes': 1, 'No': 0, 'No internet service': -1})
df['StreamingTV'] = df['StreamingTV'].map({'Yes': 1, 'No': 0, 'No internet service': -1})
df['StreamingMovies'] = df['StreamingMovies'].map({'Yes': 1, 'No': 0, 'No internet service': -1})
df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['Contract'] = df['Contract'].map({'Month-to-month': 1, 'One year': 2, 'Two year': 3})

#one hot encoding
df = pd.concat([df, pd.get_dummies(df['gender']), pd.get_dummies(df['InternetService']), pd.get_dummies(df['PaymentMethod'])], axis = 1)
df.drop(['gender', 'InternetService', 'PaymentMethod'], axis = 1, inplace = True)

#filling missing values with mean
df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace = True)

#dropping primary key column
df.drop('customerID', axis = 1, inplace = True)

#splitting data into test and train
churn_train, churn_test = train_test_split(df, test_size=0.25, random_state = 7)

churn_train_X = churn_train
churn_train_y = churn_train['Churn']
churn_test_X = churn_test
Churn_test_y = churn_test['Churn']
churn_train_X.drop('Churn', axis = 1, inplace = True)
churn_test_X.drop('Churn', axis = 1, inplace = True)

#fitiing classification models
fit_classification_models(churn_train_X, churn_train_y, churn_test_X, Churn_test_y, 'scikit_learn_pkg/metrics/telco', classes)
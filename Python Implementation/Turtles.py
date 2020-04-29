#!/usr/bin/env python
# coding: utf-8

#Importing modules to handle data
import pandas as pd
import numpy as np

#classification models 
from utils import fit_classification_models

#module for splitting data into test and train
from sklearn.model_selection import train_test_split

df = pd.read_csv('turtles.csv')

# select subset of columns
df = df[['Species', 'Dead_Alive', 'Gear', 'SCL_notch', 'SCL_tip', 'SCW', 'CCL_notch', 'TestLevel_Before', 'Entangled']]
#'Circumference', # has 13k na values and only 63 actual entries
#'Girth', # has 11k na value and only 2k actual entries
#'Depth_mid', # has 13k na values and only 181 actual entries
#'Tail', # has 13k na values and only 8 actual values
#'Weight', # has 11k na values and only 2k actual entries
#'Cap_Region', # 11k inshore and only 141 offshore
#'TestLevel_After', # 12k na values and only 1167 entries

# drop any record without a species label
df.dropna(subset=['Species'],inplace=True)

# other species have < 10 total tags
df = df.loc[df['Species'].isin(['Loggerhead','Green','Kemps_Ridley'])]

# correct capitalization
df['Dead_Alive'].replace({'Alive':'alive','Dead':'dead'},inplace=True)

# categories with < 10 captures --> 'other' label
df['Gear'].replace(
    {cat:'Other' for cat in ['Shrimp trawl','General public sighting','Channel net','Headstart','Flounder trawl']},
    inplace=True)

for col in ['SCL_notch','SCL_tip','SCW','CCL_notch','TestLevel_Before']:
    # 0 is put in instead of na --> correct to na
    df[col].replace({0:np.nan},inplace=True)
    
    # get mean by species
    means = df.groupby('Species')[col].transform('mean') 
    
    # fill na with species mean
    df[col].fillna(means,inplace=True)

# make object from boolean
df['Entangled'].replace({True:'entangled',False:'free'},inplace=True)

for col in df.columns:
    print (f'col: {col}')
    col_dtype = df[col].dtype
    print (f'dtype: {col_dtype}')
    print (f'na values: {df[col].isnull().sum()}')
    if col_dtype == int or col_dtype==float:
        print(df[col].describe())
    else:
        print(df[col].value_counts())
    print ('\n')

df.head()

df.to_csv('turtles_clean.csv',index=False)

#Converting text to integers in columns
df['Dead_Alive'] = df['Dead_Alive'].map({'alive': 1, 'dead': 0})
df['Entangled'] = df['Entangled'].map({'free': 1, 'entangled': 0})

#one hot encoding
df = pd.concat([df, pd.get_dummies(df['Gear'])], axis = 1)
df.drop('Gear', axis = 1, inplace = True)

#splitting data into test and train
train, test = train_test_split(df, test_size=0.25, random_state = 7)

train_X = train[train.columns[1:]]
train_y = train['Species']
test_X = test[train.columns[1:]]
test_y = test['Species']

#fitiing classification models
fit_classification_models(train_X, train_y, test_X, test_y, 'Turtles')
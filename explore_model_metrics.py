''' Explore PySpark MLlib Models Metrics on Iris and Turtles Datasets '''

import numpy as np
import pandas as pd 
import seaborn as sns 
sns.set(style='whitegrid')

def multiple_bar_plot(df,dataset):
    df = pd.melt(df,id_vars='model',var_name='metric')
    plt = sns.catplot(x='metric',y='value',hue='model',data=df,kind='bar')
    plt.set(title='Model Metrics for %s Dataset'%dataset,ylabel='metric',xlabel='')
    #plt.despine()
    plt.set_xticklabels(rotation=45)
    plt.savefig('out/bar_chart_metrics/%s.png'%dataset.lower())


datasets = ['Auto','Housing','Wine','Iris','Telco','Turtles']
for dataset in datasets:
    name = dataset.lower()
    df = pd.read_csv('data/%s/%s_model_metrics.csv'%(name,name))
    multiple_bar_plot(df,dataset)

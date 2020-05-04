''' Explore PySpark MLlib Models Metrics '''

import numpy as np
import pandas as pd 
import seaborn as sns 
sns.set(style='whitegrid')

def multiple_bar_plot(tool,dataset,df):
    df = pd.melt(df,id_vars='model',var_name='metric')
    plt = sns.catplot(x='metric',y='value',hue='model',data=df,kind='bar')
    plt.set(title='%s Model Metrics for %s Dataset'%(tool,dataset),ylabel='metric',xlabel='')
    plt.despine(left=True)
    plt.set_xticklabels(rotation=45)
    plt.savefig('%s_pkg/out/%s.png'%(tool.lower(),dataset.lower()))

tools = ['Spark','Scikit_Learn','R']
datasets = ['Auto','Housing','Wine','Iris','Telco','Turtles']
for tool in tools:
    for dataset in datasets:
        df = pd.read_csv('%s_pkg/metrics/%s_metric.csv'%(tool.lower(),dataset.lower()))
        multiple_bar_plot(tool,dataset,df)

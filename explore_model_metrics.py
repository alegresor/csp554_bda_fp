''' Explore PySpark MLlib Models Metrics on Iris and Turtles Datasets '''

import numpy as np
import pandas as pd 
from matplotlib import pyplot

df_iris = pd.read_csv('data/iris/iris_model_metrics.csv')
df_turtles = pd.read_csv('data/turtles/turtle_model_metrics.csv')

def multiple_bar_plot(df,dataset):
    n = len(df.columns)-1
    idx = np.arange(n)
    width = .25
    fig,ax = pyplot.subplots()
    for i in range(len(df)):
        vals = df.iloc[i,1:]
        ax.bar(idx+i*width,vals,width,label=df.iloc[i,0])
    for spine in ['top','right','left','bottom']: ax.spines[spine].set_visible(False)
    ax.set_xticks(idx+width)
    ax.set_xticklabels(list(df.columns)[1:])
    ax.legend(frameon=False,loc=(0,1),ncol=len(df))
    fig.suptitle('%s Dataset MLlib Model Metrics'%dataset)
    pyplot.savefig('out/metrics_bar_chart.%s.png'%dataset.lower(),dpi=200)

multiple_bar_plot(df_iris,'Iris')
multiple_bar_plot(df_turtles,'Turtles')
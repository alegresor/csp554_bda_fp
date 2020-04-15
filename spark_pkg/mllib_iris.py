'''
MLlib models for iris dataset
command to create log: python spark_pkg/mllib_iris.py > spark_pkg/logs/mllib_iris.log
'''

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from spark_pkg.mllib_util import encode_classifier_data, model

# setup
spark = SparkSession.builder.master("local").appName('iris-app').getOrCreate()
spark.sparkContext.setLogLevel('ERROR') # only show pyspark errors

# read data
struct = StructType([
    StructField('sepal_length', FloatType(), True),
    StructField('sepal_width', FloatType(), True),
    StructField('petal_length', FloatType(), True),
    StructField('petal_width', FloatType(), True),
    StructField('species', StringType(), True)])
df = spark.read.schema(struct).csv('data/iris/iris.csv',header=False)
df.printSchema()
df.show(5)

# one hot encoding
df = encode_classifier_data(df, 
    label_col = 'species',
    categorical_cols = [],
    numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

# split into train and test datsets
train,test = df.randomSplit([.75,.25], seed=7)

# modeling
model(train,test,'data/iris/iris_model_metrics.csv')
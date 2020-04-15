'''
MLlib models for turtles dataset
command to create log: python spark_pkg/mllib_turtles.py > spark_pkg/logs/mllib_turtles.log
'''

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from spark_pkg.mllib_util import encode_classifier_data, model

# setup
spark = SparkSession.builder.master("local").appName('mllib_turtles').getOrCreate()
spark.sparkContext.setLogLevel('ERROR') # only show pyspark errors

# read data
struct = StructType([    
    StructField('Species', StringType(), True),
    StructField('Dead_Alive', StringType(), True),
    StructField('Gear', StringType(), True),
    StructField('SCL_notch', FloatType(), True),    
    StructField('SCL_tip', FloatType(), True),    
    StructField('SCW', FloatType(), True),    
    StructField('CCL_notch', FloatType(), True),    
    StructField('TestLevel_Before', FloatType(), True),    
    StructField('Entangled', StringType(), True)])
df = spark.read.schema(struct).csv('data/turtles/turtles_clean_spark_sql.csv',header=False)
df.printSchema()
df.show(5)

# one hot encoding
df = encode_classifier_data(df, 
    label_col = 'Species',
    categorical_cols = ['Dead_Alive','Gear','Entangled'],
    numeric_cols = ['SCL_notch', 'SCL_tip', 'SCW', 'CCL_notch','TestLevel_Before'])

# split into train and test datsets
train,test = df.randomSplit([.75,.25], seed=7)

# modeling
model(train,test,'data/turtles/turtle_model_metrics.csv')
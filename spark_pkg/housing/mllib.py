''' MLlib models '''

from pyspark.sql import SparkSession
from spark_pkg.util.sql import summarize_df
import pyspark.sql.functions as f
from pyspark.sql.types import *
from spark_pkg.util.mllib import encode_data, run_regression_models

# setup
spark = SparkSession.builder.master("local").appName('auto').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

# read data
struct = StructType([
    StructField('idx', FloatType(), True),
    StructField('crim', FloatType(), True),
    StructField('zn', FloatType(), True),
    StructField('indus', FloatType(), True),
    StructField('chas', StringType(), True),    
    StructField('nox', FloatType(), True),    
    StructField('rm', FloatType(), True),  
    StructField('age', FloatType(), True),  
    StructField('dis', FloatType(), True),  
    StructField('rad', FloatType(), True),  
    StructField('tax', FloatType(), True),  
    StructField('ptratio', FloatType(), True),  
    StructField('black', FloatType(), True),  
    StructField('lstat', FloatType(), True),      
    StructField('label', FloatType(), True)]) # median housing value
df = spark.read.schema(struct).csv('data/housing/Boston.csv',header=False)
df.printSchema()
df.show(5)

# drop idx, the unique id column 
df = df.drop('idx')

# summarize dataframe
numeric_cols = [col for col,dtype in df.dtypes if dtype=='float']
summarize_df(spark,df,numeric_cols)

# one hot encoding
numeric_cols.remove('label')
df = encode_data(df,
        categorical_cols = ['chas'],
        numeric_cols = numeric_cols,
        predict_col = 'label',
        encode_predict_col = False)

# split into train and test datsets
train,test = df.randomSplit([.75,.25], seed=7)

# modeling
run_regression_models(train,test,'spark_pkg/metrics/housing_metric.csv')

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
    StructField('label', FloatType(), True), # mpg
    StructField('cylinders', StringType(), True),
    StructField('displacement', FloatType(), True),
    StructField('horsepower', FloatType(), True),
    StructField('weight', FloatType(), True),    
    StructField('acceleration', FloatType(), True),    
    StructField('model_year', StringType(), True),  
    StructField('origin', StringType(), True),  
    StructField('car_name', StringType(), True)])
df = spark.read.schema(struct).csv('data/auto/auto-mpg.csv',header=False)
df.printSchema()
df.show(5)

# drop (total 7) cars with 4 or 3 cylinders
df = df.filter((df['cylinders']!='3') & (df['cylinders']!='5'))

# fill missing horsepower values with column mean 
df = df.na.fill(df.na.drop().agg(f.avg('horsepower')).first()[0],'horsepower')

# drop car_name, the unique id column 
df = df.drop('car_name')

# summarize dataframe
numeric_cols = [col for col,dtype in df.dtypes if dtype=='float']
summarize_df(spark,df,numeric_cols)

# one hot encoding
df = encode_data(df,
        categorical_cols = ['cylinders','model_year','origin'],
        numeric_cols = ['displacement','horsepower','weight','acceleration'],
        predict_col = 'label',
        encode_predict_col = False)

# split into train and test datsets
train,test = df.randomSplit([.75,.25], seed=7)

# modeling
run_regression_models(train,test,'data/auto/auto_model_metrics.csv')

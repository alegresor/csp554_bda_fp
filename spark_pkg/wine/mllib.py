''' MLlib models '''

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from spark_pkg.util.mllib import encode_data, run_regression_models

# setup
spark = SparkSession.builder.master("local").appName('mllib_wine').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

# read data
struct = StructType([
    StructField('fixed_acidity', FloatType(), True),
    StructField('volatile_acidity', FloatType(), True),
    StructField('citric_acid', FloatType(), True),
    StructField('residual_sugar', FloatType(), True),    
    StructField('chlorides', FloatType(), True),    
    StructField('free_sulfur_dioxide', FloatType(), True),    
    StructField('total_sulfur_dioxide', FloatType(), True),    
    StructField('density', FloatType(), True),    
    StructField('pH', FloatType(), True),
    StructField('sulphates', FloatType(), True),
    StructField('alcohol', FloatType(), True),
    StructField('label', FloatType(), True), # quality
    StructField('color', StringType(), True)])
df = spark.read.schema(struct).csv('data/wine/wine_clean_spark_sql.csv',header=False)
df.printSchema()
df.show(5)

# one hot encoding
df = encode_data(df,
        categorical_cols = [col for col,dtype in df.dtypes if dtype=='string'],
        numeric_cols = [col for col,dtype in df.dtypes if dtype=='float' and col!='label'],
        predict_col = 'label',
        encode_predict_col = False)

# split into train and test datsets
train,test = df.randomSplit([.75,.25], seed=7)

# modeling
run_regression_models(train,test,'spark_pkg/metrics/wine_metrics.csv')

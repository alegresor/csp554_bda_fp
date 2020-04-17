''' MLlib models for '''

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from spark_pkg.util.mllib import encode_data, run_classification_models

# setup
spark = SparkSession.builder.master("local").appName('mllib_telco').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

# read data
struct = StructType([    
    StructField('gender', StringType(), True),
    StructField('senior', StringType(), True),
    StructField('partner', StringType(), True),
    StructField('dependents', StringType(), True),    
    StructField('tenure', FloatType(), True),    
    StructField('phone_service', StringType(), True),    
    StructField('multiple_lines', StringType(), True),    
    StructField('internet', StringType(), True),    
    StructField('security', StringType(), True),    
    StructField('backup', StringType(), True),    
    StructField('protection', StringType(), True),    
    StructField('support', StringType(), True),    
    StructField('tv', StringType(), True),    
    StructField('movies', StringType(), True),    
    StructField('contract', StringType(), True),    
    StructField('paperless', StringType(), True),    
    StructField('method', StringType(), True),    
    StructField('monthly', FloatType(), True),    
    StructField('churn', StringType(), True),    
    StructField('months', FloatType(), True)])
df = spark.read.schema(struct).csv('data/telco/telco_clean_spark_sql.csv',header=False)
df.printSchema()
df.show(5)

# one hot encoding
df = encode_data(df,
        categorical_cols = [col for col,dtype in df.dtypes if dtype=='string' and col!='churn'],
        numeric_cols = [col for col,dtype in df.dtypes if dtype!='string'],
        predict_col = 'churn',
        encode_predict_col = True)

# split into train and test datsets
train,test = df.randomSplit([.75,.25], seed=7)

# modeling
run_classification_models(train,test,'data/telco/telco_model_metrics.csv',classes=2)
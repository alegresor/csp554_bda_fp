''' MLlib models '''

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from spark_pkg.util.mllib import encode_data, run_classification_models

# setup
spark = SparkSession.builder.master("local").appName('mllib_turtles').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

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
df = encode_data(df,
        categorical_cols = [col for col,dtype in df.dtypes if dtype=='string' and col!='Species'],
        numeric_cols = [col for col,dtype in df.dtypes if dtype=='float'],
        predict_col = 'Species',
        encode_predict_col = True)

# split into train and test datsets
train,test = df.randomSplit([.75,.25], seed=7)

# modeling
run_classification_models(train,test,'data/turtles/turtle_model_metrics.csv',classes=3)
''' MLlib models '''

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from spark_pkg.util.sql import summarize_df
from spark_pkg.util.mllib import encode_data, run_classification_models

# setup
spark = SparkSession.builder.master("local").appName('iris-app').getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

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
summarize_df(spark,df,[col for col,dtype in df.dtypes if dtype=='float' and col])

# one hot encoding
df = encode_data(df,
        categorical_cols = [],
        numeric_cols = [col for col,dtype in df.dtypes if dtype=='float'],
        predict_col = 'species',
        encode_predict_col = True)

# split into train and test datsets
train,test = df.randomSplit([.75,.25], seed=7)

# modeling
run_classification_models(train,test,'spark_pkg/metrics/iris_metric.csv',classes=3)
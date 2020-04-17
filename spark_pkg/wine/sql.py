''' Clean dataset with spark sql '''

from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as f
from spark_pkg.util.sql import summarize_df

# setup
spark = SparkSession.builder.master("local").appName("sql_wine").getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

# read wine datasets
df_red = spark.read.csv('data/wine/winequality-red.csv', header=True, inferSchema=True, sep=';').withColumn('color',f.lit('red'))
df_white = spark.read.csv('data/wine/winequality-white.csv', header=True, inferSchema=True, sep=';').withColumn('color',f.lit('white'))

# combine red and white wine datasets
df = df_red.union(df_white)

# summarize dataframe
numeric_cols = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide',
                'total sulfur dioxide','density','pH','sulphates','alcohol','quality']
summarize_df(spark,df,numeric_cols)

# output dataset
df.write.mode('overwrite').csv('data/wine/wine_clean_spark_sql.csv',header=False)
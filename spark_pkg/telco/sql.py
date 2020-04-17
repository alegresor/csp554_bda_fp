''' Clean dataset with spark sql '''

from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as f
from spark_pkg.util.sql import summarize_df

# setup
spark = SparkSession.builder.master("local").appName("sql_wine").getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

# read dataset
df = spark.read.csv('data/telco/WA_Fn-UseC_-Telco-Customer-Churn.csv', header=True, inferSchema=True)
df = df.withColumn('months',df['TotalCharges']/df['MonthlyCharges'])
df = df.drop('TotalCharges','customerID')
df = df.na.drop()

# summarize dataframe
numeric_cols = ['tenure','MonthlyCharges','months']
summarize_df(spark,df,numeric_cols)

# output dataset
df.write.mode('overwrite').csv('data/telco/telco_clean_spark_sql.csv',header=False)

''' Clean dataset with spark sql '''

from pyspark.sql import SparkSession, SQLContext, Window
from pyspark.sql.types import *
import pyspark.sql.functions as f
from spark_pkg.util.sql import summarize_df

# setup
spark = SparkSession.builder.master("local").appName("sql_turtles").getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

# read turtles dataset
df = spark.read.csv('data/turtles/turtles.csv', header=True, inferSchema=True)

# clean dataset
#    filter attributes and only keep records with turtles in majority 3 classes
#       the following attributes were not included for the below reasons
#           Circumference: has 13k na values and only 63 actual entries
#           Girth: has 11k na value and only 2k actual entries
#           Depth_mid: has 13k na values and only 181 actual entries
#           Tail: has 13k na values and only 8 actual values
#           Weight: has 11k na values and only 2k actual entries
#           Cap_Region: 11k inshore and only 141 offshore
#           TestLevel_After: 12k na values and only 1167 entries
df.createOrReplaceTempView('dfsql')
df = spark.sql('''
    select Species, lower(Dead_Alive) as Dead_Alive, Gear, SCL_notch, SCL_tip, SCW, CCL_notch, TestLevel_Before, Entangled
    from dfsql
    where Species in ('Loggerhead', 'Green', 'Kemps_Ridley')''')
#    bin Gear categories with under 10 records into 'other' category
df = df.replace(['Shrimp trawl','General public sighting','Channel net','Headstart','Flounder trawl'],'other','Gear')
#    replace na/0 values ini numeric columsn with Species class mean  
numeric_cols = ['SCL_notch','SCL_tip','SCW','CCL_notch','TestLevel_Before']
df = df.replace(0,None,numeric_cols) # 0 > nan
window = Window.partitionBy('Species')
for col_str in numeric_cols:
    df = df.withColumn(col_str,
            f.when(f.col(col_str).isNull(),
                f.avg(f.col(col_str)).over(window))\
                .otherwise(f.col(col_str)))
# Entangled from boolean to string
df = df.withColumn('Entangled', 
        f.col('Entangled').cast('string'))\
            .replace(['true','false'], ['entangled', 'free'], subset='Entangled')

# summarize data
summarize_df(spark,df,numeric_cols)

# output dataset
df.write.mode('overwrite').csv('data/turtles/turtles_clean_spark_sql.csv',header=False)
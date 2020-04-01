# imports
#    pyspark general
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import isnan, when, count, col
#    pyspark ml
from pyspark.ml.classification import LogisticRegression,RandomForestClassifier,NaiveBayes,MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
#    local
from mllib_pkg.util import encode_classifier_data

# setup
spark = SparkSession.builder.appName('turtle-app').getOrCreate()
spark.sparkContext.setLogLevel('ERROR') # only show pyspark errors

print('\n'*5)
verbose = False

# read data
df = spark.read.csv('data/turtles/tagged_turtles.csv',header=True,inferSchema=True)
df.printSchema()


# imports
#    pyspark general
from pyspark.sql import SparkSession
from pyspark.sql.types import *
#    pyspark ml
from pyspark.ml.classification import LogisticRegression,RandomForestClassifier,NaiveBayes,MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
#    local
from mllib_pkg.util import encode_classifier_data


# setup
spark = SparkSession.builder.appName('iris-app').getOrCreate()
spark.sparkContext.setLogLevel('ERROR') # only show pyspark errors

print('\n'*5)
verbose = False

# read data
struct = StructType([
    StructField('sepal_length', FloatType(), True),
    StructField('sepal_width', FloatType(), True),
    StructField('petal_length', FloatType(), True),
    StructField('petal_width', FloatType(), True),
    StructField('species', StringType(), True)])
df = spark.read.schema(struct).csv('data/iris.csv',header=False)
df.printSchema()
print(df.show(5))

# one hot encoding
df = encode_classifier_data(df, 
    label_col = 'species',
    categorical_cols = [],
    numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

# split into train and test datsets
train,test = df.randomSplit([.75,.25], seed=7)

# modeling
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
metrics_f = '%-25s Accuracy: %.3f'
#    logistic regression
lr_model = LogisticRegression().fit(train)
lr_predictions = lr_model.transform(test)
lr_accuracy = evaluator.evaluate(lr_predictions)
print metrics_f%('Logistic Regression',lr_accuracy)
if verbose:
    print(lr_model.coefficientMatrix)
    print(lr_model.interceptVector)
    lr_predictions.select('label','rawPrediction','prediction','probability').show(10)
#    random forest
rf_model = RandomForestClassifier(numTrees=3,maxDepth=2,seed=7).fit(train)
rf_predictions = rf_model.transform(test)
rf_accuracy = evaluator.evaluate(rf_predictions)
print metrics_f%('Random Forest',rf_accuracy)
#    naive bayes
nb_model = NaiveBayes(smoothing=1.0, modelType="multinomial").fit(train)
nb_predictions = nb_model.transform(test)
nb_accuracy = evaluator.evaluate(nb_predictions)
print metrics_f%('Naive Bayes',nb_accuracy)

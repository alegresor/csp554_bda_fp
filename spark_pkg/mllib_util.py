''' Utility functions for spark_pkg/mllib_*.py '''

from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import *
from pyspark.ml.regression import *
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder,TrainValidationSplit
from pyspark.ml.evaluation import MulticlassClassificationEvaluator,RegressionEvaluator


def encode_data(df, categorical_cols, numeric_cols, predict_col, encode_predict_col):
    """
    Args:
        categorical_cols (list): list of collumns to be one-hot encoded (does not include predict_col for classification)
        numeric_cols (list): numeric columns
        predict_col (string): attribute to predict
        encode_predict_col (boolean): should the predict_col be encoded (classification) or not (regression)
    Returns:
        DataFrame with columns
            'label': one hot encoded label column for classification. Not included for regression
            'features': numeric and one hot encoded variables. Included for both classificaiton and regression
    """
    cols = df.columns
    stages = []
    # one hot encoding stages for categorical predictor variables
    for categoricalCol in categorical_cols:
        stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
        encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
        stages += [stringIndexer, encoder]
    # possibly one hot encode the predict_col if this is a classification problem
    predict_col_tf = predict_col
    if encode_predict_col:
        predict_col_tf = 'label'
        predict_col_stringIdx = StringIndexer(inputCol = predict_col, outputCol=predict_col_tf)
        stages += [predict_col_stringIdx]
    assemblerInputs = [c + "classVec" for c in categorical_cols] + numeric_cols
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    stages += [assembler]
    # pipeline stages
    pipeline = Pipeline(stages = stages)
    pipelineModel = pipeline.fit(df)
    df = pipelineModel.transform(df)
    # return appropriote subset of DataFrame
    selectedCols = [predict_col_tf,'features']
    df = df.select(selectedCols)
    return df

def evaluate_models(models,train,test,metric_file_path,Evaluator,metric_names):
    """
    Evaluate list of models
    Args:
        models (list): list of models s.t. predictoins = models[i].transform(test)
        train (DataFrame): training dataset
        test (DataFrame): testing dataset
        metric_file_path (str): path to file to output metrics
        Evaluator (from pyspark.ml.evaluation): handle of an evaluator
        metric_names (list): list of metric names for evaluator to evaluate
    """
    f = open(metric_file_path,'w')
    f.write('model,'+','.join(metric_names)+'\n')
    for model in models:
        model_name = type(model).__name__
        metric_vals = [None]*len(metric_names)
        predictions = model.fit(train).transform(test)
        print model_name
        for i in range(len(metric_names)):
            metric_vals[i] = Evaluator(metricName=metric_names[i]).evaluate(predictions)
            print '\t%15s: %.3f'%(metric_names[i],metric_vals[i])
        f.write(model_name+','+','.join(str(val) for val in metric_vals)+'\n')
    f.close()

def run_classification_models(train,test,metric_file_path,classes):
    """
    Modeling and metrics for classification models
    Args:
        train (DataFrame): training dataset
        test (DataFrame): testing dataset
        metric_file_path (str): path to file to output metrics
        classes (int): number of unique labels
    Note:
        Did not train MultilayerPerceptronClassifier is it requires feature size and output
        size and therefore does not generalize well to our vanilla/black-box testing
    """
    models = []
    models.append( LogisticRegression() )
    models.append( DecisionTreeClassifier(seed=7) )
    models.append( RandomForestClassifier(seed=7) )
    models.append( OneVsRest(classifier=LogisticRegression()) )
    models.append( NaiveBayes() )
    if classes==2:
        models.append( GBTClassifier(seed=7) )
        models.append( LinearSVC() )
    metric_names = ['accuracy','weightedRecall','weightedPrecision','f1']
    evaluate_models(models,train,test,metric_file_path,MulticlassClassificationEvaluator,metric_names)

def run_regression_models(train,test,metric_file_path):
    """
    Modeling and metrics for regression models
    Args:
        train (DataFrame): training dataset
        test (DataFrame): testing dataset
        metric_file_path (str): path to file to output metrics
    Note:
        Did not train IsotonicRegression is it requires wieghts column
        which does not generalize well to our vanilla/black-box testing
    """
    models = []
    # linear regression with cross validation
    lr = LinearRegression(maxIter=10)
    paramGrid = ParamGridBuilder()\
        .addGrid(lr.regParam, [0.001,0.001,.05]) \
        .addGrid(lr.fitIntercept, [False, True])\
        .addGrid(lr.elasticNetParam,[0.0, 0.05, 1.0])\
        .build()
    tvs = TrainValidationSplit(estimator=lr,estimatorParamMaps=paramGrid,evaluator=RegressionEvaluator())
    models.append( tvs )
    models.append( DecisionTreeRegressor() )
    models.append( RandomForestRegressor() )
    models.append( GBTRegressor() )
    metric_names = ['r2','rmse','mae']
    evaluate_models(models,train,test,metric_file_path,RegressionEvaluator,metric_names)

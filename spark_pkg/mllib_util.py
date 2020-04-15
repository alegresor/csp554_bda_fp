''' Utility functions for spark_pkg '''

# imports
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression,RandomForestClassifier,NaiveBayes,MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def encode_classifier_data(df, label_col, categorical_cols, numeric_cols):
    """
    From: https://towardsdatascience.com/machine-learning-with-pyspark-and-mllib-solving-a-binary-classification-problem-96396065d2aa
    Args:
        label_col (string): label column to be one-hot encoded
        categorical_cols (list): list of collumns to be one-hot encoded (does not include label_col)
        numeric_cols (list): numeric columns
    Returns:
        dataframe with new columns 'label' (one hot encoded) and 'features' (includes numeric and one hot encoded variables)
    """
    # one hot encoding
    cols = df.columns
    stages = []
    for categoricalCol in categorical_cols:
        stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
        encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
        stages += [stringIndexer, encoder]
    label_stringIdx = StringIndexer(inputCol = label_col, outputCol = 'label')
    stages += [label_stringIdx]
    assemblerInputs = [c + "classVec" for c in categorical_cols] + numeric_cols
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    stages += [assembler]
    # pipeline
    pipeline = Pipeline(stages = stages)
    pipelineModel = pipeline.fit(df)
    df = pipelineModel.transform(df)
    selectedCols = ['label', 'features'] + cols
    df = df.select(selectedCols)
    return df

def model(train,test,metric_file_path):
    """
    Modeling and metric aggregation
    Args:
        train (DataFrame): training dataset
        test (DataFrame): testing dataset
        metric_file_path (str): path to file to output metrics
    """
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    metric_names = ['accuracy','weightedRecall','weightedPrecision','f1']
    f = open(metric_file_path,'w')
    f.write('model,'+','.join(metric_names)+'\n')
    def get_model_metrics(model_name,predictions):
        metric_vals = [None]*len(metric_names)
        print model_name
        for i in range(len(metric_names)):
            metric_vals[i] = evaluator.evaluate(predictions,{evaluator.metricName: metric_names[i]})
            print '\t%15s: %.3f'%(metric_names[i],metric_vals[i])
        f.write(model_name+','+','.join(str(val) for val in metric_vals)+'\n')
    # logistic regression
    lr_model = LogisticRegression().fit(train)
    lr_predictions = lr_model.transform(test)
    # uncomment below for advanced logistic regression metrics
    '''print(lr_model.coefficientMatrix)
    print(lr_model.interceptVector)
    lr_predictions.select('label','rawPrediction','prediction','probability').show(10)'''
    get_model_metrics('Logistic Regression',lr_predictions)
    # random forest
    rf_model = RandomForestClassifier(numTrees=3,maxDepth=2,seed=7).fit(train)
    rf_predictions = rf_model.transform(test)
    get_model_metrics('Random Forest',rf_predictions)
    #  naive bayes
    nb_model = NaiveBayes(smoothing=1.0, modelType="multinomial").fit(train)
    nb_predictions = nb_model.transform(test)
    get_model_metrics('Naive Bayes',nb_predictions)
    f.close()
''' Utility functions for spark_pkg '''

# imports
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

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
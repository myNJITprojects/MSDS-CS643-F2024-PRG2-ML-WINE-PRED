import sys
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

def main(input_file):
    spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

    # Read the CSV file
    data = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(input_file)

    # Select features and label
    features = data.select(
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
        "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
    )
    label = data.select("quality")

    # Assemble features
    assembler = VectorAssembler(inputCols=features.columns, outputCol="features")
    assembled_data = assembler.transform(features)

    # Combine features and label
    final_data = assembled_data.join(label, how='inner')

    # Split data into training and testing sets
    (trainingData, testData) = final_data.randomSplit([0.7, 0.3], seed=12345)

    # Create a logistic regression model
    lr = LogisticRegression(maxIter=100, featuresCol="features", labelCol="quality")

    # Create a pipeline
    pipeline = Pipeline(stages=[lr])

    # Train the model
    model = pipeline.fit(trainingData)

    # Make predictions on the new data
    predictions = model.transform(data)

    # Print
    predictions.show()
    
    spark.stop()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python wine_predictor.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    main(input_file)
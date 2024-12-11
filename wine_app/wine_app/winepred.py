from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier

# from google.colab import drive
# drive.mount('/content/drive')

# Create a SparkSession
spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

# Read the CSV file (adjust the path as needed)
if 'drive' in globals():  # If Google Drive is mounted
  data = spark.read.csv("/content/drive/MyDrive/Graduate/2024 Fall/CS-643-863/Homeworks/Program 2/TrainingDataset.csv", header=True, inferSchema=True)
else:  # If the file is directly uploaded
  data = spark.read.csv("/home/app/wine_app/data/TrainingDataset.csv", header=True, inferSchema=True)

# Select the features and label
features = data.select(
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
    "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
)
label = data.select("quality")

# Assemble the features into a vector
assembler = VectorAssembler(inputCols=features.columns, outputCol="features")
assembled_data = assembler.transform(features)

# Combine the features and label into a single DataFrame
final_data = assembled_data.join(label, how='inner')

# Split the data into training and testing sets
(trainingData, testData) = final_data.randomSplit([0.7, 0.3], seed=12345)

# Create a logistic regression model
lr = LogisticRegression(maxIter=100, featuresCol="features", labelCol="quality")

# Create a pipeline
pipeline = Pipeline(stages=[lr])

# Train the model
model = pipeline.fit(trainingData)

# Make predictions on the test data
predictions = model.transform(testData)

# Evaluate the model using F1 score
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1") # Specify labelCol and predictionCol
f1_score = evaluator.evaluate(predictions)

print("F1 Score: ", f1_score)

# Stop the SparkSession
spark.stop()
import boto3
import os
from io import BytesIO
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier

from s3Utils import *


try:
  from google.colab import drive
  drive.mount('/content/drive')
except Exception as ex:
  print("Not running on Google Colab. Continue...\n")

# ---------  GLOBAL CONFIGS ----------
modelPath="/path/to/folder"
csvFile="file.csv"
s3Bucket="bucket_name"
fileKey=f"/path/to/folder/{csvFile}"
s3ModelPath="model"
# --------- SPARK CONFIGS ----------
spark_master = "machine hostname"
spark_memory = "10g"
spark_cores = "3"
spark_executors = "4"
# -----------------------------------

# Create a SparkSession
spark = SparkSession.builder \
  .appName("WineQty_ModelPrediction") \
  .master(spark_master) \
  .config("spark.executor.memory", spark_memory) \
  .config("spark.executor.cores", spark_cores) \
  .config("spark.num.executors", spark_executors) \
  .getOrCreate()

# Read the CSV file (adjust the path as needed)
if 'drive' in globals():  # If Google Drive is mounted
  data = spark.read.csv("/content/drive/MyDrive/Graduate/2024 Fall/CS-643-863/Homeworks/Program 2/TrainingDataset.csv", header=True, inferSchema=True)
else:  
  # If the file is directly uploaded
  downloadFile(csvFile, s3Bucket, fileKey)
  data = spark.read.csv(csvFile, header=True, inferSchema=True).repartition(4)


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

# Save the trained model
model_save_path = modelPath
model.write().overwrite().save(model_save_path)
print(f"Model saved at: {model_save_path}")
print(f"Uploading model to {s3Bucket} ...")

#Remove if folder exists
if folderExists(s3Bucket, modelPath):
  print(f"Folder '{modelPath}' exists in the bucket.")
  print(f"Deleting {modelPath}...")
  deleteS3Folder(s3Bucket,modelPath)
  print("Deleted.")

uploadModel(s3Bucket, modelPath, s3ModelPath)

# Make predictions on the test data
predictions = model.transform(testData)

# Evaluate the model using F1 score
evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1") # Specify labelCol and predictionCol
f1_score = evaluator.evaluate(predictions)

print("------------------------------")
print("WINE Predicition")
print("F1 Score: ", f1_score)
print("------------------------------")

# Stop the SparkSession
spark.stop()

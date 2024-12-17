import sys
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler


def predict(model_path, input_data_path):
    try:
        # Initialize Spark session
        spark = SparkSession.builder.appName("WinePrediction").getOrCreate()
        
        # Load input data
        data = spark.read.csv(input_data_path, header=True, inferSchema=True)

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

        # Load the trained model
        model = RandomForestClassificationModel.load(model_path)
        
        # Make predictions
        predictions = model.transform(final_data)
        
        # Show predictions
        print("Predictions:")
        predictions.select("features", "prediction").show()
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict.py <path_to_model> <path_to_input_data>")
        sys.exit(1)

    input_data_path = sys.argv[1]
    model_path = sys.argv[2]
    predict(input_data_path, model_path)

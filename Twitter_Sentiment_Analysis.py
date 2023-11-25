# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Pyspark 3
#     language: python
#     name: pyspark3
# ---

"""
Twitter Sentiment Analysis
The objective of this task is to detect hate speech in tweets. For the sake of simplicity, we say a
tweet contains hate speech if it has a racist or sexist sentiment associated with it. So, the task is
to classify racist or sexist tweets from other tweets.
Formally, given a training sample of tweets and labels, where label '1' denotes the tweet is
racist/sexist and label '0' denotes the tweet is not racist/sexist, your objective is to predict the
labels on the test dataset.
Let’s say we receive hundreds of comments per second and we want to keep the platform clean
by blocking the users who post comments that contain hate speech. So, whenever we receive
the new text, we will pass that into the pipeline and get the predicted sentiment.
"""

# Import SparkSession for Spark SQL functionality
from pyspark.sql import SparkSession  
# Import StreamingContext for Spark Streaming functionality
from pyspark.streaming import StreamingContext  
# Import ML feature transformers
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer  
# Import Logistic Regression classifier
from pyspark.ml.classification import LogisticRegression 
# Import ML Pipeline for chaining multiple stages
from pyspark.ml import Pipeline 
# Import col function for DataFrame column operations
from pyspark.sql.functions import col  

# +
"""
process_batch function - It processes a batch of data by converting it to a DataFrame, 
applying the same preprocessing steps as in the training phase, making predictions using 
the trained model, and taking appropriate actions based on the predictions. 
It handles exceptions that may occur during the processing and handles the case when the 
batch is empty.
"""

def process_batch(batch_rdd):
    # Check if the batch RDD is empty
    if not batch_rdd.isEmpty():
        try:
            # Convert RDD to DataFrame
            batch_df = spark.createDataFrame(batch_rdd, ["tweet"])

            # Apply the same preprocessing steps as in the training phase
            preprocessed_batch_df = model.transform(batch_df)

            # Get predictions using the trained model
            predictions_batch = preprocessed_batch_df.select("tweet", "prediction")

            # Filter out hate speech tweets (prediction = 1)
            filtered_predictions = predictions_batch.filter(col("prediction") == 1)

            # Take appropriate actions based on predictions
            if not filtered_predictions.isEmpty():
                # Perform actions for hate speech tweets (e.g., block users, flag tweets, etc.)
                print("Racist or sexist sentiment:")
                filtered_predictions.select("tweet").show(truncate=False)
            else:
                print("Not racist or sexist sentiment.")
        except Exception as e:
            print("Error occurred while processing batch:", str(e))
    else:
        print("No data in the batch")


# -

if __name__ == "__main__":
    # Create a SparkSession
    spark = SparkSession.builder.appName("Twitter Sentiment Analysis").getOrCreate()

    # Load and preprocess the training data
    train_data = spark.read.csv("Twitter_Sentiment_Analysis/train.csv", header=True, inferSchema=True)
    train_data.cache()  # Cache the training data for better performance

    # Perform basic analysis on the data
    total_rows = train_data.count()
    positive_rows = train_data.filter(col("label") == 1).count()
    negative_rows = train_data.filter(col("label") == 0).count()
    print("Total rows:", total_rows)
    print("Positive (label 1) rows:", positive_rows)
    print("Negative (label 0) rows:", negative_rows)
    print("% of positive rows:", (positive_rows / total_rows) * 100)
    print("% of negative rows:", (negative_rows / total_rows) * 100)

    # Define the machine learning pipeline
    # Tokenize the input text into individual words
    tokenizer = RegexTokenizer(inputCol="tweet", outputCol="tokens", pattern=r"\W")
    # Remove stop words (common words that do not carry much information)
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    # Convert the filtered tokens into a numerical vector representation
    vectorizer = CountVectorizer(inputCol="filtered_tokens", outputCol="features")
    # Define the logistic regression classifier
    classifier = LogisticRegression(featuresCol="features", labelCol="label")
    # Create the pipeline by chaining the stages together
    pipeline = Pipeline(stages=[tokenizer, remover, vectorizer, classifier])

    # Train the model
    model = pipeline.fit(train_data)

    # Initialize a Spark Streaming context with a batch duration of 3 seconds
    batch_duration = 3
    ssc = StreamingContext(spark.sparkContext, batch_duration)

    # Specify the port and hostname for the DStream
    port = 9999
    hostname = "localhost"

    # Create a DStream from the streaming data source
    dstream = ssc.socketTextStream(hostname, port)

    # Apply the processing function to each batch of data in the DStream
    dstream.foreachRDD(process_batch)

    # Start the Spark Streaming context and wait for termination
    try:
        ssc.start()
        ssc.awaitTermination()
    except KeyboardInterrupt:
        # Stop the Spark Streaming context if the program is manually interrupted
        ssc.stop()
        print("Streaming stopped by the user")

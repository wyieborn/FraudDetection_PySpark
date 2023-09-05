


import time
# importing the required pyspark library
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col, sum, when, lag, abs, sqrt, pow, floor, current_date
from pyspark.sql.functions import datediff, to_timestamp, hour, asin, sin, cos, radians, sqrt
from pyspark.sql.window import Window


#randomforest classifier and logistic regression
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
# for encoding, vectorization, and pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
# model evaluation
from pyspark.ml.evaluation import BinaryClassificationEvaluator



def age_segment(df):
    # Calculate age based on current date and DOB
    df = df.withColumn("age", floor(datediff(current_date(), col("dob")) / 365))

    # Create the "age_group" column based on the given logic
    df = df.withColumn(
        "age_group",
        when(col("age") < 18, "under 18")
        .when((col("age") >= 18) & (col("age") <= 30), "18-30")
        .when((col("age") >= 31) & (col("age") <= 50), "31-50")
        .when(col("age") >= 50, "over 50")
        .otherwise("Unknown")
    )

    #dropping columns after transformation
    columns_ = ("age")
    df = df.drop(*columns_)

    return df

def pop_segment(df):
    # Creating the "city_pop_segment" column
    df = df.withColumn(
        "city_pop_segment",
        when(col("city_pop") < 10000, "few density")
        .when((col("city_pop") > 10000) & (col("city_pop") < 50000), "normal density")
        .when(col("city_pop") > 50000, "high density")
        .otherwise("Unknown")
    )

    #dropping columns after transformation
    columns_ = ("city_pop")
    df = df.drop(*columns_)

    return df

def unix_segment(df):
    # Defining a window specification partitioned by 'cc_num' and ordered by 'unix_time'
    window_spec = Window.partitionBy("cc_num").orderBy("unix_time")

    # Compute the difference between consecutive 'unix_time' values within each group
    df = df.withColumn("most_recent", col("unix_time") - lag("unix_time", 1).over(window_spec))
    # Fill na with -1 or other negative value and not 0 because there might be cases where the first transaction for a credit card is not present in the dataset.
    df = df.withColumn("most_recent", when(col("most_recent").isNull(), -1).otherwise(col("most_recent")))


    # Converting "most_recent" time from second to hour .
    df = df.withColumn("most_recent", (col("most_recent") / 3600.0))

    # Creating the "recent_segment" by dividing the day into 4 section and default values to most_recent_transaction for < 1
    df = df.withColumn(
        "recent_segment",
        when(col("most_recent") < 1, "most_recent_transaction")
        .when((col("most_recent") > 1) & (col("most_recent") < 6), "within 6 hours")
        .when((col("most_recent") > 6) & (col("most_recent") < 12), "after 6 hours")
        .when((col("most_recent") > 12) & (col("most_recent") < 24), "after half-Day")
        .when(col("most_recent") > 24, "after 24 hours")
        .otherwise("first transaction")
    )

    #dropping columns after transformation
    columns_ = ("most_recent")
    df = df.drop(*columns_)

    return df

def get_displacement(df):
    # Calculatig absolute differences between 'lat' and 'merch_lat'
    df = df.withColumn("diff_lat", abs(col("lat") - col("merch_lat")))
    # Calculating absolute differences between 'long' and 'merch_long'
    df = df.withColumn("diff_long", abs(col("long") - col("merch_long")))

    # since the lat and long difference for each degree is equal to 110 kilometers(approximately) and using pythogorous formula
    # Calculating the "displacement" column using the formula
    df = df.withColumn("displacement",sqrt(pow(col("diff_lat") * 110, 2) + pow(col("diff_long") * 110, 2)))

    #dropping columns after transformation
    columns_ = ("diff_lat","diff_long")
    df = df.drop(*columns_)

    return df


# for more accurate long distance calculation.
def distance_using_haversine(df):
    # Convert latitude and longitude from degrees to radians
    df_with_radians = df.withColumn("lat_rad", radians(col("lat"))) \
                        .withColumn("long_rad", radians(col("long"))) \
                        .withColumn("merch_lat_rad", radians(col("merch_lat"))) \
                        .withColumn("merch_long_rad", radians(col("merch_long")))

    # Calculate the Haversine distance
    df_with_distance = df_with_radians.withColumn(
        "distance",
        asin(
            sqrt(
                sin((col("merch_lat_rad") - col("lat_rad")) / 2) ** 2 +
                cos(col("lat_rad")) * cos(col("merch_lat_rad")) *
                sin((col("merch_long_rad") - col("long_rad")) / 2) ** 2
            )
        ) * 2 * 6371  # Earth's radius in kilometers
    )

    #dropping columns after transformation
    columns_ = ("lat_rad","long_rad","merch_lat_rad","merch_long_rad")
    df_with_distance = df_with_distance.drop(*columns_)

    return df_with_distance

def feature_engineering(df):
  

    # Handle missing values
    df = df.na.drop()

    # taking hour of day from time
    df = df.withColumn("hour", hour(to_timestamp(col("trans_date_trans_time"), "yyyy-MM-dd HH:mm:ss")))

    #segmented values and transformations
    df = pop_segment(df)
    df = age_segment(df)
    df = unix_segment(df)
    df = get_displacement(df)


    #dropping columns that are not useful for the model with no unique patterns and the features that we already prepared from
    columns_ = ("_c0","first","last","dob","unix_time","city","zip","street", "lat", "long", "merch_lat","merch_long","trans_date_trans_time","cc_num","merchant","job","state","gender","trans_num")

    df = df.drop(*columns_)

    return df

def data_pipeline_assembling(data):

    #getting string data type columns
    categorical_cols = [x for (x, dataType) in data.dtypes if dataType == "string"]
    #getting numerical data types column
    # numerical_cols = [c for c in ready_train.columns if c not in categorical_columns]
    numerical_cols = [x for (x, dataType) in data.dtypes if dataType == "double"] 
    numerical_cols += ["hour"]

    # Apply StringIndexer to categorical columns
    indexers = [StringIndexer(inputCol=col, outputCol=col + "_index").fit(data) for col in categorical_cols]
    indexer_pipeline = Pipeline(stages=indexers)
    # fitting the test and train data to indexer pipeline
    indexed_data = indexer_pipeline.fit(data).transform(data)

    # Assemble features
    feature_columns = numerical_cols + [col + "_index" for col in categorical_cols]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

    # assemble data for both train and test with respect to indexed pipelines
    assembled_data = assembler.transform(indexed_data)


    return assembled_data

def randomforest_model(ready_train, ready_test):

    print("\nStarting to create indexers and pipeline for random forest.............\n")
    assembled_data = data_pipeline_assembling(ready_train)
    assembled_data_test = data_pipeline_assembling(ready_test)
    print("\nProcess completed to create indexers and pipeline for randomforest................\n")


    # Define the classifier
    rf = RandomForestClassifier(labelCol="is_fraud", featuresCol="features", numTrees=100, maxDepth=10, seed=42)

    # Train the model
    print("\nRandom Forest Classifier Model Training Started............................\n")
    model = rf.fit(assembled_data)
    print("\nRandom Forest Classifier Model Training Completed.....................\n")
    # Make predictions on the test set
    predictions = model.transform(assembled_data_test)


    return model, predictions

def logistic_regresstion_model(ready_train, ready_test):

  

    print("\nStarting to create indexers and pipeline for logistic regression..............\n")
    assembled_data = data_pipeline_assembling(ready_train)
    assembled_data_test = data_pipeline_assembling(ready_test)
    print("\nProcess completed to create indexers and pipeline for logistic regression................\n")


    train_ = assembled_data.select(
    col("features").alias("features"),
    col("is_fraud").alias("label"),
    )
    test_ = assembled_data_test.select(
    col("features").alias("features"),
    col("is_fraud").alias("label"),
    )
    # Train the model
    print("\nLogistic Regression Model Training Started............................\n")
    model = LogisticRegression().fit(train_)
    print("\nLogistic Regression Model Training Completed.....................\n")
    # Make predictions on the test set
    predictions = model.transform(test_)


    return model, predictions
  
def evaluate_model(predictions, label = "is_fraud"):

    # Evaluate the model
    evaluator = BinaryClassificationEvaluator(labelCol=label, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    roc_auc = evaluator.evaluate(predictions)
    print("ROC Area Under Curver:", roc_auc)
    
if __name__ == "__main__":
    #initiate spark session
    spark_session = SparkSession.builder.appName('_FraudDetection').getOrCreate()


    # define input path for the data
    input_path_train = "fraudTrain.csv"
    input_path_test = "fraudTest.csv"


    #CSV file can be downloaded from the link mentioned above.
    data_train = spark_session.read.csv(input_path_train,
                        inferSchema=True,header=True) # , schema=schema #if schema needed

    data_test = spark_session.read.csv(input_path_test,
                        inferSchema=True,header=True)



    # performing our pre feature selection and engineering
    ready_train = feature_engineering(data_train)
    ready_test = feature_engineering(data_test)
    
    # Record start time--------------------------------------------------------------
    start_time = time.time()

    # Random Forest Classifier
    model_rf, predictions_rf =  randomforest_model(ready_train, ready_test)

    #evaluating the random forest model
    print("Random Forest Model Evaluation ..............\n")
    evaluate_model(predictions_rf)


    # Record end time
    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Time taken to train and evaluate Random Forest Classifier: {elapsed_time:.6f} seconds")
    
    
    # Record start time-------------------------------------------------------------------
    start_time = time.time()

    # Logistic Regression Classifier
    model_lr, predictions_lr = logistic_regresstion_model(ready_train,ready_test)

    #evaluating the random forest model
    print("Logistic Regression Model Evaluation ..............\n")
    evaluate_model(predictions_lr,"label")

    # Record end time
    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Time taken to train and evaluate Logistic Regression: {elapsed_time:.6f} seconds")

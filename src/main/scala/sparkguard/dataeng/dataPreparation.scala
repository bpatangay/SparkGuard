import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}

// Enable setting to capture only error logs
spark.sparkContext.setLogLevel("ERROR")

// Loading the downloaded credit card dataset for processing the fraud detection analysis
val ccDataset = spark.read.option("header", "true").option("inferSchema", "true").csv("data/creditcard.csv")

/* Extraction of all the features excluding the target column - Class from the available dataset.
Collation of all the extracted features into a single dense vector column required by Spark ML models as the input and set up of vector assembler.
Application of the configured transformation to the dataset */
val featureColumns = ccDataset.columns.filter(_ != "Class")
val vecAssembler= new VectorAssembler().setInputCols(featureColumns).setOutputCol("combinedFeatures")
val assembledDataframe = vecAssembler.transform(ccDataset)

/* 
Configuration of standard scaler model to align the unscaled features such as Time and Amount with rest normalized features which had already undergone PCA transformations. 
*/
val scaler = new StandardScaler().setInputCol("combinedFeatures").setOutputCol("outputFeatures").setWithStd(true).setWithMean(true)
val dataPPModel = scaler.fit(assembledDataframe)
val scaledDataframe = dataPPModel.transform(assembledDataframe)

/* Selection and transformation of required columns. Performing data split - 70 percent for training, 15 percent each for validation and test purposes and writing it in parquet data format for structure preservance
*/
val outputDataframe = scaledDataframe.select("outputFeatures", "Class").withColumnRenamed("outputFeatures", "features").withColumnRenamed("Class", "label")
val Array(trainDF, validateDF, testDF) = outputDataframe.randomSplit(Array(0.7, 0.15, 0.15), seed = 1234L)
trainDF.write.mode("overwrite").parquet("data/trainingData.parquet")
validateDF.write.mode("overwrite").parquet("data/validationData.parquet")
testDF.write.mode("overwrite").parquet("data/testData.parquet")
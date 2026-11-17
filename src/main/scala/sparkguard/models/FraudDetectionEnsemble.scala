import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.linalg.{Vector => MLVector}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.ml.functions.vector_to_array
import org.apache.spark.storage.StorageLevel
import java.io.PrintWriter
import org.apache.spark.sql.SparkSession
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier

//Pull dataset files from parquets created by Data Engineering File
val trainDF = spark.read.parquet("data/trainingData.parquet")
val validationDF = spark.read.parquet("data/validationData.parquet")
val testDF = spark.read.parquet("data/testData.parquet")


//Voting Ensemble will consists of an SVM, a Neural Network, and an XGBoost Model
//----------------------------------------------------------------------------------
//SVM Section

//Convert dataframes into LabeledPoint RDDs for the SVM
val trainDataSVM = trainDF.rdd.map {row => 
	val label = row.getAs[Int]("label").toDouble
	val features = row.getAs[MLVector]("features")
	
	LabeledPoint(label, Vectors.dense(features.toArray))
}

val validationDataSVM = validationDF.rdd.map {row => 
	val label = row.getAs[Int]("label").toDouble
	val features = row.getAs[MLVector]("features")
	
	LabeledPoint(label, Vectors.dense(features.toArray))
}
val testDataSVM = testDF.rdd.map {row => 
	val label = row.getAs[Int]("label").toDouble
	val features = row.getAs[MLVector]("features")
	
	LabeledPoint(label, Vectors.dense(features.toArray))
}

//Train the SVM model
val iterations = 100 
val svm_model = SVMWithSGD.train(trainDataSVM, iterations)

//Clear Threshold
svm_model.clearThreshold()


//Collect all the raw scores and true labels for the data points
val scoresAndLabels = validationDataSVM.map { point =>
  (svm_model.predict(point.features), point.label)
}.collect()

//Use every value that appears as a possible threshold
val thresholds = scoresAndLabels.map(_._1).distinct


//Evaluate all thresholds based on validation set and pick the one that results in the highest number of correct predictions
val best = thresholds.map { threshold =>
  val correct = scoresAndLabels.count {
    case (score, label) =>
      (if (score >= threshold) 1.0 else 0.0) == label
  }
   (threshold, correct.toDouble / scoresAndLabels.length)
}.maxBy(_._2)

//print out validation threshold and val accuracy
println()
println(s"Best threshold: ${best._1}")
println(s"Best validation accuracy: ${best._2 * 100}%")
println()

//set the model to use the best threshold found
svm_model.setThreshold(best._1)

val svm_predictions = testDataSVM.map { 
	point => val prediction = svm_model.predict(point.features)
	(prediction, point.label)
	}
	

//----------------------------------------------------------------------------------
// XGBoost Model

//Pull dataset files from parquets created by Data Engineering File
val xgboost_trainDF = spark.read.parquet("data/trainingData.parquet").withColumn("label", col("label").cast("double"))
val xgboost_validationDF = spark.read.parquet("data/validationData.parquet").withColumn("label", col("label").cast("double"))
val xgboost_testDF = spark.read.parquet("data/testData.parquet").withColumn("label", col("label").cast("double"))

//Class imbalance ratio used to weight the fraud class
val negCount = trainDF.filter(col("label") === 0.0).count().toDouble
val posCount = trainDF.filter(col("label") === 1.0).count().toDouble

//Train the model
println("Training XGBoost...")
val classifier = new XGBoostClassifier(Map(
"objective"        -> "binary:logistic",
"eval_metric"      -> "aucpr",
"tree_method"      -> "hist",
"max_depth"        -> 6,
"eta"              -> 0.1,
"num_round"        -> 200,
"num_workers"      -> 1,
"seed"             -> 1234,
"scale_pos_weight" -> negCount / posCount
)).setFeaturesCol("features").setLabelCol("label")
val xgboost_model = classifier.fit(xgboost_trainDF)

//Collect all the fraud probabilities (raw scores) and true labels for the validation set
val xgboost_scoresAndLabels = xgboost_model.transform(xgboost_validationDF).select("probability", "label").rdd.map { row =>
  (row.getAs[MLVector]("probability")(1), row.getAs[Double]("label"))
}.collect()

//Use every value that appears as a possible threshold
val thresholds = xgboost_scoresAndLabels.map(_._1).distinct

//Evaluate all thresholds based on validation set and pick the one that results in the highest fraud-class F1
println("Searching for best threshold on validation set...")
val best = thresholds.map { threshold =>
val xgboost_tp = xgboost_scoresAndLabels.count { case (s, l) => s >= threshold && l == 1.0 }
val xgboost_fp = xgboost_scoresAndLabels.count { case (s, l) => s >= threshold && l == 0.0 }
val xgboost_fn = xgboost_scoresAndLabels.count { case (s, l) => s < threshold && l == 1.0 }
val xgboost_p = if (xgboost_tp + xgboost_fp > 0) xgboost_tp.toDouble / (xgboost_tp + xgboost_fp) else 0.0
val xgboost_r = if (xgboost_tp + xgboost_fn > 0) xgboost_tp.toDouble / (xgboost_tp + xgboost_fn) else 0.0
val xgboost_f1 = if (xgboost_p + xgboost_r > 0) 2.0 * xgboost_p * xgboost_r / (xgboost_p + xgboost_r) else 0.0
(threshold, xgboost_f1)
}.maxBy(_._2)

//print out validation threshold and validation F1
println()
println(s"Best threshold: ${best._1}")
println(s"Best validation F1: ${best._2 * 100}%")
println()
val xgboost_bestThreshold = best._1

val xgboost_predictions = xgboost_model.transform(xgboost_testDF).select("probability", "label").rdd.map { 
	row => val score = row.getAs[MLVector]("probability")(1)
	(if (score >= xgboost_bestThreshold) 1.0 else 0.0, row.getAs[Double]("label"))
}





//----------------------------------------------------------------------------------

//Neural Network
// --------------------------- Configuration -------------------------------
val trainPath = "data/trainingData.parquet"
val validationPath = "data/validationData.parquet"
val testPath = "data/testData.parquet"
val outputDir = "results/neural_network"
val hiddenLayers = Array(64, 32)
val maxIterations = 100
val blockSize = 128
val seed = 42L
val thresholdMetric = "f1" // "f1" or "accuracy"; SVM uses accuracy
val thresholdCandidates = (1 to 99).map(_ / 100.0).toArray

// ------------------------------ Data ------------------------------------
val rawTrain = spark.read.parquet(trainPath)
val rawValidation = spark.read.parquet(validationPath)
val rawTest = spark.read.parquet(testPath)

// Explicitly cast labels to Double, as required by Spark ML.
def prepare(df: DataFrame): DataFrame = {
  require(df.columns.contains("label") && df.columns.contains("features"),
    "Input parquet must contain label and features columns")
  df.select(col("label").cast("double").as("label"), col("features")).na.drop(Seq("label", "features"))
}
val nnTrain = prepare(rawTrain).persist(StorageLevel.MEMORY_AND_DISK)
val nnValidation = prepare(rawValidation).persist(StorageLevel.MEMORY_AND_DISK)
val nnTest = prepare(rawTest).persist(StorageLevel.MEMORY_AND_DISK)

val trainCount = nnTrain.count()
val validationCount = nnValidation.count()
val testCount = nnTest.count()
require(trainCount > 0 && validationCount > 0 && testCount > 0, "All splits must be nonempty")

val inputSize = nnTrain.select("features").first().getAs[MLVector](0).size
val layers = Array(inputSize) ++ hiddenLayers ++ Array(2)
println(s"Training rows: $trainCount; Validation rows: $validationCount; Test rows: $testCount")
println(s"Neural network layers: ${layers.mkString(" -> ")}")

// ----------------------------- Training ---------------------------------
val nn = new MultilayerPerceptronClassifier().setLabelCol("label").setFeaturesCol("features").setPredictionCol("prediction").setProbabilityCol("probability").setRawPredictionCol("rawPrediction").setLayers(layers).setMaxIter(maxIterations).setBlockSize(blockSize).setSeed(seed).setSolver("l-bfgs")

val trainingStart = System.nanoTime()
val nnModel = nn.fit(nnTrain)
val trainingSeconds = (System.nanoTime() - trainingStart) / 1e9
println(f"Training time: $trainingSeconds%.2f seconds")

// ------------------------- Validation threshold -------------------------
// probability(1) is the predicted probability of fraud.
val validationScores = nnModel.transform(nnValidation).select(col("label"), col("probability")).rdd.map(r => (r.getAs[MLVector](1)(1), r.getDouble(0))).persist(StorageLevel.MEMORY_AND_DISK)

// Thresholds are fixed in advance and chosen using validation data only.
// Aggregate counts on executors rather than collecting all predictions to driver.
val thresholdStats = thresholdCandidates.map { t =>
  val counts = validationScores.map { case (score, label) =>
    val pred = if (score >= t) 1.0 else 0.0
    val tp = if (pred == 1.0 && label == 1.0) 1L else 0L
    val fp = if (pred == 1.0 && label == 0.0) 1L else 0L
    val tn = if (pred == 0.0 && label == 0.0) 1L else 0L
    val fn = if (pred == 0.0 && label == 1.0) 1L else 0L
    (tp, fp, tn, fn)
  }.reduce { (a, b) => (a._1+b._1, a._2+b._2, a._3+b._3, a._4+b._4) }
  val (tp, fp, tn, fn) = counts
  val precision = if (tp+fp == 0) 0.0 else tp.toDouble/(tp+fp)
  val recall = if (tp+fn == 0) 0.0 else tp.toDouble/(tp+fn)
  val f1 = if (precision+recall == 0) 0.0 else 2*precision*recall/(precision+recall)
  val accuracy = (tp+tn).toDouble/(tp+fp+tn+fn)
  (t, accuracy, f1)
}
val bestThresholdResult = if (thresholdMetric == "accuracy") {
  thresholdStats.maxBy(x => (x._2, x._3))
} else {
  require(thresholdMetric == "f1", "thresholdMetric must be f1 or accuracy")
  thresholdStats.maxBy(x => (x._3, x._2))
}
val bestThreshold = bestThresholdResult._1
println(f"Selected threshold: $bestThreshold%.2f ($thresholdMetric on validation set)")
println(f"Validation accuracy: ${bestThresholdResult._2*100}%.4f%%")
println(f"Validation fraud F1: ${bestThresholdResult._3*100}%.4f%%")
validationScores.unpersist()

// --------------------------- Final test ---------------------------------
val inferenceStart = System.nanoTime()
val testScores = nnModel.transform(nnTest).select(col("label"), col("probability")).rdd.map(r => (r.getAs[MLVector](1)(1), r.getDouble(0))).persist(StorageLevel.MEMORY_AND_DISK)

val nn_predictions = testScores.map {
  case (score, label) =>
    (
      if (score >= bestThreshold) 1.0 else 0.0,
      label
    )
}


//Test to make sure everything works
println("SVM")
svm_predictions.take(10).foreach(println)
println()
println("XGBoost")
xgboost_predictions.take(10).foreach(println)
println()
println("NN")
nn_predictions.take(10).foreach(println)
println()

//Aggregation
val ensemblePredictions =
	svm_predictions.zip(xgboost_predictions).zip(nn_predictions).map {
      case (
        ((svmPred, actual),
         (xgbPred, _)),
         (nnPred, _)
      ) =>
        val voteCount =
          svmPred.toInt +
          xgbPred.toInt +
          nnPred.toInt

        val finalPred =
          if (voteCount >= 2) 1.0 else 0.0

        (finalPred, actual)
   }
val truePositives = ensemblePredictions.filter{case (prediction, label) => prediction == 1.0 && label == 1.0}.count()


val falsePositives =ensemblePredictions.filter{case (prediction, label) => prediction == 1.0 && label == 0.0}.count()

val trueNegatives = ensemblePredictions.filter{case (prediction, label) => prediction == 0.0 && label == 0.0}.count()

val falseNegatives = ensemblePredictions.filter{case (prediction, label) => prediction == 0.0 && label == 1.0}.count()

// Calculate fraud-class metrics.

val accuracy = ((truePositives + trueNegatives).toDouble / (truePositives + trueNegatives + falsePositives + falseNegatives))

val fraudPrecision = if (truePositives + falsePositives > 0) truePositives.toDouble / (truePositives + falsePositives)else 0.0

val fraudRecall =if (truePositives + falseNegatives > 0) truePositives.toDouble / (truePositives + falseNegatives) else 0.0

val fraudF1 = if (fraudPrecision + fraudRecall > 0) 2.0 * fraudPrecision * fraudRecall /(fraudPrecision + fraudRecall) else 0.0

val binaryMetrics = new BinaryClassificationMetrics(ensemblePredictions)
val rocAuc = binaryMetrics.areaUnderROC()
val prAuc = binaryMetrics.areaUnderPR()

  println("\n========== ENSEMBLE RESULTS ==========")
  println(f"Accuracy:       ${accuracy * 100}%.4f%%")
  println()
  println("Fraud-class metrics:")
  println(f"Precision:      ${fraudPrecision * 100}%.4f%%")
  println(f"Recall:         ${fraudRecall * 100}%.4f%%")
  println(f"F1 score:       ${fraudF1 * 100}%.4f%%")
  println(f"ROC-AUC:           $rocAuc%.6f")
  println(f"PR-AUC:            $prAuc%.6f")
  println()
  println("Confusion matrix:")
  println(f"True positives (fraud detected):     $truePositives%d")
  println(f"False positives (legitimate flagged): $falsePositives%d")
  println(f"True negatives (legitimate cleared): $trueNegatives%d")
  println(f"False negatives (fraud missed):      $falseNegatives%d")
  println("====================================")
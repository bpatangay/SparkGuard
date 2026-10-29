import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.linalg.{Vector => MLVector}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.ml.functions.vector_to_array
import org.apache.spark.storage.StorageLevel
import java.io.PrintWriter

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
  df.select(col("label").cast("double").as("label"), col("features"))
    .na.drop(Seq("label", "features"))
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
val nn = new MultilayerPerceptronClassifier()
  .setLabelCol("label")
  .setFeaturesCol("features")
  .setPredictionCol("prediction")
  .setProbabilityCol("probability")
  .setRawPredictionCol("rawPrediction")
  .setLayers(layers)
  .setMaxIter(maxIterations)
  .setBlockSize(blockSize)
  .setSeed(seed)
  .setSolver("l-bfgs")

val trainingStart = System.nanoTime()
val nnModel = nn.fit(nnTrain)
val trainingSeconds = (System.nanoTime() - trainingStart) / 1e9
println(f"Training time: $trainingSeconds%.2f seconds")

// ------------------------- Validation threshold -------------------------
// probability(1) is the predicted probability of fraud.
val validationScores = nnModel.transform(nnValidation)
  .select(col("label"), col("probability"))
  .rdd.map(r => (r.getAs[MLVector](1)(1), r.getDouble(0)))
  .persist(StorageLevel.MEMORY_AND_DISK)

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
val testScores = nnModel.transform(nnTest)
  .select(col("label"), col("probability"))
  .rdd.map(r => (r.getAs[MLVector](1)(1), r.getDouble(0)))
  .persist(StorageLevel.MEMORY_AND_DISK)

val confusion = testScores.map { case (score, label) =>
  val pred = if (score >= bestThreshold) 1.0 else 0.0
  (if (pred == 1.0 && label == 1.0) 1L else 0L,
   if (pred == 1.0 && label == 0.0) 1L else 0L,
   if (pred == 0.0 && label == 0.0) 1L else 0L,
   if (pred == 0.0 && label == 1.0) 1L else 0L)
}.reduce { (a, b) => (a._1+b._1, a._2+b._2, a._3+b._3, a._4+b._4) }

val (tp, fp, tn, fn) = confusion
val accuracy = (tp+tn).toDouble/(tp+fp+tn+fn)
val fraudPrecision = if (tp+fp == 0) 0.0 else tp.toDouble/(tp+fp)
val fraudRecall = if (tp+fn == 0) 0.0 else tp.toDouble/(tp+fn)
val fraudF1 = if (fraudPrecision+fraudRecall == 0) 0.0 else 2*fraudPrecision*fraudRecall/(fraudPrecision+fraudRecall)

val binaryMetrics = new BinaryClassificationMetrics(testScores)
val rocAuc = binaryMetrics.areaUnderROC()
val prAuc = binaryMetrics.areaUnderPR()
val inferenceSeconds = (System.nanoTime() - inferenceStart) / 1e9

println("\n========== NEURAL NETWORK RESULTS ==========")
println(f"Accuracy:          ${accuracy*100}%.4f%%")
println(f"Fraud precision:   ${fraudPrecision*100}%.4f%%")
println(f"Fraud recall:      ${fraudRecall*100}%.4f%%")
println(f"Fraud F1:          ${fraudF1*100}%.4f%%")
println(f"ROC-AUC:           $rocAuc%.6f")
println(f"PR-AUC:            $prAuc%.6f")
println(s"True positives:    $tp")
println(s"False positives:   $fp")
println(s"True negatives:    $tn")
println(s"False negatives:   $fn")
println(f"Training seconds:  $trainingSeconds%.2f")
println(f"Test eval seconds: $inferenceSeconds%.2f")
println("============================================\n")

// ----------------------------- Save -------------------------------------
// Saving the model uses Spark filesystem semantics (local or HDFS).
// Use a fresh outputDir or overwrite explicitly if re-running.
nnModel.write.overwrite().save(s"$outputDir/model")

// Save test scores + predictions, distributed as parquet.
val testPredictionDF = nnModel.transform(nnTest)
  .withColumn("fraud_probability", vector_to_array(col("probability"))(1))
  .withColumn("fraud_prediction", when(col("fraud_probability") >= lit(bestThreshold), lit(1.0)).otherwise(lit(0.0)))
  .select("label", "fraud_probability", "fraud_prediction")
testPredictionDF.write.mode("overwrite").parquet(s"$outputDir/test_predictions.parquet")

// Save a one-row summary as CSV.
import spark.implicits._
Seq(("NeuralNetwork", inputSize, hiddenLayers.mkString("-"), maxIterations,
  thresholdMetric, bestThreshold, accuracy, fraudPrecision, fraudRecall,
  fraudF1, rocAuc, prAuc, tp, fp, tn, fn, trainingSeconds, inferenceSeconds))
  .toDF("model", "input_features", "hidden_layers", "max_iterations",
    "threshold_metric", "threshold", "accuracy", "fraud_precision", "fraud_recall",
    "fraud_f1", "roc_auc", "pr_auc", "tp", "fp", "tn", "fn",
    "training_seconds", "test_eval_seconds")
  .coalesce(1).write.mode("overwrite").option("header", "true")
  .csv(s"$outputDir/metrics_csv")

println(s"Saved model, predictions, and metrics under: $outputDir")
testScores.unpersist()
nnTrain.unpersist()
nnValidation.unpersist()
nnTest.unpersist()

package sparkguard
//Imports
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.apache.spark.ml.linalg.{Vector => MLVector}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier

object XGBoostRealData {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("SparkGuard XGBoost - Real Dataset")
      .master("local[*]")
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    try {
      //Pull dataset files from parquets created by Data Engineering File
      val trainDF = spark.read.parquet("data/trainingData.parquet")
        .withColumn("label", col("label").cast("double"))
      val validationDF = spark.read.parquet("data/validationData.parquet")
        .withColumn("label", col("label").cast("double"))
      val testDF = spark.read.parquet("data/testData.parquet")
        .withColumn("label", col("label").cast("double"))

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
      val model = classifier.fit(trainDF)

      //Collect all the fraud probabilities (raw scores) and true labels for the validation set
      val scoresAndLabels = model.transform(validationDF)
        .select("probability", "label")
        .rdd
        .map { row =>
          (row.getAs[MLVector]("probability")(1), row.getAs[Double]("label"))
        }.collect()

      //Use every value that appears as a possible threshold
      val thresholds = scoresAndLabels.map(_._1).distinct

      //Evaluate all thresholds based on validation set and pick the one that results in the highest fraud-class F1
      println("Searching for best threshold on validation set...")
      val best = thresholds.map { threshold =>
        val tp = scoresAndLabels.count { case (s, l) => s >= threshold && l == 1.0 }
        val fp = scoresAndLabels.count { case (s, l) => s >= threshold && l == 0.0 }
        val fn = scoresAndLabels.count { case (s, l) => s < threshold && l == 1.0 }
        val p = if (tp + fp > 0) tp.toDouble / (tp + fp) else 0.0
        val r = if (tp + fn > 0) tp.toDouble / (tp + fn) else 0.0
        val f1 = if (p + r > 0) 2.0 * p * r / (p + r) else 0.0
        (threshold, f1)
      }.maxBy(_._2)

      //print out validation threshold and validation F1
      println()
      println(s"Best threshold: ${best._1}")
      println(s"Best validation F1: ${best._2 * 100}%")
      println()

      //With the best threshold, evaluate the model on the test set.
      val bestThreshold = best._1
      val scoresAndLabelsTest = model.transform(testDF)
        .select("probability", "label")
        .rdd
        .map { row =>
          (row.getAs[MLVector]("probability")(1), row.getAs[Double]("label"))
        }.cache()

      val predictions = scoresAndLabelsTest.map { case (score, label) =>
        (if (score >= bestThreshold) 1.0 else 0.0, label)
      }.cache()

      //ROC-AUC and PR-AUC use the continuous scores, not the thresholded predictions
      val binaryMetrics = new BinaryClassificationMetrics(scoresAndLabelsTest)
      val rocAuc = binaryMetrics.areaUnderROC()
      val prAuc = binaryMetrics.areaUnderPR()

      // Calculate the confusion matrix.
      val truePositives = predictions.filter{case (prediction, label) => prediction == 1.0 && label == 1.0}.count()

      val falsePositives = predictions.filter{case (prediction, label) => prediction == 1.0 && label == 0.0}.count()

      val trueNegatives = predictions.filter{case (prediction, label) => prediction == 0.0 && label == 0.0}.count()

      val falseNegatives = predictions.filter{case (prediction, label) => prediction == 0.0 && label == 1.0}.count()

      // Calculate fraud-class metrics.

      val accuracy = ((truePositives + trueNegatives).toDouble / (truePositives + trueNegatives + falsePositives + falseNegatives))

      val fraudPrecision = if (truePositives + falsePositives > 0) truePositives.toDouble / (truePositives + falsePositives) else 0.0

      val fraudRecall = if (truePositives + falseNegatives > 0) truePositives.toDouble / (truePositives + falseNegatives) else 0.0

      val fraudF1 = if (fraudPrecision + fraudRecall > 0) 2.0 * fraudPrecision * fraudRecall / (fraudPrecision + fraudRecall) else 0.0

      println("\n========== XGBOOST RESULTS ==========")
      println(f"Accuracy:       ${accuracy * 100}%.4f%%")
      println()
      println("Fraud-class metrics:")
      println(f"Precision:      ${fraudPrecision * 100}%.4f%%")
      println(f"Recall:         ${fraudRecall * 100}%.4f%%")
      println(f"F1 score:       ${fraudF1 * 100}%.4f%%")
      println(f"ROC-AUC:        $rocAuc%.6f")
      println(f"PR-AUC:         $prAuc%.6f")
      println()
      println("Confusion matrix:")
      println(f"True positives (fraud detected):     $truePositives%d")
      println(f"False positives (legitimate flagged): $falsePositives%d")
      println(f"True negatives (legitimate cleared): $trueNegatives%d")
      println(f"False negatives (fraud missed):      $falseNegatives%d")
      println("=====================================")
    } finally {
      spark.stop()
    }
  }
}

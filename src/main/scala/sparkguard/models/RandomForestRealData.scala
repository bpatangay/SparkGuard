package sparkguard
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
object RandomForestRealData {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("SparkGuard Random Forest - Real Dataset")
      .master("local[*]")
      .getOrCreate()
    import spark.implicits._
    spark.sparkContext.setLogLevel("ERROR")
    try {
      println("Loading credit card fraud dataset...")
      val data = spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv("data/creditcard.csv")
      println(s"Total transactions: ${data.count()}")
      println("Dataset columns:")
      data.printSchema()
      // Separate the target label from the feature columns.
      val featureColumns = data.columns.filter(_ != "Class")
      // Assemble transaction columns into Spark ML features.
      val assembler = new VectorAssembler()
        .setInputCols(featureColumns)
        .setOutputCol("features")
      val assembledData = assembler.transform(data)
        .select("features", "Class")
        .withColumnRenamed("Class", "label")
      // This split is for a standalone local experiment.
      // Coordinate with the team before replacing shared splits.
      val Array(trainingData, testData) =
        assembledData.randomSplit(Array(0.8, 0.2), seed = 42L)
      println(s"Training rows: ${trainingData.count()}")
      println(s"Test rows: ${testData.count()}")
      // Train the Random Forest model.
      val classifier = FraudDetectionModel.buildClassifier(
        labelCol = "label",
        numTrees = 100,
        maxDepth = 10
      )
      println("Training Random Forest...")
      val model = classifier.fit(trainingData)
      println("Generating test predictions...")
      val predictions = model.transform(testData)
      println("Sample predictions:")
      predictions
        .select("label", "prediction", "probability")
        .show(10, truncate = false)
      // Overall accuracy.
      // Evaluate overall accuracy.
      val accuracyEvaluator =
        new MulticlassClassificationEvaluator()
          .setLabelCol("label")
          .setPredictionCol("prediction")
          .setMetricName("accuracy")

      val accuracy = accuracyEvaluator.evaluate(predictions)

      // Calculate the confusion matrix.
      val truePositives = predictions
        .filter($"label" === 1.0 && $"prediction" === 1.0)
        .count()

      val falsePositives = predictions
        .filter($"label" === 0.0 && $"prediction" === 1.0)
        .count()

      val trueNegatives = predictions
        .filter($"label" === 0.0 && $"prediction" === 0.0)
        .count()

      val falseNegatives = predictions
        .filter($"label" === 1.0 && $"prediction" === 0.0)
        .count()

      // Calculate fraud-class metrics.
      val fraudPrecision =
        if (truePositives + falsePositives > 0)
          truePositives.toDouble / (truePositives + falsePositives)
        else 0.0

      val fraudRecall =
        if (truePositives + falseNegatives > 0)
          truePositives.toDouble / (truePositives + falseNegatives)
        else 0.0

      val fraudF1 =
        if (fraudPrecision + fraudRecall > 0)
          2.0 * fraudPrecision * fraudRecall /
            (fraudPrecision + fraudRecall)
        else 0.0

      println("\n========== RANDOM FOREST RESULTS ==========")
      println(f"Accuracy:       ${accuracy * 100}%.4f%%")
      println()
      println("Fraud-class metrics:")
      println(f"Precision:      ${fraudPrecision * 100}%.4f%%")
      println(f"Recall:         ${fraudRecall * 100}%.4f%%")
      println(f"F1 score:       ${fraudF1 * 100}%.4f%%")
      println()
      println("Confusion matrix:")
      println(f"True positives (fraud detected):     $truePositives%d")
      println(f"False positives (legitimate flagged): $falsePositives%d")
      println(f"True negatives (legitimate cleared): $trueNegatives%d")
      println(f"False negatives (fraud missed):      $falseNegatives%d")
      println("===========================================")

    } finally {
      spark.stop()
    }
  }
}

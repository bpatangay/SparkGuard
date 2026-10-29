
//Imports
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.linalg.{Vector => MLVector}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

//Pull dataset files from parquets created by Data Engineering File
val trainDF = spark.read.parquet("data/trainingData.parquet")
val validationDF = spark.read.parquet("data/validationData.parquet")
val testDF = spark.read.parquet("data/testData.parquet")

//Convert dataframes into LabeledPoint RDDs for the SVM
val trainData = trainDF.rdd.map {row => 
	val label = row.getAs[Int]("label").toDouble
	val features = row.getAs[MLVector]("features")
	
	LabeledPoint(label, Vectors.dense(features.toArray))
}

val validationData = validationDF.rdd.map {row => 
	val label = row.getAs[Int]("label").toDouble
	val features = row.getAs[MLVector]("features")
	
	LabeledPoint(label, Vectors.dense(features.toArray))
}
val testData = testDF.rdd.map {row => 
	val label = row.getAs[Int]("label").toDouble
	val features = row.getAs[MLVector]("features")
	
	LabeledPoint(label, Vectors.dense(features.toArray))
}




//Train the model
val iterations = 100 
val model = SVMWithSGD.train(trainData, iterations)

//Clear Threshold
model.clearThreshold()


//Collect all the raw scores and true labels for the data points
val scoresAndLabels = validationData.map { point =>
  (model.predict(point.features), point.label)
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
model.setThreshold(best._1)


//With the threshold in place, evaluate the model on the test set. 

val predictions = testData.map { 
	point => val prediction = model.predict(point.features)
	(prediction, point.label)
	}

// Calculate the confusion matrix.
val truePositives = predictions.filter{case (prediction, label) => prediction == 1.0 && label == 1.0}.count()


val falsePositives =predictions.filter{case (prediction, label) => prediction == 1.0 && label == 0.0}.count()

val trueNegatives = predictions.filter{case (prediction, label) => prediction == 0.0 && label == 0.0}.count()

val falseNegatives = predictions.filter{case (prediction, label) => prediction == 0.0 && label == 1.0}.count()

// Calculate fraud-class metrics.

val accuracy = ((truePositives + trueNegatives).toDouble / (truePositives + trueNegatives + falsePositives + falseNegatives))

val fraudPrecision = if (truePositives + falsePositives > 0) truePositives.toDouble / (truePositives + falsePositives)else 0.0

val fraudRecall =if (truePositives + falseNegatives > 0) truePositives.toDouble / (truePositives + falseNegatives) else 0.0

val fraudF1 = if (fraudPrecision + fraudRecall > 0) 2.0 * fraudPrecision * fraudRecall /(fraudPrecision + fraudRecall) else 0.0

  println("\n========== SVM RESULTS ==========")
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
  println("====================================")


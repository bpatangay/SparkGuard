package sparkguard
import org.apache.spark.ml.classification.{
  RandomForestClassificationModel,
  RandomForestClassifier
}
import org.apache.spark.sql.DataFrame
object FraudDetectionModel {
  /**
   * Builds a Random Forest classifier for fraud detection.
   *
   * The input DataFrame must contain:
   * - features: a vector of assembled transaction features
   * - label: the target class (0 = legitimate, 1 = fraudulent)
   *
   * @param labelCol name of the target column
   * @param numTrees number of trees in the forest
   * @param maxDepth maximum depth of each tree
   * @return configured Random Forest classifier
   */
  def buildClassifier(
                       labelCol: String = "label",
                       numTrees: Int = 100,
                       maxDepth: Int = 10
                     ): RandomForestClassifier = {
    new RandomForestClassifier()
      .setLabelCol(labelCol)
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setProbabilityCol("probability")
      .setNumTrees(numTrees)
      .setMaxDepth(maxDepth)
      .setSeed(42)
  }
  /**
   * Trains the Random Forest using prepared training data.
   *
   * @param trainingData training DataFrame containing features and label
   * @param labelCol name of the target column
   * @return trained Random Forest model
   */
  def train(
             trainingData: DataFrame,
             labelCol: String = "label"
           ): RandomForestClassificationModel = {
    val classifier = buildClassifier(
      labelCol = labelCol
    )
    classifier.fit(trainingData)
  }
  /**
   * Generates predictions for a prepared DataFrame.
   *
   * @param model trained Random Forest model
   * @param data DataFrame containing the features column
   * @return DataFrame with prediction and probability columns
   */
  def predict(
               model: RandomForestClassificationModel,
               data: DataFrame
             ): DataFrame = {
    model.transform(data)
  }
}
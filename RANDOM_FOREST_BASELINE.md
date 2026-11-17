SparkGuard — Random Forest Baseline Results

1. Overview

This experiment establishes the Random Forest baseline for the SparkGuard credit card fraud detection project.

The model was implemented in Scala using Apache Spark MLlib and evaluated on the Kaggle Credit Card Fraud Detection dataset.

2. Experimental Setup

* Model: Random Forest Classifier
* Dataset: Credit Card Fraud Detection dataset
* Total transactions: 284,807
* Training/validation/test split: 70% / 15% / 15%
* Random seed: 1234
* Training rows: 199,586
* Validation rows: 42,619
* Test rows: 42,602
* Number of trees: 100
* Maximum tree depth: 10
* Execution environment: Local Mac, Apache Spark 3.5.3, Scala 2.12.18, Java 17
* Execution time: Approximately 43 seconds

The model was trained using the training set and evaluated using the held-out test set. The validation set was created but not used for hyperparameter tuning in this experiment.

3. Results

Metric	Result
Accuracy	99.9484%
ROC-AUC	0.9649
PR-AUC	0.7497
Fraud precision	92.3077%
Fraud recall	72.7273%
Fraud F1 score	81.3559%

Confusion Matrix

Outcome	Count
True positives — fraud detected	48
False positives — legitimate transactions flagged	4
True negatives — legitimate transactions cleared	42,532
False negatives — fraud missed	18

4. Interpretation

The Random Forest model achieved a ROC-AUC of 0.9649, indicating strong ability to distinguish fraudulent from legitimate transactions across classification thresholds.

The PR-AUC of 0.7497 provides an additional measure of performance that is particularly relevant to this highly imbalanced fraud detection dataset.

The model correctly identified 48 fraudulent transactions and missed 18. Four legitimate transactions were incorrectly flagged as fraudulent.

Although accuracy was 99.9484%, accuracy alone is insufficient for evaluating fraud detection performance. Precision, recall, F1-score, ROC-AUC, and PR-AUC provide a more complete evaluation.

5. Limitations and Next Steps

The validation partition was not used for hyperparameter tuning, and no class weighting was applied in this experiment.

These results establish a standalone baseline. Comparisons with other project models should use consistent data partitions and evaluation procedures.

Future work should include:

* Hyperparameter tuning using the validation set.
* Evaluating class weighting to address class imbalance.
* Comparing Random Forest with SVM, XGBoost, Neural Networks, and the Voting Ensemble.
* Comparing ROC-AUC and PR-AUC across all models.
* Measuring training execution time and scalability.
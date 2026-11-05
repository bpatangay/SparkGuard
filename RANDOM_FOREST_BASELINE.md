SparkGuard — Random Forest Baseline Results

1. Overview

This experiment establishes the Random Forest baseline for the SparkGuard credit card fraud detection project. The model was implemented in Scala using Apache Spark MLlib and evaluated on the Kaggle Credit Card Fraud Detection dataset (creditcard.csv).

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
* Execution time: Approximately 46 seconds

The model was trained using the training set and evaluated using the held-out test set. The validation set was created and counted but was not used for hyperparameter tuning in this baseline experiment.

These results come from a standalone local experiment and should not be treated as final team-wide comparative results until all models use the same shared data partitions and evaluation procedure.

3. Results

Metric	Result
Accuracy	99.9484%
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

The model correctly identified 48 fraudulent transactions and missed 18 fraudulent transactions in the test set. Four legitimate transactions were incorrectly flagged as fraudulent.

Although overall accuracy was 99.9484%, accuracy alone is insufficient for evaluating this highly imbalanced dataset. Fraud precision, recall, and F1 score provide more meaningful information about fraud detection performance.

In this experiment, the model achieved 92.3077% fraud precision and 72.7273% fraud recall, resulting in an F1 score of 81.3559%.

5. Limitations and Next Steps

The validation partition was not used for tuning in this experiment, and no class weighting was applied.

The results establish a standalone baseline. The model should be compared with the other project models only after confirming that they use the same data partitions and evaluation procedure.

Future work should use the validation set for model selection and reserve the test set for final evaluation. Metrics such as PR-AUC and ROC-AUC should also be reported where available.
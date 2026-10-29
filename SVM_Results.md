SparkGuard - SVM Results

1. Overview

This experiment establishes the effectiveness of the SVM model for the SparkGuard credit card fraud detection project.

The model was implemented in Scala using Apache Spark MLLib and evaluted on the credit card fraud dataset

2. Experimental setup

* Model: Support Vector Machine
* Dataset: Credit card fraud dataset
* Training/validation/test split: 70% / 15% / 15%
* Random seed: 1234L
* Number of Iterations: 100

3. Results
Accuracy: 99.9297%
Precision: 84.8837%
Recall: 81.1111%
F1 Score: 82.9545%

Confusion Matrix:
True Positives (Fraud detected): 73
False Positives (Legitimate flagged): 13
True Negatives (Legitimate cleared): 42594
False Negatives (Fraud missed): 17

4. Interpretation

The model correctly identified 72 fraudulent transactions and missed 17 fraudulent transactions in the test set. 13 legitimate transactions were incorrectly flagged as fraud.

Although accuracy is extremely high at 99.9297%, this score alone is misleading considering the very large class imbalance betweenr legitimate and fraudulent transactions. As such, metrics such as precision, recall, and F1 score provide more meaningful insights into this model's performance.

5. Next Steps

While the general effectiveness metrics for the SVM approach have been gathered, these results need to be compared with the results of the other models to determine the overall most effective solution.

SparkGuard — Random Forest Baseline Results

1. Overview

This experiment establishes the initial Random Forest baseline for the SparkGuard credit card fraud detection project.

The model was implemented in Scala using Apache Spark MLlib and evaluated on the credit card fraud dataset.

2. Experimental Setup

* Model: Random Forest Classifier
* Dataset: Credit card fraud dataset
* Training/test split: 80% / 20%
* Random seed: 42
* Number of trees: 100
* Maximum tree depth: 10
* Execution environment: Local Mac
* Execution time: Approximately 43 seconds

These results come from an independent local experiment and have not yet been evaluated on the team’s shared test dataset.

3. Results

Metric	Result
Accuracy	99.9542%
Fraud precision	92.3077%
Fraud recall	78.2609%
Fraud F1 score	84.7059%

Confusion Matrix

Outcome	Count
True positives — fraud detected	72
False positives — legitimate transactions flagged	6
True negatives — legitimate transactions cleared	56,664
False negatives — fraud missed	20

4. Interpretation

The model correctly identified 72 fraudulent transactions and missed 20 fraudulent transactions in the test set. Six legitimate transactions were incorrectly flagged as fraudulent.

Although overall accuracy was 99.9542%, accuracy alone is insufficient for evaluating this highly imbalanced dataset. Fraud precision, recall, and F1 score provide more meaningful information about fraud detection performance.

5. Limitations and Next Steps

These results establish a preliminary baseline. The model should be evaluated on the team’s agreed test dataset before its performance is compared directly with the other models.

Future comparisons should use consistent data splits and evaluation metrics across all models.
SparkGuard - XGBoost Results

1. Overview

This experiment establishes the effectiveness of the XGBoost model for the SparkGuard credit card fraud detection project.

The model was implemented in Scala using Apache Spark and XGBoost4J-Spark and evaluated on the credit card fraud dataset.

2. Experimental setup

* Model: XGBoost (gradient-boosted trees)
* Dataset: Credit card fraud dataset
* Training/validation/test split: 70% / 15% / 15%
* Random seed: 1234L (split), 1234 (model)
* Number of Iterations: 200 boosting rounds (max depth 6, learning rate 0.1, class weight scale_pos_weight)
* Decision threshold: 0.9974, tuned on the validation set by fraud-class F1

3. Results
Accuracy: 99.9413%
Precision: 93.6170%
Recall: 66.6667%
F1 Score: 77.8761%
ROC-AUC: 0.956209
PR-AUC: 0.791736

Confusion Matrix:
True Positives (Fraud detected): 44
False Positives (Legitimate flagged): 3
True Negatives (Legitimate cleared): 42533
False Negatives (Fraud missed): 22

4. Interpretation

The model correctly identified 44 fraudulent transactions and missed 22 fraudulent transactions in the test set. 3 legitimate transactions were incorrectly flagged as fraud.

Precision is high, but recall is lower, so the model misses a notable share of fraud. As a reference, an earlier run with the default 0.5 threshold gave Precision 91.2281%, Recall 78.7879%, and F1 84.5528%. Accuracy alone is misleading because of the large class imbalance.

Note: this test set had 42,602 rows (66 fraud), which differs from the SVM test set size, so the models were not scored on identical rows.

5. Next Steps

Compare these results with the other models on the same test split to determine the most effective solution.


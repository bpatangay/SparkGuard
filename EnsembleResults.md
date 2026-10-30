# SparkGuard - Neural Network Results

## 1. Overview

This experiment evaluates the Voting Ensemble  Approach for the SparkGuard
credit card fraud detection project.

The model consists of three sub-models which work together to vote on a particular classification: SVM, XGBoost, and a Neural Network. 
It was implemented in Scala using Apache Spark MLlib and XGBoost4J-Spark, and tested on the credit card fraud dataset. 

## 2. Experimental Setup
#General Information
-   Model: Voting Ensemble (Support Vector Machine, XGBoost, Neural Network)
-   Dataset: Credit card fraud dataset
-   Training/validation/test split: 70% / 15% / 15%
-   Random seed: 1234L
-   Training samples: 199,515
-   Validation samples: 42,581
-   Test samples: 42,711

#Neural Network Setup
-   Neural Network layers: 30 → 64 → 32 → 2
-   Number of iterations: 100
-   Selected classification threshold: 0.60

#Support Vector Machine Setup
-   Number of Iterations: 100

#XGboost Setup
-   Random seed:1234 (model)
-   Number of Iterations: 200 boosting rounds (max depth 6, learning rate 0.1, class weight scale_pos_weight)

## 3. Results

-   Accuracy: 99.9508%
-   Precision: 93.6709%
-   Recall: 82.2222%
-   F1 Score: 87.5740%

**Confusion Matrix:**

-   True Positives (Fraud detected): 74
-   False Positives (Legitimate flagged): 5
-   True Negatives (Legitimate cleared): 42,602
-   False Negatives (Fraud missed): 16

## 4. Interpretation

The Voting Ensemble correctly detected 74 fraudulent transactions and
missed 16 fraud cases in the test set. It also incorrectly flagged 5
legitimate transactions as fraud.

The model achieved 99.9508% accuracy, but accuracy alone does not tell
the full story because the dataset contains a much larger number of legitimate transactions.
The precision of 93.6709% shows that most transactions flagged as fraud
were actually fraudulent. However, the recall of 82.2222% shows that the
model still missed some fraud cases. The F1 score gives a more balanced
view of its performance.

## 5. Next Steps

The results from the Voting Ensemble will be compared with the other models in the
project. 
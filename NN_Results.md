# SparkGuard - Neural Network Results

## 1. Overview

This experiment evaluates the Neural Network model for the SparkGuard
credit card fraud detection project.

The model was implemented in Scala using Apache Spark MLlib and tested
on the credit card fraud dataset.

## 2. Experimental Setup

-   Model: Neural Network (Multilayer Perceptron)
-   Dataset: Credit card fraud dataset
-   Training/validation/test split: 70% / 15% / 15%
-   Network layers: 30 → 64 → 32 → 2
-   Number of iterations: 100
-   Selected classification threshold: 0.60
-   Training samples: 199,515
-   Validation samples: 42,581
-   Test samples: 42,711
-   Training time: 71.41 seconds

## 3. Results

-   Accuracy: 99.9251%
-   Precision: 88.7324%
-   Recall: 72.4138%
-   F1 Score: 79.7468%
-   ROC-AUC: 0.928420
-   PR-AUC: 0.755943

**Confusion Matrix:**

-   True Positives (Fraud detected): 63
-   False Positives (Legitimate flagged): 8
-   True Negatives (Legitimate cleared): 42,616
-   False Negatives (Fraud missed): 24

## 4. Interpretation

The Neural Network correctly detected 63 fraudulent transactions and
missed 24 fraud cases in the test set. It also incorrectly flagged 8
legitimate transactions as fraud.

The model achieved 99.9251% accuracy, but accuracy alone does not tell
the full story because most transactions in the dataset are legitimate.
The precision of 88.7324% shows that most transactions flagged as fraud
were actually fraudulent. However, the recall of 72.4138% shows that the
model still missed some fraud cases. The F1 score gives a more balanced
view of its performance.

## 5. Next Steps

The Neural Network results will be compared with the other models in the
project. 

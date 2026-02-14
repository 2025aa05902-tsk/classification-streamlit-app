# classification-streamlit-app
This repository contains the Implementation of multiple classification models &amp; Builds an interactive Streamlit web application to demonstrate models

### Problem statement
- Implement multiple classification models
- Build an interactive Streamlit web application to demonstrate your models 
- Deploy the app on Streamlit Community Cloud (FREE) 
- Share clickable links for evaluation

### 📂Dataset description
Name: The Adult Income Classification dataset
- The dataset predicts whether a person’s income exceeds **$50K** per year based on census data. Contains ~48,842 instances
- It contains demographic and employment-related attributes such as: ["age", "workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country"]
- Target Variable: `<=50K` and `>50K`

### Models used
1. Logistic Regression 
2. Decision Tree Classifier 
3. K-Nearest Neighbor Classifier 
4. Naive Bayes Classifier - Gaussian or Multinomial 
5. Ensemble Model - Random Forest 
6. Ensemble Model - XGBoost

| ML Model Name                                              | Training Time | Accuracy  | AUC      | Precision | Recall   | F1       | MCC      |
|---------------------------------------------------|--------------|----------|----------|----------|----------|----------|----------|
| Logistic Regression                               | 1.798065     | 0.844444 | 0.901550 | 0.732591 | 0.586530 | 0.651474 | 0.558591 |
| Decision Tree Classifier                          | 0.411856     | 0.847098 | 0.884649 | 0.746414 | 0.580285 | 0.652949 | 0.564119 |
| K-Nearest Neighbor Classifier                     | 0.026320     | 0.829851 | 0.876465 | 0.685097 | 0.580285 | 0.628351 | 0.522008 |
| Naive Bayes Classifier - Gaussian or Multinomial  | 0.098686     | 0.565616 | 0.806723 | 0.357877 | 0.947368 | 0.519506 | 0.351953 |
| Ensemble Model - Random Forest                    | 8.425462     | 0.858043 | 0.915644 | 0.777199 | 0.599019 | 0.676574 | 0.595758 |
| Ensemble Model - XGBoost                          | 4.226436     | 0.869320 | 0.926285 | 0.788986 | 0.645406 | 0.710010 | 0.632021 |


### 📊 Model Performance Summary and Observations

| ML Model            | Observation about Model Performance |
|---------------------|-------------------------------------|
| Logistic Regression | Serves as a strong baseline with **84.44% accuracy** and **0.90 AUC**, indicating good class separation. Precision (0.73) is higher than recall (0.59), meaning it is more conservative in predicting high income (>50K). |
| Decision Tree       | Slightly improves accuracy to **84.71%**, but lower AUC (0.88) suggests weaker probability ranking compared to Logistic Regression. Similar recall (~0.58) indicates limited improvement in capturing positive cases. |
| KNN                 | Achieves **82.99% accuracy** and **0.88 AUC**, but lower precision (0.69) and F1-score (0.63) suggest less balanced performance. Very fast training time, but may scale poorly during prediction. |
| Naive Bayes         | Despite a high recall of **0.95**, accuracy drops significantly to **56.56%** with very low precision (0.36). This indicates excessive false positives, making it less reliable despite capturing most positive cases. |
| Random Forest       | Improves performance with **85.80% accuracy** and **0.92 AUC**. Better precision (0.78) and improved MCC (0.60) show stronger overall classification balance compared to single-tree models. |
| XGBoost             | Best overall model with **86.93% accuracy**, highest **AUC (0.93)**, and strongest **F1-score (0.71)**. Balanced precision (0.79) and recall (0.65) indicate robust and well-generalized performance. |

Link to [Classification Model App]("https://classification-app-2025aa05902.streamlit.app/")

# Predicting Online Shoppers Purchasing Intention with Machine Learning

Overview:
This repository contains a detailed implementation of three different machine learning models - Logistic Regression, Random Forest, and XGBoost - to predict whether an online session will end in a purchase or not based on user session data. The project aims to explore various aspects of machine learning, including data preprocessing, model building, feature selection, hyperparameter tuning, and model evaluation.

Dataset:
The dataset used for this project is the ‘Online Shoppers Purchasing Intention Dataset’ from UCI ML Repository, which contains various features of online shopping sessions. It has 12,330 entries, each representing an online browsing session. It includes both numerical and categorical data across 18 columns, with the target variable being the Revenue boolean column, indicating whether a session resulted in a purchase.

Model Implementation:
-	Logistic Regression: Standard logistic regression model with class weight adjustment, important feature importance analysis, and hyperparameter tuning.
-	Random Forest: Ensemble method with class weight adjustment, feature importance analysis, and hyperparameter tuning.
-	XGBoost: Gradient boosting model with class weight adjustment, feature importance analysis, and hyperparameter tuning.

Feature Importance: 
Reduced feature sets to improve model performance and simplify the models.

Model Evaluation: 
Each model is evaluated based on accuracy, sensitivity, and specificity. Cross-validation is performed to assess model stability.



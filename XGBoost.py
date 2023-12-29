import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel


# Load the dataset
file_path = 'data/online_shoppers_intention.csv'
dataset = pd.read_csv(file_path)

# Displaying basic information about the dataset
dataset_info = dataset.info()
dataset_description = dataset.describe()
first_few_rows = dataset.head()

# Checking for missing values
missing_values = dataset.isnull().sum()

print(dataset_info, dataset_description.to_string(), '\n', first_few_rows.to_string(), '\n', missing_values)

# Defining the categorical and numerical features
categorical_features = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']
numerical_features = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
                      'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']


dataset_num = dataset.drop(['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend'], axis=1)

# Check for class imbalance in the target variable
class_distribution = dataset['Revenue'].value_counts(normalize=True)

# Analyze correlations
correlation_matrix = dataset_num.corr()

# Explore skewness in numerical features
skewness = dataset[numerical_features].skew()

print(class_distribution, '\n', correlation_matrix.to_string(), skewness)

# Function to calculate sensitivity and specificity from the confusion matrix
def calc_sensitivity_specificity(conf_matrix):
    TN, FP, FN, TP = conf_matrix.ravel()
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    return sensitivity, specificity


# Preprocessing:
X = dataset.drop('Revenue', axis=1)
y = dataset['Revenue']

# Encoding categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Apply preprocessing to categorical features and concatenate with numerical features
X_categorical_processed = preprocessor.fit_transform(X[categorical_features])
X_numerical = X[numerical_features]

# Convert processed categorical features back to DataFrame
feature_names_categorical = preprocessor.get_feature_names_out()
X_categorical = pd.DataFrame(X_categorical_processed.toarray(), columns=feature_names_categorical)

# Concatenate processed categorical features with numerical features
X_processed = pd.concat([X_numerical.reset_index(drop=True), X_categorical.reset_index(drop=True)], axis=1)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=0)

# Training the XGB model
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Predicting and evaluating
y_pred_xgb = xgb_model.predict(X_test)
accuracy_XGB, conf_matrix_XGB = accuracy_score(y_test, y_pred_xgb), confusion_matrix(y_test, y_pred_xgb)
sensitivity_XGB, specificity_XGB = calc_sensitivity_specificity(conf_matrix_XGB)

summary = {"XGB": {
        "Accuracy": accuracy_XGB,
        "Confusion Matrix": conf_matrix_XGB,
        "Sensitivity": sensitivity_XGB,
        "Specificity": specificity_XGB}}

print(summary)


# Train the XGB model with balanced class
class_counts = y_train.value_counts()
scale_pos_weight = class_counts[0] / class_counts[1]
xgb_balanced = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
xgb_balanced.fit(X_train, y_train)

# Predicting and evaluating
y_pred_xgb_balanced = xgb_balanced.predict(X_test)
accuracy_xgb_balanced, conf_matrix_xgb_balanced = accuracy_score(y_test, y_pred_xgb_balanced), confusion_matrix(y_test, y_pred_xgb_balanced)
sensitivity_xgb_balanced, specificity_xgb_balanced = calc_sensitivity_specificity(conf_matrix_xgb_balanced)

summary_balanced = {
    "Balanced XGB": {
        "Accuracy": accuracy_xgb_balanced,
        "Confusion Matrix": conf_matrix_xgb_balanced,
        "Sensitivity": sensitivity_xgb_balanced,
        "Specificity": specificity_xgb_balanced}}

print(summary_balanced)

# Create SelectFromModel
feature_names = X_train.columns.tolist()
selector = SelectFromModel(xgb_balanced, prefit=True)
selector.feature_names_in_ = feature_names

# Transforming the datasets
X_train_top_features = selector.transform(X_train)
X_test_top_features = selector.transform(X_test)

# Number of features selected
selected_features_count_xgb = X_train_top_features.shape[1]
print("Number of selected features:", selected_features_count_xgb)

# Building XGB model using top features
xgb_balanced_top_features = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
xgb_balanced_top_features.fit(X_train_top_features, y_train)

# Predicting and evaluating
y_pred_top = xgb_balanced_top_features.predict(X_test_top_features)
accuracy_balanced_top, conf_matrix_balanced_top = accuracy_score(y_test, y_pred_top), confusion_matrix(y_test, y_pred_top)
sensitivity_balanced_top, specificity_balanced_top = calc_sensitivity_specificity(conf_matrix_balanced_top)

summary_balanced_selected_features = {
    "Balanced XGB Top Features": {
        "Accuracy": accuracy_balanced_top,
        "Confusion Matrix": conf_matrix_balanced_top,
        "Sensitivity": sensitivity_balanced_top,
        "Specificity": specificity_balanced_top}}

print(summary_balanced_selected_features)



# Define the parameter grid
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'n_estimators': [100, 200, 300],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 1.5, 2]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=xgb_balanced, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(X_train_top_features, y_train)

# Best parameters
print("Best parameters found: ", grid_search.best_params_)

# Defining best parameters
best_params = {
    'colsample_bytree': 0.9,
    'learning_rate': 0.01,
    'max_depth': 5,
    'n_estimators': 300,
    'reg_alpha': 0.5,
    'reg_lambda': 1.5,
    'subsample': 0.7
}

# Training the XGB model with the best parameters
xgb_tuned = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight, **best_params)
xgb_tuned.fit(X_train_top_features, y_train)

# Predicting and evaluating
y_pred_tuned = xgb_tuned.predict(X_test_top_features)
accuracy_tuned, conf_matrix_tuned = accuracy_score(y_test, y_pred_tuned), confusion_matrix(y_test, y_pred_tuned)
sensitivity_tuned, specificity_tuned = calc_sensitivity_specificity(conf_matrix_tuned)

summary_tuned = {
    "Balanced XGB Top Features Tuned": {
        "Accuracy": accuracy_tuned,
        "Confusion Matrix": conf_matrix_tuned,
        "Sensitivity": sensitivity_tuned,
        "Specificity": specificity_tuned}}

print(summary_tuned)


# Perform cross-validation
cv_scores_balanced = cross_val_score(xgb_tuned, X_train_top_features, y_train, cv=5)

cv_results = {"Cross Validation Results": {
        "CV Mean Accuracy": cv_scores_balanced.mean(),
        "CV Std Accuracy": cv_scores_balanced.std()}}

print(cv_results)
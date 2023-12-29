import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
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

# Training the Random Forest model
rf_model = RandomForestClassifier(random_state=0)
rf_model.fit(X_train, y_train)

# Predicting and evaluating
y_pred_rf = rf_model.predict(X_test)
accuracy_rf, conf_matrix_rf = accuracy_score(y_test, y_pred_rf), confusion_matrix(y_test, y_pred_rf)
sensitivity_rf, specificity_rf = calc_sensitivity_specificity(conf_matrix_rf)

summary = {"Random Forest": {
        "Accuracy": accuracy_rf,
        "Confusion Matrix": conf_matrix_rf,
        "Sensitivity": sensitivity_rf,
        "Specificity": specificity_rf}}

print(summary)


# Adjusting the Random Forest model with balanced class weights
rf_balanced = RandomForestClassifier(class_weight='balanced', random_state=0)
rf_balanced.fit(X_train, y_train)

# Predicting and evaluating
y_pred_rf_balanced = rf_balanced.predict(X_test)
accuracy_rf_balanced, conf_matrix_rf_balanced = accuracy_score(y_test, y_pred_rf_balanced), confusion_matrix(y_test, y_pred_rf_balanced)
sensitivity_rf_balanced, specificity_rf_balanced = calc_sensitivity_specificity(conf_matrix_rf_balanced)

summary_balanced = {"Balanced Random Forest": {
        "Accuracy": accuracy_rf_balanced,
        "Confusion Matrix": conf_matrix_rf_balanced,
        "Sensitivity": sensitivity_rf_balanced,
        "Specificity": specificity_rf_balanced}}

print(summary_balanced)

# Create SelectFromModel
feature_names = X_train.columns.tolist()
selector_rf = SelectFromModel(rf_balanced, prefit=True)
selector_rf.feature_names_in_ = feature_names

# Transforming the datasets
X_train_top_features = selector_rf.transform(X_train)
X_test_top_features = selector_rf.transform(X_test)

# Number of features selected
selected_features_count = X_train_top_features.shape[1]
print("Number of selected features", selected_features_count)

# Building Random Forest model using top features
rf_top_features = RandomForestClassifier(class_weight='balanced', random_state=0)
rf_top_features.fit(X_train_top_features, y_train)

# Predicting and evaluating
y_pred_rf = rf_top_features.predict(X_test_top_features)
accuracy_rf_top, conf_matrix_rf_top = accuracy_score(y_test, y_pred_rf), confusion_matrix(y_test, y_pred_rf)
sensitivity_rf_top, specificity_rf_top = calc_sensitivity_specificity(conf_matrix_rf)

summary_balanced_top_features = {"Balanced Random Forest Top Features": {
        "Accuracy": accuracy_rf_top,
        "Confusion Matrix": conf_matrix_rf_top,
        "Sensitivity": sensitivity_rf_top,
        "Specificity": specificity_rf_top}}

print(summary_balanced_top_features)


# Hyperparameter tuning
# Setting up hyperparameter grids
param_grid_rf = {'n_estimators': [50, 100, 150],'max_depth': [10, 20, None],'min_samples_split': [2, 5, 10]}

# Setting up GridSearchCV
grid_search_rf = GridSearchCV(rf_balanced, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_rf.fit(X_train_top_features, y_train)

# Best parameters
best_params_rf = grid_search_rf.best_params_
print(best_params_rf)

# Defining best parameters
best_params_rf = {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 100}

# Training the Random Forest model with the best parameters
rf_tuned = RandomForestClassifier(class_weight='balanced', **best_params_rf, random_state=0)
rf_tuned.fit(X_train_top_features, y_train)

# Evaluating the tuned Random Forest model
y_pred_rf_tuned = rf_tuned.predict(X_test_top_features)
accuracy_rf_tuned, conf_matrix_rf_tuned = accuracy_score(y_test, y_pred_rf_tuned), confusion_matrix(y_test, y_pred_rf_tuned)
sensitivity_rf_tuned, specificity_rf_tuned = calc_sensitivity_specificity(conf_matrix_rf_tuned)

# Creating a summary of the evaluation metrics for the tuned models
summary_tuned = {"Balanced Random Forest Top Features Tuned": {
        "Accuracy": accuracy_rf_tuned,
        "Confusion Matrix": conf_matrix_rf_tuned,
        "Sensitivity": sensitivity_rf_tuned,
        "Specificity": specificity_rf_tuned}}

print(summary_tuned)


# Performing cross-validation
cv_scores_balanced_rf = cross_val_score(rf_tuned, X_train_top_features, y_train, cv=5, scoring='accuracy')

cv_results = {"Cross Validation Results": {
        "CV Mean Accuracy": cv_scores_balanced_rf.mean(),
        "CV Std Accuracy": cv_scores_balanced_rf.std()}}

print(cv_results)
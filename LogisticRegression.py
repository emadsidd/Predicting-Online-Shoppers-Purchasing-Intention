import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
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

# Encoding categorical features and scaling numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', PowerTransformer(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Apply preprocessing and split the dataset
X_processed = preprocessor.fit_transform(X)
X_train_p, X_test_p, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=0)


# Convert X_train and X_test back to DataFrame
feature_names = preprocessor.get_feature_names_out()
X_train = pd.DataFrame(X_train_p.toarray(), columns=feature_names)
X_test = pd.DataFrame(X_test_p.toarray(), columns=feature_names)

# Training the Logistic Regression model
logreg_model = LogisticRegression(max_iter=10000, random_state=0)
logreg_model.fit(X_train, y_train)

# Predicting and evaluating
y_pred_logreg = logreg_model.predict(X_test)
accuracy_logreg, conf_matrix_logreg = accuracy_score(y_test, y_pred_logreg), confusion_matrix(y_test, y_pred_logreg)
sensitivity_logreg, specificity_logreg = calc_sensitivity_specificity(conf_matrix_logreg)

summary = {"Logistic Regression": {
        "Accuracy": accuracy_logreg,
        "Confusion Matrix": conf_matrix_logreg,
        "Sensitivity": sensitivity_logreg,
        "Specificity": specificity_logreg}}

print(summary)


# Adjusting the Logistic Regression model with balanced class weights
logreg_balanced = LogisticRegression(max_iter=10000, class_weight='balanced', random_state=0)
logreg_balanced.fit(X_train, y_train)

# Predicting and evaluating
y_pred_logreg_balanced = logreg_balanced.predict(X_test)
accuracy_logreg_balanced, conf_matrix_logreg_balanced = accuracy_score(y_test, y_pred_logreg_balanced), confusion_matrix(y_test, y_pred_logreg_balanced)
sensitivity_logreg_balanced, specificity_logreg_balanced = calc_sensitivity_specificity(conf_matrix_logreg_balanced)

summary_balanced = {"Balanced Logistic Regression": {
        "Accuracy": accuracy_logreg_balanced,
        "Confusion Matrix": conf_matrix_logreg_balanced,
        "Sensitivity": sensitivity_logreg_balanced,
        "Specificity": specificity_logreg_balanced}}

print(summary_balanced)


# Create SelectFromModel
feature_names = X_train.columns.tolist()
selector_logreg = SelectFromModel(logreg_balanced, prefit=True)
selector_logreg.feature_names_in_ = feature_names

# Transforming the datasets
X_train_top_features = selector_logreg.transform(X_train)
X_test_top_features = selector_logreg.transform(X_test)

# Get the number of features selected
selected_features_count_logreg = X_train_top_features.shape[1]
print("Number of selected features:", selected_features_count_logreg)


# Building Logistic Regression model using top features
logreg_top_features = LogisticRegression(class_weight='balanced', max_iter=10000, random_state=0)
logreg_top_features.fit(X_train_top_features, y_train)

# Predicting and evaluating
y_pred_logreg = logreg_top_features.predict(X_test_top_features)
accuracy_logreg_top, conf_matrix_logreg_top = accuracy_score(y_test, y_pred_logreg), confusion_matrix(y_test, y_pred_logreg)
sensitivity_logreg_top, specificity_logreg_top = calc_sensitivity_specificity(conf_matrix_logreg)

summary_balanced_top_features = {"Balanced Logistic Regression Top Features": {"Accuracy": accuracy_logreg_top,
        "Confusion Matrix": conf_matrix_logreg_top,
        "Sensitivity": sensitivity_logreg_top,
        "Specificity": specificity_logreg_top}}

print(summary_balanced_top_features)


# Hyperparameter tuning
# Setting up hyperparameter grids
param_grid_logreg = {'C': [0.01, 0.1, 1, 10, 100]}

# Setting up GridSearchCV
grid_search_logreg = GridSearchCV(logreg_balanced, param_grid_logreg, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_logreg.fit(X_train_top_features, y_train)

# Best parameters
best_params_logreg = grid_search_logreg.best_params_
print(best_params_logreg)

# Defining best parameters
best_params_logreg = {'C': 0.01}


# Training the Logistic Regression model with the best parameters
logreg_tuned = LogisticRegression(class_weight='balanced', C=best_params_logreg['C'], max_iter=10000, random_state=0)
logreg_tuned.fit(X_train_top_features, y_train)

# Predicting and evaluating
y_pred_logreg_tuned = logreg_tuned.predict(X_test_top_features)
accuracy_logreg_tuned, conf_matrix_logreg_tuned = accuracy_score(y_test, y_pred_logreg_tuned), confusion_matrix(y_test, y_pred_logreg_tuned)
sensitivity_logreg_tuned, specificity_logreg_tuned = calc_sensitivity_specificity(conf_matrix_logreg_tuned)

summary_tuned = {"Balanced Logistic Regression Top Features Tuned": {
        "Accuracy": accuracy_logreg_tuned,
        "Confusion Matrix": conf_matrix_logreg_tuned,
        "Sensitivity": sensitivity_logreg_tuned,
        "Specificity": specificity_logreg_tuned}}

print(summary_tuned)

# Performing cross-validation
cv_scores_balanced_logreg = cross_val_score(logreg_tuned, X_train_top_features, y_train, cv=5, scoring='accuracy')

cv_results = {"Cross Validation Results": {
        "CV Mean Accuracy": cv_scores_balanced_logreg.mean(),
        "CV Std Accuracy": cv_scores_balanced_logreg.std()}}

print(cv_results)




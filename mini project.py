# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Load the training dataset
file_path = '/kaggle/input/diabetes-classification/train.csv'
train_data = pd.read_csv(file_path)

# Identifying the columns with medically unreasonable zero values
columns_to_replace_zeros = ['glucose_concentration', 'blood_pressure', 
                            'skin_fold_thickness', 'serum_insulin', 'bmi']

# Replacing zeros with NaN for later imputation
for col in columns_to_replace_zeros:
    train_data[col].replace(0, np.nan, inplace=True)

# Checking for missing values after replacement
missing_values = train_data[columns_to_replace_zeros].isnull().sum()
missing_values_percentage = (missing_values / len(train_data)) * 100

# Imputing missing values with the mean of each column
for col in columns_to_replace_zeros:
    mean_value = train_data[col].mean()
    train_data[col].fillna(mean_value, inplace=True)

# Checking if all missing values have been imputed
imputed_missing_values = train_data[columns_to_replace_zeros].isnull().sum()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Preparing the data
X = train_data.drop(['diabetes', 'p_id'], axis=1)  # Features
y = train_data['diabetes']  # Target variable

# Splitting the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Creating a Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_scaled, y_train)

# Predicting on the validation set
y_pred = clf.predict(X_val_scaled)

# Evaluating the model
accuracy = accuracy_score(y_val, y_pred)
report = classification_report(y_val, y_pred)

import matplotlib.pyplot as plt

# Extracting feature importances from the model
feature_importances = clf.feature_importances_
features = X.columns

# Creating a DataFrame to display feature names and their importance scores
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

# Sorting the DataFrame based on feature importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plotting feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Feature Importances in Random Forest Classifier')
plt.gca().invert_yaxis()  # Invert the Y-axis for better readability
plt.show()

# Selecting top important features based on the previous feature importance analysis
# We will select the top N features. For this example, let's use the top 4 features
top_features = feature_importance_df['Feature'].head(4).tolist()

# Preparing the data with selected features
X_selected_features = train_data[top_features]
y = train_data['diabetes']

# Splitting the data into training and validation sets
X_train_sel, X_val_sel, y_train, y_val = train_test_split(X_selected_features, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler_sel = StandardScaler()
X_train_sel_scaled = scaler_sel.fit_transform(X_train_sel)
X_val_sel_scaled = scaler_sel.transform(X_val_sel)

# Creating a new Random Forest Classifier
clf_sel = RandomForestClassifier(random_state=42)
clf_sel.fit(X_train_sel_scaled, y_train)

# Predicting on the validation set
y_pred_sel = clf_sel.predict(X_val_sel_scaled)

# Evaluating the model
accuracy_sel = accuracy_score(y_val, y_pred_sel)
report_sel = classification_report(y_val, y_pred_sel)

# Load the test dataset
test_file_path = '/kaggle/input/diabetes-classification/test.csv'
test_data = pd.read_csv(test_file_path)

# Preparing the test data (excluding the patient ID)
X_test = test_data.drop(['p_id','no_times_pregnant','blood_pressure','skin_fold_thickness','serum_insulin'], axis=1)

# Standardizing the test data
X_test_scaled = scaler_sel.transform(X_test)

# Predicting diabetes for the test data
test_predictions = clf_sel.predict(X_test_scaled)

# Adding predictions to the test dataset for review
test_data['diabetes'] = test_predictions

# Formatting the predictions in the required format
submission = test_data[['p_id', 'diabetes']]

# Saving the formatted predictions to a CSV file
submission_file_path = '/kaggle/working/diabetes_predictions_submission.csv'
submission.to_csv(submission_file_path, index=False)
# Import libraries
import copy
import time
# library for math
import numpy as np
# library for math
import pandas as pd
# library for graphs
import matplotlib.pyplot as plt
# library for stats and stats visuals
import seaborn as sns
# Machine Learning in Python
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
# Import required libraries for model evaluation
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error, mean_absolute_error
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Set up MLflow for experiment tracking
import mlflow
from mlflow import MlflowClient
from mlflow.models import Model
from mlflow.models.signature import ModelSignature

"""
Define Variables
"""

EXPERIMENT_NAME = "Test"  # MLflow experiment name

"""
ML Flow: Machine Learning

Autologging in Microsoft Fabric extends the MLflow autologging capabilities by automatically capturing the values of input parameters and output metrics of a machine learning model as it is being trained. This information is then logged to the workspace, where it can be accessed and visualized using the MLflow APIs or the corresponding experiment in the workspace. To learn more about autologging, see [Autologging in Microsoft Fabric](https://aka.ms/fabric-autologging).
"""

mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.autolog(disable=True)  # Disable MLflow autologging

# Record the start time
start_time = time.time()

import os

desired_working_directory =  'C://Users//NicholasKenney//OneDrive - Key2 Consulting//Desktop//Classification'

# Change the working directory
os.chdir(desired_working_directory)

"""
Load dataset 
"""
iris = load_iris()
X = iris.data
y = iris.target

"""
Cleaning Data 
"""
def clean_data(X):
    # Rename column 0 to 'sepal_length'
    X = X.rename(columns={0: 'sepal_length'})
    # Rename column 1 to 'sepal_width'
    X = X.rename(columns={1: 'sepal_width'})
    # Rename column 2 to 'petal_length'
    X = X.rename(columns={2: 'petal_length'})
    # Rename column 3 to 'petal_width'
    X = X.rename(columns={3: 'petal_width'})
    return X

# Loaded variable 'X' from kernel state
X = pd.DataFrame(X.tolist() if len(X.shape) > 2 else X)

X_clean = clean_data(X.copy())
X_clean.head()

def clean_data(y):
    # Rename column 0 to 'species'
    y = y.rename(columns={0: 'species'})
    return y

# Loaded variable 'y' from kernel state
y = pd.DataFrame(y.tolist() if len(y.shape) > 2 else y)

y_clean = clean_data(y.copy())
y_clean.head()

X = X_clean
y = y_clean

"""
train_test_split
"""
# X_train: Features of the training set
# X_test: Features of the testing set
# y_train: Labels of the training set
# y_test: Labels of the testing set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

"""
Modeling
"""
clf = tree.DecisionTreeClassifier()
# Train the model using the training set
clf.fit(X_train, y_train)
clf.get_params()
"""
feature importances
"""
# Get feature importances
feature_importances = clf.feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance Percentage': feature_importances})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance Percentage', ascending=False)

# Display the feature importances
print("Feature Importances:")
print(feature_importance_df)

# Visualize feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance Percentage'])
plt.xlabel('Importance Percentage')
plt.title('Feature Importances')
plt.show()

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, rounded=True)
# Visualize the decision tree
plt.show()

"""
predictions
"""
# Make predictions on the testing set
predictions = clf.predict(X_test)

# Create a DataFrame with new data and predictions
result_df = pd.DataFrame(data=X_test, columns=X_test.columns)
result_df['True_Label'] = y_test  # Assuming y_test is the true label for the corresponding examples
result_df['Predicted_Label'] = predictions

# Display the DataFrame
print(result_df)
# Log the DataFrame
result_df.to_csv("Baseline.csv")

"""
metrics
"""
# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Display additional metrics like classification report and confusion matrix
print(classification_report(y_test, predictions))
cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

"""
Create Random data 
"""
# Create a shallow copy of the original DataFrame
copied_df = copy.copy(X_test)
def clean_data(copied_df):
    # Select column: 'sepal_length'
    copied_df['sepal_length'] = np.random.uniform(low=1.0, high=100.0, size=len(copied_df))  # random values between 1 and 100
    # Select column: 'sepal_width'
    copied_df['sepal_width'] = np.random.uniform(low=1.0, high=100.0, size=len(copied_df))  # random values between 1 and 100
    # Select column: 'petal_length'
    copied_df['petal_length'] = np.random.uniform(low=1.0, high=100.0, size=len(copied_df))  # random values between 1 and 100
    # Select column: 'petal_width'
    copied_df['petal_width'] = np.random.uniform(low=1.0, high=100.0, size=len(copied_df))  # random values between 1 and 100
    # Round columns 'sepal_length', 'sepal_width' and 2 other columns (Number of decimals: 1)
    copied_df = copied_df.round({'sepal_length': 1, 'sepal_width': 1, 'petal_length': 1, 'petal_width': 1})
    return copied_df

copied_df_clean = clean_data(copied_df.copy())
copied_df_clean.head()
new_data = copied_df_clean

"""
new predictions
"""
# Assuming 'new_data' contains the features of new instances
new_predictions = clf.predict(new_data)

# Create a DataFrame with new data and predictions
result_df = pd.DataFrame(data=new_data, columns=new_data.columns)
result_df['True_Label'] = y_test  # Assuming y_test is the true label for the corresponding examples
result_df['Predicted_Label'] = new_predictions

# Display the DataFrame
print(result_df)
# Log the DataFrame
result_df.to_csv("Prediction_Result.csv")

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, new_predictions)
print(f"Accuracy: {accuracy}")

# Display additional metrics like classification report and confusion matrix
print(classification_report(y_test, new_predictions))
cm = confusion_matrix(y_test, new_predictions)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Function to plot histograms for each feature in original and new data
def plot_data_distribution(original_data, new_data):
    num_features = original_data.shape[1]  # Number of features in the dataset
    num_rows = 2  # Number of rows in the subplot grid
    num_cols = (num_features + 1) // 2  # Number of columns in the subplot grid

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 8))
    fig.suptitle('Distribution of Features in Original and New Data', fontsize=16)

    for i, feature in enumerate(original_data.columns):
        row_index = i // num_cols
        col_index = i % num_cols

        sns.histplot(original_data[feature], kde=True, ax=axes[row_index, col_index], color='blue', label='Original Data')
        sns.histplot(new_data[feature], kde=True, ax=axes[row_index, col_index], color='orange', label='New Data')

        axes[row_index, col_index].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to prevent overlap
    plt.show()

# Plot the distribution of features
plot_data_distribution(X_test, new_data)


# Function to plot line plots for each feature in original and new data
def plot_data_distribution(original_data, new_data):
    num_features = original_data.shape[1]  # Number of features in the dataset
    fig, axes = plt.subplots(nrows=1, ncols=num_features, figsize=(16, 4))
    fig.suptitle('Distribution of Features in Original and New Data', fontsize=16)

    for i, feature in enumerate(original_data.columns):
        sns.kdeplot(original_data[feature], ax=axes[i], color='blue', label='Original Data')
        sns.kdeplot(new_data[feature], ax=axes[i], color='orange', label='New Data')

        axes[i].set_title(feature)
        axes[i].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to prevent overlap
    plt.show()

# Plot the distribution of features
plot_data_distribution(X_test, new_data)


"""
Come back:
Important to add logging
"""
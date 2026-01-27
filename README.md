# Breast Cancer Classification using Logistic Regression

Overview:
This notebook demonstrates a machine learning workflow to predict breast cancer diagnosis (Malignant or Benign) using a Logistic Regression model. The data is sourced from a Snowflake database and processed using Python's data science libraries.

Data Source:
The dataset is retrieved from a Snowflake database, specifically from the CANCER table within the breast_cancer_db.PUBLIC schema.

Preprocessing Steps:
Diagnosis Mapping: The 'DIAGNOSIS' column, originally categorical ('M' for Malignant, 'B' for Benign), is converted into numerical representation (1 for Malignant, 0 for Benign).
Feature and Target Split: The dataset is split into features (X) and the target variable (y), with 'ID' and 'DIAGNOSIS' columns excluded from features.
Train-Test Split: The data is divided into training and testing sets with a 80/20 ratio (test_size=0.2) to evaluate model performance.
Feature Scaling: Numerical features are standardized using StandardScaler to ensure that all features contribute equally to the model training process.
Model Training:
A Logistic Regression model is used for classification. The model is trained on the scaled training data (x_train_scaled) and their corresponding labels (y_train).

Model Performance:
After training, the model's performance is evaluated on the test set. The key metrics are as follows:

Accuracy
0.9737

Confusion Matrix
[[70  1]
 [ 2 41]]
Classification Report
              precision    recall  f1-score   support

           0       0.97      0.99      0.98        71
           1       0.98      0.95      0.96        43

    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114

Conclusion:
The Logistic Regression model achieved high accuracy in predicting breast cancer diagnosis, as indicated by the performance metrics.

# Breast Cancer Diagnosis: KNN and Naive Bayes Classification with Snowflake Data

This project demonstrates the application of machine learning models (K-Nearest Neighbors and Gaussian Naive Bayes) for breast cancer diagnosis using data fetched from a Snowflake data warehouse.

Project Overview:
The goal of this notebook is to classify breast cancer as either benign (B) or malignant (M) based on various features extracted from cell nuclei. The process involves:

Data Ingestion: Connecting to Snowflake to retrieve the breast cancer dataset.
Data Preprocessing: Handling missing values, and scaling numerical features.
Model Training: Implementing and training a K-Nearest Neighbors (KNN) classifier and a Gaussian Naive Bayes classifier.
Model Evaluation: Assessing the performance of both models using accuracy score, confusion matrix, and classification report.
Technologies Used:
Python: The primary programming language.
pandas: For data manipulation and analysis.
snowflake-connector-python: To establish a connection with Snowflake and fetch data.
scikit-learn: For machine learning algorithms (KNN, Gaussian Naive Bayes, train_test_split, StandardScaler).
Getting Started
To run this notebook, you will need to:

1. Install Dependencies
Ensure you have the necessary Python libraries installed. You can install them using pip:

pip install pandas snowflake-connector-python scikit-learn
2. Snowflake Connection
Update the Snowflake connection details in the notebook with your credentials:

import snowflake.connector

conn = snowflake.connector.connect(
    user = 'YOUR_SNOWFLAKE_USER',
    password = 'YOUR_SNOWFLAKE_PASSWORD',
    account = 'YOUR_SNOWFLAKE_ACCOUNT',
    database = 'YOUR_DATABASE',
    schema = 'YOUR_SCHEMA',
    warehouse = 'YOUR_WAREHOUSE'
)
Make sure the cancer table exists in your specified database and schema, containing the breast cancer data with columns matching the notebook's expectations.

3. Run the Notebook
Execute the cells sequentially to:

Connect to Snowflake and load the data.
Explore and preprocess the dataset.
Train and evaluate the KNN model.
Train and evaluate the Gaussian Naive Bayes model.
Results
The models' performance metrics (accuracy, precision, recall, f1-score) are displayed in the notebook. For reference, the provided notebook shows the following results:

# K-Nearest Neighbors (KNN)
Accuracy: ~95.6%
Classification Report:
               precision    recall  f1-score   support

           B       0.96      0.97      0.96        66
           M       0.96      0.94      0.95        48

    accuracy                           0.96       114
   macro avg       0.96      0.95      0.95       114
weighted avg       0.96      0.96      0.96       114
# Gaussian Naive Bayes
Accuracy: ~92.9%
Classification Report:
               precision    recall  f1-score   support

           B       0.94      0.94      0.94        66
           M       0.92      0.92      0.92        48

    accuracy                           0.93       114
   macro avg       0.93      0.93      0.93       114
weighted avg       0.93      0.93      0.93       114
These results indicate that both models perform well on this dataset, with the KNN model showing slightly higher accuracy in this particular split.

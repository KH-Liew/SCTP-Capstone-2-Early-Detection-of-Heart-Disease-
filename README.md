# Capstone on Predictive Analytics using Machine Learning
## Problem Statement
Heart disease is a leading cause of mortality worldwide. 
Early detection and accurate prediction of heart disease can significantly improve patient outcomes by enabling timely intervention and preventive measures. 
The objective of this project is to develop a machine learning model that can effectively predict the presence or absence of heart disease based on various medical 
and lifestyle factors.

## Research Objectives
The goal of this project is to build a predictive model that can accurately classify individuals as either having or not having heart disease. 
By analysing a comprehensive set of patient attributes, including demographic, clinical and lifestyle factors, the model aims to identify patterns 
and features indicative of the presence or absence of heart disease.

## Data Information
The dataset provided for this project is a subset of clinical data worked on for a project with a healthcare organisation to understand the influence 
of various factors on the heart disease.

## _About the Data_
* Age | Objective Feature | age | int (days) 
* Height | Objective Feature | height | int (cm) | 
* Weight | Objective Feature | weight | float (kg) | 
* Gender | Objective Feature | gender | 1: Female, 2: Male | 
* Systolic blood pressure | Examination Feature | ap_hi | int | 
* Diastolic blood pressure | Examination Feature | ap_lo | int | 
* Cholesterol | Examination Feature | cholesterol | 1: normal, 2: above normal, 3: well above normal | 
* Glucose | Examination Feature | gluc | 1: normal, 2: above normal, 3: well above normal | 
* Smoking | Subjective Feature | smoke | binary | 
* Alcohol intake | Subjective Feature | alco | binary | 
* Physical activity | Subjective Feature | active | binary | 
* Presence or absence of cardiovascular disease | Target Variable | 
* cardio | binary |

## Load Initial Libraries
pandas, numpy, matplotlib, seaborn
> _import the ML classification algorithms when required_

## Import the Data
cardio_data.csv

## Data Exploration & Cleaning
* Remove duplicate data
* Check for empty data cells, remove rows (if any)
* Change column 'age' data from days to years
* Change column 'weight' to int
* Check data set after cleaning

## Observations from basic exploration
There are total 70000 records/rows and 16 columns in the dataset. All independent variables have int values. The disease column which is the target, 
has Total Positive records: 34979, Total Negative records: 35021 There is no missing value in any row/column. Hence, no need to treat the missing values.

## Visual Analysis (EDA - Exploratory Data Analysis)
## _data transformation, identify and address outliers / any anormalies_
* added 'bmi' column as int
* drop id, date, country, occupation columns not vital for modeling
* remove values in height, weight, Systolic, diastolic blood pressure that fall below 2.5% or above 97.5% of a given range

## Correlation Analysis
## _Correlation Matrix_
![image](https://github.com/KH-Liew/SCTP-Capstone-2-Early-Detection-of-Heart-Disease-/assets/155032208/a79243cd-e6ca-48f8-80af-b016b85ad264)

## Training Models
* import the function/module to randomly split the data from sklearn.model_selection import train_test_split
* select the predictors and target
* verify the features and target variables
* split the data into train and test and verify Train Test split with Target Value

## Data Modeling
## _Import the algorithms_
* from sklearn.metrics import confusion_matrix, classification_report
* from sklearn.linear_model import LogisticRegression
* from sklearn.neighbors import KNeighborsClassifier
* from sklearn.svm import SVC
* from sklearn.tree import DecisionTreeClassifier
* from sklearn.preprocessing import StandardScaler

## Model Performance Evaluation
* The accuracy of logistic regression is: 72.18675179569034 %
* The accuracy of KNN model is: 66.27427507315775 %
* The accuracy of SVM model is: 71.38201649374834 %
* The accuracy of Decision Tree model is: 63.0220803405161 %

Based on accuracy of the model, it looks like Logistic Regression and SVM model can be used.
Accuracy score is a reliable performance metric if the data is balanced. In our case too, the data is balanced so we can conclude that for 
the type of analysis we have done on the type of data, logistic regression, KNN classifier or SVM classifier - all three are suitable model.

## Confusion Matrix and Classification Report
![image](https://github.com/KH-Liew/SCTP-Capstone-2-Early-Detection-of-Heart-Disease-/assets/155032208/d91673a9-8abc-41b1-a07a-7a4ccbc31143)
![image](https://github.com/KH-Liew/SCTP-Capstone-2-Early-Detection-of-Heart-Disease-/assets/155032208/f593d845-30b6-45b5-9359-58ddfea4522a)
![image](https://github.com/KH-Liew/SCTP-Capstone-2-Early-Detection-of-Heart-Disease-/assets/155032208/81c66f9a-c99b-4bb0-a2ae-73c73bde60b4)
![image](https://github.com/KH-Liew/SCTP-Capstone-2-Early-Detection-of-Heart-Disease-/assets/155032208/4649a51d-b514-401b-9bca-9a3cad14e576)

## Display Classification Report
<img width="322" alt="image" src="https://github.com/KH-Liew/SCTP-Capstone-2-Early-Detection-of-Heart-Disease-/assets/155032208/18f7cf24-b732-4d89-8db9-578be5ad6861">
<img width="332" alt="image" src="https://github.com/KH-Liew/SCTP-Capstone-2-Early-Detection-of-Heart-Disease-/assets/155032208/5ea38f67-08b9-4a00-bf63-b88b9583f023">

## Insights and Comments
The model performance shows accuracy scores ranging from approximately 62% to 72% for all four models, with the highest score achieved by 
Logistic Regression. However, since our goal is to capture all positive cases in this scenario, we should prioritize the Recall score when 
selecting a model for prediction. According to the Recall score, Logistic Regression performs the best with an accuracy of 65%.



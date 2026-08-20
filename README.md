# Healthcare Medical Condition Classification using Logistic Regression

## Project Overview

This project applies **Machine Learning classification techniques** to a healthcare dataset to predict a patient's **Medical Condition** based on available patient-related information.

The primary model used in this project is **Multiclass Logistic Regression**. The project demonstrates a complete machine learning workflow, including data exploration, feature selection, preprocessing, model training, prediction, and evaluation.

The project is implemented using **Python and Jupyter Notebook**.

---

## Objective

The objective of this project is to develop a classification model capable of predicting a patient's medical condition from features such as:

* Age
* Gender
* Blood Type
* Admission Type
* Medication
* Test Results

The target variable is:

```text
Medical Condition
```

The problem is treated as a **multiclass classification problem** because the target contains multiple medical-condition categories.

---

## Dataset

The dataset contains patient and hospital-related information, including:

| Feature            | Description             |
| ------------------ | ----------------------- |
| Name               | Patient name            |
| Age                | Patient age             |
| Gender             | Patient gender          |
| Blood Type         | Patient blood type      |
| Medical Condition  | Target variable         |
| Date of Admission  | Hospital admission date |
| Doctor             | Assigned doctor         |
| Hospital           | Hospital name           |
| Insurance Provider | Insurance company       |
| Billing Amount     | Hospital billing amount |
| Room Number        | Patient room number     |
| Admission Type     | Type of admission       |
| Discharge Date     | Hospital discharge date |
| Medication         | Medication provided     |
| Test Results       | Medical test result     |

---

## Feature Selection

Not every dataset column was used for model training.

The following variables were excluded:

```text
Name
Doctor
Hospital
Insurance Provider
Billing Amount
Room Number
Date of Admission
Discharge Date
```

### Reasons for Exclusion

**Name**
Patient names are identifiers and do not provide meaningful predictive information.

**Doctor and Hospital**
These variables can have high cardinality and may cause the model to learn administrative patterns rather than meaningful patient characteristics.

**Insurance Provider**
Insurance information is primarily administrative and is not directly relevant to identifying the patient's medical condition.

**Billing Amount**
Billing amount represents a financial outcome and is not an appropriate medical predictor for this classification task. Including it could also introduce unwanted information leakage depending on the prediction scenario.

**Room Number**
Room numbers are identifiers and do not have meaningful medical significance.

**Date of Admission and Discharge Date**
Raw dates were excluded from the initial model because they require feature engineering before they can be meaningfully incorporated into a machine learning model.

---

## Selected Features

The final feature set consists of:

```text
Age
Gender
Blood Type
Admission Type
Medication
Test Results
```

Target:

```text
Medical Condition
```

---

## Machine Learning Workflow

The project follows the following workflow:

```text
Dataset
   |
   v
Data Loading
   |
   v
Data Inspection
   |
   v
Exploratory Data Analysis
   |
   v
Feature Selection
   |
   v
Train-Test Split
   |
   v
Categorical Encoding
   |
   v
Feature Scaling
   |
   v
Logistic Regression
   |
   v
Predictions
   |
   v
Model Evaluation
```

---

## Exploratory Data Analysis

Before model development, exploratory data analysis was performed to understand the dataset and identify potential patterns.

The analysis includes:

* Dataset dimensions
* Data types
* Missing values
* Duplicate records
* Numerical statistics
* Categorical distributions
* Age distribution
* Gender distribution
* Blood type distribution
* Medical condition distribution
* Medication distribution
* Medical condition vs medication
* Medical condition vs billing amount
* Admission type vs billing amount
* Age vs billing amount

Visualizations were created using **Matplotlib** and **Seaborn**.

---

## Data Preprocessing

The dataset contains both numerical and categorical features.

### Numerical Features

```text
Age
```

The numerical feature is standardized using:

```python
StandardScaler()
```

### Categorical Features

```text
Gender
Blood Type
Admission Type
Medication
Test Results
```

Categorical variables are transformed using:

```python
OneHotEncoder()
```

This converts categorical values into numerical representations that can be processed by the Logistic Regression algorithm.

---

## Train-Test Split

The dataset is divided into training and testing subsets.

```text
80% → Training
20% → Testing
```

The training set is used to train the model, while the testing set is reserved for evaluating performance on unseen data.

A fixed `random_state` is used to make the experiment reproducible.

---

## Machine Learning Model

### Logistic Regression

Logistic Regression is used as the primary classification algorithm.

Although its name contains "Regression," Logistic Regression is primarily used for **classification problems**.

Since `Medical Condition` contains multiple classes, the model performs **multiclass classification**.

The implementation uses Scikit-learn's:

```python
LogisticRegression()
```

The preprocessing and model are combined into a Scikit-learn `Pipeline`.

This ensures that preprocessing is performed consistently during both training and prediction.

---

## Model Evaluation

The model is evaluated using several metrics rather than relying only on accuracy.

### Accuracy

Measures the overall percentage of correctly classified observations.

### Precision

Measures how many observations predicted as a particular class actually belong to that class.

### Recall

Measures how many observations belonging to a particular class were correctly identified.

### F1-Score

The harmonic mean of precision and recall.

### Confusion Matrix

Provides a detailed view of correct and incorrect predictions for each medical-condition class.

---

## Current Model Performance

The initial Logistic Regression model achieved approximately:

```text
Accuracy: 16.80%
```

This relatively low accuracy is important to analyze rather than simply treating it as a model failure.

The dataset contains multiple medical-condition classes, so a result around 16.7% can be close to random classification when there are six approximately balanced classes.

The low performance also suggests that the currently selected features may not contain enough predictive information to reliably distinguish between the different medical conditions.

Further analysis is therefore required before drawing conclusions about model effectiveness.

---

## Limitations

Several limitations should be considered:

1. The dataset may contain synthetic or artificially generated healthcare records.
2. The selected features may have weak relationships with the target variable.
3. Logistic Regression assumes relatively simple decision boundaries and may not capture complex relationships between features.
4. The model should not be interpreted as a clinical diagnostic system.
5. A larger and clinically validated dataset would be required for real-world medical applications.
6. Model performance should be evaluated using cross-validation rather than relying solely on a single train-test split.

---

## Future Improvements

Future versions of the project can include:

* Cross-validation
* Hyperparameter tuning
* Class imbalance analysis
* Feature engineering
* Feature importance analysis
* Comparison with K-Nearest Neighbors
* Decision Tree
* Random Forest
* Gradient Boosting
* Support Vector Machine
* Model comparison using F1-score
* ROC-AUC analysis where appropriate
* Improved handling of temporal features
* More extensive feature engineering

The next stage of the project will focus on **cross-validation and comparing Logistic Regression with other classification algorithms**.

---

## Technologies Used

* Python
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## Project Structure

```text
Healthcare-Medical-Condition-Classification/
│
├── healthcare_dataset.csv
├── Healthcare_Medical_Condition_Classification.ipynb
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository and install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
Healthcare_Medical_Condition_Classification.ipynb
```

---

## Conclusion

This project demonstrates an end-to-end machine learning workflow for **multiclass healthcare classification** using Logistic Regression.

The project emphasizes not only model development but also **data understanding, responsible feature selection, preprocessing, and appropriate evaluation**.

The initial results demonstrate that model selection alone is not sufficient; understanding the relationship between the available features and the target variable is equally important. Further experimentation with cross-validation, feature engineering, and alternative classification algorithms will be required to determine whether the prediction task can be improved.

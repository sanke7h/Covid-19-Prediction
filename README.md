# High Accuracy COVID-19 Prediction Using Optimized Union Ensemble Feature Selection Approach

This repository contains implementation of the above research paper, which uses a union ensemble feature selection method followed by Hyperparameter optimization using a genetic algorithm to predict COVID-19.

## Data Preprocessing Steps

### 1. Dataset Loading and Initial Inspection

- Loaded the original `master_dataset.csv` file
- Fixed column alignment issues in the original dataset
- The original dataset has 59 columns.
- Select 27 relevant columns for analysis

### 2. Feature Selection

Selected 27 key features for analysis, including:
- Demographic factors: sex, age, BMI
- Health behaviors: smoking, alcohol, cannabis, amphetamines, cocaine
- Social factors: contacts count, working environment
- Risk reduction behaviors: mask usage, social distancing, single, covid19_contact
- Pre-existing medical conditions: asthma, heart disease, diabetes, lung diseases, compromised immune system, hiv_positive, hypertension, kidney diseases, other chronic disesases
- Target variable: COVID-19 positive status

### 3. Data Cleaning and Transformation

#### Handling Age Data
- Converted age range strings (e.g., '20_30') to numerical values using the average (25)
- Filled missing age values with the mean

#### Handling Missing Values
- Filled categorical missing values with mode (most frequent value)
- Filled numerical missing values with mean

#### Categorical Variable Encoding
- Applied one-hot encoding to nominal variables:
  - Sex (male, female, other, undefined)
  - Smoking status (never, quit, vape, light/medium/heavy)
  - Working environment (home, never, stopped, travel critical, travel non-critical)

#### Numerical Variable Processing
- Converted drug use columns to numeric format
- Applied Min-Max scaling to normalize all features

### 4. Class Imbalance Handling

- Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset
- Created a 1:3 ratio of positive to negative COVID-19 cases
- Resulted in a balanced dataset with 1,348,342 samples

## Final Dataset

The final preprocessed dataset contains 41 features (after one-hot encoding) and is ready for machine learning model development.

## Features in Final Dataset

- Normalized numerical values (age, BMI, alcohol consumption, etc.)
- Binary health indicators (asthma, diabetes, etc.)
- One-hot encoded categorical variables
- Target variable: COVID-19 positive status (0 or 1)

## Data Preparation

The dataset was preprocessed to handle missing values with SMOTE applied for class imbalance and encoding for categorical variables. The processed dataset was split into:
- 70% training data
- 15% validation data
- 15% test data

Features were scaled using StandardScaler for applicable models.

## Feature Selection Methods

Three feature selection techniques were implemented, each selecting the top 15 features:

1. **MIFS (Mutual Information Feature Selection)**
   - Features: bmi, contacts_count, age, alcohol, rate_reducing_risk_single, rate_reducing_mask, covid19_symptoms, cannabis, covid19_contact, sex_male, sex_female, working_stopped, working_never, smoking_never, working_travel critical

2. **RFE (Recursive Feature Elimination)**
   - Features: age, alcohol, cannabis, contacts_count, rate_reducing_risk_single, covid19_symptoms, covid19_contact, asthma, other_chronic, nursing_home, sex_female, sex_male, smoking_quit10, smoking_yesmedium, working_stopped

3. **RidgeCV-based Feature Selection**
   - Features: covid19_symptoms, age, rate_reducing_risk_single, covid19_contact, alcohol, sex_male, sex_female, nursing_home, contacts_count, working_never, cannabis, heart_disease, working_stopped, asthma, smoking_yesmedium

Additionally, a **Union Ensemble** approach was utilized, combining features from all three methods, resulting in 21 unique features.

## Models Evaluated

Four classification models were trained and evaluated:
- Linear SVM (LSVM)
- Logistic Regression
- Gradient Boosting
- AdaBoost

## Results

### Table: Training on all features (no feature selection)

| Model | Validation Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1 | Test AUC |
|-------|---------------------|---------------|----------------|-------------|---------|----------|
| LSVM | 0.8698 | 0.8690 | 0.8465 | 0.5815 | 0.6894 | N/A |
| LogisticRegression | 0.8677 | 0.8669 | 0.8201 | 0.5991 | 0.6924 | 0.9034 |
| GradientBoosting | 0.9599 | 0.9594 | 0.9591 | 0.8749 | 0.9151 | 0.9889 |
| AdaBoost | 0.9296 | 0.9288 | 0.9156 | 0.7880 | 0.8470 | 0.9697 |

### Feature Selection Method Comparison

### Table 1: Training on MIFS FS Subset

| Model | Validation Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1 | Test AUC |
|-------|---------------------|---------------|----------------|-------------|---------|----------|
| LSVM | 0.8690 | 0.8678 | 0.8449 | 0.5773 | 0.6859 | N/A |
| LogisticRegression | 0.8664 | 0.8655 | 0.8180 | 0.5944 | 0.6885 | 0.9005 |
| GradientBoosting | 0.9608 | 0.9603 | 0.9585 | 0.8791 | 0.9171 | 0.9884 |
| AdaBoost | 0.9258 | 0.9254 | 0.9082 | 0.7805 | 0.8395 | 0.9700 |

### Table 2: Training on RFE FS Subset

| Model | Validation Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1 | Test AUC |
|-------|---------------------|---------------|----------------|-------------|---------|----------|
| LSVM | 0.8699 | 0.8687 | 0.8475 | 0.5790 | 0.6880 | N/A |
| LogisticRegression | 0.8675 | 0.8663 | 0.8208 | 0.5950 | 0.6899 | 0.9017 |
| GradientBoosting | 0.9595 | 0.9585 | 0.9525 | 0.8778 | 0.9136 | 0.9868 |
| AdaBoost | 0.9305 | 0.9300 | 0.9147 | 0.7938 | 0.8500 | 0.9693 |

### Table 3: Training on RidgeCV FS Subset

| Model | Validation Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1 | Test AUC |
|-------|---------------------|---------------|----------------|-------------|---------|----------|
| LSVM | 0.8697 | 0.8685 | 0.8467 | 0.5788 | 0.6876 | N/A |
| LogisticRegression | 0.8674 | 0.8664 | 0.8208 | 0.5957 | 0.6904 | 0.9011 |
| GradientBoosting | 0.9605 | 0.9596 | 0.9541 | 0.8807 | 0.9160 | 0.9872 |
| AdaBoost | 0.9302 | 0.9294 | 0.9129 | 0.7933 | 0.8490 | 0.9693 |

### Table 1: Training on union of MIFS and RFE features

| Model | Validation Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1 | Test AUC |
|-------|---------------------|---------------|----------------|-------------|---------|----------|
| LSVM | 0.8703 | 0.8691 | 0.8485 | 0.5798 | 0.6888 | N/A |
| LogisticRegression | 0.8672 | 0.8662 | 0.8200 | 0.5954 | 0.6899 | 0.9024 |
| GradientBoosting | 0.9581 | 0.9574 | 0.9579 | 0.8679 | 0.9107 | 0.9877 |
| AdaBoost | 0.9296 | 0.9288 | 0.9156 | 0.7880 | 0.8470 | 0.9697 |

### Table 2: Training on union of MIFS and RidgeCV features

| Model | Validation Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1 | Test AUC |
|-------|---------------------|---------------|----------------|-------------|---------|----------|
| LSVM | 0.8699 | 0.8686 | 0.8481 | 0.5780 | 0.6875 | N/A |
| LogisticRegression | 0.8675 | 0.8662 | 0.8201 | 0.5955 | 0.6900 | 0.9016 |
| GradientBoosting | 0.9600 | 0.9594 | 0.9592 | 0.8746 | 0.9150 | 0.9881 |
| AdaBoost | 0.9296 | 0.9288 | 0.9156 | 0.7880 | 0.8470 | 0.9697 |

### Table 3: Training on union of RFE and RidgeCV features

| Model | Validation Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1 | Test AUC |
|-------|---------------------|---------------|----------------|-------------|---------|----------|
| LSVM | 0.8703 | 0.8689 | 0.8472 | 0.5804 | 0.6889 | N/A |
| LogisticRegression | 0.8678 | 0.8668 | 0.8218 | 0.5964 | 0.6912 | 0.9019 |
| GradientBoosting | 0.9595 | 0.9585 | 0.9525 | 0.8778 | 0.9136 | 0.9868 |
| AdaBoost | 0.9305 | 0.9300 | 0.9147 | 0.7938 | 0.8500 | 0.9693 |





### Union Ensemble Feature Selection

| Model | Test Accuracy | Test Precision | Test Recall | Test F1 | Test AUC |
|-------|--------------|---------------|------------|---------|----------|
| LSVM | 0.8691 | 0.8475 | 0.5810 | 0.6894 | N/A |
| LogisticRegression | 0.8665 | 0.8206 | 0.5963 | 0.6907 | 0.9025 |
| GradientBoosting | 0.9574 | 0.9579 | 0.8679 | 0.9107 | 0.9877 |
| AdaBoost | 0.9288 | 0.9156 | 0.7880 | 0.8470 | 0.9697 |

### PCA Feature Extraction (15 components)

| Model | Test Accuracy | Test Precision | Test Recall | Test F1 | Test AUC |
|-------|--------------|---------------|------------|---------|----------|
| LSVM | 0.8476 | 0.8132 | 0.5067 | 0.6244 | N/A |
| LogisticRegression | 0.8499 | 0.7929 | 0.5410 | 0.6432 | 0.8851 |
| GradientBoosting | 0.8652 | 0.8409 | 0.5683 | 0.6783 | 0.8989 |
| AdaBoost | 0.8558 | 0.7970 | 0.5678 | 0.6632 | 0.8870 |

## Key Findings

1. **Gradient Boosting** consistently performed best across all feature subsets, achieving over 95% accuracy and F1 scores above 0.91.
2. **Feature selection** maintained or slightly improved model performance while reducing dimensionality.
3. The **Union Ensemble** approach provided comparable performance to using all features.
4. **PCA** performed worse than the other feature selection methods, suggesting that the original features contain important information that may be lost in the transformation.
5. Important features for COVID-19 prediction include: covid19_symptoms, age, contacts_count, alcohol consumption, and covid19_contact.


## Genetic Algorithm Hyperparameter Optimization

The project uses a genetic algorithm framework to search for optimal hyperparameters across different machine learning models. After optimization, the best-performing models are evaluated on test data and analyzed using SHAP (SHapley Additive exPlanations) to understand feature importance.

## Components

### Genetic Algorithm Framework

The `ga_hpo_model` function provides a generic GA implementation for hyperparameter optimization with:
- Random individual generation
- Fitness evaluation
- Selection based on fitness scores
- Crossover between selected individuals
- Mutation of offspring
- Generational evolution tracking

### Optimized Models

Four classification algorithms are optimized:

1. **AdaBoost**
   - Hyperparameters: `n_estimators` (50-200), `learning_rate` (0.01-1.0)
   - Best parameters: `n_estimators=184`, `learning_rate=0.978`
   - Validation accuracy: 0.9608
   - Test accuracy: 0.9603

2. **Gradient Boosting**
   - Hyperparameters: `n_estimators` (50-200), `learning_rate` (0.01-1.0)
   - Best parameters: `n_estimators=188`, `learning_rate=0.883`
   - Validation accuracy: 0.9907
   - Test accuracy: 0.9902

3. **Linear SVM**
   - Hyperparameters: `C` (0.001-100)
   - Best parameters: `C=9.749`
   - Validation accuracy: 0.8704
   - Test accuracy: 0.8691

4. **Logistic Regression**
   - Hyperparameters: `C` (0.001-100)
   - Best parameters: `C=0.626`
   - Validation accuracy: 0.8676
   - Test accuracy: 0.8665

### SHAP Analysis

The code includes SHAP analysis for all optimized models to explain their predictions:
- KernelExplainer is used for model-agnostic explanations
- Various visualization types:
  - Beeswarm summary plots for feature impact distribution
  - Bar plots for global feature importance

## Results

Gradient Boosting achieved the highest accuracy (99.02% on test data), significantly outperforming other models:
Here's a simple table for the final test accuracy results:

| Model | Final Test Accuracy |
|-------|---------------------|
| Gradient Boosting | 0.9902 |
| AdaBoost | 0.9603 |
| Linear SVM | 0.8691 |
| Logistic Regression | 0.8665 |

## Visualization

SHAP visualizations reveal which features contribute most to each model's predictions, providing interpretability alongside performance optimization.

### Adaboost

![image](https://github.com/user-attachments/assets/0bf1ce8c-e979-4edc-9fca-e08cee6b6b6a)

![image](https://github.com/user-attachments/assets/133e9bab-c1d9-4694-a0e6-03d003083e40)

### Gradient Boosting
![image](https://github.com/user-attachments/assets/f07c4734-7e9b-4998-80fa-2eabcb15beea)
![image](https://github.com/user-attachments/assets/ccb66ecb-4a0f-477c-84f2-403f1a57aa3e)

### LSVM

![image](https://github.com/user-attachments/assets/fd7aed65-f980-453d-a898-abdcce9969d1)

![image](https://github.com/user-attachments/assets/0f6bc780-4b5c-4c93-a0e9-cd66a55c1b7a)

### Logistic Regression

![image](https://github.com/user-attachments/assets/9e8d8360-d40b-45bf-9ae4-491c2d945a5b)

![image](https://github.com/user-attachments/assets/9ae67147-185b-45fb-9fb1-f492f25224e8)

## Novelty 


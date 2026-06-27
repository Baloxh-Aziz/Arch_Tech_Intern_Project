# 🚢 Titanic Survival Classification

A machine learning project that predicts whether a passenger survived the Titanic disaster using Logistic Regression, based on features like age, gender, passenger class, fare, and family size.

## 📋 Overview

This project walks through a complete end-to-end ML workflow on the classic Titanic dataset:

1. **Data Loading & Inspection** — exploring shape, columns, data types, and summary statistics
2. **Data Cleaning** — handling missing values and dropping unreliable columns
3. **Exploratory Data Analysis (EDA)** — univariate and bivariate analysis to uncover survival patterns
4. **Feature Engineering** — encoding categorical variables and creating a `Family_Size` feature
5. **Model Training** — building a Logistic Regression classifier
6. **Model Evaluation** — accuracy score, confusion matrix, and classification report

## 📊 Dataset

The dataset (`train.csv`) is the classic [Titanic dataset](https://www.kaggle.com/c/titanic) containing 891 passenger records with 12 original columns:

| Column | Description |
|---|---|
| PassengerId | Unique identifier |
| Survived | Target variable (0 = Did Not Survive, 1 = Survived) |
| Pclass | Passenger class (1st, 2nd, 3rd) |
| Name | Passenger name |
| Sex | Gender |
| Age | Age in years |
| SibSp | Number of siblings/spouses aboard |
| Parch | Number of parents/children aboard |
| Ticket | Ticket number |
| Fare | Ticket fare |
| Cabin | Cabin number |
| Embarked | Port of embarkation (C, Q, S) |

## 🧹 Data Cleaning

- **Age** — 177 missing values filled with the column mean
- **Embarked** — 2 missing values filled with the mode
- **Cabin** — dropped entirely (687 missing values, ~77% of the column)
- **Family_Size** — new feature created from `SibSp + Parch + 1`, after which `SibSp` and `Parch` were dropped

## 🔍 Exploratory Data Analysis

The notebook explores survival patterns across multiple dimensions:

- Survival distribution (overall class balance)
- Survival by gender, passenger class, and age
- Fare distribution and its relationship with class and survival
- Family size vs survival
- Correlation analysis between numerical features and survival
- Outlier detection in Age and Fare using boxplots

**Key Insights:**
- Female passengers had the highest survival rate
- First-class passengers survived more than lower classes
- Higher fare passengers had better survival chances
- Children had slightly better survival outcomes
- Passengers with smaller families had better survival rates

## 🤖 Model

| Detail | Value |
|---|---|
| Algorithm | Logistic Regression |
| Features used | Pclass, Sex, Age, Fare, Family_Size, Embarked |
| Train/Test Split | 80% / 20% (stratified) |
| Encoding | Sex (male=0, female=1), Embarked (S=0, C=1, Q=2) |

## 📈 Results

| Metric | Score |
|---|---|
| Training Accuracy | ~80.9% |
| Test Accuracy | ~78.2% |

**Confusion Matrix (Test Set):**

| | Predicted: Did Not Survive | Predicted: Survived |
|---|---|---|
| **Actual: Did Not Survive** | 92 | 18 |
| **Actual: Survived** | 20 | 49 |

The model performs slightly better at identifying non-survivors than survivors, consistent with the class imbalance in the dataset (~62% did not survive vs ~38% survived). A full classification report (precision, recall, F1-score) is included in the notebook for a more detailed breakdown.

## 🛠️ Tech Stack

- **Python 3**
- **pandas** — data manipulation
- **numpy** — numerical operations
- **matplotlib** & **seaborn** — data visualization
- **scikit-learn** — model training and evaluation (`LogisticRegression`, `train_test_split`, `accuracy_score`, `confusion_matrix`, `classification_report`)

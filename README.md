# 🧠 Acquisition Price Prediction – Machine Learning Project

This project predicts the **acquisition price** of tech companies using regression and classification models. The dataset was built by merging multiple real-world datasets on company acquisitions, financials, and founders. The analysis involves data preprocessing, exploratory data analysis (EDA), feature engineering, and machine learning model development.

---

## 📁 Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Techniques Used](#techniques-used)
- [Models](#models)
- [Results](#results)
- [Key Insights](#key-insights)
- [Next Steps](#next-steps)
- [Team](#team)

---

## 📌 Overview

We developed regression models to predict acquisition prices and classification models to categorize deals into size classes (Small, Medium, Large). Our goal was to understand which features influence acquisition valuations and how accurately we can predict them using machine learning.

---

## 🎯 Objectives

- Predict **exact acquisition prices** using regression
- Classify acquisitions into **deal size categories**
- Perform feature selection and multicollinearity checks
- Compare performance of base models and ensemble methods

---

## 📊 Dataset

We merged 4 sources:
- Acquisitions (336 records)
- Acquiring Companies (36 records)
- Acquired Companies (310 records)
- Founders & Board Members (382 records)

After preprocessing, the final dataset contained **336 acquisitions** and **49 features**.

### 🔧 Feature Engineering:
- `Company_Age = Acquisition Year - Founding Year`
- `Funding_per_Employee = Total Funding / (Employees + 1)`
- `Has_IPO = Binary flag for public companies`
- Market categories, location, and acquisition terms were also encoded

---

## 🛠️ Techniques Used

- Data Cleaning and Merging
- Feature Engineering & Transformation (Log, Ratios)
- Correlation & VIF Analysis
- Model Training & Hyperparameter Tuning
- Ensemble Learning
- Classification & Regression Evaluation

---

## 🤖 Models

### 📉 Regression Models:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- Ensemble Averaging (Best)

### 📊 Classification Models:
- Random Forest Classifier
- SVM (Linear Kernel)
- K-Nearest Neighbors
- Voting & Stacking Ensembles

---

## 📈 Results

| Model                | RMSE     | MAE     | R²     | MAPE   |
|---------------------|----------|---------|--------|--------|
| Linear Regression    | $1.74B   | $1.31B  | -0.20  | 6.68%  |
| Gradient Boosting    | $1.63B   | $1.28B  | -0.05  | 6.55%  |
| **Ensemble (Best)**  | **$1.60B** | **$1.26B** | **-0.01** | **6.42%** |

### 🥇 Best Classifier:  
- **SVM** with 83.6% test accuracy

---

## 💡 Key Insights

- **Total Funding** and **Company Age** were the most predictive features
- **Cloud**, **Software**, and **Mobile** industries had the highest average prices
- Public companies had higher acquisition prices
- Location (California, San Francisco) was a strong predictor
- Simple models like **Linear SVM** outperformed complex models in some cases

---

## 🔮 Next Steps

- Add revenue, EBITDA, or deal structure data for better accuracy
- Explore deep learning or BERT embeddings for textual features
- Perform time-series analysis on acquisition trends
- Build a live dashboard or web app for prediction

---

## 👨‍💻 Team

- **Marwan Ashraf**  
- John Boulos  
- Mariam Ragab  
- Mohamed Abdelmomen  
- Yomna Omar  
- Lina Yasser

---

## 📄 Report

The full project report with visualizations and model comparisons can be found here:  
📎 `Machine Learning Project Report.pdf` (attach it or link to it if hosted)

---

## 📬 Contact

If you're interested in collaboration or have any questions, feel free to reach out via +201144579459


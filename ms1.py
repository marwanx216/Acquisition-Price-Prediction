import json
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.ticker as ticker

# ==============================================
# DATA LOADING AND PREPROCESSING
# ==============================================

""" Data Loading and Initial Inspection """
# Load all datasets
acquisitions = pd.read_csv('Acquisitions.csv')
acquiring = pd.read_csv('Acquiring Tech Companies.csv')
acquired = pd.read_csv('Acquired Tech Companies.csv')
founders = pd.read_csv('Founders and Board Members.csv')

# Display basic info
print("Acquisitions shape:", acquisitions.shape)
print("Acquiring shape:", acquiring.shape)
print("Acquired shape:", acquired.shape)
print("Founders shape:", founders.shape)

""" Data Merging """
# Merge acquisitions with acquiring companies
merged = pd.merge(acquisitions, acquiring,
                  left_on='Acquiring Company',
                  right_on='Acquiring Company',
                  how='left',
                  suffixes=('_acquisition', '_acquiring'))

# Merge with acquired companies
merged = pd.merge(merged, acquired,
                  left_on='Acquired Company',
                  right_on='Company',
                  how='left',
                  suffixes=('', '_acquired'))
merged.to_csv('t1.csv',index=False)
print("\nMerged dataset shape:", merged.shape)
print("Columns in merged data:", merged.columns.tolist())

""" Handling Missing Values """
# Check missing values percentage
missing = merged.isnull().sum() / len(merged) * 100
print("\nMissing values percentage:\n", missing[missing > 0].sort_values(ascending=False))

# Handle specific columns:
# Price - our target variable - drop rows with missing price
merged = merged.dropna(subset=['Price'])

# Clean numeric columns before filling NA
numeric_cols = ['Year Founded', 'Number of Employees', 'Total Funding ($)']
for col in numeric_cols:
    if col in merged.columns:
        # Remove commas and convert to numeric
        merged[col] = merged[col].astype(str).str.replace(',', '').replace('nan', np.nan)
        merged[col] = pd.to_numeric(merged[col], errors='coerce')
        # Now fill NA with median
        merged[col] = merged[col].fillna(merged[col].median())
    else:
        print(f"Warning: Column {col} not found in merged data")

# Categorical columns - fill with mode or 'Unknown'
cat_cols = ['Status', 'Terms', 'Market Categories']
for col in cat_cols:
    if col in merged.columns:
        merged[col] = merged[col].fillna('Unknown')
    else:
        print(f"Warning: Column {col} not found in merged data")

""" Data Cleaning """
# Clean Price column if it exists
if 'Price' in merged.columns:
    # Remove $ and commas, convert to float
    merged['Price'] = merged['Price'].str.replace('[\$,]', '', regex=True).astype(float)

    # Log transformation for skewed prices
    merged['Log_Price'] = np.log1p(merged['Price'])
else:
    print("Error: Price column not found")

# Convert to datetime
if 'Deal announced on' in merged.columns:
    merged['Deal announced on'] = pd.to_datetime(merged['Deal announced on'], errors='coerce')
    # Extract year if needed
    merged['Deal_Year'] = merged['Deal announced on'].dt.year

# Clean text columns
text_cols = ['Description', 'Tagline', 'Market Categories']
for col in text_cols:
    if col in merged.columns:
        merged[col] = merged[col].str.lower().str.strip()

# Extract market categories as dummy variables
if 'Market Categories' in merged.columns:
    categories = merged['Market Categories'].str.get_dummies(sep=', ')
    merged = pd.concat([merged, categories], axis=1)

""" Feature Engineering """
# Company age at acquisition
if 'Year of acquisition announcement' in merged.columns and 'Year Founded' in merged.columns:
    merged['Company_Age'] = merged['Year of acquisition announcement'] - merged['Year Founded']

# Funding per employee - ensure both columns are numeric first
if 'Total Funding ($)' in merged.columns and 'Number of Employees' in merged.columns:
    merged['Funding_per_Employee'] = merged['Total Funding ($)'] / (merged['Number of Employees'] + 1)

# Acquisition count for acquiring company
if 'Acquiring Company' in merged.columns:
    acq_count = merged['Acquiring Company'].value_counts().to_dict()
    merged['Acquirer_Experience'] = merged['Acquiring Company'].map(acq_count)

# Has IPO flag
if 'IPO' in merged.columns:
    merged['Has_IPO'] = merged['IPO'].notna().astype(int)

""" Handling Categorical Variables """
# Encode important categorical features
cat_to_encode = ['Status', 'Terms', 'Country (HQ)']
for col in cat_to_encode:
    if col in merged.columns:
        le = LabelEncoder()
        merged[col + '_encoded'] = le.fit_transform(merged[col].astype(str))
    else:
        print(f"Warning: Categorical column {col} not found for encoding")

""" Final Data Preparation """
# Select relevant features for modeling
features = []
potential_features = [
    'Company_Age', 'Number of Employees', 'Total Funding ($)',
    'Funding_per_Employee', 'Acquirer_Experience', 'Has_IPO',
    'Status_encoded', 'Terms_encoded', 'Country (HQ)_encoded'
]

# Add only features that exist in the dataframe
for feat in potential_features:
    if feat in merged.columns:
        features.append(feat)

# Add category dummy variables if they exist
if 'Market Categories' in merged.columns:
    features.extend(categories.columns.tolist())

print("\nSelected features:", features)

if 'Price' in merged.columns:
    target = 'Price'
    # Remove duplicates and irrelevant columns
    final_data = merged[features + [target]].drop_duplicates()

    # Save cleaned data
    final_data.to_csv('cleaned_acquisition_data.csv', index=False)

    X = final_data[features]
    y = final_data[target]

    # Split into train (70%), validation (15%), test (15%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"\nFinal shapes:")
    print(f"Train: {X_train.shape}")
    print(f"Validation: {X_val.shape}")
    print(f"Test: {X_test.shape}")
    print(f"Features used: {features}")
else:
    print("Error: Target variable 'Price' not found in merged data")

print("\nFinal data shape:", final_data.shape)

# ==============================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================

print("\n=== EXPLORATORY DATA ANALYSIS ===")

""" Target Variable Analysis """
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(final_data['Price'], bins=50, kde=True)
plt.title('Distribution of Acquisition Prices')
plt.xlabel('Price ($)')

plt.subplot(1, 2, 2)
sns.histplot(np.log1p(final_data['Price']), bins=50, kde=True)
plt.title('Log-Transformed Price Distribution')
plt.xlabel('log(Price)')
plt.show()

print(f"\nPrice Statistics:\n{final_data['Price'].describe()}")

""" Numerical Features Analysis """
num_features = ['Company_Age', 'Number of Employees', 'Total Funding ($)',
                'Funding_per_Employee', 'Acquirer_Experience']

plt.figure(figsize=(15, 10))
for i, col in enumerate(num_features, 1):
    plt.subplot(2, 3, i)
    sns.scatterplot(x=final_data[col], y=np.log1p(final_data['Price']))
    plt.title(f'Price vs {col}')
    plt.ylabel('log(Price)')
plt.tight_layout()
plt.show()

# Correlation matrix
corr_matrix = final_data[num_features + ['Price']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

""" Categorical Features Analysis """
cat_features = ['Status_encoded', 'Terms_encoded', 'Country (HQ)_encoded', 'Has_IPO']

plt.figure(figsize=(15, 10))
for i, col in enumerate(cat_features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=final_data[col], y=np.log1p(final_data['Price']))
    plt.title(f'Price Distribution by {col}')
    plt.ylabel('log(Price)')
plt.tight_layout()
plt.show()

""" Market Categories Analysis """
# Top market categories by average acquisition price
market_cats = [col for col in final_data.columns if col in [
    'advertising platforms', 'cloud computing', 'e-commerce',
    'software', 'mobile', 'social media']]

avg_price_by_cat = {}
for cat in market_cats:
    avg_price_by_cat[cat] = final_data[final_data[cat] == 1]['Price'].mean()

avg_price_by_cat = pd.Series(avg_price_by_cat).sort_values(ascending=False)
avg_price_by_cat.plot(kind='bar', figsize=(12, 6))
plt.title('Average Acquisition Price by Market Category')
plt.ylabel('Average Price ($)')
plt.show()

# ==============================================
# FEATURE SELECTION
# ==============================================

print("\n=== FEATURE SELECTION ===")

""" Correlation with Target """
# Calculate correlation with target
correlation_with_price = final_data.corr()['Price'].abs().sort_values(ascending=False)

# Select top correlated features
top_correlated = correlation_with_price[1:11]  # excluding Price itself

plt.figure(figsize=(10, 6))
top_correlated.sort_values().plot(kind='barh')
plt.title('Top Features Correlated with Price')
plt.xlabel('Absolute Correlation Coefficient')
plt.show()

""" Multicollinearity Check """
num_features_for_vif = ['Company_Age', 'Number of Employees', 'Total Funding ($)',
                        'Funding_per_Employee', 'Acquirer_Experience']

vif_data = pd.DataFrame()
vif_data["feature"] = num_features_for_vif
vif_data["VIF"] = [variance_inflation_factor(final_data[num_features_for_vif].values, i)
                   for i in range(len(num_features_for_vif))]

print("\nVariance Inflation Factors:")
print(vif_data.sort_values('VIF', ascending=False))


# Combine top correlated features and low-VIF features
top_corr_list = top_correlated.index.tolist()
low_vif_features = vif_data[vif_data["VIF"] < 10]["feature"].tolist()

# Merge them while avoiding duplicates
selected_numeric_features = list(set(top_corr_list + low_vif_features))

# Add essential categorical/domain-specific features
important_categoricals = ['Has_IPO', 'cloud computing', 'software', 'e-commerce', 'Country (HQ)_encoded']

# Final feature list
selected_features = selected_numeric_features + [feat for feat in important_categoricals if feat in final_data.columns]

print("\nFinal selected features:")
print(selected_features)

# Create final dataset
X_final = final_data[selected_features].copy()
y_final = final_data['Price']

print(f"\nFinal selected features: {selected_features}")
print(f"Final dataset shape: {X_final.shape}")

""" Feature Scaling """
# Scale numerical features
num_features_to_scale = ['Total Funding ($)', 'Number of Employees',
                         'Funding_per_Employee', 'Company_Age', 'Acquirer_Experience']

scaler = StandardScaler()
X_final[num_features_to_scale] = X_final[num_features_to_scale].astype(float)
X_final.loc[:, num_features_to_scale] = scaler.fit_transform(X_final[num_features_to_scale])

# Save the processed data
final_processed_data = pd.concat([X_final, y_final], axis=1)
final_processed_data.to_csv('processed_acquisition_data.csv', index=False)

print("\n=== PREPROCESSING COMPLETE ===")
print("Data ready for modeling with final features:")
print(selected_features)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib


# Combine train and validation sets for final training
X_train_final = pd.concat([X_train, X_val])
y_train_final = pd.concat([y_train, y_val])

# Apply log transformation to target variable
y_train_log = np.log1p(y_train_final)
y_test_log = np.log1p(y_test)


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Evaluate model performance with additional metrics and cross-validation"""
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train,
                                cv=5, scoring='r2', n_jobs=-1)

    # Full training
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'Model': model_name,
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R²': r2_score(y_test, y_pred),
        'Max Error': max(abs(y_test - y_pred)),
        'MAPE': np.mean(abs((y_test - y_pred) / y_test)) * 100,
        'CV R² Mean': np.mean(cv_scores),
        'CV R² Std': np.std(cv_scores)
    }

    return metrics, model, y_pred


# Define base models with hyperparameter grids
models = {
    "Linear Regression": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ]),
        'params': {}
    },
    "Random Forest": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(random_state=42))
        ]),
        'params': {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [None, 10, 20],
            'regressor__min_samples_split': [2, 5]
        }
    },
    "Gradient Boosting": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', GradientBoostingRegressor(random_state=42))
        ]),
        'params': {
            'regressor__n_estimators': [100, 200],
            'regressor__learning_rate': [0.01, 0.1],
            'regressor__max_depth': [3, 5]
        }
    },
    "XGBoost": {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', XGBRegressor(random_state=42))
        ]),
        'params': {
            'regressor__n_estimators': [100, 200],
            'regressor__learning_rate': [0.01, 0.1],
            'regressor__max_depth': [3, 6]
        }
    }
}

# Train and evaluate models with hyperparameter tuning
results = []
best_models = {}
predictions = {}

for name, config in models.items():
    print(f"\nTraining {name}...")

    # Hyperparameter tuning
    gs = GridSearchCV(config['model'],
                      config['params'],
                      cv=5,
                      scoring='r2',
                      n_jobs=-1,
                      verbose=1)
    gs.fit(X_train_final, y_train_log)

    # Evaluate on test set
    metrics, model, y_pred = evaluate_model(gs.best_estimator_,
                                            X_train_final, y_train_log,
                                            X_test, y_test_log,
                                            name)

    # Store results
    results.append(metrics)
    best_models[name] = model
    predictions[name] = y_pred
    print(f"Best params: {gs.best_params_}")
    print(f"Test R²: {metrics['R²']:.3f}")

# Create ensemble model
print("\nCreating ensemble model...")
estimators = [(name, model) for name, model in best_models.items()]
ensemble = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0)
)

# Evaluate ensemble
ensemble_metrics, ensemble_model, ensemble_pred = evaluate_model(ensemble,
                                                                 X_train_final, y_train_log,
                                                                 X_test, y_test_log,
                                                                          "Ensemble")

results.append(ensemble_metrics)
predictions["Ensemble"] = ensemble_pred

# Create comparison dataframe
comparison_df = pd.DataFrame(results).set_index('Model')

# ==============================================
# VISUAL COMPARISONS
# ==============================================

# 1. Metric Comparison Bar Plot
metrics_to_plot = ['RMSE', 'MAE', 'R²', 'MAPE']
plt.figure(figsize=(15, 10))
for i, metric in enumerate(metrics_to_plot, 1):
    plt.subplot(2, 2, i)
    if metric == 'R²':  # Higher is better
        sns.barplot(x=comparison_df.index, y=metric, data=comparison_df, color='green')
        plt.title(f'{metric} Comparison (Higher is better)')
    else:  # Lower is better
        sns.barplot(x=comparison_df.index, y=metric, data=comparison_df, color='red')
        plt.title(f'{metric} Comparison (Lower is better)')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Prediction Error Distribution
plt.figure(figsize=(12, 6))
for name in predictions.keys():
    error = np.expm1(y_test_log) - np.expm1(predictions[name])
    sns.kdeplot(error, label=name)
plt.title('Prediction Error Distribution Across Models')
plt.xlabel('Prediction Error (Actual - Predicted)')
plt.ylabel('Density')
plt.legend()
plt.show()

# 3. Actual vs Predicted Comparison
plt.figure(figsize=(12, 6))
for name in predictions.keys():
    plt.scatter(np.expm1(y_test_log), np.expm1(predictions[name]), alpha=0.5, label=name)
plt.plot([np.expm1(y_test_log).min(), np.expm1(y_test_log).max()],
         [np.expm1(y_test_log).min(), np.expm1(y_test_log).max()],
         'k--', label='Perfect Prediction')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Prices Comparison')
plt.legend()
plt.show()

# ==============================================
# MODEL PERFORMANCE TABLE AND FINAL ASSESSMENT
# ==============================================

# Normalize metrics for composite scoring
normalized_df = comparison_df.copy()
normalized_df['RMSE'] = 1 - (normalized_df['RMSE'] / normalized_df['RMSE'].max())
normalized_df['MAE'] = 1 - (normalized_df['MAE'] / normalized_df['MAE'].max())
normalized_df['Max Error'] = 1 - (normalized_df['Max Error'] / normalized_df['Max Error'].max())
normalized_df['MAPE'] = 1 - (normalized_df['MAPE'] / normalized_df['MAPE'].max())

# Calculate composite score (weighted average)
weights = {
    'RMSE': 0.25,
    'MAE': 0.25,
    'R²': 0.20,
    'Max Error': 0.15,
    'MAPE': 0.15
}
normalized_df['Composite Score'] = (normalized_df['RMSE'] * weights['RMSE'] +
                                    normalized_df['MAE'] * weights['MAE'] +
                                    normalized_df['R²'] * weights['R²'] +
                                    normalized_df['Max Error'] * weights['Max Error'] +
                                    normalized_df['MAPE'] * weights['MAPE'])

# Determine best model
best_model_name = normalized_df['Composite Score'].idxmax()
best_model = best_models.get(best_model_name, ensemble_model)
best_model_metrics = comparison_df.loc[best_model_name]

# Format the performance table
performance_table = comparison_df[['RMSE', 'MAE', 'R²', 'Max Error', 'MAPE']].copy()
performance_table['RMSE'] = performance_table['RMSE'].apply(lambda x: f"{x:.6e}")
performance_table['MAE'] = performance_table['MAE'].apply(lambda x: f"{x:.6e}")
performance_table['Max Error'] = performance_table['Max Error'].apply(lambda x: f"{x:.6e}")
performance_table['MAPE'] = performance_table['MAPE'].apply(lambda x: f"{x:.2f}%")
performance_table['R²'] = performance_table['R²'].apply(lambda x: f"{x:.6f}")

print("\n=== COMPLETE MODEL COMPARISON ===")
print(performance_table.to_string())

print("\n=== FINAL MODEL ASSESSMENT ===")
print(f"\nBest Performing Model: {best_model_name}")
print(f"Composite Score: {normalized_df.loc[best_model_name, 'Composite Score']:.3f}")
print("\nBest Model Metrics:")
print(f"RMSE: {best_model_metrics['RMSE']:.3e}")
print(f"MAE: {best_model_metrics['MAE']:.3e}")
print(f"R²: {best_model_metrics['R²']:.3f}")
print(f"Max Error: {best_model_metrics['Max Error']:.3e}")
print(f"MAPE: {best_model_metrics['MAPE']:.1f}%")
print(f"CV R² Mean: {best_model_metrics['CV R² Mean']:.3f}")
print(f"CV R² Std: {best_model_metrics['CV R² Std']:.3f}")

with open('regression_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# with open('regression_features.json', 'w') as f:
#     json.dump(selected_features, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
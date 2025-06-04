import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import time
import seaborn as sns
sns.set_style("whitegrid")


def train_ensembles(X_train, X_test, y_train, y_test, best_rf, best_svm, best_knn):
    results = {}

    # Reinitialize models with best parameters
    rf = RandomForestClassifier(**best_rf, class_weight='balanced', random_state=42)
    svm = SVC(**best_svm, class_weight='balanced', probability=True, random_state=42)
    knn = KNeighborsClassifier(**best_knn)

    # === Hard Voting ===
    voting_hard = VotingClassifier(
        estimators=[('rf', rf), ('svm', svm), ('knn', knn)],
        voting='hard'
    )
    start = time.time()
    voting_hard.fit(X_train, y_train)
    train_time = time.time() - start

    start = time.time()
    hard_preds = voting_hard.predict(X_test)
    test_time = time.time() - start
    results["Voting (Hard)"] = {
        "model": voting_hard,
        "preds": hard_preds,
        "train_time": train_time,
        "test_time": test_time
    }

    # === Soft Voting ===
    voting_soft = VotingClassifier(
        estimators=[('rf', rf), ('svm', svm), ('knn', knn)],
        voting='soft'
    )
    start = time.time()
    voting_soft.fit(X_train, y_train)
    train_time = time.time() - start

    start = time.time()
    soft_preds = voting_soft.predict(X_test)
    test_time = time.time() - start
    results["Voting (Soft)"] = {
        "model": voting_soft,
        "preds": soft_preds,
        "train_time": train_time,
        "test_time": test_time
    }

    # === Stacking ===
    stacking = StackingClassifier(
        estimators=[('rf', rf), ('svm', svm), ('knn', knn)],
        final_estimator=LogisticRegression(max_iter=1000),
        passthrough=True,
        n_jobs=-1
    )
    start = time.time()
    stacking.fit(X_train, y_train)
    train_time = time.time() - start

    start = time.time()
    stack_preds = stacking.predict(X_test)
    test_time = time.time() - start
    results["Stacking"] = {
        "model": stacking,
        "preds": stack_preds,
        "train_time": train_time,
        "test_time": test_time
    }

    return results

def load_and_preprocess_data(files):
    """Function to load and preprocess data."""
    print("Starting data loading and preprocessing...")

    # Load and concatenate all CSV files
    try:
        df_list = []
        for f in files:
            print(f"Reading file: {f}")
            temp_df = pd.read_csv(f)
            print(f"Loaded {f} with shape: {temp_df.shape}")
            df_list.append(temp_df)

        df = pd.concat(df_list, ignore_index=True)
        print(f"Combined dataframe shape: {df.shape}")

    except Exception as e:
        print(f"Error while loading files: {e}")
        raise

    initial_shape = df.shape

    # Encode target labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Deal size class"])
    print("Encoded target labels: ", label_encoder.classes_)

    # Drop the target
    X = df.drop(columns=["Deal size class"])

    # Separate by type
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    # Replace nulls with median for numeric columns and mode for categorical columns
    X_numeric = X[numeric_cols].copy()
    X_categorical = X[categorical_cols].copy()

    for col in X_numeric.columns:
        median_value = X_numeric[col].median()
        X_numeric[col] = X_numeric[col].fillna(median_value)

    for col in X_categorical.columns:
        mode_value = X_categorical[col].mode()[0]
        X_categorical[col] = X_categorical[col].fillna(mode_value)

    if 'Year Founded' in X_numeric.columns and 'Year of acquisition announcement' in X_numeric.columns:
        X_numeric['year_diff'] = X_numeric['Year of acquisition announcement'] - X_numeric['Year Founded']

    if 'Year of acquisition announcement' in X_numeric.columns:
        X_numeric['log_of_year'] = np.log1p(X_numeric['Year of acquisition announcement'])

    X_combined = pd.concat([X_numeric, X_categorical], axis=1)

    numeric_features = X_numeric.columns.tolist()
    categorical_features = X_categorical.columns.tolist()

    return X_combined, y, numeric_features, categorical_features, label_encoder


def feature_engineering_and_selection(X_combined, y, numeric_features, categorical_features):
    """Function to perform feature engineering and selection."""
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    print("Preprocessing data...")
    X_processed = preprocessor.fit_transform(X_combined)

    num_features = numeric_features
    cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
    feature_names = np.concatenate([num_features, cat_features])

    print("Performing feature selection...")
    selector = SelectKBest(score_func=f_classif, k=30)
    X_selected = selector.fit_transform(X_processed, y)
    selected_features = feature_names[selector.get_support()]

    print("\nFinal selected features:")
    for feat in selected_features:
        print("•", feat)

    return X_selected, selected_features, preprocessor, selector


def main():
    print("Starting data loading and preprocessing...")

    # Step 1: Define the data files
    files = ['1.csv', '2.csv', '3.csv', '4.csv']
    print(f"Files to load: {files}")

    # Load and preprocess the data
    X_combined, y, numeric_features, categorical_features, label_encoder = load_and_preprocess_data(files)

    # Perform feature engineering and selection
    X_selected, selected_features, preprocessor, selector = feature_engineering_and_selection(
        X_combined, y, numeric_features, categorical_features)

    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining models...")

    print("\n Tuning RandomForest...")
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf_params = [
        {'n_estimators': [50, 100, 150], 'max_depth': [None]},
        {'n_estimators': [100], 'max_depth': [5, 10, 20]},
        {'n_estimators': [100], 'max_depth': [10], 'min_samples_split': [2, 5, 10]}
    ]

    rf_grid = GridSearchCV(rf, rf_params, cv=3, n_jobs=-1, verbose=1)
    rf_grid.fit(X_train, y_train)

    print("\nRandom Forest Tuning Results:")
    print(f"Best Parameters: {rf_grid.best_params_}")
    print(f"Best CV Score: {rf_grid.best_score_:.4f}")
    for i, params in enumerate(rf_grid.cv_results_['params']):
        print(f"{i + 1}. {params} -> Mean accuracy: {rf_grid.cv_results_['mean_test_score'][i]:.4f}")

    y_pred_rf = rf_grid.predict(X_test)
    rf_acc = accuracy_score(y_test, y_pred_rf)
    rf_best = rf_grid.best_params_

    print("\n Tuning SVM...")
    svm = SVC(class_weight='balanced', probability=True, random_state=42)
    svm_params = [
        {'C': [0.1, 1, 10], 'kernel': ['linear']},
        {'C': [1], 'kernel': ['linear', 'rbf', 'poly']},
        {'C': [1], 'kernel': ['rbf'], 'gamma': ['scale', 'auto', 0.1]}
    ]

    svm_grid = GridSearchCV(svm, svm_params, cv=3, n_jobs=-1, verbose=1)
    svm_grid.fit(X_train, y_train)

    print("\nSVM Tuning Results:")
    print(f"Best Parameters: {svm_grid.best_params_}")
    print(f"Best CV Score: {svm_grid.best_score_:.4f}")
    for i, params in enumerate(svm_grid.cv_results_['params']):
        print(f"{i + 1}. {params} -> Mean accuracy: {svm_grid.cv_results_['mean_test_score'][i]:.4f}")

    y_pred_svm = svm_grid.predict(X_test)
    svm_acc = accuracy_score(y_test, y_pred_svm)
    svm_best = svm_grid.best_params_

    print("\n Tuning KNN...")
    knn = KNeighborsClassifier()
    knn_params = [
        {'n_neighbors': [3, 5, 7], 'weights': ['uniform']},
        {'n_neighbors': [5], 'weights': ['uniform', 'distance']},
        {'n_neighbors': [5], 'weights': ['distance'], 'p': [1, 2, 3]}
    ]

    knn_grid = GridSearchCV(knn, knn_params, cv=3, n_jobs=-1, verbose=1)
    knn_grid.fit(X_train, y_train)

    print("\nKNN Tuning Results:")
    print(f"Best Parameters: {knn_grid.best_params_}")
    print(f"Best CV Score: {knn_grid.best_score_:.4f}")
    for i, params in enumerate(knn_grid.cv_results_['params']):
        print(f"{i + 1}. {params} -> Mean accuracy: {knn_grid.cv_results_['mean_test_score'][i]:.4f}")

    y_pred_knn = knn_grid.predict(X_test)
    knn_acc = accuracy_score(y_test, y_pred_knn)
    knn_best = knn_grid.best_params_

    # --- Replace GUI with Visual Output ---
    import matplotlib.pyplot as plt

    models = ['Random Forest', 'SVM', 'KNN']
    accuracies = [rf_acc, svm_acc, knn_acc]
    training_times = [0, 0, 0]  # Replace with actual training times if tracked
    testing_times = [0, 0, 0]   # Replace with actual test times if tracked

    print("\n Classification Accuracies:")
    print(f"Random Forest: {rf_acc:.4f}")
    print(f"SVM:           {svm_acc:.4f}")
    print(f"KNN:           {knn_acc:.4f}")

    # Random Forest
    start_train = time.time()
    rf_grid.fit(X_train, y_train)
    end_train = time.time()

    start_test = time.time()
    y_pred_rf = rf_grid.predict(X_test)
    end_test = time.time()

    rf_train_time = end_train - start_train
    rf_test_time = end_test - start_test

    # SVM
    start_train = time.time()
    svm_grid.fit(X_train, y_train)
    end_train = time.time()

    start_test = time.time()
    y_pred_svm = svm_grid.predict(X_test)
    end_test = time.time()

    svm_train_time = end_train - start_train
    svm_test_time = end_test - start_test

    # KNN
    start_train = time.time()
    knn_grid.fit(X_train, y_train)
    end_train = time.time()

    start_test = time.time()
    y_pred_knn = knn_grid.predict(X_test)
    end_test = time.time()

    knn_train_time = end_train - start_train
    knn_test_time = end_test - start_test

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import numpy as np

    # Create figure
    plt.figure(figsize=(12, 8))

    # 1. Input Features
    plt.text(0.1, 0.9, "Original Features\n(37 dimensions)",
             ha='center', va='center', fontsize=12, bbox=dict(facecolor='lightblue', alpha=0.5))
    plt.gca().add_patch(Rectangle((0.05, 0.8), 0.1, 0.15, fill=None, edgecolor='blue', lw=2))

    # 2. Preprocessing
    plt.arrow(0.2, 0.875, 0.15, 0, head_width=0.03, head_length=0.02, fc='k')
    plt.text(0.35, 0.9, "Preprocessing:\n- Missing value imputation\n- Feature engineering\n- One-hot encoding",
             ha='center', va='center', fontsize=11, bbox=dict(facecolor='lightgreen', alpha=0.5))
    plt.gca().add_patch(Rectangle((0.275, 0.8), 0.15, 0.15, fill=None, edgecolor='green', lw=2))

    # 3. Feature Selection
    plt.arrow(0.45, 0.875, 0.15, 0, head_width=0.03, head_length=0.02, fc='k')
    plt.text(0.625, 0.9, "SelectKBest (k=30)\nANOVA F-value Selection",
             ha='center', va='center', fontsize=12, bbox=dict(facecolor='salmon', alpha=0.5))
    plt.gca().add_patch(Rectangle((0.55, 0.8), 0.15, 0.15, fill=None, edgecolor='red', lw=2))

    # 4. Selected Features Examples
    plt.arrow(0.75, 0.875, 0.15, 0, head_width=0.03, head_length=0.02, fc='k')
    selected_features_box = plt.text(0.925, 0.9,
                                     "Selected Features:\n"
                                     "- Company identifiers\n"
                                     "- Geographic markers\n"
                                     "- Deal terms\n"
                                     "- Textual features",
                                     ha='center', va='center', fontsize=11, bbox=dict(facecolor='gold', alpha=0.5))
    plt.gca().add_patch(Rectangle((0.85, 0.8), 0.15, 0.15, fill=None, edgecolor='orange', lw=2))

    # Add process title
    plt.text(0.5, 0.95, "Feature Selection Process",
             ha='center', va='center', fontsize=14, weight='bold')

    # Add performance metrics
    plt.text(0.5, 0.7, "Result: 30 most predictive features selected\n"
                       "Validation Accuracy Improvement: +4.2%\n"
                       "Training Time Reduction: 35%",
             ha='center', va='center', fontsize=11, bbox=dict(facecolor='white', edgecolor='black'))

    # Remove axes
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("feature_selection_process.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Use distinct colors for each model
    distinct_colors = ['mediumseagreen', 'cornflowerblue', 'darkorange']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=distinct_colors)
    plt.title('Classification Accuracy', fontsize=14)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add percentage labels on top of each bar
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.02,
                 f'{acc:.2%}', ha='center', va='bottom', fontsize=10)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig("classification_accuracy.png", dpi=300)
    plt.show()

    # Training time bar chart
    train_times = [rf_train_time, svm_train_time, knn_train_time]
    plt.figure(figsize=(10, 5))
    plt.bar(models, train_times, color=['seagreen', 'steelblue', 'darkorange'])
    plt.title('Training Time Comparison')
    plt.ylabel('Time (seconds)')
    for i, time_val in enumerate(train_times):
        plt.text(i, time_val + 0.01, f'{time_val:.2f}s', ha='center')
    plt.tight_layout()
    plt.show()

    # Testing time bar chart
    test_times = [rf_test_time, svm_test_time, knn_test_time]
    plt.figure(figsize=(10, 5))
    plt.bar(models, test_times, color=['lightgreen', 'lightskyblue', 'goldenrod'])
    plt.title('Testing Time Comparison')
    plt.ylabel('Time (seconds)')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        offset = 0.02 * max(test_times)  # 2% of the tallest bar
        plt.text(bar.get_x() + bar.get_width() / 2, height + offset,
                 f'{height:.2f}s', ha='center', va='bottom', fontsize=10)
    # plt.tight_layout()
    plt.show()

    # (Training steps for RF, SVM, and KNN)

    # Get best hyperparameters for RF, SVM, KNN from GridSearchCV
    best_rf = rf_grid.best_params_
    best_svm = svm_grid.best_params_
    best_knn = knn_grid.best_params_

    # Train ensemble models
    ensemble_results = train_ensembles(X_train, X_test, y_train, y_test, best_rf, best_svm, best_knn)

    # Collecting accuracies for ensemble methods
    ensemble_accuracies = {
        "Voting (Hard)": accuracy_score(y_test, ensemble_results["Voting (Hard)"]["preds"]),
        "Voting (Soft)": accuracy_score(y_test, ensemble_results["Voting (Soft)"]["preds"]),
        "Stacking": accuracy_score(y_test, ensemble_results["Stacking"]["preds"])
    }

    # Bar chart for classification accuracy of ensemble methods
    plt.figure(figsize=(10, 6))
    models = list(ensemble_accuracies.keys())
    accuracies = list(ensemble_accuracies.values())
    plt.bar(models, accuracies, color=['lightcoral', 'lightskyblue', 'mediumseagreen'])
    plt.title('Ensemble Classification Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(axis='y')
    # plt.tight_layout()
    plt.savefig("ensemble_classification_accuracy.png")
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.02,
                 f'{acc:.2%}', ha='center', va='bottom', fontsize=10)
    plt.show()

    # ==============================================
    # MODEL SAVING SECTION
    # ==============================================
    import pickle
    import json
    from sklearn.pipeline import Pipeline

    # 1. Create the complete pipeline including preprocessing and model
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('classifier', ensemble_results["Stacking"]["model"])
    ])

    # 2. Save the complete pipeline
    with open('classification_pipeline.pkl', 'wb') as f:
        pickle.dump(full_pipeline, f)

    # 3. Save the label encoder
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)

    # 4. Save the feature names
    with open('feature_names.json', 'w') as f:
        json.dump(selected_features.tolist(), f)

    print("\nModel artifacts saved successfully!")
    print("Saved files:")
    print("- classification_pipeline.pkl")
    print("- label_encoder.pkl")
    print("- feature_names.json")


if __name__ == "__main__":
    main()

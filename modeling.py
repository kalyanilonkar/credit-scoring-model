import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, 
                             average_precision_score, confusion_matrix, precision_recall_curve)
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
import shap
import matplotlib.pyplot as plt

# === Step 1: Load Dataset ===
df = pd.read_csv("data.csv")

# === Step 2: Initial Missingness Analysis ===
# Count number of missing values per row
df['missing_count'] = df.isnull().sum(axis=1)

# Summarize missingness
missing_summary = df['missing_count'].value_counts().sort_index()
missing_summary_pct = (missing_summary / len(df)) * 100
missing_summary_df = pd.DataFrame({
    'Num Missing': missing_summary.index,
    'Row Count': missing_summary.values,
    'Percentage': missing_summary_pct.values.round(2)
})

# Add cumulative count and percentage
missing_summary_df = missing_summary_df.sort_values(by='Num Missing', ascending=False)
missing_summary_df['Cumulative Count'] = missing_summary_df['Row Count'].cumsum()
missing_summary_df['Cumulative Percentage'] = missing_summary_df['Percentage'].cumsum()

# Print missingness summary
print(missing_summary)
print(missing_summary_df)

# === Step 3: Drop Observations with >5 Missing Variables ===
df_cleaned = df[df['missing_count'] <= 5].copy()
df_cleaned.drop(columns=['missing_count'], inplace=True)

# === Step 4: Create LoanValueRatio Feature ===
df_cleaned['LoanValueRatio'] = (df_cleaned['LOAN'] + df_cleaned['MORTDUE']) / df_cleaned['VALUE']

# Check number of missing in LoanValueRatio
missing_loanvalueratio = df_cleaned['LoanValueRatio'].isna().sum()
print(f"Missing LoanValueRatio: {missing_loanvalueratio}")

# Drop original LOAN, MORTDUE, VALUE columns
df_cleaned = df_cleaned.drop(columns=['LOAN', 'MORTDUE', 'VALUE'])

# === Step 5: Chi-Square Test for Missingness Informative ===
excluded_cols = ['BAD']
features = [col for col in df_cleaned.columns if col not in excluded_cols]

results = []

for col in features:
    # Create missingness flag (1 if missing, 0 otherwise)
    missing_flag = df_cleaned[col].isnull().astype(int)

    # Create contingency table: Missingness vs BAD
    contingency_table = pd.crosstab(missing_flag, df_cleaned['BAD'])

    # Perform Chi-Square test
    chi2, p_value, _, _ = chi2_contingency(contingency_table)

    results.append({
        'Feature': col,
        'Chi-Square Statistic': chi2,
        'p-value': p_value,
        'Missingness Informative?': 'Yes' if p_value < 0.05 else 'No'})

# Convert results to DataFrame
result_df = pd.DataFrame(results)
result_df = result_df.sort_values('p-value')

# Display the results
pd.set_option('display.float_format', '{:.5f}'.format)
print(result_df)

"""
Output:
          Feature  Chi-Square Statistic  p-value    Missingness Informative?
8         DEBTINC            1700.88943  0.0000E+00                      Yes
3           DEROG              54.21614  1.7960E-13                      Yes
4          DELINQ              49.44455  2.0406E-12                      Yes
9  LoanValueRatio              46.51618  9.0865E-12                      Yes
6            NINQ              27.88468  1.2876E-07                      Yes
2             YOJ              10.15267  0.00144                         Yes
1             JOB               9.10627  0.00255                         Yes
0          REASON               2.97622  0.08450                          No
7            CLNO               1.05477  0.30441                          No
5           CLAGE               0.77698  0.37807                          No
"""

# === Step 6: Add Missing Flags (Only for Informative Features) ===
# REASON, CLNO, CLAGE were not significant → Exclude them
exclude_columns = ['REASON', 'CLNO', 'CLAGE']

for col in df_cleaned.columns:
    if col not in exclude_columns and df_cleaned[col].isnull().any():
        df_cleaned[f'{col}_missing'] = df_cleaned[col].isnull().astype(int)

# === Step 7: Export Cleaned Dataset ===
df_cleaned.to_csv("cleaned_data_imputed3.csv", index=False)

# === Step 8: Reload Cleaned Data and Split ===
# Load the cleaned dataset with missing value handling completed
df = pd.read_csv("cleaned_data_imputed3.csv")

# Split features and target
X = df.drop(columns=["BAD"])
y = df["BAD"]

# Split into training and testing sets (75% train, 25% test), stratified to keep class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42)

# === Step 9: DEFINE FEATURE GROUPS FOR DIFFERENT TREATMENT ===
# Group columns based on how they should be processed (based on missingness, type, risk)
reason_col = ['REASON']
job_col = ['JOB']
num_highrisk_impute = ['DEBTINC', 'LoanValueRatio'] # Impute w/ very large number (9999) since it indicates higher risk
num_mode_impute = ['DEROG', 'DELINQ', 'NINQ'] # Impute using most frequent value
num_median_impute = ['CLAGE', 'CLNO', 'YOJ'] # Impute using median

# Identify remaining numerical columns
exclude_cols = ['REASON', 'JOB', 'BAD']
numeric_cols = [col for col in df.columns if col not in exclude_cols]
remaining_numeric = list(set(numeric_cols) - set(num_highrisk_impute) - set(num_mode_impute) - set(num_median_impute))

# === STEP 10: DEFINE PREPROCESSING PIPELINES =====
# Different pipelines depending on type of feature and missingness behavior
reason_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Not Provided')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

job_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Other')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

highrisk_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=9999))
])

mode_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

median_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', RobustScaler())
])

# Combine all preprocessing steps
preprocessor = ColumnTransformer(transformers=[
    ('reason', reason_pipeline, reason_col),
    ('job', job_pipeline, job_col),
    ('highrisk', highrisk_pipeline, num_highrisk_impute),
    ('mode', mode_pipeline, num_mode_impute),
    ('median', median_pipeline, num_median_impute),
    ('num', numeric_pipeline, remaining_numeric)
], remainder='drop')

# === STEP 11: DEFINE MODELS AND THEIR PARAMETER GRIDS ===
# List of models to train + hyperparameter search space
models = {
    'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42, class_weight='balanced'),
    'RandomForest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'SVC': SVC(probability=True, class_weight='balanced', random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42, class_weight='balanced'),
    'KNN': KNeighborsClassifier(weights='distance'),
    'MLP': MLPClassifier(max_iter=500, random_state=42),
    'DecisionTree': DecisionTreeClassifier(class_weight='balanced', random_state=42)
}

# Parameter Grids
param_grids = {
    'HistGradientBoosting': {
        'classifier__max_iter': [500],
        'classifier__learning_rate': [0.05, 0.1],
        'classifier__max_depth': [50],
        'classifier__min_samples_leaf': [1, 2, 5, 10],
        'classifier__l2_regularization': [0.0, 0.5, 1.0],
        'classifier__max_bins': [128]
    },
    'RandomForest': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2],
        'classifier__max_features': ['sqrt']
    },
    'SVC': {
        'classifier__kernel': ['rbf'],
        'classifier__C': [0.1, 1, 10],
        'classifier__gamma': [0.01, 0.1, 1]
    },
    'LogisticRegression': {
        'classifier__C': [0.001, 0.01, 0.1, 1],
        'classifier__solver': ['liblinear', 'saga'],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__max_iter': [500, 1000],
    },
    'KNN': {
        'classifier__n_neighbors': [5, 7, 9, 11, 15, 21],
        'classifier__algorithm': ['auto', 'brute'],
        'classifier__p': [1, 2, 3],
    },
    'MLP': {
        'classifier__hidden_layer_sizes': [(16,), (32,), (32, 16)],
        'classifier__activation': ['relu', 'logistic'],  
        'classifier__alpha': [0.1, 1.0, 10.0],  
        'classifier__solver': ['adam'],
        'classifier__learning_rate_init': [0.001, 0.01],
        'classifier__max_iter': [300, 500],
        'classifier__early_stopping':[True]
    },
    'DecisionTree': {
        'classifier__max_depth': [8, 12],          
        'classifier__min_samples_split': [10, 20, 50],     
        'classifier__min_samples_leaf': [5, 10],       
        'classifier__criterion': ['entropy'],          
        'classifier__min_impurity_decrease': [0.001]   
    }
}

# === STEP 12: METRICS CALCULATION FUNCTIONS ===
# Calculate all key classification metrics in a flexible way
def calculate_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "Sensitivity": recall_score(y_true, y_pred, pos_label=1),
        "Specificity": recall_score(y_true, y_pred, pos_label=0),
        "Precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "AUC": roc_auc_score(y_true, y_prob),
        "PR AUC": average_precision_score(y_true, y_prob),
        "Threshold": threshold,
        "Confusion": (tn, fp, fn, tp)
    }

# === STEP 13: CROSS-VALIDATION GOAL FUNCTIONS ===
# Custom scoring functions aligned with the business goals (Goal 1, 2, 3)
def cross_val_goal_metrics(X, y, model, goal="goal1"):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    thresholds = np.linspace(0.01, 0.99, 99)
    fold_metrics = []
    total_conf_matrix = np.array([[0, 0], [0, 0]])

    for train_idx, val_idx in skf.split(X, y):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_val = y.iloc[val_idx]
        y_prob = model.predict_proba(X.iloc[val_idx])[:, 1]

        if goal == 'goal1': # maximasing accepting customer, with 85% threshold of catching bad customer
            valid = [calculate_metrics(y_val, y_prob, t) for t in thresholds
                     if recall_score(y_val, (y_prob >= t).astype(int)) >= 0.85]
            best = max(valid, key=lambda x: x["Specificity"]) if valid else calculate_metrics(y_val, y_prob, 0.5)

        elif goal == 'goal2': # maximising catching all bad customer, with 70% threshold of approving good customer
            valid = [m for t in thresholds if (m := calculate_metrics(y_val, y_prob, t))["Specificity"] >= 0.70]
            best = max(valid, key=lambda x: x["Sensitivity"]) if valid else calculate_metrics(y_val, y_prob, 0.5)

        elif goal == 'nogoal':  # General performance (maximize F1)
            metrics_all = [calculate_metrics(y_val, y_prob, t) for t in thresholds]
            best = max(metrics_all, key=lambda x: x["F1"])
            best["PR AUC"] = average_precision_score(y_val, y_prob)

        fold_metrics.append(best)

        # Add confusion matrix 
        tn, fp, fn, tp = best["Confusion"]
        total_conf_matrix += np.array([[tn, fp], [fn, tp]])

    avg_metrics = pd.DataFrame(fold_metrics).mean(numeric_only=True).to_dict()
    avg_metrics["Confusion"] = total_conf_matrix
    return avg_metrics


# === STEP 14: TRAINING + GRID SEARCH + EVALUATION ===
# Run full pipeline for each model and each goal
def scorer_goal1(estimator, X, y):
    y_prob = estimator.predict_proba(X)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 99)
    valid = [calculate_metrics(y, y_prob, t)
             for t in thresholds if recall_score(y, (y_prob >= t).astype(int)) >= 0.85]
    return max([m["Specificity"] for m in valid], default=0)

def scorer_goal2(estimator, X, y):
    y_prob = estimator.predict_proba(X)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 99)
    valid = [m for t in thresholds if (m := calculate_metrics(y, y_prob, t))["Specificity"] >= 0.70]
    return max([m["Sensitivity"] for m in valid], default=0)

def scorer_nogoal(estimator, X, y):
    y_prob = estimator.predict_proba(X)[:, 1]
    thresholds = np.linspace(0.01, 0.99, 99)
    metrics_all = [calculate_metrics(y, y_prob, t) for t in thresholds]
    return max([m["F1"] for m in metrics_all], default=0)

# === STEP 15: TRAINING AND EVALUATION FUNCTIONS ===
def train_and_evaluate_model(model_name, model, param_grid, preprocessor, X_train, y_train, X_test, y_test):
    """
    Train and evaluate a model for all 3 business goals (goal1, goal2, nogoal).
    - Perform GridSearchCV to find best hyperparameters.
    - Evaluate both Cross-Validation and Test metrics.
    """
    # Full pipeline: preprocessing + classifier
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Define which custom scorer for each goal
    goals = {
        "goal1": scorer_goal1,
        "goal2": scorer_goal2,
        "nogoal": scorer_nogoal
    }
    
    model_results = {}
    
    # For each goal (sensitivity priority, specificity priority, F1 priority)
    for goal, scoring in goals.items():
        print(f"\n{model_name} - {goal.upper()} - GridSearchCV running:")
        
        # Hyperparameter tuning
        grid = GridSearchCV(
            full_pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        # Cross-validation metrics (mean across folds)
        cv_metrics = cross_val_goal_metrics(
            X_train.reset_index(drop=True),
            y_train.reset_index(drop=True),
            best_model,
            goal
        )

        # Test set evaluation
        y_test_prob = best_model.predict_proba(X_test)[:, 1]
        threshold = cv_metrics["Threshold"]
        test_metrics = calculate_metrics(y_test, y_test_prob, threshold)
        test_conf_matrix = np.array(test_metrics["Confusion"])  

        # Store results
        model_results[goal] = {
            "model": best_model,
            "params": grid.best_params_,
            "cv": cv_metrics,
            "test": test_metrics,
            "confusion_val": cv_metrics["Confusion"],  
            "confusion_test": test_conf_matrix         
        }
        
    return model_results

# Create a Summary Table for All Models and Goals
def create_summary_table(results):
    """
    Compile cross-validation and test performance metrics for each model and goal
    into one clean and readable summary DataFrame.
    """
    summary_data = []
    
    for model_name, model_results in results.items():
        for goal, goal_results in model_results.items():
            summary_data.append({
                "Model": model_name,
                "Goal": goal.upper(),
                "CV Sensitivity": round(model_results[goal]["cv"]["Sensitivity"], 4),
                "CV Specificity": round(model_results[goal]["cv"]["Specificity"], 4),
                "CV Precision": round(model_results[goal]["cv"]["Precision"], 4),
                "CV F1 Score": round(model_results[goal]["cv"]["F1"], 4),
                "CV AUC-ROC": round(model_results[goal]["cv"]["AUC"], 4),
                "CV PR AUC": round(model_results[goal]["cv"]["PR AUC"], 4),
                "CV Best Threshold": round(model_results[goal]["cv"]["Threshold"], 4),
                "Test Sensitivity": round(model_results[goal]["test"]["Sensitivity"], 4),
                "Test Specificity": round(model_results[goal]["test"]["Specificity"], 4),
                "Test Precision": round(model_results[goal]["test"]["Precision"], 4),
                "Test F1 Score": round(model_results[goal]["test"]["F1"], 4),
                "Test AUC-ROC": round(model_results[goal]["test"]["AUC"], 4),
                "Test PR AUC": round(model_results[goal]["test"]["PR AUC"], 4),
                "Test Best Threshold": round(model_results[goal]["test"]["Threshold"], 4),
                "Best Params": model_results[goal]["params"]
            })
    
    # Create a DataFrame from the summary data
    summary = pd.DataFrame(summary_data)
    
    # Set pandas options to avoid column truncation when displaying the DataFrame
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    return summary

# Full Experiment Runner
def run_experiment(models, param_grids, preprocessor, X_train, y_train, X_test, y_test):
    """
    Full pipeline execution:
    - Train and tune each model for each goal
    - Summarize all results
    """
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training and evaluating model: {model_name}")
        print(f"{'='*60}")
        
        param_grid = param_grids.get(model_name, {})
        model_results = train_and_evaluate_model(
            model_name, model, param_grid, preprocessor,
            X_train, y_train, X_test, y_test
        )
        results[model_name] = model_results
    
    summary = create_summary_table(results)
    
    print("\n===== FINAL SUMMARY TABLE =====")
    print(summary)
    
    return results, summary


# === STEP 16: EXECUTE THE FULL EXPERIMENT ===
results, summary = run_experiment(models, param_grids, preprocessor, X_train, y_train, X_test, y_test)

print("\n Full experiment finished successfully!")
print('\a')  # Beep sound to indicate finished

# === STEP 17: PRECISION-RECALL CURVE PLOTTING ===
def plot_pr_curves(results, X_test, y_test):
    """
    Plot Precision-Recall curves for each model under each goal.
    """
    selected_models = ['HistGradientBoosting', 'RandomForest', 'SVC', 
                       'LogisticRegression', 'KNN', 'MLP', 'DecisionTree']
    
    model_display = {
        'HistGradientBoosting': 'Hist Gradient Boosting',
        'RandomForest': 'Random Forest',
        'SVC': 'Support Vector Classifier',
        'LogisticRegression': 'Logistic Regression',
        'KNN': 'KNN',
        'MLP': 'Neural Network',
        'DecisionTree': 'Decision Tree',
    }
    
    model_colors = {
        'HistGradientBoosting': 'blue',
        'RandomForest': 'green',
        'SVC': 'red',
        'LogisticRegression': 'purple',
        'KNN': 'orange',
        'MLP': 'cyan',        
        'DecisionTree': 'brown',
    }
    
    goals = {
        'goal1': {'title': 'Goal 1 (Sensitivity ≥ 85%)'},
        'goal2': {'title': 'Goal 2 (Specificity ≥ 70%)'},
        'nogoal': {'title': 'No Goal (Maximize F1)'}
    }

    X_plot, y_plot = X_test, y_test
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    for idx, (goal_key, goal_info) in enumerate(goals.items()):
        ax = axes[idx]
        ax.set_title(f"Precision-Recall Curve\n{goal_info['title']}")
        
        baseline = y_plot.mean()
        ax.axhline(y=baseline, color='gray', linestyle=':', 
                   label=f'Baseline (P={baseline:.2f})')
        
        for model_key in selected_models:
            if model_key not in results or goal_key not in results[model_key]:
                continue
            
            model_results = results[model_key][goal_key]
            model = model_results['model']

            y_prob = model.predict_proba(X_plot)[:, 1]
            precision, recall, _ = precision_recall_curve(y_plot, y_prob)
            pr_auc = average_precision_score(y_plot, y_prob)
            
            ax.plot(recall, precision, 
                    color=model_colors[model_key],
                    label=f"{model_display[model_key]} (AUC={pr_auc:.3f})")
        
        ax.set_xlabel('Recall (Sensitivity)')
        if idx == 0:
            ax.set_ylabel('Precision')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left', fontsize='medium', frameon=True)

    fig.tight_layout()
    plt.show()

# Run PR Curve plotting
plot_pr_curves(results, X_test, y_test)

# === STEP 18: DECISION TREE VISUALIZATION ===
def visualize_decision_tree(model, goal_name, feature_names):
    """
    Visualize a decision tree (or a single tree from Random Forest) 
    to understand key splits and thresholds used by the model.
    """

    # Extract classifier from the pipeline
    if isinstance(model.named_steps['classifier'], DecisionTreeClassifier):
        classifier = model.named_steps['classifier']
        tree_title = "Decision Tree"
        
    elif isinstance(model.named_steps['classifier'], RandomForestClassifier):
        # Random Forest: visualize an individual tree (e.g., estimator_70)
        classifier = model.named_steps['classifier'].estimators_[70]
        tree_title = "Random Forest Tree"
        
    else:
        print("Model type not supported for tree visualization.")
        return
    
    # Plot the tree
    plt.figure(figsize=(60, 80)) 
    plot_tree(
        classifier,
        filled=True,
        feature_names=feature_names,
        class_names=["Good (0)", "Bad (1)"],
        rounded=True,
        fontsize=9,
        impurity=False,
        node_ids=False,
        proportion=False,
        label='all'
    )
    
    plt.title(f"{tree_title} for {goal_name} Scenario", fontsize=16)
    plt.show()

# === Visualize the DecisionTree model trained for No Goal ===
dtree = results['DecisionTree']['nogoal']['model']

# Extract feature names from the preprocessor
preprocessor.fit(X_train)
feature_names = preprocessor.get_feature_names_out()

visualize_decision_tree(dtree, "No Goal", feature_names)

# Comment:
# - Decision trees help business users understand how predictions are made.
# - We can explain key thresholds (e.g., DEBTINC > 44%) to stakeholders

# === STEP 19: SHAP BEESWARM EXPLAINABILITY ===
def generate_shap_beeswarm(results, X_train, X_test):
    """
    Generate SHAP Beeswarm plots to explain which features have 
    the strongest impact on the model's prediction and in which direction.
    """
    shap.initjs()

    for goal in ['goal1', 'goal2', 'nogoal']:
        print(f"\n Generating SHAP analysis for {goal.upper()}")

        # Select best model for each goal
        if goal == 'goal1':
            best_model = results['HistGradientBoosting']['goal1']['model']
            best_name = 'HistGradientBoosting'
                
        elif goal == 'goal2':
            best_model = results['RandomForest']['goal2']['model']
            best_name = 'RandomForest'
                
        elif goal == 'nogoal':
            best_model = results['DecisionTree']['nogoal']['model']
            best_name = 'DecisionTree'

        print(f"Using model: {best_name}")

        # Get preprocessor and classifier
        preprocessor = best_model.named_steps['preprocessor']
        clf = best_model.named_steps['classifier']

        # Transform data
        X_train_proc = preprocessor.transform(X_train)
        X_test_proc = preprocessor.transform(X_test)

        # Get transformed feature names
        try:
            feature_names = preprocessor.get_feature_names_out()
        except AttributeError:
            feature_names = []
            for name, trans, cols in preprocessor.transformers_:
                if hasattr(trans, 'get_feature_names_out'):
                    feature_names.extend(trans.get_feature_names_out(cols))
                else:
                    feature_names.extend(cols)

        n_features = len(feature_names)

        # Use appropriate SHAP explainer based on model type
        if isinstance(clf, (RandomForestClassifier, DecisionTreeClassifier, HistGradientBoostingClassifier)):
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_test_proc)

            # Handle output shapes
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For multi-class
            elif shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]

        # Plot SHAP beeswarm
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_test_proc,
            feature_names=feature_names,
            plot_type="dot",
            show=False,
            max_display=n_features
        )
        plt.title(f"SHAP Beeswarm for {best_name} ({goal.upper()})")
        plt.tight_layout()

        # Save plot
        fname = f"shap_beeswarm_{best_name}_{goal}.png"
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        print(f" SHAP plot saved as: {fname}")
        plt.close()

        # Optional: print top SHAP impacts
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        print("\nTop Features by Average SHAP Value:")
        for name, val in sorted(zip(feature_names, mean_abs_shap), key=lambda x: -x[1]):
            print(f"{name:40s}: {val:.5f}")

# === Generate SHAP plots ===
generate_shap_beeswarm(results, X_train, X_test)

print("\n All explainability plots successfully generated!")
print('\a')  # Beep again to signal full run


# === Appendix: confusion matrix for validation and test set ===
for model_name, goals in results.items():
    for goal_name, goal_data in goals.items():
        print(f"\nModel: {model_name} | Goal: {goal_name.upper()}")
        print("Validation Confusion Matrix:")
        print(pd.DataFrame(goal_data["confusion_val"], 
                           index=["Actual 0", "Actual 1"], 
                           columns=["Pred 0", "Pred 1"]))
        
        # Reshape the confusion_test if it's a flat array
        test_conf = np.array(goal_data["confusion_test"]).reshape(2, 2)
        print("Test Confusion Matrix:")
        print(pd.DataFrame(test_conf, 
                           index=["Actual 0", "Actual 1"], 
                           columns=["Pred 0", "Pred 1"]))

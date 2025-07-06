# =============================
# STEP 1: Initial Data Assessment
# =============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegressionCV

# Load raw dataset
df = pd.read_csv("data.csv")

# Count missing values per row
df['missing_count'] = df.isnull().sum(axis=1)

# Drop rows with > 5 missing values
df_cleaned = df[df['missing_count'] <= 5].copy()
df_cleaned.drop(columns=['missing_count'], inplace=True)

# Summary statistics
missing_summary = df['missing_count'].value_counts().sort_index()
missing_summary_pct = (missing_summary / len(df)) * 100

missing_summary_df = pd.DataFrame({
    'Num Missing': missing_summary.index,
    'Row Count': missing_summary.values,
    'Percentage': missing_summary_pct.values.round(2)
})
missing_summary_df = missing_summary_df.sort_values(by='Num Missing', ascending=False)
missing_summary_df['Cumulative Count'] = missing_summary_df['Row Count'].cumsum()
missing_summary_df['Cumulative Percentage'] = missing_summary_df['Percentage'].cumsum()

print(missing_summary_df)

# =============================
# STEP 2: Feature Correlation & Engineering
# =============================

df_cleaned['LoanValueRatio'] = (df_cleaned['LOAN'] + df_cleaned['MORTDUE']) / df_cleaned['VALUE']
print(f"Missing LoanValueRatio: {df_cleaned['LoanValueRatio'].isna().sum()}")
df_cleaned = df_cleaned.drop(columns=['LOAN', 'MORTDUE', 'VALUE'])

# =============================
# STEP 3: Missing Value Informativeness
# =============================

excluded_cols = ['BAD']
features = [col for col in df_cleaned.columns if col not in excluded_cols]
results = []

for col in features:
    missing_flag = df_cleaned[col].isnull().astype(int)
    contingency_table = pd.crosstab(missing_flag, df['BAD'])
    if contingency_table.shape == (2, 2):
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        results.append({
            'Feature': col,
            'Chi-Square Statistic': chi2,
            'p-value': p_value,
            'Missingness Informative?': 'Yes' if p_value < 0.05 else 'No'
        })

result_df = pd.DataFrame(results).sort_values('p-value')
print(result_df)

# =============================
# STEP 4: Handling Missing Values: Imputation Flags
# =============================

exclude_columns = ['REASON', 'CLNO', 'CLAGE']
for col in df_cleaned.columns:
    if col not in exclude_columns and df_cleaned[col].isnull().any():
        df_cleaned[f'{col}_missing'] = df_cleaned[col].isnull().astype(int)

df_cleaned.to_excel("cleaned_data_with_missing_flags.xlsx", index=False)
df_cleaned.to_csv("cleaned_data_imputed.csv", index=False)

# =============================
# STEP 5: Outlier Handling & Export
# =============================

df = pd.read_csv("cleaned_data_imputed.csv")
df_cleaned_no_missing_flags = df.drop(columns=[col for col in df.columns if col.endswith('_missing')])
df_cleaned_no_missing_flags.to_csv("cleaned_data_no_missing_flags.csv", index=False)

# =============================
# STEP 6: Data Splitting Strategy
# =============================

df = pd.read_csv("cleaned_data_no_missing_flags.csv")
X = df.drop(columns=["BAD"])
y = df["BAD"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# =============================
# STEP 7: Feature Engineering & Ranking
# =============================

reason_col = ['REASON']
job_col = ['JOB']
num_highrisk_impute = ['DEBTINC', 'LoanValueRatio']
num_mode_impute = ['DEROG', 'DELINQ', 'NINQ']
num_median_impute = ['CLAGE', 'CLNO', 'YOJ']
exclude_cols = ['REASON', 'JOB', 'BAD']
numeric_cols = [col for col in df.columns if col not in exclude_cols]
remaining_numeric = list(set(numeric_cols) - set(num_highrisk_impute) - set(num_mode_impute) - set(num_median_impute))

reason_pipeline = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Not Provided')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])
job_pipeline = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='Other')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])
highrisk_pipeline = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value=9999))])
mode_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent'))])
median_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median'))])
numeric_pipeline = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value=0)), ('scaler', RobustScaler())])

preprocessor = ColumnTransformer([
    ('reason', reason_pipeline, reason_col),
    ('job', job_pipeline, job_col),
    ('highrisk', highrisk_pipeline, num_highrisk_impute),
    ('mode', mode_pipeline, num_mode_impute),
    ('median', median_pipeline, num_median_impute),
    ('num', numeric_pipeline, remaining_numeric)
])

X_full = pd.concat([X_train, X_test])
y_full = pd.concat([y_train, y_test]).reset_index(drop=True)

preprocessor.fit(X_full)
X_processed_full = preprocessor.transform(X_full)

reason_features = preprocessor.named_transformers_['reason'].named_steps['encoder'].get_feature_names_out(['REASON'])
job_features = preprocessor.named_transformers_['job'].named_steps['encoder'].get_feature_names_out(['JOB'])
all_features = np.concatenate([numeric_cols, reason_features, job_features])
X_df = pd.DataFrame(X_processed_full, columns=all_features)

mi_scores = mutual_info_classif(X_df, y_full)
mi_series = pd.Series(mi_scores, index=X_df.columns)

log_l1 = LogisticRegressionCV(penalty='l1', solver='liblinear', cv=5, random_state=42, scoring='roc_auc', max_iter=1000)
log_l1.fit(X_df, y_full)
log_l1_importance = pd.Series(np.abs(log_l1.coef_).flatten(), index=X_df.columns)

ranking_df = pd.DataFrame({
    "MI Importance": mi_series,
    "MI Rank": mi_series.rank(ascending=False),
    "L1 Logistic Importance": log_l1_importance,
    "L1 Logistic Rank": log_l1_importance.rank(ascending=False)
})
ranking_df["Average Rank"] = ranking_df[["MI Rank", "L1 Logistic Rank"]].mean(axis=1)
ranking_df = ranking_df.sort_values("Average Rank")

ranking_df.to_csv("final_feature_ranking_detailed.csv")

# =============================
# STEP 8: Visualizations
# =============================

# Histograms by BAD group
custom_palette = {0: '#377eb8', 1: '#e41a1c'}
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x=col, hue='BAD', bins=30, kde=True, stat="density", common_norm=False, palette=custom_palette)
    plt.title(f'Distribution of {col} by BAD')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.tight_layout()
    plt.show()

# Categorical count plots
categorical_cols = ['REASON', 'JOB']
for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col, hue='BAD')
    plt.title(f'{col} vs BAD')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Top 20 features by MI & L1
top_mi = ranking_df.sort_values("MI Importance", ascending=True).tail(20)
top_l1 = ranking_df.sort_values("L1 Logistic Importance", ascending=True).tail(20)

plt.figure(figsize=(12, 6))
plt.barh(top_mi.index, top_mi["MI Importance"])
plt.xlabel("Mutual Information Importance")
plt.title("Top Features by Mutual Information")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.barh(top_l1.index, top_l1["L1 Logistic Importance"])
plt.xlabel("L1 Logistic Regression Importance")
plt.title("Top Features by Logistic Regression (L1)")
plt.tight_layout()
plt.show()

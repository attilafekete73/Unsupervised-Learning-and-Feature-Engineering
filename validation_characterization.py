import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load & Preprocess (Re-running previous state)
try:
    df = pd.read_csv('mental-heath-in-tech-2016_20161114.csv')
except FileNotFoundError:
    print("Error: CSV file not found.")
    exit()

def clean_gender(gender):
    if isinstance(gender, str):
        gender = gender.lower().strip()
        if gender in ['male', 'm', 'man', 'cis male', 'mail', 'male ']:
            return 'Male'
        elif gender in ['female', 'f', 'woman', 'femake', 'cis female']:
            return 'Female'
        else:
            return 'Other'
    return 'Other'

if 'What is your gender?' in df.columns:
    df['What is your gender?'] = df['What is your gender?'].apply(clean_gender)

cols_to_drop = ['Why or why not?', 'Why or why not?.1', 'What US state or territory do you live in?', 
                'What US state or territory do you work in?', 'What country do you work in?']
df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

cat_cols = df_clean.select_dtypes(include=['object']).columns
num_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
    ])

X_processed = preprocessor.fit_transform(df_clean)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_pca)

# 2. Deep Dive Characterization
# We want to see how demographic/work factors distribute across clusters
characterization_cols = [
    'Are you self-employed?',
    'Do you have a family history of mental illness?',
    'Do you work remotely?',
    'Is your employer primarily a tech company/organization?',
    'What is your age?',
    'What is your gender?'
]

# Filter for cols that actually exist
valid_char_cols = [c for c in characterization_cols if c in df.columns]

print("--- Cluster Characterization (Demographics & Work Context) ---")
for col in valid_char_cols:
    print(f"\nAnalysis of: {col}")
    if df[col].dtype == 'object':
        # Categorical: Show distribution
        print(df.groupby('Cluster')[col].value_counts(normalize=True).unstack().fillna(0))
    else:
        # Numerical: Show mean/median
        print(df.groupby('Cluster')[col].describe()[['mean', '50%', 'std']])

# 3. Check for "Missingness" in Cluster 2
# Let's see if Cluster 2 is defined by having NaNs in the original data
# We'll check the % of missing values per cluster for a key question
key_q = 'Does your employer provide mental health benefits as part of healthcare coverage?'
print(f"\nMissing Value Analysis for question: '{key_q}'")
missing_analysis = df.groupby('Cluster')[key_q].apply(lambda x: x.isnull().mean())
print(missing_analysis)

# Filter for Cluster 0
df_c0 = df[df['Cluster'] == 0].copy()
print(f"Focusing on Cluster 0. Sample size: {len(df_c0)}")

# Re-assess columns for this specific subgroup
# We drop columns that might have become constant (zero variance) in this subset
# e.g., if everyone in Cluster 0 said "Yes" to something, it's no longer a useful feature for sub-clustering
nunique = df_c0.nunique()
cols_to_drop_sub = nunique[nunique == 1].index.tolist()
df_c0_clean = df_c0.drop(columns=cols_to_drop_sub)
print(f"Dropped constant columns for this subgroup: {cols_to_drop_sub}")

# Drop the previous 'Cluster' column and the raw text columns dropped previously
cols_to_drop_general = ['Why or why not?', 'Why or why not?.1', 'What US state or territory do you live in?',
                        'What US state or territory do you work in?', 'What country do you work in?', 'Cluster']
# Only drop if they exist
df_c0_clean = df_c0_clean.drop(columns=[c for c in cols_to_drop_general if c in df_c0_clean.columns])


# Identify types again for the subset
cat_cols_c0 = df_c0_clean.select_dtypes(include=['object']).columns
num_cols_c0 = df_c0_clean.select_dtypes(include=['int64', 'float64']).columns

# Pipeline for the subset
preprocessor_c0 = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols_c0),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols_c0)
    ])

# Preprocess
X_c0_processed = preprocessor_c0.fit_transform(df_c0_clean)

# PCA
pca_c0 = PCA(n_components=2)
X_c0_pca = pca_c0.fit_transform(X_c0_processed)

# Elbow Method for Sub-clustering
inertia_c0 = []
k_range = range(1, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_c0_pca)
    inertia_c0.append(kmeans.inertia_)

# Plot Elbow
plt.figure(figsize=(8, 4))
plt.plot(k_range, inertia_c0, marker='o')
plt.title('Elbow Method for Cluster 0 Sub-groups')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Apply K-Means (Assuming k=3 for sub-segmentation)
k_sub = 3
kmeans_c0 = KMeans(n_clusters=k_sub, random_state=42, n_init=10)
sub_clusters = kmeans_c0.fit_predict(X_c0_pca)
df_c0['SubCluster'] = sub_clusters

# Visualization of Sub-clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_c0_pca[:, 0], y=X_c0_pca[:, 1], hue=sub_clusters, palette=['#ff9999', '#66b3ff', '#99ff99'], s=50)
plt.title('Sub-segmentation of "The Vulnerable & Supported" (Cluster 0)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(title='Sub-Cluster')
plt.show()

# Characterize the Sub-clusters
# We look for differences in trust and perception of employer support
deep_dive_qs = [
    'Do you feel that your employer takes mental health as seriously as physical health?',
    'Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?',
    'Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?',
    'Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment resources provided by your employer?',
    'Gender' # Let's check gender again if available (it was cleaned into 'What is your gender?')
]
# Fix gender column name if needed
if 'What is your gender?' in df_c0.columns:
    deep_dive_qs[-1] = 'What is your gender?'

print("\n--- Sub-Cluster Characterization (Cluster 0 Only) ---")
for col in deep_dive_qs:
    if col in df_c0.columns:
        print(f"\nAnalysis of: {col}")
        print(df_c0.groupby('SubCluster')[col].value_counts(normalize=True).unstack().fillna(0))
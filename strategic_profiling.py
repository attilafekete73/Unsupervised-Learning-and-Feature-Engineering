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

# 1. Load Data
try:
    df = pd.read_csv('mental-heath-in-tech-2016_20161114.csv')
except FileNotFoundError:
    # Fallback if file not found (should not happen based on context)
    pass

# 2. Data Cleaning & Preprocessing
# Clean Gender
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

# Drop columns
cols_to_drop = ['Why or why not?', 'Why or why not?.1', 'What US state or territory do you live in?',
                'What US state or territory do you work in?', 'What country do you work in?']
df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Feature Engineering Pipeline
cat_cols = df_clean.select_dtypes(include=['object']).columns
num_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
    ])

X_processed = preprocessor.fit_transform(df_clean)

# 3. Global Clustering
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_pca)

# 4. Sub-Clustering (Cluster 0)
df_c0 = df[df['Cluster'] == 0].copy()
nunique = df_c0.nunique()
cols_to_drop_sub = nunique[nunique == 1].index.tolist()
df_c0_clean = df_c0.drop(columns=cols_to_drop_sub + ['Cluster'])

cat_cols_c0 = df_c0_clean.select_dtypes(include=['object']).columns
num_cols_c0 = df_c0_clean.select_dtypes(include=['int64', 'float64']).columns

preprocessor_c0 = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols_c0),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols_c0)
    ])

X_c0_processed = preprocessor_c0.fit_transform(df_c0_clean)
pca_c0 = PCA(n_components=2)
X_c0_pca = pca_c0.fit_transform(X_c0_processed)
kmeans_c0 = KMeans(n_clusters=3, random_state=42, n_init=10)
df_c0['SubCluster'] = kmeans_c0.fit_predict(X_c0_pca)

# 5. Generate Plots for Report

# Plot A: Global Segmentation Pie Chart
plt.figure(figsize=(8, 8))
cluster_counts = df['Cluster'].value_counts().sort_index()
# Cluster 2 is Contractors (100% self-employed). Cluster 0 is Aware. Cluster 1 is Unaware.
labels_global = ['Cluster 0: The Aware (Target)', 'Cluster 1: The Unaware', 'Cluster 2: Contractors']
plt.pie(cluster_counts, labels=labels_global, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#99ff99', '#ffcc99'])
plt.title('Global Employee Segmentation')
plt.savefig('global_segmentation.png')
plt.show()

# Plot B: Target Group (Cluster 0) Breakdown Pie Chart
plt.figure(figsize=(8, 8))
sub_counts = df_c0['SubCluster'].value_counts().sort_index()
# Labels based on previous analysis: 0=Safe, 1=Skeptics, 2=Uncertain
labels_sub = ['Sub 0: The Safe & Supported', 'Sub 1: The Fearful Skeptics', 'Sub 2: The Uncertain Middle']
plt.pie(sub_counts, labels=labels_sub, autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff', '#99ff99'])
plt.title('Breakdown of Target Group (Cluster 0)')
plt.savefig('target_breakdown.png')
plt.show()

# Plot C: Trust Gap Bar Chart
q_trust = 'Do you feel that your employer takes mental health as seriously as physical health?'
if q_trust in df_c0.columns:
    plt.figure(figsize=(10, 6))
    trust_dist = df_c0.groupby('SubCluster')[q_trust].value_counts(normalize=True).unstack().fillna(0)
    # Plotting "Yes" responses
    sns.barplot(x=labels_sub, y=trust_dist.get('Yes', 0), palette='viridis')
    plt.title('Perception of Employer Support by Sub-Group')
    plt.ylabel('Percentage responding "Yes"')
    plt.ylim(0, 1)
    plt.savefig('trust_gap.png')
    plt.show()
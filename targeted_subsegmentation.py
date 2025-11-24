import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

try:
    df = pd.read_csv('mental-heath-in-tech-2016_20161114.csv')
except FileNotFoundError:
    pass

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


target_mask = df_c0['SubCluster'].isin([1, 2])
df_c0['GroupType'] = np.where(target_mask, 'Target (Risk)', 'Safe')

profile_cols = [
    'How many employees does your company or organization have?',
    'Is your employer primarily a tech company/organization?',
    'Do you work remotely?',
    'What is your gender?',
    'What is your age?',
    'Do you have a family history of mental illness?'
]

valid_profile_cols = [c for c in profile_cols if c in df_c0.columns]

print(f"Total size of Cluster 0: {len(df_c0)}")
print(f"Size of Target Group (Risk): {target_mask.sum()} ({target_mask.mean():.1%})")
print(f"Size of Safe Group: {(~target_mask).sum()} ({(~target_mask).mean():.1%})")

print("\n--- Profile of the Target Group (The 'Quick Win' Candidates) ---")
for col in valid_profile_cols:
    print(f"\nFeature: {col}")
    if df_c0[col].dtype == 'object':
        print(df_c0[df_c0['GroupType'] == 'Target (Risk)'][col].value_counts(normalize=True).head(3))
    else:
        print(df_c0[df_c0['GroupType'] == 'Target (Risk)'][col].describe()[['mean', 'std']])

print("\n--- Comparing Company Size Distribution (Target vs Safe) ---")
print(df_c0.groupby('GroupType')['How many employees does your company or organization have?'].value_counts(normalize=True).unstack().fillna(0))
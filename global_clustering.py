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

try:
    df = pd.read_csv('mental-heath-in-tech-2016_20161114.csv')
except FileNotFoundError:
    print("Error: CSV file not found. Please ensure the file name is correct.")
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
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')), 
            ('scaler', StandardScaler())
        ]), num_cols),
        
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols)
    ])

print("Preprocessing data...")
X_processed = preprocessor.fit_transform(df_clean)

print("Running PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

print("Calculating Elbow Method...")
inertia = []
k_range = range(1, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method to find optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

#k based on Elbow plot
k_optimal = 3 
print(f"Applying k-Means with k={k_optimal}...")
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)

df['Cluster'] = clusters

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette="coolwarm", s=50)
plt.title('Mental Health Survey Clusters (PCA Reduced)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(title='Cluster')
plt.show()

print("\n--- Cluster Interpretation (Top responses per group) ---")
target_questions = [
    'Does your employer provide mental health benefits as part of healthcare coverage?',
    'Do you think that discussing a mental health disorder with your employer would have negative consequences?',
    'Would you feel comfortable discussing a mental health disorder with your coworkers?'
]

valid_questions = [q for q in target_questions if q in df.columns]

for i in range(k_optimal):
    print(f"\n--- Cluster {i} Analysis ---")
    
    cluster_data = df[df['Cluster'] == i][valid_questions]
    
    modes = cluster_data.mode()
    
    if not modes.empty:
        print(modes.iloc[0])
    else:
        print("Result: mostly missing values (Non-responsive group)")
        print(f"Sample size: {len(cluster_data)}")
        print(f"Missing values in this group:\n{cluster_data.isnull().sum()}")
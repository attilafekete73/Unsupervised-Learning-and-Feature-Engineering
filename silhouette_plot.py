import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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

# based on elbow plot
k_optimal = 3 
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)
df['Cluster'] = clusters

fig, ax1 = plt.subplots(1, 1)
fig.set_size_inches(8, 6)

ax1.set_xlim([-0.1, 1])
ax1.set_ylim([0, len(X_pca) + (k_optimal + 1) * 10])

silhouette_avg = silhouette_score(X_pca, clusters)
sample_silhouette_values = silhouette_samples(X_pca, clusters)

y_lower = 10
for i in range(k_optimal):
    ith_cluster_silhouette_values = sample_silhouette_values[clusters == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.nipy_spectral(float(i) / k_optimal)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)

    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

    y_lower = y_upper + 10 

ax1.set_title("Silhouette Plot for the Mental Health Survey Clusters")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster label")

ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
ax1.set_yticks([]) 
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

print(f"Average Silhouette Score: {silhouette_avg:.2f}")
plt.show()

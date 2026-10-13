# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Create dataset

# Convert data into DataFrame
df=pd.read_csv("student-scores.csv", nrows=50)

print(df.head())

print(df.describe())

# Create K-Means model
kmeans = KMeans(n_clusters=2, random_state=0)

# Use only numeric subject-score columns
features = df.filter(regex="_score$")
kmeans.fit(features)

# Add cluster number to the original data
df["Cluster"] = kmeans.labels_

print("\nClustered Data:")
print(df)

# Display centroids
print("\nCentroids:")
print(kmeans.cluster_centers_)

# Plot the clusters
plt.scatter(df['chemistry_score'], df['physics_score'], c=df['Cluster'])

# Plot centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker='X',
    s=200
)

plt.xlabel("Chemistry Score")
plt.ylabel("Physics Score")
plt.title("K-Means Clustering")
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Reload the data
data = pd.read_csv('Data/Spotify_Youtube.csv')

# Select and scale the three columns
features = data[['Liveness', 'Energy', 'Loudness']].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Elbow Method to determine optimal K
inertia = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow graph
plt.figure(figsize=(8, 4))
plt.plot(k_values, inertia, marker='o')
plt.title("Elbow Method to Determine Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (SSE)")
plt.grid(True)
plt.show()

# Prepare the 3D data list
x_3d = X_scaled.tolist()

# Choose number of clusters
num_C = 3
num_inits = 10
num_max_iter = 300
km = KMeans(n_clusters=num_C, n_init=num_inits, max_iter=num_max_iter, random_state=42)

y_km = km.fit_predict(x_3d)
c_centers = km.cluster_centers_

# Organize data points by clusters
k_clusters = {}
for i in range(num_C):
    k_clusters[str(i)] = [[], [], []]

for i in range(num_C):
    for j in range(len(y_km)):
        if y_km[j] == i:
            n_x, n_y, n_z = x_3d[j]
            k_clusters[str(i)][0].append(n_x)
            k_clusters[str(i)][1].append(n_y)
            k_clusters[str(i)][2].append(n_z)

# Plot 3D K-means clusters
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')
colors = ['red', 'blue', 'green']

for idx, i in enumerate(k_clusters):
    x, y, z = k_clusters[i]
    ax.scatter(x, y, z, label=f'Cluster {i}', alpha=0.6)

# Plot cluster centers
for i in c_centers:
    x, y, z = i
    ax.scatter(x, y, z, marker='*', s=200, c='black', label='Centroid')

ax.set_title("3D K-Means Clustering on Spotify/YouTube Data")
ax.set_xlabel("Liveness (scaled)")
ax.set_ylabel("Energy (scaled)")
ax.set_zlabel("Loudness (scaled)")
ax.legend()
plt.show()

#Hierarchical Clustering
linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(10, 6))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()

hier_clusters = fcluster(linked, t=3, criterion='maxclust')

# Plot 3D hierarchical clustering
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

for i in range(1, 4):
    cluster_points = X_scaled[hier_clusters == i]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f"Cluster {i}", alpha=0.6)

ax.set_title("3D Hierarchical Clustering on Spotify/YouTube Data")
ax.set_xlabel("Liveness (scaled)")
ax.set_ylabel("Energy (scaled)")
ax.set_zlabel("Loudness (scaled)")
ax.legend()
plt.show()
#Clustering [Agrupamento]

#Geração de dados com MakeBlobs (Gaussian Blobs)
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

#Plotagem
import matplotlib.pyplot as plt
plt.scatter(X[:,0], X[:,1], s=50)

#Descobrindo os agrupamentos
from sklearn.cluster import KMeans
import numpy as np
kmeans = KMeans(n_clusters=4).fit(X)
y_kmean = kmeans.predict(X)

#Plotando o Kmeans
plt.scatter(X[:,0], X[:,1], c=y_kmean, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], s=200, alpha=0.5, c='black');

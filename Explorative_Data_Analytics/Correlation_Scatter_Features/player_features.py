import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import preprocessing

data = pd.read_csv('player_data2.csv')
data.head()

scaledData = preprocessing.MinMaxScaler().fit_transform(data)
scaledData = pd.DataFrame(scaledData, columns = data.columns)
scaledData.head ()

image, axes = plt.subplots (figsize = (8,6))
axes.scatter (scaledData.feature_1, scaledData.feature_2, s = 50)
axes.set_title ('PLAYER FEATURES', fontsize = 20)
axes.set_ylabel ('Feature 1', fontsize = 20)
axes.set_xlabel ('Feature 2', fontsize = 20)

kmeans = KMeans (n_clusters = 6)
kmeans.fit (scaledData)
cluster_centers = kmeans.cluster_centers_

image, axes = plt.subplots (figsize = (8,6))

axes.scatter (scaledData.feature_1, scaledData.feature_2, s = 50, c = kmeans.labels_, cmap = "brg")
axes.scatter (cluster_centers[:, 0], cluster_centers[:, 1], c = 'black', s = 300, alpha = 0.5, marker = '*')
axes.set_title ('PLAYER FEATURES', fontsize = 20)
axes.set_ylabel ('Property 1', fontsize = 20)
axes.set_xlabel ('Feature 2', fontsize = 20)
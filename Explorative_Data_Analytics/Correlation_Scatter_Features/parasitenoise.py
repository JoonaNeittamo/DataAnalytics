import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

data = pd.read_csv('loisia.csv')

image, axes = plt.subplots (figsize = (8,6))
axes.scatter(data.y, data.x, s = 0.015)
axes.set_title('Parasite with noise', fontsize = 20)

dbscan = DBSCAN(eps=50, min_samples=110).fit(data)
dbscan.fit(data)

image,axes = plt.subplots(figsize = (10,13))
axes.scatter(data.y, data.x, c = dbscan.labels_, s = 0.23)
axes.set_title("Parasite clusterings", fontsize = 13)
# Clustering Analysis

This repository contains code for clustering analysis using DBSCAN and KMeans algorithms.

## parasitenoise.py

This script performs clustering on parasite data with noise using the DBSCAN algorithm.

### Key Steps:
1. **Data Loading**: Reads the dataset from a CSV file (`loisia.csv`).
2. **Visualization**: Plots the initial scatter plot of the data.
3. **DBSCAN Clustering**: Applies the DBSCAN algorithm to the data.
4. **Cluster Visualization**: Plots the clustered data.

## player_features.py

This script performs clustering on player features using the KMeans algorithm.

### Key Steps:
1. **Data Loading**: Reads the dataset from a CSV file (`player_data2.csv`).
2. **Data Scaling**: Scales the features using Min-Max scaling.
3. **Visualization**: Plots the scatter plot of the scaled data.
4. **KMeans Clustering**: Applies the KMeans algorithm to the scaled data.
5. **Cluster Visualization**: Plots the clustered data along with cluster centers.

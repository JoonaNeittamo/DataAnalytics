import pandas as pd
import matplotlib.pyplot as axis
import numpy as np
from sklearn.feature_selection import RFECV

data = pd.read_csv("Boston_housing.csv", header=0)

data.head(5)

predictors = data.drop('pollution_(nitrous_oxide)', axis=1)

target = data['median_price']

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=1)

rfecv = RFECV(estimator=model, step=1, cv=5, scoring='r2')
rfecv.fit(predictors, target)

print('Features: ', rfecv.n_features_)
print('Support: ', rfecv.support_)
print('Estimators: ', rfecv.estimator_.feature_importances_)
print('Rankings: ', rfecv.ranking_)

axis.figure(figsize=(16, 9))
axis.title('Recursive Feature Elimination with Cross-Validation(RFEC)', fontsize=18, fontweight='bold', pad=20)
axis.xlabel('Number of features selected', fontsize=14, labelpad=20)
axis.ylabel('% Correct Classification', fontsize=14, labelpad=20)
axis.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='blue', linewidth=3)

axis.show()
import pandas as pd
import sklearn
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import matplotlib.pyplot as axis
import pydotplus

data = pd.read_csv("Boston_housing_modified.csv", header=0)
predictors = data.drop(['median_price'],axis=1) # Features
target = data['median_price'] # Target variable

from sklearn.ensemble import RandomForestRegressor


model = RandomForestRegressor(n_estimators=100, random_state=1)

from sklearn.feature_selection import RFECV


rfecv = RFECV(estimator=model, step=1, cv=5, scoring='r2')
rfecv.fit(predictors, target)

axis.figure(figsize=(16, 9))

axis.title('Recursive Feature Elimination with Cross-Validation(RFEC)', fontsize=18, fontweight='bold', pad=20)
axis.xlabel('Number of features selected', fontsize=14, labelpad=20)
axis.ylabel('% Correct Classification', fontsize=14, labelpad=20)
axis.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='blue', linewidth=3)

axis.show()


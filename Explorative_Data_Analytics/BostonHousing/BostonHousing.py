import pandas as pd

data = pd.read_csv("Boston_housing_modified.csv", header=0)
data.head()

import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

data = pd.read_csv("Boston_housing_modified.csv", header=0)
predictors = data.drop(['median_price'],axis=1) # Features
target = data['median_price'] # Target variable


from sklearn.model_selection import train_test_split
predictors_teach, predictors_test, target_teach, target_test = sklearn.model_selection.train_test_split(predictors, target, test_size=0.3, random_state=1) # 70% training and 30% test
decision_tree = DecisionTreeRegressor(min_impurity_decrease=0.02,max_depth=4 ,min_samples_leaf=20)
decision_tree = decision_tree.fit(predictors_teach, target_teach)
dot_data = StringIO()

export_graphviz(decision_tree, out_file=dot_data,
                filled=True, rounded=True,impurity=False, proportion=True,precision=2,
                special_characters=True, feature_names = predictors.columns,class_names=['survived','died'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())

feature_importances = pd.DataFrame({'predictor': predictors.columns,
                   'importance': decision_tree.feature_importances_}).\
                    sort_values('importance', ascending = False)


feature_importances.head()

prediction = decision_tree.predict(predictors_test)

from sklearn.metrics import r2_score


accuracy = r2_score(target_test, prediction)


print("r2_score:",accuracy)
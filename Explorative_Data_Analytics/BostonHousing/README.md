# Boston Housing Data Analysis

## BostonHousing

This document demonstrates the use of a Decision Tree Regressor to predict median house prices based on various features of the Boston Housing dataset.

### Key Steps:
1. **Data Loading**: Load the modified Boston housing data from a CSV file.
2. **Feature Selection**: Select features and target variables from the dataset.
3. **Train-Test Split**: Split the data into training and testing sets (70% training, 30% testing).
4. **Model Training**: Train a Decision Tree Regressor with specific parameters (`min_impurity_decrease=0.02`, `max_depth=4`, `min_samples_leaf=20`).
5. **Visualization**: Export the trained decision tree to a graph.
6. **Feature Importance**: Calculate and display the importance of each feature.
7. **Prediction & Evaluation**: Predict the target variable for the test set and evaluate the model using the R² score.

## BostonHousing-G_Boosting

This document illustrates the application of Gradient Boosting on the Boston Housing dataset to improve prediction accuracy.

### Key Steps:
1. **Data Loading**: Load the modified Boston housing data from a CSV file.
2. **Feature Selection**: Identify and select features and target variables.
3. **Model Initialization**: Initialize the Gradient Boosting Regressor.
4. **Model Training**: Fit the model to the dataset.
5. **Evaluation**: Evaluate the model performance using appropriate metrics.

## Feature_Elim_WCross-Validation

This document shows how to perform Recursive Feature Elimination with Cross-Validation (RFECV) using a Random Forest Regressor on the Boston Housing dataset.

### Key Steps:
1. **Data Loading**: Load the Boston housing data from a CSV file.
2. **Feature Selection**: Drop the 'pollution_(nitrous_oxide)' feature and select the target variable.
3. **Model Initialization**: Initialize a Random Forest Regressor with 100 estimators.
4. **RFECV**: Apply RFECV to select the most important features.
5. **Results**: Display the number of selected features, feature support, feature importance, and rankings.
6. **Visualization**: Plot the RFECV results to show the number of features selected versus the cross-validated score.

---

For more details, refer to the respective PDF files, you can also view the code by opening the .py files.

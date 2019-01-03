import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
import math
#data preprocessing
train = pd.read_csv('train.csv',index_col=0)
test = pd.read_csv('test.csv',index_col=0)
train_target = train['Sale_Price']
test_target = test['Sale_Price']
train = train.drop(columns=['Sale_Price'])
test = test.drop(columns=['Sale_Price'])

#dummy coding
categorical_features = [col for col in train.columns if train[col].dtypes =='object']
train = pd.get_dummies(train,columns = categorical_features)
test = pd.get_dummies(test,columns = categorical_features)
train_features,test_features = train.align(test,join = 'inner',axis=1)

#feature selection

# Extract feature names
feature_names = list(train_features.columns)
# Convert to np array
features_tree = np.array(train_features)
labels = np.array(train_target).reshape((-1, ))
# Empty array for feature importances
feature_importance_values = np.zeros(len(feature_names))
for i in range(50):
    model = lgb.LGBMRegressor(n_estimators=1000, learning_rate = 0.05, verbose = -1)
    # If training using early stopping need a validation set
    model.fit(features_tree, labels)

    # Record the feature importances
    feature_importance_values += model.feature_importances_ / 50

feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
# Sort features according to importance
feature_importances = feature_importances.sort_values('importance', ascending = False).reset_index(drop = True)
# Normalize the feature importances to add up to one
feature_importances['normalized_importance'] = feature_importances['importance'] / feature_importances['importance'].sum()
feature_importances['cumulative_importance'] = np.cumsum(feature_importances['normalized_importance'])
# Extract the features with zero importance
record_zero_importance = feature_importances[feature_importances['importance'] == 0.0]

#drop zero_importance features
train_features = train_features.drop(columns=list(record_zero_importance['feature']))
train_features,test_features = train_features.align(test_features,join = 'inner',axis=1)

# Replace the inf and -inf with nan (required for later imputation)
train_features = train_features.replace({np.inf: np.nan, -np.inf: np.nan})
test_features = test_features.replace({np.inf: np.nan, -np.inf: np.nan})

#impute the data
from sklearn.preprocessing import StandardScaler,Imputer
im = Imputer(strategy = 'median')
im.fit(train_features)
train_features = im.fit_transform(train_features)
test_features = im.fit_transform(test_features)


#scale the data
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(train_features)
# Apply transform to both the training set and the test set.
train_features = scaler.transform(train_features)
test_features = scaler.transform(test_features)

train_target = np.array(train_target).reshape((-1, ))
test_target = np.array(test_target).reshape((-1, ))

#hyperparameter training
# Loss function to be optimized
# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
loss = ['ls', 'lad', 'huber']

# Number of trees used in the boosting process
n_estimators = [100, 500, 900, 1100, 1500]

# Maximum depth of each tree
max_depth = [2, 3, 5, 10, 15]

# Minimum number of samples per leaf
min_samples_leaf = [1, 2, 4, 6, 8]

# Minimum number of samples to split a node
min_samples_split = [2, 4, 6, 10]

# Maximum number of features to consider for making splits
max_features = ['auto', 'sqrt', 'log2', None]

# Define the grid of hyperparameters to search
hyperparameter_grid = {'loss': loss,
                       'n_estimators': n_estimators,
                       'max_depth': max_depth,
                       'min_samples_leaf': min_samples_leaf,
                       'min_samples_split': min_samples_split,
                       'max_features': max_features
                       }


# Create the model to use for hyperparameter tuning
model = GradientBoostingRegressor(random_state = 42)


# Set up the random search with 4-fold cross validation
random_cv = RandomizedSearchCV(estimator=model,
                               param_distributions=hyperparameter_grid,
                               cv=4, n_iter=25,
                               scoring = 'neg_mean_absolute_error',
                               n_jobs = -1, verbose = 1,
                               return_train_score = True,
                               random_state=42)

# Fit on the training data
random_cv.fit(train_features,train_target)

random_cv.best_estimator_

# Create a range of trees to evaluate
trees_grid = {'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]}

model = GradientBoostingRegressor(loss = 'huber', max_depth = 3,
                                  min_samples_leaf = 2,
                                  min_samples_split = 4,
                                  max_features = 'auto',
                                  random_state = 42)

# Grid Search Object using the trees range and the random forest model
grid_search = GridSearchCV(estimator = model, param_grid=trees_grid, cv = 4,
                           scoring = 'neg_mean_absolute_error', verbose = 1,
                           n_jobs = -1, return_train_score = True)


# Fit the grid search
grid_search.fit(train_features,train_target)

# Select the best model
final_model = grid_search.best_estimator_
final_model.fit(train_features,train_target)
result = final_model.predict(test_features)



mysubmission1 = pd.DataFrame()
mysubmission1["PID"] = test['PID']
mysubmission1['Sale_Price'] = result
mysubmission1.to_csv("mysubmission1.txt",sep=",",index = False)

#select the second best model
second_model = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='huber', max_depth=3,
             max_features='auto', max_leaf_nodes=None,
             min_impurity_decrease=0.0, min_impurity_split=None,
             min_samples_leaf=2, min_samples_split=4,
             min_weight_fraction_leaf=0.0, n_estimators=700,
             presort='auto', random_state=42, subsample=1.0, verbose=0,
             warm_start=False)


second_model.fit(train_features,train_target)
result = second_model.predict(test_features)
mysubmission2 = pd.DataFrame()
mysubmission2["PID"] = test['PID']
mysubmission2['Sale_Price'] = result
mysubmission2.to_csv("mysubmission2.txt",sep=",",index = False)

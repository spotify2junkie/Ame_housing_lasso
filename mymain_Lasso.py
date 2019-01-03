

import random
random.seed(42)


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
# for sqrt
import math
from feature_selector import FeatureSelector

from sklearn.metrics import mean_squared_error as mse
import copy


#load datasets
test_id = list(np.arange(0, 2930,3))
data = pd.read_csv('Ames_data.csv')
train_id = list(set(np.arange(0, 2930))-set(test_id))

train = data.iloc[train_id,:]
test = data.iloc[test_id,:]

#extract train-test target variable
train_target = pd.DataFrame(np.log(train['Sale_Price']))
test_target = pd.DataFrame(np.log(test['Sale_Price']))

#drop the tatget variable from train-test
train = train.drop(columns=['Sale_Price','MS_Zoning','Street','Utilities','Land_Slope','Condition_2',
                            'Roof_Matl', 'Heating', 'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF',
                            'Three_season_porch', 'Pool_Area', 'Misc_Val', 'Longitude','Latitude',
                           'Alley','Bsmt_Cond','Central_Air','Electrical','Functional','Garage_Qual'])
test = test.drop(columns=['Sale_Price','MS_Zoning','Street','Utilities','Land_Slope','Condition_2',
                            'Roof_Matl', 'Heating', 'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF',
                            'Three_season_porch', 'Pool_Area', 'Misc_Val', 'Longitude','Latitude',
                         'Alley','Bsmt_Cond','Central_Air','Electrical','Functional','Garage_Qual'])


# In[3]:


train['int_liv_lot_area'] = train['Gr_Liv_Area']*train['Lot_Area']
train['int_liv_total_bsmt'] = train['Gr_Liv_Area']*train['Total_Bsmt_SF']
train['int_live_garage_liv'] = train['Gr_Liv_Area']*train['Garage_Yr_Blt']
train['int_first_area_garage'] = train['First_Flr_SF']*train['Garage_Area']

test['int_liv_lot_area'] = test['Gr_Liv_Area']*test['Lot_Area']
test['int_liv_total_bsmt'] = test['Gr_Liv_Area']*test['Total_Bsmt_SF']
test['int_live_garage_liv'] = test['Gr_Liv_Area']*test['Garage_Yr_Blt']
test['int_first_area_garage'] = test['First_Flr_SF']*test['Garage_Area']


# In[4]:


#dummy coding process
categorical_features = [col for col in train.columns if train[col].dtypes =='object']
train = pd.get_dummies(train,columns = categorical_features)
test = pd.get_dummies(test,columns = categorical_features)

#make sure train-test has same shape and columns
train_features,test_features = train.align(test,join = 'inner',axis=1)


# In[5]:


fs = FeatureSelector(data = train_features, labels = train_target)

fs.identify_all(selection_params = {'missing_threshold': 0.8, 'correlation_threshold': 0.8,
                                    'task': 'regression', 'eval_metric': 'l2',
                                     'cumulative_importance': 0.95})


def rmse(true,predicted):
    return math.sqrt(mse(true,predicted))




from sklearn.preprocessing import StandardScaler,Imputer,PolynomialFeatures

im = Imputer(strategy = 'median')
im.fit(train_features)

train_features_np = im.transform(train_features)
test_features_np = im.transform(test_features)

print(np.where(~np.isfinite(train_features_np)))
print(np.where(~np.isfinite(test_features_np)))

#scale the data
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(train_features_np)
# Apply transform to both the training set and the test set.
train_features_np = scaler.transform(train_features_np)
test_features_np = scaler.transform(test_features_np)




train_target = np.array(train_target).reshape((-1, 1))
train_target.shape

#scale the data
scalery = StandardScaler()
# Fit on training set only.
scalery.fit(train_target)

y_compute = scalery.transform(train_target)

y_compute = np.array(y_compute).reshape((-1, ))


y_compute.shape



def one_step_lasso(r, x, lam):
    xx = np.sum(np.square(x))
    xr = np.sum(np.dot(r,x))
    b = (np.abs(xr) -lam/2)/xx
    b = sign(xr)*ifelse(b>0, b, 0)
    return b



def sign(arg):
    if arg > 0:
        return 1
    elif arg == 0:
        return 0
    else:
        return -1

def ifelse(arg,a,b):
    if arg == True:
        return a
    else:
        return b


def mylasso(X,y,lam,n_iter = 50):
    """
    X: n-by-p design matrix,make sure it's scaled and centered
    y:n-by-1 response vector,make sure it's centered
    lam:lambda value
    n_iter: number of iterations
    """
    b = np.repeat(0.00000000, train_features_np.shape[1])
    r = y
    iteration = 0
    d = np.size(X, 1)
    while iteration < n_iter:
        for j in range(d):

            #update the residual vector
            r = r + np.dot(train_features_np[:, j],b[j])

            #apply one step lasso
            b[j] = one_step_lasso(r,X[:, j],lam)
            r = r - np.dot(X[:, j],one_step_lasso(r,X[:, j],lam))
        iteration += 1
    return b





coefficient1 = mylasso(train_features_np,y_compute,0.869749)
predicited1 = test_features_np.dot(coefficient1)
result1 = scalery.inverse_transform(predicited1)


mysubmission3 = pd.DataFrame()
mysubmission3["PID"] = test['PID']
mysubmission3['Sale_Price'] = np.exp(result1)
mysubmission3.to_csv("mysubmission3.txt",sep=",",index = False)

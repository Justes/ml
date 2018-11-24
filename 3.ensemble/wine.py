#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    data = pd.read_csv('wine.data', header=None)
    x, y = data.iloc[:, 1:], data[0]
    x = MinMaxScaler().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=1)

    lr = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), cv=3)
    lr.fit(x_train, y_train.ravel())
    print('alpha %.2f' % lr.alpha_)
    y_train_pred = lr.predict(x_train)
    y_test_pred = lr.predict(x_test)
    print('Logistic train acc ', accuracy_score(y_train, y_train_pred))
    print('Logistic test acc ', accuracy_score(y_test, y_test_pred))

    rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=5, oob_score=True)
    rf.fit(x_train, y_train.ravel())
    print('OOB Score=%.5f' % rf.oob_score_)
    y_train_pred = rf.predict(x_train)
    y_test_pred = rf.predict(x_test)
    print('rf train acc ', accuracy_score(y_train, y_train_pred))
    print('rf test acc ', accuracy_score(y_test, y_test_pred))

    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=2)
    gb.fit(x_train, y_train.ravel())
    y_train_pred = gb.predict(x_train)
    y_test_pred = gb.predict(x_test)
    print('GBDT train acc ', accuracy_score(y_train, y_train_pred))
    print('GBDT test acc ', accuracy_score(y_test, y_test_pred))

    y_train[y_train == 3] = 0
    y_test[y_test == 3] = 0
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_train, 'train'),(data_test, 'eval')]
    params = {'max_depth':1, 'eta':0.9, 'silent':1, 'objective':'multi:softmax', 'num_class':3}
    bst = xgb.train(params, data_train, num_boost_round=5, evals=watch_list)
    y_train_pred = bst.predict(data_train)
    y_test_pred = bst.predict(data_test)
    print('XGBoost train acc ', accuracy_score(y_train, y_train_pred))
    print('XGBoost test acc ', accuracy_score(y_test, y_test_pred))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    path = 'iris.data'
    data = pd.read_csv(path, header=None)
    x, y = data[list(range(4))], data[4]
    y = pd.Categorical(y).codes

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=50, random_state=1)
    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_train, 'train'), (data_test, 'eval')]
    # softmax 多分类，要设置 num_class 类别个数
    params = {'max_depth':4, 'eta':0.3, 'silent':1, 'objective':'multi:softmax', 'num_class':3}
    bst = xgb.train(params, data_train, num_boost_round=6, evals=watch_list)
    y_hat = bst.predict(data_test)
    result = y_test == y_hat
    print('acc: ', float(np.sum(result)) / len(y_hat))

    # Cs正则化参数的倒数, 若Cs的类型是 int 代表从 10^(-4),到 10^4 之间按等比数列取多少个数
    models = [('LogisticRegression', LogisticRegressionCV(Cs=10, cv=3)),
              ('RandomForest', RandomForestClassifier(n_estimators=30, criterion='gini'))
            ]
    for name, model in models:
        model.fit(x_train, y_train)
        print(name, 'train acc: ', accuracy_score(y_train, model.predict(x_train)))
        print(name, 'test acc: ', accuracy_score(y_test, model.predict(x_test)))


# In[ ]:





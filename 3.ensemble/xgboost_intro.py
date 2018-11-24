#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xgboost as xgb
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# 自定义一阶导 二阶导
def g_h(y_hat, y):
    p = 1.0 / (1.0 + np.exp(-y_hat))
    g = p - y.get_label()
    h = p * (1.0 - p)
    return g, h

def error_rate(y_hat, y):
    return 'error', float(sum(y.get_label() != (y_hat > 0.5))) / len(y_hat)

if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    data_train = xgb.DMatrix('agaricus_train.txt')
    data_test = xgb.DMatrix('agaricus_test.txt')

    # max_depth 树的最大深度
    # eta 学习率,即步长,更新中减少的步长来防止过拟合
    # silent (0,1) , 0 打印运行信息, 1 静默模式,不打印
    # objective 使用的基函数 binary:logistic 二分类逻辑回归
    param = {'max_depth':3, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
    watchlist = [(data_test, 'eval'), (data_train, 'train')]
    n_round = 7
    bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist, obj=g_h, feval=error_rate)
    y_hat = bst.predict(data_test)
    y = data_test.get_label()
    print(y_hat)
    print(y)
    error = sum(y != (y_hat > 0.5))
    error_rate = float(error) / len(y_hat)
    print('sample sum ', len(y_hat))
    print('error sum %4d' % error)
    print('error rate %.5f%%' % (100 * error_rate))


# In[ ]:





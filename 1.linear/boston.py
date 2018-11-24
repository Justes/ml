#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import pandas as pd
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

def not_empty(s):
    return s != ''

if __name__ == "__main__":
    warnings.filterwarnings(action='ignore')
    np.set_printoptions(suppress=True)
    file_data = pd.read_csv('housing.data', header=None)
    data = np.empty((len(file_data), 14))
    
    # 字符串转数字
    for i, d in enumerate(file_data.values):
        d = list(map(float, list(filter(not_empty, d[0].split(' ')))))
        data[i] = d
    x, y = np.split(data, (13,), axis=1)
    print('sample number: %d , feature number: %d' % x.shape)
    #print(y.shape)
    y = y.ravel()
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    model = Pipeline([
        ('ss', StandardScaler()),
        ('poly', PolynomialFeatures(degree=3, include_bias=True)),
        ('linear', ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.99, 1], alphas=np.logspace(-3, 2, 5), fit_intercept=False, max_iter=1e3, cv=3))
        ])

    model.fit(x_train, y_train)
    linear = model.get_params()['linear']
    print('alpha ', linear.alpha_)
    print('l1 ratio ', linear.l1_ratio_)
    #print('coefficient ', linear.coef_)

    order = y_test.argsort(axis=0)
    y_test = y_test[order]
    x_test = x_test[order]
    y_pred = model.predict(x_test)
    r2 = model.score(x_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    print('R2: ', r2)
    print('mean square: ', mse)

    t = np.arange(len(y_pred))
    mpl.rcParams['font.sans-serif'] = ['simhei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure()
    plt.plot(t, y_test, 'r-', lw=2, label='real')
    plt.plot(t, y_pred, 'g-', lw=2, label='pred')
    plt.legend(loc='best')
    plt.title('predict of boston houre price', fontsize=18)
    plt.xlabel('sample number', fontsize=15)
    plt.ylabel('houre price', fontsize=15)
    plt.grid()
    plt.show()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

def f(x):
    return 0.5 * np.exp(-(x+3) ** 2) + np.exp(-x ** 2) + 1.5 * np.exp(-(x - 3) ** 2)

if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = ['simhei']
    mpl.rcParams['axes.unicode_minus'] = False

    np.random.seed(0)
    N = 200
    x = np.random.rand(N) * 10 - 5
    x = np.sort(x)
    y = f(x) + 0.05 * np.random.randn(N)
    x.shape = -1, 1

    degree = 6
    n_estimators = 50
    max_samples = 0.5

    ridge = RidgeCV(alphas=np.logspace(-3, 2, 20), fit_intercept=False)
    ridged = Pipeline([('poly', PolynomialFeatures(degree=degree)),('Ridge', ridge)])

    # max_samples： int or float, optional (default=1.0)。决定从x_train抽取去训练 基估计器的样本数量。int 代表抽取数量，float代表抽取比例
    # max_features : int or float, optional (default=1.0)。决定从x_train抽取去训练 基估计器的特征数量。int 代表抽取数量，float代表抽取比例
    bagging_ridged = BaggingRegressor(ridged, n_estimators=n_estimators, max_samples=max_samples)

    dtr = DecisionTreeRegressor(max_depth=9)
    regs = [
            ('DecisionTree', dtr),
            ('Ridge(%d Degree)' % degree, ridged),
            ('Bagging Ridge(%d Degree)' % degree, bagging_ridged),
            ('Bagging DecisionTree', BaggingRegressor(dtr, n_estimators=n_estimators, max_samples=max_samples))
            ]

    plt.figure(figsize=(12, 9))
    plt.plot(x, y, 'ro', mec='k', label='train')
    plt.plot(x_test, f(x_test), color='k', lw=3, ls='-', label='real')
    clrs = '#FF2020', 'm', 'y', 'g'
    
    for i, (name, reg) in enumerate(regs):
        reg.fit(x, y)
        label = '%s, $R^2$=%.2f' % (name, reg.score(x, y))
        y_test = reg.predict(x_test.reshape(-1, 1))
        plt.plot(x_test, y_test, color=clrs[i], lw=(i+1)*0.5, label=label, zorder=6-i)
    
    plt.legend(loc='upper left')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('regressor fitting: samples_rate(%.1f), n_trees(%d)' % (max_samples, n_estimators), fontsize=15)
    plt.ylim(-0.2, 1.1*y.max())
    plt.grid()
    plt.show()


# In[ ]:





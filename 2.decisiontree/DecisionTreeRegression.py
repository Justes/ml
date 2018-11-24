#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor

if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = ['simhei']
    mpl.rcParams['axes.unicode_minus'] = False

    N = 100
    x = np.random.rand(N) * 6 - 3
    x.sort()
    y = np.sin(x) + np.random.randn(N) * 0.05
    x = x.reshape(-1, 1)

    dt = DecisionTreeRegressor(criterion='mse', max_depth=9)
    dt.fit(x, y)
    x_test = np.linspace(-3, 3, 50).reshape(-1, 1)
    y_hat = dt.predict(x_test)

    plt.figure()
    plt.plot(x, y, 'r*', markersize=10, markeredgecolor='k', label='real')
    plt.plot(x_test, y_hat, 'g-', lw=2, label='predict')
    plt.legend(loc='upper left', fontsize=12)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.title('decision tree regressor')
    plt.show()

    depth = [2, 4, 6, 8, 10]
    clr = 'rgbmy'
    dtr = DecisionTreeRegressor(criterion='mse')
    plt.figure()
    plt.plot(x, y, 'ro', ms=5, mec='k', label='real')
    x_test = np.linspace(-3, 3, 50).reshape(-1, 1)

    for d, c in zip(depth, clr):
        dtr.set_params(max_depth=d)
        dtr.fit(x, y)
        y_hat = dtr.predict(x_test)
        plt.plot(x_test, y_hat, '-', color=c, lw=2, markeredgecolor='k', label='Depth = %d' %d)
    plt.legend(loc='upper left')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.title('decision tree regression')
    plt.show()


# In[ ]:





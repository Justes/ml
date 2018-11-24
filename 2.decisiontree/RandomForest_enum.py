#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = ['simhei']
    mpl.rcParams['axes.unicode_minus'] = False

    iris_feature = 'sepal length', 'sepal width', 'petal length', 'petal width'
    path = 'iris.data'
    data = pd.read_csv(path, header=None)
    x_prime = data[list(range(4))]
    y = pd.Categorical(data[4]).codes
    x_prime_train, x_prime_test, y_train, y_test = train_test_split(x_prime, y, test_size=0.3, random_state=0)

    feature_pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]

    plt.figure(figsize=(12,9))
    for i, pair in enumerate(feature_pairs):
        x_train = x_prime_train[pair]
        x_test = x_prime_test[pair]
        
        model = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=5, oob_score=True)
        model.fit(x_train, y_train)

        N = 500
        x1_min, x2_min = x_train.min()
        x1_max, x2_max = x_train.max()
        t1 = np.linspace(x1_min, x1_max, N)
        t2 = np.linspace(x2_min, x2_max, N)
        x1, x2 = np.meshgrid(t1, t2)
        x_show = np.stack((x1.flat, x2.flat), axis=1)

        y_train_pred = model.predict(x_train)
        acc_train = accuracy_score(y_train, y_train_pred)
        y_test_pred = model.predict(x_test)
        acc_test = accuracy_score(y_test, y_test_pred)
        print('feature ', iris_feature[pair[0]], ' + ', iris_feature[pair[1]])
        print('OOB score ', model.oob_score_)
        print('train acc %.4f%%' % (100 * acc_train))
        print('test acc %.4f%%' % (100 * acc_test))

        cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
        cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
        y_hat = model.predict(x_show)
        y_hat = y_hat.reshape(x1.shape)
        plt.subplot(2, 3, i+1)
        # contour 绘制等高线, levels=[0, 1] 只画 0 和 1 的线 (根据 y_hat 的值)
        plt.contour(x1, x2, y_hat, colors='k', levels=[0, 1], antialiased=True, linestyles='--')
        plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)
        plt.scatter(x_train[pair[0]], x_train[pair[1]], 20, y_train, edgecolors='k', cmap=cm_dark, label='train')
        plt.scatter(x_test[pair[0]], x_test[pair[1]], 100, y_test, edgecolors='k', marker='*', cmap=cm_dark, label='test')
        plt.xlabel(iris_feature[pair[0]])
        plt.ylabel(iris_feature[pair[1]])
        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.grid()
    plt.suptitle('random forest')
    plt.show()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    iris_features = 'sepal length', 'sepal width', 'petal length', 'petal width'
    path = 'iris.data'
    data = pd.read_csv(path, header=None)
    x, y = data[[0, 1]], pd.Categorical(data[4]).codes
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=1)

    # C越大分类效果越好,但可能过拟合, 默认为1
    # kernel: linear 线性核, rbf 高斯核
    # 高斯核有个参数 gamma, gamma值越小,分类界面越连续;gamma值越大,分类界面越“散”,分类效果越好,但有可能会过拟合。
    # ovr: one v rest 一个类别与其他类别进行划分
    # ovo: one v one 类别两两之间划分,用二分类模拟多分类
    clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
    clf.fit(x_train, y_train.ravel())
    print(clf.score(x_train, y_train))
    print('train acc ', accuracy_score(y_train, clf.predict(x_train)))
    print(clf.score(x_test, y_test))
    print('test acc ', accuracy_score(y_test, clf.predict(x_test)))
    #print('decision_function: \n', clf.decision_function(x_train))
    print('\n predict: \n', clf.predict(x_train))

    x1_min, x2_min = x.min()
    x1_max, x2_max = x.max()
    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]
    grid_test = np.stack((x1.flat, x2.flat), axis=1)
    grid_hat = clf.predict(grid_test)
    grid_hat = grid_hat.reshape(x1.shape)

    mpl.rcParams['font.sans-serif'] = ['simhei']
    mpl.rcParams['axes.unicode_minus'] = False

    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    plt.figure()
    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
    plt.scatter(x[0], x[1], 50, y, edgecolors='k', cmap=cm_dark)
    plt.scatter(x_test[0], x_test[1], 120, facecolors='none', zorder=10)
    plt.xlabel(iris_features[0])
    plt.ylabel(iris_features[1])
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('iris svm')
    plt.grid(b=True, ls=':')
    plt.tight_layout(pad=1.5)
    plt.show()


# In[ ]:





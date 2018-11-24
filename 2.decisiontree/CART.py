#!/usr/bin/env python
# coding: utf-8

# In[63]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import pydotplus
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = ['simhei']
    mpl.rcParams['axes.unicode_minus'] = False

    iris_feature_E = 'sepal length', 'sepal width', 'petal length', 'petal width'
    iris_feature = '花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'
    iris_class = 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'

    path = 'iris.data'
    data = pd.read_csv(path, header=None)
    x = data[list(range(4))]
    y = pd.Categorical(data[4]).codes
    #x, y = np.split(data, (4,), axis=1)
    #y = LabelEncoder().fit_transform(data[4])
    x = x.iloc[:, :2]
    # x = x[[0, 1]]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(x_train, y_train)
    y_test_hat = model.predict(x_test)
    print('acc %.2f%%' % (100 * accuracy_score(y_test, y_test_hat)))

    """
    with open('iris.dot', 'w') as f:
        tree.export_graphviz(model, out_file=f)
    dot_data = tree.export_graphviz(model, out_file=None, feature_names=iris_feature_E[0:2], class_names=iris_class, filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf('iris.pdf')
    f = open('iris.png', 'wb')
    f.write(graph.create_png())
    f.close()
    """

    N, M = 50, 50
    x1_min, x2_min = x.min()
    x1_max, x2_max = x.max()
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)
    x_show = np.stack((x1.flat, x2.flat), axis=1)
    print(x_show.shape)

    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    y_show_hat = model.predict(x_show)
    print(y_show_hat.shape)
    y_show_hat = y_show_hat.reshape(x1.shape)
    
    plt.figure()
    plt.pcolormesh(x1, x2, y_show_hat, cmap=cm_light)
    # 测试集 zorder 制图顺序， 值小的先画
    plt.scatter(x_test[0], x_test[1], 30, c=y_test.ravel(), edgecolors='k', zorder=10, cmap=cm_dark)
    #plt.scatter(x_test[0], x_test[1], 20, c=y_test.ravel(), edgecolors='k', zorder=10, cmap=cm_dark, marker='*')
    # 全部数据
    plt.scatter(x[0], x[1], 10, c=y.ravel(), edgecolors='k', cmap=cm_dark)
    plt.xlabel(iris_feature[0], fontsize=13)
    plt.ylabel(iris_feature[1], fontsize=13)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('鸢尾花数据的决策树分类', fontsize=15)
    plt.grid(b=True, ls=':', color='#606060')
    plt.show()

    result = (y_test_hat == y_test)
    acc = np.mean(result)
    print('acc: %.2f%%' % (100 * acc))

    depth = np.arange(1, 15)
    err_list = []
    for d in depth:
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=d)
        clf.fit(x_train, y_train)
        y_test_hat = clf.predict(x_test)
        result = (y_test_hat == y_test)
        err = 1 - np.mean(result)
        err_list.append(err)
        print(d, ' error rate %.2f%%' % (100 * err))

    plt.figure()
    plt.plot(depth, err_list, 'ro-', markeredgecolor='k', lw=2)
    plt.xlabel('depth', fontsize=13)
    plt.ylabel('error rate', fontsize=13)
    plt.title('Decision Tree and Overfit')
    plt.grid(b=True, ls=':', color='#606060')
    plt.show()


# In[ ]:





# In[ ]:





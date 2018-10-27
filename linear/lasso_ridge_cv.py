#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso, Ridge

if __name__ == "__main__":
    path = 'Advertising.csv'
    data = pd.read_csv(path)
    x = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']

    mpl.rcParams['font.sans-serif'] = ['simhei']
    mpl.rcParams['axes.unicode_minus'] = False

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # lambda 超参数
    # np.logspace 创建等比数列 (第四个参数默认base=10)
    # 以10为底 从 -3 到 2 取 10个
    alpha_can = np.logspace(-3, 2, 10)
    # 禁止用科学计数法
    np.set_printoptions(suppress=True)
    print('alpha_can = ', alpha_can, '\n')

    models = {'Lasso' : Lasso(), 'Ridge' : Ridge()}
    for index, model in models.items():
        # lasso 加入 L1 正则项
        # model = Lasso()
        # Ridge 加入 L2 正则项
        # model = Ridge()

        print(index)
        # 利用 GridSearchCV 自动调参
        model = GridSearchCV(model, param_grid={'alpha' : alpha_can}, cv=5)
        model.fit(x_train, y_train)
        print('超参数: ', model.best_params_)

        order = y_test.argsort()
        x_test_v = x_test.values[order]
        y_test_v = y_test.values[order]
        y_hat = model.predict(x_test_v)
        print('R2 =',model.score(x_test_v, y_test_v))
        mse = np.average((y_hat - y_test_v) ** 2)
        rmse = np.sqrt(mse)
        print('mse =', mse, 'rmse =', rmse)

        t = np.arange(len(x_test_v))
        plt.figure()
        plt.plot(t, y_test_v, 'r-', lw=2, label='real')
        plt.plot(t, y_hat, 'g-', lw=2, label='pred')
        plt.title('real & predict', fontsize=18)
        plt.legend(loc='upper left')
        plt.grid(b=True, ls=':')
        plt.show()

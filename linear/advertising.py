#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    path = 'Advertising.csv'
    data = pd.read_csv(path)
    x = data[['TV','Radio','Newspaper']]
    y = data['Sales']

    mpl.rcParams['font.sans-serif'] = ['simhei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 图像背景
    plt.figure(facecolor='w')
    # r g m品红  o圆点 ^上三角形 v 下三角形
    plt.plot(data['TV'], y, 'ro', label='TV')
    plt.plot(data['Radio'], y, 'g^', label='Radio')
    plt.plot(data['Newspaper'], y, 'mv', label='Newspaper')
    # x轴表示  字体大小
    plt.xlabel('广告花费',fontsize=16)
    plt.ylabel('销售额',fontsize=16)
    plt.title('广告花费与销售额对比数据')
    # 图例位置 右下
    plt.legend(loc='lower right')
    # 显示网格 用虚线
    plt.grid(b=True, ls=':')
    plt.show()

    plt.figure(facecolor='w', figsize=(9, 10))
    # 将图像分成 3 * 1 ， 占用第1个
    # 第一个3 是分成几个子图 ，第二个 1 是 这个子图分成几块， 第三个1 是 占用子图第几块
    plt.subplot(311)
    plt.plot(data['TV'], y, 'ro')
    plt.title('TV')
    plt.grid(b=True, ls=':')

    plt.subplot(312)
    plt.plot(data['Radio'], y, 'g^')
    plt.title('Radio')
    plt.grid(b=True, ls=':')
    
    plt.subplot(313)
    plt.plot(data['Newspaper'], y, 'mv')
    plt.title('Newspaper')
    plt.grid(b=True, ls=':')

    plt.tight_layout()
    plt.show()
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    print(type(x_test))
    print(x_train.shape, y_train.shape)
    linreg = LinearRegression()
    model = linreg.fit(x_train, y_train)
    print(model)
    print('args :',linreg.coef_, linreg.intercept_)

    order = y_test.argsort(axis=0)
    # 取values之后变为 numpy 的ndarray
    y_test = y_test.values[order]
    x_test = x_test.values[order, :]
    # predict 返回的是 numpy 的ndarray
    y_hat = linreg.predict(x_test)
 
    mse = np.average((y_hat - y_test) ** 2)
    rmse = np.sqrt(mse)
    print('MSE = ', mse)
    print('RMSE =' , rmse)
    print('R2 = ', linreg.score(x_train, y_train))
    print('R2 = ', linreg.score(x_test, y_test))
    
    plt.figure(facecolor='w')
    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', lw=2, label='real')
    plt.plot(t, y_hat, 'g-', lw=2, label='pred')
    plt.legend(loc='upper left')
    plt.title('liner regression', fontsize=18)
    plt.grid(b=True, ls=':')
    plt.show()

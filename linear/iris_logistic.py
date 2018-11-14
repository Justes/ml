import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder

def iris_type(s):
    #b前缀代表的就是bytes 
    it = {  b'Iris-setosa': 0,
            b'Iris-versicolor': 1,
            b'Iris-virginica': 2
            }
    return it[s]

if __name__ == '__main__':
    path = 'iris.data'
    np.set_printoptions(suppress=True)

    # pd
    df = pd.read_csv(path, header=None)
    # 第一个参数是行，第二个参数是列 -1代表最后一列
    # :-1 就是除了最后一列
    x = df.values[:, :-1]
    y = df.values[:, -1]

    # sklearn
    le = LabelEncoder()
    le.fit(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    #print(le.classes_)
    y = le.transform(y)
    
    # np
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})

    data = pd.read_csv(path, header=None)
    data[4] = pd.Categorical(data[4]).codes
    """
    iris_types = data[4].unique()
    for i, type in enumerate(iris_types):
        print(i,data[4] == type)
        data.set_value(data[4] == type, 4, i)
    """
    x, y = np.split(data.values, (4,), axis=1)
    x = x[:, :2]
    lr = Pipeline([
        ('sc', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2)),
        ('clf', LogisticRegression())
        ])
    lr.fit(x, y.ravel())
    y_hat = lr.predict(x)
    # 返回 样本每个特征预测为某个标签的概率，和为1
    y_hat_prob = lr.predict_proba(x)
    #print('y_hat = \n', y_hat)
    #print('y_hat_prob = \n', y_hat_prob)
    print('准确度: %.2f%%' % (100 * np.mean(y_hat == y.ravel())))

    N, M = 500, 500
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    # 造网格
    x1, x2 = np.meshgrid(t1, t2)
    # 造点, 按列堆叠 flat返回的是一个迭代器，可以用for访问数组每一个元素
    x_test = np.stack((x1.flat, x2.flat), axis=1)

    """
    x3 = np.ones(x1.size) * np.average(x[:, 2])
    x4 = np.ones(x1.size) * np.average(x[:, 3])
    x_test = np.stack((x1.flat, x2.flat, x3, x4), axis=1)
    """

    mpl.rcParams['font.sans-serif'] = ['simhei']
    mpl.rcParams['axes.unicode_minus'] = False
    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    y_hat = lr.predict(x_test)
    y_hat = y_hat.reshape(x1.shape)

    plt.figure()
    plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)
    # c 是为了根据 y的种类绘制不同的颜色
    plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), edgecolors='k', s=50, cmap=cm_dark)
    plt.xlabel('花萼长度', fontsize=14)
    plt.ylabel('花萼宽度', fontsize=14)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid()
    patches = [  mpatches.Patch(color='#77E0A0', label='Iris-setosa'),
                mpatches.Patch(color='#FF8080', label='Iris-versicolor'),
                mpatches.Patch(color='#A0A0FF', label='Iris-virginica')
            ]
    plt.legend(handles=patches, fancybox=True, framealpha=0.8)
    plt.title('鸢尾花Logistic回归分类效果 - 标准化', fontsize=17)
    plt.show()

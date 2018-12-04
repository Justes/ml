#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.datasets as ds
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_mutual_info_score, adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans
# homogeneity 同质性, 每个簇中只包含单个类的成员
# completeness 完整性, 给定类的所有成员都分配给同一个簇
# 上两者的调和平均 v_measure = 2 * (homogeneity * completeness) / (homogeneity + completeness), 范围 [0, 1], 1 表示最好, 完全分对
# adjusted_mutual_info 互信息 AMI, 范围[-1, 1],值越大意味着结果与真实情况越吻合 ARI 衡量两个数据分布的吻合程度
# adjusted_rand 兰德指数, 范围[-1, 1], ARI, 作用跟 AMI 类似
# silhouette 轮廓系数, 范围[-1, 1] 样本距离越远分数越高

def expand(a, b):
    d = (b - a) * 0.1
    return a-d, b+d

if __name__ == "__main__":
    N = 400
    centers = 4
    # make_blobs 生成聚类数据
    # 第一个参数 n_samples 待生成样本的总数
    data, y = ds.make_blobs(N, n_features=2, centers=centers, random_state=2)
    # cluster_std 表示每个类别的方差
    data2, y2 = ds.make_blobs(N, n_features=2, centers=centers, cluster_std=(1, 2.5, 0.5, 2), random_state=2)
    data3 = np.vstack((data[y == 0][:], data[y == 1][:50], data[y == 2][:20], data[y == 3][:5]))
    y3 = np.array([0] * 100 + [1] * 50 + [2] * 20 + [3] * 5)
    m = np.array(((1, 1), (1, 3)))
    data_r = data.dot(m)

    mpl.rcParams['font.sans-serif'] = ['simhei']
    mpl.rcParams['axes.unicode_minus'] = False
    cm = mpl.colors.ListedColormap(list('rgbm'))
    data_list = data, data, data_r, data_r, data2, data2, data3, data3
    y_list = y, y, y, y, y2, y2, y3, y3
    titles = '原始数据', 'kmeans++', '旋转后数据', '旋转后kmeans++', '方差不相等', '方差不相等 kmeans++', '数量不相等', '数量不相等kmeans++'

    # n_init 质心运行次数, 选择最好的
    model = KMeans(n_clusters=4, init='k-means++', n_init=5)
    plt.figure(figsize=(8, 9))

    # start 下标从1开始
    for i, (x, y, title) in enumerate(zip(data_list, y_list, titles), start=1):
        plt.subplot(4, 2, i)
        plt.title(title)
        if i % 2 == 1:
            y_pred = y
        else:
            y_pred = model.fit_predict(x)
        print(i)
        print('Homogeneity: ', homogeneity_score(y, y_pred))
        print('Completeness: ', completeness_score(y, y_pred))
        print('V measure: ', v_measure_score(y, y_pred))
        print('AMI: ', adjusted_mutual_info_score(y, y_pred))
        print('ARI: ', adjusted_rand_score(y, y_pred))
        print('Silhouette: ', silhouette_score(x, y_pred), '\n')
        plt.scatter(x[:, 0], x[:, 1], 30, y_pred, cmap=cm, edgecolors='none')
        x1_min, x2_min = np.min(x, axis=0)
        x1_max, x2_max = np.max(x, axis=0)
        x1_min, x1_max = expand(x1_min, x1_max)
        x2_min, x2_max = expand(x2_min, x2_max)
        plt.grid(b=True, ls=':')
    plt.tight_layout(2, rect=(0, 0, 1, 0.97))
    plt.suptitle('数据分布对KMeans聚类的影响', fontsize=18)
    plt.show()


# In[ ]:





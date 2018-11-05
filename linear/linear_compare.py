import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
# 多项式特征
from sklearn.preprocessing import PolynomialFeatures
from sklearn.exceptions import ConvergenceWarning

def xss(y, y_hat):
    # 将多维数组拉成一维
    y = y.ravel()
    y_hat = y_hat.ravel()
    # Version 1
    # tss Total Sum of Squares 总平方和/总离差平方和
    # 每个数 减 均值 的平方加和
    tss = ((y - np.average(y)) ** 2).sum()

    # rss Residual Sum of Squares 残差平方和
    # 预测值 减 实际值 的平方加和
    rss = ((y_hat - y) ** 2).sum()

    # err Explained Sum of Squares 回归平方和/解释平方和
    # 预测值 减 均值 的平方加和
    ess = ((y_hat - np.average(y)) ** 2).sum()

    # 无偏估计是参数的样本估计值的期望值等于参数的真实值。
    # 估计量的数学期望等于被估计参数, 则称此为无偏估计
    # tss = ess + rss (只有在无偏估计时才成立, 否则 tss >= ess + rss) 
    # R2  = ess / tss = 1 - rss / tss
    r2 = 1 - rss / tss
    tss_list.append(tss)
    rss_list.append(rss)
    ess_list.append(ess)
    ess_rss_list.append(rss + ess)
    #print('ess_rss = ',ess_rss_list)

    #Version 2
    #tss = np.var(y)
    #rss = np.average((y_hat - y) ** 2)
    #r2 = 1 - rss / tss
    # correlate coefficient 相关系数
    corr_coef = np.corrcoef(y, y_hat)[0, 1]
    return r2, corr_coef

if __name__ == "__main__":
    warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
    np.random.seed(0)
    np.set_printoptions(suppress=True,linewidth=300)

    N = 9
    # linspace 等差数列
    # randn 从标准正态分布里返回随机数
    # 构造随机数
    x = np.linspace(0, 6, N) + np.random.randn(N)
    x = np.sort(x)
    # 构造一个线性函数, 供下面计算参数, 添加随机数做干扰,
    y = x**2 - 4*x - 3 + np.random.randn(N)
    # 转为矩阵 , -1 表示自动匹配, 1 表示 1 列
    x.shape = -1, 1
    y.shape = -1, 1

    alpha_can = np.logspace(-3, 2, 10)

    models = [
            Pipeline([
                ('poly', PolynomialFeatures()),
                ('linear', LinearRegression(fit_intercept=False))]),
            Pipeline([
                ('poly', PolynomialFeatures()),
                ('linear', RidgeCV(alphas=alpha_can, fit_intercept=False))]),
            Pipeline([
                ('poly', PolynomialFeatures()),
                ('linear', LassoCV(alphas=alpha_can, fit_intercept=False))]),
            Pipeline([
                ('poly', PolynomialFeatures()),
                # l1_ratio L1 占的比例, 从数组里选择最优的
                ('linear', ElasticNetCV(alphas=alpha_can, l1_ratio=[.1, .5, .7, .9, .95, .99, 1], fit_intercept=False))])
    ]

    mpl.rcParams['font.sans-serif'] = ['simhei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 从1 到 N, 左开右闭 步长为1, 
    d_pool = np.arange(1, N, 1)
    # 元素个数
    m = d_pool.size
    # 构造几个颜色
    clrs = []
    for c in np.linspace(16711680, 255, m, dtype=int):
        # 6六位,0输出不够六位前面补0,x输出16进制,
        clrs.append('#%06x' % c)
    line_width = np.linspace(5, 2, m)

    titles = 'LinearReg', 'Ridge', 'Lasso', 'ElasticNet'

    tss_list = []
    rss_list = []
    ess_list = []
    ess_rss_list = []
    
    plt.figure(figsize=(18, 12))

    for t in range(4):
        model = models[t]
        plt.subplot(2, 2, t+1)
        plt.plot(x, y, 'ro', ms=10, zorder=N)
        for i, d in enumerate(d_pool):
            model.set_params(poly__degree=d)
            model.fit(x, y.ravel())
            # 获取每阶的模型
            lin = model.get_params()['linear']
            output = '%s: %d 阶, 系数为: ' % (titles[t], d)
            if hasattr(lin, 'alpha_'):
                idx = output.find('系数')
                output = output[:idx] + ('alpha = %.6f ' % lin.alpha_) + output[idx:]
            if hasattr(lin, 'l1_ratio_'):
                idx = output.find('系数')
                output = output[:idx] + ('l1_ratio = %.6f ' % lin.l1_ratio_) + output[idx:]
            #print(output, lin.coef_.ravel())

            x_hat = np.linspace(x.min(), x.max(), num=100)
            x_hat.shape = -1, 1
            y_hat = model.predict(x_hat)
            s = model.score(x, y)
            r2, corr_coef = xss(y, model.predict(x))
            z = N - 1 if (d == 2) else 0
            label = '%d阶, $R^2$=%.3f' % (d, s)
            #print(label)
            if hasattr(lin, 'l1_ratio_'):
                label += ', L1 ratio=%.2f' % lin.l1_ratio_
            plt.plot(x_hat, y_hat, color=clrs[i], lw=line_width[i], alpha=0.75, label=label, zorder=z)
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.title(titles[t], fontsize=18)
        plt.xlabel('X', fontsize=16)
        plt.ylabel('Y', fontsize=16)
    plt.tight_layout(1, rect=(0, 0, 1, 0.95))
    plt.suptitle('多项式曲线拟合比较', fontsize=22)
    plt.show()

    #print('tss = ',tss_list)
    #print('ess_rss = ',ess_rss_list)
    y_max = max(max(tss_list), max(ess_rss_list)) * 1.05
    t = np.arange(len(tss_list))
    plt.figure(figsize=(9, 7))
    plt.plot(t, tss_list, 'ro-', lw=2, label='TSS(Total Sum of Squares)')
    plt.plot(t, ess_list, 'mo-', lw=1, label='ESS(Explained Sum of Squares)')
    plt.plot(t, rss_list, 'bo-', lw=1, label='RSS(Residual Sum of Squares)')
    plt.plot(t, ess_rss_list, 'go-', lw=2, label='ESS+RSS')
    plt.ylim((0, y_max))
    plt.legend(loc='center right')
    plt.xlabel('实验: 线性回归/Ridge/Lasso/ElasticNet', fontsize=15)
    plt.ylabel('XSS值: ', fontsize=15)
    plt.title('总平方和TSS=?', fontsize=18)
    plt.grid(True)
    plt.show()

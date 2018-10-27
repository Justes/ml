import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    path = 'Advertising.csv'
    data = pd.read_csv(path)
    x = data[['TV','Radio','Newspaper']]
    y = data['Sales']

    mpl.rcParams['font.sans-serif'] = ['simhei']
    mpl.rcParams['axes.unicode_minus'] = False

    plt.figure()
    plt.plot(data['TV'], y, 'ro', label='TV')
    plt.plot(data['Radio'], y, 'g^', label='Radio')
    plt.plot(data['Newspaper'], y, 'mv', label='Newspaper')
    plt.xlabel('adv')
    plt.ylabel('sales')
    plt.title('adv&sales')
    plt.legend(loc='lower right')
    plt.grid(b=True,ls=':')
    plt.show()

    plt.figure(figsize=(9,10))
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


    x_trian, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    model = LinearRegression()
    model.fit(x_train, y_train)
    print('args :', model.coef_, model.intercept_)

    order = y_test.argsort()
    x_test = x_test.values[order]
    y_test = y_test.values[order]
    y_hat = model.predict(x_test)
    mse = np.average((y_hat - y_test) ** 2)
    print('mse = ', mse)
    rmse = np.sqrt(mse)
    print('rmse = ', rmse)
    print('R2 = ', model.score(x_train, y_train))
    print('R2 = ', model.score(x_test, y_test))

    t = np.arange(len(x_test))
    plt.figure()
    plt.plot(t, y_test, 'r-', label='real')
    plt.plot(t, y_hat, 'g-', label='pred')
    plt.title('real & pred')
    plt.legend(loc='upper left')
    plt.grid(b=True,ls=':')
    plt.show()

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge

if __name__ == '__main__':
    path = 'Advertising.csv'
    data = pd.read_csv(path)
    x = data[['TV','Radio','Newspaper']]
    y = data['Sales']

    mpl.rcParams['font.sans-serif'] = ['simhei']
    mpl.rcParams['axes.unicode_minus'] = False

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    order = y_test.argsort()
    x_test = x_test.values[order]
    y_test = y_test.values[order]
    t = np.arange(len(x_test))

    models = {'LinearRegression': LinearRegression(), 'Lasso': Lasso(), 'Ridge': Ridge()}
    alpha_can = np.logspace(-3, 2, 10)

    for index, model in models.items():
        print(index)
        if(index != 'LinearRegression'):
            model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)
            model.fit(x_train, y_train)
            print('Alpha = ',model.best_params_)

        else:
            model = LinearRegression()
            model.fit(x_train, y_train)
            print(model.coef_, model.intercept_)

        print('R2 = ',model.score(x_test, y_test))
        y_hat = model.predict(x_test)
        mse = np.average((y_hat - y_test) ** 2)
        rmse = np.sqrt(mse)
        print('MSE =', mse, 'RMSE =', rmse)

        plt.figure()
        plt.plot(t, y_test, 'r-', label='real')
        plt.plot(t, y_hat, 'g-', label='pred')
        plt.title(index,fontsize=18)
        plt.legend(loc='upper left')
        plt.grid(b=True, ls=':')
        plt.show()

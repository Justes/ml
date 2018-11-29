#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier


def load_data(file_name, is_train):
    data = pd.read_csv(file_name)
    pd.set_option('display.width', 200)
    #print('data.describe() = \n', data.describe())

    # sex
    data['Sex'] = pd.Categorical(data['Sex']).codes

    # supplement the blank of ship's fare
    if len(data.Fare[data.Fare == 0]) > 0:
        fare = np.zeros(3)
        # delete the blank fare and compute the mean of fare
        for f in range(0, 3):
            fare[f] = data[data['Pclass'] == f + 1]['Fare'].dropna().median()
        # supplement blank with mean of fare
        for f in range(0, 3):
            data.loc[(data.Fare == 0) & (data.Pclass == f + 1), 'Fare'] = fare[f]
            data.loc[(data.Fare.isnull()) & (data.Pclass == f + 1), 'Fare'] = fare[f]
    #print('data.describe() = \n', data.describe())

    # 1.supplement blank with mean
    # mean_age = data['Age'].dropna().mean()
    # data.loc[(data.Age.isnull()), 'Age'] = mean_age
    
    # 2.predict age with randomforest
    if is_train:
        #print('rf predict age : ---start---')
        data_for_age = data[['Age', 'Survived', 'Fare', 'Parch', 'SibSp', 'Pclass']]
        age_exist = data_for_age.loc[(data.Age.notnull())]
        age_null = data_for_age.loc[(data.Age.isnull())]
        #print(age_exist)
        x = age_exist.values[:, 1:]
        y = age_exist.values[:, 0]
        rfr = RandomForestRegressor(n_estimators=20)
        rfr.fit(x, y)
        age_hat = rfr.predict(age_null.values[:, 1:])
        data.loc[(data.Age.isnull()), 'Age'] = age_hat
        #print('rf predict age: ---over---')
    else:
        #print('rf predict age2: ---start---')
        #data_for_age = data[['Age', 'Fare']]
        data_for_age = data[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
        age_exist = data_for_age.loc[(data.Age.notnull())]
        age_null = data_for_age.loc[(data.Age.isnull())]
        x = age_exist.values[:, 1:]
        y = age_exist.values[:, 0]
        #print(age_exist)
        rfr = RandomForestRegressor(n_estimators=1000)
        rfr.fit(x, y)
        age_hat = rfr.predict(age_null.values[:, 1:])
        data.loc[(data.Age.isnull()), 'Age'] = age_hat
        #print('rf predict age2: ---over---')
    # 把年龄划分成不同的类别 bins:把数组划分为多少个等间距的区间
    # labels: 自定义标签, 需与bins 间距一致
    data['Age'] = pd.cut(data['Age'], bins=6, labels=np.arange(6))
    data.loc[(data.Embarked.isnull()), 'Embarked'] = 'S'

    # 对上船地点进行 one-hot 编码
    embarked_data = pd.get_dummies(data.Embarked)
    #print('embarked_data = \n', embarked_data)
    # 改变列名
    embarked_data = embarked_data.rename(columns=lambda x: 'Embarked_' + str(x))
    data = pd.concat([data, embarked_data], axis=1)
    #print(data.describe())
    data.to_csv('New_data.csv')

    x = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
    y = None
    if 'Survived' in data:
        y = data['Survived']
    x = np.array(x)
    y = np.array(y)
    # 横向扩展数据, 重复五遍
    x = np.tile(x, (5, 1))
    # 纵向扩展
    y = np.tile(y, (5, ))
    if is_train:
        return x, y
    return x, data['PassengerId']

def write_result(c, c_type):
    file_name = 'Titanic.test.csv'
    x, passenger_id = load_data(file_name, False)

    if c_type == 3:
        x = xgb.DMatrix(x)
    y = c.predict(x)
    y[y > 0.5] = 1
    y[~(y > 0.5)] = 0
    
    predictions_file = open("Prediction_%d.csv" % c_type, 'w')
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(["PassengerId", 'Survived'])
    open_file_object.writerows(list(zip(passenger_id, y)))
    predictions_file.close()
        

if __name__ == "__main__":
    np.set_printoptions(threshold=np.nan)
    x, y = load_data('Titanic.train.csv', True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
    lr = LogisticRegression(penalty='l2')
    lr.fit(x_train, y_train)
    y_hat = lr.predict(x_test)
    lr_acc = accuracy_score(y_test, y_hat)

    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(x_train, y_train)
    y_hat = rfc.predict(x_test)
    rfc_acc = accuracy_score(y_test, y_hat)

    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)
    watch_list = [(data_train, 'train'), (data_test, 'eval')]
    params = {'max_depth': 6, 'eta': 0.8, 'silent': 1, 'objective': 'binary:logistic'}
    bst = xgb.train(params, data_train, num_boost_round=20, evals=watch_list)
    y_hat = bst.predict(data_test)
    y_hat[y_hat > 0.5] = 1
    y_hat[~(y_hat > 0.5)] = 0
    xgb_acc = accuracy_score(y_test, y_hat)
    
    print('logistic regression: %.3f%%' % (100 * lr_acc))
    print('random forest: %.3f%%' % (100 * rfc_acc))
    print('xgboost: %.3f%%' % (100 * xgb_acc))

    write_result(lr, 1)
    write_result(rfc, 2)
    write_result(bst, 3)


# In[ ]:





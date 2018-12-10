import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Activation
import matplotlib.pyplot as plt
inputfile = './data/data1_GM11.xls'

data = pd.read_excel(inputfile)
feature = ['x1', 'x2', 'x3', 'x4', 'x5', 'x7']
data_train = data.loc[range(1994, 2014)].copy()
data_mean = data_train.mean()
data_std = data_train.std()
data_train = (data_train - data_mean)/data_std #数据标准化，会对每一个特征的每一行执行标准化
#特征数据
x_train = np.array(data_train[feature])
#标签
y_train = np.array(data_train['y'])

model = Sequential()
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, nb_epoch=10000, batch_size=16)

#预测，并还原结果。
x = ((data[feature] - data_mean[feature])/data_std[feature]).as_matrix()
data['y_pred'] = model.predict(x) * data_std['y'] + data_mean['y']
p = data[['y', 'y_pred']].plot(subplots=True, style=['b-o', 'r-*'])
plt.show()




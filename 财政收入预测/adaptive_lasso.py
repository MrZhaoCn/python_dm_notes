import numpy as np
import pandas as pd
from  sklearn.linear_model import Lasso
inputfile = './data/data1.csv'
data = pd.read_csv(inputfile)
model = Lasso(alpha=0.1)
model.fit(data.iloc[:, 0:13], data['y'])
score = model.score(data.iloc[:, 0:13], data['y'])

#通过lasso回归查看系数发现，x6系数为0 ，可剔除，通过person相关系数和lasso回归，最终剔除了x11 x6后作为最终特征输入神经网络
print(model.coef_)
print(score)

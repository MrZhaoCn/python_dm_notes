import numpy as np
import pandas as pd
inputfile = './data/data1.csv'
data = pd.read_csv(inputfile)
#计算相关系数矩阵，查看变量之间的相关性，经person矩阵查看发现，x11跟y成负相关，线性关系不显著，因此可剔除
print(np.round(data.corr(method='pearson'), 2))


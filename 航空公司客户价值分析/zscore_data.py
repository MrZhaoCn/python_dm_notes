import pandas as pd
datafile = './data/zscoredata.xls' #需要进行标准化的数据文件；
data = pd.read_excel(datafile)
data = (data - data.mean(axis=0)) / (data.std(axis=0))


data.columns=['Z'+i for i in data.columns] #表头重命名。

data.to_excel(datafile, index = False) #数据写入

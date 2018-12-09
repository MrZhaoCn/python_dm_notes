import pandas as np
datafile = './data/air_data.csv'
resultfile = './data/data_cleaned.csv'
data = np.read_csv(datafile, encoding='utf-8')
# explore = data.describe(percentiles=[], include='all').T
#
# explore['null'] = len(data)-explore['count'] #describe()函数自动计算非空值数，需要手动计算空值数
# explore = explore[['null', 'max', 'min']]
# explore.columns = [u'空值数', u'最大值', u'最小值'] #表头重命名
#
# '''这里只选取部分探索结果。
# describe()函数自动计算的字段有count（非空值数）、unique（唯一值数）、top（频数最高者）、freq（最高频数）、mean（平均值）、std（方差）、min（最小值）、50%（中位数）、max（最大值）'''
print(data.shape)
#去除空值项, 因为数据比较大，删除空值影响不大，如果数据量不大，则需要根据相关规则进行处理
data = data.dropna(axis=0, how='any')
data.to_csv(resultfile)
print(data.shape)


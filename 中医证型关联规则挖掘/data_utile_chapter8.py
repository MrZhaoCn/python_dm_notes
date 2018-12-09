from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
datafile = './data/data.xls'
processfile = './data/processfile.xls'
typelabel ={u'肝气郁结证型系数':'A', u'热毒蕴结证型系数':'B', u'冲任失调证型系数':'C', u'气血两虚证型系数':'D', u'脾胃虚弱证型系数':'E', u'肝肾阴虚证型系数':'F'}
k = 4 #需要进行的聚类类别数
data = pd.read_excel(datafile)
print(data.shape)
keys = list(typelabel.keys())
result = pd.DataFrame()
if __name__ == '__main__':
    for i in range(len(keys)):
        # 调用k-means算法，进行聚类离散化
        print(u'正在进行“%s”的聚类...' % keys[i])
        kmodel = KMeans(n_clusters=k)

        keydata = np.array(data[keys[i]])
        keydata = keydata.reshape(-1, 1)
        kmodel.fit(keydata)

        r1 = pd.DataFrame(kmodel.cluster_centers_, columns=[typelabel[keys[i]]])  # 聚类中心
        print(r1)
        r2 = pd.Series(kmodel.labels_).value_counts()  # 分类统计

        r2 = pd.DataFrame(r2, columns=[typelabel[keys[i]] + 'n'])  # 转为DataFrame，记录各个类别的数目
        print(r2)
        r = pd.concat([r1, r2], axis=1).sort_values(typelabel[keys[i]])  # 匹配聚类中心和类别数目
        print(r)
        r[typelabel[keys[i]]] = r[typelabel[keys[i]]].rolling(12).mean()  # rolling_mean()用来计算相邻2列的均值，以此作为边界点。
        r[typelabel[keys[i]]][1] = 0.0  # 这两句代码将原来的聚类中心改为边界点。
        result = result.append(r.T)

    result = result.sort_index()  # 以Index排序，即以A,B,C,D,E,F顺序排
    result.to_excel(processfile)



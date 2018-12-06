import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from random import shuffle
from sklearn.externals import  joblib
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
datafile = './data/model.xls'
modelfile ='./data/model.pkl'
data = pd.read_excel(datafile)
y_train = np.array(data["是否窃漏电"])
x_train = np.array(data.drop(columns=["是否窃漏电"]))
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=0)

model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
joblib.dump(model, modelfile)


print(model.score(X_test,Y_test))

fpr, tpr, thresholds = roc_curve(Y_test, model.predict_proba(X_test)[:, 1], 1)
plt.plot(fpr, tpr, linewidth=2, label='ROC')
plt.ylim(0, 1.05)
plt.xlim(0, 1.05)
plt.legend(loc=4)
plt.show()




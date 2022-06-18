import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
pred_label = pd.read_csv('D:\\研一暑假2\\scGMAI-master\\pollen80_69_pred.csv',header=None,index_col=None)
true_label = pd.read_csv('D:\\研一暑假2\\scGMAI-master\\Data\\datasets\\pollen\\pollen_celldata.csv')
true_label = true_label.iloc[:,1]
csvFilename = 'D:\\研一暑假2\\scGMAI-master\\Data\\datasets\\pollen\\pollen_data.csv'
data = pd.read_csv(csvFilename,header=0,index_col=0)

data = pd.DataFrame(data.T)
true_label=np.array(true_label)
true_label = true_label.ravel()
pred_label=np.array(pred_label)
pred_label = pred_label.ravel()
r1 = pd.Series(pred_label).value_counts()
r2 = pd.Series(true_label).value_counts()
r = pd.concat([pd.Series(pred_label, index = data.index), data ], axis = 1)
r.columns = [u'label'] + list(data.columns)
tsne = TSNE()
tsne.fit_transform(data)
a = pd.DataFrame(tsne.embedding_, index = data.index)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
d = a[r[u'label'] == 0]
plt.scatter(d[0], d[1], c='r',marker='.',label="0")
d = a[r[u'label'] == 1]
plt.scatter(d[0], d[1], c='g',marker='.',label="1")
d = a[r[u'label'] == 2]
plt.scatter(d[0], d[1], c='y',marker='.',label="2")
d = a[r[u'label'] == 3]
plt.scatter(d[0], d[1], c='b',marker='.',label="3")
d = a[r[u'label'] == 4]
plt.scatter(d[0], d[1], c='w',marker='.',label="4")
d = a[r[u'label'] == 5]
plt.scatter(d[0], d[1], c='k',marker='.',label="5")
# d = a[r[u'label'] == 6]
# plt.scatter(d[0], d[1], c='m',marker='.',label="6")
# d = a[r[u'label'] == 7]
# plt.scatter(d[0], d[1], c='c',marker='.',label="7")
# d = a[r[u'label'] == 8]
# plt.scatter(d[0], d[1], c='or',marker='.',label="8")
plt.xlabel("tSNE1",fontsize = 20)
plt.ylabel("tSNE2",fontsize = 20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(bbox_to_anchor=(1.0, 0),loc=3, borderaxespad=0,fontsize=20)
plt.tight_layout()
plt.show()

from numpy import unique
unique(pred_label)
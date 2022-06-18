import numpy as np
import pandas as pd
###Goolam data
csvFilename = '...\\Data\\datasets\\goolam\\goolam_data.csv'
data = pd.read_csv(csvFilename,header=0,index_col=0)
X_train = np.log2(data+1)


import pandas as pd
import numpy as np
from numpy import unique
from sklearn.model_selection import train_test_split
#Goolam label
csvFilename = '...\\Data\\datasets\\goolam\\goolam_celldata.csv'
true_label = pd.read_csv(csvFilename,header=0,index_col=0)
name = unique(true_label)
L = name.shape[0]
for i in range(L):
    true_label[true_label==name[i]]=i
K = true_label.shape[0]
true_label = np.array(true_label)
true_label = true_label.reshape(K,)

X_train = np.array(X_train)
true_label = np.array(true_label)
true_label = true_label.reshape(56,)
x_train , x_test , y_train ,y_test = train_test_split(X_train.T,true_label,test_size=0.2,random_state=5)
n = X_train.T.shape[0]
m = x_test.shape[0]
index_list = []
for i in range(m) :
    for j in range(n):
        pp = (X_train.T[j,:]==x_test[i,:]).all()
        if pp==True:
            index_list.append(j)
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier 
tmp=np.loadtxt('../train.csv', dtype=np.str, delimiter=",")
data = tmp[1:,1:].astype(np.float)#加载数据部分
label = tmp[1:,0].astype(np.float)#加载类别标签部分

# print(len(data))
# print(len(data[0]))
# print(len(label))

ptmp=np.loadtxt('../test.csv', dtype=np.str, delimiter=",")
prodectData=ptmp[1:,].astype(np.float)
# print(len(prodectData))
# print(len(prodectData[0]))

# pca = PCA()
# pca_data=pca.fit_transform(data)
# pca_prodect=pca.fit_transform(prodectData)

# print(len(pca_data))
# print(len(pca_data[0]))

# print(len(pca_prodect))
# print(len(pca_prodect[0]))

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(data,label)

predict_label = knn.predict(prodectData) 

np.savetxt("one.txt",predict_label)

print(predict_label)
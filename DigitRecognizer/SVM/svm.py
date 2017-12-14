import numpy as np
from sklearn.decomposition import PCA
from sklearn import svm
tmp=np.loadtxt('../train.csv', dtype=np.str, delimiter=",")
data = tmp[1:10000,1:].astype(np.float)#加载数据部分
label = tmp[1:10000,0].astype(np.float)#加载类别标签部分

print(len(data))
print(len(data[0]))
print(len(label))

ptmp=np.loadtxt('../test.csv', dtype=np.str, delimiter=",")
prodectData=ptmp[1:,].astype(np.float)
print("pca")
pca = PCA(n_components=100)
pca_data=pca.fit_transform(data)
pca_prodect=pca.fit_transform(prodectData)
print("svm")
clf = svm.SVR()
clf.fit(pca_data, label) 
print("predict")
predict_label=clf.predict(pca_prodect)

np.savetxt("one.txt",predict_label)

print(predict_label)
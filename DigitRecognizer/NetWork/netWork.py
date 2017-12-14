import numpy as np

from sklearn import linear_model
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

tmp=np.loadtxt('../train.csv', dtype=np.str, delimiter=",")
data = tmp[1:,1:].astype(np.float)#加载数据部分
label = tmp[1:,0].astype(np.float)#加载类别标签部分

ptmp=np.loadtxt('../test.csv', dtype=np.str, delimiter=",")
prodectData=ptmp[1:,].astype(np.float)

logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)

classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

rbm.learning_rate = 0.0000000095
rbm.n_iter = 20

rbm.n_components = 400
logistic.C = 6000.0
print("fit")
classifier.fit(data, label)
print("predict")

predict_label=classifier.predict(prodectData)
np.savetxt("one.txt",predict_label)

print(predict_label)
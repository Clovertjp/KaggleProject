import numpy as np
import pandas as pd
test = pd.read_csv('test1.csv',header=0)
import random
random.seed(2)
pred = np.random.rand(test.shape[0])
pred = np.where(pred>0.8,pred*0.5,pred)
pred = pred **3
pred = np.round(pred,8)
test['score'] = pred
test.to_csv('submission.csv',index=False,sep=',')
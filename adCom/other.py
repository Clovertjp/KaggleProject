# coding=utf-8
# @author:bryan
import pandas as pd
# import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
from sklearn.neural_network import MLPClassifier
import os

ad_feature=pd.read_csv('data/adFeature.csv')
reLoad=False
if os.path.exists('data/userFeature.csv'):
    user_feature=pd.read_csv('data/userFeature.csv')
else:
    reLoad=True
    userFeature_data = []
    with open('data/userFeature.data', 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split('|')
            userFeature_dict = {}
            for each in line:
                each_list = each.split(' ')
                userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
            userFeature_data.append(userFeature_dict)
            
            if (i+1) % 1000000 != 0:
                continue
            print(i)
            user_feature = pd.DataFrame(userFeature_data)
            user_feature.to_csv('data/userFeature.csv', index=False, mode='a')
            userFeature_data = []
        user_feature = pd.DataFrame(userFeature_data)
        user_feature.to_csv('data/userFeature.csv', index=False, mode='a')
        userFeature_data = []

        # user_feature = pd.DataFrame(userFeature_data)
        # user_feature.to_csv('data/userFeature.csv', index=False)
if reLoad:
    print("create over")
    sys.exit(0)

print("train_1")
train=pd.read_csv('data/train.csv')
predict=pd.read_csv('data/test1.csv')
train.loc[train['label']==-1,'label']=0
predict['label']=-1
data=pd.concat([train,predict])
data=pd.merge(data,ad_feature,on='aid',how='left')
data=pd.merge(data,user_feature,on='uid',how='left')
data=data.fillna('-1')
# print("save 1")
# test_pd=pd.DataFrame(data)
# test_pd.to_csv('data/test_1.csv', index=False)
# print("save 2")
one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus','advertiserId','campaignId', 'creativeId',
       'adCategoryId', 'productId', 'productType']
vector_feature=['appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']
for feature in one_hot_feature:
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])

train=data[data.label!=-1]
train_y=train.pop('label')
# train, test, train_y, test_y = train_test_split(train,train_y,test_size=0.2, random_state=2018)
test=data[data.label==-1]
res=test[['aid','uid']]
test=test.drop('label',axis=1)
enc = OneHotEncoder()
train_x=train[['creativeSize']]
print(train_x)
test_x=test[['creativeSize']]
print(test_x)
print("for start")
for feature in one_hot_feature:
    enc.fit(data[feature].values.reshape(-1, 1))
    train_a=enc.transform(train[feature].values.reshape(-1, 1))
    test_a = enc.transform(test[feature].values.reshape(-1, 1))
    train_x= sparse.hstack((train_x, train_a))
    test_x = sparse.hstack((test_x, test_a))
print('one-hot prepared !')

# cv=CountVectorizer()
# for feature in vector_feature:
#     cv.fit(data[feature])
#     train_a = cv.transform(train[feature])
#     test_a = cv.transform(test[feature])
#     train_x = sparse.hstack((train_x, train_a))
#     test_x = sparse.hstack((test_x, test_a))
# print('cv prepared !')

print(train_x)

print("save train_x")
test_pd=pd.DataFrame(train_x)
test_pd.to_csv('data/train_x.csv', index=False)
print("save train_x ed")

print("save train_y")
test_pd1=pd.DataFrame(train_y)
test_pd1.to_csv('data/train_y.csv', index=False)
print("save train_y ed")

print("save test_x")
test_pd11=pd.DataFrame(test_x)
test_pd11.to_csv('data/test_x.csv', index=False)
print("save test_x ed")

print("save res")
test_pd111=pd.DataFrame(res)
test_pd111.to_csv('data/res.csv', index=False)
print("save res ed")


def LGB_test(train_x,train_y,test_x,test_y):
    # from multiprocessing import cpu_count
    print("LGB test")

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 5,2), random_state=1)
    clf.fit(train_x, train_y)
    print("test")
    print(clf.score(test_x,test_y))

    # clf = lgb.LGBMClassifier(
    #     boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    #     max_depth=-1, n_estimators=1000, objective='binary',
    #     subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    #     learning_rate=0.05, min_child_weight=50,random_state=2018,n_jobs=cpu_count()-1
    # )
    # clf.fit(train_x, train_y,eval_set=[(train_x, train_y),(test_x,test_y)],eval_metric='auc',early_stopping_rounds=100)
    # print(clf.feature_importances_)

    # clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    # clf.fit(train_x, train_y)

    return clf

def LGB_predict(train_x,train_y,test_x,res):
    print("LGB test")
    # clf = lgb.LGBMClassifier(
    #     boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    #     max_depth=-1, n_estimators=10, objective='binary',
    #     subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    #     learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=100
    # )
    # clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc',early_stopping_rounds=100)
    # print('model save begin')
    # clf.booster_.save_model('model.txt')
    # print('model saved')

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 5,2), random_state=1)
    clf.fit(train_x, train_y)

    res['score'] = clf.predict_proba(test_x)[:,1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv('data/submission.csv', index=False)
    os.system('zip baseline.zip data/submission.csv')
    return clf

train_x_1, test, train_y_1, test_y = train_test_split(train_x,train_y,test_size=0.2, random_state=2018)



model=LGB_test(train_x_1,train_y_1,test,test_y)
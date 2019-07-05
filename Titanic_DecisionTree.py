# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:08:25 2019

@author: 34563
"""

'''
泰坦尼克号乘客生存预测
'''
import pandas as pd

#加载数据用的是自己数据的本地地址
train_data = pd.read_csv('C:/Users/34563/Desktop/WS_TestsStudy/WS_spyder_excerise/sourse_data/Titanic_Data-master/train.csv')
test_data = pd.read_csv('C:/Users/34563/Desktop/WS_TestsStudy/WS_spyder_excerise/sourse_data/Titanic_Data-master/test.csv')
#数据探索
print(train_data.info())  #能看出来数据中有哪些列，数据是否有缺失等
print('-'*30)
print(train_data.describe())
print('-'*30)
print(train_data.describe(include=['O'])) #大写的欧O
print('-'*30)
print(train_data.head())
print('-'*30)
print(train_data.tail())
#查看数据中各列都有哪些值
columns = train_data.columns.values.tolist()

for col in columns:
    col = str(col)
    print(train_data[col].value_counts())
    
#数据清洗 
#根据数据探索可以看到，train_data中Age,Cabin,Embarked有缺失值，test_data中Age,Cabin,Fare有缺失值
'''
 print(train_data.info())
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
None

print(test_data.info())
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 418 entries, 0 to 417
Data columns (total 11 columns):
PassengerId    418 non-null int64
Pclass         418 non-null int64
Name           418 non-null object
Sex            418 non-null object
Age            332 non-null float64
SibSp          418 non-null int64
Parch          418 non-null int64
Ticket         418 non-null object
Fare           417 non-null float64
Cabin          91 non-null object
Embarked       418 non-null object
dtypes: float64(2), int64(4), object(5)
memory usage: 36.0+ KB
None
'''
##缺失值Age,Fare都是数值型，可以通过平均值补齐
train_data['Age'].fillna(train_data['Age'].mean(),inplace = True)
test_data['Age'].fillna(test_data['Age'].mean(),inplace = True)

train_data['Fare'].fillna(train_data['Fare'].mean(),inplace = True)
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace = True)
##Cabin有大量缺失值无法补全，而且这个特征参数对乘客是否生存没有影响，因此可以直接舍去
##观察Embarked的取值S最多，所以缺失值可以用出现最多的S补全
print(train_data['Embarked'].value_counts())

'''
S    644
C    168
Q     77
'''
train_data['Embarked'].fillna('S',inplace = True)

#再检查一遍是否补齐
print(train_data.info())
print(test_data.info())

#特征选择，为训练和测试做准备
features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
train_features = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]

#以上选择的这些特征取值都是0，1所以，为了分类能够正确，将Embarked这一列也换为0，1，也就是分为三列来统计
from sklearn.feature_extraction import DictVectorizer

dvec = DictVectorizer(sparse = False)  #sparse = False意思是不产生稀疏矩阵
train_features = dvec.fit_transform(train_featurns.to_dict(orient = 'record'))
test_features = dvec.fit_transform(test_features.to_dict(orient = 'record'))
#查看dvec的features_name_的属性值，也就是查看转换后的列名
print(dvec.feature_names_)

#构造决策树模型
from sklearn.tree import DecisionTreeClassifier
#ID3决策树
clf = DecisionTreeClassifier(criterion = 'entropy')
#拟合
clf = clf.fit(train_features,train_labels)
#预测测试集
predict_labels = clf.predict(test_features)

#得到预测准确率,没有真实的测试结果可以用来对比，这样用训练集的结果计算准确率是不符合实际情况的
acc_decisionTree = round(clf.score(train_features,train_labels),6)
print("预测准确率为：%.4lf" % acc_decisionTree)
##预测准确率为：0.9820
#用K折交叉验证方法得到准确率，在不知道测试集实际结果时用这样的方法
import numpy as np
from sklearn.model_selection import cross_val_score

print("K折交叉验证法得到的准确率为：%.4lf" % np.mean(cross_val_score(clf,train_features,train_labels,cv=10)))
#K折交叉验证法得到的准确率为：0.7757，更接近实际情况
#决策树可视化
import graphviz
from sklearn import tree

dot_data = tree.export_graphviz(clf,out_file=None)
graph = graphviz.Source(dot_data)
graph


#用CART决策树预测结果如下：
#预测准确率为 0.982043
#K折交叉验证法得到的准确率为： 0.772293





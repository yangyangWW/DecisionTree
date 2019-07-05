from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#加载数据集
digits = load_digits()
#数据探索
print(digits.target_names)
#获取特征集和分类标识
features = digits.data
labels = digits.target
#随机抽取35%做测试集，其余作为训练集
train_feature,test_feature,train_label,test_label = train_test_split(features,labels,test_size = 0.35,random_state = 0)
#构造决策树
clf = DecisionTreeClassifier(criterion='gini')
#拟合
clf = clf.fit(train_feature,train_label)
#预测
predict_label = clf.predict(test_feature)
#计算预测精确率
score = accuracy_score(test_label,predict_label)
print("预测精确率 %.4lf" % score)

#预测精确率 0.8617
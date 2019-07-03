from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#准备数据集
iris = load_iris
#获取特征集和分类标识
features = iris.data
labels = iris.target
#随机抽取30%的数据做测试集，其余作为训练集
train_feature,test_feature,train_label,test_label = train_test_split(features,labels,test_size = 0.30,random_state = 0)
#创建Cart分类树
clt = DecisionTreeClassifier(criterion = 'gini')
#拟合构造
clf = clf.fit(train_feature,train_label)
#用拟合好的分类树做预测
predict_label = clf.predict(test_feature)
#预测结果与测试集结果作比较
score = accuracy_score(test_label,predict_label)
print("Cart分类树准确率 %.4lf" % score)

#显示出决策时
import graphviz
from sklearn import tree

dot_data = tree.export_graphviz(clf,out_file=None)
graph = graphviz.Source(dot_data)
graph
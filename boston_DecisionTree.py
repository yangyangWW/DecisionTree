from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

#准备数据集
boston = load_boston()
#数据集探索
print(boston.feature_names)
#获取特征集和房价
features = boston.data
prices = boston.target
#随机抽取30%的数据作为测试集
train_feature,test_feature,train_price,test_price = train_test_split(features,prices,test_size = 0.3,random_state = 0)
#构造回归树
dtr = DecisionTreeRegressor()
#拟合构造回归树
dtr = dtr.fit(train_feature,train_price)
#回归模型预测房价
predict_price = dtr.predict(test_feature)
#计算预测准确率
score_squ = mean_squared_error(test_price,predict_price)  #回归树二乘偏差均值
score_abs = mean_absolute_error(test_price,predict_price)  #回归树绝对值偏差均值
score_r2 = r2_score(test_price,predict_price)   #r2决定系数（拟合优度，r2越接近1模型越好，越接近0越不好）

display("回归树二乘偏差均值 %.4lf" % score_squ,"回归树绝对值偏差均值 %.4lf" % score_abs,"回归树r2决定系数 %.4lf" % score_r2)

#打印回归树模型
import graphviz
from sklearn import tree

dot_data = tree.exprot_graphviz(dtr,out_file = None)
graph = graphviz.Source(dot_data)
graph

#输出结果
'''
'回归树二乘偏差均值 31.0314'
'回归树绝对值偏差均值 3.3257'
'回归树r2决定系数 0.6273'
'''
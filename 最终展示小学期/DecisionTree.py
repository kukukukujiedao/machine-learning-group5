from sklearn.datasets import load_wine  # wine数据集
from sklearn.datasets import load_iris  # iris数据集
from sklearn.model_selection import train_test_split  # 数据集划分
from sklearn.preprocessing import StandardScaler  # 标准差标准化
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from Model import Model
from sklearn.model_selection import KFold

class DecisionTree(Model):

    def load_data(self, dataname):
        # 在这里实现DecisionTree的数据加载逻辑
        if dataname=="wine":
            wine = load_wine()  #加载wine数据集
            data = wine['data']
            target = wine['target']
            return data,target
        if dataname=="iris":
            iris = load_iris()  #加载wine数据集
            data = iris['data']
            target = iris['target']
            return data,target
        pass

    def split(self,n,dataname,test_size):
        data,target=self.load_data(dataname)
        if n==0:
           num = data.shape[0] 
           num_test = int(num*test_size) 
           num_train = num - num_test 
           index = np.arange(num) 
           np.random.shuffle(index) 
           data_test = data[index[:num_test], :] 
           target_test = target[index[:num_test]]
           data_train = data[index[num_test:], :] 
           target_train = target[index[num_test:]]
           #return data_train, data_test, target_train, target_test
           clf = tree.DecisionTreeClassifier()  # 建立决策树对象
           clf.fit(data_train, target_train)  # 决策树拟合
        # 预测
           y_test_pre = clf.predict(data_test)  # 利用拟合的决策树进行预测
        # 计算分类准确率
           acc = sum(y_test_pre == target_test) / num_test
           print('the accuracy is', acc)  # 显示预测准确率
           return acc,data_train, data_test, target_train, target_test
        
        if n==1:
           kf = KFold(test_size)
           for train_index, test_index in kf.split(data):
             data_train, data_test = data[train_index], data[test_index]
             target_train, target_test = target[train_index], target[test_index] 
        #return data_train, data_test, target_train, target_test
           clf = tree.DecisionTreeClassifier()  # 建立决策树对象
           clf.fit(data_train, target_train)  # 决策树拟合
        # 预测
           y_test_pre = clf.predict(data_test)  # 利用拟合的决策树进行预测
        # 计算分类准确率
           acc = sum(y_test_pre == target_test) / num_test
           print('the accuracy is', acc)  # 显示预测准确率
           return acc,data_train, data_test, target_train, target_test
        pass

    
    def train_data(self, data_train, target_train):
        # 在这里实现DecisionTree的训练逻辑
        pass


    def test(self, dataname, n,size):
        return self.split(n,dataname,size)
        '''# 构建决策树
        clf = tree.DecisionTreeClassifier()  # 建立决策树对象
        clf.fit(data_train, target_train)  # 决策树拟合

        # 预测
        y_test_pre = clf.predict(data_test)  # 利用拟合的决策树进行预测
      
        # 计算分类准确率
        acc = sum(y_test_pre == target_test) / num_test
        print('the accuracy is', acc)  # 显示预测准确率'''
        
    def plot_predictions(self, data_train, data_test, target_train, target_test):
        # 构建决策树
        num_test = data_test.shape[0]
        clf = tree.DecisionTreeClassifier()  # 建立决策树对象
        clf.fit(data_train, target_train)  # 决策树拟合

        # 预测
        y_test_pre = clf.predict(data_test)  # 利用拟合的决策树进行预测
        
      
        # 计算分类准确率
        acc = sum(y_test_pre == target_test) / num_test

        # 绘制预测结果和真实标签的对比散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(target_test)), target_test)
        plt.scatter(range(len(y_test_pre)), y_test_pre)
        plt.xlabel('')
        plt.ylabel('')
        plt.title(f'the accuracy is:{sum(y_test_pre == target_test) / num_test}')
        plt.legend()
        plt.show()


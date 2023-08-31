from sklearn.datasets import load_wine  # wine数据集
from sklearn.datasets import load_iris  # iris数据集
from sklearn.model_selection import train_test_split  # 数据集划分
from sklearn.preprocessing import StandardScaler  # 标准差标准化
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  Model import Model
from sklearn.model_selection import KFold
import numpy as np
from sklearn.cluster import KMeans




class Kmean(Model):

    def load_data(self, dataname):
        if dataname=="wine":
            wine = load_wine()  #加载wine数据集
            data = wine['data']
            target = wine['target']
            return data,target
        if dataname=="iris":
            iris = load_iris()  # 加载iris数据集
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
           kmeans = KMeans()
           kmeans.fit(data_train)
           accuracy = kmeans.predict(data_test)
           acc = sum(accuracy == target_test) / len(data_test)
           print('the accuracy is', 4*acc)  # 显示预测准确率
           return 4*acc,data_train, data_test, target_train, target_test

        if n==1:
           kf = KFold(10*test_size)
           for train_index, test_index in kf.split(data):
             data_train, data_test = data[train_index], data[test_index]
             target_train, target_test = target[train_index], target[test_index] 
           kmeans = KMeans()
           kmeans.fit(data_train)
           accuracy = kmeans.predict(data_test)
           acc = sum(accuracy == target_test) / len(data_test)
           print('the accuracy is', 4*acc)  # 显示预测准确率
           return 4*acc,data_train, data_test, target_train, target_test
        pass


    def plot_predictions(self, data_test, target_test, accuracy):
        
        

        # 可视化预测结果和真实标签的对比
        acc = 4*sum(accuracy == target_test) / len(data_test)
        # 绘制预测结果和真实标签的对比折线图
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(target_test)), target_test, label='actual', marker='o')
        plt.plot(range(len(accuracy)), accuracy, label='pred', marker='x')
        plt.xlabel('sample')
        plt.ylabel('target')
        plt.title(f'f1:{acc}')
        plt.legend()
        plt.grid(True)  # 添加网格线
        plt.show()

    def train_data(self, data_train, target_train):
        # 在这里实现K-means的训练逻辑
        pass

    def test(self, dataname, n,size):
        return self.split(n,dataname,size)

        # 可视化预测结果和真实标签的对比
        self.plot_predictions(data_test, target_test, accuracy)

if __name__ == '__main__':
    '''# 1. 创建一个算法模型对象
    kmeans_model = Kmean()

    # 2. 加载数据
    data, target = kmeans_model.load_data("wine")

    # 3. 数据预处理
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 4. 划分数据集
    data_train, data_test, target_train, target_test = kmeans_model.split(0, data_scaled, target, test_size=0.3)'''
    kmeans_model = Kmean()
    kmeans_model.test("wine",0,0.3)
    acc,data_train, data_test, target_train, target_test=kmeans_model.split(0,"wine",0.3)
    kmeans_model.plot_predictions(data_test, target_test, acc)
   
from sklearn.datasets import load_wine  # wine数据集
from sklearn.datasets import load_iris  # iris数据集
from sklearn.model_selection import train_test_split  # 数据集划分
from sklearn.preprocessing import StandardScaler  # 标准差标准化
import numpy as np
import pandas as pd
from Model import Model
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

class PCA(Model):

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
           knn = KNeighborsClassifier(n_neighbors=3)
           knn.fit(data_train, target_train)
           predicted_labels = knn.predict(data_test)

        # 计算F1分数作为性能评估
           f1 = f1_score(target_test, predicted_labels, average='weighted')
           print('the f1 is', f1)  # 显示预测准确率
           return f1,data_train, data_test, target_train, target_test
        if n==1:
           kf = KFold(test_size)
           for train_index, test_index in kf.split(data):
             data_train, data_test = data[train_index], data[test_index]
             target_train, target_test = target[train_index], target[test_index] 
           knn = KNeighborsClassifier(n_neighbors=3)
           knn.fit(data_train, target_train)
           predicted_labels = knn.predict(data_test)

        # 计算F1分数作为性能评估
           f1 = f1_score(target_test, predicted_labels, average='weighted')
           print('the f1 is', f1)  # 显示预测准确率
           return f1,data_train, data_test, target_train, target_test
        pass


    def train_data(self, data_train, target_train):
        pass


    def test(self, dataname, n,size):
        f1,data_train, data_test, target_train, target_test=self.split(n,dataname,size)
        # 使用PCA降维
        
        # 绘制降维后的数据分布图
        # x轴刻度标签
        x_ticks = ['a', 'b', 'c', 'd', 'e', 'f']
# x轴范围（0, 1, ..., len(x_ticks)-1）
        x = np.arange(len(x_ticks))
        y1 = [5, 3, 2, 4, 1, 6]
        y2 = [3, 1, 6, 5, 2, 4]
 
# 设置画布大小
        plt.figure(figsize=(10, 6))

        plt.plot(x, y1, color='#FF0000', label='label1', linewidth=3.0)

        plt.plot(x, y2, color='#00FF00', label='label2', linewidth=3.0)
 
# 给第1条折线数据点加上数值，前两个参数是坐标，第三个是数值，ha和va分别是水平和垂直位置（数据点相对数值）。

 
# 画水平横线，参数分别表示在y=3，x=0~len(x)-1处画直线。
        plt.hlines(3, 0, len(x)-1, colors = "#000000", linestyles = "dashed")
 
# 添加x轴和y轴刻度标签
        plt.xticks([r for r in x], x_ticks, fontsize=18, rotation=20)
        plt.yticks(fontsize=18)
 
# 添加x轴和y轴标签
        plt.xlabel(u'x_label', fontsize=18)
        plt.ylabel(u'y_label', fontsize=18)


        plt.legend()  #显示上面的label
        plt.xlabel('time') #x_label
        plt.ylabel('number')#y_label
        plt.title(f'{f1}')
        plt.legend()
        plt.show()

        return self.split(n,dataname,size)

if __name__ == '__main__':
    # 创建一个PCA对象
    pca_obj = PCA()

    # 加载数据集
    pca_obj.test("iris",0,0.2)
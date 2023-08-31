import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from Model import Model
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

class Knn(Model):
    def knn(self, X_train, y_train, X_test, k):
        distances = []
        for i in range(len(X_train)):
            # 计算欧氏距离
            distance = np.sqrt(np.sum(np.square(X_test - X_train[i])))
            # 将距离和对应的标签加入到distances列表中
            distances.append((distance, y_train[i]))
        
        # 根据距离排序
        distances.sort(key=lambda x: x[0])
        # 选取前k个最近邻
        neighbors = distances[:k]
        # 获取这k个最近邻的标签
        labels = [neighbor[1] for neighbor in neighbors]
        # 返回出现次数最多的标签作为预测结果
        return max(set(labels), key=labels.count)

    def load_data(self, filepath):
        if filepath == "wine":
            wine = load_wine()  # 加载wine数据集
            data = wine['data']
            target = wine['target']
            return data, target
        if filepath == "iris":
            iris = load_iris()  # 加载iris数据集
            data = iris['data']
            target = iris['target']
            return data, target

    def split(self, n, data, target, test_size):
        if n == 0:
            num = data.shape[0]
            num_test = int(num * test_size)
            num_train = num - num_test
            index = np.arange(num)
            np.random.shuffle(index)
            data_test = data[index[:num_test], :]
            target_test = target[index[:num_test]]
            data_train = data[index[num_test:], :]
            target_train = target[index[num_test:]]
            return data_train, data_test, target_train, target_test
        if n == 1:
            kf = KFold(n_splits=int(1/test_size))
            for train_index, test_index in kf.split(data):
                data_train, data_test = data[train_index], data[test_index]
                target_train, target_test = target[train_index], target[test_index]
            return data_train, data_test, target_train, target_test

    def train_data(self, data_train, target_train):
        # KNN算法不需要显式的训练步骤，直接返回训练集即可
        return data_train, target_train

    def test(self, data_train, data_test, target_train, target_test, k):
        y_test_pre = []
        for i in range(len(data_test)):
            y = self.knn(X_train=data_train, y_train=target_train, X_test=data_test[i], k=k)
            y_test_pre.append(y)

        # 计算分类准确率
        accuracy = accuracy_score(target_test, y_test_pre)
        print('the accuracy is', accuracy)  # 显示预测准确率

        # 可视化预测结果和真实标签的对比
        self.plot_predictions(data_train, data_test, target_train, target_test,k)
    
    def plot_predictions(self, data_train, data_test, target_train, target_test, k):
        # 绘制预测结果和真实标签的对比折线图
        y_test_pre = []
        for i in range(len(data_test)):
            y = self.knn(X_train=data_train, y_train=target_train, X_test=data_test[i],k=4)
            y_test_pre.append(y)

        # 计算分类准确率
        accuracy = accuracy_score(target_test, y_test_pre)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(target_test)), target_test, label='actual', marker='o')
        plt.plot(range(len(y_test_pre)), y_test_pre, label='pred', marker='x')
        plt.xlabel('sample')
        plt.ylabel('target')
        plt.title(f'{accuracy}')
        plt.legend()
        plt.grid(True)  # 添加网格线
        plt.show()


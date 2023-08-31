from sklearn.datasets import load_wine  # wine数据集
from sklearn.datasets import load_iris  # iris数据集
from sklearn.model_selection import train_test_split  # 数据集划分
from sklearn.preprocessing import StandardScaler  # 标准差标准化
import numpy as np
import matplotlib.pyplot as plt
from Model import Model
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

class LR(Model):

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
        #留出法
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
           stdScale = StandardScaler().fit(data_train) #生成规则（建模）
           wine_trainScaler = stdScale.transform(data_train)#对训练集进行标准化
           wine_testScaler = stdScale.transform(data_test)#用训练集训练的模型对测试集标准化

           LRmodel = LinearRegression()
           LRmodel.fit(data_train, target_train)
           predictions = LRmodel.predict(data_test)
           f1 = r2_score(target_test, predictions)
           print('the f1 is', f1)  # 显示预测准确率
           return predictions,f1,data_train, data_test, target_train, target_test
        #交叉验证法
        if n==1:
           kf = KFold(30*test_size)
           for train_index, test_index in kf.split(data):
             data_train, data_test = data[train_index], data[test_index]
             target_train, target_test = target[train_index], target[test_index] 
           stdScale = StandardScaler().fit(data_train) #生成规则（建模）
           wine_trainScaler = stdScale.transform(data_train)#对训练集进行标准化
           wine_testScaler = stdScale.transform(data_test)#用训练集训练的模型对测试集标准化

           LRmodel = LinearRegression()
           LRmodel.fit(data_train, target_train)
           predictions = LRmodel.predict(data_test)
           f1 = r2_score(target_test, predictions)
           print('the f1 is', f1)  # 显示预测准确率
           return predictions,f1,data_train, data_test, target_train, target_test
        pass
    
    def train_data(self, data_train, target_train):
        """根据训练数据集X_train,y_train 训练Linear Regression模型"""
        assert data_train.shape[0] == target_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        # np.ones((len(X_train), 1)) 构造一个和X_train 同样行数的，只有一列的全是1的矩阵
        # np.hstack 拼接矩阵
        X_b = np.hstack([np.ones((len(data_train), 1)), data_train])
        # X_b.T 获取矩阵的转置
        # np.linalg.inv() 获取矩阵的逆
        # dot() 矩阵点乘
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(target_train)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self
        pass


    def test(self, dataname, n,size):
        predictions,f1,data_train, data_test, target_train, target_test=self.split(n,dataname,size)
       
        
        plt.scatter(target_test, predictions)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{f1}')
        plt.show()
        self.split(n,dataname,size)



#random
    

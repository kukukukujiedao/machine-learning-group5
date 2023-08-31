from sklearn import datasets
from sklearn.datasets import load_wine  # wine数据集
from sklearn.datasets import load_iris  # iris数据集
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from Model import Model
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Svm(Model):
    #  data=[]
    #  target=[]
     def load_data(self, dataname):
        # 在这里实现DecisionTree的数据加载逻辑
        if dataname=="wine":
            wine = load_wine()  #加载wine数据集
            data = wine['data']
            target = wine['target']
            return data,target
        if dataname=="iris":
            iris = load_iris()  # 加载wine数据集
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
           clf = svm.SVC(kernel='linear')
         # 使用训练数据拟合SVM分类器
           clf.fit(data_train, target_train)
         # 使用测试数据进行预测
           y_pred = clf.predict(data_test)
         #print('the predict values are', y_pred)  # 显示结果

        # 打印模型精度
           accuracy = metrics.accuracy_score(target_test, y_pred)
           print("Accuracy:", accuracy)
        
           cm = metrics.confusion_matrix(target_test, y_pred)
           return accuracy,cm,data_train, data_test, target_train, target_test
        if n==1:
           kf = KFold(test_size)
           for train_index, test_index in kf.split(data):
             data_train, data_test = data[train_index], data[test_index]
             target_train, target_test = target[train_index], target[test_index] 
           clf = svm.SVC(kernel='linear')
         # 使用训练数据拟合SVM分类器
           clf.fit(data_train, target_train)
         # 使用测试数据进行预测
           y_pred = clf.predict(data_test)
         #print('the predict values are', y_pred)  # 显示结果

        # 打印模型精度
           accuracy = metrics.accuracy_score(target_test, y_pred)
           print("Accuracy:", accuracy)
        
           cm = metrics.confusion_matrix(target_test, y_pred)
           return accuracy,cm,data_train, data_test, target_train, target_test
        pass
     
     def train_data(self, data_train, target_train):
        self.model = svm.SVC(kernel='linear')
        self.model.fit(data_train, target_train)
        pass
     
     def test(self, dataname, n,size):
         accuracy,cm,data_train, data_test, target_train, target_test=self.split(n,dataname,size)

    # Plot confusion matrix as a heatmap
         plt.figure(figsize=(8, 6))
         sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues")
         plt.xlabel("Predicted")
         plt.ylabel("Actual")
         plt.title(f'{accuracy}')
         plt.show()
         

#random

if __name__ == '__main__':
   svm_obj = Svm()

    # 加载数据集
   svm_obj.test("iris",0,0.2)

   






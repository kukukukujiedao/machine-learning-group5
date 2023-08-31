from Model import Model
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score
import matplotlib.pyplot as plt

#逻辑回归算法
class Logistic(Model):
    
    #加载数据集
    def load_data(self, dataname):
        if dataname == "iris":
            datas = load_iris()
            x= datas.data
            y= datas.target

        elif dataname == "wine":
            datas = load_wine()
            #放入DataFrame中便于查看
            x= datas.data
            y= datas.target

        #数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(x)
        return x, y, X_scaled

    #数据分割并训练
    #k折(k=6)
    def split_data_K_Fold(self, data, ObservationIndex):
        #加载数据
        x,y,X_scaled = self.load_data(data)
        #k折分割
        kfolds = KFold(n_splits=6)
        for train_index,test_index in kfolds.split(X_scaled):
            # 准备交叉验证的数据
            x_train_fold = X_scaled[train_index]
            y_train_fold = y[train_index]
            x_test_fold = X_scaled[test_index]
            y_test_fold = y[test_index]

            # 训练
            lgf = self.train_data(x_train_fold,y_train_fold)

            #评估
            tmp = self.Evaluations(lgf, x_test_fold, y_test_fold, ObservationIndex)
            printer = tmp
            return printer

    #random
    def split_data_Random(self, data, size, ObservationIndex):
        #加载数据
        x,y,X_scaled = self.load_data(data)
        #random分割
        x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=size, random_state=3)
        
        #训练
        lgf=self.train_data(x_train, y_train)
        
        #评估
        tmp = self.Evaluations(lgf, x_test, y_test, ObservationIndex)
        printer =  tmp
        return printer
    
    #训练模型
    def train_data(self, X_train, y_train):
        lg = LogisticRegression(max_iter=10000)
        lgf = lg.fit(X_train, y_train)
        return lgf

    #评估模型
    def Evaluations(self, model, x_test, y_test, ObservationIndex):
        y_pred = model.predict(x_test)
        if ObservationIndex == "acc":
            AccuracyScore = accuracy_score(y_test,y_pred)
            printer = AccuracyScore
            
        elif ObservationIndex == "f1":
            F1Score = f1_score(y_test,y_pred, average='micro')
            printer = F1Score

        return printer

    #测试模型
    def test(self, dataname, n, size, ObservationIndex):
        
        if n == 0:
            
            if ObservationIndex == "f1":
                return self.split_data_K_Fold(dataname, ObservationIndex)
        
        elif n == 1:
            
            if ObservationIndex == "f1":
                return self.split_data_Random(dataname, size, ObservationIndex)
            
    def plot_predictions(self, data, size, n):
        # 加载数据
        x, y, X_scaled = self.load_data(data)
        # random分割
        x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=size, random_state=3)
        
        # 训练模型
        lgf = self.train_data(x_train, y_train)
        
        # 预测结果
        y_pred = lgf.predict(x_test)
        
        # 绘制预测结果和真实标签的对比
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(y_test)), y_test, label='actual')
        plt.scatter(range(len(y_pred)), y_pred, label='pred')
        plt.plot(range(len(y_pred)), y_pred, 'r--', label='line')
        plt.xlabel('')
        plt.ylabel('')
        plt.title(f'{self.test(data, n, size,"f1")}')
        plt.legend()
        plt.show()
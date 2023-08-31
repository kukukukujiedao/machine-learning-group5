from Model import Model
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score,f1_score
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

class GradientBoosting(Model):

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
            gbrt = self.train_data(x_train_fold,y_train_fold)

            #评估
            tmp = self.Evaluations(gbrt, x_test_fold, y_test_fold, ObservationIndex)
            printer =  tmp
            return printer
    
    #random
    def split_data_Random(self, data, size, ObservationIndex):
        #加载数据
        x,y,X_scaled = self.load_data(data)
        #random分割
        x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=size, random_state=3)
        
        #训练
        gbrt=self.train_data(x_train, y_train)
        
        #评估
        tmp = self.Evaluations(gbrt, x_test, y_test, ObservationIndex)
        printer =  tmp
        return printer
    
    #训练模型
    def train_data(self, X_train, y_train):
        gbrt = GradientBoostingRegressor(learning_rate = 0.3, max_depth=2, n_estimators=3, random_state=42)
        gbrtf = gbrt.fit(X_train,y_train)
        return gbrtf

    #评估模型
    def Evaluations(self, model, x_test, y_test, ObservationIndex):
        y_pred = model.predict(x_test)
        #数据转int
        predictions = [round(value) for value in y_pred]
        if ObservationIndex == "acc":
            AccuracyScore = accuracy_score(y_test,predictions)
            printer = AccuracyScore
            
        elif ObservationIndex == "f1":
            F1Score = f1_score(y_test,predictions, average='micro')
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
    
    def plot_predictions(self, data, size,n):
        # 加载数据
        x, y, X_scaled = self.load_data(data)
        if n == 0:
        # random分割
         x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=size, random_state=3)
        
        # 训练模型
        gbrt = self.train_data(x_train, y_train)

        # 预测结果
        y_pred = gbrt.predict(x_test)

        # 数据转int
        predictions = [round(value) for value in y_pred]

        # 设置中文字体
        font = FontProperties(fname='C:\Windows\Fonts\simsunb.ttf')

        # 绘制预测结果和真实标签的对比
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(y_test)), y_test)
        plt.scatter(range(len(predictions)), predictions)
        plt.plot(range(len(predictions)), predictions, 'r--')
        plt.xlabel('', fontproperties=font)  # 使用中文字体
        plt.ylabel('', fontproperties=font)  # 使用中文字体
        plt.title(f'f1:{a.test(data, n, size, "f1")} ', fontproperties=font)  # 使用中文字体
        plt.legend(prop=font)  # 使用中文字体
        plt.show()

if __name__ == '__main__':
    '''# 1. 创建一个算法模型对象
    a = GradientBoosting()
    # 2. 调用模型对象的方法
    print(a.test("iris", 0, 0.5, "f1"))
    print(a.test("wine", 1, 0.7, "f1"))'''
    a = GradientBoosting()
    # 2. 调用模型对象的方法
    print(a.test("iris", 0, 0.5, "f1"))
    print(a.test("wine", 1, 0.7, "f1"))

    # 可视化预测结果和真实标签的对比
    a.plot_predictions("iris", 0.5,0)
    a.plot_predictions("wine", 0.7,1)
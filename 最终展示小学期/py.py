import eel
import Model
from LR import LR
from DecisionTree import DecisionTree
from Kmean import Kmean
from PCA import PCA
from knn import Knn
from randomforest import randomforest
from Svm import Svm
from GradientBoosting import GradientBoosting
from NBC import NaiveBayes
from Logistic import Logistic
eel.init('C:\最终展示小学期')

@eel.expose
def Lr(dataset,split,divider):
   LLL = LR()
   x = LLL.test(dataset,divider, split)


@eel.expose
def Decision(dataset,split,divider):
   DDD = DecisionTree()
   x = DDD.test(dataset,divider, split)
   acc,data_train, data_test, target_train, target_test=DDD.split(split,dataset,divider)
   DDD.plot_predictions(data_train, data_test, target_train, target_test)

@eel.expose
def kmean(dataset,split,divider):
   KKK = Kmean()
   x = KKK.test(dataset,divider, split)
   acc,data_train, data_test, target_train, target_test=KKK.split(split,dataset,divider)
   KKK.plot_predictions(data_train, data_test, target_train, target_test)

@eel.expose
def KNN(dataset,split,divider):
   KKK = Knn()
   data,target = KKK.load_data(dataset)
   data_train, data_test, target_train, target_test = KKK.split(divider,data, target, split)
   KKK.train_data(data_train, target_train)
   KKK.test(data_train, data_test, target_train, target_test,k=3)


@eel.expose
def pca(dataset,split,divider):
   PPP = PCA()
   x=PPP.test(dataset,divider, split)


@eel.expose
def randomf(dataset,split,divider):
   RRR = randomforest()
   x=RRR.test(dataset,divider, split)


@eel.expose
def svm(dataset,split,divider):
   SSS = Svm()
   x = SSS.test(dataset,divider, split)
   


@eel.expose
def boost(dataset,split,divider):
   BOO = GradientBoosting()
   x = BOO.test(dataset,divider, split, "f1")
   BOO.plot_predictions(dataset, split,divider)


@eel.expose
def nbc(dataset,split,divider):
   NNN = GradientBoosting()
   x = NNN.test(dataset,divider, split, "f1")
   NNN.plot_predictions(dataset, split,divider)

@eel.expose
def logi(dataset,split,divider):
   LLLL = Logistic()
   x = LLLL.test(dataset,divider, split, "f1")
   LLLL.plot_predictions(dataset, split,divider)


if __name__ == '__main__' :

  eel.start('html.html')
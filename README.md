# machine-learning-group5
## 一、成员分工
### （一）小组成员
- 组长：连航  前端页面设计与代码实现、前后端通信实现，后端架构设计，文档总和
- 组员：
- 薛泰阳 前后端通信代码编写，后端架构设计与代码实现，文档算法部分编辑
- 蒋世良宸 后端代码实现，文档第一部分编辑
- 殷泽楷 后端代码实现，文档第二部分编辑

## 二、项目总述 
程序首先在python端运行后会跳出机器学习项目实践前端页面
页面提供参数选择，参数选择包括经典数据集选择，数据分割比例选择，数据分割器选择，模型学习率选择，机器学习算法选择
再前端页面选择好各个参数后点击启动按钮，程序即可通过eel库中expose函数在JavaScript中调用后端设计的函数，完成后端函数相关参数的配置
从而完成模型训练和评估并通过python中绘图函数将机器算法结果展示在前端页面，项目完成展示。

## 三、算法原理介绍
### 1. 线性回归算法
线性回归是一种基本的统计学和机器学习算法，用于建立和分析变量之间线性关系的模型。它适用于解决回归问题，其中目标是预测一个连续的数值输出，基于一个或多个输入特征。
线性回归的基本思想是假设自变量（输入特征）和因变量（输出）之间存在一个线性关系，可以用一条直线来近似表示。这条直线被称为“回归线”，它的方程形式为：<br>
![image](https://github.com/kukukukujiedao/machine-learning-group5/assets/143621522/d4646a78-04c9-480a-aa7b-9340900df0a6)
其中：
y 是预测的因变量
x1​,x2​,…,xn​ 是自变量
b0​ 是截距，表示当所有自变量为0时的预测输出
b1​,b2​,…,bn​ 是自变量的系数，表示对应特征对输出的影响
线性回归的目标是找到合适的系数b0​,b1​,…,bn使得预测值与实际观测值之间的平方误差最小化
其中m是样本数量，yi是第i个样本的实际观测值，^yi是模型预测的值
### 2. 逻辑回归算法
逻辑回归是一种用于解决分类问题的统计学和机器学习算法，尽管名字中带有"回归"两个字，但实际上它用于分类任务，而不是回归任务。逻辑回归主要用于二分类问题，也可以通过一些扩展应用于多分类问题。
逻辑回归使用一种称为"逻辑函数"或"sigmoid函数"的特殊函数来建立输入特征和输出类别之间的关系。逻辑函数的数学表达式如下：<br>
![image](https://github.com/kukukukujiedao/machine-learning-group5/assets/143621522/3b4ceab3-fa88-404d-8cc9-7b0b9a9b8ee2)<br>
在这里,z 是输入特征的线性组合：<br>
![image](https://github.com/kukukukujiedao/machine-learning-group5/assets/143621522/17f3d14d-53f2-4b29-85a0-88090a37aa20)<br>
在逻辑回归模型中，b0,b1...bn是模型的权重，而x1,x2...xn是输入特征。逻辑函数将任何实数值映射到一个介于0和1之间的概率值，表示预测样本属于某一类别的概率。如果概率大于等于一个阈值（通常是0.5），则将样本分类为正类；否则，将其分类为负类。训练逻辑回归模型的目标是找到合适的权重，使得模型能够最好地拟合训练数据。这通常涉及最大化似然函数（或等价地，最小化对数似然损失），以便使模型的预测与实际观测值尽可能吻合。
### 3. 决策树算法
决策树是一种常用的机器学习算法，适用于分类和回归任务。它的工作原理类似于人们在做决策时提出一系列问题，每个问题都基于输入数据的特征，帮助将数据逐步划分成不同的类别或数值区间。这些问题构成了树状结构，树的每个节点代表一个问题，每个分支代表问题的不同答案，而叶节点则对应着最终的分类或回归结果。决策树从数据中学习出最佳的问题和划分方式，使其在预测新数据时能够高效地进行决策。它易于理解、可解释，还能够处理非线性关系和复杂的特征交互。
### 4. 朴素贝叶斯算法
朴素贝叶斯算法是一种基于贝叶斯定理的机器学习算法，用于分类和文本分类等任务。它假设各个特征之间相互独立，从而简化问题，尤其适用于文本数据。
算法的核心思想是根据输入特征的条件概率来进行分类。它会计算给定每个类别的情况下，输入特征的条件概率，然后选择具有最高概率的类别作为最终预测结果。朴素贝叶斯的训练过程涉及计算每个特征在每个类别下的概率，而预测过程则是基于这些概率进行决策。假设输入样本有n个特征，我们需要计算每个类别下的联合概率，即 P(类别 | 特征1, 特征2, …, 特征n)。根据贝叶斯定理，可以计算出后验概率 P(类别 | 特征1, 特征2, …, 特征n) = P(特征1, 特征2, …, 特征n | 类别) * P(类别) / P(特征1, 特征2, …, 特征n)。对于给定的输入样本，计算其属于每个类别的后验概率，并选择具有最高后验概率的类别作为预测结果。
在朴素贝叶斯模型中，假设所有特征之间是相互独立的，这是“朴素”的原因。虽然这个假设在现实中很少成立，但朴素贝叶斯模型仍然可以取得不错的分类效果，且计算效率高。
### 5. K最近邻算法
K最近邻算法（KNN）是一种常见的监督学习算法，用于分类和回归任务。它的工作原理简单直观。
KNN的基本想法是，根据输入数据的特征，找到与之最接近的训练数据点（最近邻），然后根据这些最近邻的类别或值来做出预测。这里的“K”代表着选择的最近邻数量，通常通过交叉验证等技术确定。在分类问题中，KNN通过投票机制来决定预测类别：如果K个最近邻中大多数属于某个类别，则预测为该类别。在回归问题中，KNN计算K个最近邻的平均值或加权平均值作为预测结果。KNN的优势在于简单易懂，适用于各种数据类型和复杂度不一的问题。然而，它对数据规模和特征空间的维度较为敏感，需要考虑适当的距离度量方法。
### 6. 支持向量机算法
支持向量机（SVM）是一种强大的监督学习算法，用于分类和回归任务。它的目标是在特征空间中找到一个能够最大化不同类别之间的边界距离的超平面，从而实现分类。SVM的关键思想是找到一个“最大间隔超平面”，即使不同类别的数据点与超平面的距离最大化。这样的超平面有助于在新数据点出现时，能更准确地进行分类。支持向量指的是离超平面最近的一些数据点，它们在分类决策中起着重要作用。在处理非线性问题时，SVM使用“核函数”将数据映射到高维空间，使其在高维空间中变得线性可分。常见的核函数包括线性核、多项式核和高斯核等。
SVM具有较强的泛化能力，适用于小样本和高维度数据，且对于噪声和过拟合具有一定的鲁棒性。  
### 7. 随机森林算法
随机森林是一种集成学习算法，用于解决分类和回归问题。它基于决策树构建而来，通过集成多个决策树的预测结果，从而提高模型的性能和泛化能力。
随机森林的核心思想是构建多个决策树，并在每个树的训练过程中引入两种随机性：随机抽取训练数据和随机选择特征。每个决策树都会根据随机抽取的样本和特征进行训练，从而降低过拟合风险。最终的预测结果由所有决策树的投票或平均得出。
随机森林具有良好的泛化能力：通过集成多个决策树，减少过拟合的可能性、能够处理大量特征：可以处理高维数据，无需进行特征选择等优点
由于随机森林能够有效地处理复杂的特征关系和噪声，且不需要太多的参数调整，因此在许多领域取得了很好的效果，如金融、医疗和自然语言处理等。
### 8. K均值聚类算法
K均值聚类算法是一种常用的无监督学习算法，用于将数据集中的样本划分为若干个类别（簇），使得同一类别内的样本相似度较高，而不同类别之间的相似度较低。
K均值算法的工作原理是通过迭代优化的方式找到K个中心点，然后将每个样本分配到最近的中心点所代表的类别。在每次迭代中，会更新中心点位置，使其成为同一类别内样本的平均值。迭代过程将持续进行，直到中心点不再变化或者达到预定的迭代次数。
K均值聚类的优势在于简单易懂，适用于大多数数据类型。它对于数据分布相对均匀、簇间距离明显的情况效果较好。然而，K值的选择对于聚类效果至关重要，不同的初始中心点选择可能导致不同的聚类结果。此外，K均值对于非球形簇、大小差异较大的簇和噪声敏感度较高。
在应用中，可以结合启发式方法或者轮廓系数等指标来帮助选择合适的K值，并使用多次运行取最佳结果。
### 9. 降维算法
降维模型用于减少高维数据集的维度，并保留最重要的特征。降维的目的是降低数据存储和计算复杂度，并提高模型的效率和泛化能力。常用的降维方法包括主成分分析（PCA）和线性判别分析（LDA）。在本项目中用的降维方法是主成分分析，主成分分析是一种无监督的降维技术，通过将原始数据投影到新的低维子空间来实现降维。
其算法原理：先计算原始数据的协方差矩阵，再对协方差矩阵进行特征值分解，得到特征值和对应的特征向量。然后根据特征值的大小排序特征向量，选取前k个特征向量作为主成分（k为降维后的目标维度）。最后将原始数据投影到所选的主成分上，得到降维后的数据。
PCA的关键思想是通过寻找协方差矩阵的主要特征向量，找到能够最好地解释原始数据变化的低维表示。通常情况下，选择的前几个主成分可以保留大部分的数据方差，从而更好地表示数据。
### 10. 梯度增强算法
梯度增强（Gradient Boosting）是一种强大的机器学习技术，用于解决分类和回归问题。它是集成学习的一种形式，通过组合多个弱学习器（通常是决策树）来构建一个更强大的模型。
梯度增强的基本思想是迭代地训练一系列弱学习器，每一轮都试图纠正前一轮的错误。它通过计算预测值与实际值之间的残差，然后让下一个弱学习器针对这个残差进行拟合。随着迭代的进行，每个弱学习器都在之前的弱学习器的基础上进行优化，从而逐步提升模型的预测性能。
梯度增强的优势在于能够处理复杂的非线性问题，且对于噪声和异常值有一定的鲁棒性。它通常能够获得比单个弱学习器更好的泛化性能。然而，梯度增强的参数调整较多，需要合适的学习率、迭代轮数和弱学习器类型等。
## 四、算法性能比较与分析

## 五、架构设计思路
### （一）前端架构
前端主要通过HTML、CSS、JavaScript三种语言进行编写。
- HTML   首先通过HTML完成对前端页面中根据设计需要完成每个板块盒子的设置，给出大致盒子叠加的关系，同时规划好整个网页的布局与分割。其次完成各个盒子内元素的设计，如文字，图片，单选框复选框等组件，合理根据前端页面内容填充盒子。需要注意的是，对于不同盒子或者元素标签的命名是需要考究的，以便后续CSS渲染过程中选择器的合理使用。
- CSS    CSS在前端设计中的作用是完成对HTML代码的美化。对于盒子设置合适的大小与边框，同时利用定位或流动等方式完成对盒子位置的摆放，使整体页面观感上整洁美观。对于盒子内元素，如字体大小颜色样式和段落间隔，照片的变形和选中时的浮动效果，单选框复选框样式的美化，都是通过CSS进行的。多数动态效果通过伪类选择器实现，checked和hover等伪类选择器可以很好的将某些组件在选中或停浮前后的动态效果展示出来。
- JavaScript   JavaScript在前端架构中主要有两个作用。首先是通过监听器函数和checked属性完成对图片和单选框的绑定，实现选中图片即选中单选框的效果。<br/>
  其次，JavaScript通过python中自带的eel库来完成对后端函数的调用。在JavaScript函数中，可以直接使用eel库暴露给我们的后端函数，从而可以通过对前端页面中选择框组件的分支判断语句来实现参数传递，从而后端可以根据前端页面用户所选的选项完成参数接受。后端完成模型训练和评估后，还可以在被JavaScript调用的情况下把评估结果完成展示。
  

### （二）前后端通信架构
> 在前后端通信方面我们选择使用eel这个python库。
> eel允许你使用Python构建简单的桌面应用程序，并通过使用Web前端技术（如HTML、CSS和JavaScript）在桌面应用程序中嵌入一个Web界面。Eel的目标是让开发者能够以Python为主要语言来创建跨平台的桌面应用，同时能够利用Web技术来实现动态和交互式的用户界面。它具有轻量级、简单易用的优点，还可以实现实时交互，可以在Python代码中调用JavaScript函数，反之亦然，从而实现Python和Web界面之间的实时交互。
基本用法就是将一个HTML文件嵌入到Python应用中。你可以在Python中定义一个函数，然后通过JavaScript来调用这个函数。<br/>
主要使用到的函数如下：<br/>
eel.expose(function, **kwargs)： 这个装饰器用于将Python函数暴露给前端 JavaScript。在前端代码中，你可以通过 eel.function_name() 来调用这个函数，并且它会返回一个 Promise 对象。<br/>
eel.init(path, allowed_extensions=[])： 这个函数用于初始化 Eel 应用。你需要提供 HTML 文件的路径，Eel 将会从这个路径加载你的前端界面。你还可以通过 allowed_extensions 参数设置允许的文件扩展名。<br/>
eel.start(html=None, options=None, suppress_error=True)： 这个函数用于启动 Eel 应用。你可以提供一个 HTML 文件的路径来指定初始页面，也可以在参数中设置一些选项。<br/>
其中在eel.start函数中使用了 WebSocket 技术来实现实时的双向通信，以便在服务器和客户端之间传递数据。在这个函数中，run_lambda() 函数启动了一个 WebSocket 服务器，允许用户端通过 WebSocket 连接与服务器进行实时通信。WebSocket 服务器的具体实现可能在 register_eel_routes() 函数中，从而确保 WebSocket 连接能够被正常处理。<br/>
简单来说eel让我们更简单的完成websocket所做的前后端通信的工作，使工作量和代码行数大大减少。<br/>
在本程序中，首先使用expose函数将十种算法后端代码暴露给JavaScript

### （三）后端架构
> 后端分为数据集，数据分割，训练和评估四个功能模块，模型模块设置父类，其余设为成员函数分别为load_data函数，Splitr函数，train_data函数、test函数、plot_predictions函数。  
#### 1. 数据集模块
-load_data函数里包含从sklearn.datasets中调用的load_wine()函数和load_iris()函数
通过形参dataname的值来调用不同的数据集，再通过赋值，返回data和target的值。
#### 2. 数据分割模块
若前端返回形参n=0，则该函数首先根据随机状态设置随机种子，再获取样本数量和测试集大小，然后随机选择测试集的索引，最后依据索引构建训练集和测试集，然后训练决策树模型并计算准确率。
若前端返回形参n=1，该函数首先进行m次放回抽样，得到训练集的序号，然后将剩下的序号记为测试集序号，最后产生训练/测试集，然后训练决策树模型并计算准确率。
#### 3. 模型模块
Model共有八个类，分别为DecisionTree类，SVM类，GBDT类，NB类，LinearRegression类，LogisticRegression类，KNN类和KMeans类。  
- DecisionTree类，SVM类，GBDT类，NBC类，Linear类，Logistic类，KNN类，KMeans类
    - 主要函数为fit函数和predict函数，其他函数根据这两个函数所需来写。
    - fit函数，即训练函数。该函数负责模型的训练。
    - predict函数，即预测函数。该函数负责模型的预测。
- randomForest类
    - 继承自DecisionTree类，主要函数为fit函数和predict函数，同时会用到DecisionTree类来构建所需的决策树。
- PCA类
    - LinearRegression类，LogisticRegression类，KNN类和KMeans类中调用了该类的transform函数。
- 当数据的维度大于4时，调用该函数对数据进行降维处理。
#### 4. 训练模块
通过train_data函数中的StandardScaler()标准化函数和fit_transform()函数进行数据的训练。
#### 5. 评估模块
   -test()函数：当前端返回的形参n为0时，调用split_data_Random()函数，从而求得留出法下，模型的准确率等数据；当前端返回的形参n为1时，调用split_data_K_Fold()函数，从而求得交叉验证法下，模型的准确率等数据；
   -plot_predictions（）函数：对模型以及数据进行可视化处理，更利于观察和分析。

## 六、前后端代码仓库链接
https://github.com/kukukukujiedao/machine-learning-group5

## 七、代码运行方式说明
运行code文件夹中的main.py文件，即下图中红框标记的文件  
![image](https://github.com/stupid-vegetable-bird/machine_learning_group4/assets/97822083/4fd61545-fa8e-47f3-922b-6c8af0d69ee0)

## 八、代码运行截图
![image](https://github.com/stupid-vegetable-bird/machine_learning_group4/assets/97822083/6a99b913-e810-4fcb-803c-154172db77cd)  
运行后的初始界面  

![image](https://github.com/stupid-vegetable-bird/machine_learning_group4/assets/97822083/38abdd5f-d904-49b6-bfaf-d9636eef7336)  
数据集、分割器。分割比例、模型、评估指标的选择及文字形式的运行结果  

![image](https://github.com/stupid-vegetable-bird/machine_learning_group4/assets/97822083/faf3f6a4-e90c-43f8-967c-412b173e39cd)  
图片形式的运行结果显示

## 九、每位组员的收获与感悟
### 1. 李青琪
经过小学期和暑假的学习，学到了手写机器学习模型代码、搭建前端和实现前后端通信的相关知识并加以实践，对项目开发以及团队合作开发有了进一步的了解与体会，对机器学习整个流程也有了更深入的认识。要手写机器学习模型代码就要去学习模型背后的算法的逻辑、了解并掌握相关的知识，在本次实践当中，如何参透算法的数学逻辑与内容并将其以代码的形式来更好地实现是我重点面对的问题。除此之外，在后端部分对代码进行整改也让我明白了良好的代码风格的重要性，以及在项目创建过程中如何与团队成员携手遵循共同且良好的代码风格。在前端和前后端通信部分，对websocket有了初步的学习与简单的实践，学到了前后端通信的知识与前后端通信的搭建。
### 2. 柴菲儿
在平台搭建的过程中，我负责了前端和各个模块的优化整改部分。因为对前端了解甚浅，所以最开始理解框架的时候花费了很多时间，学会了html、css和js等等，从而可以写出用于展示一个机器学习模型评估的界面。在前后端连接的过程中回顾了一些计算机网络的知识，学习了websocket接口和前后端的连接。这一部分以前没有接触过，这次将计算机网络的知识应用于实践，是一次很好的锻炼。在后端优化部分，我学会了使用机器学习模型算法工厂，从而实现模型的自动化和可扩展部署，也加深了对算法的理解。同时我们小组在写代码的过程中一直使用github， 使用Pull Request功能，从而促进了团队成员之间的代码审查、讨论和修改，提高了我们的合作能力。整个过程中我很感谢我的队友，我们会一起解决问题，整个过程中互相鼓励，让项目能够顺利地进行下去。
### 3. 张晨宇
在这个小学期的学习过程中，收获最大的是在小组合作的过程中进一步的提升了自己与成员沟通，正确地表达自己的想法的能力，这对于更好地完成学期任务以及生活中更好地相处起到了良好的促进作用。另一方面，本次的小学期也让我学习到了不少html上面的内容，也加强了对于python的学习。从算法实现到前端界面设计，再到前后端的连接，整个过程从熟悉到不熟悉，也克服了不少困难。很多时候，都多亏了小组内的成员们，我们在一起共同探讨，每个人发挥着自己所擅长的能力，使得这个项目的完成也较之顺利。
### 4. 吴  倩
在整个平台搭建的过程中，我主要负责了前后端连接和后端代码书写、调试的部分。在后端有对决策树和随机森林模型进行书写，之前使用模型大部分都是通过库和函数实现，通过这次实践，对模型进行手写，更好地了解了算法的原理和具体实现，在分割数据集和模型评估也有书写，最后进行了调试，更好地了解机器学习的整体过程。在前后端连接的过程中回顾了一些计算机网络的知识，学习了websocket的通信过程。实践过程中也复习了很多语法，收获满满。这次主要的收获在于前后端的搭建连接过程，这一部分以前没有接触过，对框架有了初步的了解。对于机器学习算法的研究我们还可以更加深入，比如数据集体量不够大，用户可以选择自行导入数据等等，都可以再次进行完善。在调试过程中也遇到了很多麻烦，比如前端限定模型算法选择的地方，更换了很多种方法才得以解决。整个过程中我很感谢我的队友，我们会一起解决问题，整个过程中互相鼓励，让项目能够顺利地进行下去。

## 十、其他重要的内容

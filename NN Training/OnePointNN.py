import numpy as np
# from AllPointTrainDataProcess import Get_LinkNum
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


# 加载数据
data = np.load('./TrainData/all_point_data_4.npz')

# 所有数据aa
# 维度说明：采样点个数*时间序列数*（Link_Num+1）
Data_ALL = data['all_point_data']
# print(Data_ALL.shape)

# 采样点个数
Point_Num = len(Data_ALL[:])
# 时间序列长度
Time_Num = len(Data_ALL[0, :])

# ----------------每次运行前注意预测长度是否修改-------------------------
def Get_LinkNum():
    return 4

point = 3951  #116.408 39.91608 天安门故宫附近
# point = 4946  #116.368 39.9956   北京科技大学附近
# point = 5675  #116.6 40.054   首都机场附近
point_list = np.array([[116.408, 39.916], [116.368, 39.996], [116.6, 40.054]])

# print(Data_ALL[point, :, :].shape, Data_ALL[point, :, :])

# point与经纬度转化关系
# 北京市经纬度边界
# bounds = [115.7, 39.4, 117.4, 41.6]
bounds = [116.0, 39.6, 116.8, 40.2]
MaxX = bounds[2]
MinX = bounds[0]
MaxY = bounds[3]
MinY = bounds[1]
# 坐标采样个数,即Sample_Num*Sample_Num
Sample_Num = 100
deltaX = (MaxX-MinX)/Sample_Num
deltaY = (MaxY-MinY)/Sample_Num

point_X = (point % 100) * deltaX + MinX
point_Y = (point / 100) * deltaX + MinY
print(point_X, point_Y)


Data_ALL_Trans = Data_ALL.copy()

# 对每个采样点进行归一化处理
# MinMaxScaler():
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min
min_max_scaler = preprocessing.MinMaxScaler()
# for i in range(0, Point_Num):
Data_ALL_Trans[point, :, :] = min_max_scaler.fit_transform(Data_ALL[point, :, :])

# 对所有数据进行归一化处理

# Data_ALL = Data_ALL.reshape(Point_Num*Time_Num, len(Data_ALL[0, 0, :]))
# min_max_scaler = preprocessing.MinMaxScaler()
# Data_ALL = min_max_scaler.fit_transform(Data_ALL)
# Data_ALL = np.expand_dims(Data_ALL, axis=0)
# Data_ALL = Data_ALL.reshape(Point_Num, Time_Num, len(Data_ALL[0, 0, :]))

# standard_scaler = preprocessing.StandardScaler() # 通过均值方差规范化
# for i in range(0, Point_Num):
#     standard_scaler.fit(Data_ALL[i, :, :])
#     Data_ALL[i, :, :] = standard_scaler.transform(Data_ALL[i, :, :])

# 数据还原
# min_max_scaler.inverse_transform(data)

X_data = Data_ALL_Trans[:, :, :Get_LinkNum()]
Y_data = Data_ALL_Trans[:, :, Get_LinkNum()]
# print(X_data.shape, Y_data.shape)

# 训练集测试集分类
Split = 700

X_train_data = X_data[point, :Split, :]
Y_train_data = Y_data[point, :Split]
X_test_data = X_data[point, Split:, :]
Y_test_data = Y_data[point, Split:]


# 多层感知器MLPRegressor参数说明
# solver=‘sgd',  MLP的求解方法：L-BFGS 在小数据上表现较好，Adam 较为鲁棒，SGD在参数调整较优时会有最佳表现（分类效果与迭代次数）；SGD标识随机梯度下降。
# alpha:L2的参数：MLP是可以支持正则化的，默认为L2，具体参数需要调整
# hidden_layer_sizes=(32, 1) hidden层2层,第一层32个神经元，第二层1个神经元)，2层隐藏层，也就有3层神经网络
# activation='relu'，激活函数
# max_iter=200，最大迭代次数

clf = MLPRegressor(solver='sgd', alpha=1e-4, learning_rate_init=0.01, hidden_layer_sizes=(18, 1), activation='relu', max_iter=1000, random_state=1)
clf.fit(X_train_data, Y_train_data)

y_train_hat = clf.predict(X_train_data)
y_test_hat = clf.predict(X_test_data)
score = clf.score(X_test_data, Y_test_data)
print("score:", score)

# 评价指标
# 计算训练集误差MSE
MSE_Train = mean_squared_error(y_train_hat, Y_train_data)
MAE_Train = mean_absolute_error(y_train_hat, Y_train_data)
EVS_Train = explained_variance_score(y_train_hat, Y_train_data)
R2_Train = r2_score(y_train_hat, Y_train_data)
# print("Train ERROR(MSE) = ", MSE_Train)
# print("Train MAE = ", MAE_Train)
# print("Train EVS = ", EVS_Train)
# print("Train R2 = ", R2_Train)

# 计算测试机误差MSE
MSE_Test = mean_squared_error(y_test_hat, Y_test_data)
MAE_Test = mean_absolute_error(y_test_hat, Y_test_data)
EVS_Test = explained_variance_score(y_test_hat, Y_test_data)
R2_Test = r2_score(y_test_hat, Y_test_data)
# print("Test ERROR(MSE) = ", MSE_Test)
# print("Test MAE = ", MAE_Test)
# print("Test EVS = ", EVS_Test)
# print("Test R2 = ", R2_Test)

# 打印最优系数
# cengindex = 0
# for wi in clf.coefs_:
#     cengindex += 1  # 表示底第几层神经网络。
#     print('第%d层网络层:' % cengindex)
#     print('权重矩阵维度:', wi.shape)
#     print('系数矩阵：\n', wi)

# 画loss曲线
# print(clf.loss_curve_, len(clf.loss_curve_))
# loss_x = range(len(clf.loss_curve_))
# plt.figure(1, figsize=(8, 6))
# plt.plot(loss_x, clf.loss_curve_, color="blue", label="Loss Curve", linewidth=2)
# plt.xlabel("Iteration")
# plt.ylabel("Loss")
# plt.title("Loss Curve of MLP")
# plt.legend()

# 归一化还原
test_data_trans = np.c_[X_test_data, y_test_hat]
train_data_trans = np.c_[X_train_data, y_train_hat]
data_all_trans = np.concatenate((train_data_trans, test_data_trans))

# 数据还原
data_all = min_max_scaler.inverse_transform(data_all_trans)
# print(data_all.shape, data_all)

y_raw = Data_ALL[point, :, Get_LinkNum()]
y_train_raw = Data_ALL[point, :len(X_train_data), Get_LinkNum()]
y_test_raw = Data_ALL[point, len(X_train_data):, Get_LinkNum()]

y_train_hat = data_all[:len(X_train_data), Get_LinkNum()]
y_test_hat = data_all[len(X_train_data):, Get_LinkNum()]

# 评价指标
# 训练集
# Mean squared error（均方误差）
MSE_Train = mean_squared_error(y_train_hat, y_train_raw)

# Mean absolute error（平均绝对误差）,给定数据点的平均绝对误差，一般来说取值越小，模型的拟合效果就越好。
MAE_Train = mean_absolute_error(y_train_hat, y_train_raw)

# explained_variance_score：解释方差分，这个指标用来衡量我们模型对数据集波动的解释程度，如果取值为1时，模型就完美，越小效果就越差。
EVS_Train = explained_variance_score(y_train_hat, y_train_raw)

# R2,R方可以理解为因变量y中的变异性能能够被估计的多元回归方程解释的比例，它衡量各个自变量对因变量变动的解释程度，其取值在0与1之间，其值越接近1，
# 则变量的解释程度就越高，其值越接近0，其解释程度就越弱。一般来说，增加自变量的个数，回归平方和会增加，残差平方和会减少，所以R方会增大；
# 反之，减少自变量的个数，回归平方和减少，残差平方和增加。
R2_Train = r2_score(y_train_hat, y_train_raw)
print("Train ERROR(MSE) = ", MSE_Train)
print("Train MAE = ", MAE_Train)
print("Train EVS = ", EVS_Train)
print("Train R2 = ", R2_Train)

# 计算测试集
MSE_Test = mean_squared_error(y_test_hat, y_test_raw)
MAE_Test = mean_absolute_error(y_test_hat, y_test_raw)
EVS_Test = explained_variance_score(y_test_hat, y_test_raw)
R2_Test = r2_score(y_test_hat, y_test_raw)
print("Test ERROR(MSE) = ", MSE_Test)
print("Test MAE = ", MAE_Test)
print("Test EVS = ", EVS_Test)
print("Test R2 = ", R2_Test)

# 用来正常显示中文标签,并设置字体大小
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 12

# 可视化拟合数据与真实结果对比
plt.figure(1, figsize=(9, 4), dpi=200)

x = range(0, len(y_raw))
plt.scatter(x, y_raw, color="blue", label="采样点", linewidth=3, s=0.5, alpha=0.4)

xx = range(0, len(y_train_raw))
plt.plot(xx, y_train_hat, color="green", label="拟合曲线", linewidth=1)

xxx = range(len(y_train_raw)+1, len(y_raw)+1)
plt.plot(xxx, y_test_hat, color="red", label="预测曲线", linewidth=1)

plt.xlabel("时间(×100s)")
plt.ylabel("出租车核密度值")
plt.xlim(0, len(y_raw)+1)
# title_name = "神经网络回归单点(经纬度:" + str(point_X) + "," + str(point_Y) + ")拟合效果与预测效果对比图"
title_name = "神经网络回归天安门故宫附近(经纬度:" + str(point_list[0, 0]) + "," + str(point_list[0, 1]) + ")拟合效果与预测效果对比图"
plt.title(title_name)
plt.legend()


# 图片存储在HotSpotPlot目录
# plt_name = './SVRPredictData/OnePoint_Predict_'+str(point_X)+"_"+str(point_Y)+"_"+str(int(clf.best_params_['C']))
plt_name = './NNPredictData/OnePoint_Predict_'+str(point_X)+"_"+str(point_Y)+"_"+"天安门"
plt.savefig(plt_name+'.png')
plt.savefig(plt_name+'.svg')

plt.show()


# # 可视化拟合数据与真实结果对比
# xx = range(0, len(y_train_raw))
# plt.figure(2, figsize=(8, 6))
# plt.scatter(xx, y_train_raw, color="red", label="Sample Point", linewidth=3, s=1)
# plt.plot(xx, y_train_hat, color="orange", label="Fitting Line", linewidth=2)
# plt.xlabel("Time")
# plt.ylabel("Value")
# plt.title("Train Fit Line of MLP")
# plt.legend()
#
# # 可视化预测结果与真实结果对比
# xx = range(0, len(y_test_raw))
# plt.figure(3, figsize=(8, 6))
# plt.scatter(xx, y_test_raw, color="red", label="Sample Point", linewidth=3, s=1)
# plt.plot(xx, y_test_hat, color="orange", label="Fitting Line", linewidth=2)
# plt.xlabel("Time")
# plt.ylabel("Value")
# # plt.ylim(0, 1)
# plt.title("Predict of MLP")
# plt.legend()
# plt.show()










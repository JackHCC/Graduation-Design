import numpy as np
# from AllPointTrainDataProcess import Get_LinkNum
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd


# 加载数据
data = np.load('./TrainData/all_point_data_5.npz')

# 所有数据
# 维度说明：采样点个数*时间序列数*（Link_Num+1）
Data_ALL = data['all_point_data']
print(Data_ALL.shape)

# ----------------每次运行前注意预测长度是否修改-------------------------
def Get_LinkNum():
    return 5

# 采样点个数
Point_Num = len(Data_ALL[:])
# 时间序列长度
Time_Num = len(Data_ALL[0, :])
# print(Point_Num)

# 取第一个Point进行预测
point = 5555


Data_ALL_Trans = Data_ALL.copy()

# 对每个采样点进行归一化处理
# MinMaxScaler():
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min
min_max_scaler = preprocessing.MinMaxScaler()
# for i in range(0, Point_Num):
Data_ALL_Trans[point, :, :] = min_max_scaler.fit_transform(Data_ALL[point, :, :])

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


# 评指指标汇总
MAE_ALL = []
MSE_ALL = []
# EVS_ALL = []
R2_ALL = []

# 三层网络神经元最优神经元测试隐藏层神经元个数对模型训练影响
for Hidden_Point1 in range(1, 30):
    for Hidden_Point2 in range(1, 30):
        clf = MLPRegressor(solver='sgd', alpha=1e-4, learning_rate_init=0.01, hidden_layer_sizes=(Hidden_Point1, Hidden_Point2, 1), activation='relu',
                       max_iter=1000, random_state=1).fit(X_train_data, Y_train_data)

        y_train_hat = clf.predict(X_train_data)
        y_test_hat = clf.predict(X_test_data)
        # score = clf.score(X_test_data, Y_test_data)
        # print("score:", score)

        # 归一化还原
        test_data_trans = np.c_[X_test_data, y_test_hat]
        train_data_trans = np.c_[X_train_data, y_train_hat]
        data_all_trans = np.concatenate((train_data_trans, test_data_trans))

        # 数据还原
        data_all = min_max_scaler.inverse_transform(data_all_trans)

        y_raw = Data_ALL[point, :, Get_LinkNum()]
        y_train_raw = Data_ALL[point, :len(X_train_data), Get_LinkNum()]
        y_test_raw = Data_ALL[point, len(X_train_data):, Get_LinkNum()]

        y_train_hat = data_all[:len(X_train_data), Get_LinkNum()]
        y_test_hat = data_all[len(X_train_data):, Get_LinkNum()]

        # 计算训练集
        # Mean squared error（均方误差）
        MSE_Train = mean_squared_error(y_train_hat, y_train_raw)
        # Mean absolute error（平均绝对误差）,给定数据点的平均绝对误差，一般来说取值越小，模型的拟合效果就越好。
        MAE_Train = mean_absolute_error(y_train_hat, y_train_raw)
        # explained_variance_score：解释方差分，这个指标用来衡量我们模型对数据集波动的解释程度，如果取值为1时，模型就完美，越小效果就越差。
        # EVS_Train = explained_variance_score(y_train_hat, y_train_raw)
        # R2,R方可以理解为因变量y中的变异性能能够被估计的多元回归方程解释的比例，它衡量各个自变量对因变量变动的解释程度，其取值在0与1之间，其值越接近1，
        # 则变量的解释程度就越高，其值越接近0，其解释程度就越弱。一般来说，增加自变量的个数，回归平方和会增加，残差平方和会减少，所以R方会增大；
        # 反之，减少自变量的个数，回归平方和减少，残差平方和增加。
        R2_Train = r2_score(y_train_hat, y_train_raw)

        # 计算测试集
        MSE_Test = mean_squared_error(y_test_hat, y_test_raw)
        MAE_Test = mean_absolute_error(y_test_hat, y_test_raw)
        # EVS_Test = explained_variance_score(y_test_hat, y_test_raw)
        # R2_Test = r2_score(y_test_hat, y_test_raw)
        R2_Test = clf.score(X_test_data, Y_test_data)

        # MSE_Train = mean_squared_error(y_train_hat, Y_train_data)
        # MAE_Train = mean_absolute_error(y_train_hat, Y_train_data)
        # EVS_Train = explained_variance_score(y_train_hat, Y_train_data)
        # R2_Train = r2_score(y_train_hat, Y_train_data)
        #
        # MSE_Test = mean_squared_error(y_test_hat, Y_test_data)
        # MAE_Test = mean_absolute_error(y_test_hat, Y_test_data)
        # EVS_Test = explained_variance_score(y_test_hat, Y_test_data)
        # R2_Test = r2_score(y_test_hat, Y_test_data)

        MAE_ALL = np.append(MAE_ALL, MAE_Test)
        MSE_ALL = np.append(MSE_ALL, MSE_Test)
        # EVS_ALL = np.append(EVS_ALL, EVS_Test)
        R2_ALL = np.append(R2_ALL, R2_Test)

        print("Hidden(1,2) = ", Hidden_Point1, Hidden_Point2, "|Test ERROR(MSE) = ", MSE_Test, "|Test MAE = ", MAE_Test, "|Test R2 = ", R2_Test)


# print(MAE_ALL, MSE_ALL)
# print("MAE_min_index:(Hidden_Point)", np.argmin(MAE_ALL)+1, "|MAE_min:", MAE_ALL.min())
Best_Hidden_Point = np.argmin(MSE_ALL)+1
print("MSE_min_index:(Hidden_Point)", Best_Hidden_Point, "|Hidden1:", Best_Hidden_Point//29+1, "|Hidden2:", Best_Hidden_Point%29, "|MSE_min:", MSE_ALL.min())
Best_Hidden_Point_R2 = np.argmax(R2_ALL)+1
print("R2_max_index:(Hidden_Point)", Best_Hidden_Point_R2, "|Hidden1:", Best_Hidden_Point_R2//29+1, "|Hidden2:", Best_Hidden_Point_R2%29, "|R2_max:", R2_ALL.max())

# 评价指标数据汇总
Sum_eva = np.r_[np.array(range(1, 29*29+1)).reshape(1, 29*29), MSE_ALL.reshape(1, len(MSE_ALL)), MAE_ALL.reshape(1, len(MAE_ALL)), R2_ALL.reshape(1, len(R2_ALL))]
# print(Sum_eva.shape, Sum_eva)

Sum_eva = pd.DataFrame(data=Sum_eva)
save_data_path = 'NNPredictData/Three_layer_MSE_MAE_R2.csv'
Sum_eva.to_csv(save_data_path, index=False, header=0)

# 最优Hidden_Point下回归
# clf = MLPRegressor(solver='sgd', alpha=1e-4, learning_rate_init=0.01, hidden_layer_sizes=(Best_Hidden_Point//29+1, Best_Hidden_Point%29, 1), activation='relu',
#                        max_iter=1000, random_state=1).fit(X_train_data, Y_train_data)
# y_test_hat = clf.predict(X_test_data)
#
# # 可视化预测结果与真实结果对比
# xx = range(0, len(Y_test_data))
# plt.figure(1, figsize=(8, 6))
# plt.scatter(xx, Y_test_data, color="red", label="Sample Point", linewidth=3)
# plt.plot(xx, y_test_hat, color="orange", label="Fitting Line", linewidth=2)
# plt.xlabel("Time")
# plt.ylabel("Value")
# plt.title("Predict of MLP(Best Hidden Point Num)")
# plt.legend()
# plt.show()




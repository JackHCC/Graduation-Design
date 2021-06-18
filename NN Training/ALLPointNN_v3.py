import numpy as np
# from AllPointTrainDataProcess import Get_LinkNum
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import time
import pandas as pd

# 加载数据
data = np.load('./TrainData/all_point_data_4.npz')

# 所有数据aa
# 维度说明：采样点个数*时间序列数*（Link_Num+1）
Data_ALL = data['all_point_data']
print(Data_ALL.shape)

# ----------------每次运行前注意预测长度是否修改-------------------------
def Get_LinkNum():
    return 4

# 采样点个数
Point_Num = len(Data_ALL[:])
# 时间序列长度
Time_Num = len(Data_ALL[0, :])

# 训练集测试集分类
Split = 700

# print(Data_ALL[point, :, :].shape, Data_ALL[point, :, :])

Data_ALL_Trans = Data_ALL.copy()
Data_ALL_Test = Data_ALL.copy()

# # 取第一个Point进行预测
# point = 5555

# 评价指标数组，包含每个Point的预测指标
EVS_ALL = []
MAE_ALL = []
MSE_ALL = []
R2_ALL = []

Best_Hidden_ALL = []

start = time.process_time()

for point in range(Point_Num):

    # 进度
    print(point, '/9999')

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
    MAE_Select = []
    for Hidden_Point1 in range(1, 30):
        for Hidden_Point2 in range(1, 30):
            clf = MLPRegressor(solver='sgd', alpha=1e-4, learning_rate_init=0.01, hidden_layer_sizes=(Hidden_Point1, Hidden_Point2, 1), activation='relu',
                       max_iter=1000, random_state=1)
            clf.fit(X_train_data, Y_train_data)

            # 训练数据拟合
            y_train_hat = clf.predict(X_train_data)
            # 模型测试预测
            y_test_hat = clf.predict(X_test_data)
            # y_hat = clf.predict(X_train_data)

            # 选取最优的Hidden_Num
            MAE_Test = mean_absolute_error(y_test_hat, Y_test_data)
            MAE_Select = np.append(MAE_Select, MAE_Test)

    # 最佳神经元个数
    Best_Hidden_Point_MAE = np.argmin(MAE_Select) + 1
    Best_Hidden_1 = Best_Hidden_Point_MAE//29+1
    Best_Hidden_2 = Best_Hidden_Point_MAE % 29

    clf = MLPRegressor(solver='sgd', alpha=1e-4, learning_rate_init=0.01, hidden_layer_sizes=(Best_Hidden_1,
                                                                                              Best_Hidden_2, 1),
                       activation='relu',
                       max_iter=1000, random_state=1)
    clf.fit(X_train_data, Y_train_data)

    # 训练数据拟合
    y_train_hat = clf.predict(X_train_data)
    # 模型测试预测
    y_test_hat = clf.predict(X_test_data)


    # 归一化还原
    test_data_trans = np.c_[X_test_data, y_test_hat]
    train_data_trans = np.c_[X_train_data, y_train_hat]
    data_all_trans = np.concatenate((train_data_trans, test_data_trans))

    # 数据还原
    data_all = min_max_scaler.inverse_transform(data_all_trans)
    # print(data_all.shape, data_all)
    Data_ALL_Test[point] =data_all

    # 计算评价指标提取
    y_train_raw = Data_ALL[point, :len(X_train_data), Get_LinkNum()]
    y_test_raw = Data_ALL[point, len(X_train_data):800, Get_LinkNum()]
    y_train_hat = data_all[:len(X_train_data), Get_LinkNum()]
    y_test_hat = data_all[len(X_train_data):800, Get_LinkNum()]

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
    # print("Train ERROR(MSE) = ", MSE_Train)
    # print("Train MAE = ", MAE_Train)
    # print("Train EVS = ", EVS_Train)
    # print("Train R2 = ", R2_Train)

    # 计算测试集
    MSE_Test = mean_squared_error(y_test_hat, y_test_raw)
    MAE_Test = mean_absolute_error(y_test_hat, y_test_raw)
    EVS_Test = explained_variance_score(y_test_hat, y_test_raw)
    R2_Test = r2_score(y_test_hat, y_test_raw)
    MSE_ALL = np.append(MSE_ALL, MSE_Test)
    MAE_ALL = np.append(MAE_ALL, MAE_Test)
    EVS_ALL = np.append(EVS_ALL, EVS_Test)
    R2_ALL = np.append(R2_ALL, R2_Test)
    # print("Test ERROR(MSE) = ", MSE_Test)
    # print("Test MAE = ", MAE_Test)
    # print("Test EVS = ", EVS_Test)
    # print("Test R2 = ", R2_Test)

end = time.process_time()
print("Data Process 4 Step time:", end-start)

# DataALLTest异常点处理,小于0的值按0计算
Data_ALL_Test[Data_ALL_Test < 0] = 0

# print(Data_ALL.shape, Data_ALL_Test.shape)

# 异常值剔除
EVS_ALL = [element for element in EVS_ALL if element >= 0]
R2_ALL = [element for element in R2_ALL if element >= 0]
# MSE_ALL = filter(lambda a: a <= np.mean(MSE_ALL), MSE_ALL)
# MAE_ALL = filter(lambda a: a <= np.mean(MAE_ALL), MAE_ALL)


print("EVS:", np.mean(EVS_ALL))
print("MAE:", np.mean(MAE_ALL))
print("MSE:", np.mean(MSE_ALL))
print("R2 Score:", np.mean(R2_ALL))

# 评价指标数据汇总
Sum_eva = np.array([[Get_LinkNum(),
                    np.mean(MSE_ALL), np.max(MSE_ALL), np.min(MSE_ALL),
                    np.mean(MAE_ALL), np.max(MAE_ALL), np.min(MAE_ALL),
                    np.mean(EVS_ALL), np.max(EVS_ALL), np.min(EVS_ALL),
                    np.mean(R2_ALL), np.max(R2_ALL), np.min(R2_ALL),
                    end-start]])

Sum_eva = pd.DataFrame(data=Sum_eva)
save_data_path = 'NNPredictData/Sum_Params_v2.csv'
Sum_eva.to_csv(save_data_path, index=False, mode='a', header=0)

save_name = './NNPredictData/all_y_hat_nn_' + str(Get_LinkNum())
np.savez(save_name + '.npz', all_y_hat=Data_ALL_Test)

# 画loss曲线
# print(clf.loss_curve_, len(clf.loss_curve_))
# loss_x = range(len(clf.loss_curve_))
# plt.figure(1, figsize=(8, 6))
# plt.plot(loss_x, clf.loss_curve_, color="blue", label="Loss Curve", linewidth=2)
# plt.xlabel("Iteration")
# plt.ylabel("Loss")
# plt.title("Loss Curve of MLP")
# plt.legend()

# 可视化预测结果与真实结果对比
# xx = range(0, len(Y_test_data))
# plt.figure(2, figsize=(8, 6))
# plt.scatter(xx, Y_test_data, color="red", label="Sample Point", linewidth=3)
# plt.plot(xx, y_test_hat, color="orange", label="Fitting Line", linewidth=2)
# plt.xlabel("Time")
# plt.ylabel("Value")
# plt.title("Predict of MLP")
# plt.legend()
# plt.show()







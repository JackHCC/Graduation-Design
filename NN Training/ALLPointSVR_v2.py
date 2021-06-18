import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import time
import pandas as pd


# 加载数据
data = np.load('./TrainData/all_point_data_2.npz')

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
    return 2

# 最优化C
C = []

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

    # SVR预测
    # SVR参数说明
    # kernel:指定要在算法中使用的内核类型，它必须是 'linear'，'poly'，'rbf'， 'sigmoid'，'precomputed' ，‘callable’。
    # degree：int，可选（默认= 3）多项式核函数的次数（'poly'），被所有其他内核忽略。
    # gamma：float，（默认='auto'）,'rbf'，'poly' 和 'sigmoid' 的核系数。当前默认值为 'auto'，它使用1 / n_features。
    # tol：float，（默认值= 1e-3）容忍停止标准。
    # C ：float，可选（默认= 1.0）错误术语的惩罚参数C，正则化参数，越小正则化越强。
    # shrinking ： 布尔值，可选（默认 = True）是否使用收缩启发式。
    # cache_size ： float，可选，指定内核缓存的大小（以MB为单位）。
    # max_iter ： int，optional（默认值= -1） 求解器内迭代的硬限制，或无限制的-1。

    clf = SVR(kernel='linear', C=10)
    clf.fit(X_train_data, Y_train_data)

    # 最优参数寻找 GridSearchCV,适合数据量不大
    # param_grid = {'C': np.linspace(1, 1000, 10)}
    # clf = GridSearchCV(SVR(kernel='linear'), param_grid, cv=10)
    #
    # clf.fit(X_train_data, Y_train_data)
    # C = np.append(C, clf.best_params_)
    # print("best param: {0}\nbest score: {1}".format(clf.best_params_, clf.best_score_))

    # 训练数据拟合
    y_train_hat = clf.predict(X_train_data)

    # 模型测试预测
    y_test_hat = clf.predict(X_test_data)
    # y_hat = clf.predict(X_train_data)

    # # 评价指标
    # # 训练集
    # # Mean squared error（均方误差）
    # MSE_Train = mean_squared_error(y_train_hat, Y_train_data)
    #
    # # Mean absolute error（平均绝对误差）,给定数据点的平均绝对误差，一般来说取值越小，模型的拟合效果就越好。
    # MAE_Train = mean_absolute_error(y_train_hat, Y_train_data)
    #
    # # explained_variance_score：解释方差分，这个指标用来衡量我们模型对数据集波动的解释程度，如果取值为1时，模型就完美，越小效果就越差。
    # EVS_Train = explained_variance_score(y_train_hat, Y_train_data)
    #
    # # R2,R方可以理解为因变量y中的变异性能能够被估计的多元回归方程解释的比例，它衡量各个自变量对因变量变动的解释程度，其取值在0与1之间，其值越接近1，
    # # 则变量的解释程度就越高，其值越接近0，其解释程度就越弱。一般来说，增加自变量的个数，回归平方和会增加，残差平方和会减少，所以R方会增大；
    # # 反之，减少自变量的个数，回归平方和减少，残差平方和增加。
    # R2_Train = r2_score(y_train_hat, Y_train_data)
    # print("Train ERROR(MSE) = ", MSE_Train)
    # print("Train MAE = ", MAE_Train)
    # print("Train EVS = ", EVS_Train)
    # print("Train R2 = ", R2_Train)
    #
    # # 计算测试集
    # MSE_Test = mean_squared_error(y_test_hat, Y_test_data)
    # MAE_Test = mean_absolute_error(y_test_hat, Y_test_data)
    # EVS_Test = explained_variance_score(y_test_hat, Y_test_data)
    # R2_Test = r2_score(y_test_hat, Y_test_data)
    # print("Test ERROR(MSE) = ", MSE_Test)
    # print("Test MAE = ", MAE_Test)
    # print("Test EVS = ", EVS_Test)
    # print("Test R2 = ", R2_Test)


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

print(Data_ALL.shape, Data_ALL_Test.shape)

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
                    # np.mean(C), np.max(C), np.min(C),
                    end-start]])

Sum_eva = pd.DataFrame(data=Sum_eva)
save_data_path = 'SVRPredictData/Sum_Params_C10.csv'
Sum_eva.to_csv(save_data_path, index=False, mode='a', header=0)

save_name = './SVRPredictData/all_y_hat_svr_' + str(Get_LinkNum())
np.savez(save_name + '.npz', all_y_hat=Data_ALL_Test)

# # 可视化拟合数据与真实结果对比
# xx = range(0, len(y_train_raw))
# plt.figure(2, figsize=(8, 6))
# plt.scatter(xx, y_train_raw, color="red", label="Sample Point", linewidth=3, s=1)
# plt.plot(xx, y_train_hat, color="orange", label="Fitting Line", linewidth=2)
# plt.xlabel("Time")
# plt.ylabel("Value")
# plt.title("Train Fit Line of SVR")
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
# plt.title("Predict of SVR")
# plt.legend()
# plt.show()


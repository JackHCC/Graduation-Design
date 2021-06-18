import numpy as np
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KernelDensity
import pandas as pd
import math
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from SecondtoTime import sec2time

# 最佳窗口长度 3
# 加载数据
data = np.load('./TrainData/all_point_data_3.npz')

# 所有数据aa
# 维度说明：采样点个数*时间序列数*（Link_Num+1）
Data_ALL = data['all_point_data']
# print(Data_ALL.shape)



def Get_LinkNum():
    return 3

# 采样点个数
Point_Num = len(Data_ALL[:])
# 时间序列长度
Time_Num = len(Data_ALL[0, :])

# 训练集测试集分类
Split = 700

Data_ALL_Trans = Data_ALL.copy()
Data_ALL_Test = Data_ALL.copy()
# point = 5555


for point in range(0, Point_Num):
    # 进度
    print(point, "/9999")
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
    # X_test_data = X_data[point, Split:, :]
    # Y_test_data = Y_data[point, Split:]
    # print(X_train_data.shape, Y_train_data.shape)

    # clf = MLPRegressor(solver='sgd', alpha=1e-4, learning_rate_init=0.01, hidden_layer_sizes=(19, 1), activation='relu',
    #                        max_iter=1000, random_state=1).fit(X_train_data, Y_train_data)

    # clf = SVR(kernel='linear', C=10).fit(X_train_data, Y_train_data)
    # clf = SVR(kernel='rbf', C=10).fit(X_train_data, Y_train_data)

    clf = MLPRegressor(solver='sgd', alpha=1e-4, learning_rate_init=0.01, hidden_layer_sizes=(18, 1),
                       activation='relu',
                       max_iter=1000, random_state=1).fit(X_train_data, Y_train_data)

    Window = Y_train_data[-Get_LinkNum():]
    # print(Window)
    all_y_hat = []
    Window_ALL = []

    for i in range(0, Time_Num-Split):
        New_predict_time = clf.predict(Window.reshape(1, -1))

        # print(New_predict_time)

        # X_train_data = np.concatenate((X_train_data, Window.reshape(1, -1)))
        # Y_train_data = np.append(Y_train_data, New_predict_time.reshape(1, -1))
        # print(New_predict_time[0])

        # print(X_train_data.shape, Y_train_data.shape)

        all_y_hat = np.append(all_y_hat, New_predict_time[0])
        Window_ALL = np.append(Window_ALL, Window)
        Window = np.delete(Window, 0)
        Window = np.append(Window, New_predict_time)

    Window_ALL = np.reshape(Window_ALL, [Time_Num-Split, Get_LinkNum()])

    test_data_trans = np.c_[Window_ALL, all_y_hat.reshape(-1, 1)]
    data_all_trans = np.concatenate((Data_ALL_Trans[point, :Split, :], test_data_trans))

    # 数据还原
    data_all = min_max_scaler.inverse_transform(data_all_trans)
    # print(data_all.shape, data_all)
    Data_ALL_Test[point] =data_all

    # # 计算评价指标提取
    # # y_train_raw = Data_ALL[point, :Split, Get_LinkNum()]
    # y_test_raw = Data_ALL[point, Split:, Get_LinkNum()]
    # # y_train_hat = data_all[:Split, Get_LinkNum()]
    # y_test_hat = data_all[Split:, Get_LinkNum()]
    #
    # # 计算测试集
    # MSE_Test = mean_squared_error(y_test_hat, y_test_raw)
    # MAE_Test = mean_absolute_error(y_test_hat, y_test_raw)
    # MSE_ALL = np.append(MSE_ALL, MSE_Test)
    # MAE_ALL = np.append(MAE_ALL, MAE_Test)


print(Data_ALL_Test.shape)


MAE_ALL = []
MSE_ALL = []

# 将时间尺度的MAE---> 空间尺度的MAE
for Time in range(Split, Time_Num):
    y_test_pred = Data_ALL_Test[:, Time, Get_LinkNum()]
    y_test_raw = Data_ALL[:, Time, Get_LinkNum()]
    # print(y_test_pred.shape, y_test_raw.shape)
    # print(Time)

    # 计算测试集
    MSE_Test = mean_squared_error(y_test_pred, y_test_raw)
    MAE_Test = mean_absolute_error(y_test_pred, y_test_raw)
    MSE_ALL = np.append(MSE_ALL, MSE_Test)
    MAE_ALL = np.append(MAE_ALL, MAE_Test)

print(MAE_ALL)
MAE_MSE_ALL = [MAE_ALL, MSE_ALL]

Sum_eva = pd.DataFrame(data=MAE_MSE_ALL)
save_data_path = 'SVRPredictData/Sum_Params_Time_Queue.csv'
Sum_eva.to_csv(save_data_path, index=False, mode='a', header=0)

# # 用来正常显示中文标签,并设置字体大小
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['font.size'] = 12
#
# # 可视化拟合数据与真实结果对比
# plt.figure(1, figsize=(9, 4), dpi=200)
#
# x = range(0, len(y_raw))
# plt.scatter(x, y_raw, color="blue", label="采样点", linewidth=3, s=0.5)
#
# xx = range(0, len(y_train_raw))
# plt.plot(xx, y_train_hat, color="green", label="拟合曲线", linewidth=1)
#
# xxx = range(len(y_train_raw)+1, len(y_raw)+1)
# plt.plot(xxx, y_test_hat, color="red", label="预测曲线", linewidth=1)
#
# plt.xlabel("时间(×100s)")
# plt.ylabel("核密度估计值")
# plt.xlim(0, len(y_raw)+1)
# title_name = "支持向量回归单点(经纬度:" + str(point_X) + "," + str(point_Y) + ")拟合效果与预测效果对比图"
# plt.title(title_name)
# plt.legend()

# # 加载真实数据
# # 数据读取time：57~58s
# data = pd.read_csv(r'../Data Process/RawGPSData.csv', header=None)
# data.columns = ['VehicleNum', 'Stime', 'Lng', 'Lat']
#
# Link_Num = 3
#
# # 北京市经纬度边界
# # bounds = [115.7, 39.4, 117.4, 41.6]
# bounds = [116.0, 39.6, 116.8, 40.2]
# MaxX = bounds[2]
# MinX = bounds[0]
# MaxY = bounds[3]
# MinY = bounds[1]
#
# # 坐标采样个数,即Sample_Num*Sample_Num
# Sample_Num = 100
#
# # 计算采样间隔
# Sample_Space_X = (MaxX - MinX) / Sample_Num
# Sample_Space_Y = (MaxY - MinY) / Sample_Num
#
# # 采样矩阵，Sample_Num*Sample_Num
# Sample_X, Sample_Y = np.mgrid[MinX:MaxX:complex(0, Sample_Num), MinY:MaxY:complex(0, Sample_Num)]  # 步长实数为间隔，虚数为个数
# # Sample_X, Sample_Y = np.mgrid[MinX:MaxX:Sample_Space_X, MinY:MaxY:Sample_Space_Y]  # 步长实数为间隔，虚数为个数
# Sample_Metric = np.transpose(np.vstack((Sample_X.ravel(), Sample_Y.ravel())))
# # print(Sample_Metric)
# # print(Sample_Metric.shape)
#
# # 计算地球经纬度与公里数转换 1度多少千米
# deltaLng_km = (MaxX-MinX)*(2 * math.pi * 6371 * math.cos((MinY + MaxY) * math.pi/360))/360
# deltaLat_km = (MaxY-MinY)*(2 * math.pi * 6371)/360
# # print(deltaLng_km, deltaLat_km)
#
# # 计算最佳核密度估计最佳带宽
# # Method1:
# # bandwidth1 = np.power(0.68 * Point_Num, -0.2) * np.sqrt((MaxX - MinX) * (MaxY - MinY))/(np.sqrt(deltaLng_km*deltaLat_km))
# bandwidth1 = 0.5/np.sqrt(deltaLng_km*deltaLat_km)
# # bandwidth1 = np.power(0.68*Point_Num, -0.2)*np.sqrt((MaxX-MinX)*(MaxY-MinY))/(np.sqrt(111*55))
# # print("Best Bandwidth:", bandwidth1)
# # Method2:计算中心位置，计算距离方差和中位数
#
#
# # 计算对比的预测时间点（70000-86400）
# test_sec = 72000
#
# MAE_ALL = []
# MSE_ALL = []
#
# for test_sec in range(70100, 82800, 100):
#
#     dense_predict = ALL_y_hat[:, int(test_sec/100), Link_Num]
#     # print(dense_predict.shape)
#
#     data_10 = data[data['Stime'] == test_sec]
#
#     Point_Num = len(data_10)
#     X = np.array(data_10[['Lng', 'Lat']])
#
#     # KDEFunction
#     # kde, dense = KDEFunction.KDE(X, Sample_Metric, bandwidth1)
#     kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth1).fit(X)
#     log_dense = kde.score_samples(Sample_Metric)
#     dense_real = np.exp(log_dense)
#     # print(dense_real.shape)
#
#     MSE = mean_squared_error(dense_predict, dense_real)
#     MAE = mean_absolute_error(dense_predict, dense_real)
#     MSE_ALL = np.append(MSE_ALL, MSE)
#     MAE_ALL = np.append(MAE_ALL, MAE)
#     # print("ERROR(MSE) = ", MSE)
#     # print("MAE = ", MAE)
#
#     # 进度
#     print(test_sec, "/82800")
#
# print(MSE_ALL)
# MSE_Time = pd.DataFrame(data=MSE_ALL.reshape(1, len(MSE_ALL)))
# MAE_Time = pd.DataFrame(data=MAE_ALL.reshape(1, len(MAE_ALL)))
# save_data_path = 'SVRPredictData/MSE_Time_Window_3.csv'
# save_data_path2 = 'SVRPredictData/MAE_Time_Window_3.csv'
# MSE_Time.to_csv(save_data_path, index=False, mode='a', header=0)
# MAE_Time.to_csv(save_data_path2, index=False, mode='a', header=0)

# plt.figure(dpi=300)
# plt.plot(range(701, 828), MSE_ALL, color='r', marker='o', linewidth=1, markeredgecolor='b', markersize='2',
#          markeredgewidth=1)
#
# # ------------------------------------------------------------------------------------------------------------------
# plt.xlabel()


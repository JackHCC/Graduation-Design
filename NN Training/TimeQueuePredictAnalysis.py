import numpy as np
from sklearn.neighbors import KernelDensity
import pandas as pd
import math
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from SecondtoTime import sec2time

# 最佳窗口长度 3
# data = np.load('./SVRPredictData/all_y_hat_svr_3.npz')
# data = np.load('./NNPredictData/all_y_hat_nn_4.npz')
data = np.load('./NNPredictData/all_y_hat_lstm_3.npz')
ALL_y_hat = data['all_y_hat']

# 加载真实数据
# 数据读取time：57~58s
data = pd.read_csv(r'../Data Process/RawGPSData.csv', header=None)
data.columns = ['VehicleNum', 'Stime', 'Lng', 'Lat']

Link_Num = 3

# 北京市经纬度边界
# bounds = [115.7, 39.4, 117.4, 41.6]
bounds = [116.0, 39.6, 116.8, 40.2]
MaxX = bounds[2]
MinX = bounds[0]
MaxY = bounds[3]
MinY = bounds[1]

# 坐标采样个数,即Sample_Num*Sample_Num
Sample_Num = 100

# 计算采样间隔
Sample_Space_X = (MaxX - MinX) / Sample_Num
Sample_Space_Y = (MaxY - MinY) / Sample_Num

# 采样矩阵，Sample_Num*Sample_Num
Sample_X, Sample_Y = np.mgrid[MinX:MaxX:complex(0, Sample_Num), MinY:MaxY:complex(0, Sample_Num)]  # 步长实数为间隔，虚数为个数
# Sample_X, Sample_Y = np.mgrid[MinX:MaxX:Sample_Space_X, MinY:MaxY:Sample_Space_Y]  # 步长实数为间隔，虚数为个数
Sample_Metric = np.transpose(np.vstack((Sample_X.ravel(), Sample_Y.ravel())))
# print(Sample_Metric)
# print(Sample_Metric.shape)

# 计算地球经纬度与公里数转换 1度多少千米
deltaLng_km = (MaxX-MinX)*(2 * math.pi * 6371 * math.cos((MinY + MaxY) * math.pi/360))/360
deltaLat_km = (MaxY-MinY)*(2 * math.pi * 6371)/360
# print(deltaLng_km, deltaLat_km)

# 计算最佳核密度估计最佳带宽
# Method1:
# bandwidth1 = np.power(0.68 * Point_Num, -0.2) * np.sqrt((MaxX - MinX) * (MaxY - MinY))/(np.sqrt(deltaLng_km*deltaLat_km))
bandwidth1 = 0.5/np.sqrt(deltaLng_km*deltaLat_km)
# bandwidth1 = np.power(0.68*Point_Num, -0.2)*np.sqrt((MaxX-MinX)*(MaxY-MinY))/(np.sqrt(111*55))
# print("Best Bandwidth:", bandwidth1)
# Method2:计算中心位置，计算距离方差和中位数


# 计算对比的预测时间点（70000-86400）
test_sec = 72000

MAE_ALL = []
MSE_ALL = []
R2_ALL = []

for test_sec in range(70100, 82800, 100):

    dense_predict = ALL_y_hat[:, int(test_sec/100), Link_Num]
    # print(dense_predict.shape)

    data_10 = data[data['Stime'] == test_sec]

    Point_Num = len(data_10)
    X = np.array(data_10[['Lng', 'Lat']])

    # KDEFunction
    # kde, dense = KDEFunction.KDE(X, Sample_Metric, bandwidth1)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth1).fit(X)
    log_dense = kde.score_samples(Sample_Metric)
    dense_real = np.exp(log_dense)
    # print(dense_real.shape)

    MSE = mean_squared_error(dense_predict, dense_real)
    MAE = mean_absolute_error(dense_predict, dense_real)
    R2 = r2_score(dense_predict, dense_real)
    MSE_ALL = np.append(MSE_ALL, MSE)
    MAE_ALL = np.append(MAE_ALL, MAE)
    R2_ALL = np.append(R2_ALL, R2)
    # print("ERROR(MSE) = ", MSE)
    # print("MAE = ", MAE)

    # 进度
    print(test_sec, "/82800")

print(MSE_ALL)
MSE_Time = pd.DataFrame(data=MSE_ALL.reshape(1, len(MSE_ALL)))
MAE_Time = pd.DataFrame(data=MAE_ALL.reshape(1, len(MAE_ALL)))
R2_Time = pd.DataFrame(data=R2_ALL.reshape(1, len(R2_ALL)))
save_data_path = 'NNPredictData/LSTM_MSE_Time_Window_3.csv'
save_data_path2 = 'NNPredictData/LSTM_MAE_Time_Window_3.csv'
save_data_path3 = 'NNPredictData/LSTM_R2_Time_Window_3.csv'
MSE_Time.to_csv(save_data_path, index=False, mode='a', header=0)
MAE_Time.to_csv(save_data_path2, index=False, mode='a', header=0)
R2_Time.to_csv(save_data_path3, index=False, mode='a', header=0)

# plt.figure(dpi=300)
# plt.plot(range(701, 828), MSE_ALL, color='r', marker='o', linewidth=1, markeredgecolor='b', markersize='2',
#          markeredgewidth=1)
#
# # ------------------------------------------------------------------------------------------------------------------
# plt.xlabel()


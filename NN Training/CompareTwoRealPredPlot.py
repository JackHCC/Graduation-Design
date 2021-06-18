import numpy as np
from PlotMethodSet import plot_contour_predict
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import pandas as pd
import math
import plot_map
from SecondtoTime import sec2time

# --------------------注意切换不同步长数据---------------------
# 加载预测SVR数据
# data = np.load('./SVRPredictData/all_y_hat_svr_3.npz')
# ALL_y_hat = data['all_y_hat']

# 加载预测MLP数据
data = np.load('./NNPredictData/all_y_hat_nn_4.npz')
ALL_y_hat = data['all_y_hat']

# 加载预测LSTM数据
# data = np.load('./NNPredictData/all_y_hat_lstm_3.npz')
# ALL_y_hat = data['all_y_hat']


# 更换预测窗口长度时更换--------------------------------------
Link_Num = 4

# 计算对比的预测时间点（70000-86400）
test_sec = 75600
# test_sec = 72000
# test_sec = 79200

# 加载预测NN数据
# data = np.load('all_y_hat_nn_3.npz')
# ALL_y_hat = data['all_y_hat']

# 加载真实数据
# 数据读取time：57~58s
data = pd.read_csv(r'../Data Process/RawGPSData.csv', header=None)
data.columns = ['VehicleNum', 'Stime', 'Lng', 'Lat']

data_10 = data[data['Stime'] == test_sec]
# print(data_10)

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
print(Sample_Metric.shape)



Point_Num = len(data_10)
X = np.array(data_10[['Lng', 'Lat']])


# 计算地球经纬度与公里数转换 1度多少千米
deltaLng_km = (MaxX-MinX)*(2 * math.pi * 6371 * math.cos((MinY + MaxY) * math.pi/360))/360
deltaLat_km = (MaxY-MinY)*(2 * math.pi * 6371)/360
print(deltaLng_km, deltaLat_km)

# 计算最佳核密度估计最佳带宽
# Method1:
# bandwidth1 = np.power(0.68 * Point_Num, -0.2) * np.sqrt((MaxX - MinX) * (MaxY - MinY))/(np.sqrt(deltaLng_km*deltaLat_km))
bandwidth1 = 0.5/np.sqrt(deltaLng_km*deltaLat_km)
# bandwidth1 = np.power(0.68*Point_Num, -0.2)*np.sqrt((MaxX-MinX)*(MaxY-MinY))/(np.sqrt(111*55))
# print("Best Bandwidth:", bandwidth1)
# Method2:计算中心位置，计算距离方差和中位数

# KDEFunction
# kde, dense = KDEFunction.KDE(X, Sample_Metric, bandwidth1)
kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth1).fit(X)
log_dense = kde.score_samples(Sample_Metric)
dense = np.exp(log_dense)


# 用来正常显示中文标签,并设置字体大小
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 6

fig = plt.figure(dpi=300)

# 真实热点图------------------------------------
ax = fig.add_subplot(1, 2, 1)
plt.sca(ax)

# 背景
plot_map.plot_map(plt, bounds, zoom=12, style=3)

# 散点图分布
# plt.scatter(data_10['Lng'], data_10['Lat'], s=1, alpha=0.2)

# 画核密度等高线图
# 作图
Contour_X = Sample_X
Contour_Y = Sample_Y
Contour_Z = np.reshape(dense, Contour_X.shape)

# fig = plt.figure()
axis = fig.gca()
axis.set_xlim(MinX, MaxX)
axis.set_ylim(MinY, MaxY)
# contourf_set = axis.contourf(Contour_X, Contour_Y, Contour_Z, levels=np.linspace(0, Contour_Z.max(), 12), alpha=0.7, cmap='Reds')
contourf_set = axis.contourf(Contour_X, Contour_Y, Contour_Z, alpha=0.7, cmap='Reds')

# 设置colorbar
cbar = fig.colorbar(contourf_set, fraction=0.035, pad=0.05)
cbar.set_clim(0, Contour_Z.max())

# contour_set = axis.contour(Contour_X, Contour_Y, Contour_Z, colors='r')
# axis.clabel(contour_set, inline=1, fontsize=5)
axis.set_xlabel('Longitude')
axis.set_ylabel('Latitude')
plt.title(str(sec2time(test_sec)) + "密度分布真实图", fontsize=10)


# 预测图-----------------------------------------------
ax = fig.add_subplot(1, 2, 2)
plt.sca(ax)

# 背景
plot_map.plot_map(plt, bounds, zoom=12, style=3)

# plt = plot_contour_predict(ALL_y_hat, Plot_Time=int(test_sec/100), Figure_Num=1)

Plot_Time = int(test_sec/100)

# 采样点个数
# Point_Num = len(ALL_y_hat[:])
# print(Point_Num)

# 加载经纬度数据
data1 = np.load('../KDE/all_f_data.npz')
# 经纬度数据提取
Data_ALL = data1['all_data'][0, :, :2]
print(Data_ALL.shape)

# 将两组数据合并
Predict_Con_data = np.c_[Data_ALL, ALL_y_hat[:, Plot_Time, Link_Num]]
print(Predict_Con_data.shape)

# 从所有数据中经纬度数据极值提取
# MaxX = np.max(Data_ALL[:, 0])
# MinX = np.min(Data_ALL[:, 0])
# MaxY = np.max(Data_ALL[:, 1])
# MinY = np.min(Data_ALL[:, 1])
# Sample_Num = int(np.sqrt(Point_Num))

# 获取采样坐标点
Contour_X = Data_ALL[:, 0].reshape(Sample_Num, Sample_Num)
Contour_Y = Data_ALL[:, 1].reshape(Sample_Num, Sample_Num)
print(Contour_X.shape, Contour_Y.shape)

Contour_Z = np.reshape(ALL_y_hat[:, Plot_Time, Link_Num], Contour_X.shape)

# 作图
axis = fig.gca()
axis.set_xlim(MinX, MaxX)
axis.set_ylim(MinY, MaxY)

# contourf_set = axis.contourf(Contour_X, Contour_Y, Contour_Z, alpha=0.7, levels=np.linspace(0, Contour_Z.max(), 12), cmap='Reds')
contourf_set = axis.contourf(Contour_X, Contour_Y, Contour_Z, alpha=0.7, cmap='Reds')

# 设置colorbar
cbar = fig.colorbar(contourf_set, fraction=0.035, pad=0.05)
cbar.set_clim(0, Contour_Z.max())

# contour_set = axis.contour(Contour_X, Contour_Y, Contour_Z, colors='k')
# axis.clabel(contour_set, inline=0.2, fontsize=0.5)

axis.set_xlabel('Longitude')
axis.set_ylabel('Latitude')

# plt.title(str(sec2time(test_sec)) + "密度分布预测图，预测窗口长：" + str(Link_Num))
plt.title(str(sec2time(test_sec)) + "密度分布预测图", fontsize=10)
plt.subplots_adjust(top=0.92, bottom=0.08, right=0.96, left=0.1, hspace=0, wspace=0.3)

# 图片存储在HotSpotPlot目录
plt_name = './PredictHotSpotNN/HotSpot_Real_Predict_MLP_'+str(test_sec)+'_'+str(Link_Num)+'update'
# plt_name = './PredictHotSpotNN/HotSpot_Real_Predict_LSTM_'+str(test_sec)+'_'+str(Link_Num)+'update'
# plt_name = './PredictHotSpotSVR/HotSpot_Real_Predict_SVR_'+str(test_sec)+'_'+str(Link_Num)+'update'
plt.savefig(plt_name+'.png')
plt.savefig(plt_name+'.svg')

plt.show()



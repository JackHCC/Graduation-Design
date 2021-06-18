import numpy as np
from PlotMethodSet import plot_contour_predict
import matplotlib.pyplot as plt
import plot_map

# --------------------注意切换不同步长数据---------------------
# 加载预测SVR数据
# data = np.load('./SVRPredictData/all_y_hat_svr_5.npz')
# ALL_y_hat = data['all_y_hat']

# 加载预测NN数据
data = np.load('./NNPredictData/all_y_hat_nn_5.npz')
ALL_y_hat = data['all_y_hat']

# 读取shapefile文件
# shp = r'../Beijing Map Data/北京市.shp'
# bj = geopandas.GeoDataFrame.from_file(shp, encoding='utf-8')

# 北京市经纬度边界
# bounds = [115.7, 39.4, 117.4, 41.6]
bounds = [116.0, 39.6, 116.8, 40.2]

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

fig = plt.figure(dpi=300)
ax = plt.subplot(111)
plt.sca(ax)
# fig.tight_layout(rect=(0.05, 0.1, 1, 0.9))

# 背景
plot_map.plot_map(plt, bounds, zoom=12, style=3)

plt = plot_contour_predict(ALL_y_hat, Plot_Time=720, Figure_Num=1)

plt.title("预测图")

# 图片存储在HotSpotPlot目录
plt_name = './PredictHotSpotSVR/HotSpotPredict_'+str(72000)
plt.savefig(plt_name+'.png')
# cax = plt.axes([0.13, 0.32, 0.02, 0.3])
# plt.colorbar(cax=cax)
plt.show()

# SVR预测热点图源码

# # 采样点个数
# Point_Num = len(ALL_y_hat[:])
# print(Point_Num)
#
# # ALL_y_hat_pro = []
# # for i in range(4):
# #     ALL_y_hat_pro = np.append(ALL_y_hat_pro, ALL_y_hat)
# #
# # ALL_y_hat_pro = ALL_y_hat_pro.reshape(4, Point_Num, len(ALL_y_hat[0, :]))
# #
# # print(ALL_y_hat_pro.shape)
# #
# # for i in range(Point_Num):
# #     ALL_y_hat[i, :] = get_minmax_scaler().inverse_transform(ALL_y_hat_pro[:, i, :].T)[:, 0]
#
#
# print(ALL_y_hat.shape)
# print(ALL_y_hat[0, :])
#
# # 加载经纬度数据
# data1 = np.load('../KDE/all_f_data.npz')
# # 经纬度数据提取
# Data_ALL = data1['all_data'][0, :, :2]
# print(Data_ALL.shape)
#
# # 将两组数据合并
# Predict_SVR_data = np.c_[Data_ALL, ALL_y_hat]
# print(Predict_SVR_data.shape)
#
#
# # 从所有数据中经纬度数据极值提取
# MaxX = np.max(Data_ALL[:, 0])
# MinX = np.min(Data_ALL[:, 0])
# MaxY = np.max(Data_ALL[:, 1])
# MinY = np.min(Data_ALL[:, 1])
#
#
# Sample_Num = int(np.sqrt(Point_Num))
# # 获取采样坐标点
# Contour_X = Data_ALL[:, 0].reshape(Sample_Num, Sample_Num)
# Contour_Y = Data_ALL[:, 1].reshape(Sample_Num, Sample_Num)
# print(Contour_X.shape, Contour_Y.shape)
#
# Contour_Z = np.reshape(ALL_y_hat[:, 30], Contour_X.shape)
#
# # 作图
# fig = plt.figure()
# axis = fig.gca()
# axis.set_xlim(MinX, MaxX)
# axis.set_ylim(MinY, MaxY)
#
# contourf_set = axis.contourf(Contour_X, Contour_Y, Contour_Z, cmap='Reds')
#
# contour_set = axis.contour(Contour_X, Contour_Y, Contour_Z, colors='k')
# axis.clabel(contour_set, inline=1, fontsize=10)
#
# axis.set_xlabel('Longitude')
# axis.set_ylabel('Latitude')
#
# # plt_name = './KDE_Random'
# # plt.savefig(plt_name+'.jpg')
# plt.show()

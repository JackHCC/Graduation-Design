import pandas as pd
import numpy as np
import KDEFunction
import time
import math
import matplotlib.pyplot as plt
import plot_map
from SecondtoTime import sec2time

# 读取GPS数据
data = pd.read_csv(r'../Data Process/RawGPSData.csv', header=None)
data.columns = ['VehicleNum', 'Stime', 'Lng', 'Lat']
# print(data.shape)
# print(data.head(5))

# 获取出租车数据经纬度最值
# maxLng = data['Lng'].max()
# maxLat = data['Lat'].max()
# minLng = data['Lng'].min()
# minLat = data['Lat'].min()

# bounds
# print(maxLng, minLng, maxLat, minLat)
# bounds = [117.9963607788086, 115.20406341552734, 40.87900924682617, 39.00091552734375]
# bounds = [115.20406341552734, 39.00091552734375, 117.9963607788086, 40.87900924682617]

# 北京市经纬度边界【最小经度，最小纬度，最大经度，最大纬度】
# bounds = [115.7, 39.4, 117.4, 41.6]
bounds = [116.0, 39.6, 116.8, 40.2]
# bounds = [116.15, 39.7, 116.6, 40.1]
MaxX = bounds[2]
MinX = bounds[0]
MaxY = bounds[3]
MinY = bounds[1]

# 计算地球经纬度与公里数转换 1度多少千米
deltaLng_km = (MaxX-MinX)*(2 * math.pi * 6371 * math.cos((MinY + MaxY) * math.pi/360))/360
deltaLat_km = (MaxY-MinY)*(2 * math.pi * 6371)/360
# print(deltaLng_km, deltaLat_km)

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

# 将所有特征数据存入三维数组
all_feature_data = []

# 计时
start = time.process_time()

# 设置带宽0.25km
bandwidth_km = 0.5
bandwidth1 = bandwidth_km/np.sqrt(deltaLng_km * deltaLat_km)

# 一天秒数计算 86400
one_day_second = 60 * 60 * 24

# 每小时提取一张
for i in range(0, int(one_day_second), 3600):
    data_10 = data[data['Stime'] == i]
    # print(data_10)
    Point_Num = len(data_10)
    X = np.array(data_10[['Lng', 'Lat']])

    # 计算最佳核密度估计最佳带宽
    # Method1:
    # bandwidth1 = np.power(0.68 * Point_Num, -0.2) * np.sqrt((MaxX - MinX) * (MaxY - MinY)) / (np.sqrt(deltaLng_km*deltaLat_km))
    # bandwidth1 = np.power(0.68*Point_Num, -0.2)*np.sqrt((MaxX-MinX)*(MaxY-MinY))/(np.sqrt(111*55))
    # print("Best Bandwidth:", bandwidth1)
    # Method2:计算中心位置，计算距离方差和中位数

    # KDEFunction
    kde, dense = KDEFunction.KDE(X, Sample_Metric, bandwidth1)

    # 北京市经纬度边界
    fig = plt.figure(dpi=300)
    ax = plt.subplot(111)
    plt.sca(ax)
    fig.tight_layout(rect=(0.05, 0.1, 1, 0.9))

    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']

    # 背景
    plot_map.plot_map(plt, bounds, zoom=12, style=3)

    # 散点图分布
    plt.scatter(data_10['Lng'], data_10['Lat'], s=1, alpha=0.25)

    # 画核密度等高线图
    # 作图
    Contour_X = Sample_X
    Contour_Y = Sample_Y
    Contour_Z = np.reshape(dense, Contour_X.shape)

    # fig = plt.figure()
    axis = fig.gca()
    axis.set_xlim(MinX, MaxX)
    axis.set_ylim(MinY, MaxY)
    # contourf_set = axis.contourf(Contour_X, Contour_Y, Contour_Z, levels=np.linspace(0, Contour_Z.max(), 10), alpha=0.5, cmap='Reds')
    contourf_set = axis.contourf(Contour_X, Contour_Y, Contour_Z, alpha=0.7, cmap='Reds')

    # 设置colorbar
    cbar = fig.colorbar(contourf_set)
    cbar.set_clim(0, Contour_Z.max())
    # cbar.set_clim(Contour_Z.min(), Contour_Z.max())

    # contour_set = axis.contour(Contour_X, Contour_Y, Contour_Z, colors='r')
    # axis.clabel(contour_set, inline=1, fontsize=5)

    axis.set_xlabel('Longitude')
    axis.set_ylabel('Latitude')
    plt.title(str(sec2time(i)) + "出租车热点分布情况(周一)" + "/带宽：" + str(bandwidth_km))

    # 图片存储在HotSpotPlot目录
    plt_name = './HotSpotPlot/' + str(int(i/3600))
    plt.savefig(plt_name + '.png')

    # 动态观察程序进展
    print(i, "/82800")

end = time.process_time()
print("Data Process 2 Step time:", end - start)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import KDEFunction
import time
import plot_map
import math
from SecondtoTime import sec2time

# 读取GPS数据
# 数据读取time：57~58s
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


# 计算100秒时的热点数据
test_sec = 28800
data_10 = data[data['Stime'] == test_sec]
# print(data_10)
Point_Num = len(data_10)
X = np.array(data_10[['Lng', 'Lat']])

# 计时
start = time.process_time()

# # 计算地球经纬度与公里数转换 1度多少千米
deltaLng_km = (MaxX-MinX)*(2 * math.pi * 6371 * math.cos((MinY + MaxY) * math.pi/360))/360
deltaLat_km = (MaxY-MinY)*(2 * math.pi * 6371)/360
# print(deltaLng_km, deltaLat_km)

# 公里数
bandwidth_list = [0.1, 0.25, 0.5, 0.75, 1, 2, 3, 6, 10]
# 不同带宽分析
bandwidth1 = np.array(bandwidth_list)/np.sqrt(deltaLat_km*deltaLng_km)

# 0.25-0.5km效果最佳

print(bandwidth1)

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 3.5
# plt.rcParams['font.size'] = 4
fig = plt.figure(dpi=300)


for i in range(len(bandwidth1)):

    # KDEFunction
    kde, dense = KDEFunction.KDE(X, Sample_Metric, bandwidth1[i])

    # 打印进度
    print(i, '/8', dense.max(), dense.min())


    # 保存特征数据num*3 经度|纬度|核密度估计值
    # KDEFunction.SaveNPZ(Sample_Metric, dense, "fdata")

    # 画核密度等高线图
    # KDEFunction.PlotContour(MinX, MaxX, MinY, MaxY, Sample_Space_X, Sample_Space_Y, kde, plotbool=True, savebool=False, picname="KDE_Random")

    # 北京市经纬度边界
    # fig = plt.figure(dpi=300)
    # ax = plt.subplot(111)
    # plt.sca(ax)
    # fig.tight_layout(rect=(0.05, 0.1, 1, 0.9))
    # plt.subplot(3, 3, i+1)

    # 散点图分布
    # plt.scatter(data_10['Lng'], data_10['Lat'], s=1, alpha=0.5)

    # 画核密度等高线图
    # 作图
    Contour_X = Sample_X
    Contour_Y = Sample_Y
    Contour_Z = np.reshape(dense, Contour_X.shape)

    # axis = fig.gca()
    axis = fig.add_subplot(3, 3, i+1)

    # 背景
    plot_map.plot_map(plt, bounds, zoom=12, style=3)
    # plt.axis('off')
    plt.axis('on')
    axis.set_xlim(MinX, MaxX)
    axis.set_ylim(MinY, MaxY)

    contourf_set = axis.contourf(Contour_X, Contour_Y, Contour_Z, levels=np.linspace(0, Contour_Z.max(), 12), alpha=0.7, cmap='Reds')
    # contourf_set = axis.contourf(Contour_X, Contour_Y, Contour_Z, alpha=0.6, cmap='Reds')

    # 设置colorbar
    cbar = fig.colorbar(contourf_set, fraction=0.04, pad=0.04)
    cbar.set_clim(Contour_Z.min(), Contour_Z.max())

    # contour_set = axis.contour(Contour_X, Contour_Y, Contour_Z, colors='r')
    # axis.clabel(contour_set, inline=1, fontsize=5)

    # axis.set_xlabel('Longitude')
    # axis.set_ylabel('Latitude')
    plt.title("带宽" + str(bandwidth_list[i]) + "km时" + "出租车热点分布（8点）", fontsize='7')


# axis on
plt.subplots_adjust(top=0.92, bottom=0.08, right=0.96, left=0.04, hspace=0.32, wspace=0.25)
# axis off
# plt.subplots_adjust(top=0.92, bottom=0.08, right=0.96, left=0.04, hspace=0.4, wspace=0.15)

end = time.process_time()
print("Data Process 2 Step time:", end - start)

# 图片存储在HotSpotPlot目录
plt_name = './KDEbandAnaysis/HotSpot_band_list_update_7'
plt.savefig(plt_name+'.png')
plt.savefig(plt_name+'.svg')

plt.show()




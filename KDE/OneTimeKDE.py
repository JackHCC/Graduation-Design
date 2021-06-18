import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import KDEFunction
import time

# 计时
start = time.process_time()

# 读取GPS数据
data = pd.read_csv(r'../Data Process/RawGPSData.csv', header=None)
data.columns = ['VehicleNum', 'Stime', 'Lng', 'Lat']
# print(data.shape)
# print(data.head(5))

end = time.process_time()
print("Data Process 2 Step time:", end - start)

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
bounds = [115.7, 39.4, 117.4, 41.6]
MaxX = bounds[2]
MinX = bounds[0]
MaxY = bounds[3]
MinY = bounds[1]

# 坐标采样个数,即每个时点采集100*100个位置的密度信息(测试使用20)
Sample_Num = 20

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
data_10 = data[data['Stime'] == 100]
# print(data_10)
Point_Num = len(data_10)
X = np.array(data_10[['Lng', 'Lat']])

# 计算最佳核密度估计最佳带宽
# Method1:
bandwidth1 = np.power(0.68 * Point_Num, -0.2) * np.sqrt((MaxX - MinX) * (MaxY - MinY))
# bandwidth1 = np.power(0.68*Point_Num, -0.2)*np.sqrt((MaxX-MinX)*(MaxY-MinY))/(np.sqrt(111*55))
# print("Best Bandwidth:", bandwidth1)
# Method2:计算中心位置，计算距离方差和中位数

# KDEFunction
kde, dense = KDEFunction.KDE(X, Sample_Metric, bandwidth1)

feature_data = np.c_[Sample_Metric, dense]

print(feature_data.shape)

# 画核密度等高线图
# 作图
Contour_X = Sample_X
Contour_Y = Sample_Y
Contour_Z = np.reshape(dense, Contour_X.shape)

fig = plt.figure()
axis = fig.gca()
axis.set_xlim(MinX, MaxX)
axis.set_ylim(MinY, MaxY)

contourf_set = axis.contourf(Contour_X, Contour_Y, Contour_Z, cmap='Reds')

contour_set = axis.contour(Contour_X, Contour_Y, Contour_Z, colors='k')
axis.clabel(contour_set, inline=1, fontsize=10)

axis.set_xlabel('Longitude')
axis.set_ylabel('Latitude')

plt_name = './KDE_Random'
plt.savefig(plt_name+'.jpg')
plt.show()
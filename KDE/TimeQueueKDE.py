import pandas as pd
import numpy as np
import KDEFunction
import time
import math

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

# 坐标采样个数,即每个时点采集100*100个位置的密度信息(测试使用20)
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

# 一天秒数计算 86400
one_day_second = 60 * 60 * 24

# 设置带宽0.5km
bandwidth_km = 0.5
bandwidth1 = bandwidth_km/np.sqrt(deltaLng_km * deltaLat_km)

# 864:一天的秒数除以100,即每百秒提取一次
for i in range(int(one_day_second / 100)):
    data_10 = data[data['Stime'] == 100 * i]
    # print(data_10)
    Point_Num = len(data_10)
    X = np.array(data_10[['Lng', 'Lat']])

    # 计算最佳核密度估计最佳带宽
    # Method1:
    # bandwidth1 = np.power(0.68 * Point_Num, -0.2) * np.sqrt((MaxX - MinX) * (MaxY - MinY)) / 5
    # bandwidth1 = np.power(0.68*Point_Num, -0.2)*np.sqrt((MaxX-MinX)*(MaxY-MinY))/(np.sqrt(111*55))
    # print("Best Bandwidth:", bandwidth1)
    # Method2:计算中心位置，计算距离方差和中位数

    # KDEFunction
    kde, dense = KDEFunction.KDE(X, Sample_Metric, bandwidth1)

    feature_data = np.c_[Sample_Metric, dense]

    # 将每个时序数据放入同一个三维数组并存储
    all_feature_data = np.append(all_feature_data, feature_data)

    # 动态观察程序进展
    print(i, "/863")

# 输出所有处理的密度特征
all_feature_data = all_feature_data.reshape(864, len(Sample_Metric[:]), len(feature_data[0, :]))

f_name = 'all_f_data'
np.savez(f_name + '.npz', all_data=all_feature_data)

end = time.process_time()
print("Data Process 2 Step time:", end - start)

print(all_feature_data.shape)
print(all_feature_data[0])

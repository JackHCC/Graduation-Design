import scipy.io as scio
import pandas as pd
import numpy as np
import time

# 原始数据路径
filepath1 = '../RawData/interpolated_time_01.mat'
filepath2 = '../RawData/interpolated_position_01.mat'
# 加载数据
time_data = scio.loadmat(filepath1)
position_data = scio.loadmat(filepath2)
# 获取数据数组
array_time = time_data['interpolated_time']
array_position = position_data['interpolated_position']

# for i in range(20):
#     if array_position[0, i].any():
#         print(array_position[0, i][1])

print(array_position[0, 0][:])
print(array_time[0, 0][:])


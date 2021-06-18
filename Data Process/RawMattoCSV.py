import scipy.io as scio
import pandas as pd
import numpy as np
import time

# 原始数据路径
filepath1 = '../RawData/interpolated_time_07.mat'
filepath2 = '../RawData/interpolated_position_07.mat'
# 加载数据
time_data = scio.loadmat(filepath1)
position_data = scio.loadmat(filepath2)
# 获取数据数组
array_time = time_data['interpolated_time']
array_position = position_data['interpolated_position']

print(array_time.shape)
print(array_position.shape)

k = 0
array_all = []

start = time.process_time()

# 数据转换
for i in range(0, len(array_time[0, :])):
    if array_time[0, i].any():
        k = k+1
        array_i = np.ones([len(array_time[0, i]), 1])*i
        array_all_one = np.c_[array_i, array_time[0, i], array_position[0, i]/1e5]
        if i == 0:
            array_all = array_all_one
        else:
            array_all = np.concatenate((array_all, array_all_one))
    else:
        pass
    print(i)

df_data_all = pd.DataFrame(data=array_all)
save_data_path = 'RawGPSData7.csv'

end = time.process_time()
print("Data Process 1 Step time:", end-start)

print(k)
print(array_all.shape, array_all)

df_data_all.to_csv(save_data_path, index=False, header=0)

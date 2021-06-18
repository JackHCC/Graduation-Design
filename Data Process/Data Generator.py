import numpy as np


test_data=np.random.random((100, 100, 2))*10  #随机生成100（时间序列）*100（此时刻位置的点）*2（位置信息）的数据


print(test_data)
np.savez('rdata', test_data=test_data)

# with open('../RawData/rdata.txt',mode='w') as f:
#     f.write(test_data)

# 载入数据
# data=np.load('rdata.npz')
# data.files


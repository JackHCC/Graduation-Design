import numpy as np
import time

# 加载数据
data = np.load('../KDE/all_f_data.npz')

# 所有数据
Data_ALL = data['all_data']
print(Data_ALL.shape)
print(Data_ALL[0, 0, 2])

# 训练数据构建
# 每N个连续数据点预测下一时刻数据点，N=3
# 滑动窗口最后一个数据只能作为Y，前N个数据不能作为Y，因此数据长度ALL-N
Link_Num = 9 # 预测数据量LinK_Num-->Next One

# 获取Link_Num
# def Get_LinkNum():
#     return Link_Num

# ALL_Train_Data = np.empty(shape=[len(Data_ALL[:, 0, 2])-Link_Num, Link_Num+1])
ALL_Train_Data = []

start = time.process_time()

# 训练数据构造
for k in range(0, len(Data_ALL[0, :, 2])):
    X_data = []
    Y_data = []

    Point_Data = Data_ALL[:, k, 2]
    # 训练数据构造
    for i in range(0, len(Point_Data)-Link_Num):
        for j in range(0, Link_Num):
            X_data = np.append(X_data, Point_Data[i+j])
        Y_data = np.append(Y_data, Point_Data[i+j+1])

    X_data = X_data.reshape(len(Point_Data)-Link_Num, Link_Num)
    Y_data = Y_data.reshape(len(Point_Data)-Link_Num, 1)

    Train_data = np.c_[X_data, Y_data]
    ALL_Train_Data = np.append(ALL_Train_Data, Train_data)

    # 动态观察程序进展
    print(k, "/9999")


# 维度说明：采样点个数*时间序列数*（Link_Num+1）
ALL_Train_Data = ALL_Train_Data.reshape(len(Data_ALL[0, :, 2]), len(Train_data[:, 0]), len(Train_data[0, :]))
print(ALL_Train_Data.shape)
# print(ALL_Train_Data[99, :, :], Train_data)

end = time.process_time()
print("Data Process 3 Step time:", end-start)

# 保存训练数据
np.savez('./TrainData/all_point_data_9.npz', all_point_data=ALL_Train_Data)











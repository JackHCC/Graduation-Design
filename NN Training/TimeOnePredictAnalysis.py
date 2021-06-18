import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, LSTM
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KernelDensity
import pandas as pd
import math
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from SecondtoTime import sec2time

# 最佳窗口长度 3
# 加载数据
data = np.load('./TrainData/all_point_data_3.npz')

# 所有数据aa
# 维度说明：采样点个数*时间序列数*（Link_Num+1）
Data_ALL = data['all_point_data']
# print(Data_ALL.shape)


def Get_LinkNum():
    return 3

# 采样点个数
Point_Num = len(Data_ALL[:])
# 时间序列长度
Time_Num = len(Data_ALL[0, :])

# 训练集测试集分类
Split = 700

Data_ALL_Trans = Data_ALL.copy()
Data_ALL_Test = Data_ALL.copy()


point = 5555

# 对每个采样点进行归一化处理
# MinMaxScaler():
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min
min_max_scaler = preprocessing.MinMaxScaler()
# for i in range(0, Point_Num):
Data_ALL_Trans[point, :, :] = min_max_scaler.fit_transform(Data_ALL[point, :, :])

# 数据还原
# min_max_scaler.inverse_transform(data)

X_data = Data_ALL_Trans[:, :, :Get_LinkNum()]
Y_data = Data_ALL_Trans[:, :, Get_LinkNum()]
# print(X_data.shape, Y_data.shape)

X_train_data = X_data[point, :Split, :]
# # LSTM预测需要加这行——————————————————————————————————————————
# X_train_data = np.expand_dims(X_train_data, 2)
# # LSTM预测需要加这行——————————————————————————————————————————
Y_train_data = Y_data[point, :Split]
# X_test_data = X_data[point, Split:, :]
# Y_test_data = Y_data[point, Split:]
# print(X_train_data.shape, Y_train_data.shape)

# clf = MLPRegressor(solver='sgd', alpha=1e-4, learning_rate_init=0.01, hidden_layer_sizes=(19, 1), activation='relu',
#                        max_iter=1000, random_state=1).fit(X_train_data, Y_train_data)

# clf = SVR(kernel='linear', C=10).fit(X_train_data, Y_train_data)
clf = SVR(kernel='poly', degree=5, coef0=0.04, gamma=0.2, C=1).fit(X_train_data, Y_train_data)
# 建立LSTM模型
# tf.random.set_seed(1)
# model = tf.keras.Sequential([
#     # LSTM(100, return_sequences=True),
#     # Dropout(0.1),
#     LSTM(100),
#     Dropout(0.1),
#     Dense(1)
# ])
# model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error')  # 损失函数用均方误差
# clf = model.fit(X_train_data, Y_train_data, batch_size=64, epochs=50,
#                 validation_freq=1)

Window = Y_train_data[-Get_LinkNum():]
# print(Window)
all_y_hat = []
Window_ALL = []

for i in range(0, Time_Num-Split):
    New_predict_time = clf.predict(Window.reshape(1, -1))
    # New_predict_time = model.predict(Window.reshape(1, -1))
    print(New_predict_time)

    # print(New_predict_time)

    # X_train_data = np.concatenate((X_train_data, Window.reshape(1, -1)))
    # Y_train_data = np.append(Y_train_data, New_predict_time.reshape(1, -1))
    # print(New_predict_time[0])

    # print(X_train_data.shape, Y_train_data.shape)

    all_y_hat = np.append(all_y_hat, New_predict_time[0])
    Window_ALL = np.append(Window_ALL, Window)
    Window = np.delete(Window, 0)
    Window = np.append(Window, New_predict_time)

Window_ALL = np.reshape(Window_ALL, [Time_Num-Split, Get_LinkNum()])

test_data_trans = np.c_[Window_ALL, all_y_hat.reshape(-1, 1)]
data_all_trans = np.concatenate((Data_ALL_Trans[point, :Split, :], test_data_trans))

# 数据还原
data_all = min_max_scaler.inverse_transform(data_all_trans)
# print(data_all.shape, data_all)
Data_ALL_Test[point] =data_all

y_test_pred = Data_ALL_Test[point, Split:, Get_LinkNum()]
y_test_raw = Data_ALL[point, Split:, Get_LinkNum()]

print(y_test_pred.shape, y_test_raw.shape)

# 用来正常显示中文标签,并设置字体大小
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 12

# 可视化拟合数据与真实结果对比
plt.figure(1, figsize=(9, 4), dpi=200)


x = range(Split, Split+len(y_test_raw))
plt.scatter(x, y_test_raw, color="blue", label="采样点", linewidth=3, s=0.5)

xx = range(Split, Split+len(y_test_pred))
plt.plot(xx, y_test_pred, color="green", label="拟合曲线", linewidth=1)


plt.xlabel("时间(×100s)")
plt.ylabel("核密度估计值")
plt.xlim(Split, Split+len(y_test_raw))
# title_name = "支持向量回归单点(经纬度:" + str(point_X) + "," + str(point_Y) + ")拟合效果与预测效果对比图"
# plt.title(title_name)
plt.legend()

plt.show()





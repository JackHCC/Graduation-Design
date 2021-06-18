import numpy as np
# from AllPointTrainDataProcess import Get_LinkNum
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, LSTM
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt



# 加载数据
data = np.load('./TrainData/all_point_data_3.npz')

# 所有数据aa
# 维度说明：采样点个数*时间序列数*（Link_Num+1）
Data_ALL = data['all_point_data']
# print(Data_ALL.shape)

# 采样点个数
Point_Num = len(Data_ALL[:])
# 时间序列长度
Time_Num = len(Data_ALL[0, :])

# ----------------每次运行前注意预测长度是否修改-------------------------
def Get_LinkNum():
    return 3

# point = 3951  #116.408 39.91608 天安门故宫附近
# point = 4946  #116.368 39.9956   北京科技大学附近
point = 5675  #116.6 40.054   首都机场附近
point_list = np.array([[116.408, 39.916], [116.368, 39.996], [116.6, 40.054]])

# print(Data_ALL[point, :, :].shape, Data_ALL[point, :, :])

# point与经纬度转化关系
# 北京市经纬度边界
# bounds = [115.7, 39.4, 117.4, 41.6]
bounds = [116.0, 39.6, 116.8, 40.2]
MaxX = bounds[2]
MinX = bounds[0]
MaxY = bounds[3]
MinY = bounds[1]
# 坐标采样个数,即Sample_Num*Sample_Num
Sample_Num = 100
deltaX = (MaxX-MinX)/Sample_Num
deltaY = (MaxY-MinY)/Sample_Num

point_X = (point % 100) * deltaX + MinX
point_Y = (point / 100) * deltaX + MinY
print(point_X, point_Y)


Data_ALL_Trans = Data_ALL.copy()

# 对每个采样点进行归一化处理
# MinMaxScaler():
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min
min_max_scaler = preprocessing.MinMaxScaler()
# for i in range(0, Point_Num):
Data_ALL_Trans[point, :, :] = min_max_scaler.fit_transform(Data_ALL[point, :, :])


X_data = Data_ALL_Trans[:, :, :Get_LinkNum()]
Y_data = Data_ALL_Trans[:, :, Get_LinkNum()]
# print(X_data.shape, Y_data.shape)

# 训练集测试集分类
Split = 700

X_train_data = X_data[point, :Split, :]
X_train_data = np.expand_dims(X_train_data, 2)
Y_train_data = Y_data[point, :Split]
X_test_data = X_data[point, Split:, :]
X_test_data = np.expand_dims(X_test_data, 2)
Y_test_data = Y_data[point, Split:]

print(X_train_data.shape, Y_train_data.shape)

# 建立LSTM模型
tf.random.set_seed(1)

model = tf.keras.Sequential([
    # LSTM(100, return_sequences=True),
    # Dropout(0.1),
    LSTM(100),
    Dropout(0.1),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error')  # 损失函数用均方误差

clf = model.fit(X_train_data, Y_train_data, batch_size=64, epochs=50, validation_data=(X_test_data, Y_test_data),
                validation_freq=1)

model.summary()

# # 画loss曲线
# loss = clf.history['loss']
# val_loss = clf.history['val_loss']
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.show()

y_train_hat = model.predict(X_train_data)
y_test_hat = model.predict(X_test_data)
print(y_test_hat.shape)
# score = clf.score(X_test_data, Y_test_data)
# print("score:", score)

X_test_data = np.squeeze(X_test_data)
X_train_data = np.squeeze(X_train_data)
print(X_test_data.shape)

# 归一化还原
test_data_trans = np.c_[X_test_data, y_test_hat]
train_data_trans = np.c_[X_train_data, y_train_hat]
data_all_trans = np.concatenate((train_data_trans, test_data_trans))

# 数据还原
data_all = min_max_scaler.inverse_transform(data_all_trans)
# print(data_all.shape, data_all)

y_raw = Data_ALL[point, :, Get_LinkNum()]
y_train_raw = Data_ALL[point, :len(X_train_data), Get_LinkNum()]
y_test_raw = Data_ALL[point, len(X_train_data):, Get_LinkNum()]

y_train_hat = data_all[:len(X_train_data), Get_LinkNum()]
y_test_hat = data_all[len(X_train_data):, Get_LinkNum()]

# 评价指标
# 训练集
# Mean squared error（均方误差）
MSE_Train = mean_squared_error(y_train_hat, y_train_raw)

# Mean absolute error（平均绝对误差）,给定数据点的平均绝对误差，一般来说取值越小，模型的拟合效果就越好。
MAE_Train = mean_absolute_error(y_train_hat, y_train_raw)

# explained_variance_score：解释方差分，这个指标用来衡量我们模型对数据集波动的解释程度，如果取值为1时，模型就完美，越小效果就越差。
EVS_Train = explained_variance_score(y_train_hat, y_train_raw)

# R2,R方可以理解为因变量y中的变异性能能够被估计的多元回归方程解释的比例，它衡量各个自变量对因变量变动的解释程度，其取值在0与1之间，其值越接近1，
# 则变量的解释程度就越高，其值越接近0，其解释程度就越弱。一般来说，增加自变量的个数，回归平方和会增加，残差平方和会减少，所以R方会增大；
# 反之，减少自变量的个数，回归平方和减少，残差平方和增加。
R2_Train = r2_score(y_train_hat, y_train_raw)
print("Train ERROR(MSE) = ", MSE_Train)
print("Train MAE = ", MAE_Train)
print("Train EVS = ", EVS_Train)
print("Train R2 = ", R2_Train)

# 计算测试集
MSE_Test = mean_squared_error(y_test_hat, y_test_raw)
MAE_Test = mean_absolute_error(y_test_hat, y_test_raw)
EVS_Test = explained_variance_score(y_test_hat, y_test_raw)
R2_Test = r2_score(y_test_hat, y_test_raw)
print("Test ERROR(MSE) = ", MSE_Test)
print("Test MAE = ", MAE_Test)
print("Test EVS = ", EVS_Test)
print("Test R2 = ", R2_Test)

# 用来正常显示中文标签,并设置字体大小
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 12

# 可视化拟合数据与真实结果对比
plt.figure(1, figsize=(9, 4), dpi=200)

x = range(0, len(y_raw))
plt.scatter(x, y_raw, color="blue", label="采样点", linewidth=3, s=0.5, alpha=0.4)

xx = range(0, len(y_train_raw))
plt.plot(xx, y_train_hat, color="green", label="拟合曲线", linewidth=1)

xxx = range(len(y_train_raw)+1, len(y_raw)+1)
plt.plot(xxx, y_test_hat, color="red", label="预测曲线", linewidth=1)

plt.xlabel("时间(×100s)")
plt.ylabel("出租车核密度值")
plt.xlim(0, len(y_raw)+1)
# title_name = "神经网络回归单点(经纬度:" + str(point_X) + "," + str(point_Y) + ")拟合效果与预测效果对比图"
# title_name = "LSTM天安门故宫附近(经纬度:" + str(point_list[0, 0]) + "," + str(point_list[0, 1]) + ")拟合效果与预测效果对比图"
# title_name = "LSTM北京科技大学附近(经纬度:" + str(point_list[1, 0]) + "," + str(point_list[1, 1]) + ")拟合效果与预测效果对比图"
title_name = "LSTM首都机场附近(经纬度:" + str(point_list[2, 0]) + "," + str(point_list[2, 1]) + ")拟合效果与预测效果对比图"
plt.title(title_name)
plt.legend()


# 图片存储在HotSpotPlot目录
# plt_name = './NNPredictData/LSTM_OnePoint_Predict_'+str(point_X)+"_"+str(point_Y)+"_"+"天安门"
# plt_name = './NNPredictData/LSTM_OnePoint_Predict_'+str(point_X)+"_"+str(point_Y)+"_"+"北京科技大学"
plt_name = './NNPredictData/LSTM_OnePoint_Predict_'+str(point_X)+"_"+str(point_Y)+"_"+"首都机场"
# plt.savefig(plt_name+'.png')
# plt.savefig(plt_name+'.svg')

plt.show()








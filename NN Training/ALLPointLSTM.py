import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, LSTM
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error


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

# 训练集测试集分类
Split = 700

# 评价指标数组，包含每个Point的预测指标
MAE_ALL = []
MSE_ALL = []
R2_ALL = []

Data_ALL_Trans = Data_ALL.copy()
Data_ALL_Test = Data_ALL.copy()

for point in range(Point_Num):
    # 进度
    print(point, '/9999')

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

    # model.summary()

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

    # R2,R方可以理解为因变量y中的变异性能能够被估计的多元回归方程解释的比例，它衡量各个自变量对因变量变动的解释程度，其取值在0与1之间，其值越接近1，
    # 则变量的解释程度就越高，其值越接近0，其解释程度就越弱。一般来说，增加自变量的个数，回归平方和会增加，残差平方和会减少，所以R方会增大；
    # 反之，减少自变量的个数，回归平方和减少，残差平方和增加。
    R2_Train = r2_score(y_train_hat, y_train_raw)
    # print("Train ERROR(MSE) = ", MSE_Train)
    # print("Train MAE = ", MAE_Train)
    # print("Train EVS = ", EVS_Train)
    # print("Train R2 = ", R2_Train)

    # 计算测试集
    MSE_Test = mean_squared_error(y_test_hat, y_test_raw)
    MAE_Test = mean_absolute_error(y_test_hat, y_test_raw)
    R2_Test = r2_score(y_test_hat, y_test_raw)
    # print("Test ERROR(MSE) = ", MSE_Test)
    # print("Test MAE = ", MAE_Test)
    # print("Test EVS = ", EVS_Test)
    # print("Test R2 = ", R2_Test)

    MSE_ALL = np.append(MSE_ALL, MSE_Test)
    MAE_ALL = np.append(MAE_ALL, MAE_Test)
    R2_ALL = np.append(R2_ALL, R2_Test)

# DataALLTest异常点处理,小于0的值按0计算
Data_ALL_Test[Data_ALL_Test < 0] = 0

# 异常值剔除
R2_ALL = [element for element in R2_ALL if element >= 0]

print("MAE:", np.mean(MAE_ALL))
print("MSE:", np.mean(MSE_ALL))
print("R2 Score:", np.mean(R2_ALL))

# 评价指标数据汇总
Sum_eva = np.array([[Get_LinkNum(),
                    np.mean(MSE_ALL), np.max(MSE_ALL), np.min(MSE_ALL),
                    np.mean(MAE_ALL), np.max(MAE_ALL), np.min(MAE_ALL),
                    np.mean(R2_ALL), np.max(R2_ALL), np.min(R2_ALL)]])

Sum_eva = pd.DataFrame(data=Sum_eva)
save_data_path = 'NNPredictData/LSTM_Sum_Params.csv'
Sum_eva.to_csv(save_data_path, index=False, mode='a', header=0)

save_name = './NNPredictData/all_y_hat_lstm_' + str(Get_LinkNum())
np.savez(save_name + '.npz', all_y_hat=Data_ALL_Test)










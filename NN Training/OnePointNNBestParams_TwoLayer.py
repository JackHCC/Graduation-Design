import numpy as np
# from AllPointTrainDataProcess import Get_LinkNum
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd


# 加载数据
data = np.load('./TrainData/all_point_data_4.npz')

# 所有数据
# 维度说明：采样点个数*时间序列数*（Link_Num+1）
Data_ALL = data['all_point_data']
print(Data_ALL.shape)

# ----------------每次运行前注意预测长度是否修改-------------------------
def Get_LinkNum():
    return 4

# 采样点个数
Point_Num = len(Data_ALL[:])
# 时间序列长度
Time_Num = len(Data_ALL[0, :])
# print(Point_Num)

# 取第一个Point进行预测
point = 5555


Data_ALL_Trans = Data_ALL.copy()

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

# 训练集测试集分类
Split = 700
X_train_data = X_data[point, :Split, :]
Y_train_data = Y_data[point, :Split]
X_test_data = X_data[point, Split:, :]
Y_test_data = Y_data[point, Split:]


# 评指指标汇总
MAE_ALL = []
MSE_ALL = []
# EVS_ALL = []
R2_ALL = []

# 测试隐藏层神经元个数对模型训练影响
for Hidden_Point in range(1, 30):
    clf = MLPRegressor(solver='sgd', alpha=1e-4, learning_rate_init=0.01, hidden_layer_sizes=(Hidden_Point, 1), activation='relu',
                       max_iter=1000, random_state=1).fit(X_train_data, Y_train_data)
    y_train_hat = clf.predict(X_train_data)
    y_test_hat = clf.predict(X_test_data)
    # score = clf.score(X_test_data, Y_test_data)
    # print("score:", score)

    # 归一化还原
    test_data_trans = np.c_[X_test_data, y_test_hat]
    train_data_trans = np.c_[X_train_data, y_train_hat]
    data_all_trans = np.concatenate((train_data_trans, test_data_trans))

    # 数据还原
    data_all = min_max_scaler.inverse_transform(data_all_trans)

    y_raw = Data_ALL[point, :, Get_LinkNum()]
    y_train_raw = Data_ALL[point, :len(X_train_data), Get_LinkNum()]
    y_test_raw = Data_ALL[point, len(X_train_data):, Get_LinkNum()]

    y_train_hat = data_all[:len(X_train_data), Get_LinkNum()]
    y_test_hat = data_all[len(X_train_data):, Get_LinkNum()]

    # 计算训练集
    # Mean squared error（均方误差）
    MSE_Train = mean_squared_error(y_train_hat, y_train_raw)
    # Mean absolute error（平均绝对误差）,给定数据点的平均绝对误差，一般来说取值越小，模型的拟合效果就越好。
    MAE_Train = mean_absolute_error(y_train_hat, y_train_raw)
    # explained_variance_score：解释方差分，这个指标用来衡量我们模型对数据集波动的解释程度，如果取值为1时，模型就完美，越小效果就越差。
    # EVS_Train = explained_variance_score(y_train_hat, y_train_raw)
    # R2,R方可以理解为因变量y中的变异性能能够被估计的多元回归方程解释的比例，它衡量各个自变量对因变量变动的解释程度，其取值在0与1之间，其值越接近1，
    # 则变量的解释程度就越高，其值越接近0，其解释程度就越弱。一般来说，增加自变量的个数，回归平方和会增加，残差平方和会减少，所以R方会增大；
    # 反之，减少自变量的个数，回归平方和减少，残差平方和增加。
    R2_Train = r2_score(y_train_hat, y_train_raw)

    # 计算测试集
    MSE_Test = mean_squared_error(y_test_hat, y_test_raw)
    MAE_Test = mean_absolute_error(y_test_hat, y_test_raw)
    # EVS_Test = explained_variance_score(y_test_hat, y_test_raw)
    # R2_Test = r2_score(y_test_hat, y_test_raw)
    R2_Test = clf.score(X_test_data, Y_test_data)

    # MSE_Train = mean_squared_error(y_train_hat, Y_train_data)
    # MAE_Train = mean_absolute_error(y_train_hat, Y_train_data)
    # EVS_Train = explained_variance_score(y_train_hat, Y_train_data)
    # R2_Train = r2_score(y_train_hat, Y_train_data)
    #
    # MSE_Test = mean_squared_error(y_test_hat, Y_test_data)
    # MAE_Test = mean_absolute_error(y_test_hat, Y_test_data)
    # EVS_Test = explained_variance_score(y_test_hat, Y_test_data)
    # R2_Test = r2_score(y_test_hat, Y_test_data)

    MAE_ALL = np.append(MAE_ALL, MAE_Test)
    MSE_ALL = np.append(MSE_ALL, MSE_Test)
    # EVS_ALL = np.append(EVS_ALL, EVS_Test)
    R2_ALL = np.append(R2_ALL, R2_Test)

    print("Hidden = ", Hidden_Point, "|Test ERROR(MSE) = ", MSE_Test, "|Test MAE = ", MAE_Test, "|Test R2 = ", R2_Test,
          "|Train ERROR(MSE) = ", MSE_Train, "|Train MAE = ", MAE_Train, "|Train R2 = ", R2_Train)


Best_Hidden_Point_MSE = np.argmin(MSE_ALL)+1
print("MSE_min_index:(Hidden_Point)", Best_Hidden_Point_MSE, "|MSE_min:", MSE_ALL.min())
Best_Hidden_Point_R2 = np.argmax(R2_ALL)+1
print("R2_max_index:(Hidden_Point)", Best_Hidden_Point_R2, "|R2_max:", R2_ALL.max())

# 评价指标数据汇总
Sum_eva = np.r_[np.array(range(1, 30)).reshape(1, 29), MSE_ALL.reshape(1, len(MSE_ALL)), MAE_ALL.reshape(1, len(MAE_ALL)), R2_ALL.reshape(1, len(R2_ALL))]
# print(Sum_eva.shape, Sum_eva)

Sum_eva = pd.DataFrame(data=Sum_eva)
save_data_path = 'NNPredictData/Two_layer_MSE_MAE_R2.csv'
Sum_eva.to_csv(save_data_path, index=False, header=0)


# 画最优参数条形图

# label_list = [2, 3, 4, 5, 6, 7, 8, 9]
label_list = range(1, 30)
mse_list = MSE_ALL   # 纵坐标值1
mae_list = MAE_ALL     # 纵坐标值2
r2_list = R2_ALL      # 纵坐标值3

# 设置中文字体和负号正常显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 4

# label_list = ['2014', '2015', '2016', '2017']    # 横坐标刻度显示值
# num_list1 = [20, 30, 15, 35]      # 纵坐标值1
# num_list2 = [15, 30, 40, 20]      # 纵坐标值2
x = range(len(label_list))
"""
绘制条形图
left:长条形中点横坐标
height:长条形高度
width:长条形宽度，默认值0.8
label:为后面设置legend准备
"""
plt.figure(1, figsize=(8, 4), dpi=300)

width = 0.2

# rects1 = plt.bar(left=x, height=mse_list, width=width, alpha=0.8, color='white', edgecolor="k", label="MSE", hatch='///')
# rects2 = plt.bar(left=[i + width*1 for i in x], height=mae_list, width=width, color='white', edgecolor="k", label="MAE", hatch='xxx')
rects1 = plt.bar(left=x, height=mse_list, width=width, alpha=0.8, color='r', label="MSE")
rects2 = plt.bar(left=[i + width*1 for i in x], height=mae_list, width=width, color='g', label="MAE")
# rects3 = plt.bar(left=[i + width*2 for i in x], height=r2_list, width=width, color='b', label="R2")
# plt.ylim(0, 50)     # y轴取值范围
plt.ylabel("指标")
"""
设置x轴刻度显示值
参数一：中点坐标
参数二：显示值
"""
plt.xticks([index + 0.1 for index in x], label_list)
plt.xlabel("隐藏层神经元个数")
plt.title("两层神经网络下不同隐藏层神经元个数预测误差指标对比图")
plt.legend()


text_height = 0.002
# 编辑文本
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+text_height, str(int(height*1e2)/1e2), ha="center", va="bottom")
for rect in rects2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+text_height, str(int(height*1e2)/1e2), ha="center", va="bottom")
# for rect in rects3:
#     height = rect.get_height()
#     plt.text(rect.get_x() + rect.get_width() / 2, height+text_height, str(int(height*1e4)/1e4), ha="center", va="bottom")

# 图片存储在HotSpotPlot目录
plt_name = './NNPredictData/Two_layer_MSE_MAE'
plt.savefig(plt_name+'.png')
# plt.savefig(plt_name+'.svg')

# R2
plt.figure(2, figsize=(8, 4), dpi=300)
plt.rcParams['font.size'] = 10
plt.plot(label_list, r2_list, color="k", linewidth=1, marker='o', markeredgecolor='k', markersize='2',
         markeredgewidth=1)
plt.xlabel("隐藏层神经元个数")
plt.ylabel("R2 Score")
plt.title("两层神经网络下不同隐藏层神经元个数预测R2 Score指标对比图")

plt.show()







# 最优Hidden_Point下回归
# clf = MLPRegressor(solver='sgd', alpha=1e-4, learning_rate_init=0.01, hidden_layer_sizes=(Best_Hidden_Point, 1), activation='relu',
#                        max_iter=1000, random_state=1).fit(X_train_data, Y_train_data)
# y_test_hat = clf.predict(X_test_data)
#
# # 可视化预测结果与真实结果对比
# xx = range(0, len(Y_test_data))
# plt.figure(1, figsize=(8, 6))
# plt.scatter(xx, Y_test_data, color="red", label="Sample Point", linewidth=3)
# plt.plot(xx, y_test_hat, color="orange", label="Fitting Line", linewidth=2)
# plt.xlabel("Time")
# plt.ylabel("Value")
# plt.title("Predict of MLP(Best Hidden Point Num)")
# plt.legend()
# plt.show()


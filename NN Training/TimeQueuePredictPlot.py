import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Using Window Predict data predict Future

svr_MAE = pd.read_csv(r'./SVRPredictData/MAE_Time_Window_3.csv', header=None)
svr_MSE = pd.read_csv(r'./SVRPredictData/MSE_Time_Window_3.csv', header=None)
svr_R2 = pd.read_csv(r'./SVRPredictData/R2_Time_Window_3.csv', header=None)

mlp_MAE = pd.read_csv(r'./NNPredictData/MAE_Time_Window_3.csv', header=None)
mlp_MSE = pd.read_csv(r'./NNPredictData/MSE_Time_Window_3.csv', header=None)
mlp_R2 = pd.read_csv(r'./NNPredictData/R2_Time_Window_3.csv', header=None)

lstm_MAE = pd.read_csv(r'./NNPredictData/LSTM_MAE_Time_Window_3.csv', header=None)
lstm_MSE = pd.read_csv(r'./NNPredictData/LSTM_MSE_Time_Window_3.csv', header=None)
lstm_R2 = pd.read_csv(r'./NNPredictData/LSTM_R2_Time_Window_3.csv', header=None)


# data = pd.read_csv(r'./SVRPredictData/MAE_Time_Window_3.csv', header=None)


# print(data)

# Split = 700

SVR_MAE_data = np.squeeze(np.array(svr_MAE))
SVR_MSE_data = np.squeeze(np.array(svr_MSE))
SVR_R2_data = np.squeeze(np.array(svr_R2))
NN_MAE_data = np.squeeze(np.array(mlp_MAE))
NN_MSE_data = np.squeeze(np.array(mlp_MSE))
NN_R2_data = np.squeeze(np.array(mlp_R2))
LSTM_MAE_data = np.squeeze(np.array(lstm_MAE))
LSTM_MSE_data = np.squeeze(np.array(lstm_MSE))
LSTM_R2_data = np.squeeze(np.array(lstm_R2))

print(SVR_MAE_data, SVR_MAE_data, NN_MAE_data, NN_MSE_data)
print(SVR_MAE_data.shape, SVR_MAE_data.shape, NN_MAE_data.shape, NN_MSE_data.shape)

# 画图
# 用来正常显示中文标签,并设置字体大小
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 10

plt.figure(1, dpi=200, figsize=(8, 4))
# title_name = "短时预测模型下不同时间尺度性能对比图"
# plt.title(title_name)
plt.subplot(2, 1, 1)
x = range(0, len(SVR_MAE_data))
plt.plot(x, SVR_MAE_data, color='r', marker='o', linewidth=1, markeredgecolor='r', markersize='2',
         markeredgewidth=1, label='SVR')
plt.plot(x, NN_MAE_data, color='b', marker='o', linewidth=1, markeredgecolor='b', markersize='2',
         markeredgewidth=1, label='MLP')
plt.plot(x, LSTM_MAE_data, color='g', marker='o', linewidth=1, markeredgecolor='g', markersize='2',
         markeredgewidth=1, label='LSTM')
plt.xlabel("预测时间(×100s)")
plt.ylabel("MAE")
plt.xlim(0, len(SVR_MAE_data))
# title_name = "支持向量回归单点(经纬度:" + str(point_X) + "," + str(point_Y) + ")拟合效果与预测效果对比图"
# plt.title(title_name)
plt.legend()

plt.subplot(2, 1, 2)
x = range(0, len(SVR_MSE_data))
plt.plot(x, SVR_MSE_data, color='r', marker='o', linewidth=1, markeredgecolor='r', markersize='2',
         markeredgewidth=1, label='SVR')
plt.plot(x, NN_MSE_data, color='b', marker='o', linewidth=1, markeredgecolor='b', markersize='2',
         markeredgewidth=1, label='MLP')
plt.plot(x, LSTM_MSE_data, color='g', marker='o', linewidth=1, markeredgecolor='g', markersize='2',
         markeredgewidth=1, label='LSTM')
plt.xlabel("预测时间(×100s)")
plt.ylabel("MSE")
plt.xlim(0, len(SVR_MSE_data))

plt.legend()
# plt.show()

plt_name = './NNPredictData/TimeQueue_Predict_Compare_NoWindow'
plt.savefig(plt_name+'.png')
plt.savefig(plt_name+'.svg')


# Using Test data predict Future

# data_svr_mae = pd.read_csv(r'./SVRPredictData/MAE_Time_Window_3.csv', header=None)
# data_svr_mse = pd.read_csv(r'./SVRPredictData/MSE_Time_Window_3.csv', header=None)
# data_svr_r2 = pd.read_csv(r'./SVRPredictData/R2_Time_Window_3.csv', header=None)
#
# data_mlp_mae = pd.read_csv(r'./NNPredictData/MAE_Time_Window_3.csv', header=None)
# data_mlp_mse = pd.read_csv(r'./NNPredictData/MSE_Time_Window_3.csv', header=None)
# data_mlp_r2 = pd.read_csv(r'./NNPredictData/R2_Time_Window_3.csv', header=None)
#
# data_svr_mae = np.squeeze(np.array(data_svr_mae))
# data_svr_mse = np.squeeze(np.array(data_svr_mse))
# data_svr_r2 = np.squeeze(np.array(data_svr_r2))
# data_mlp_mae = np.squeeze(np.array(data_mlp_mae))
# data_mlp_mse = np.squeeze(np.array(data_mlp_mse))
# data_mlp_r2 = np.squeeze(np.array(data_mlp_r2))
#
#
# print(data_svr_r2.shape,data_mlp_r2.shape)
#
# plt.figure(2, dpi=200, figsize=(8, 4))
# plt.subplot(1, 3, 1)
# x = range(0, len(data_svr_mae))
# plt.plot(x, data_svr_mae, color='r', marker='o', linewidth=1, markeredgecolor='r', markersize='2',
#          markeredgewidth=1, label='SVR')
# plt.plot(x, data_mlp_mae, color='b', marker='o', linewidth=1, markeredgecolor='b', markersize='2',
#          markeredgewidth=1, label='MLP')
# plt.xlabel("时间(×100s)")
# plt.ylabel("MAE")
# # title_name = "支持向量回归单点(经纬度:" + str(point_X) + "," + str(point_Y) + ")拟合效果与预测效果对比图"
# # plt.title(title_name)
# plt.legend()
#
# plt.subplot(1, 3, 2)
# x = range(0, len(data_svr_mse))
# plt.plot(x, data_svr_mse, color='r', marker='o', linewidth=1, markeredgecolor='r', markersize='2',
#          markeredgewidth=1, label='SVR')
# plt.plot(x, data_mlp_mse, color='b', marker='o', linewidth=1, markeredgecolor='b', markersize='2',
#          markeredgewidth=1, label='MLP')
# plt.xlabel("时间(×100s)")
# plt.ylabel("MSE")
# # title_name = "支持向量回归单点(经纬度:" + str(point_X) + "," + str(point_Y) + ")拟合效果与预测效果对比图"
# # plt.title(title_name)
# plt.legend()
#
# plt.subplot(1, 3, 3)
# x = range(0, len(data_svr_r2))
# plt.plot(x, data_svr_r2, color='r', marker='o', linewidth=1, markeredgecolor='r', markersize='2',
#          markeredgewidth=1, label='SVR')
# plt.plot(x, data_mlp_r2, color='b', marker='o', linewidth=1, markeredgecolor='b', markersize='2',
#          markeredgewidth=1, label='MLP')
# plt.xlabel("时间(×100s)")
# plt.ylabel("R2")
#
# # title_name = "支持向量回归单点(经纬度:" + str(point_X) + "," + str(point_Y) + ")拟合效果与预测效果对比图"
# # plt.title(title_name)
# plt.legend()

plt.show()
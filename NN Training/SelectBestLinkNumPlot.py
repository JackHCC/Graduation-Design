import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 加载CSV评价指标数据
# data = pd.read_csv(r'./SVRPredictData/Sum_Params_C10.csv', header=None)

# 加载MLP评价指标数据
data = pd.read_csv(r'./NNPredictData/Sum_Params_v2.csv', header=None)
# print(data)

data = np.array(data)

# label_list = data[:8, 0]
label_list = [2, 3, 4, 5, 6, 7, 8, 9]
mse_list = data[:8, 1]   # 纵坐标值1
mae_list = data[:8, 4]     # 纵坐标值2
evs_list = data[:8, 7]   # 纵坐标值3
r2_list = data[:8, 10]      # 纵坐标值4

# print(label_list)


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
plt.figure(figsize=(8, 4), dpi=300)

# width = 0.3
#
# # rects1 = plt.bar(left=x, height=mse_list, width=width, alpha=0.8, color='white', edgecolor="k", label="MSE", hatch='///')
# # rects2 = plt.bar(left=[i + width*1 for i in x], height=mae_list, width=width, color='white', edgecolor="k", label="MAE", hatch='xxx')
# rects1 = plt.bar(left=x, height=mae_list, width=width, alpha=0.8, color='r', label="MAE")
# rects2 = plt.bar(left=[i + width*1 for i in x], height=mse_list, width=width, color='g', label="MSE")
# # plt.ylim(0, 50)     # y轴取值范围
# plt.ylabel("指标")
# """
# 设置x轴刻度显示值
# 参数一：中点坐标
# 参数二：显示值
# """
# plt.xticks([index + 0.15 for index in x], label_list)
# plt.xlabel("滑动窗口长度")
# plt.title("不同滑动窗口长度预测指标对比图")
# plt.legend()
#
#
# text_height = 0.002
# # 编辑文本
# for rect in rects1:
#     height = rect.get_height()
#     plt.text(rect.get_x() + rect.get_width() / 2, height+text_height, str(int(height*1e4)/1e4), ha="center", va="bottom")
# for rect in rects2:
#     height = rect.get_height()
#     plt.text(rect.get_x() + rect.get_width() / 2, height+text_height, str(int(height*1e4)/1e4), ha="center", va="bottom")
#
#
# plt.plot(range(len(label_list)), mae_list, color="b", linewidth=1, marker='o', markeredgecolor='b', markersize='2',
#          markeredgewidth=1, alpha=0.7)




width = 0.24

rects1 = plt.bar(left=x, height=mse_list, width=width, alpha=0.8, color='red', label="MSE")
rects2 = plt.bar(left=[i + width*1 for i in x], height=mae_list, width=width, color='green', label="MAE")
rects3 = plt.bar(left=[i + width*2 for i in x], height=evs_list, width=width, color='blue', label="EVS")
rects4 = plt.bar(left=[i + width*3 for i in x], height=r2_list, width=width, color='orange', label="R2 Score")
# plt.ylim(0, 50)     # y轴取值范围
plt.ylabel("指标")
"""
设置x轴刻度显示值
参数一：中点坐标
参数二：显示值
"""
plt.xticks([index + 0.36 for index in x], label_list)
plt.xlabel("滑动窗口长度")
plt.title("不同滑动窗口长度预测指标对比图")
plt.legend()


text_height = 0.01
# 编辑文本
for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+text_height, str(int(height*1e4)/1e4), ha="center", va="bottom")
for rect in rects2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+text_height, str(int(height*1e4)/1e4), ha="center", va="bottom")
for rect in rects3:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+text_height, str(int(height*1e4)/1e4), ha="center", va="bottom")
for rect in rects4:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+text_height, str(int(height*1e4)/1e4), ha="center", va="bottom")

# 图片存储在SVR目录
# plt_name = './SVRPredictData/Window_MSE_MAE_Point'
# plt.savefig(plt_name+'.png')
# plt.savefig(plt_name+'.svg')

# 储存在NN目录
# plt_name = './NNPredictData/NN_Window_MSE_MAE_EVS_R2_Point'
# plt.savefig(plt_name+'.png')

# plt.savefig(plt_name+'.svg')

plt.show()

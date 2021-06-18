import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas
import plot_map
import seaborn as sns

# 读取GPS数据
# data = pd.read_csv(r'RawGPSData.csv', header=None)
# data.columns = ['VehicleNum', 'Stime', 'Lng', 'Lat']
# print(data.head(5))

# 获取时间为10的所有样本车辆
# data_10 = data[data['Stime'] == 10]
# print(data_10)

# 读取shapefile文件
# shp = r'../Beijing Map Data/北京市.shp'
# bj = geopandas.GeoDataFrame.from_file(shp, encoding='utf-8')

# 绘制北京市地图
# bj.plot()
# plt.show()

# 北京市经纬度边界
bounds1 = [115.7, 39.4, 117.4, 41.6]
bounds2 = [116.0, 39.6, 116.8, 40.2]
# bounds = [116.15, 39.7, 116.6, 40.1]

# fig = plt.figure(dpi=300)
# ax = plt.subplot(2, 1, 1)
plt.subplot(1, 2, 1)
# plt.sca(ax)
# fig.tight_layout(rect=(0.05, 0.1, 1, 0.9))

# 背景
plot_map.plot_map(plt, bounds1, zoom=12, style=3)

ax = plt.gca()
# 默认框的颜色是黑色，第一个参数是左上角的点坐标
# 第二个参数是宽，第三个参数是长
ax.add_patch(plt.Rectangle((116.0, 39.6), 0.8, 0.6, color="red", fill=False, linewidth=1, alpha=0.8))


#plot scatters
# cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#9DCC42', '#FFFE03', '#F7941D', '#E9420E', '#FF0000'], 256)

# plt.scatter(data_10['Lng'], data_10['Lat'], s=1, alpha=0.6)
plt.axis('off')
plt.xlim(bounds1[0], bounds1[2])
plt.ylim(bounds1[1], bounds1[3])


# fig = plt.figure(dpi=300)
plt.subplot(1, 2, 2)
# ax = plt.subplot(2, 1, 2)
# plt.sca(ax)
# fig.tight_layout(rect=(0.05, 0.1, 1, 0.9))

# 背景
plot_map.plot_map(plt, bounds2, zoom=12, style=3)

#plot scatters
# cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#9DCC42', '#FFFE03', '#F7941D', '#E9420E', '#FF0000'], 256)

# plt.scatter(data_10['Lng'], data_10['Lat'], s=1, alpha=0.6)
plt.axis('off')
plt.xlim(bounds2[0], bounds2[2])
plt.ylim(bounds2[1], bounds2[3])

plt.subplots_adjust(top=1, bottom=0, right=0.96, left=0.04, hspace=0, wspace=0.05)
plt.margins(0, 0)

#定义colorbar位置
# cax = plt.axes([0.13, 0.32, 0.02, 0.3])
#绘制热力图
# sns.kdeplot(data['Lng'], data['Lat'],
#             alpha=0.8, #透明度
#             gridsize=100, #绘图精细度，越高越慢
#             bw=0.03,   #高斯核大小（经纬度），越小越精细
#             shade=True,
#             shade_lowest=False,
#             cbar=True,
#             cmap=cmap,
#             ax=ax,  #指定绘图位置
#             cbar_ax=cax  #指定colorbar位置
#            )

# 加比例尺和指北针
# plot_map.plotscale(ax, bounds=bounds, textsize=10, compasssize=1, accuracy=2000, rect=[0.06, 0.03])

plt_name = './BeijingMap/beijing_all_to_116.0_39.6_116.8_40.2'
plt.savefig(plt_name + '.svg')
plt.savefig(plt_name + '.png')

plt.show()

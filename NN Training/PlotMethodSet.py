import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learn_curve(estimator, title, X, y, ylim = None, cv=None, n_jobs=-1, train_sizes=np.linspace(.1, 1., 5)):
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("train exs")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_score_mean = np.mean(train_scores, axis=1)
    train_score_std = np.std(train_scores, axis=1)
    test_score_mean = np.mean(test_scores, axis=1)
    test_score_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_score_mean - train_score_std,
                     train_score_mean + train_score_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_score_mean - test_score_std,
                     test_score_mean + test_score_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_score_mean, 'o-', color='r', label='train score')
    plt.plot(train_sizes, test_score_mean, 'o-', color='g', label='cross-validation score')

    plt.legend(loc='best')
    return plt


# 预测数据Predict_data：Point_Num*Test时序长度，图序号Figure_Num
def plot_contour_predict(Predict_data, Plot_Time = 0, Figure_Num = None):
    # 采样点个数
    Point_Num = len(Predict_data[:])
    # print(Point_Num)

    # 加载经纬度数据
    data1 = np.load('../KDE/all_f_data.npz')
    # 经纬度数据提取
    Data_ALL = data1['all_data'][0, :, :2]
    print(Data_ALL.shape)

    # 将两组数据合并
    Predict_Con_data = np.c_[Data_ALL, Predict_data[:, Plot_Time, 5]]
    print(Predict_Con_data.shape)

    # 从所有数据中经纬度数据极值提取
    MaxX = np.max(Data_ALL[:, 0])
    MinX = np.min(Data_ALL[:, 0])
    MaxY = np.max(Data_ALL[:, 1])
    MinY = np.min(Data_ALL[:, 1])

    Sample_Num = int(np.sqrt(Point_Num))
    # 获取采样坐标点
    Contour_X = Data_ALL[:, 0].reshape(Sample_Num, Sample_Num)
    Contour_Y = Data_ALL[:, 1].reshape(Sample_Num, Sample_Num)
    print(Contour_X.shape, Contour_Y.shape)

    Contour_Z = np.reshape(Predict_data[:, Plot_Time, 5], Contour_X.shape)

    # 作图
    fig = plt.figure(Figure_Num)
    axis = fig.gca()
    axis.set_xlim(MinX, MaxX)
    axis.set_ylim(MinY, MaxY)

    # contourf_set = axis.contourf(Contour_X, Contour_Y, Contour_Z, alpha=0.7, levels=np.linspace(0, Contour_Z.max(), 12), cmap='Reds')
    contourf_set = axis.contourf(Contour_X, Contour_Y, Contour_Z, alpha=0.7, cmap='Reds')

    # 设置colorbar
    cbar = fig.colorbar(contourf_set, fraction=0.05, pad=0.05)
    cbar.set_clim(0, Contour_Z.max())

    # contour_set = axis.contour(Contour_X, Contour_Y, Contour_Z, colors='k')
    # axis.clabel(contour_set, inline=0.2, fontsize=0.5)

    axis.set_xlabel('Longitude')
    axis.set_ylabel('Latitude')

    # plt_name = './KDE_Random'
    # plt.savefig(plt_name+'.jpg')
    return plt

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# 核密度估计，输入：输入数据|采样矩阵|最适带宽；输出：KDE模型，采样矩阵获取的密度分布矩阵
def KDE(Input_Data, Sample_Metric, Bandwidth):
    kde = KernelDensity(kernel='gaussian', bandwidth=Bandwidth).fit(Input_Data)

    # some method of KDE:
    # fit(X, y=None, sample_weight=None) Fit the Kernel Density model on the data.
    # get_params(deep=True) Get parameters for this estimator.
    model_params = kde.get_params(deep=True)

    # sample(n_samples=1, random_state=None) Generate random samples from the model.
    # sample_point=kde.sample(10)

    # score(X, y=None) Compute the total log probability density under the model.
    # log_dense_total=kde.score(X)

    # score_samples(X) Evaluate the log density model on the data.
    log_dense = kde.score_samples(Sample_Metric)

    dense = np.exp(log_dense)

    return kde, dense


# 根据密度数据画等高线图
# 输入：X，Y最大最小值，采样间隔，kde模型，i标识（用于遍历区分）
def PlotContour(MinX, MaxX, MinY, MaxY, Sample_Space_X, Sample_Space_Y, kde, plotbool=False, savebool=False, picname='', i=''):
    # 画核密度等高线图

    Contour_X, Contour_Y = np.mgrid[MinX:MaxX:Sample_Space_X, MinY:MaxY:Sample_Space_Y]  # 步长实数为间隔，虚数为个数
    positions = np.vstack([Contour_X.ravel(), Contour_Y.ravel()])

    Contour_Z = np.reshape(np.exp(kde.score_samples(positions.T)), Contour_X.shape)

    # 作图
    fig = plt.figure()
    axis = fig.gca()
    axis.set_xlim(MinX, MaxX)
    axis.set_ylim(MinY, MaxY)

    contourf_set = axis.contourf(Contour_X, Contour_Y, Contour_Z, cmap='Reds')

    contour_set = axis.contour(Contour_X, Contour_Y, Contour_Z, colors='k')
    axis.clabel(contour_set, inline=1, fontsize=10)

    axis.set_xlabel('Longitude')
    axis.set_ylabel('Latitude')


    plt_name = picname

    if savebool == True:
        plt.savefig(plt_name + str(i) + '.jpg')
        plt.close()
    else:
        plt.close()

    if plotbool == True:
        plt.show()

# 文件存储
# 输入：采样矩阵，密度矩阵，文件名，标识i
def SaveNPZ(Sample_Metric, dense, filename, i=''):
    feature_data = np.c_[Sample_Metric, dense]

    f_name = filename
    np.savez(f_name + str(i) + '.npz', feature_data=feature_data)
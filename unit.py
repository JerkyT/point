import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import yaml
from src.modelnetC import ModelNet40_C
from src.modelnetC6 import ModelNet40_C6
from src.modelnet import ModelNet40Ply2048
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde

np.set_printoptions(threshold=np.inf)

def normalize_array(arr):
    # 矩阵归一化
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr

def normalize_positive(matrix):
    positive_values = matrix[matrix > 0]
    min_val = np.min(positive_values)
    max_val = np.max(positive_values)
    matrix[matrix > 0] = (matrix[matrix > 0] - min_val) / (max_val - min_val)
    return matrix

def draw(dir_path):
    res_yaml = yaml.load(open(os.path.dirname(dir_path[:-1]) + '/res.yaml', 'r', encoding="utf-8").read(), Loader=yaml.FullLoader)
    res_yaml2 = yaml.load(open(os.path.dirname(dir_path[:-1]).replace('_ST', '_ATDKD') + '/res.yaml', 'r', encoding="utf-8").read(), Loader=yaml.FullLoader)
    shapleys = {}

    # 循环遍历目录及其子目录下的所有 .npy 文件
    for file_path in glob.iglob(os.path.join(dir_path, '**/*_p.npy'), recursive=True):
        file_path = file_path.replace('\\', '/')
        key = file_path.split('/')[-2][0:-2]
        level = file_path.split('/')[-2].split('_')[-1]
        if not key in shapleys:
            shapleys[key] = {}

        shapleys[key][level] = file_path
        l = ["rotation", "shear", "distortion", "distortion_rbf", "distortion_rbf_inv",
         "uniform", "upsampling", "gaussian", "impulse", "background",
        "occlusion", "lidar", "density_inc", "density", "cutout"]

    for i, key in enumerate(l):
        x, y, y2 = [], [], []
        x_oa, y_oa, y2_oa = [], [], []
        for level in shapleys[key]:
            shapley = np.load(shapleys[key][level]).mean(-1) * 10
            shapley_DKD = np.load(shapleys[key][level].replace('_ST', '_ATDKD').replace('test100', 'test1000')).mean(-1) * 10
            # print(shapley_DKD.shape)
            # shapley = normalize_positive(shapley)
            # shapley_DKD = normalize_positive(shapley_DKD)
            # shapley = shapley[sources > 0.5]
            # shapley_DKD = shapley_DKD[sources2 > 0.5]

            y.append(np.mean(shapley))
            y2.append(np.mean(shapley_DKD))
            x.append(int(level))

            y_oa.append(res_yaml[key + '-' + level]['OA'])
            y2_oa.append(res_yaml2[key + '-' + level]['OA'])
            x_oa.append(level)
        x, y, y2 = np.array(x), np.array(y), np.array(y2)

        plt.subplot(3, 5, i+1)
        plt.bar(x, y, 0.25)
        plt.bar(x + 0.25, y2, 0.25)
        # plt.bar(x, abs(y - 0.008), 0.25)
        # plt.bar(x + 0.25, abs(y2 - 0.008), 0.25)
        # plt.ylim(-0.005, 0.015)
        plt.title(key)

    plt.subplots_adjust(wspace = 0.3, hspace = 0.3) # 调整子图间距
    plt.show()

def draw_line_chart(dir_path):
    res_yaml = yaml.load(open(os.path.dirname(dir_path) + '/res.yaml', 'r', encoding="utf-8").read(), Loader=yaml.FullLoader)
    res_yaml2 = yaml.load(open(os.path.dirname(dir_path).replace('_ST', '_ATDKD') + '/res.yaml', 'r', encoding="utf-8").read(), Loader=yaml.FullLoader)

    shapleys = {}

    # 循环遍历目录及其子目录下的所有 .npy 文件
    for file_path in glob.iglob(os.path.join(dir_path, '**/*_s.npy'), recursive=True):
        file_path = file_path.replace('\\', '/')
        key = file_path.split('/')[-2][0:-2]
        level = file_path.split('/')[-2].split('_')[-1]
        if not key in shapleys:
            shapleys[key] = {}

        shapleys[key][level] = file_path

    keys = ["rotation", "shear", "distortion", "distortion_rbf", "distortion_rbf_inv",
         "uniform", "upsampling", "gaussian", "impulse", "background",
        "occlusion", "lidar", "density_inc", "density", "cutout"]

    for i, key in enumerate(keys):
        x, y_oa, y2_oa = [], [], []
        s1 , s2 = [], []
        for level in shapleys[key]:
            x.append(int(level))
            # y_oa.append(res_yaml[key + '-' + str(level)]['OA'])
            # y2_oa.append(res_yaml2[key + '-' + str(level)]['OA'])

            sources = np.load(shapleys[key][level])
            sources2 = np.load(shapleys[key][level].replace('_ST', '_ATDKD'))

            s1.append(np.mean(sources))
            s2.append(np.mean(sources2))
            
        plt.subplot(3, 5, i + 1)
        plt.plot(x, s1, marker='o', linewidth=1, linestyle = '-', alpha = 1)
        plt.plot(x, s2, marker='o', linewidth=1, linestyle = '-', alpha = 1)
        plt.title(key)

    plt.subplots_adjust(wspace = 0.3, hspace = 0.3) # 调整子图间距
    plt.show()

def draw_histogram(dir_path, severity = 5):
    shapleys = {}

    # 循环遍历目录及其子目录下的所有 .npy 文件
    for file_path in glob.iglob(os.path.join(dir_path, '**/*_s.npy'), recursive=True):
        file_path = file_path.replace('\\', '/')
        key = file_path.split('/')[-2][0:-2]
        level = file_path.split('/')[-2].split('_')[-1]
        if not key in shapleys:
            shapleys[key] = {}

        shapleys[key][level] = file_path

    keys = ["rotation", "shear", "distortion", "distortion_rbf", "distortion_rbf_inv",
         "uniform", "upsampling", "gaussian", "impulse", "background",
        "occlusion", "lidar", "density_inc", "density", "cutout"]

    for i, key in enumerate(keys):
        sources = np.load(shapleys[key][str(severity)])
            
        plt.subplot(3, 5, i + 1)
        plt.hist(sources, bins=30, color='lightgray', alpha=0.7, label='Data Histogram')
        plt.title(key)

    plt.subplots_adjust(wspace = 0.3, hspace = 0.3) # 调整子图间距
    plt.show()

def draw2(dataset = '15', keys = ['rotation'], color_interval = np.array([[0, 0, 0], [255, 255, 255]]), interval = [-0.01, 0.1], index = 1, dir_path = './PointNet2Encoder_TR/test100/', type = '3d', fig = None, h = 1, w = 6, s = 0, mm = None):
    mean, min_max, weights, point_clouds, sources = [], [], [], [], []
    weights_clean = np.load('./data/shaply_ModelNet40Ply2048/PointNetEncoder_ST/test100/final.npy')[index].mean(1) * 10
    # weights_clean = normalize_array(weights_clean)
    # colors = linear_calculation(weights_clean, interval, color_interval)
    if fig == None:
        fig = plt.figure(figsize=(60, 10))
    
    for i, key in enumerate(keys):
        for j in range(1, 6):
            if dataset == '15':
                modelnet_c = ModelNet40_C(corruption = key, severity = j)
                weight = np.load(dir_path + key + '_' + str(j) + '/final_p.npy')[index].mean(1) * 10
                source = np.load(dir_path + key + '_' + str(j) + '/final_s.npy')[index]
                # print(key, ' ', np.argmin(source))
                # exit()
            else:
                modelnet_c = ModelNet40_C6(corruption = key, severity = j - 1)
                weight = np.load(dir_path + key + '_' + str(j - 1) + '/final_p.npy')[index].mean(1) * 10
                source = np.load(dir_path + key + '_' + str(j - 1) + '/final_s.npy')[index]
            # weight = normalize_array(weight)
            point_clouds.append(modelnet_c[index]['pos'])
            
            weights.append(weight)
            min_max.append(weight.min())
            min_max.append(weight.max())
            mean.append(weight.mean())
            sources.append(source)
        if not mm==None:
            min_max = mm

        step = (max(min_max) - min(min_max)) / len(color_interval)
        interval = [min(min_max) + step * s for s in range(len(color_interval))]

        modelnet = ModelNet40Ply2048()
        point_clean = modelnet[index]['pos']
        ax = fig.add_subplot(h, w, s + 1, projection='3d')
        colors = linear_calculation(weights_clean, interval, color_interval)
        ax.scatter(point_clean[:, 2], point_clean[:, 0], point_clean[:, 1], c = colors / 255, s = 6)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        ax.axis('off')
        # ax.set_title('clean' + ' ' + str(round(weights_clean.mean(),2)))
        # ax.set_title(str(round(max(min_max), 4)) + ' ' + str(round(min(min_max), 4)))

        for j, weight in enumerate(weights):
            point_cloud = point_clouds[j]
            if type == '3d':
                # interval = [min(min_max), max(min_max)]
                # colors = np.zeros_like(point_cloud)
                # colors[weight <= 0.02] = np.array([0, 0, 0])
                # colors[weight > 0.02] = np.array([255, 255, 255])
                colors = linear_calculation(weight, interval, color_interval)
                ax = fig.add_subplot(h, w, s + i * 5 + j + 1 + 1, projection='3d')
                ax.scatter(point_cloud[:, 2], point_cloud[:, 0], point_cloud[:, 1], c = colors / 255, s = 6)
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_zlim(-1, 1)

                ax.axis('off')

            ax.set_title(key + " " + str(round(sources[j], 4)))

        mean, min_max, weights, point_clouds, sources = [], [], [], [], []
    # plt.subplots_adjust(wspace=-0.2, hspace=-0.37)
    # plt.tight_layout()
    cbar_ax = fig.add_axes([0.92, 0.25, 0.005, 0.50])
    cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', color_interval / 255, N=256)
    # norm = BoundaryNorm(interval, cmap.N + 1)

    norm = Normalize(vmin=mm[0], vmax=mm[1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, extend='both', shrink=0.1, cax=cbar_ax) #, orientation='horizontal')
    plt.yticks(fontsize=16)
    # cbar.ax.set_xlabel('Value') # 设置颜色条标签

    # 显示图像
    # if count >= 3:
    # save_path = './' + keys[0]
    # if not os.path.exists(save_path): 
    #     os.makedirs(save_path)
    return fig

    # plt.savefig(save_path + '/' + str(index) + '.png')
    # if fig == None:
    #     plt.show()
    #     plt.close()

def linear_calculation(weights, interval, color_interval):
    """
    weights.shape: 1024
    x = [0 , 1, 2, 3]
    rgb = [ [0 , 0 , 0], [128, 128, 128], [96, 96, 96], [255, 255, 255]]
    """
    ks, bs = [], []
    for i in range(len(interval) - 1):
        k = (color_interval[i + 1] - color_interval[i]) / (interval[i + 1] - interval[i]) # [k1, k2 , k3]
        b = color_interval[i] - k * interval[i]

        ks.append(k)
        bs.append(b)
    """
    ks: [ [[k11, k12 , k13]], [k21, k22 , k23], [k31, k32 , k33]]
    bs: [ [[b11, b12 , b13]], [b21, b22 , b23], [b31, b32 , b33]]
    """
    j = np.zeros_like(weights)

    j[weights < interval[0]] = 0
    j[weights > interval[-1]] = len(interval)

    for i in range(len(interval) - 1):
        j[(weights > interval[i]) & (weights < interval[i + 1])] = i + 1

    x = np.zeros((weights.shape[0], 3))
    x[j == 0] = color_interval[0]
    x[j == len(interval)] = color_interval[-1]

    for i in range(1, len(interval)):
        w = weights[j == i][:, np.newaxis]
        w = np.concatenate([w, w, w], -1)
        x[j == i] = ks[i - 1] * w + bs[i - 1]

    return x

def draw_one(point_clouds, weights = [], sources = [], color_interval = None, save_path = './1.png'):
    # 创建一个3D图形对象
    fig = plt.figure()
    if len(weights) == 0:
        # 提取点的坐标
        point_cloud = point_clouds

        x = point_cloud[:, 2]
        y = point_cloud[:, 0]
        z = point_cloud[:, 1]
        ax = fig.add_subplot(111, projection='3d')

        # 绘制点云
        ax.scatter(x, y, z, marker='.')

        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        # 显示图形
        plt.show()
        return
    row = 2

    for i, weight in enumerate(weights):
        point_cloud = point_clouds[i // 2]
        x = point_cloud[:, 2]
        y = point_cloud[:, 0]
        z = point_cloud[:, 1]

        ax = fig.add_subplot(len(weights) // row + 1 if len(weights) % row > 0 else len(weights) // row, row if len(weights) >= row else len(weights), i + 1, projection='3d')

        step = (max(weight) - min(weight)) / len(color_interval)
        interval = [min(weight) + step * s for s in range(len(color_interval))]

        colors = linear_calculation(weight, interval, color_interval) / 255

        # 绘制点云
        ax.scatter(x, y, z, c = colors, marker='.', s = 20)

        # 设置坐标轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.axis('off')
        ax.set_title(str( np.round(sources[i], 2)) )

    # cbar_ax = fig.add_axes([0.92, 0.25, 0.005, 0.50])
    # cbar_ax = fig.add_axes([0.92, 0.25, 0.005, 0.50])
    # cmap = mcolors.LinearSegmentedColormap.from_list('my_cmap', color_interval / 255, N=256)
    # norm = Normalize(vmin=min(weight), vmax=max(weight))
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # plt.colorbar(sm, ax=ax, extend='both', shrink=0.4, cax=cbar_ax, orientation='horizontal')
    # plt.subplots_adjust(wspace = -0.0, hspace = 0.0) # 调整子图间距

    # 保存图片并截取指定区域
    
    # 显示图形
    # plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    # plt.close()

def normal_distribution(datas, fig = None, labels = [], color = '#ffa556', show = False):
    # 计算正态分布的参数
    log = []
    labels = ['Clean', 'corruptions']
    if fig is None:
        fig, ax = plt.figure()

    for i, data in enumerate(datas):
        if len(labels) == 0:
            plt.hist(data, bins=30, color = color[i], alpha=0.3)
        else:
            plt.hist(data, bins=30, color = color[i], alpha=0.3, label=labels[i])

    # 绘制直方图
    ax1 = plt.twinx()
    for i, data in enumerate(datas):
        mean = np.mean(data)
        std = np.std(data)

        # 生成一系列x值，用于绘制曲线
        # x = np.linspace(mean - 3 * std, mean + 3 * std, 100)
        x = np.linspace(0, 1, 100)
        # 计算每个x值对应的高斯分布的y值
        y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

        kde = gaussian_kde(data)

        # x = np.linspace(0, 1, 100)
        ax1.plot(x, y, color = color[i])

        plt.text(mean + 0.02, (0.1 + 0.75 * i) * max(y), f"Mean: {mean:.2f}\nStd: {std:.2f}", color = color[i], ha='left', fontweight='bold')
        plt.axvline(mean, color = color[i], linestyle = '--')
        # 高斯核
        # ax1.plot(x, kde(x), color = color[i])
        log.append(y)
        ax1.set_xlim(0 , 1)

    kl_divergence = np.sum(np.where(log[0] != 0, log[0] * np.log(log[0] / log[1]), 0))
    plt.title(str(round(kl_divergence, 2)))
    if show:
        # 显示图形
        plt.show()

    return

# 定义拟合函数
def linear_function(x, a, b, c):
    return a * x + b

# 定义非线性函数
def nonlinear_function(x, a, b, c):
    return a * np.sin(b * x) + c

# 拟合曲线
from scipy.optimize import curve_fit

def fit(x, y):
    index = np.argsort(x)
    x = x[index]
    y = y[index]
    z1 = np.polyfit(x, y, 3) # 用3次多项式拟合
    p1 = np.poly1d(z1)
    yvals=p1(x) # 也可以使用yvals=np.polyval(z1,x)
    plt.plot(x, yvals, 'r',label='polyfit values')
    # 绘制散点图和拟合曲线
    plt.scatter(x, y, color='blue', label='Data', s = 5)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Scatter Plot with Fit')
    plt.legend()

    plt.show()
    plt.close()

def get_class_w(weight):
    # 获取每一个类的shapely矩阵
    from labels import dic
    pkl = {}
    for i in range(weight.shape[0]):
        key = str(dic[str(i)])
        if key in pkl:
            pkl[key].append(weight[i])
        else:
            pkl[key] = [weight[i], ]
    return pkl

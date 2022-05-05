import os
import sklearn
import warnings
import copy
import random
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
%matplotlib inline

file_dir = './data'
csv_files = os.listdir(file_dir)
print(csv_files)

# df 作为最后输出的 DataFrame 初始化为空
df = pd.DataFrame()
feature = ['cpc', 'cpm']
df_features = []
for col in feature:
    infix = col + '.csv'
    path = os.path.join(file_dir, infix)
    df_feature = pd.read_csv(path)
    # 将两个特征存储起来用于后续连接
    df_features.append(df_feature)

# 2 张 DataFrame 表按时间连接
df = pd.merge(left=df_features[0], right=df_features[1])
df.head()

# 获取 df 数据信息
df.info()

df.describe()

# 将 timestamp 列转化为时间类型
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 将 df 数据按时间序列排序，方便数据展示
df = df.sort_values(by='timestamp').reset_index(drop=True)
df.head()

# 按照时间轴绘制 cpc 和 cpm 指标数据
fig, axes = plt.subplots(2, 1)
df.plot(kind='line', x='timestamp', y='cpc', figsize=(20, 10), ax=axes[0], title='cpc', color='g')
df.plot(kind='line', x='timestamp', y='cpm', figsize=(20, 10), ax=axes[1], title='cpm', color='b')
plt.tight_layout();

sns.heatmap(df.corr(), cmap='coolwarm', annot=True);

# 绘制 cpc 和 cpm 的散点图，其中横坐标是 cpc，纵坐标是 cpm
plt.scatter(x=df['cpc'],y=df['cpm'],alpha=0.5);
# 绘制数据集中心点，横坐标是 cpc 的平均值，纵坐标是 cpm 的平均值 
plt.scatter(x=df['cpc'].mean(),y=df['cpm'].mean(),c='red',alpha=0.8);

def simple_distance(data):
    """
    计算当前点（cpc，cpm）到（cpc_mean，cpm_mean）的几何距离（L2 范数）
    :param data: ataDFrame 包含cpc、cpm 列
    :return: Series，每一列cpc、cpm到平均值值点的距离大小
    """
    mean = np.array([data['cpc'].mean(), data['cpm'].mean()])
    distance = []
    for i in range(0, len(data)):
        point = np.array(data.iloc[i, 1:3])
        # 求当前点（cpc，cpm）到平均值点（cpc_mean，cpm_mean）之间的几何距离（L2 范数）
        distance.append(np.linalg.norm(point - mean))
    distance = pd.Series(distance)
    return distance

df['distance'] = simple_distance(df[df.columns])
df.head()

df.describe()

# 按照时间轴绘制 cpc 和 cpm 指标数据
fig,axes = plt.subplots(3,1)
df.plot(kind='line', x = 'timestamp', y = 'cpc',      figsize=(20,10), ax = axes[0], title='cpc')
df.plot(kind='line', x = 'timestamp', y = 'cpm',      figsize=(20,10), ax = axes[1], title='cpm')
df.plot(kind='line', x = 'timestamp', y = 'distance', figsize=(20,10), ax = axes[2], title='distance')
plt.tight_layout()

ratio = 0.005
num_anomaly = int(len(df) * ratio)
threshould = df['distance'].sort_values(ascending=False).reset_index(drop=True)[num_anomaly]
print('阈值距离：' + str(threshould))

df['is_anomaly'] = df['distance'].apply(lambda x: x > threshould)
df.head()

normal = df[df['is_anomaly'] == 0]
anormal = df[df['is_anomaly'] == 1]
plt.scatter(x = normal['cpc'], y = normal['cpm'], c = 'blue', alpha = 0.5)
plt.scatter(x = anormal['cpc'],y = anormal['cpm'],c = 'red', alpha=0.5);

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['cpc','cpm']] = scaler.fit_transform(df[['cpc','cpm']])
df.describe()

# 假设异常数据比例
ratio = 0.005
num_anomaly = int(len(df) * ratio)
df['distance2'] = simple_distance(df)
threshould = df['distance2'].sort_values(ascending=False).reset_index(drop=True)[num_anomaly]
print('阈值距离：'+str(threshould))
df['is_anomaly2'] = df['distance2'].apply(lambda x: x > threshould)
normal = df[df['is_anomaly2'] == 0]
anormal = df[df['is_anomaly2'] == 1]
plt.scatter(x=normal['cpc'],y=normal['cpm'],c='blue',alpha=0.5)
plt.scatter(x=anormal['cpc'],y=anormal['cpm'],c='red',alpha=0.5);

a = df.loc[df['is_anomaly2'] == 1, ['timestamp', 'cpc']] 
plt.figure(figsize=(20,10))
plt.plot(df['timestamp'], df['cpc'], color='blue')
plt.scatter(a['timestamp'],a['cpc'], color='red')
plt.show()

# 尝试引入非线性关系
df['cpc X cpm'] = df['cpm'] * df['cpc']
df['cpc / cpm'] = df['cpc'] / df['cpm']

# 尝试获取时间关系
df['hours'] = df['timestamp'].dt.hour
df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)

df1 = df[['cpc','cpm','cpc X cpm','cpc / cpm','hours','daylight']]
df1.corr()

sns.heatmap(df1.corr(),cmap='coolwarm',annot=True);

from sklearn.decomposition import PCA

#在进行特征变换之前先对各个特征进行标准化
# columns = ['cpc', 'cpm', 'cpc X cpm', 'cpc / cpm']\
columns = ['cpc','cpm','cpc X cpm','cpc / cpm','hours','daylight']
# print(df)
data = df[columns]
# print(data)
scaler = StandardScaler()
data = scaler.fit_transform(data)
data = pd.DataFrame(data, columns=columns)

#通过 n_components 指定需要降低到的维度
n_components =3
pca = PCA(n_components=n_components)
data = pca.fit_transform(data)
print(data)
data = pd.DataFrame(data,columns=['D' + str(i+1) for i in range(n_components)])
# data.head()


# d=data.values
# s=np.sqrt(np.sum((d[0,:] - d[1,:]) ** 2))
# print(d[0,:])
# print(d[1,:])
# print(s)
print(data)

# 特征变换空间（特征矩阵），根据我们指定的n_components = k的值，选择方差最大的 k 个值所对应的的特征向量组成的特征矩阵。
print(pd.DataFrame(pca.components_, columns=['Dimension' + str(i + 1) for i in range(pca.n_features_)]))

# 每个保留特征占所有特征的方差百分比
pca.explained_variance_ratio_

# 保留 n 个特征的方差
pca.explained_variance_

var_explain = pca.explained_variance_ratio_
# 梯形累计和，axis=0，按照行累加。axis=1，按照列累加。axis不给定具体值，就把数组当成一个一维数组
cum_var_explian = np.cumsum(var_explain)

plt.figure(figsize=(10, 5))
plt.bar(range(len(var_explain)), var_explain, alpha=0.3, align='center', label='the interpreted independently variance ')
plt.step(range(len(cum_var_explian)), cum_var_explian, where='mid', label='the cumulative interpretation variance')
plt.ylabel('the explain variance rate')
plt.xlabel('PCA')
plt.legend(loc='best')
plt.show()

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(1,figsize=(6,6))
ax = Axes3D(fig)
ax.scatter(data.iloc[:,0],data.iloc[:,1],data.iloc[:,2],edgecolors='k');

from sklearn.metrics import calinski_harabasz_score,silhouette_score
from itertools import chain
class KMeans():


    """
    Parameters
    ----------
    n_clusters 指定了需要聚类的个数，这个超参数需要自己调整，会影响聚类的效果
    n_init 指定计算次数，算法并不会运行一遍后就返回结果，而是运行多次后返回最好的一次结果，n_init即指明运行的次数
    max_iter 指定单次运行中最大的迭代次数，超过当前迭代次数即停止运行
    """
    def __init__(
                self,
                n_clusters=5,
                n_init=50,
                max_iter=10000
                ):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.mycluster=[]
        
    def fit(self, x):
        """
        用fit方法对数据进行聚类
        :param x: 输入数据
        :best_centers: 簇中心点坐标 数据类型: ndarray
        :best_labels: 聚类标签 数据类型: ndarray
        :return: self
        """
        ###################################################################################
        #### 请勿修改该函数的输入输出 ####
        ###################################################################################
        # #
        kk=0
#         minsum=10000.0
        maxscore=-11
        while kk<self.n_init:
            
            kk+=1
            m = np.shape(x)[0]
            # 初始化一个矩阵来存储每个点的簇分配结果
            # clusterAssment包含两个列:一列记录簇索引值,第二列存储误差(误差是指当前点到簇质心的距离,后面会使用该误差来评价聚类的效果)
            clusterAssment = np.mat(np.zeros((m, 2)))
            label = np.mat(np.zeros((m, 1)))
            clusterChange = True
            it=0
            # 创建质心,随机K个质心

            m, n = x.shape  # 把数据集的行数和列数赋值给m,n
            # 初始化质心,创建(k,n)个以零填充的矩阵
            centroids = np.zeros((self.n_clusters, n))
            # 循环遍历特征值
            xdata=x.values
            for i in range(self.n_clusters):
                index = int(np.random.uniform(0, m))
                centroids[i,:] = xdata[index,:]
                # 计算每一列的质心,并将值赋给centroids
    #             centroids[i, :] = x[index, :]


            # 初始化标志变量,用于判断迭代是否继续,如果True,则继续迭代
            while clusterChange and it<self.max_iter:
                clusterChange = False
                it+=1
                # 遍历所有样本（行数）
                for i in range(m):
                    minDist = 100000.0
                    minIndex = -1
                    # 遍历所有数据找到距离每个点最近的质心,
                    # 可以通过对每个点遍历所有质心并计算点到每个质心的距离来完成
                    for j in range(self.n_clusters):
                        # 计算数据点到质心的距离
                        # 计算距离是使用distMeas参数给出的距离公式,默认距离函数是distEclud
                        distance=np.sqrt(np.sum((centroids[j,:] - xdata[i,:]) ** 2))
    #                     distance = self.distEclud(centroids[j,:], x[i,:])
                        # 如果距离比minDist(最小距离)还小,更新minDist(最小距离)和最小质心的index(索引)
                        if distance < minDist:
                            minDist = distance
                            minIndex = j
                    # 如果任一点的簇分配结果发生改变,则更新clusterChanged标志
                    if clusterAssment[i, 0] != minIndex:
                        clusterChange = True
                        # 更新簇分配结果为最小质心的index(索引),minDist(最小距离)
                        clusterAssment[i, :] = minIndex, minDist
                        label[i,:]=minIndex
                # 遍历所有质心并更新它们的取值
                for j in range(self.n_clusters):
                    # 通过数据过滤来获得给定簇的所有点
                    pointsInCluster = xdata[np.nonzero(clusterAssment[:, 0].A == j)[0]]
                    # 计算所有点的均值,axis=0表示沿矩阵的列方向进行均值计算
                    centroids[j, :] = np.mean(pointsInCluster, axis=0)
            # 返回所有的类质心与点分配结果

            # #
            ###################################################################################
            ############# 在生成 main 文件时, 请勾选该模块 ############# 
            ###################################################################################


    #         best_centers = 
    #         best_labels = 
    #         self.cluster_centers_ = best_centers
    #         self.labels_ = best_labels
#             sum=0.0
            score=0
            score = silhouette_score(data,label.A)
#             print(score)
            if score>maxscore:
                maxscore=score
                print("maxscore",maxscore)
                self.cluster_centers_ = centroids
                
    #         self.mycluster=pd.Series(clusterAssment.A)
#                 self.labels_ = label.A
                self.labels_=np.array(list(chain.from_iterable(label.A)))
                self.mycluster=clusterAssment
#             for i in range(m):
#                 sum+=clusterAssment[i,1]
#             if sum<minsum:
#                 minsum=sum
#                 print("minsun %d",minsum)
#                 self.cluster_centers_ = centroids
                
#     #         self.mycluster=pd.Series(clusterAssment.A)
# #                 self.labels_ = label.A
#                 self.labels_=np.array(list(chain.from_iterable(label.A)))
#                 self.mycluster=clusterAssment
        
        return self

from itertools import chain

kmeans = KMeans(n_clusters=5, n_init=50, max_iter=10000)
kmeans.fit(data)
print(type(kmeans.labels_))
print(type(kmeans.mycluster))
print(type(kmeans.cluster_centers_))
print(kmeans.mycluster[0,1])
print(kmeans.mycluster)
print(kmeans.cluster_centers_)

labels = pd.Series(kmeans.labels_)
fig = plt.figure(1,figsize=(6,6))
ax = Axes3D(fig)
ax.scatter(data.iloc[:,0],data.iloc[:,1],data.iloc[:,2],c=labels.astype('float'),edgecolors='k');
# ax.scatter(data.iloc[:,0],data.iloc[:,1],data.iloc[:,2],edgecolors='k');

from sklearn.metrics import calinski_harabasz_score,silhouette_score

score1 = calinski_harabasz_score(data,kmeans.labels_)
score2 = silhouette_score(data,kmeans.labels_)

print('calinski_harabasz_score:', score1)
print('silhouette_score:', score2)

# 寻找最佳聚类数目
from sklearn.metrics import calinski_harabasz_score,silhouette_score

'''
n_clusters 指定了需要聚类的个数，这个超参数需要自己调整，会影响聚类的效果
init 指明初始聚类中心点的初始化方式，kmeans++是一种初始化方式，还可以选择为random
n_init 指定计算次数，算法并不会运行一遍后就返回结果，而是运行多次后返回最好的一次结果，n_init即指明运行的次数
max_iter 指定单次运行中最大的迭代次数，超过当前迭代次数即停止运行
'''
score1_list = []
score2_list = []
for i in range(2,10):
    kmeans = KMeans(n_clusters=i,n_init=8,max_iter=100)
    kmeans.fit(data)
    score1 = round(calinski_harabasz_score(data,kmeans.labels_), 2)
    score2 = round(silhouette_score(data,kmeans.labels_), 2)
    score1_list.append(score1)
    score2_list.append(score2)
    print('聚类数目:%s  calinski_harabasz_score:%-10s  silhouette_score:%-10s'%(i,score1,score2))

def get_distance(data, kmeans, n_features):
    """
    计算距离函数
    :param data: 训练 kmeans 模型的数据
    :param kmeans: 训练好的 kmeans 模型
    :param n_features: 计算距离需要的特征的数量
    :return: 每个点距离自己簇中心的距离
    """
#     print(type(kmeans.cluster_centers_))
#     print(type(kmeans.labels_))
#     print(kmeans.cluster_centers_[0])
#     print(kmeans.labels_[0])
#     distance = []
#     for i in range(0,len(data)):
#         point = np.array(data.iloc[i,:n_features])
#         center = kmeans.cluster_centers_[kmeans.labels_[i],:n_features]
#         distance.append(np.linalg.norm(point - center))
#     distance = pd.Series(distance)
#     print(distance)
    distance = []
    for i in range(0,len(data)):
        point = np.array(data.iloc[i,:n_features])
        center = kmeans.cluster_centers_[int(kmeans.labels_[i]),:n_features]
        distance.append(np.linalg.norm(point - center))
    distance = pd.Series(distance)
#     print(distance)
    return distance

from copy import deepcopy
ratio = 0.01
num_anomaly = int(len(data) * ratio)
new_data = deepcopy(data)
new_data['distance'] = get_distance(new_data,kmeans,n_features=len(new_data.columns))
threshould = new_data['distance'].sort_values(ascending=False).reset_index(drop=True)[num_anomaly]
print('阈值距离：'+str(threshould))

# 根据阈值距离大小判断每个点是否是异常值
new_data['is_anomaly'] = new_data['distance'].apply(lambda x: x > threshould)
normal = new_data[new_data['is_anomaly'] == 0]
anormal = new_data[new_data['is_anomaly'] == 1]
new_data.head()

fig = plt.figure(1, figsize=(6, 6))
ax = Axes3D(fig)
ax.scatter(anormal.iloc[:, 0], anormal.iloc[:, 1], anormal.iloc[:, 2], c='red', edgecolors='k')
ax.scatter(normal.iloc[:, 0], normal.iloc[:, 1], normal.iloc[:, 2], c='blue', edgecolors='k');

#保存模型，用于后续的测试打分
from sklearn.externals import joblib
joblib.dump(kmeans, './results/model.pkl')
joblib.dump(scaler, './results/scaler.pkl')
joblib.dump(pca, './results/pca.pkl')

a = df.loc[new_data['is_anomaly'] == 1, ['timestamp', 'cpc']] 
plt.figure(figsize=(20,6))
plt.plot(df['timestamp'], df['cpc'], color='blue')
# 聚类后 cpc 的异常点
plt.scatter(a['timestamp'],a['cpc'], color='red')
plt.show()

a = df.loc[new_data['is_anomaly'] == 1, ['timestamp', 'cpm']] 
plt.figure(figsize=(20,6))
plt.plot(df['timestamp'], df['cpm'], color='blue')
# 聚类后 cpm 的异常点
plt.scatter(a['timestamp'],a['cpm'], color='red')
plt.show()

import os
import sklearn
import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA
from copy import deepcopy
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """
    数据处理及特征工程等
    :param df: 读取原始 csv 数据，有 timestamp、cpc、cpm 共 3 列特征
    :return: 处理后的数据, 返回 pca 降维后的特征
    """
    # 请使用joblib函数加载自己训练的 scaler、pca 模型，方便在测试时系统对数据进行相同的变换
    # ====================数据预处理、构造特征等========================
    # 例如
    #     df['hours'] = df['timestamp'].dt.hour
    #     df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 将 df 数据按时间序列排序，方便数据展示
    df = df.sort_values(by='timestamp').reset_index(drop=True)
    df['cpc X cpm'] = df['cpm'] * df['cpc']
    df['cpc / cpm'] = df['cpc'] / df['cpm']
    df['hours'] = df['timestamp'].dt.hour
    df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)
    columns = ['cpc', 'cpm', 'cpc X cpm', 'cpc / cpm', 'hours', 'daylight']
    data = df[columns]
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    # 通过 n_components 指定需要降低到的维度
    n_components = 3
    pca = PCA(n_components=n_components)
    data = pca.fit_transform(data)
    data = pd.DataFrame(data, columns=['Dimesion' + str(i + 1) for i in range(n_components)])
    # ========================  模型加载  ===========================
  

    return data
def get_distance(data, kmeans, n_features):
    """
    计算样本点与聚类中心的距离
    :param data: preprocess_data 函数返回值，即 pca 降维后的数据
    :param kmeans: 通过 joblib 加载的模型对象，或者训练好的 kmeans 模型
    :param n_features: 计算距离需要的特征的数量
    :return:每个点距离自己簇中心的距离，Series 类型
    """
    # ====================计算样本点与聚类中心的距离========================
    distance = []
    for i in range(0,len(data)):
        point = np.array(data.iloc[i,:n_features])
        center = kmeans.cluster_centers_[int(kmeans.labels_[i]),:n_features]
        distance.append(np.linalg.norm(point - center))
    distance = pd.Series(distance)
    return distance
    

from copy import deepcopy
def get_anomaly(data, kmeans, ratio):
    """
    检验出样本中的异常点，并标记为 True 和 False，True 表示是异常点

    :param data: preprocess_data 函数返回值，即 pca 降维后的数据，DataFrame 类型
    :param kmean: 通过 joblib 加载的模型对象，或者训练好的 kmeans 模型
    :param ratio: 异常数据占全部数据的百分比,在 0 - 1 之间，float 类型
    :return: data 添加 is_anomaly 列，该列数据是根据阈值距离大小判断每个点是否是异常值，元素值为 False 和 True
    """
    # ====================检验出样本中的异常点========================
#     num_anomaly = int(len(data) * ratio)
    
    
    
#     data['is_anomaly'] = None


    ratio = 0.03
    num_anomaly = int(len(data) * ratio)
    new_data = deepcopy(data)
    new_data['distance'] = get_distance(new_data,kmeans,n_features=len(new_data.columns))
    threshould = new_data['distance'].sort_values(ascending=False).reset_index(drop=True)[num_anomaly]
    new_data['is_anomaly'] = new_data['distance'].apply(lambda x: x > threshould)
#     normal = new_data[new_data['is_anomaly'] == 0]
#     anormal = new_data[new_data['is_anomaly'] == 1]
    
    return new_data

def predict(preprocess_data):
    """
    该函数将被用于测试，请不要修改函数的输入输出，并按照自己的模型返回相关的数据。
    在函数内部加载 kmeans 模型并使用 get_anomaly 得到每个样本点异常值的判断
    :param preprocess_data: preprocess_data函数的返回值，一般是 DataFrame 类型
    :return:is_anomaly:get_anomaly函数的返回值，各个属性应该为（Dimesion1,Dimension2,......数量取决于具体的pca），distance,is_anomaly，请确保这些列存在
            preprocess_data:  即直接返回输入的数据
            kmeans: 通过joblib加载的对象
            ratio:  异常点的比例，ratio <= 0.03   返回非异常点得分将受到惩罚！
    """
    # 异常值所占比率
    ratio = 0.03
    # 加载模型 
    kmeans = joblib.load('./results/model.pkl')
    # 获取异常点数据信息
    is_anomaly = get_anomaly(preprocess_data, kmeans, ratio)
    
    return is_anomaly, preprocess_data, kmeans, ratio


































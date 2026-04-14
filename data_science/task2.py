import numpy as np
import matplotlib.pyplot as plt

# 1. 准备给定的二维数据点
# 数据来源于作业文档中的表格数据及第三页的补充数据
data = np.array([
    [-0.5200, 1.8539], [2.5849, 2.2481], [0.9919, 1.9234], [2.9443, 3.7382],
    [-0.4240, 3.6220], [1.7762, 2.6264], [2.0581, 2.0918], [1.5754, 1.1924],
    [1.7971, 1.5387], [0.4869, 0.5940], [7.8736, 7.6255], [8.1850, 7.5291],
    [9.3666, 9.7513], [8.4139, 8.7532], [10.5374, 8.0650], [9.1401, 7.7072],
    [7.1372, 8.0828], [8.5458, 8.7662], [8.3479, 10.2368], [9.1033, 8.3269],
    [3.7794, 4.8633], [3.7210, 4.6794], [3.2663, 4.5548], [3.9355, 5.0016],
    [2.5560, 5.2594], [4.6123, 4.0442], [2.6765, 3.6859], [3.3384, 4.2267]
])


# 2. 编写 K-均值聚类算法
def kmeans(X, k, max_iters=100):
    """
    K-Means 聚类算法实现
    :param X: 数据集，numpy 数组
    :param k: 聚类簇的数目
    :param max_iters: 最大迭代次数
    :return: 聚类中心 (centroids) 和每个样本的标签 (labels)
    """
    # 随机选择 k 个样本作为初始聚类中心
    np.random.seed(42)  # 设置随机种子以保证结果可复现
    initial_indices = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[initial_indices]

    for i in range(max_iters):
        # 计算每个样本到所有中心点的欧氏距离
        # X[:, np.newaxis] 维度变为 (N, 1, 2)，与 centroids (k, 2) 广播计算
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

        # 将样本分配给距离最近的中心，产生簇标签
        labels = np.argmin(distances, axis=1)

        # 重新计算每个簇的平均值，作为新的聚类中心
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])

        # 如果中心点不再变化（或者变化极其微小），则算法收敛，提前跳出循环
        if np.all(centroids == new_centroids):
            print(f"算法在第 {i + 1} 次迭代后收敛。")
            break

        centroids = new_centroids

    return centroids, labels


# 3. 运行算法并进行可视化
if __name__ == "__main__":
    # 根据数据分布设定聚类数 k=2
    k = 2
    centroids, labels = kmeans(data, k)

    # 绘制聚类结果散点图
    plt.figure(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e']  # 设置两类的颜色 (蓝、橙)

    # 绘制每个数据点
    for j in range(k):
        cluster_data = data[labels == j]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1],
                    c=colors[j], s=50, label=f'Cluster {j + 1}')

    # 绘制最终的聚类中心点
    plt.scatter(centroids[:, 0], centroids[:, 1],
                c='red', marker='X', s=200, edgecolors='k', label='Centroids')

    plt.title('K-Means Clustering Analysis')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
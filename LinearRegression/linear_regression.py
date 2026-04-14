import numpy as np
from utils.features import prepare_for_training


class LinearRegression:
    # label是真实值
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        """
        1.对数据进行预处理操作
        2.先得到所有的特征个数
        3.初始化参数矩阵
        """
        # 预处理函数，在utils包下
        (data_processed,
         features_mean,
         features_deviation) = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data=True)

        self.data = data_processed  # 预处理的结果
        self.labels = labels  # 传入的参数
        self.features_mean = features_mean  # 预处理的结果
        self.features_deviation = features_deviation  # 预处理的结果
        self.polynomial_degree = polynomial_degree  # 传入的参数
        self.sinusoid_degree = sinusoid_degree  # 传入的参数
        self.normalize_data = normalize_data  # 传入的参数(也可能是默认的参数)

        num_features = self.data.shape[1]  # 特征指标(也就是列数)，shape是给出一个元组，
        self.theta = np.zeros((num_features, 1))  # 有多少个特征就有多少个θ

    def train(self, alpha, num_iterations=500):  # α是学习率，也就是步长，一般设置比较小合适
        """
                    训练模块，执行梯度下降
        """
        cost_history = self.gradient_descent(alpha, num_iterations)
        return self.theta, cost_history

    def gradient_descent(self, alpha, num_iterations):
        """
                    实际迭代模块，会迭代num_iterations次
                    梯度下降
        """
        cost_history = []
        for _ in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data, self.labels))  # 添加损失函数
        return cost_history

    def gradient_step(self, alpha):
        """
                    梯度下降参数更新计算方法，注意是矩阵运算
        """
        num_examples = self.data.shape[0]  # 样本数就是行，列是特征数
        prediction = LinearRegression.hypothesis(self.data, self.theta)
        delta = prediction - self.labels  # 预测值-真实值
        theta = self.theta
        # 下面使用小批量梯度下降法更新θ
        # 下面np.dot(delta.T, self.data)就是通过矩阵运算，实现求和的效果，最后一个转置是为了展示正常
        theta = theta - alpha * (1 / num_examples) * (np.dot(delta.T, self.data)).T
        self.theta = theta

    def cost_function(self, data, labels):
        """
                    损失计算方法
        """
        num_examples = data.shape[0]  # 数据的行数就是样本数
        delta = LinearRegression.hypothesis(self.data, self.theta) - labels
        # 下面求损失函数，也就是目标函数
        cost = (1 / 2) * np.dot(delta.T, delta) / num_examples  # dot就是求点积，
        return cost[0][0]  # 返回损失计算结果

    @staticmethod
    def hypothesis(data, theta):  # 预测值y
        predictions = np.dot(data, theta)
        # data是原始数据，theta是前进的方向，也就是跟着方向走
        return predictions

    def get_cost(self, data, labels):
        data_processed = prepare_for_training(data,
                                              self.polynomial_degree,
                                              self.sinusoid_degree,
                                              self.normalize_data
                                              )[0]
        # 上面这个函数返回data_processed, features_mean, features_deviation，我们只需要data即可，所以[0]

        return self.cost_function(data_processed, labels)

    def predict(self, data):
        """
                    用训练的参数模型，与预测得到回归值结果
        """
        data_processed = prepare_for_training(data,
                                              self.polynomial_degree,
                                              self.sinusoid_degree,
                                              self.normalize_data
                                              )[0]

        predictions = LinearRegression.hypothesis(data_processed, self.theta)

        return predictions

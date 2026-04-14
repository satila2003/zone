import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

data = pd.read_csv('../data/world-happiness-report-2017.csv')

# 得到训练和测试数据
# 这个drop解释是去除关于与train_data具有相同索引的所有行，这样test_data就包含了除了训练集之外的所有数据
train_data = data.sample(frac=0.8)  # 从data数据集中随机抽取 80% 的数据作为一个新的数据集
test_data = data.drop(train_data.index)

# 输入特征名和输出特征名
input_param_name = 'Economy..GDP.per.Capita.'
output_param_name = 'Happiness.Score'

x_train = train_data[[input_param_name]].values
y_train = train_data[[output_param_name]].values

x_test = test_data[input_param_name].values
y_test = test_data[output_param_name].values

plt.scatter(x_train, y_train, label='Train data')
plt.scatter(x_test, y_test, label='test data')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happy')
plt.legend()
plt.show()

# 上面是展示数据长什么样子，下面是训练数据


num_iterations = 500
learning_rate = 0.01

linear_regression = LinearRegression(x_train, y_train)  # 创建实例
(theta, cost_history) = linear_regression.train(learning_rate, num_iterations)

print('开始时的损失：', cost_history[0])
print('训练后的损失：', cost_history[-1])

plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iter')
plt.ylabel('cost')
plt.title('GD')
plt.show()

# 下面开始绘制预测线(测试数据)
predictions_num = 100
x_predictions = np.linspace(x_train.min(), x_train.max(), predictions_num).reshape(predictions_num, 1)
y_predictions = linear_regression.predict(x_predictions)

plt.scatter(x_train, y_train, label='Train data')
plt.scatter(x_test, y_test, label='test data')
plt.plot(x_predictions, y_predictions, 'r', label='Prediction')
plt.xlabel(input_param_name)
plt.ylabel(output_param_name)
plt.title('Happy')
plt.legend()
plt.show()

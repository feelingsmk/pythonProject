import numpy as np
import joblib
from sklearn import svm
import matplotlib.pyplot as plt

x = [[1, 2], [1, 3], [1, 4], [1, 5], [2, 2], [2, 1], [3, 1], [4, 1], [5, 1],
     [-1, -2], [-1, -3], [-1, -4], [-1, -5], [-2, -2], [-2, -1], [-3, -1], [-4, -1], [-5, -1]]
x = np.array(x)
y = [1, 1, 1, 1, 1, 1, 1, 1,1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
y = np.array(y)
# 训练模型
model = svm.SVC(C=10, kernel='linear')
model.fit(x, y)

# 预测
a = [[-1,-1]]
a_pre = model.predict(a)
print("a_pre:", a_pre)
# 对应的支持向量
Support_vector = model.support_vectors_
print("Support_vector:", Support_vector)
# 线性分类对应的参数
w = model.coef_
print("w:", w)
b = model.intercept_
print("b:", b)
# 训练集散点图
# plt.scatter(x[:, 0], x[:, 1]
#
# if w[0, 1] != 0:
#     xx = np.arange(0, 20, 0.1)
#     # 最佳分类线
#     yy = -w[0, 0] / w[0, 1] * xx - b / w[0, 1]
#     plt.scatter(xx, yy, s=4)
#     # 支持向量
#     b1 = Support_vector[0, 1] + w[0, 0] / w[0, 1] * Support_vector[0, 0]
#     b2 = Support_vector[1, 1] + w[0, 0] / w[0, 1] * Support_vector[1, 0]
#     yy1 = -w[0, 0] / w[0, 1] * xx + b1
#     plt.scatter(xx, yy1, s=4)
#     yy2 = -w[0, 0] / w[0, 1] * xx + b2
#     plt.scatter(xx, yy2, s=4)
# else:
#     xx = np.ones(100) * (-b) / w[0, 0]
#     yy = np.arange(0, 10, 0.1)
#     plt.scatter(xx, yy)
# plt.show()

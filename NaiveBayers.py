import numpy as np
from sklearn.naive_bayes import GaussianNB

# GaussianNB 继承高斯朴素贝叶斯，特征可能性被假设为高斯
model = GaussianNB()
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
print(X)
model.fit(X, Y)
predicted = model.predict([[-0.8, -0.1]])
print(predicted)

from sklearn import neighbors
import numpy as np

knn = neighbors.KNeighborsClassifier()
data = np.array([[3, 104], [2, 100], [1, 81], [101, 10], [99, 5], [98, 2]])
lables = np.array([1, 1, 1, 2, 2, 2])
knn.fit(data, lables.ravel())
print(knn.predict(np.array([[58, 9]])))

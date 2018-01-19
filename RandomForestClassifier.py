from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=1000,bootstrap=True)
#决策树的个数，越多越好，但是性能就会越差，至少100左右
#bootstrap=True：是否有放回的采样。
model.fit(X,y)
predict = model.predict(x_test)

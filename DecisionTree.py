# coding:utf-8
import numpy as np
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split

data = []
labels = []
with open("D:/scikit-learnDemos/tree") as ifile:
    for line in ifile:
        tokens = line.strip().split(' ')
        data.append([float(tk) for tk in tokens[:-1]])
        labels.append(tokens[-1])
x = np.array(data)
labels = np.array(labels)
y = np.zeros(labels.shape)  # 返回来一个给定形状和类型的用0填充的数组
y[labels == 'fat'] = 1
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
clf = tree.DecisionTreeClassifier(criterion='gini')
# DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
# max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
# min_samples_split=2, min_weight_fraction_leaf=0.0,
# presort=False, random_state=None, splitter='best')

print(clf)
clf.fit(x_train, y_train)

with open("tree.doc", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

print(clf.feature_importances_)

ans = clf.predict(x_test)
print(x_test)
print('ans',ans)
print("y_test",y_test)
print(np.mean(ans == y_test))
# 准确率与召回率
# 准确率：某个类别在测试结果中被正确测试的比率
# 召回率：某个类别在真实结果中被正确预测的比率

precision, recall, thresholds = precision_recall_curve(y_test, clf.predict(x_test))
answer = clf.predict_proba(x)[:, 1]
print(classification_report(y, answer, target_names=['thin', 'fat']))

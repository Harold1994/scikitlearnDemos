# coding:utf-8
import pandas as pd
import statsmodels.api as am
import pylab as pl
import numpy as np

# numpy: Python 的语言扩展，定义了数字的数组和矩阵
#  pandas: 直接处理和操作数据的主要 package
#  statsmodels: 统计和计量经济学的 package，包含了用于参数评估和统计测试的实用工具
#  pylab: 用于生成统计图

df = pd.read_csv('D:/scikit-learnDemos/lr-binary.csv')
print(df.head())
df.columns = ['admit', 'gre', 'gpa', 'prestige']
# 使用 pandas 的函数 describe 来给出数据的摘要–describe
print(df.describe())
print(df.std())
#频率表，表示 prestige 与 admin 的值相应的数量关系
print(pd.crosstab(df['admit'],df['prestige'],rownames=['admit']))
df.hist()
pl.show()

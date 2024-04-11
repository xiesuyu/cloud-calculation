import numpy as np
import pandas as pd
import hvplot.pandas
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets  # 导入数据集
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn import ensemble
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
import warnings

plt.style.use("fivethirtyeight")
# Pandas中只显示3位小数
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

boston = fetch_openml(name='boston', version=1, as_frame=True, parser='auto')
X = boston.data   # 特征值
y = boston.target  # 目标变量

df = pd.DataFrame(
    X,
    columns = boston.feature_names
)
df['CHAS'] = df['CHAS'].astype(float)
df['RAD'] = df['RAD'].astype(float)
df["MEDV"] = y
# print(df.head())
# print(df.columns)
# print(df.dtypes)
#print(df.info())
# print(df.shape)
# print(df.isnull().sum())
# print(df.describe())
corr = df.corr()
# sns.heatmap(
#     corr,
#     annot=True,
#     fmt=".2f",
#     cmap="magma"
# )
# plt.show()

#print(corr["MEDV"].sort_values())

# sns.pairplot(df[["LSTAT","INDUS","PTRATIO","MEDV"]]) # 绝对值靠前3的特征
# plt.show()

X = df.drop("MEDV",axis=1)
y = df[["MEDV"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=123)
# print("X_train shape",X_train.shape)
# print("y_train shape",y_train.shape)
# print("X_test shape",X_test.shape)
# print("y_test shape",y_test.shape)

# 模型实例化
le = LinearRegression()
# 拟合过程
le.fit(X_train, y_train)
# 得到回归系数
coef1 = le.coef_  # 13个回归系数

#print(coef1)

predict1 = le.predict(X_test)

# print(predict1[:5])
# # 得分
# print("Score：", le.score(X_test, y_test))
# print("RSME：", np.sqrt(mean_squared_error(y_test, predict1)))

#回归系数
le_df = pd.DataFrame()
le_df["name"] = X.columns.tolist()
le_df["coef"] = coef1.reshape(-1,1)
# print(le_df)

#真实值和预测值的对比
test_pre = pd.DataFrame({"test": y_test["MEDV"].tolist(),"pre": predict1.flatten()})
# print(test_pre)
# test_pre.plot(figsize=(18,10))
# plt.show()

#真实值大于预测值的比例
# print(len(test_pre.query("test > pre")) / len(test_pre))

#模型评价

# #测试集上评价
# #将真实值和预测值的散点分布图画在坐标轴上
# plt.scatter(y_test, predict1, label="test")
# plt.plot([y_test.min(), y_test.max()],
#          [y_test.min(), y_test.max()],
#          'k--',
#          lw=3,
#          label="predict"
#         )
# plt.show()

# #整体数据集评价
# #我们对整个数据集X上进行建模：
predict_all = le.predict(X)
# print("Score：", le.score(X, y))  # 统一换成整体数据集
# print("RSME：", np.sqrt(mean_squared_error(y, predict_all)))

#比较整体数据集上的真实值和预测值：
all_pre = pd.DataFrame({"test": y["MEDV"].tolist(),
                         "pre": predict_all.flatten()
                        })
# print(all_pre)

# all_pre.plot(figsize=(18,10))
# plt.show()

# plt.scatter(y, predict_all, label="y_all")
# plt.plot([y.min(), y.max()],
#          [y.min(), y.max()],
#          'k--',
#          lw=3,
#          label="all_predict"
#         )
# plt.show()

#模型改进

#数据标准化
# 实例化
ss = StandardScaler()
# 特征数据
X = ss.fit_transform(X)
# 目标变量
y = ss.fit_transform(y)
# 先切分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

#决策树回归
tr = DecisionTreeRegressor(max_depth=2)
tr.fit(X_train, y_train)
# 预测值
tr_pre = tr.predict(X_test)
# 模型评分
# print('Score:{:.4f}'.format(tr.score(X_test, y_test)))
# # RMSE(标准误差)
# print('RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test,tr_pre))))

#GradientBoosting（梯度提升）
gb = ensemble.GradientBoostingRegressor()
gb.fit(X_train, y_train)
gb_pre=gb.predict(X_test)
# 模型评分
# print('Score:{:.4f}'.format(gb.score(X_test, y_test)))
# # RMSE(标准误差)
# print('RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test,gb_pre))))

#Lasso回归
lo = Lasso()
lo.fit(X_train, y_train)
lo_pre=lo.predict(X_test)
# 模型评分
# print('Score:{:.4f}'.format(lo.score(X_test, y_test)))
# # RMSE(标准误差)
# print('RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test,lo_pre))))

#SVR-支持向量回归
linear_svr = SVR(kernel="linear")
linear_svr.fit(X_train, y_train)
linear_svr_pre = linear_svr.predict(X_test)
# 模型评分
print('Score:{:.4f}'.format(linear_svr.score(X_test, y_test)))
# RMSE(标准误差)
print('RMSE:{:.4f}'.format(np.sqrt(mean_squared_error(y_test,linear_svr_pre))))


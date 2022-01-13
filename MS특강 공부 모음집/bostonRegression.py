#1 데이터 불러오기
from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


#2데이터 불러오기
boston_data = load_boston()
#3 데이터 읽기
print(boston_data)
print("*"*100)

boston_pd = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
boston_pd['MEDV'] = boston_data.target
print(boston_pd.head())

#5 빈 값 확인
print(boston_pd.isnull().sum())


#6 집 값 분포 시각화
sns.displot(boston_pd['MEDV'])
plt.show()

#7 feature-target 상관관계 확인

correlation_matrix = boston_pd.corr().round(2)
plt.figure(figsize=(10,10))
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()

#8 x/y 분리

x = boston_pd.drop(['MEDV', 'NOX', 'DIS', 'TAX'], axis=1)
y= boston_pd['MEDV']
print(x.shape, y.shape)

#9 학습셋/평가셋 분리

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=9)

print(x_train[:5])

#10 data 정규화

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train[:5])

#11 Regression 학습

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)

#12 모델 평가

y_predict = lr.predict(x_test)
print(f"테스트 데이터 : {x_test[0:5]}")
print(f"테스트 값 결과 :{y_predict[0:5]}")
#print(f"asddas : {y_test.values}")
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_predict)
print(y_test.shape, y_predict.shape)
r2 = r2_score(y_test,y_predict)


print(f"MSE = {mse}")
print(lr.score(x_test, y_test))
print(f"R2 = {r2}")
print(y_test, y_predict)
print(y_test.shape, y_predict.shape, x_test.shape, x_train.shape)
#13 model 시각화
plt.scatter(y_test, y_predict)
plt.plot([0, 50], [0, 50], '--k')
plt.tight_layout()
plt.show()
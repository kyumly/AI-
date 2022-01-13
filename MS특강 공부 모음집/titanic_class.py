#사이키런을 활용한 타이타닉 탑승객 예측 classification

import pandas as pd


#1 데이터 불러오기
data = pd.read_csv('데이터/Titanic_dataset.csv')
print(data.head())

#2 데이터 확인하기

print(data.describe(include='all'))

#3빠진 값 확인
print(data.isnull().sum())

#4 사용하지 않을 featrue 제거
#inplace 다음값이 채운다
data.drop(['cabin', 'boat', 'home.dest', 'name', 'ticket', 'body'], axis=1, inplace=True)
data.info()

print(data.isnull().sum())

#5. Fare
print(data.fare.mean())
data.loc[data.fare.isnull(), 'fare'] = data.fare.mean()
print(data.isnull().sum())

#6. Age
print(data.age.mean())
data.loc[data.age.isnull(), 'age'] = data.age.mean()
print(data.isnull().sum())

#7. embarked
print(data.groupby('embarked').size())
data.loc[data.embarked.isnull(), 'embarked'] = 'S'
print(data.isnull().sum())

#시각화
import seaborn as sns
import matplotlib.pyplot as plt

print(data.survived.value_counts(normalize=True))
sns.countplot(data.survived)
plt.title('Count of survived')
plt.show()

#8-1 성별에 따른 생존자 수 여성일경우 생존할 확률이 남성에 비해 2배 가량 높다
sns.countplot(data.gender, hue=data.survived)
plt.title('Relationship between Gender and survived')
plt.show()

#8-2 선실 등급에 따른 생존 여부 선실등급이 3등급일 때는 생존하지 못하는 사람의 비율이, 1등급 일때는 생존하는 사람의 비율보다 높다
sns.kdeplot(data.pclass, data.survived)
plt.title('Relation between Class and survived')
plt.show()

#9 데이터 변환
print(data.groupby('gender').size())
data.loc[data.gender == 'male', 'gender'] = 0
data.loc[data.gender == 'female', 'gender'] = 1

data.loc[data.embarked == 'S', 'embarked'] = 0
data.loc[data.embarked == 'Q', 'embarked'] = 1
data.loc[data.embarked == 'C', 'embarked'] = 2
print(data.head())

#10 X/Y 분리
X = data.drop('survived', axis=1)
Y = data.survived

print(X[:5])
print(Y[:5])

#11 훈련셋/평가셋 분리

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=109, test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


#12. 모델 학습
from sklearn.linear_model import LogisticRegression

lgr = LogisticRegression(max_iter=1000, C=20)
lgr.fit(X_train, Y_train)
#13 모델 성능 평가
from sklearn.metrics import accuracy_score
y_predict = lgr.predict(X_test)
acc = accuracy_score(Y_test, y_predict)
print(acc)
print(lgr.score(X_test, Y_test))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, y_predict)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.show()
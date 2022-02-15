# AI-
AI공부
혼자 공부하는 머신러닝+딥러닝 실습

# 1장 머신러닝이란?

데이터를 학습시켜 기준을 세우는것 
ex) ML하기전 코드 : if x < 30
    ML 한 후 : 데이터를 가지고 기준을 찾음

***
# 2장 데이터 다루기
## 2.1훈련 세트와 테스트 세트  
### 지도 학습과 비지도 학습
    머신 러닝 알고리즘은 크게 2가지로 지도 학습, 비지도 학습으로 나눌수 있음
    지도 학습은 : 입력(input), 타킷(target) 
    비지도 학습 : 입력(input)
    
    특성, 속성 : 각각에 레이블(클래스)들의 특징을 뽑아서 학습 모델에 정답에 도출하는 데이터, 특징이 많으면, 클래스을 파악하기 쉬움
    ex) 
    라벨, 클래스 : 특징들에 대한 결과물

### 훈련 세트와 테스트 세트
    훈련 세트, 테스트 세트 : 모델을 훈련할 때 훈련 세트를 사용하고, 평가는 테스트 세트로 함
    데이터들을 분리 하는 이유는 샘플링 편향, 과대 적합, 과소 적합에 대한 논쟁이 있음   
### 샘플링 편향
    샘플링 편향이란 훈련, 테스트 데이터들이 골고루 섞여야 한다. 만약에 안 섞이면, 한곳에 데이터들이 몰려, 제대로 된 훈련하지 못한다.
    
## 2.2 데이터 전처리
    
### 데이터 전처리를 하는 이유?
1. ML를 하는데 가장 중요한 것은 데이터이다. 만약 데이터가 null, 부정확한 값이 있다면 등, 제대로 된 기계 학습을 할 수 없다. 그리고 데이터편향을 방지 하기위해, 데이터도 적절하게 석어야 한다.<>
2. 샘플들의 특성을 파악해 클래스들간에 상관관계를 파악해야한다. 필요 없는 특성은 학습하는 전에 빼야 한다.
3. 특성값을 일정한 기준으로 맞춰야 한다. 이런 작업들을 데이터 전처리라고 한다.

### 기준 맞추기
1. 데이터를 표현 방식을 맞춰야 한다. 만약에 x축, y축간에 스케일이 맞지 않는다면, 정확한 데이터 분석이 불가능 하다. 그래서 스케일을 맞추기 위해 데이터 정규화 과정을 해야한다.
2. 데이터 정규화에는 2가지 방식이 있다. 범위 변환, Z 변환이 있다. 범위 변환은 정규화후 최소값을 0, 최대값을 1로, 'Z 변환 '은 정규화 후의 변수 평균값이 0, 표준편차가 1이 되도록 변환한다. (정규화 : 범위변환, 표준화 : Z변환)
3. MinMaxScaler -> 범위 변환, StandardScaler -> Z 변환
4. 사용 방법 : 패키지 import -> mc = import한 값 -> mc.transform(데이터 프레임)
#### 표준편차 vs 표준점수
    분산은 데이터에서 평균을 뺀 값을 모두 제곱한 다음 평균을 내어 구함, 표준 편차는 분산의 제곱근으로 데이터가 분산된 정도를 나타낸다.
    표준 점수는 데이터가 원점에서 몇 표준편차만큼 떨어져 있는지를 나타내는 값
        
***
# 3장 회귀 알고리즘과 모델 규제
## 3.1 K-최근접 이웃 회귀
### k-최근접 이웃 회귀란?
지도 학습 알고리즘은 크게 분류와 회귀로 나눈다.<br>
회귀 : 클래스 중 하나로 분류하는 것이 아니라 임의의 **어떤 숫자를 분류**하는 문제<br>
분류 : 클래스 중 하나를 **분류**하는 문제<hr>
    ex) 농어의 길이를 가지고 농어에 무게를 파악
    ex) 길이, 무게, 높이를 가지고 농어인지, 방어인지 분류
<hr>
    K-최근접 이웃회귀 : 예측하려는 샘플에 가장 가까운 샘플 K개를 선택해, 이 샘플들의 클래스를 확인하여 다수 클래스를 새로운 샘플의 클래스로 예측한다.<br>
    데이터에 대한 이해 : sklearn에서는 모든 하이퍼매개변수를 행렬(2차원 배열)형태로 받아야 한다. (데이터베이스 형태)<br>
    ex) test_array = test_array.reshape(-1,1) 1개의 특성을 가지는 행렬

### 결정계수(R^2)
 모델.score() 입력하면, 해당 클래스에 대한 점수가 나온다. 분류일 경우 분류 확률로 표시되고,<br> 
 회귀는 결정계수(R^2)라고 부른다.<br>
 결정 계수란 회귀에서 정확한 값을 측정하는것은 힘들다. 즉 예측값에 대한 정확도로 알 수 있다.<br>
 R^2 = 1 - (타킷-예측)^2 합 / (타킷-평균)^2 합 계산된다. <br>
 0 ~ 1 값으로 나타난다.
### 과대 적합 vs 과소 적합
과대 적합 : 훈련 세틍만 잘 맞는 모델이라 테스트 세트와 나중에 실전에 투입하야 새로운 샘플에 대한 예측을 만들 때 잘 동작하지 않는다.<br>
과소 적합 : 훈련 세트보다 테스트 세트의 점수가 높거나 두 점수가 모두 너무 낮은 결우 과소 접합이라 한다.


## 3.2선형 회귀
### K-최근접 이웃의 한계
모델이 예측했던 값과 정확히 일치하지만, K-최근접 이웃 회귀는 가장 가까운 샘플을 찾아 타킷에 평균을 구하기 때문에, 새로운 샘플이<br>
훈련 세트의 범위를 벗어나면 엉뚱한 값을 예측한다 -> 해결 방법으로는 선형회귀를 사용한다.

### 선형 회귀
특성이 하나인 경우 어떤 직선을 학습하는 알고리즘 -> 하나의 특성을 추출하는것이 제일 중요함<br>
y = ax + b 
a : 기울기(가중치), b : y절편, x : 특성, y : 라벨<br>
lr.coef_(기울기), lr.intercept_(절편)
### 다항회귀
선형회귀에 있던 일차함수 형식에서 n차 함수식으로 변경하는 기법이다. 주의할 점으로는 다중 회귀랑 헷갈리면 안된다. 다중 회귀는 여러개의 특성을 가지고 회귀를 분석하는 역할을 하지만,<br>
다항회귀는 하나의 특성을 변형시켜 모델을 훈련시키는 기법이다.<br>
[다중/다항 회귀에 대한 설명](https://dodonam.tistory.com/236) - 데이터과학님 설명!!
***

## 3.3 특성 공학과 규제
### 다중 회귀
다중 회귀는 하나의 특성을 사용하는 것이 아니라, N개의 특성을 가지고 모델을 학습하는 알고리즘이다.
### 특성공학
    기존의 특성을 사용해 새로운 특성을 뽑아내는 작업을 특성공학이라고 한다. 사이키런에서 제공하고 있는 polynomialFeatures를 사용해 특성을 여러개 만든다.

### 규제
1. 규제는 머신러닝 모델이 훈련 세트를 너무 과도하게 학습하지 못하도록 훼방하는 것을 말한다. 즉 과대 적합이 되지 않도록 만드는 것이다.
2. 규제를 하기전에 정규화를 통해 모든 값들에 대한 scaler 조정해야한다. 정규화를 통해 변환을 시도한다. (StandarScaler클래스 사용) 
3. 선형 회귀 모델에 규제를 추가한 모델을 릿지와 라쏘라고 한다.

### 릿지 회귀 vs 라쏘 
1. 릿지 : 계수를 제곱한 값으로 규제를 실행한다. 
2. 라쏘 : 계수의 절대값을 기준으로 실행한다.
3. 공통점 : alpha값 사용하여 계수를 줄이는 역할을 실행한다.<br>
[참고사이트](https://rk1993.tistory.com/entry/Ridge-regression%EC%99%80-Lasso-regression-%EC%89%BD%EA%B2%8C-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0) - 우주먼지의 하루님!!
***
4장 다양한 분류 알고리즘
***
5장 트리 알고리즘
***
6장 비지도 학습
***
### 참고문헌
1. 혼자 공부하는 머신러닝
2. 파이 데이터 분석

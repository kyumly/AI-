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
#### ML를 하는데 가장 중요한 것은 데이터이다. 만약 데이터가 null, 부정확한 값이 있다면 등, 제대로 된 기계 학습을 할 수 없다. 그리고 데이터편향을 방지 하기위해, 데이터도 적절하게 석어야 한다.
#### 샘플들의 특성을 파악해 클래스들간에 상관관계를 파악해야한다. 필요 없는 특성은 학습하는 전에 빼야 한다.
    
###
###
###
        
        
***
3장 회귀 알고리즘과 모델 규제
***
4장 다양한 분류 알고리즘
***
5장 트리 알고리즘
***
6장 비지도 학습

# 2. 지도 학습

지도 학습은 훈련데이터와 레이블이 있고 훈련을 통해서 완성된 모델로 새로 입력으로 들어온 데이터 포인트의 레이블을 예측하는 머신러닝 방법이다.



### 2.1 분류와 회귀

분류는 미래 정해져 있는 클래스 중에서 데이터 포인트에 맞는 레이블을 고르는 문제이다. 분류는 크게 두 가지 범주로 나눌 수 있는데 두 가지 레이블 중 하나를 고르는 **이진분류(Binary classification)** 과 세 가지 이상의 레이블 중 하나를 고르는 **다중분류(Multiclass classification)** 이다. 이진 분류에서 학습하고자 하는 대상을 **양성(Positive) 클래스** 라고 하고, 그 반대의 대상을 **음성(Negative) 클래스** 라고 한다. 컴퓨터 비전에서는 암 환자들의 MRI 등의 이미지를 보고 암인지 아닌지를 판별 할 수 있는데 이 경우에 이미지에 암 종양이 발견될 경우 양성 클래스, 아닐 경우 음성 클래스라고 할 수 있다. 다중 분류에서는 어떤 꽃의 이미지를 보여주고, 이 꽃이 어떤 꽃인지를 예측하는 모델을 만들 수 있다. 

회귀는 데이터 포인트의 특성를 고려하여 알맞는 **부동소수점수(Floating point value)** 를 찾아내는 문제이다. 대표적으로 어떤 사람의 학력, 집안 형편, 성별, 인종, 교육 수준, 주거지 등의 정보로 이 사람이 얼마만큼의 연봉을 받을 것인지를 예측하는 문제가 있고, 뉴스나 여러가지 웹 페이지에서 나타나는 단어의 빈도수를 체크하여 주가를 예측하는 문제도 있을 수 있다. 

분류와 회귀 문제를 구분하는 방법은 레이블이 **Continuous** 한지 **Discrete** 한지를 살펴보면 알 수 있다.



### 2.2 일반화, 과대적합, 과소적합



일반적으로 훈련데이터로 학습된 모델은 테스트 데이터에도 잘 맞을 것이라고 예상한다. 실제로 훈련데이터로 학습한 모델이 테스트 데이터에도 준수한 성적을 보일 때, 이 모델은 **일반화(Generalization)** 이 잘 되었다고 말할 수 있다. 그러나 훈련된 모델이 항상 일반화가 잘 되는 것은 아니다. 예를 들어서 어떤 동물이 곰인지 아닌지를 판단하는 분류기를 만들어 본다고 가정한다. 그런데 학습 데이터로 반달 가슴곰의 이미지만 입력데이터로 주입시킨다. 이 모델을 테스트 할 때, 반달 가슴곰의 이미지를 주입시키면 매우 뛰어난 성능을 보일 수 있겠지만 북극곰의 이미지를 넣었을때, 반달 가슴곰 모양의 석상 이미지를 주입시킬 때, 과연 곰인지 아닌지를 잘 판별 할 수 있을까. 이렇게 훈련 데이터의 특성만 기억하는 모델을 **과대적합(Overfitting)** 되었다고 말한다. 반대로 반달 가슴곰의 이미지를 아무리 넣어도 훈련과정에서 정확도가 나아지지 않는다. 이 때는 훈련 데이터를 점검하거나 모델의 개선이 필요한데, 이를 **과소적합(Underfitting)** 되었다고 이야기 한다. 과대적합인지 과소적합인지 아니면 일반화가 잘 되었는지를 판단하는 대표적인 방법 한 가지는 훈련 과정 간의 정확도나 손실 값을 확인 하는 것이다.



![과대, 과소 적합](./Figure/2_2.JPG)

##### 2.2.1 모델 복잡도와 데이셋 크기의 관계



보통 데이셋의 특성이 다양하고 갯수가 많아질수록 **모델복잡도(Model capacity)** 가 커지기 마련이다. 실제로 과대적합의 경우, 중복없는 다양한 데이터를 더 모으는 것이 문제 해결에 도움이 된다. 과대 적합이 일어 났다면 모델을 조작하기 전에 데이터를 점검해보고 데이터의 양이 너무 적거나, 중복이 되었거나, 특성이 너무 적을 때, 다양한 데이터를 추가하면 성능 개선에 큰 도움이 된다. 과소 적합의 경우, 데이터 자체보다는 모델을 개선하는 방향이 더 도움이 된다.



### 2.3 지도 학습 알고리즘



##### 2.3.1 예제에 사용할 데이터셋



지도 학습 알고리즘에 사용될 예제 데이터 셋은 다음과 같다. 

- 이진 분류 데이터 셋

  ![이진 분류 데이터 셋](./Figure/2_3_1_1.JPG)

- 회귀 데이터 셋

  ![회귀 데이터 셋](./Figure/2_3_1_2.JPG)



(* 특성이 적은 데이터셋(저차원 데이터셋)에서 얻은 직관이 특성이 많은 데이터 셋(고차원 데이터셋)에서 그대로 유지되지 않을 수 있음.)



- scikit-learn의 위스콘신 유방암 데이터 셋

```python 
In: 
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
print(f"cancer.keys(): {cancer.keys()}")
print(f"Data shape: {cancer.data.shape}")
target_number = {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}
print(f"The number of samples per class:\n{target_number}")
print(f"Features' name:\n{cancer.feature_names}")
```

```python 
Out:
cancer.keys(): dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
Data shape: (569, 30)
The number of samples per class:
{'malignant': 212, 'benign': 357}
Features' name:
['mean radius' 'mean texture' 'mean perimeter' 'mean area'
 'mean smoothness' 'mean compactness' 'mean concavity'
 'mean concave points' 'mean symmetry' 'mean fractal dimension'
 'radius error' 'texture error' 'perimeter error' 'area error'
 'smoothness error' 'compactness error' 'concavity error'
 'concave points error' 'symmetry error' 'fractal dimension error'
 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
 'worst smoothness' 'worst compactness' 'worst concavity'
 'worst concave points' 'worst symmetry' 'worst fractal dimension']    
```



- 보스턴 주택가격 데이터셋과 특성 공학을 거친 보스턴 주택 가격 데이터 셋

```python 
In:
from sklearn.datasets import load_boston

boston = load_boston()
print(f"Data shape: {boston.data.shape}")
# Has been gone through feature engineering
X, y = mglearn.datasets.load_extended_boston()
print(f"X.shape: {X.shape}")    
```

```python 
Out:
Data shape: (506, 13)
X.shape: (506, 104)    
```



##### 2.3.2 K-최근접 이웃



***K*-NN(*K*-Nearest Neighbors)** 알고리즘은 어떤 포인트 하나에 대하여 k개의 이웃을 찾아서 그 중에 가장 많은 레이블을 그 데이터 포인트의 레이블로 지정하는 알고리즘이다.



##### k-최근접 이웃 분류



- 이웃의 숫자가 1개일 때

  ![이웃의 숫자 1개](./Figure/2_3_2_1.JPG)

- 이웃의 숫자가 3개일 때

  ![이웃의 숫자가 3개](./Figure/2_3_2_2.JPG)



실제로 scikit-learn에서 k-최근접 이웃 알고리즘을 적용하는 방법은 다음과 같다.



```python 
from sklearn.model_selection import train_test_split
from introduction_to_ml_with_python import mglearn

X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
```

먼저 데이터를 훈련 세트와 테스트 세트로 나눈다.



```python 
In:
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=3)    
```

scikit-learn에서 KNeighborsClassfier를 Import하고 이웃의 숫자를 지정한다.



```python 
In:
clf.fit(X_train, y_train)    
```

```python 
Out:
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=3, p=2,
                     weights='uniform')    
```

알고리즘 모델 객체의 fit메소드를 호출하면 입력으로 받은 데이터로 모델을 훈련시킨다. 여기서는 우리가 지정한 이웃의 숫자를 제외하고 기본적인 몇가지 매개변수들을 미리 셋팅이 되어 있음을 확인할 수 있다.



```python 
In:
print(f"Test set prediction: {clf.predict(X_test)}")    
```

```python 
Out:
Test set prediction: [1 0 1 0 1 0 0]    
```

테스트 데이터로 예측을 진행하고자 할 때는 알고리즘 모델 객체의 predict 메소드를 호출한다. 결과의 각 원소들은 예측 레이블을 뜻한다.



```python 
In:
print(f"Test set prediction score: {clf.score(X_test, y_test):.2f}")
```

```python 
Out:
Test set prediction score: 0.86
```

모델이 얼마나 잘 일반화 되었는지 확인하기 위해서 score 메소드를 호출할 수 있다.






# 데이터 표현과 특성 공학

데이터을 데이터들의 성격으로 분리해보면 **연속형 특성(Continuous feature)** , **범주형 특성(Categorical feature)** , **이산형 특성(Discrete feature)** 로 나눌 수 있다. 

이렇게 데이터가 어떤 형태의 특성으로 구성되어 있는가로 구성되어 있는가를 살펴봐야 하지만 그 보다 중요한 것은 데이터를 어떻게 표현하는가이다. 예컨대 데이터의 스케일을 조정하지 않으면 데이터 특성의 범위에 따라 결과 값에 차이가 생길 것이다. 또 특성의 상호작용(특성 간의 곱 등)이나 일반적인 다항식을 투가 특성으로 도움이 될 수 있음을 Chapter2에서 살펴봤다.

특정 애플리케이션에 가장 적합한 데이터 표현을 찾는 것을 **특성 공학(Feature engineering)** 이라고 한다. 올바른 데이터 표현은 지도 학습 모델에서 적절한 매개변수를 선택하는 것보다 성능에 더 큰 영향을 끼친다.



### 4.1 범주형 변수

예제에 사용할 데이터는 1994년 인구 조사 데이터베이스에서 추출한 미국 성인의 소득 데이터셋이다. 

[UCI 미국 성인의 소득 데이터]: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/

이 데이터 셋을 사용해 어떤 근로자의 수입(income)이 50,000 달러를 초과하는지, 이하일지를 예측하려고 한다.  아래의 표에서 age, hours-per-week은 연속형 특성이지만 workclass, education, sex, occupation은 범주형 특성이다. 이런 특성들은 고정된 목록 중 하나를 값으로 가지며, 정략적이 아니고 정성적인 속성이다.

![](./Figure/4_1_1.JPG)

범주형 특성을 로지스틱 회귀 분류기에서 학습시키기 위해서(입력 특성 x[i]에 삽입하기 위해서)는 텍스트 데이터가 아닌 숫자여야 한다. 



##### 4.1.1 원-핫-인코딩(가변수)

범주형 변수를 표현하는 데 가장 널리 쓰이는 방법은 **원-핫-인코딩(One-hot-encoding)** 이다(**원-아웃-오브-엔 인코딩(One-out-of-N encoding)** or **가변수(Dummy variable)** ). 가변수는 범주형 변수를 0 또는 1 값을 가진 하나 이상의 새로운 특성으로 바꾼다.

예를 들어서 workclass 특성에는 "Government Employee", "Private Employee", "Self Employed", "Self Employed Incorporated"란 값이 있는데 어떤 사람의 workclass 값에 해당하는 특성은 1이 되고 나머지 세 특성은 0이 된다.



우선 pandas를 이용해 CSV 파일에서 데이터를 읽는다. 

```python 
import pandas as pd
import os

data = pd.read_csv(
    os.path.join(mglearn.datasets.DATA_PATH, "adult.data"),
    header=None, index_col=False,
    names=['age', 'workclass', 'fnlwgt', 'education',  'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'gender',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
           'income'])
data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week', 'occupation', 'income']]
display(data.head())
```

![](./Figure/4_1_1_1.JPG)



##### 범주형 데이터 문자열 확인하기

데이터셋을 일고 나면, 열에 어떤 의미가 있는 범주형 데이터가 있는지 확인해 보는 것이 좋다.

열의 내용을 확인하는 좋은 방법은 pandas에서 Series 객체의 value_counts 메소드를 사용하여 유일한 값이 각각 며 번 나타나는지 출력해 보는 것이다.

```python
In:
print(data.gender.value_counts())
```

```python 
Out:
 Male      21790
 Female    10771
Name: gender, dtype: int64
```



pandas에서는 get_dummies 함수를 사용해 데이터를 매우 쉽게 인코딩할 수 있다. get_dummies 함수는 객체 타입(문자열 같은)이나 범주형을 가진 열을 자동으로 변환해 준다.

```python 
import pprint

print("원본 특성:\n", list(data.columns), "\n")
data_dummies = pd.get_dummies(data)
print("get_dummies 후의 특성:", )
pprint.pprint(list(data_dummies.columns), width=100, compact=True)
```

```python 
원본 특성:
 ['age', 'workclass', 'education', 'gender', 'hours-per-week', 'occupation', 'income'] 

get_dummies 후의 특성:
['age', 'hours-per-week', 'workclass_ ?', 'workclass_ Federal-gov', 'workclass_ Local-gov',
 'workclass_ Never-worked', 'workclass_ Private', 'workclass_ Self-emp-inc',
 'workclass_ Self-emp-not-inc', 'workclass_ State-gov', 'workclass_ Without-pay', 'education_ 10th',
 'education_ 11th', 'education_ 12th', 'education_ 1st-4th', 'education_ 5th-6th',
 'education_ 7th-8th', 'education_ 9th', 'education_ Assoc-acdm', 'education_ Assoc-voc',
 'education_ Bachelors', 'education_ Doctorate', 'education_ HS-grad', 'education_ Masters',
 'education_ Preschool', 'education_ Prof-school', 'education_ Some-college', 'gender_ Female',
 'gender_ Male', 'occupation_ ?', 'occupation_ Adm-clerical', 'occupation_ Armed-Forces',
 'occupation_ Craft-repair', 'occupation_ Exec-managerial', 'occupation_ Farming-fishing',
 'occupation_ Handlers-cleaners', 'occupation_ Machine-op-inspct', 'occupation_ Other-service',
 'occupation_ Priv-house-serv', 'occupation_ Prof-specialty', 'occupation_ Protective-serv',
 'occupation_ Sales', 'occupation_ Tech-support', 'occupation_ Transport-moving', 'income_ <=50K',
 'income_ >50K']
```



연속형 특성인 age나 hours-per-week는 그대로지만 범주형 특성은 값마다 새로운 특성으로 확장되었다.

```python 
data_dummies.head()[data_dummies.head().columns[:6]]
```

![](./Figure/4_1_1_2.JPG)



DataFrame 객체의 values 속성을 이용해 값을 NumPy 배열로 바꿀수 있고 이 배열로 머신러닝 모델을 학습시킨다. 모델을 학습하기 전에는 이 데이터로부터 타깃값을 분리해야 한다. 

(pandas에서 열 인덱싱은 범위 끝을 포함한다. 'a':'b' 라 하면 'b'를 포함한다. 이와 달리 NumPy 배열의 슬라이싱은 마지막 범위를 포함하지 않는다. np.arange(11)[0:10]은 인덱스 10인 항목을 포함하지 않는다.)

여기서는 특성을 포함한 열, 즉 age부터 occupation_ Transport-moving까지 모든 열을 추출한다. 이때 타깃을 뺀 모든 특성이 포함된다.

```python 
In:
features = data_dummies.loc[:, 'age':'occupation_ Transport-moving']
X = features.values
y = data_dummies['income_ >50K'].values
print(f"X.shape: {X.shape} y.shape: {y.shape}")
```

```python 
Out:
X.shape: (32561, 44) y.shape: (32561,)
```

```python 
In:
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print(f"테스트 점수: {logreg.score(X_test, y_test)}")
```

```python 
Out:
테스트 점수: 0.8067804937968308
```

주의할 점은 훈련 데이터와 테스트 데이터를 모두 포함하는 DataFrame을 사용해 get_dummies 함수를 호출하든지, 각각 get_dummies를 호출한 후에 훈련 세트와 테스트 세트의 열 이름을 비교해서 같은 속성인지 확인해야 한다. 각 열의 클래스 값들이 훈련 데이터와 테스트 데이터에 고루 분포하지 않을 가능성이 있기 때문이다.



##### 4.1.2 숫자로 표현된 범주형 특성

범주형 특성은 종종 숫자로 인코딩된다. 특성의 값이 숫자라고 해서 연속형 특성으로 항상 다뤄지는 것은 아니다. 인코딩된 값 사이에 어떤 순서도 없으면, 이 특성은 이산적이라고 생각해야 한다. 예컨대 별 다섯 개 만점으로 매긴 평점 데이터는 보통 범주형으로 다루지만, 평균을 구하는 등 연속형으로 다루기 한다. pandas의 get_dummies 함수는 숫자 특성은 모두 연속형이라고 여겨 가변수를 만들지 않는다. 이를 해결하기 위한 방법은 2가지이다.

- 어떤 열이 연속형인지 범주형인지를 지정할 수 있는 scikit-learn의 OneHotEncoder
- DataFrame에 있는 숫자로 된 열을 문자열로 바꾼다.

```python 
demo_df = pd.DataFrame({'숫자 특성': [0, 1, 2, 1],
                        '범주형 특성': ['양말', '여우', '양말', '상자']})
display(demo_df)
```

![](./Figure/4_1_2_1.JPG)

```python 
display(pd.get_dummies(demo_df))
```

![](./Figure/4_1_2_2.JPG)

 ```python 
demo_df['숫자 특성'] = demo_df['숫자 특성'].astype(str)
display(pd.get_dummies(demo_df, columns=['숫자 특성', '범주형 특성']))
 ```

![](./Figure/4_1_2_3.JPG)



### 4.2 구간 분할, 이산화 그리고 선형 모델, 트리 모델

데이터를 가장 잘 표현하는 방법은 데이터가 가진 의미뿐 아니라 어떤 모델을 사용하는지에 따라 다르다. 선형 모데로가 트리 기반 모델은 특성의 표현 방식으로 인해 미치는 영향이 매우 다르다.

```python 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

X, y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)

reg = DecisionTreeRegressor(min_samples_split=3).fit(X, y)
plt.plot(line, reg.predict(line), label='결정 트리')

reg = LinearRegression().fit(X, y)
plt.plot(line, reg.predict(line), '--', label='선형 회귀')

plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("회귀 출력")
plt.xlabel("입력 특성")
plt.legend(loc='best')
```

![](./Figure/4_2_1.JPG)

선형 모델은 선형 관계로만 모델링하므로 특성이 하나일 때는 직성으로 나타난다. 결정 트리는 이 데이터로 훨씬 복잡한 모델을 만들 수 있다. 그러나 이는 데이터의 표현 형태에 따라 굉장히 달라지는데 그 중 하나는 한 특성을 여러 특성으로 나누는 **구간 분할(Bining)** or **이산화(Discretization)** 이다. 

예를 들어 -3에서 3까지의 입력 값 범위가 10개의 구간으로 구성된다면 다음과 같이 구간 분할을 할 수 있다.

```python 
In:
bins = np.linspace(-3, 3, 11)
print(f"구간: {bins}")
```

```python 
Out:
구간: [-3.  -2.4 -1.8 -1.2 -0.6  0.   0.6  1.2  1.8  2.4  3. ]
```

그 다음 각 데이터 포인트가 어느 구간에 속하는지 기록한다(np.digitize 함수는 시작점을 포함하고 종료점은 포함하지 않는다. 첫번째 구간인 1은  -3 <= x < -2.4이다.  -3보다 작은 데이터는 0, 3보다 크거나 같은 데이터는 11이 된다).

```python 
In:
which_bin = np.digitize(X, bins=bins)
print("\n데이터 포인트:\n", X[:5])
print("\n데이터 포인트의 소속 구간:\n", which_bin[:5])
```

```python 
Out:
데이터 포인트:
 [[-0.75275929]
 [ 2.70428584]
 [ 1.39196365]
 [ 0.59195091]
 [-2.06388816]]

데이터 포인트의 소속 구간:
 [[ 4]
 [10]
 [ 8]
 [ 6]
 [ 2]]
```



연속형 특성을 각 데이터 포인트가 어느 구간에 속했는지로 인코딩한 범주형 특성으로 변환하였으면 원-핫-인코딩으로 변환 할 수 있다.

```python 
In:
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
encoder.fit(which_bin)
X_binned = encoder.transform(which_bin)
print(X_binned[:5])
```

```python 
Out:
[[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]]
```

```python 
In:
print(f"X_binned.shape: {X_binned.shape}")
```

```python 
Out:
X_binned.shape: (100, 10)
```

```python 
line_binned = encoder.transform(np.digitize(line, bins=bins))

reg = LinearRegression().fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label='구간 선형 회귀')

reg = DecisionTreeRegressor(min_samples_split=3).fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), '--', label='구간 결정 트리')
plt.plot(X[:, 0], y, 'o', c='k')
plt.vlines(bins, -3, 3, linewidth=1, alpha=.2)
plt.legend(loc='best')
plt.ylabel("회귀 출력")
plt.xlabel("입력 특성")
```

![](./Figure/4_2_2.JPG)

구간으로 나눈 특성을 사용하기 전과 비교해보면, 각 구간에서 다른 값을 가지고 있으므로 선형 모델이 훨씬 유연해진 것을 알 수 있다. 반면에 결정 트리는 덜 유연해졌다. 트리 모델은 데이터를 자유롭게 나눠 학습할 수 있으므로 특성의 값을 구간으로 나누는 것이 아무런 득이 되지 않는다. 결정 트리는 데이터셋에서 예측을 위한 가장 좋은 구간을 학습한다고 볼 수 있다. 거기다가 구간 분할은 특성마다 따로 해야 하나, 결정 트리는 한 번에 여러 특성을 살필 수 있다. 선형 모델은 이런 변환으로부터 큰 이득을 얻었다.

용량이 매우 크고 고차원 데이터셋이라 선형 모델을 사용해야 한다면 구간 분할이 모델 성능을 높이는 데 아주 좋은 방법이 될 수 있다. 



### 4.3 상호 작용과 다항식

특성을 풍부하게 나타내는 다른 방법은 원본 데이터에 **상호 작용(Interaction)** 과 **다항식(Polynomial)** 을 추가하는 것이다.

선형 모델에 기울기를 추가하는 방법은 구간으로 분할된 데이터에 원래 특성을 추가하는 것이다. 

 ```python 
In:
X_combined = np.hstack([X, X_binned])
print(X_combined.shape)
 ```

```python 
Out:
(100, 11)
```

```python 
reg = LinearRegression().fit(X_combined, y)

line_combined = np.hstack([line, line_binned])
plt.plot(line, reg.predict(line_combined), label='원본 특성을 더한 선형 회귀')

for bin in bins:
  plt.plot([bin, bin], [-3, 3], ":", c='k', linewidth=1)
plt.legend(loc='best')
plt.ylabel("회귀 출력")
plt.xlabel("입력 특성")
plt.plot(X[:, 0], y, 'o', c='k')
```

![](./Figure/4_3_1.JPG)



다음은 구간 특성과 원본 특성의 곱을 추가한 새로운 특성을 추가했을 때이다.

```python 
In:
X_product = np.hstack([X_binned, X * X_binned])
print(X_product.shape)
```

```python 
Out:
(100, 20)
```

```python 
reg = LinearRegression().fit(X_product, y)

line_product = np.hstack([line_binned, line * line_binned])
plt.plot(line, reg.predict(line_product), label='원본 특성을 곱합 선형 회귀')

for bin in bins:
  plt.plot([bin, bin], [-3, 3], ':', c='k', linewidth=1)

plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("회귀 출력")
plt.xlabel("입력 특성")
plt.legend(loc='best')
```

![](./Figure/4_3_2.JPG)



원본 특성의 다항식을 추가하는 방법도 존재한다.

```python 
In:
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=10, include_bias = False)
poly.fit(X)
X_poly = poly.transform(X)
print(f"X_poly.shape: {X_poly.shape}")
```

```python 
Out:
X_poly.shape: (100, 10)
```

```python 
In:
print(f"X 원소:\n{X[:5]}")
print(f"X_poly 원소:\n{X_poly[:5]}")
```

```python 
Out:
X 원소:
[[-0.75275929]
 [ 2.70428584]
 [ 1.39196365]
 [ 0.59195091]
 [-2.06388816]]
X_poly 원소:
[[-7.52759287e-01  5.66646544e-01 -4.26548448e-01  3.21088306e-01
  -2.41702204e-01  1.81943579e-01 -1.36959719e-01  1.03097700e-01
  -7.76077513e-02  5.84199555e-02]
 [ 2.70428584e+00  7.31316190e+00  1.97768801e+01  5.34823369e+01
   1.44631526e+02  3.91124988e+02  1.05771377e+03  2.86036036e+03
   7.73523202e+03  2.09182784e+04]
 [ 1.39196365e+00  1.93756281e+00  2.69701700e+00  3.75414962e+00
   5.22563982e+00  7.27390068e+00  1.01250053e+01  1.40936394e+01
   1.96178338e+01  2.73073115e+01]
 [ 5.91950905e-01  3.50405874e-01  2.07423074e-01  1.22784277e-01
   7.26822637e-02  4.30243318e-02  2.54682921e-02  1.50759786e-02
   8.92423917e-03  5.28271146e-03]
 [-2.06388816e+00  4.25963433e+00 -8.79140884e+00  1.81444846e+01
  -3.74481869e+01  7.72888694e+01 -1.59515582e+02  3.29222321e+02
  -6.79478050e+02  1.40236670e+03]]
```



각 특성의 차수를 알려주는 get_feature_names 메소드를 사용해 특성의 의미를 알 수 있다.

```python 
In:
print(f"항 이름:\n{poly.get_feature_names()}") 
```

```python 
Out:
항 이름:
['x0', 'x0^2', 'x0^3', 'x0^4', 'x0^5', 'x0^6', 'x0^7', 'x0^8', 'x0^9', 'x0^10']
```



다항식 특성을 선형 모델과 함께 사용하면 **다항 회귀(Polynomail regression)** 모델이 된다.

```python 
reg = LinearRegression().fit(X_poly, y)

line_poly = poly.transform(line)
plt.plot(line, reg.predict(line_poly), label='다항 선형 회귀')
plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("회귀 출력")
plt.xlabel("입력 특성")
plt.legend(loc="best")
```

![](./Figure/4_3_3.JPG)



고차원 다항식은 데이터가 부족한 여역에서는 민감하게 동작할 수 있다.

```python 
from sklearn.svm import SVR

for gamma in [1, 10]:
  svr = SVR(gamma=gamma).fit(X, y)
  plt.plot(line, svr.predict(line), label=f"SVR gamma={gamma}")

plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("회귀 출력")
plt.xlabel("입력 특성")
plt.legend(loc="best")
```

![](./Figure/4_3_4.JPG)



보스턴 주택 가격 데이터 셋에 앞에서 언급한 내용을 반영하면 다음과 같다.

```python 
In:
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state=0)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

poly = PolynomialFeatures(degree=2).fit(X_train_scaled)
X_train_poly = poly.transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
print(f"X_train.shape: {X_train.shape}")
print(f"X_train_poly.shape: {X_train_poly.shape}")
```

```python 
Out:
X_train.shape: (379, 13)
X_train_poly.shape: (379, 105)
```

```python 
In:
import pprint

print("다항 특성 이름:")
pprint.pprint(poly.get_feature_names(), width=75, compact=True)
```

```python 
Out:
다항 특성 이름:
['1', 'x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10',
 'x11', 'x12', 'x0^2', 'x0 x1', 'x0 x2', 'x0 x3', 'x0 x4', 'x0 x5',
 'x0 x6', 'x0 x7', 'x0 x8', 'x0 x9', 'x0 x10', 'x0 x11', 'x0 x12', 'x1^2',
 'x1 x2', 'x1 x3', 'x1 x4', 'x1 x5', 'x1 x6', 'x1 x7', 'x1 x8', 'x1 x9',
 'x1 x10', 'x1 x11', 'x1 x12', 'x2^2', 'x2 x3', 'x2 x4', 'x2 x5', 'x2 x6',
 'x2 x7', 'x2 x8', 'x2 x9', 'x2 x10', 'x2 x11', 'x2 x12', 'x3^2', 'x3 x4',
 'x3 x5', 'x3 x6', 'x3 x7', 'x3 x8', 'x3 x9', 'x3 x10', 'x3 x11', 'x3 x12',
 'x4^2', 'x4 x5', 'x4 x6', 'x4 x7', 'x4 x8', 'x4 x9', 'x4 x10', 'x4 x11',
 'x4 x12', 'x5^2', 'x5 x6', 'x5 x7', 'x5 x8', 'x5 x9', 'x5 x10', 'x5 x11',
 'x5 x12', 'x6^2', 'x6 x7', 'x6 x8', 'x6 x9', 'x6 x10', 'x6 x11', 'x6 x12',
 'x7^2', 'x7 x8', 'x7 x9', 'x7 x10', 'x7 x11', 'x7 x12', 'x8^2', 'x8 x9',
 'x8 x10', 'x8 x11', 'x8 x12', 'x9^2', 'x9 x10', 'x9 x11', 'x9 x12',
 'x10^2', 'x10 x11', 'x10 x12', 'x11^2', 'x11 x12', 'x12^2']
```



상호 작용이 있는 특성 데이터와 없는 데이터에 대해 Ridge 회귀와 랜덤 포레스트에서의 성능을 비교하면 다음과 같다.

```python 
In:
from sklearn.linear_model import Ridge

ridge = Ridge().fit(X_train_scaled, y_train)
print(f"상호 작용 특성이 없을 때 점수: {ridge.score(X_test_scaled, y_test):.3f}")
ridge = Ridge().fit(X_train_poly, y_train)
print(f"상호 작용 특성이 있을 때 점수: {ridge.score(X_test_poly, y_test):.3f}")
```

```python 
Out:
상호 작용 특성이 없을 때 점수: 0.621
상호 작용 특성이 있을 때 점수: 0.753
```

```python 
In:
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=0).fit(X_train_scaled, y_train)
print(f"상호 작용 특성이 없을 때 점수: {rf.score(X_test_scaled, y_test):.3f}")
rf = RandomForestRegressor(n_estimators=100, random_state=0).fit(X_train_poly, y_train)
print(f"상호 작용 특성이 있을 때 점수: {rf.score(X_test_poly, y_test)}")
```

```python 
Out:
상호 작용 특성이 없을 때 점수: 0.795
상호 작용 특성이 있을 때 점수: 0.7736296204464514
```

위와 같이 Ridge에서는 상호작용과 다항식 특성이 성능을 크게 높였으나 랜덤 포레스트에서는 오히려 성능이 조금 줄어 들었다.



### 4.4 일변량 비선형 변환

트리 기반 모델은 특성의 순서에만 영향을 받으나(트리 기반 모델의 max_features 매개변수는 트리의 각 분기에서 사용될 후보 특성의 개수를 제한한다. 랜덤 포레스트는 "auto"가 기본 값으로 특성 개수의 제곱근만큼 사용하고 의사결정 트리와 그래디언트 부스팅 트리의 기본값은 "None"으로 특성을 모두 사용한다) 선형 모델과 신경망은 각 특성의 스케일과 분포에 밀접하게 연관 되어 있다. 특히 선형 회귀에서는 특성과 타깃값 사이에 비선형성이 있으면 모델을 만들기 어렵다. log, exp 함수 등은 데이터의 스케일을 변경해 선형 모델과 신경망의 성능을 올리는 데 도움을 주기도 한다.

대부분의 모델은 각 특성이 (회귀에서는 타깃 포함) 정규 분포와 비슷할 때 최고의 성능을 낸다(확률적 요소를 가진 많은 알고리즘의 이론이 정규분포를 근간으로 한다). 정수 카운트 데이터는 음수가 없으며 특별한 통계 패턴을 따르기 때문에 이런 변환이 도움이 되기도 한다.

```python 
In:
rnd = np.random.RandomState(0)
X_org = rnd.normal(size=(1000, 3))
w = rnd.normal(size=3)
X = rnd.poisson(10 * np.exp(X_org))
y = np.dot(X_org, w)
print(X[:10, 0])
```

```python 
Out:
[ 56  81  25  20  27  18  12  21 109   7]
```

```python 
In:
print(f"특성 출현 횟수:\n{np.bincount(X[:, 0])}")
```

```python 
Out:
특성 출현 횟수:
[28 38 68 48 61 59 45 56 37 40 35 34 36 26 23 26 27 21 23 23 18 21 10  9
 17  9  7 14 12  7  3  8  4  5  5  3  4  2  4  1  1  3  2  5  3  8  2  5
  2  1  2  3  3  2  2  3  3  0  1  2  1  0  0  3  1  0  0  0  1  3  0  1
  0  2  0  1  1  0  0  0  0  1  0  0  2  2  0  1  1  0  0  0  0  1  1  0
  0  0  0  0  0  0  1  0  0  0  0  0  1  1  0  0  1  0  0  0  0  0  0  0
  1  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1]
```

![](./Figure/4_4_1.JPG)

위와 같은 분포는 푸아송(Poisson) 분포로 카운트 데이터의 전형적인 분포이다. 

(푸아송 분포는 단위 시간 안에 일어날 이벤트 횟수를 표현하는 확률 분포이다.)

![](./Figure/4_4_2.JPG)



선형 모델은 이런 데이터를 잘 처리하지 못한다.

```python 
In:
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
score = Ridge().fit(X_train, y_train).score(X_test, y_test)
print(f"테스트 점수: {score:.3f}")
```

```python 
Out:
테스트 점수: 0.622
```



이를 로그 스케일로 변환하면 도움이 될 수 있다.

```python 
X_train_log = np.log(X_train + 1)
X_test_log = np.log(X_test + 1)
plt.rc('font', size=12)
plt.rc('axes', labelsize=15)
plt.figure(figsize=(10, 8))
plt.hist(X_train_log[:, 0], bins=25, color='gray')
plt.ylabel("출현 횟수")
plt.xlabel("값")
plt.xlim(left=0)
```

![](./Figure/4_4_3.JPG)

변환 후 데이터 분포를 보면 데이터의 분포가 덜 치우쳐 있으며 매우 큰 값의 이상치가 보이지 않는다.

```python 
In:
score = Ridge().fit(X_train_log, y_train).score(X_test_log, y_test)
print(f"테스트 점수: {score:.3f}") 
```

```python 
Out:
테스트 점수: 0.875
```



이 예에서는 모든 특성이 같은 속성을 가지고 있지만 실제로 이런 경우는 드물며, 일부 특성만 변환하거나 특성마다 모두 다르게 변환하기도 한다. 이런 변환은 트리 기반 모델에서는 불필요하지만 선형 모델에서는 필수이다.  회귀에서 타깃 변수 y를 변환 하는 것이 좋을 때도 있다(위와 같은 카운트를 예측하는 경우). 

구간 분할, 다항식, 상호 작용은 데이터가 주어진 상황에서 모델의 성능에 큰 영향을 줄 수 있다. 

- 선형 모델, 나이브 베이즈 모델 같은 덜 복잡한 모델 - 구간 분할, 다항식, 상호 작용 등이 성능에 기여하는 바가 큼.
- 트리 기반 모델 - 스스로 중요한 상호작용을 찾아낼 수 있고 대부분의 경우 데이터를 명시적으로 변환할 필요가 없음.
- SVM, 최근접 이웃, 신경망 - 이따금 이득을 볼 수 있으나 선형 모델과 같이 영향이 뚜렷하지 않음.



### 4.5 특성 자동 선택

새로운 특성을 만들다보면 원본 특성의 수 이상으로 증가하기 쉬운데 특성이 추가될수록 모델은 더 복잡해지고 과대적합될 가능성도 높아진다. 보통 가장 유용한 특성만 선택하고 나머지는 무시해서 특성의 수를 최적화 하는 것이 좋다. 어떤 특성이 좋은지 파악하기 위한 전략으로 **변량 통계(Univariate statistics)** , **모델 기반 선택(Model-based selection)** , **반복적 선택(Iterative selection)** 이 있다. 이들 모두 지도 학습학습이다.



##### 4.5.1 일변량 통계

일변량 통계에서는 개개의 특성과 타깃 사이에 중요한 통계적 관계가 있는지를 계산한다.  그런 다음 깊게 관련되어 있다고 판단 되는 특성을 선택한다. 분류에서는 **분산 분석(ANOVA, Analysis of variance)** 라고도 한다(데이터를 클래스별로 나누어 평균을 비교하는 방법. 분산 분석으로 계산한 어떤 특성의 F-통계값이 높으면 그 특성은 클래스별 평균이 서로 다르다는 뜻). 이 방법의 핵심은  **일변량(Univariate)** , 즉 각 특성이 독립적으로 평가된다는 점이다. 따라서 다른 특성과 깊게 연관된 특성은 선택되지 않는다. 일변량 분석은 계산이 빠르고 평가를 위해 모델을 만들 필요가 없다. 

scikit-learn에서는 일변량 분석으로 특성을 선택하려면 분류에서는 f_classif를, 회귀에서는 f_regression을 선택하여 테스트 하고(

[참고]: http://vassarstats.net/textbook/ch14pt1.html

), 계산한 *p*-값(*p*-value)에 기초하여 특성을 제외한다. 이런 방식들은 매우 높은 *p*-값을 가진(타깃값과 연관성이 작을 것 같은) 특성을 제외할 수 있도록 임계값을 조정하는 매개변수를 사용한다.  SelectKBest는 고정된 k개의 특성을 선택하고 SelectPercentile은 지정된 비율만큼 특성을 선택한다.

```python 
In:
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectPercentile

cancer = load_breast_cancer()
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))

X_w_noise = np.hstack([cancer.data, noise])

X_train, X_test, y_train, y_test = train_test_split(
    X_w_noise, cancer.target, random_state=0, test_size=.5)
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)

X_train_selected = select.transform(X_train)

print(f"X_train.shape: {X_train.shape}")
print(f"X_train_selected.shape: {X_train_selected.shape}")
```

```python 
Out:
X_train.shape: (284, 80)
X_train_selected.shape: (284, 40)
```

```python 
#선택된 특성을 참 거짓으로 표시해주어 어떤 특성이 선택되었는지 확인 가능
mask = select.get_support()
print(mask)
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("특성 번호")
```

``````
Out:
[ True  True  True  True  True  True  True  True  True False  True False
  True  True  True  True  True  True False False  True  True  True  True
  True  True  True  True  True  True False False False  True False  True
 False False  True False False False False  True False False  True False
 False  True False  True False False False False False False  True False
  True False False False False  True False  True False False False False
  True  True False  True False False False False]
``````

![](./Figure/4_5_1_1.JPG)

```python 
In:
from sklearn.linear_model import LogisticRegression

X_test_selected = select.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train, y_train)
print(f"전체 특성을 사용한 점수: {lr.score(X_test, y_test):.3f}")
lr.fit(X_train_selected, y_train)
print(f"선택된 일부 특성을 사용한 점수: {lr.score(X_test_selected, y_test):.3f}")
```

```python 
Out:
전체 특성을 사용한 점수: 0.916
선택된 일부 특성을 사용한 점수: 0.919
```



##### 4.5.2 모델 기반 특성 선택

모델 기반 특성 선택은 지도 학습 머신러닝 모델을 사용하여 특성의 중요도를 평가해서 가장 중요한 특성들만 선택한다.  특성 선택에 사용하는 지도 학습 모델이 최종 사용 모델과 같을 필요는 없다. 모델이 특성의 중요도를 평가하기 때문에 당연하게도 모델은 각 특성의 중요도를 측정하여 순서를 매길 수 있어야 한다. 예를 들어 결정 트리와 이를 기반으로 하는 모델은 각 특성의 중요도가 담겨 있는 feature_importances_ 속성을 제공한다. 선형 모델 계수의 절댓값도 특성의 중요도를 재는 데 사용할 수 있다. L1 규제를 사용한 모델이 일부 특성의 계수만 학습함을 보았는데 이를 다른 모델을 위한 전처리 단계에 사용할 수 있다. 일변량 분석과는 다르게 모델 기반 특성 선택은 한 번에 모든 특성을 고려하므로 상호작용 부분을 반영할 수 있는 가능성이 있다. 

```python 
In:
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

select = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42), 
    threshold='median')
select.fit(X_train, y_train)
X_train_l1 = select.transform(X_train)
print(f"X_train.shape: {X_train.shape}")
print(f"X_train_l1.shape: {X_train_l1.shape}")
```

```python 
Out:
X_train.shape: (284, 80)
X_train_l1.shape: (284, 40)
```

 (L1 규제가 없는 모델을 사용할 경우 SelectFromModel threshold 매개변수의 기본 값은 "mean"이다. 또한 "1.2*median"과 같이 비율로 나타낼 수 있다.)

```python 
mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("특성 번호")
```

![](./Figure/4_5_2_1.JPG)

```python 
In:
from sklearn.linear_model import LogisticRegression

X_test_l1 = select.transform(X_test)
score = LogisticRegression().fit(X_train_l1, y_train).score(X_test_l1, y_test)
print(f"테스트 점수: {score:.3f}")
```

```python 
Out:
테스트 점수: 0.919
```



##### 4.5.3 반복적 특성 선택

**반복적 특성 선택(Iterative Feature Selection)** 에서는 특성의 수가 각기 다른 모델이 만들어진다.

두 가지 방법이 있는데 첫 번째는 특성을 하나도 선택하지 않은 상태에서 어떤 종료 조건에 도달할 때까지 하나씩 추가하는 방법이 있고 두 번째는 모든 특성을 가지고 시작해서 어떤 종료 조건이 될 때까지 특성을 하나씩 제거해가는 방법이다. 일련의 모델이 만들어지기 때문에 계산 비용이 훨씬 많이 든다. **재귀적 특성 제거(Recursive feature elimination, RFE)** 가 이런 방법 중 하나이다. 모든 특성으로 시작해서 모델을 만들고 특성 중요도가 가장 낮은 특성을 제거하여 새로운 모델을 만든다. 이런식으로 특성의 개수가 미리 정해진만큼 남을때까지 계속한다. 

(회귀 모델에서 사용하는 반복적인 선택 방법인 전진 선택법(Foward stepwise selection)과 후진 선택법(Backward stepwise selection)은 scikit-learn에서 제공하지 않으나 score 함수의 R^2 값으로 위와 같이 구현이 가능하다.)

```python 
from sklearn.feature_selection import RFE

select = RFE(RandomForestClassifier(n_estimators=100, random_state=42), 
             n_features_to_select=40)

select.fit(X_train, y_train)
mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("특성 번호")
```

![](./Figure/4_5_3_1.JPG)

```python 
In:
X_train_rfe = select.transform(X_train)
X_test_rfe = select.transform(X_test)

score = LogisticRegression().fit(X_train_rfe, y_train).score(X_test_rfe, y_test)
print(f"테스트 점수: {score:.3f}")
```

```python 
Out:
테스트 점수: 0.930
```

```python 
In:
print(f"테스트 점수: {select.score(X_test, y_test):.3f}")
```

```python 
Out:
테스트 점수: 0.951
```



### 4.6 전문가 지식 활용

특성 공학에서 전문가의 지식(Expert Knowledge)을 사용할 경우, 종종 초기 데이터에서 더 유용한 특성을 선책하는데 도움이 될 수 있다. 

다음은 시티바이크의 자전거 대여 데이터를 활용해 특정 날짜와 시간에 자전거를 사람들이 얼마나 대여할 것인지를 예측하는 예제이다. 

이 대여소에 대한 2015년 8월 데이터는 세 시간 간격으로 자전거 대여 횟수가 누적되어 있다. 

```python 
In:
citibike = mglearn.datasets.load_citibike()
print(f"시티바이크 데이터:\n{citibike.head()}")
```

```python 
Out:
시티바이크 데이터:
starttime
2015-08-01 00:00:00     3
2015-08-01 03:00:00     0
2015-08-01 06:00:00     9
2015-08-01 09:00:00    41
2015-08-01 12:00:00    39
Freq: 3H, Name: one, dtype: int64
```

```python 
import pandas as pd

plt.figure(figsize=(10, 3))
xticks = pd.date_range(start=citibike.index.min(), end=citibike.index.max(), freq='D')
week = ["일", "월", "화", "수", "목", "금", "토"]
xticks_name = [week[int(w)]+d for w, d in zip(xticks.strftime("%w"),
                                              xticks.strftime(" %m-%d"))]
plt.xticks(xticks, xticks_name, rotation=90, ha='left')
plt.plot(citibike, linewidth=1)
plt.xlabel("날짜")
plt.ylabel("대여횟수")
```

![](./Figure/4_6_1.JPG)



그래프를 보면 낮과 밤, 주중과 주말의 패턴의 차이를 확인할 수 있다. 여기서 23일치 184개의 데이터(24시간/3시간 = 8개씩)를 훈련 세트로, 남은 8일 치 64개의 데이터를 테스트 세트로 사용한다. 



첫번째로 시도할 방법은 POSIX 시간 표현 방법으로 날짜와 시간을 하나의 숫자로 표현한 특성을 사용한다.

```python 
y = citibike.values
X = citibike.index.astype("int64").values.reshape(-1, 1) // 10**9

n_train = 184

def eval_on_features(features, target, regressor):
    X_train, X_test = features[:n_train], features[n_train:]
    y_train, y_test = target[:n_train], target[n_train:]
    regressor.fit(X_train, y_train)
    print("테스트 세트 R^2: {:.2f}".format(regressor.score(X_test, y_test)))
    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)
    plt.figure(figsize=(10, 3))

    plt.xticks(range(0, len(X), 8), xticks_name, rotation=90, ha="left")

    plt.plot(range(n_train), y_train, label="훈련")
    plt.plot(range(n_train, len(y_test) + n_train), y_test, '-', label="테스트")
    plt.plot(range(n_train), y_pred_train, '--', label="훈련 예측")

    plt.plot(range(n_train, len(y_test) + n_train), y_pred, '--',
             label="테스트 예측")
    plt.legend(loc=(1.01, 0))
    plt.xlabel("날짜")
    plt.ylabel("대여횟수")
```



랜덤 포레스트는 데이터 전처리가 거의 필요하지 않아 맨 처음 시도해 보기 좋다.

```python 
In:
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=100, random_state=0)
eval_on_features(X, y, regressor)

Out:
테스트 세트 R^2: -0.04
```

![](./Figure/4_6_2.JPG)

랜덤 포레스트 모델은 훈련세트의 예측은 매우 정확하나 테스트 데이터에 대해선 한 가지 값으로 예측했고 R^2 이 -0.04로 거의 아무것도 학습되지 않음을 확인 할 수 있다. 랜덤 포레스트는 훈련 세트에 있는 특성의 범위 밖으로 외삽(Extrapolation)하는 능력이 없으므로 테스트 세트와 가장 가까이 있는 훈련 세트 데이터의 타깃값을 예측으로 사용한다. 



전문가 지식을 활용하여 POSIX 시간 특성을 제외하고 시간 특성만 사용해본다.

```python 
In:
X_hour = citibike.index.hour.values.reshape(-1, 1)
eval_on_features(X_hour, y, regressor)

Out:
테스트 세트 R^2: 0.60
```

![](./Figure/4_6_3.JPG)



요일 특성을 추가해본다.

```python 
In:
X_hour_week = np.hstack([citibike.index.dayofweek.values.reshape(-1, 1),
                         citibike.index.hour.values.reshape(-1, 1)])
eval_on_features(X_hour_week, y, regressor)

Out:
테스트 세트 R^2: 0.84
```

![](./Figure/4_6_4.JPG)

위 모델이 학습한 것은 8월 23일까지 요일별, 시간별 평균 대여 횟수이다(랜덤 포레스트 회귀로 만든 예측은 여러 트리가 예측한 값들의 평균이므로). 



다음은 LinearRegression 모델을 적용했을 때이다.

```python 
In:
from sklearn.linear_model import LinearRegression

eval_on_features(X_hour_week, y, LinearRegression())

Out:
테스트 세트 R^2: 0.13
```

![](./Figure/4_6_5.JPG)

선형 모델은 시간을 선형 함수로만 학습할 수 있어서 시간이 흐를수록 대여 수가 늘어나게 학습되었다. 또, 성능이 나쁘고 패턴이 이상한 이유는 요일과 시간이 정수로 인코딩되어 있어서 연속형 변수로 해석되기 때문이다.



OneHotEncoder를 사용하여 정수형을 범주형으로 바꿔보았다.

```python 
In:
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge

enc = OneHotEncoder()
X_hour_week_onehot = enc.fit_transform(X_hour_week).toarray()

eval_on_features(X_hour_week_onehot, y, Ridge())

Out:
테스트 세트 R^2: 0.62
```

![](./Figure/4_6_6.JPG)



상호특성을 사용하여 시간과 요일의 조합별 계수를 학습시켜 보았다.

```python 
In:
from sklearn.preprocessing import PolynomialFeatures

poly_transformer = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_hour_week_onehot_poly = poly_transformer.fit_transform(X_hour_week_onehot)
lr = Ridge()
eval_on_features(X_hour_week_onehot_poly, y, lr)

Out:
테스트 세트 R^2: 0.85
```

![](./Figure/4_6_7.JPG)



랜덤 포레스트와는 달리 선형 회귀 모델에서는 모델이 학습한 계수를 그래프로 나타낼 수 있다.

```python 
hour = ["%02d:00"%i for i in range(0, 24, 3)]
day = ["월", "화", "수", "목", "금", "토", "일"]
features = day + hour

features_poly = poly_transformer.get_feature_names(features)
features_nonzero = np.array(features_poly)[lr.coef_ != 0]
coef_nonzero = lr.coef_[lr.coef_ != 0]

plt.figure(figsize=(15, 2))
plt.plot(coef_nonzero, 'o')
plt.xticks(np.arange(len(coef_nonzero)), features_nonzero, rotation=90)
plt.xlabel("특성 이름")
plt.ylabel("계수 크기")
```

![](./Figure/4_6_8.JPG)



### 4.7 요약 및 정리

이번 장에서는 여러 종류의 데이터 타입(특히 범주형 변수)을 다루는 법을 정리했다. 

- 원-핫-인코딩 범주형 변수처럼 머신러닝 알고리즘에 적합한 방식으로 데이터를 표현하는 것이 중요
- 새로운 특성을 만드는 방법
- 데이터에서 특성을 유도하기 위해 전문가의 지식 활용
- 선형 모델은 구간 분할이나 상호작용 특성을 추가해 성능을 끌어올릴 있음
- 랜덤 포레스트나 SVM 같은 비선형 모델은 특성을 늘리지 않고서도 복잡한 문제 학습 가능

어떤 특성을 사용하느냐 그리고 특성과 모델의 궁합이 중요하다.
# 4. 데이터 표현과 특성 공학

데이터을 데이터들의 성격으로 분리해보면 **연속형 특성(Continuous feature)** , **범주형 특성(Categorical feature)** , **이산형 특성(Discrete feature)** 로 나눌 수 있다. 

이렇게 데이터가 어떤 형태의 특성으로 구성되어 있는가로 구성되어 있는가를 살펴봐야 하지만 그 보다 중요한 것은 데이터를 어떻게 표현하는가이다. 예컨대 데이터의 스케일을 조정하지 않으면 데이터 특성의 범위에 따라 결과 값에 차이가 생길 것이다. 또 특성의 상호작용(특성 간의 곱 등)이나 일반적인 다항식을 투가 특성으로 도움이 될 수 있음을 Chapter2에서 살펴봤다.

특정 애플리케이션에 가장 적합한 데이터 표현을 찾는 것을 **특성 공학(Feature engineering)** 이라고 한다. 올바른 데이터 표현은 지도 학습 모델에서 적절한 매개변수를 선택하는 것보다 성능에 더 큰 영향을 끼친다.



### 4.1 범주형 변수

예제에 사용할 데이터는 1994년 인구 조사 데이터베이스에서 추출한 미국 성인의 소득 데이터셋이다. 

[UCI 미국 성인의 소득 데이터](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/)

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


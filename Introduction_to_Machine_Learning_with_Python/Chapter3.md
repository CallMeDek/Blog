# 3. 비지도 학습과 데이터 전처리

**비지도 학습(Unsupervised learning)** 은 출력 값이나 정보 없이 학습 알고리즘을 가르쳐야 하는 모든 종류의 머신러닝이다. 즉, 모델은 입력 데이터만으로 데이터에서 지식을 추출할 수 있어야 한다.



### 3.1 비지도 학습의 종류

**비지도 변환(Unsupervised transformation)** 은 데이터를 새롭게 표현하여 원래보다 쉽게 해석할 수 있도록 만드는 알고리즘이다. 가장 널리 사용되는 분야는 고차원 데이터 중에 필요한 특징을 포함한 채로 특성의 수를 줄이는 **차원 축소(Dimensionality reduction)** 이다. 비지도 변환으로 소셜 미디어 등의 문서에서 이야기하는 주제들이 무엇인지 파악할 수 있는 단위나 성분을 찾을 수도 있다. 

**군집 알고리즘(Clustering)** 은 데이터를 비슷한 그룹으로 묶는 것이다. 소셜 미디어 사이트에서의 사진을 같은 사람의 얼굴 이미지로 그룹화 하는 등의 일에 사용된다.



### 3.2 비지도 학습의 도전 과제

보통 비지도 학습에서 가장 어려운 일은 알고리즘이 뭔가 유용한 것을 학습했는지에 대한 평가이다.  레이블이 없기 때문에 알고리즘에게 우리가 원하는 것을 알려줄 방법이 없다. 비지도 학습의 결과를 평가하위해서 직접 눈으로 확인하는 것이 좋은 때가 많다. 이런 이유로 비지도 학습 알고리즘은 **탐색적 분석 (EDA, Exploratory data analysis)** 단계에서 많이 사용한다. 그리고 지도 학습의 전처리 단계에서도 사용되는데 실제로 비지도 학습의 결과로 새롭게 표현된 데이터를 사용하면 지도학습의 정확도가 좋아지기도 한다. 



### 3.3 데이터 전처리와 스케일 조정

지도학습에서의 신경망과 SVM 같은 알고리즘은 데이터 스케일에 매우 민감하므로 조정이 필요하다.

![](./Figure/3.3.JPG)

1. StandardScaler는 각 특성의 평균을 0, 분산으로 1로 변경하여 모든 특성이 같은 크기를 가지게 한다. 그러나 각 특성의 최솟값과 최댓값의 크기를 제한하지는 않는다(x - x_bar / σ, 여기서 x_bar는 평균,  σ는 분산)
2. RobustScaler는 특성들이 같은 스케일을 갖는 다는 점에서는 1번과 유사하나 평균과 분산 대신 중간 값(Median)과 사분위 값(Quartile)을 사용한다. 이런 방식 때문에 **이상치(Outlier)** 에 큰 영향을 받지 않는다(x - q2 / q3 - q1, 여기서 qi는 i사분위 값).
3. MinMaxScaler는 모든 특성이 정확하게 0과 1 사이에 위치하도록 데이터를 변경한다(x - x_min / x_max - x_min, 여기서 x_min은 최솟값, x_max는 최댓값).
4. Normalizer는 특성 벡터의 유클리디안 길이가 1이 되도록 데이터 포인트를 조정한다. Normalizer의 norm 매개변수는 l1, l2, max 세가 옵션을 제공하며 유클리디안 거리를 의미하는 l2가 기본 값이다. 앞의 3가지 방식에서는 각 열(특성)의 통계치를 이용한다면 Normalizer는 행(데이터 포인트)마다 각기 정규화된다. 다른말로 하면 2차원에서는 지름이 1인 원, 3차원에서는 지름이 1인 구에 데이터 포인트를 사영(Projection)한다. 이는 특성 벡터의 길이와 상관 없이 데이터의 방향 또는 각도가 중요할 때 많이 사용한다.



##### 3.3.2 데이터 변환 적용하기

sklearn의 유방암 데이터를 MinMaxScaler를 사용하여 전처리한 결과는 다음과 같다.

```python 
In:
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1)
print(X_train.shape, X_test.shape)
```

```python 
Out:
(426, 30) (143, 30)
```

```python 
In:
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)
```

```python 
Out:
MinMaxScaler(copy=True, feature_range=(0, 1))
```

```python 
In:
X_train_scaled = scaler.transform(X_train)
print(f"변환된 후 크기: {X_train_scaled.shape}")
print(f"스케일 조정 전 특성별 최솟값:")
print(np.array(list(map(lambda x:f"{x:.3f}", X_train.min(axis=0))), dtype="float"))
print(f"스케일 조정 전 특성별 최댓값:")
print(np.array(list(map(lambda x:f"{x:.3f}", X_train.max(axis=0))), dtype="float"))
print(f"스케일 조정 후 특성별 최솟값:\n {X_train_scaled.min(axis=0)}")
print(f"스케일 조정 후 특성별 최댓값:\n {X_train_scaled.max(axis=0)}")
```

```python 
In:
변환된 후 크기: (426, 30)
스케일 조정 전 특성별 최솟값:
[6.981e+00 9.710e+00 4.379e+01 1.435e+02 5.300e-02 1.900e-02 0.000e+00
 0.000e+00 1.060e-01 5.000e-02 1.150e-01 3.600e-01 7.570e-01 6.802e+00
 2.000e-03 2.000e-03 0.000e+00 0.000e+00 1.000e-02 1.000e-03 7.930e+00
 1.202e+01 5.041e+01 1.852e+02 7.100e-02 2.700e-02 0.000e+00 0.000e+00
 1.570e-01 5.500e-02]
스케일 조정 전 특성별 최댓값:
[2.811e+01 3.928e+01 1.885e+02 2.501e+03 1.630e-01 2.870e-01 4.270e-01
 2.010e-01 3.040e-01 9.600e-02 2.873e+00 4.885e+00 2.198e+01 5.422e+02
 3.100e-02 1.350e-01 3.960e-01 5.300e-02 6.100e-02 3.000e-02 3.604e+01
 4.954e+01 2.512e+02 4.254e+03 2.230e-01 9.380e-01 1.170e+00 2.910e-01
 5.770e-01 1.490e-01]
스케일 조정 후 특성별 최솟값:
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0.]
스케일 조정 후 특성별 최댓값:
 [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 1. 1. 1. 1.]
```

```python 
In:
X_test_scaled = scaler.transform(X_test)
print(f"스케일 조정 후 특성별 최솟값:")
print(np.array(list(map(lambda x:f"{x:.3f}", X_test_scaled.min(axis=0))), dtype="float"))
print(f"스케일 조정 후 특성별 최댓값:")
print(np.array(list(map(lambda x:f"{x:.3f}", X_test_scaled.max(axis=0))), dtype="float"))
```

```python 
Out:
스케일 조정 후 특성별 최솟값:
[ 0.034  0.023  0.031  0.011  0.141  0.044  0.     0.     0.154 -0.006
 -0.001  0.006  0.004  0.001  0.039  0.011  0.     0.    -0.032  0.007
  0.027  0.058  0.02   0.009  0.109  0.026  0.     0.    -0.    -0.002]
스케일 조정 후 특성별 최댓값:
[0.958 0.815 0.956 0.894 0.811 1.22  0.88  0.933 0.932 1.037 0.427 0.498
 0.441 0.284 0.487 0.739 0.767 0.629 1.337 0.391 0.896 0.793 0.849 0.745
 0.915 1.132 1.07  0.924 1.205 1.631]
```

참고로 테스트 데이터를 조정할 때는 훈련 데이터의 통계치를 사용해야 한다. 



##### 3.3.3 훈련 데이터와 테스트 데이터의 스케일을 같은 방법으로 조정하기

```python 
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)
X_train, X_test = train_test_split(X, random_state=5, test_size=.1)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].scatter(X_train[:, 0], X_train[:, 1], c=mglearn.cm2(0), label="훈련 세트", s =60)
axes[0].scatter(X_test[:, 0], X_test[:, 1],marker='^', c=mglearn.cm2(1), label="테스트 세트", s =60)
axes[0].legend(loc='upper left')
axes[0].set_title("원본 데이터")

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=mglearn.cm2(0), label="훈련 세트", s =60)
axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1],marker='^', c=mglearn.cm2(1), label="테스트 세트", s =60)
axes[1].set_title("스케일 조정된 데이터")

test_scaler = MinMaxScaler()
test_scaler.fit(X_test)
X_test_scaled_badly = test_scaler.transform(X_test)

axes[2].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=mglearn.cm2(0), label="훈련 세트", s =60)
axes[2].scatter(X_test_scaled_badly[:, 0], X_test_scaled_badly[:, 1],marker='^', c=mglearn.cm2(1), label="테스트 세트", s =60)
axes[2].set_title("잘못 조정된 데이터")

for ax in axes:
  ax.set_xlabel("특성 0")
  ax.set_ylabel("특성 1")
```

![](./Figure/3_3_3_1.JPG)

위와 같이 테스트 데이터의 통계치를 사용하여 전처리 했을 경우, 데이터 분포가 달라져 버리는 결과를 초래할 수 있다. 또한 훈련과 테스트 데이터를 분리하기 전에 합쳤을 때의 통계치를 사용해도 문제가 되는데, 그 이유는 모델에 테스트 데이터에 대한 **정보 유출(Information leak)** 이 발생할 수 있기 때문이다.



#####  3.3.4 지도 학습에서 데이터 전처리 효과

```python 
In:
from sklearn.svm import SVC

X_train , X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
svm = SVC(C=100)
svm.fit(X_train, y_train)
print(f"테스트 세트 정확도: {svm.score(X_test, y_test):.2f}")
```

```python 
Out:
테스트 세트 정확도: 0.94
```

```python 
In:
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_sacled = scaler.transform(X_test)

svm.fit(X_train_scaled, y_train)

print(f"스케일 조정된 테스트 세트의 정확도: {svm.score(X_test_sacled, y_test):.2f}")
```

```python 
Out:
스케일 조정된 테스트 세트의 정확도: 0.97
```

```python 
In:
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm.fit(X_train_scaled, y_train)

print(f"SVM 테스트 정확도: {svm.score(X_test_sacled, y_test):.2f}")
```

```python 
Out:
SVM 테스트 정확도: 0.37
```



### 3.4 차원 축소, 특성 추출, 매니폴드 학습

비지도 학습을 이용해 데이터를 변환 시키는 가장 일반적인 이유는 시각화, 압축, 정보가 더 잘 드러나는 표현을 찾기 위함이다.



##### 3.4.1 주성분 분석(PCA)

**주성분 분석(PCA, Principal component analysis)** 는 특성들이 통계적으로 상관관계가 없도록 데이터 셋을 회전 시키는 기술이다. 회전한 뒤에는 종종 새로운 특성 중 중요한 일부만 선택되기도 한다.

![](./Figure/3_4_1_1.JPG)

첫번째 그래프를 보면 주 성분 분석이 성분 1이라고 하는 분산이 가장 큰 방향을 찾은 것을 볼 수 있다. 이 방향(혹은 벡터)이 데이터에서 가장 많은 정보를 담고 있는 방향이다. 즉, 특성들간의 상관 관계가 가장 큰 방향이다. 그 다음으로 첫 번째 방향과 직각인 방향 중에서 가장 많은 정보를 담고 있는 성분 2를 찾아낸다. 고차원에서는 무수히 많은 직각 방향이 존재할 수 있다. 여기서 방향(화살표의 머리와 꼬리)은 중요하지 않다. 이런 과정을 거쳐 찾은 성분을 **주성분(Principal component)** 이라고 한다. 일반적으로는 원본 특성의 갯수만큼 주성분이 있다. 오른쪽 상단의 그래프와 같이 주성분 1과 2를 각각 x축과 y축에 나란하도록 회전하면 두 축은 상관이 없기 때문에 데이터의 상관관계 행렬(Correlation matrix)의 대각선 방향을 제외하고는 모두 0이 된다. 하단의 그래프들과 같이 중요하다고 생각되는 성분만을 남기고 원래 방향으로 회전하는 기술은 데이터에서 노이즈를 제거하거나 주성분에서 유지되는 정보를 시각화하는데 종종 사용한다. 



##### PCA를 적용해 유방암 데이터셋 시각화하기

유방암 데이터셋은 특성을 30개나 가지고 있어서 산점도 행렬로 그리기 힘들다. 이를 양성과 음성 두 클래스에 대해 각 특성의 히스토그램으로 표현하면 다음과 같다.

```python 
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
fig, axes = plt.subplots(5, 6, figsize=(20, 10))
malignant = cancer.data[cancer.target == 0]
benign = cancer.data[cancer.target == 1]

ax = axes.ravel()

for i in range(30):
  _, bins = np.histogram(cancer.data[:, i], bins=50)
  ax[i].hist(malignant[:, i], bins=bins, color=mglearn.cm3(0), alpha=.5)
  ax[i].hist(benign[:, i], bins=bins, color=mglearn.cm3(2), alpha=.5)
  ax[i].set_title(cancer.feature_names[i])
  ax[i].set_yticks(())

ax[0].set_xlabel("특성 크기")
ax[0].set_ylabel("빈도")
ax[0].legend(["악성", "양성"], loc='best')
fig.tight_layout()
```

![](./Figure/3_4_1_2.JPG)

히스토그램을 보면 어떤 특성이 양성과 음성간의 뚜렷한 차이가 있는지는 확인 할 수 있지만 특성들 간의 상호작용이나 상호작용과 클래스와 연관점은 전혀 알 수 없다. PCA를 사용하면 주요 상호작용을 찾아낼 수 있다. 



PCA를 적용하기 전에 특성의 스케일을 조정해야한다.

```python 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)
```



PCA에서는 특잇값 분해(SVD, Singular value decomposition) 방식을 사용해 주성분을 찾는다. fit 메소드에서는 Scipy의 linalg.svd 함수를 이용해 U, s, V 배열을 구한다. transform 메소드에서는 입력 데이터와 주성분 V 행렬의 전치 행렬을 곱하여 변환된 데이터를 구한다. fit_transform에서는 U와 s를 사용해 변환된 데이터를 계산하여 고차원 데이터에서 몇개의 주성분만 고를 경우 성능이 좀 더 좋을 수 있다.

```python 
In:
from sklearn.decomposition import PCA

pca= PCA(n_components=2)
pca.fit(X_scaled)

X_pca = pca.transform(X_scaled)
print(f"원본 데이터 형태: {str(X_scaled.shape)}")
print(f"축소된 데이터 형태: {str(X_pca.shape)}")
```

```python 
Out:
원본 데이터 형태: (569, 30)
축소된 데이터 형태: (569, 2)
```

```python 
plt.figure(figsize=(8, 8))
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
plt.legend(["악성", "양성"], loc='best')
plt.gca().set_aspect("equal")
plt.xlabel("첫 번째 주성분")
plt.ylabel("두 번째 주성분")
```

![](./Figure/3_4_1_3.JPG)

PCA는 비지도 학습이므로 회전 축을 찾을 때 어떤 클래스 정보도 사용하지 않는다. 단순히 데이터 있는 상관관계만을 고려한다.



PCA의 단점은 그래프의 두 축을 해석하기가 쉽지 않다는 점이다. 주성분은 원본 데이터에 있는 어떤 방향에 대응하는 여러 특성이 조하뵌 형태이다. 이런 조합은 보통 매우 복잡하다.

```python 
In:
print(f"PCA 주성분 형태: {pca.components_.shape}")
```

```python 
Out:
PCA 주성분 형태: (2, 30)
```

```python 
In:
print(f"PCA 주성분:\n{pca.components_}")
```

```python 
Out:
PCA 주성분:
[[ 0.21890244  0.10372458  0.22753729  0.22099499  0.14258969  0.23928535
   0.25840048  0.26085376  0.13816696  0.06436335  0.20597878  0.01742803
   0.21132592  0.20286964  0.01453145  0.17039345  0.15358979  0.1834174
   0.04249842  0.10256832  0.22799663  0.10446933  0.23663968  0.22487053
   0.12795256  0.21009588  0.22876753  0.25088597  0.12290456  0.13178394]
 [-0.23385713 -0.05970609 -0.21518136 -0.23107671  0.18611302  0.15189161
   0.06016536 -0.0347675   0.19034877  0.36657547 -0.10555215  0.08997968
  -0.08945723 -0.15229263  0.20443045  0.2327159   0.19720728  0.13032156
   0.183848    0.28009203 -0.21986638 -0.0454673  -0.19987843 -0.21935186
   0.17230435  0.14359317  0.09796411 -0.00825724  0.14188335 .27533947]]
```

각 행은 주성분 하나씩을 나타내고, 중요도에 따라 정렬되어 있다(맨 처음 주성분이 가장 위). 열은 원본 데이터의 특성에 대응하는 값이다. 앞에서 언급한대로 부호는 크게 중요하지 않다.

```python 
plt.matshow(pca.components_, cmap="viridis")
plt.yticks([0, 1], ["첫 번째 주성분", "두 번째 주성분"])
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)), cancer.feature_names, rotation=60, ha='left')
plt.xlabel("특성")
plt.ylabel("주성분")
```

![](./Figure/3_4_1_4.JPG)



##### 고유얼굴(eigenface) 특성 추출

```python 
from sklearn.datasets import fetch_lfw_people

people = fetch_lfw_people(min_faces_per_person=20, resize=.7)
image_shape = people.images[0].shape
fig, axes= plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})

for target, image, ax in zip(people.target, people.images, axes.ravel()):
  ax.imshow(image)
  ax.set_title(people.target_names[target])
```

![](./Figure/3_4_1_5.JPG)

```python 
In:
print(f"people.images.shape: {people.images.shape}")
print(f"클래스 개수: {len(people.target_names)}")
```

```python 
Out:
people.images.shape: (3023, 87, 65)
클래스 개수: 62
```

```python 
In:
counts = np.bincount(people.target)
for i, (count, name) in enumerate(zip(counts, people.target_names)):
  print(f"{name:25} {count:3}", end = '    ')
  if(i + 1) % 3 == 0:
    print()
```

![](./Figure/3_4_1_6.JPG)

```python 
#데이터 편중을 없애기 위해서 사람마다 50개의 이미지만 사용
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
  mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

X_people /= 255.
```

```python 
In:
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print(f"1-최근접 이웃의 테스트 세트 점수: {knn.score(X_test, y_test):.2f}")
```

```python 
Out:
1-최근접 이웃의 테스트 세트 점수: 0.23
```



얼굴의 유사도를 측정하기 위해 원본 픽셀 공간에서 거리를 계산하는 것은 나쁜 방법이다. 이미지의 각 픽셀 값을 다른 이미지에서 동일한 위치에 있는 픽셀 값과 비교하는 방식은 사람이 얼굴 이미지를 인식하는 것과 많이 다르고 얼굴의 특징을 잡아 내기 어렵다. 예를 들어 픽셀을 비교할 때 얼굴 위치가 한 픽셀만 오른쪽으로 이동해도 큰 차이가 난다. 여기에 주성분 분석을 통해 데이터를 주성분으로 변환하여 거리를 계산할 수 있다. PCA의 **화이트닝(Whitening)** 옵션은 주성분의 스케일이 같아지도록 조정한다. 이것은 화이트닝 옵션 없이 주성분으로 변환한 뒤에 StandardScaler를 적용하는 것과 같다(PCA로 변환된 데이터의 표준편차는 linalg.svd 함수에서 반환한 특잇값 배열 s를 샘플 개수의 제곱근으로 나누어 구할 수 있다. PCA 변환은 데이터의 평균을 0으로 만들어 주므로 화이트닝 옵션에서 이 표준편차를 나누어 적용하는 것은 곧 StandardScaler 적용 한 것과 같다).

![](./Figure/3_4_1_7.JPG)

```python 
In:
pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"X_train_pca.shape: {X_train_pca.shape}")
```

```python 
Out:
X_train_pca.shape: (1547, 100)
```

```python 
In:
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca, y_train)
print(f"테스트 세트 정확도: {knn.score(X_test_pca, y_test):.2f}")
```

```python 
Out:
테스트 세트 정확도: 0.31
```

```python 
In:
print(f"pca.components_.shape: {pca.components_.shape}")
```

```python 
Out:
pca.components_.shape: (100, 5655)
```

```python 
fig, axes = plt.subplots(3, 5, figsize=(15, 12), subplot_kw={'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
  ax.imshow(component.reshape(image_shape), cmap='viridis')
  ax.set_title(f"주성분 {i+1}")
```

![](./Figure/3_4_1_8.JPG)

시각화한 주성분들은 완전히 이해할 수는 없지만 몇몇 주성분이 잡아낸 얼굴 이미지의 특징을 짐작할 수는 있다. 예를 들어 첫 번째 주성분은 얼굴과 배경의 명암 차이를 기록한 것으로 보이고 두 번째 주성분은 오른쪽과 왼쪽 조명의 차이를 담고 있는 것 같아 보인다.



주성분의 갯수에 따라 이미지가 어떻게 변하는지 확인이 가능하다.
(87x65 크기의 샘플데이터 1x5,655에 주성분의 전치행렬 5,655x100을 곱하면 100개의 새로운 특성 값을 얻는다. 여기에(1x100) 주성분(100x5,655)을 곱하면 원본 샘플(1x5,655)를 얻을 수 있다.)

![](./Figure/3_4_1_9.JPG)



```python 
mglearn.discrete_scatter(X_train_pca[:, 0], X_train_pca[:, 1], y_train)
plt.xlabel("첫 번째 주성분")
plt.ylabel("두 번째 주성분")
```

![](./Figure/3_4_1_10.JPG)



##### 3.4.2 비음수 행렬 분해(NMF)

**NMF(Non-negative matrix factorization)** 는 PCA와 비슷하게 유용한 특성을 뽑아낼 수 있고 차원 축소에도 사용할 수 있다.  PCA와 다른 점은, PCA의 경우, 데이터의 분산이 가장 크고 수직인 성분을 찾았다면 NMF에서는 음수가 아닌 성분과 계수 값을 찾는다. 즉, 주성분과 계수가 0보다 크거나 같다.

이 알고리즘은 오디오 트랙에서 특정 소리를 추출하는 것과 같은 음수가 아닌 특성을 가진 데이터에만 적용 가능하다.



##### 인위적 데이터에 NMF 적용하기

PCA를 사용할때와는 달리 NMF로 데이터를 다루려면 주어진 데이터가 양수인지 확인해야 한다. 즉 원점(0, 0)에서 데이터로 가는 방향을 추출한 것으로 음수 미포함 성분을 이해할 수 있다.

![](./Figure/3_4_2_1.JPG)

왼쪽 그래프는 성분이 둘인 NMF로, 데이터셋의 모든 포인트를 양수로 이뤄진 두 개의 성분으로 표현 가능하다. 데이터를 완벽하게 재구성할 수 있을만큼 성분이 아주 많다면(특성 개수만큼 많다면) 알고리즘은 데이터의 각 특성의 끝에 위치한 포인트를 가리키는 방향을 선택할 것이다.

(Scikit-learn에서 NMF 알고리즘은 입력 데이터 X, 변환 데이터 W, 성분 H가 X = WH를 만족하는 W, H 행렬을 구하기 위해 행렬의 L2 Norm인 프로베니우스 Norm(Frobenius norm)의 제곱으로 만든 목적 함수  ![](./Figure/3_4_2_2.JPG) 을 좌표 하강법으로 최소화 한다. 구해진 성분 H는 NMF 객체의 component_ 속성에 저장된다.)

오른쪽 그래프와 같이 하나의 성분만 사용한다면 NMF는 데이터를 가장 잘 표현할 수 있는 평균으로 향하는 성분을 만든다. PCA와는 달리, 성분 개수를 줄이면 특정 방향이 단순히 제거되는 것이 아니라 전체 성분이 완전히 바뀐다. NMF에서의 성분은 순서가 없고 모든 성분을 동등하게 취급한다.

NMF는 난수 생성 초깃 값에 따라 결과가 달라진다. 간단한 예에서는 영향을 주지 않지만 복잡한 데이터에서는 큰 차이를 만들 수 있다.

(NMF에서 기본 초기화 방식은 데이터 평균을 성분의 개수로 나눈 후 제곱근을 구하고, 그런 다음 정규 분포의 난수를 발생시켜 앞에서 구한 제곱근을 곱하여 H와 W 행렬을 만든다. 이는 데이터 평균값을 각 성분과 두 개의 행렬에 나누어 놓는 효과를 발생시킨다.)



##### 얼굴 이미지에 NMF 적용하기

다음은 NMF를 사용해 데이터를 재구성했을 때 성분의 개수에 따른 이미지의 모습이다.

![](./Figure/3_4_2_3.JPG)

PCA가 재구성 측면에서 최선의 방향을 찾는다면 NMF는 데이터를 인코딩하거나 재구성하는 용도로 사용하기 보다는 데이터에 있는 패턴을 찾는데 활용한다.



```python 
from sklearn.decomposition import NMF

nmf = NMF(n_components=15, random_state=0)
nmf.fit(X_train)
X_train_nmf = nmf.transform(X_train)
X_test_nmf = nmf.transform(X_test)
fig, axes = plt.subplots(3, 5, figsize=(15, 12), subplot_kw={'xticks': (), 'yticks': ()})
for i, (component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
  ax.imshow(component.reshape(image_shape))
  ax.set_title(f"성분 {i}")
```

![](./Figure/3_4_2_4.JPG)



위에서 성분3이 오른쪽으로 조금 돌아간 얼굴로 보인 것을 확인할 수 있다. 실제 확인해보면 다음과 같다.

```python 
compn = 3
inds = np.argsort(X_train_nmf[:, compn])[::-1]
fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
  ax.imshow(X_train[ind].reshape(image_shape))
```

![](./Figure/3_4_2_5.JPG)



```python 
compn = 7
inds = np.argsort(X_train_nmf[:, compn])[::-1]
fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
  ax.imshow(X_train[ind].reshape(image_shape))
```

![](./Figure/3_4_2_6.JPG)



다음은 임의의 합성 신호들을 NMF와 PCA로 복원한 예이다.

```python 
S = mglearn.datasets.make_signals()
plt.figure(figsize=(6, 1))
plt.plot(S, '-')
plt.xlabel("시간")
plt.ylabel("신호")
```

![](./Figure/3_4_2_7.JPG)

```python 
In:
A = np.random.RandomState(0).uniform(size=(100, 3))
X = np.dot(S, A.T)
print(f"측정 데이터 형태: {X.shape}")
```

```python 
Out:
측정 데이터 형태: (2000, 100)
```

```python 
In:
nmf = NMF(n_components=3, random_state=42)
S_ = nmf.fit_transform(X)
print(f"복원한 신호 데이터 형태 {S_.shape}")
```

```python 
Out:
복원한 신호 데이터 형태 (2000, 3)
```

```python 
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
H = pca.fit_transform(X)
```

```python 
models = [X, S, S_, H]
names = ['측정 신호 (처음 3개)', '원본 신호', 'NMF로 복원한 신호', 'PCA로 복원한 신호']
fig, axes = plt.subplots(4, figsize=(8, 4), gridspec_kw={'hspace': .5}, subplot_kw={'xticks': (), 'yticks': ()})

for model, name, ax in zip(models, names, axes):
  ax.set_title(name)
  ax.plot(model[:, :3], '-')
```

![](./Figure/3_4_2_8.JPG)



패턴 추출에 관해서는 독립 성분 분석(ICA), 요인 분석(FA), 희소 코딩(Sparse coding)-딕셔너리(Dictionary 학습)에 관해 설명하고 있는 다음 페이지를 참고.

[Scikit-learn 분해 메서드](https://scikit-learn.org/stable/modules/decomposition.html)



##### 3.4.3 t-SNE를 이용한 매니폴드 학습

**t-SNE(t-Distributed Stochastic Neighbor Embedding)** 알고리즘은 **매니폴드 학습(Manifold learning)** 알고리즘이라고 하는 시각화 알고리즘의 한 종류이다. 시각화가 목적이기 때문에 3개 이상의 특성을 뽑는 경우는 없다. 매니폴드 학습 알고리즘은 훈련 데이터를 새로운 표현으로 변환 시키지만 새로운 데이터(테스트 데이터)에는 적용할 수 없고 훈련했던 데이터만 변환 할 수 있다. 그래서 탐색적 데이터 분석에 유용하지만 지도 학습의 전처리용으로는 사용하지 않는다. t-SNE는 데이터 포인트 사이의 거리를 가장 잘 보존하는 2차원 표현을 찾는다. 각 데이터 포인트를 2차원에 무작위로 표현한 뒤 원본 특성 공간에 가까운 포인트는 가깝게, 멀리 떨어진 포인트는 멀어지게 만든다. 멀리 떨어진 포인트와 거리를 보존하는 것보다 가까이 있는 포인트에 더 많은 비중을 두는데, 이는 이웃 데이터 포인트에 대한 정보를 보존하려고 하는 것이다.

(scikit-learn의 t-SNE 구현은 쿨백-라이블러 발산(Kullback-Leibler divergence) 목적 함수를 최적하기 위해 모멘텀을 적용한 배치 경사 하강법을 사용한다.  TSNE의 method 매개변수의 기본 값은 'barnes_hut'로 그래디언트 계산의 복잡도를 O(N^2)에서 O(NlogN))으로 낮춰주는 반스-헛(Barnes-Hut) 방법이다. 'exact' 옵션은 정확한 계산을 하지만 느리므로 대량의 데이터에는 적합하지 않다.)

```python 
from sklearn.datasets import load_digits

digits = load_digits()
fig, axes = plt.subplots(2, 5, figsize=(10, 5), subplot_kw={'xticks': (), 'yticks': ()})
for ax, img in zip(axes.ravel(), digits.images):
  ax.imshow(img)
```

![](./Figure/3_4_3_1.JPG)



PCA를 사용한 데이터 변환의 시각화

```python 
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(digits.data)
digits_pca = pca.transform(digits.data)
digits_pca = pca.transform(digits.data)
colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525",
          "#A83683", "#4E655E", "#853541", "#3A3120","#535D8E"]
plt.figure(figsize=(10, 10))
plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max())
plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max())
for i in range(len(digits.data)):
  plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits.target[i]), 
           color=colors[digits.target[i]], fontdict={'weight': 'bold', 'size': 9})
plt.xlabel("첫 번째 주성분")
plt.ylabel("두 번째 주성분")
```

![](./Figure/3_4_3_2.JPG)



t-SNE를 사용한 데이터 변환의 시각화

```python 
from sklearn.manifold import TSNE

tsne = TSNE(random_state=42)
digits_tsne = tsne.fit_transform(digits.data)
colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525",
          "#A83683", "#4E655E", "#853541", "#3A3120","#535D8E"]
plt.figure(figsize=(10, 10))
plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max())
plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max())
for i in range(len(digits.data)):
  plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]), 
           color=colors[digits.target[i]], fontdict={'weight': 'bold', 'size': 9})
plt.xlabel("t-SNE 특성 0")
plt.ylabel("t-SNE 특성 1")
```

![](./Figure/3_4_3_3.JPG)

t-SNE의 매개변수 조정을 고려해 볼만한 매개변수로 perplexity와 early_exaggeration이 있다. perplexity의 값이 크면 더 많은 이웃을 포함하며 작은 그룹이 무시된다. 보통 데이터셋이 클 경우 perplexity 값도 커져야 한다. TSNE 모델은 초기 과정(early exaggeration) 단계와 최적화 단계를 가지는데 early_exaggeration 매개변수는 초기 과장 단계에서 원본 공간의 클러스터들이 얼마나 멀게 2차원에 나타낼지르 정한다. early_exaggeration의 값이 클수록 간격이 커진다.



### 3.5 군집

**군집(Clustering)** 은 데이터 셋을 **클러스터(Cluster)** 라는 그룹으로 나누는 작업이다. 클러스터 안의 데이터 포인트 끼리는 비슷하고 다른 클러스트의 데이터 포인트와는 구분된다.  군집 알고리즘은 각 데이터 포인트가 어느 클러스터에 속하는지 할당(예측)한다.



##### 3.5.1 k-평균 군집

***k*-평균(*k*-means)** 군집은 어떤 영역을 대표하는 **클러스터 중심(Cluster center)** 을 찾는다. 먼저 데이터 포인트를 가장 가까운 클러스터 중심에 할당하고, 그런 다음 클러스터에 할당된 데이터 포인트의 평균으로 클러스터 중심을 다시 지정한다. 이 과정은 클러스터에 할당되는 데이터 포인트에 변화가 없을 때까지 진행된다. 

```python 
mglearn.plots.plot_kmeans_algorithm()
```

![](./Figure/3_5_1_1.JPG)



새로운 데이터 포인트가 주어지면 k-평균 알고리즘은 가장 가까운 클러스터 중심을 할당한다.

```python 
mglearn.plots.plot_kmeans_boundaries()
```

![](C:\Users\LAB\Desktop\3_5_1_2.JPG)



```python 
In:
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y = make_blobs(random_state=1)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
```

```python 
Out:
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
       n_clusters=3, n_init=10, n_jobs=None, precompute_distances='auto',
       random_state=None, tol=0.0001, verbose=0)
```

```python 
In:
print(f"클러스터 레이블:\n{kmeans.labels_}")
```

```python 
Out:
클러스터 레이블:
[1 2 2 2 0 0 0 2 1 1 2 2 0 1 0 0 0 1 2 2 0 2 0 1 2 0 0 1 1 0 1 1 0 1 2 0 2
 2 2 0 0 2 1 2 2 0 1 1 1 1 2 0 0 0 1 0 2 2 1 1 2 0 0 2 2 0 1 0 1 2 2 2 0 1
 1 2 0 0 1 2 1 2 2 0 1 1 1 1 2 1 0 1 1 2 2 0 0 1 0 1]
```

```python 
In:
print(kmeans.predict(X))
```

```python 
Out:
[1 2 2 2 0 0 0 2 1 1 2 2 0 1 0 0 0 1 2 2 0 2 0 1 2 0 0 1 1 0 1 1 0 1 2 0 2
 2 2 0 0 2 1 2 2 0 1 1 1 1 2 0 0 0 1 0 2 2 1 1 2 0 0 2 2 0 1 0 1 2 2 2 0 1
 1 2 0 0 1 2 1 2 2 0 1 1 1 1 2 1 0 1 1 2 2 0 0 1 0 1]
```

군집은 각 데이터 포인트가 레이블을 가진다는 면에서 분류와 조금 비슷해 보이나 레이블 자체에 아무 의미가 없고 정답을 모르고 있다는 점에서 다르다. 



```python 
mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o')
mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1, 2], markers='^', markeredgewidth=2)
```

![](./Figure/3_5_1_3.JPG)

```python 
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
assignments = kmeans.labels_

mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[0])

kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
assignments = kmeans.labels_

mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[1])
```

![](./Figure/3_5_1_4.JPG)



##### k-평균 알고리즘이 실패하는 경우

데이터 셋의 클러스터의 개수를 정확하게 알고 있더라도 k-평균 알고리즘이 항상 이를 구분해낼 수 있는 것은 아니다. 각 클러스터를 정의 하는 것이 중심 하나뿐이므로 클러스터는 둥근 형태로 나타난다. k-평균은 모든 클러스터의 반경이 똑같다고 가정한다. 그래서 클러스터 중심 사이의 정확히 중간에 경계를 그린다.

```python 
X_varided, y_varied = make_blobs(n_samples=200, cluster_std=[1.0, 2.5, 0.5], random_state=170)
y_pred = KMeans(n_clusters=3, random_state=0).fit_predict(X_varided)
mglearn.discrete_scatter(X_varided[:, 0], X_varided[:, 1], y_pred)
plt.legend(['클러스터 0', '클러스터 1', '클러스터 2'], loc='best')
plt.xlabel("특성 0")
plt.ylabel("특성 1")
```

![](./Figure/3_5_1_5.JPG)

```python 
X, y = make_blobs(random_state=170, n_samples=600)
rng = np.random.RandomState(74)

transformation = rng.normal(size=(2, 2))
X = np.dot(X, transformation)

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_pred = kmeans.predict(X)

mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o')
mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1, 2], markers='^', markeredgewidth=2)
plt.xlabel("특성 0")
plt.ylabel("특성 1")
```

![](./Figure/3_5_1_6.JPG)

```python 
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=200, noise=.05, random_state=0)

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
y_pred = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=mglearn.cm2, s=60, edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='^', 
            c=[mglearn.cm2(0), mglearn.cm2(1)], s=100, linewidths=2, edgecolors='k')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
```

![](./Figure/3_5_1_7.JPG)



##### 벡터 양자화 또는 분해 메서드로서의 k-평균

k-평균은 클러스터 중심으로 각 데이터를 표현한다. 이를 각 데이터 포인트가 클러스터 중심, 즉 하나의 성분으로 표현된다고 볼 수 있다. k-평균을 이렇게 각 포인트가 하나의 성분으로 분해되는 관점으로 보는 것을 **벡터 양자화(Vector quantization)** 이라고 한다. 

```python 
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF, PCA
from sklearn.cluster import KMeans

X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
nmf = NMF(n_components=100, random_state=0)
nmf.fit(X_train)
pca = PCA(n_components=100, random_state=0)
pca.fit(X_train)
kmeans = KMeans(n_clusters=100, random_state=0)
kmeans.fit(X_train)
X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test))
X_reconstructed_kmenas = kmeans.cluster_centers_[kmeans.predict(X_test)]
X_reconstructed_nmf = np.dot(nmf.transform(X_test), nmf.components_)
fig, axes = plt.subplots(3, 5, figsize=(8, 8), subplot_kw={'xticks': (), 'yticks': ()})
fig.suptitle("추출한 성분")
for ax, comp_kmeans, comp_pca, comp_nmf in zip(axes.T, kmeans.cluster_centers_, pca.components_, nmf.components_):
  ax[0].imshow(comp_kmeans.reshape(image_shape))
  ax[1].imshow(comp_pca.reshape(image_shape), cmap='viridis')
  ax[2].imshow(comp_nmf.reshape(image_shape))

axes[0, 0].set_ylabel('kmeans')
axes[1, 0].set_ylabel('pca')
axes[2, 0].set_ylabel("nmf")

fig, axes = plt.subplots(4, 5, figsize=(8, 8), subplot_kw={'xticks': (), 'yticks': ()})
fig.suptitle("재구성")
for ax, orig, rec_kmeans, rec_pca, rec_nmf in zip(axes.T, X_test, X_reconstructed_kmenas, X_reconstructed_pca,
                                                  X_reconstructed_nmf):
  ax[0].imshow(orig.reshape(image_shape))
  ax[1].imshow(rec_kmeans.reshape(image_shape))
  ax[2].imshow(rec_pca.reshape(image_shape))
  ax[3].imshow(rec_nmf.reshape(image_shape))

axes[0, 0].set_ylabel("원본")
axes[1, 0].set_ylabel("kmeans")
axes[2, 0].set_ylabel("pca")
axes[3, 0].set_ylabel("nmf")
```

![](./Figure/3_5_1_8.JPG)



k-평균을 사용한 벡터 양자화는 입력 데이터의 차원보다 더 많은 클러스터를 사용해 데이터 인코딩을 할 수 있다. 입력 데이터가 2차원 일때, PCA와 NMF를 사용해 1차원으로 축소하면 데이터의 구조가 완전히 파괴된다. 이때 k-평균은 데이터를 보다 잘 표현 할 수 있다.

```python 
In:
X, y = make_moons(n_samples=200, noise=.05, random_state=0)

kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(X)
y_pred = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=60, cmap='Paired', edgecolors='black')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=60, marker='^', c=range(kmeans.n_clusters), linewidths=2, cmap='Paired', edgecolors='black')
plt.xlabel("특성 0")
plt.ylabel("특성 1")
print(f"클러스터 레이블:\n{y_pred}")
```

```python 
Out:
클러스터 레이블:
[4 7 6 9 7 7 4 1 4 1 8 3 7 1 0 4 2 3 2 0 5 1 2 1 8 6 7 5 6 2 4 8 1 6 4 5 3
 4 0 6 3 8 2 6 7 8 4 0 6 1 0 3 5 9 1 4 2 1 2 8 3 9 7 4 1 9 8 7 8 9 3 9 3 6
 1 9 6 4 2 3 5 8 3 5 6 8 4 8 3 5 2 4 5 0 5 7 7 3 9 6 1 5 8 4 9 6 9 8 7 2 0
 8 8 9 4 1 2 5 3 4 4 0 6 8 6 0 4 6 1 5 4 0 9 3 1 7 1 9 5 4 6 6 2 8 8 4 6 1
 2 6 3 7 4 2 3 8 1 3 2 2 6 1 2 7 3 7 2 3 7 1 2 9 0 0 6 1 5 0 0 2 7 0 5 7 5
 2 8 3 9 0 9 2 4 4 6 0 5 6 2 7]
```

![](./Figure/3_5_1_9.JPG)

각 데이터로부터 클러스터의 중심까지의 거리는 다음과 같이 구할 수 있다(transform 메소드의 리턴값,  # of samples * # of clusters).

```python 
In:
distance_features = kmeans.transform(X)
print(f"클러스터 거리 데이터의 형태 : {distance_features.shape}")
print(f"클러스터 거리:\n{distance_features}")
```

```python 
Out:
클러스터 거리 데이터의 형태 : (200, 10)
클러스터 거리:
[[1.54731274 1.03376805 0.52485524 ... 1.14060718 1.12484411 1.80791793]
 [2.56907679 0.50806038 1.72923085 ... 0.149581   2.27569325 2.66814112]
 [0.80949799 1.35912551 0.7503402  ... 1.76451208 0.71910707 0.95077955]
 ...
 [1.12985081 1.04864197 0.91717872 ... 1.50934512 1.04915948 1.17816482]
 [0.90881164 1.77871545 0.33200664 ... 1.98349977 0.34346911 1.32756232]
 [2.51141196 0.55940949 1.62142259 ... 0.04819401 2.189235   2.63792601]]
```

k-평균은 대용량 데이터셋에도 잘 작동하지만 scikit-learn은 아주 큰 대규모 데이터셋을 처리할 수 있는 MiniBatchMeans(알고리즘이 반복 될 때 전체 데이터에서 일부를 무작위로 선택해(미니 배치) 클러스터의 중심을 계산한다. batch_size 매개변수로 지정)도 제공한다.

k-평균의 단점은 무작위 초기화를 사용하여 알고리즘의 출력이 난수 초깃값에 따라 달라진다는 점이다. 서로 다른 난수 초깃 값으로 N번(기본 10번) 반복하여 최선의 결과(클러스터의 분산의 합이 작은 것)를 도출해낸다(KMeans의 n_init 매개변수는 알고리즘 전체를 다른 난수 초깃값을 사용해 반복하는 횟수를 지정한다. MiniBatchKMeans의 n_init 매개변수는 최선의 초기 클러스터 중심을 찾는 데 사용하는 반복 횟수를 지정한다). k-평균의 다른 단점은 클러스터의 모양을 가정하고 있어서 활용 범위가 비교적 제한적이고 찾으려하는 클러스터의 개수를 지정해야 한다는 점이다.



##### 3.5.2 병합 군집

**병합 군집(Agglomerative clustering)** 은 시작할 때 각 포인트를 하나의 클러스터로 지정하고, 어떤 종료 조건을 만족할 때까지 가장 비슷한 두 클러스터를 합쳐 나간다. scikit-learn에서 사용하는 조욜 조건은 클러스터 개수로, 지정된 개수의 클러스터가 남을 때까지 비슷한 클러스터를 합친다. linkage 옵션에서 가장 비슷한 클러스터를 측정하는 방법을 지정하면 이 측정은 항상 두 클러스터 사이에서 이뤄진다. 

- ward - 기본값. 모든 클러스터 내의 분산을 가장 작게 증가시키는 두 클러스터를 합침.  대체로 비슷한 크기의 클러스터 생성
- average - 클러스터 포인트 사이의 평균 거리가 가장 짧은 두 클러스터를 합침.
- complete - 최대 연결. 클러스터 포인트 사이의 최대 거리가 가장 짧은 두 클러스터를 합침.

대체로 ward 옵션을 사용하나, 클러스터에 속한 포인트 수가 많이 다를 땐(한 클러스터 다른 것보다 매우 클 때 등) average나 complete가 더 나을 수 있다.

![](./Figure/3_5_2_1.JPG)



알고리즘의 특성 상 병합 군집은 새로운 데이터 포인트에 대해서는 예측이 불가하다. 그러므로 병합 군집에는 predict 메소드가 없다. 대신 훈련 세트로 모델을 만들고 클러스터 소속 정보를 얻기 위해 fit_predict 메소드를 사용한다.

```python 
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs

X, y = make_blobs(random_state=1)

agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X)

mglearn.discrete_scatter(X[:, 0], X[:, 1], assignment)
plt.legend(["클러스터 0", "클러스터 1", "클러스터 2"], loc='best')
plt.xlabel("특성 0")
plt.ylabel('특성 1')
```

![](./Figure/3_5_2_2.JPG)



##### 계층적 군집과 덴드로그램

병합 군집은 **계층적 군집(Hierarchical clustering)** 을 만든다. 군집이 반복하여 진행되면 모든 포인트는 하나의 포인트를 가진 클러스터에서 시작하여 마지막 클러스터까지 이동한다. 각 중간 단계는 데이터에 대한 (각기 다른 개수의) 클러스터를 생성한다. 

![](./Figure/3_5_2_3.JPG)



**덴드로그램(Dendrogram)** 은 다차원 데이터셋의 계층 군집을 시각화 할 수 있다. 

```python 
from scipy.cluster.hierarchy import dendrogram, ward

X, y = make_blobs(random_state=0, n_samples=12)
linkage_array = ward(X)
dendrogram(linkage_array)

ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [7.25, 7.25], '--', c='k')
ax.plot(bounds, [4, 4], '--', c='k')

ax.text(bounds[1], 7.25, ' 두 개 클러스터', va='center', fontdict={'size': 15})
ax.text(bounds[1], 4, ' 세 개 클러스터', va='center', fontdict={'size': 15})
plt.xlabel("샘플 번호")
plt.ylabel("클러스터 거리")
```

![](./Figure/3_5_2_4.JPG)

덴드로그램의 가지의 길이는 합쳐진 클러스터가 얼마나 멀리 떨어져 있는지를 보여준다. '세 개 클러스터'로 표시한 점선이 가로지르는 세 개의 수직선의 길이가 가장 긴데 이것은 세 개에서 두 개로 될 때 꽤 먼 거리의 포인트를 모은다는 뜻이다.



##### 3.5.3 DBSCAN

**DBSCAN(Density-based spatial clustering of applications with noise)** 는 클러스터의 개수를 미리 지정할 필요가 없다. 복잡한 형상도 찾을 수 있고 어떤 클래스에도 속하지 않는 포인트를 구분할 수 있다. 병합 군집이나 *k*-평균보다는 느리지만 비교적 큰 데이터 셋에도 적용할 수 있다. 

DBSCAN은 특성 공간에서 가까이 있는 데이터가 많아 붐비는 지역의 포인트를 찾는다. 이런 지역을 특성 공간의 **밀집 지역(Dense region)** 이라고 한다. 데이터의 밀집 지역을 한 클러스터로 구성하며 비교적 비어 있는 지역을 경계로 다른 클러스터와 구분한다.

밀집 지역에 있는 포인트를 **핵심 샘플(핵심 포인트)** 라고 하며 다음과 같이 정의한다. 두 개의 매개변수 min_samples와 eps가 있는데 한 데이터 포인트 안에서 eps 거리 안에 데이터 포인트가 min_samples 만큼 있으면 이 데이터 포인트를 핵심 샘플로 분류한다(거리를 재는 방식은 metric 매개변수에서 조정 가능하며 기본 값은 'Euclidean'이다). eps보다 가까운 핵심 샘플은 DBSCAN에 의해 동일한 클러스터로 합쳐진다. 

1.   무작위로 포인트를 선택.

2. 1에서 선택한 포인트에서 eps 거리 안의 모든 포인트를 찾는다

    2-1. 만약 eps 거리 안에 있는 포인트 수가 min_samples보다 적다면 어떤 클래스에도 속하지 않

   ​        는 잡음(Noise)로 레이블한다.

    2-2. eps 거리 안에 min_samples보다 많은 포인트가 있다면 그 포인트는 핵심 샘플로 레이블하고 

   ​        새로운 클러스터 레이블을 할당한다.

3. 2에서 레이블을 할당한 포인트의 eps 거리 내의 모든 이웃을 살핀다.

    3-1. 이웃이 아직 어떤 클러스터에도 할당 되지 않았다면 바로 직전에 만들었던 클러스터 레이블     

   ​         을 할당 한다.

    3-2. 이웃이 핵심 샘플이면 그 포인틔 이웃을 차례로 방문 한다.

4. 클러스터는 eps 거리 안에 더 이상 핵심 샘플이 없을 때까지 커진다. 

5. 아직 방문하지 않은 포인트를 선택하여 같은 과정을 반복한다.

포인트의 종류는 3가지 이다(핵심, 경계 - 핵심 포인트에서 eps 거리 안에 있는 포인트, 잡음). DBSCAN을 한 데이터 셋에 여러번 실행하면 핵심 포인트의 군집은 항상 같고 매번 같은 포인트를 잡음으로 레이블 한다. 그러나 경계 포인트는 한 개 이상의 클러스터 핵심 샘플의 이웃이 될 수 있는데 이때는 포인트를 방문하는 순서에 따라 클러스터 레이블이 바뀔 수 있다. 

병합 군집과 마찬가지로 새로운 테스터 데이터에 대해 예측 할 수 없으므로 fit_predict 메소드를 사용하여 군집과 클러스터 레이블을 한 번에 계산한다.

```python 
In:
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

X, y = make_blobs(random_state=0, n_samples=12)

dbscan = DBSCAN()
clusters = dbscan.fit_predict(X)
print(f"클러스터 레이블:\n{clusters}")
```

```python 
Out:
클러스터 레이블:
[-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]    
```

여기서는 작은 데이터셋에는 적합하지 않은 eps와 min_samples의 기본 값 때문에 모든 포인트에 잡음을 의미하는 -1이 부여되었다.



```python 
In:
mglearn.plots.plot_dbscan()
```

```python 
Out:
min_samples: 2 eps: 1.000000  클러스터: [-1  0  0 -1  0 -1  1  1  0  1 -1 -1]
min_samples: 2 eps: 1.500000  클러스터: [0 1 1 1 1 0 2 2 1 2 2 0]
min_samples: 2 eps: 2.000000  클러스터: [0 1 1 1 1 0 0 0 1 0 0 0]
min_samples: 2 eps: 3.000000  클러스터: [0 0 0 0 0 0 0 0 0 0 0 0]
min_samples: 3 eps: 1.000000  클러스터: [-1  0  0 -1  0 -1  1  1  0  1 -1 -1]
min_samples: 3 eps: 1.500000  클러스터: [0 1 1 1 1 0 2 2 1 2 2 0]
min_samples: 3 eps: 2.000000  클러스터: [0 1 1 1 1 0 0 0 1 0 0 0]
min_samples: 3 eps: 3.000000  클러스터: [0 0 0 0 0 0 0 0 0 0 0 0]
min_samples: 5 eps: 1.000000  클러스터: [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]
min_samples: 5 eps: 1.500000  클러스터: [-1  0  0  0  0 -1 -1 -1  0 -1 -1 -1]
min_samples: 5 eps: 2.000000  클러스터: [-1  0  0  0  0 -1 -1 -1  0 -1 -1 -1]
min_samples: 5 eps: 3.000000  클러스터: [0 0 0 0 0 0 0 0 0 0 0 0]
```

![](./Figure/3_5_3_1.JPG)

 eps를 증가시키면(왼쪽에서 오른쪽) 하나의 클러스터에 더 많은 포인트가 포함된다. 이는 클러스터를 커지게 하고 여러 클러스터를 하나로 합치게도 만든다. min_smaples를 키우면(위에서 아래) 핵심 포인트 수가 줄어들며 잡음 포인트가 늘어난다.  eps 매개 변수는 가까운 포인트의 범위를 결정하기 때문에 중요하다. eps를 매우 작게 하면 어떤 포인트도 핵심 포인트가 되지 못하고, 모든 포인트가 잡음 포인트가 될 수 있다. eps를 매우 크게 하면 모든 포인트가 단 하나의 클러스터에 속하게 된다. min_smaples 설정은 덜 조밀한 지역에 있는 포인트들이 잡음이 될지 하나의 클러스터가 될 지를 결정하는 데 중요한 역할을 한다. min_samples를 늘리면 min_samples의 수보다 작은 클러스터들은 잡음이 된다. 따라서 min_samples는 클러스터의 최소 크기를 결정한다. 

적절한 eps 값을 쉽게 찾으려면 StandardScaler나 MinMaxScaler로 모든 특성의 스케일을 비슷한 범위로 조정해 주는 것이 좋다.

```python 
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

X, y = make_moons(n_samples=200, noise=.05, random_state=0)

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

dbscan = DBSCAN()
clusters = dbscan.fit_predict(X_scaled)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm2, s=60,
            edgecolors='black')
plt.xlabel("특성 0")
plt.ylabel("특성 1")
```

![](./Figure/3_5_3_2.JPG)




##### 3.5.4 군집 알고리즘의 비교와 평가

##### 타깃값으로 군집 평가하기

군집 알고리즘의 결과를 실제 정답 클러스터와 비교하여 평가할 수 있는 지표들이 있다. 

1(최적일 때)와 0(무작위로 분류될 때) 사이의 값을 제공하는 **ARI(Adjusted rand index)** 와 **NMI(Normalized mutual information)** 이다.

무작위로 클러스터에 포인트를 할당할 경우 ARI 값은 0에 가까워지며, 무작위 할당보다도 나쁘게 군집되면 음수 값을 가질 수 있다. NMI를 위한 함수 사용법은 ARI의 adjusted_rand_score와 같다.

```python 
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

X, y = make_moons(n_samples=200, noise=.05, random_state=0)

scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

fig, axes = plt.subplots(1, 4, figsize=(15, 3), subplot_kw={'xticks':(), 'yticks':()})

algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2), DBSCAN()]

random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X))

axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters, cmap=mglearn.cm3, s=60,
                edgecolors='black')
axes[0].set_title(f"무작위 할당 - ARI: {adjusted_rand_score(y, random_clusters):.2f}")

for ax, algorithm in zip(axes[1:], algorithms):
  clusters = algorithm.fit_predict(X_scaled)
  ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm3, s=60,
             edgecolors='black')
  ax.set_title(f"{algorithm.__class__.__name__} - ARI: {adjusted_rand_score(y, clusters):.2f}")
```

![](./Figure/3_5_4_1.JPG)



다음은 흔히 하는 실수로 군집 모델을 평가할 때 accuracy_score를 사용하는 것이다.

```python 
In:
from sklearn.metrics import accuracy_score

clusters1 = [0, 0, 1, 1, 0]
clusters2 = [1, 1, 0, 0, 1]

print(f"정확도: {accuracy_score(clusters1, clusters2):.2f}")
print(f"ARI: {adjusted_rand_score(clusters1, clusters2):.2f}")    
```

```python 
Out:
정확도: 0.00
ARI: 1.00    
```



##### 타깃값 없이 군집 평가하기

앞에서 설명한 ARI 같은 방법에는 큰 문제점이 있다. 보통 군집 알고리즘을 적용할 때는 그 결과와 비교할 타깃이 없다. 데이터가 속한 정확한 클러스터를 알고 있다면 지도 학습 모델을 만들 것이다. 그러므로 ARI나 NMI 같은 지표는 알고리즘을 개발할 때나 도움이 된다.

타깃값이 필요 없는 군집용 지표로 **실루엣 계수(Silhouette coefficient)** 가 있다.  실제로 잘 동작하지는 않는다. 실루엣 점수는 클러스터의 밀집 정도를 계산하는 것으로, 높을수록 좋고 최대 점수는 1이다(-1은 잘못된 군집, 0은 중첩된 클러스터를 뜻한다). 밀집된 클러스터가 좋긴 하나 모양이 복잡할 때는 밀집도를 활용한 평가가 잘 들어맞지 않는다.

```python 
from sklearn.metrics.cluster import silhouette_score

X, y = make_moons(n_samples=200, noise=.05, random_state=0)
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

fig, axes = plt.subplots(1, 4, figsize=(15, 3), subplot_kw={'xticks': (), 'yticks': ()})

random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X))

algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2), DBSCAN()]

axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters, cmap=mglearn.cm3, s=60,
                edgecolors='black')
axes[0].set_title(f"무작위 할당 - ARI: {silhouette_score(X_scaled, random_clusters):.2f}")

for ax, algorithm in zip(axes[1:], algorithms):
  clusters = algorithm.fit_predict(X_scaled)
  ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm3, s=60,
             edgecolors='black')
  ax.set_title(f"{algorithm.__class__.__name__} - ARI: {silhouette_score(X_scaled, clusters):.2f}")
```

![](./Figure/3_5_4_2.JPG)

위 그림과 같이 DBSCAN의 결과가 낫지만 *k*-평균의 실루엣 점수가 더 높다. 클러스터 평가에 더 적합한 전략은 견고성 기반(Robustness-based)의 지표이다. 데이터에 잡음 포인트를 추가하거나 여러 가지 매개변수 설정으로 알고리즘을 실행하고 그 결과를 비교하는 것이다.  매개변수와 데이터에 변화를 주며 반복해도 결과가 일정하다면 신뢰할만 하다고 말할 수 있다. 군집 모델이 매우 안정적이거나 실루엣 점수가 높다고 해도, 군집에 어떤 유의미한 것이 있는지 또는 군집이 데이터에 흥미로운 면을 반영하고 있는지는 알 수 없다. 이를 확인하는 유일한 방법은 클러스터를 직접 확인하는 것이다. 



##### 얼굴 데이터셋으로 군집 알고리즘 비교

LFW 데이터셋으로 군집 알고리즘을 비교한다.

```python 
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA

people = fetch_lfw_people(min_faces_per_person=20, resize=.7)
image_shape = people.images[0].shape

#데이터 편중을 없애기 위해서 사람마다 50개의 이미지만 사용
mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
  mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

X_people /= 255.

pca = PCA(n_components=100, whiten=True, random_state=0)
pca.fit_transform(X_people)
X_pca = pca.transform(X_people)
```



##### DBSCAN으로 얼굴 데이터셋 분석하기

```python 
In:
dbscan = DBSCAN()
labels = dbscan.fit_predict(X_pca)
print(f"고유한 레이블: {np.unique(labels)}")    
```

```python 
Out:
고유한 레이블: [-1]
```

```python 
In:
dbscan = DBSCAN(min_samples=3)
labels = dbscan.fit_predict(X_pca)
print(f"고유한 레이블: {np.unique(labels)}")
```

```python 
Out:
고유한 레이블: [-1]    
```

```python 
In:
dbscan = DBSCAN(min_samples=3, eps=15)
labels = dbscan.fit_predict(X_pca)
print(f"고유한 레이블: {np.unique(labels)}")
```

```python 
Out:
고유한 레이블: [-1  0]
```



잡음 포인트를 확인하면 다음과 같다

```python 
In:
#bincount는 음수를 받을 수 없어서 labels에 1을 더함.
#반환 값의 첫 번째 원소는 잡음 포인트의 수.
print(f"클러스터별 포인트 수: {np.bincount(labels+1)}")
```

```python 
Out:
클러스터별 포인트 수: [  32 2031]
```

```python 
noise = X_people[labels==-1]

fig, axes = plt.subplots(3, 9, subplot_kw={'xticks':(), 'yticks': ()}, 
                         figsize=(12, 4))
for image, ax in zip(noise, axes.ravel()):
  ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
```

![](./Figure/3_5_4_3.JPG)

손이 얼굴 앞을 가린다던지, 잔에 든 것을 마시는 사람 사진 등 특이한 것을 찾아내는 이런 종류의 분석을 **이상치 검출(Outlier dectection)** 이라고 한다. 



```python 
In:
for eps in [2*i + 1 for i in range(7)]:
  print(f"\neps={eps}")
  dbscan = DBSCAN(eps=eps, min_samples=3)
  labels = dbscan.fit_predict(X_pca)
  print(f"클러스터의 수: {len(np.unique(labels))}")
  print(f"클러스터 크기: {np.bincount(labels + 1)}")
```

```python 
Out:
eps=1
클러스터의 수: 1
클러스터 크기: [2063]

eps=3
클러스터의 수: 1
클러스터 크기: [2063]

eps=5
클러스터의 수: 1
클러스터 크기: [2063]

eps=7
클러스터의 수: 14
클러스터 크기: [2004    3   14    7    4    3    3    4    4    3    3    5    3    3]

eps=9
클러스터의 수: 4
클러스터 크기: [1307  750    3    3]

eps=11
클러스터의 수: 2
클러스터 크기: [ 413 1650]

eps=13
클러스터의 수: 2
클러스터 크기: [ 120 1943]
```

```python 
dbscan = DBSCAN(min_samples=3, eps=7)
labels = dbscan.fit_predict(X_pca)

for cluster in range(max(labels) + 1):
    mask = labels == cluster
    n_images =  np.sum(mask)
    fig, axes = plt.subplots(1, n_images, figsize=(n_images*1.5, 4),
                             subplot_kw={'xticks': (), 'yticks': ()})
    i = 0
    for image, label, ax in zip(X_people[mask], y_people[mask], axes):
        ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
        ax.set_title(people.target_names[label].split()[-1])
        i += 1
    for j in range(len(axes) - i):
        axes[j+i].imshow(np.array([[1]*65]*87), vmin=0, vmax=1)
        axes[j+i].axis('off')
```

![](./Figure/3_5_4_4.JPG)



##### k-평균으로 얼굴 데이터셋 분석하기

DBSCAN에서는 하나의 큰 클러스터 외에는 만들 수 없다는 것을 바로 앞 예제에서 확인했다. 이에 비해 병합 군집과 *K*-평균은 비슷한 크기의 클러스터들을 만들 수 있지만 클러스터 개수를 지정해야만 한다. 

```python 
In:
km = KMeans(n_clusters=10, random_state=0)
labels_km = km.fit_predict(X_pca)
print(f"k-평균 클러스터 크기: {np.bincount(labels_km)}")
```

```python 
Out:
k-평균 클러스터 크기: [282 226 105 268 151 324 202 208 208  89]
```



*k*-평균의 클러스터 중심을 pca.inverse_transform을 사용해 원본 공간으로 되돌린 후 시각화하면 다음과 같다.

```python 
fig, axes = plt.subplots(2, 5, subplot_kw={'xticks': (), 'yticks': ()},
                         figsize=(12, 4))
for center, ax in zip(km.cluster_centers_, axes.ravel()):
  ax.imshow(pca.inverse_transform(center).reshape(image_shape),
            vmin=0, vmax=1)
```

![](./Figure/3_5_4_5.JPG)



다음은 *k*-평균으로 찾은 클러스터의 중심, 중심에서 가장 가까운 5개 포인트, 클러스터 중심에서 가장 먼 5개의 포인트이다.

![](./Figure/3_5_4_6.JPG)

*k*-평균이 잡음 포인트 개념이 없기 때문에 클러스터에서 멀리 떨어진 포인트들은 중심 포인트와 관련이 별로 없어 보인다. 클러스터 수를 늘리면 알고리즘이 미세한 차이를 더 찾을 수 있지만 너무 많이 늘리면 직접 조사하는 것이 더 어려워진다.



##### 병합 군집으로 얼굴 데이터셋 분석하기

```python 
In:
agglomerative = AgglomerativeClustering(n_clusters=10)
labels_agg = agglomerative.fit_predict(X_pca)

print(f"병합 군집의 클러스터 크기: {np.bincount(labels_agg)}")
```

```python 
Out:
병합 군집의 클러스터 크기: [169 660 144 329 217  85  18 261  31 149]
```

*k*-평균보다는 크기가 고르지 않지만 DBSCAN보다는 훨씬 비슷한 크기이다.



ARI 점수를 이용해 병합 군집과 *K*-평균으로 만든 두 데이터가 비슷한지 측정한 결과는 다음과 같다.

```python 
In:
from sklearn.metrics.cluster import adjusted_rand_score

print(f"ARI: {adjusted_rand_score(labels_agg, labels_km)}")
```

```python 
Out:
ARI: 0.10292906782941566
```



덴드로그램을 그리면 다음과 같다. 단, truncate_mode='level'과 p=7을 통해 트리의 최대 깊이를 7로 제한했다.

```python 
from scipy.cluster.hierarchy import dendrogram, ward

linkage_array = ward(X_pca)
plt.figure(figsize=(20, 5))
dendrogram(linkage_array, p=7, truncate_mode='level', no_labels=True)

plt.xlabel("샘플 번호")
plt.ylabel("클러스터 거리")
ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [36, 36], '--', c='k')
```

![](./Figure/3_5_4_7.JPG)



10개의 클러스터를 그림으로 나타내면 다음과 같다. 병합 군집에서는 클러스터 중심이라는 개념이 없으므로 그냥 클러스터에 속한 몇 개의 포인트를 나타냈다. 첫 번째 이미지 왼쪽에는 각 클러스터에 속한 데이터 포인트의 수를 나타내었다.

```python 
n_clusters = 10
for cluster in range(n_clusters):
    mask = labels_agg == cluster
    fig, axes = plt.subplots(1, 10, subplot_kw={'xticks': (), 'yticks': ()},
                             figsize=(15, 8))
    axes[0].set_ylabel(np.sum(mask))
    for image, label, asdf, ax in zip(X_people[mask], y_people[mask],
                                      labels_agg[mask], axes):
        ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
        ax.set_title(people.target_names[label].split()[-1],
                     fontdict={'fontsize': 9})
```

![](./Figure/3_5_4_8.JPG)



클러스터의 수를 늘려 그 중 몇 가지 흥미로운 클러스터를 골라 시각화 하면 다음과 같다.

```python 
In:
agglomerative = AgglomerativeClustering(n_clusters=40)
labels_agg = agglomerative.fit_predict(X_pca)
print(f"병합 군집의 클러스터 크기: {np.bincount(labels_agg)}")
```

```python 
Out:
병합 군집의 클러스터 크기: [ 43 120 100 194  56  58 127  22   6  37  65  49  84  18 168  44  47  31
  78  30 166  20  57  14  11  29  23   5   8  84  67  30  57  16  22  12
  29   2  26   8]
```

```python 
n_clusters = 40
for cluster in [10, 13, 19, 22, 36]: #흥미로운 클러스터 
    mask = labels_agg == cluster
    fig, axes = plt.subplots(1, 15, subplot_kw={'xticks': (), 'yticks': ()},
                             figsize=(15, 8))
    cluster_size = np.sum(mask)
    axes[0].set_ylabel(f"#{cluster}: {cluster_size}")
    for image, label, asdf, ax in zip(X_people[mask], y_people[mask],
                                      labels_agg[mask], axes):
        ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
        ax.set_title(people.target_names[label].split()[-1],
                     fontdict={'fontsize': 9})
    for i in range(cluster_size, 15):
      axes[i].set_visible(False)
```

![](./Figure/3_5_4_9.JPG)

군집 알고리즘이 '대머리', '옆모습', '웃는 여성' 등을 뽑아낸 것으로 보인다.



##### 3.5.5 군집 알고리즘 요약

- 군집 알고리즘을 적용하고 평가하는 것은 매우 정성적인 분석과정이고 EDA 단계에서 크게 도움이 될 수 있다.
- *K*-평균, DBSCAN, 병합 군집 모두 실제 대량의 데이터 셋에 사용할 수 있고 비교적 쉽게 이해할 수 있으며 여러 개의 클러스터로 군집을 만들 수 있다.
- *K*-평균은 클러스터 중심을 사용해 클러스터를 구분한다. 각 데이터 포인트를 클러스터의 중심으로 대표할 수 있기 때문에 분해 방법(PCA, NMF 등)으로 볼 수 있다.
- DBSCAN은 클러스터에 할당되지 않은 잡음 포인트를 인식 할 수 있다. 클러스터의 개수를 자동으로 결정한다. two_moons와 같이 복잡한 클러스터의 모양을 인식할 수 있다. 크기가 많이 다른 클러스터를 만들어 내곤 한다.
- 병합 군집은 전체 데이터의 분할 계층도를 만들어 주고 덴드로그램을 이용해 확인 가능 하다.



### 3.6 요약 및 정리

데이터를 올바르게 표현하는 것은 지도학습과 비지도 학습을 잘 적용하기 위해 필수적이며, 전처리와 분해 방법은 데이터 준비 단계에서 아주 중요한 부분이다.

분해, 매니폴드 학습, 군집은 주어진 데이터에 대한 이해를 높이기 위한 필수 도구이며, 레이블 정보가 없을 때 데이터를 분석할 수 있는 유일한 방법이다. 

지도학습에서도 데이터 탐색 도구는 데이터의 특성을 잘 이해하는데 중요하며 비지도 학습의 성과를 정량화하기 어렵지만 데이터에 대한 통찰을 얻기 위해 이런 도구들을 사용할 필요가 있다.

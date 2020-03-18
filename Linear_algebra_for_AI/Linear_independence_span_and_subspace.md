# Linear independence, span, and subspace

## Recall: Linear System

![](C:\Users\choidaek\Desktop\Linear_independence_span_and_subspace1.JPG)

위와 같은 선형 시스템은 다음과 같은 벡터들의 선형 결합으로 이루어진 벡터 방정식으로 표현할 수 있다.

![](C:\Users\choidaek\Desktop\Linear_independence_span_and_subspace2.JPG)



##  Uniqueness of Solution for Ax = b

![](C:\Users\choidaek\Desktop\Linear_independence_span_and_subspace2.JPG)

이런 벡터 방정식의 해 [x1, x2, x3]는 선형 결합의 결과인 b가 Span{a1, a2, a3}에 포함될 때 존재하게 된다. 해가 존재한다는 것은 유일하게 하나 존재하거나 무수히 많거나, 두 경우가 있다. 유일하게 하나 존재하는 경우는 기저 벡터 a1, a2, a3가 선형 독립일 때이고, 무수히 많을 때는 a1, a2, a3가 선형 의존적일 때이다. 

![](C:\Users\choidaek\Desktop\Linear_independence_span_and_subspace3.JPG)



## Linear Independence

(Practical Definition) n차원 공간 상의 벡터의 집합 v1, ..., vp에 대하여 만약에 vj가 v1, ..., vj-1(j = 1, ..., p)의 선형 결합으로 표현된다면 , 즉, vj가 Span{v1, v2, ..., vj-1} (j = 1, ..., p)에 속한다면, 이런 vj가 적어도 한 개이상 발견될 경우 v1, ..., vp는 선형 종속이다. 만약에 이런 vj가 하나도 발견되지 않는다면 v1, ..., vp는 선형 독립이다. 

- 방정식의 갯수가 계수(특징, 가중치)의 갯수보다 적을 경우, 이때의 벡터들은 무조건 선형 종속이다. 그 이유는 방정식의 갯수만큼의 차원을 커버하는데 필요한 기저 벡터의 숫자는 계수(특징, 가중치)의 갯수와 같아야하는데 이미 충분히 커버하고 남기 때문에 남은 벡터들은 무조건 기저 벡터의 Span에 속할수 밖에 없게 된다.

  ![](C:\Users\choidaek\Desktop\Linear_independence_span_and_subspace4.JPG)

- 방정식의 갯수가 계수(특징, 가중치)의 갯수보다 적을 경우, 이때의 벡터들은 경우에 따라 선형 종속이 될 수도 있고 선형 독립이 될 수도 있다. 만약에 벡터들이 서로 독립이라면 선형 독립이 되고 어떤 벡터가 나머지 벡터의 선형 결합으로 표현된다면 이 벡터들은 선형 종속이 된다. 

  ![](C:\Users\choidaek\Desktop\Linear_independence_span_and_subspace5.JPG)



## Linear Independence

(Formal Definition) 어떤 Homogeneous 방정식 x1v1 + x2v2 + ... + xpvp = 0에 대하여(Homogeneous 방정식으로 판별하는 이유는 영 벡터는 모든 선형 결합의 Span에 포함되기 때문에 Ax = b에서 b 벡터가 재료 벡터들의 Span에 포함되는지 안되는지 상관할 필요가 없기 때문)

이 때의 해가 다음과 같은 Trivial solution 하나 밖에 없다면

![](C:\Users\choidaek\Desktop\Linear_independence_span_and_subspace7.JPG)

 v1, ..., vp는 해가 Trivial solution 하나 밖에 없기 때문에 선형 독립이다. 

만약에 해가, 적어도 하나의 xi가 0이 아닌, Nontrivial solution이 존재한다면 v1, ..., vp는 선형 종속이다.

![](C:\Users\choidaek\Desktop\Linear_independence_span_and_subspace6.JPG)

여기서 [1, 2] 벡터의 계수가 0이 아니고 3이라서 영 벡터로부터 [3, 6]만큼 떨어져 있는 상태라면 이 상태를 다시 영 벡터로 되돌리기 위해서는 [1, 2] 벡터(예를 들어 순서상 2번째 벡터라고 가정)를 제외한 나머지 1, 3, 4, 5번째 벡터의 계수를 잘 조정하여 [-3, -6]을 만들어 영 벡터를 만들어야 한다. 이 때 이 재료 벡터들은 선형 종속이다. 



## Two Definitions are Equivalent

v1, ..., vp가 선형 종속일 때, 어떤 Nontrivial solution이 하나 존재한다. 이때 계수 xj가 0이 아니게 하는 선형 결합에서의 마지막 인덱스 j가 있다. 그래서 

![](C:\Users\choidaek\Desktop\Linear_independence_span_and_subspace8.JPG)

쓸수 있을때, 이때의 vj는 xj라는 계수를 양변으로 나눠서 계수가 0이 아닌 벡터들의 선형 결합으로 나타낼수 있다. 

![](C:\Users\choidaek\Desktop\Linear_independence_span_and_subspace9.JPG)

![](C:\Users\choidaek\Desktop\Linear_independence_span_and_subspace10.JPG)



## Geometric Understanding of Linear Dependence

![](C:\Users\choidaek\Desktop\Linear_independence_span_and_subspace11.JPG)

위의 평면을 Span{v1, v2}라고 가정한다면 v3 벡터가 Span{v1, v2}에 포함되는지에 따라서 v1, v2과 선형 독립인지 선형 종속인지가 결정된다. 



## Linear Dependence

선형 종속적인 벡터는 더이상 Span을 확장시키지 않는다. 즉 v3 ∈ Span{v1, v2}에 대하여 Span{v1, v2} = Span{v1, v2, v3}이다. 왜냐하면 v3 = d1v1 + d2v2라고 했을 때, c1v1 + c2v2 + c3v3 = (c1 + d1)v1 + (c1 + d1)v2이고 이것은 곧 v1, v2의 선형 결합이기 때문이다.



## Linear Dependence and Linear System Solution

선형 독립적인 벡터들은 여러개의 선형 결합을 만들어 낼 수 있다. 

![](C:\Users\choidaek\Desktop\Linear_independence_span_and_subspace12.JPG)

![](C:\Users\choidaek\Desktop\Linear_independence_span_and_subspace13.JPG)
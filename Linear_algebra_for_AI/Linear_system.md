# Linear system

## Linear Equation

하나의 선형 방정식은 변수 {x1, ..., xn}에 대하여 다음과 같이 쓸 수 있는 방정식이다.

![](C:\Users\choidaek\Desktop\Linear_system1.JPG)

여기서 b와 {a1, ..., an}은 미리 알려진 실수이거나 복소수이다. 위의 식은 내적을 통해 다음과 같이도 표현할 수 있다. 

​                               ![](C:\Users\choidaek\Desktop\Linear_system2.JPG)



## Linear System: Set of Equations

하나의 선형 방정식들의 시스템(선형 시스템, 연립 방정식)은 같은 변수를 공유하는 하나 이상의 선형 방정식의 모음이다.



## Linear System Example

![](C:\Users\choidaek\Desktop\Linear_system3.JPG)

예를 들어 어떤 사람의 몸무게, 키, 흡연 여부의 데이터로 그 사람의 수명을 예측하는 문제가 있다고 가정할 때 우리는 선형 시스템을 다음과 같이 세울 수 있다. 

![](C:\Users\choidaek\Desktop\Linear_system4.JPG)

우리가 만약 적당한 x1, x2, x3 값을 찾는다면 새로운 사람의 데이터가 입력으로 들어왔을 때 선형 회귀의 문제로 그 사람의 기대수명을 구할 수 있게 된다. 



## Linear System Example

이런 선형 시스템은 행렬을 사용하여 압축된 표현으로 나타낼 수 있다. 

![](C:\Users\choidaek\Desktop\Linear_system4.JPG)

위의 시스템에서 미리 알려진 각 사람의 데이터를 하나로 모아 행렬로 나타내고 기대 수명을 모아 하나의 벡터로 나타내며 x1, x2, x3를 모아 하나의 벡터로 나타낸다. 

![](C:\Users\choidaek\Desktop\Linear_system5.JPG)



## From Multiple Equations to Single Matrix Equation

![](C:\Users\choidaek\Desktop\Linear_system6.JPG)

이 시스템은 계수행렬(Coefficient matrix) A의 각 행과 벡터x의 내적으로 나타낼수도 있고 행렬과 벡터의 Multiplication으로 나타낼수도 있다. 후자의 방법을 행렬 방정식이라고 한다. 



## Identity Matrix

항등행렬은 행렬의 각 원소 중에서 왼상단부터 오른하단까지의 대각선의 값은 모두 1이고 나머지는 0인 정사각행렬을 의미한다. 

![](C:\Users\choidaek\Desktop\Linear_system7.JPG)

n차원의 항등 행렬은 n차원의 벡터나 n x m 차원의 행렬의 모든 값을 Multiplication 연산 후에도 보존한다. 단 행렬의 경우, 피연산자 중 하나인 n x m차원의 행렬의 앞에 있는가 뒤에 있는가에 따라 항등 행렬의 차원이 달라진다. 

![](C:\Users\choidaek\Desktop\Linear_system8.JPG)



## Inverse Matrix

n차원의 정사각행렬 A에 대하여 이것의 역행렬의 정의는 다음과 같다. 

![](C:\Users\choidaek\Desktop\Linear_system9.JPG)

즉, 행렬 A의 양쪽에 각각 역행렬을 Multiplication 했을 때, n차원의 항등행렬이 나온다. 

행렬 A가 주어졌을 때, 역행렬 A를 구하는 방법은 다음과 같다. 

![](C:\Users\choidaek\Desktop\Linear_system10.JPG)

여기서 ad-bc를 2차원 행렬의 행렬식이라고 하고, det(ad-bc)라고 표기한다. 



## Solving Linear System via Inverse Matrix

역행렬과 항등행렬이 정의되어 있다면 Ax = b의 행렬 방정식의 해를 다음과 같이 구할 수 있다. 

![](C:\Users\choidaek\Desktop\Linear_system11.JPG)



## Solving Linear System via Inverse Matrix

![](C:\Users\choidaek\Desktop\Linear_system12.JPG)



## Solving Linear System via Inverse Matrix

여기서 구한 해(Solution)는 선형 시스템의 계수가 된다. 

![](C:\Users\choidaek\Desktop\Linear_system13.JPG)



## Non-Invertible Matrix A for Ax = b

행렬 A의 역행렬이 존재한다면 해는 다음과 같이 유일하게 하나 존재한다.

![](C:\Users\choidaek\Desktop\Linear_system14.JPG)

만약에 x2라는 해가 있다고 한다면 Ax2 = b를 만족해야 한다. 여기서 양변에 A의 역행렬을 곱해주면 x2 = A^-1b = x가 되기 때문에 x2 = x가 된다. 

det(ad - bc) = 0일때 역행렬이 존재하지 않는데, ad = bc에서 a : b = c : d 의 관계 즉, [[1, 2], [3, 4]]과 같은 행렬은 역행렬이 존재하지 않게 된다. 



## Does a Matrix Have an Inverse Matrix?

3차원 이상의 정사각행렬의 행렬식을 계산하는 방법은 아래 참조.

[참조 1]: https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/video-lectures/lecture-18-properties-of-determinants/
[참조 2]: https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/video-lectures/lecture-19-determinant-formulas-and-cofactors/



## Inverse Matrix Larger than 2 x 2

3차원 이상의 정사각행렬이 역행렬이 존재한다면 이를 구하는 방법은 

- 공식 활용
- 가우스 소거법(Gaussian Elimination) 



## Non-Invertible Matrix A for Ax = b

만약 A의 역행렬이 존재하지 않는다면 Ax = b의 해는 없거나 무수히 많다.

![](C:\Users\choidaek\Desktop\Linear_system15.JPG)



## Rectangular Matrix A in Ax = b

역행렬이 존재하지 않는 직사각행렬 A ∈ R^(m x n), m과 n이 다른 경우에,

m을 방정식의 숫자(위의 예제에서는 사람 데이터의 숫자), n을 변수(계수)의 숫자(특성의 개수)라고 했을 때,

- m < n인 경우의 해눈 무수히 많다(Under-determined system). 이 경우에 머신러닝에서는 특성 하나에 좌지우지 되는 리스크를 최대한 줄이는, 계수들의 범위를 선택하는 정규화(Regularization)를 수행한다.
- m > n인 경우의 해는 없다(Over-determined system). 이 경우에 머신러닝에서는 최대한 답에 근사적인 해를 구한다. 
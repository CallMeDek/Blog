# Span and subspace

## Span and Subspace

부분공간 H는 선형결합에 닫혀 있는 n차원 공간의 부분집합으로서 정의한다. 즉, 어떤 두 벡터가 H에 속해있고(u1,u2 ∈ H), 어떤 두 개의 스칼라 값 c, d가 있으며 스칼라에 의한 곱 연산과 벡터들간의 합 연산이 정의되어 있을 때 cu1+du2도 H에 속한다면 부분공간이라고 할 수 있다. 이 개념은 Span의 개념과 같다. 

![](C:\Users\LAB\Desktop\Span_and_subspace1.JPG)



## Basis of a Subspace

부분공간 H의 기저벡터는 다음과 같은 성질을 만족하는 벡터들의 집합이다. 

- 주어진 부분공간 H를 완전하게 Span해야한다.
- 벡터들이 서로 선형 독립이어야 한다.

예를 들어서 H = Span{v1, v2, v3}일 때, Span{v1, v2}가 어떤 평면을 이루고, v3 = 2v1 + 3v2  ∈ Span{v1, v2} 일때, {v1, v2}는 H의 기저 벡터이지만 {v1}, {v2}, {v1, v2, v3}는 기저 벡터라고 할 수 없다.



## Non-Uniqueness of Basis

어떤 부분공간 H를 완전히 Span하고 상호 선형 독립적인 벡터들의 집합은 유일하지는 않다. 아래 그림과 같이 부분공간 H의 한 점을 표현하는 방법은 여러개가 있을 수 있다(기저 벡터가 여러개 있다).

 ![](C:\Users\LAB\Desktop\Span_and_subspace2.JPG)



## Dimension of Subspace

부분공간 H에 대하여 각기 다른 기저벡터가 존재할 수 있지만 H를 구성하는 기저벡터들의 숫자는 유일하다. 이를 H의 차원이라고 하고 dim H라고 표기한다. 

![](C:\Users\LAB\Desktop\Span_and_subspace3.JPG)

여러개의 기저 벡터가 존재할 수 있지만 Standard basis vector(길이가 1이고 서로 수직인 벡터)를 사용하게 되면 이 기저 벡터들의 계수는 도달하고자 하는 벡터의 좌표값이 된다. 



## Column Space of Matrix

어떤 행렬 A의 열 공간(Column space)는 행렬 A의 열에 의해서 Span되는 부분공간을 의미한다.  Col A 라고  표기한다. 

![](C:\Users\LAB\Desktop\Span_and_subspace4.JPG)



## Matrix with Linearly Dependent Columns

아래와 같이 주어진 행렬 A에 대해서

![](C:\Users\LAB\Desktop\Span_and_subspace5.JPG)

3번째 열은 1번째와 2번째 열의 선형결합에 의해서 표현된다. 따라서 1번째와 2번째 열의 Span이나 1~3번째 열의 Span이나 동일하게 된다. 

![](C:\Users\LAB\Desktop\Span_and_subspace6.JPG)



## Rank of Matrix

어떤 행렬 A의 랭크는 A의 열 공간의 차원이고 rank A로 표기한다.

![](C:\Users\LAB\Desktop\Span_and_subspace7.JPG)
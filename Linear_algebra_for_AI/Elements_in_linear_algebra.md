# Elements In Linear Algebra

### Scalar, Vector, and Matrix

- Scalar: 실수 집합을 R이라고 했을 때 이 실수 집합의 원소 중 하나. 특히 하나의 스칼라 값은 R의 지수부분이 나타내는 차원이 1일 때의 집합의 원소.  소문자로 나타낸다.  

  ![](C:\Users\choidaek\Desktop\Elements_in_linear_algebra_1.JPG)

- Vector:  순서가 있는 숫자들의 리스트.  순서가 있기 때문에 숫자들의 순서가 바뀌면 다른 벡터가 된다. R의 지수부분이 나타내는 차원이 n이라고 했을 때, 이 차원 공간에 속하는 벡터는 n개의 숫자를 가진다. 볼드체의 소문자로 나타낸다. 순서가 없는 숫자들의 모음을 집합이라고 한다. 

  ![](C:\Users\choidaek\Desktop\Elements_in_linear_algebra_2.JPG)

- Matrix: 숫자들의 2차원 배열(여기서 차원은 숫자들의 갯수가 아닌 표기상의 차원을 의미). 마찬가지로 숫자들의 위치가 바뀌면 다른 행렬이 된다. R의 지수부분이 나타내는 차원이 m x n 이라고 했을 때 총 m x n개의 숫자를 가지며 m을 행렬의 행(Row)이라고 하고 n을 행렬의 열(Column)이라고 한다.  대문자로 나타낸다. 

  ![](C:\Users\choidaek\Desktop\Elements_in_linear_algebra_3.JPG)



### Column Vector and Row Vector

- 하나의 행으로 나타내는 벡터를 행 벡터(Row vector)라고 한다. 행 벡터는 크기가 1 x n인 행렬이라고 말할 수 있다. 보통 표기로 나타낼때는 열 벡터의 Transpose 연산으로 표시한다. 

  ![](C:\Users\choidaek\Desktop\Elements_in_linear_algebra_5.JPG)

- 하나의 열로 나타내는 벡터를 열 벡터(Column vector)라고 한다. 열 벡터는 크기가 n x 1인 행렬이라고 말할 수 있다. 보통 선형대수학에서 벡터를 나타낼 때는 열벡터로 나타낸다.  

  ![](C:\Users\choidaek\Desktop\Elements_in_linear_algebra_4.JPG)



### Matrix Notations

- Square matrix: 하나의 정사각행렬은 행의 숫자와 열의 숫자가 같다. 

  ![](C:\Users\choidaek\Desktop\Elements_in_linear_algebra_6.JPG)

- Rectangular matrix: 정사각행렬을 포함하고, 행의 숫자와 열의 숫자가 같지 않아도 괜찮은 행렬.

  ![](C:\Users\choidaek\Desktop\Elements_in_linear_algebra_7.JPG)

- Transpose:  어떤 행렬의 주 대각선을 기준으로 가로질러 숫자들을 복사한 행렬. 즉, 행렬의 각 행 벡터 혹은 열 벡터를 열 벡터 혹은 행 벡터로 바꾼다. 

  ![](C:\Users\choidaek\Desktop\Elements_in_linear_algebra_8.JPG)

- (i, j)-th component of A: 어떤 행렬 A의 (i, j)번째 요소라는 것은 행렬 A의 i번째 행과 j번째 열이 만나는 지점에 있는 숫자를 뜻한다. 

  ![](C:\Users\choidaek\Desktop\Elements_in_linear_algebra_9.JPG)

- i-th row vector of A: 어떤 행렬 A의 i번째 행 벡터는 행렬 A의 i번째 행에 있는 행 벡터를 의미한다.

  ![](C:\Users\choidaek\Desktop\Elements_in_linear_algebra_10.JPG)

- i-th column vector of A: 어떤 행렬 A의 i번째 열 벡터는 행렬 A의 i번째 열에 있는 열 벡터를 의미한다.

  ![](C:\Users\choidaek\Desktop\Elements_in_linear_algebra_11.JPG)



### Vector/Matrix Additions and Multiplications

- Element-wise addition: 어떤 크기가 정확히 같은 행렬 A와 B가 있을 때 두 행렬 A, B의 Element-wise addition은 두 행렬의 같은 자리의 (i, j)번째 요소를 더한다. 

  ![](C:\Users\choidaek\Desktop\Elements_in_linear_algebra_12.JPG)

- Scalar multiple: 어떤 실수 c가 있고 하나의 벡터 혹은 행렬이 있을 때, 벡터 혹은 행렬의 각 (i, j)번째 요소는 Scalar multiple에 의해 c만큼 multiply 될 수 있다. 특별히 1 x 1 행렬이 스칼라가 될 수 없는 이유는 벡터와 행렬의 경우, 스칼라에 의해 multiply 될 수 있지만 1 x 1행렬에 의해서 multiply 될 수는 없기 때문이다.

  ![](C:\Users\choidaek\Desktop\Elements_in_linear_algebra_13.JPG)

- Matrix-matrix multiplication: 어떤 두 행렬이 있을 때, 한 행의 열 벡터의 숫자와 다른 벡터의 행 벡터의 숫자가 같을 때 두 행렬의 곱을 정의할 수 있다.  개략적인 정의로 아래 예제의 세번째의 두 벡터 사이의 연산으로 1 x 1의 행렬이 만들어지는 연산을 내적(Inner product)라고 하고 네 번째의 두 벡터 사이의 연산으로 행렬이 만들어지는 연산을 외적(Outer product)이라고 한다.

  ![](C:\Users\choidaek\Desktop\Elements_in_linear_algebra_14.JPG)



### Matrix multiplication is NOT commutative

두 행렬 A, B의 multiplication 연산의 교환 법칙은 성립하지 않는다.

1. A ∈ R^(2x3), B ∈ R^(3x5) 일 때, AB는 가능하지만 BA는 B의 열 벡터의 숫자와 A의 행 벡터의 숫자가 다르므로 multiplication이 불가능하다. 

2. A ∈ R^(2x3), B ∈ R^(3x2) 일 때, AB ∈ R^(2x2) 지만 BA ∈ R^(3x3) 이므로 크기가 달라진다. 

3. A ∈ R^(2x2), B ∈ R^(2x2) 일 때, 마찬가지로 AB와 BA는 다르다. 

   ![](C:\Users\choidaek\Desktop\Elements_in_linear_algebra_15.JPG)



### Other properties

- A(B + C) = AB + AC : Distributive.
- A(BC) = (AB)C : Associative.
- (AB)^T = B^TA^T : Property of transpose.
- (AB)^-1 = B^-1A^-1 : Property of inverse.
# Linear combination, vector equation, Four views of matrix multiplication

## Linear Combinations

n차원의 벡터 p개 v1, v2, .., vp와 p개의 스칼라에 대하여 선형 결합은 다음과 같다.

![](C:\Users\choidaek\Desktop\Linear_combination_vector equation_Four_views_of_matrix_multiplication1.JPG)

여기서 p개의 스칼라 c1, ..., cp를 가중치 혹은 계수라고 한다. 선형 결합에서 가중치는 0을 포함한 실수가 될 수 있다. 



## From Matrix Equation to Vector Equation

선형 시스템의 행렬 방정식은 다음과 같이 나타낼 수 있다.

![](C:\Users\choidaek\Desktop\Linear_combination_vector equation_Four_views_of_matrix_multiplication2.JPG)

하나의 행렬 방정식은 백터 방정식으로 변형 될 수 있다.

![](C:\Users\choidaek\Desktop\Linear_combination_vector equation_Four_views_of_matrix_multiplication3.JPG)

![](C:\Users\choidaek\Desktop\Linear_combination_vector equation_Four_views_of_matrix_multiplication4.JPG)



## Existence of Solution for Ax = b

![](C:\Users\choidaek\Desktop\Linear_combination_vector equation_Four_views_of_matrix_multiplication3.JPG)

벡터 방정식에 대하여 Ax = b의 해가 존재하는 경우는?



## Span

n차원에 속하는 주어진 p개의 벡터 v1, ..., vp에 대하여 v1, ..., vp의 선형 결합으로 나타낼 수 있는 모든 집합을 Span{v1, ..., vp}으로 정의한다.  그러므로 Span{v1, ..., vp}는 임의의 스칼라 c1, ..., cp로 다음과 같이 나타낼 수 있다. 

![](C:\Users\choidaek\Desktop\Linear_combination_vector equation_Four_views_of_matrix_multiplication1.JPG)

Span{v1, ..., vp}는 v1, ..., vp에 의해 spanned(generated)된 R^n의 부분집합이라고도 한다. 



## Geometric Description of Span

![](C:\Users\choidaek\Desktop\Linear_combination_vector equation_Four_views_of_matrix_multiplication5.JPG)

만약에 3차원 공간상에 상호 독립적인 벡터 2개가 있다면 Span{v1, v2}는 v1, v2, 원점을 포함하는 R^3 상에서의 평면이다.  즉 Span{v1, v2}의 모든 점은 원점, v1, v2와 v1과 v2의 선형결합으로 나타낼 수 있게 된다. 



## Geometric Interpretation of Vector Equation

![](C:\Users\choidaek\Desktop\Linear_combination_vector equation_Four_views_of_matrix_multiplication3.JPG)

따라서 위의 기하학적인 해석에 의거하면 결국 주어진 벡터 a1, a2, a3와 스칼라 x1, x2, x3의 선형 결합이 어떤 벡터 b라고 했을 때의 스칼라 x1, x2, x3를 찾는 것은 Span{a1, a2, a3} 위에 있는 벡터 b를 만족시키는 x1, x2, x3를 찾는것과 같다. 벡터 b가 Span{a1, a2, a3} 위에 있지 않다면 해가 존재하지 않는다. 



## Matrix Multiplications as Linear Combinations of Vectors

![](C:\Users\choidaek\Desktop\Linear_combination_vector equation_Four_views_of_matrix_multiplication6.JPG)

우리는 앞서 행렬간의 곱을 왼쪽 행렬의 행 벡터와 오른쪽 행렬의 열 벡터의 내적으로 표현한 바 있다. 그런데 벡터 방정식의 개념을 이용하면 행렬과 벡터의 곱을 행렬의 열 벡터들과 벡터의 선형결합으로 볼 수 있다.

![](C:\Users\choidaek\Desktop\Linear_combination_vector equation_Four_views_of_matrix_multiplication7.JPG)



## Matrix Multiplications as Column Combinations

![](C:\Users\choidaek\Desktop\Linear_combination_vector equation_Four_views_of_matrix_multiplication8.JPG)

왼쪽 행렬의 열 벡터들을 기저(base) 혹은 재료 벡터로 보고 오른쪽 벡터를 계수 혹은 가중치로 봤을 때 앞에서 설명한 것처럼 행렬과 벡터의 곱은 기저 벡터들과 가중치의 선형결합으로 나타낼 수 있다. 

![](C:\Users\choidaek\Desktop\Linear_combination_vector equation_Four_views_of_matrix_multiplication9.JPG)

이 개념을 행렬간의 곱으로 확장시키면 위와 같다. 왼쪽 행렬의 열 벡터들을 기저 벡터로 보고 오른쪽 행렬의 각 열 벡터들을 가중치로 본다. 이를 벡터 방정식의 개념으로 선형 결합으로 나타낸다면 단지 두 개의 선형 결합이 모아져 있는 하나의 행렬의 개념으로 볼 수 있다.  이때 각 선형 결합은 서로 영향을 주지 않고, 기저 벡터들의 Span에 포함된다. 



## Matrix Multiplications as Row Combinations

(Ax)^T = x^TA^T 이므로 행 벡터들의 선형 결합으로 나타내는 곱은 왼쪽의 행 벡터가 계수, 오른쪽의 행 벡터들이 기저 벡터가 된다. 

![](C:\Users\choidaek\Desktop\Linear_combination_vector equation_Four_views_of_matrix_multiplication10.JPG)



## Matrix Multiplications as Sum of (Rank-1) Outer Products

![](C:\Users\choidaek\Desktop\Linear_combination_vector equation_Four_views_of_matrix_multiplication11.JPG)

Rank-1 외적이나 Rank-1 외적의 합계는 위와 같다. 여기서 Rank-1인 이유는 벡터간의 외적에서 왼쪽 벡터의 열 벡터의 수가 1개이기 때문이다(Rank는 어떤 행렬의 서로 독립적인 열 벡터들의 갯수, 즉 Dimension of Column space of a Matrix A이다).



## Matrix Multiplications as Sum of (Rank-1) Outer Products

Sum of (Rank-1) 외적은 머신러닝 분야에서 많이 쓰인다. 예를 들어서 다변량 가우시안 공간의 변수 간의 공분산 행렬이라든지 Style transfer의 그람 행렬등에 쓰이고 word2vec이나 PCA, SVD 등에도 쓰인다. 

![](C:\Users\choidaek\Desktop\Linear_combination_vector equation_Four_views_of_matrix_multiplication12.JPG)

100 by 50의 행렬을 100 x 1 벡터와 1 x 50 벡터의 Rank-1 외적 10개의 합으로 나타내면 5000개의 원소가 1500개로 줄어든다. 단 원래의 정보를 온전히 표현하지는 못하고 정보의 손실을 감수하고 근사적으로 표현한다. 
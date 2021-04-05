# SEARCHING FOR ACTIVATION FUNCTIONS

Prajit Ramachandran, Barret Zoph, Quoc V. Le(Google Brain)



## ABSTRACT

딥러닝 네트워크에서 어떤 Activation 함수를 사용할 것인가를 선택하는 문제는 훈련이나 훈련이 끝나고 작업의 성능에 크게 영향을 끼친다. 그동안 널리 쓰여왔던 함수는 Rectified Linear Unit이다. ReLU를 기반으로 나온 여러 연구들이 ReLU를 대체할만한 Activation 함수를 제안했으나 성능이 들쭉날쭉해서 ReLU를 대체할 수 없었다. 이 논문에서 저자들은 자동화 탐색 알고리즘으로 새로운 Activation 함수를 찾고자 했다. Exhaustive 탐색과 강화 학습 기반의 탐색을 조합해서 사용해서 저자들은 여러 새로운 Activation 함수를 발견했다. 저자들은 여기서 성능이 제일 좋은 함수로 여러 검증을 통해서 이러한 탐색 방법의 효율성을 확인했다. 저자들의 실험 결과에 따르면 Swish는 여러 데이터셋에서, 깊은 모델에서도 ReLU보다 성능이 더 좋다고 한다. 



## INTRODUCTION

보통 딥러닝 네트워크에서는 선형 변환 후에 Activation 함수가 붙는다. Activation 함수는 깊은 신경망 네트워크의 훈련을 성공시키는데 중요한 역할을 했다. 현재 가장 널리 쓰이는 Activation 함수는 ReLU이다. ReLU를 사용하면 Sigmoid나 Tanh 계열 함수를 사용하는 것보다 네트워크를 최적화 시킬때 더 쉬워지는데 그 이유는 ReLU 함수의 입력 값이 양수이면 그래디언트가 네트워크를 따라 흘러갈 수 있기 때문이다. 

많은 연구들이 ReLU를 대체하기 위한 Activation 함수를 개발하기 위해서 수행되었지만 결과적으로 ReLU를 대체하지는 못했다. ReLU는 구현하기 간편하고, 무엇보다도 안정적이라고 한다. 다른 함수들은 모델이나 데이터셋에 따라 성능 개선이 일정하지 않다. 

사람이 직접 디자인해서 Activation 함수를 찾는 것보다 자동화된 탐색 기법을 이용하여 찾는 것이 효율적이라는 것이 밝혀졌다. 예를 들어서 강화 학습 기반의 탐색으로 반복적으로 사용이 가능한 컨볼루션 Cell를 찾아내서 ImageNet Classification을 수행한 결과 사람이 직접 디자인한 모델보다 성능이 더 좋았다.

이 연구에서 저자들은 자동화된 탐색 기법을 이용하여 새로운 Activation 함수를 찾고자 했다. 저자들은 스칼라 값을 입력으로 받고 스칼라 값을 출력하는 함수를 찾기로 했다. 왜냐하면 스칼라 함수를 찾으면 네트워크 아키텍처를 변경할 필요 없이 ReLU를 대체할 수 있기 때문이다. 저자들은 여기에 Exhaustive 탐색과 강화 학습 기반의 탐색의 조합하여 사용했다. 저자들은 탐색 기법으로 스칼라 Activation 함수를 찾는 것의 효율성을 입증하기 위해서 가장 성능이 좋은 함수로 검증을 실시했다. Swish는 다음과 같다. 

![](./Figure/Searching_for_Activation_Functions1.png)

여기서 β는 상수일수도 있고 학습이 가능한 파라미터일수도 있다. 
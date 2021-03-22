# Densely Connected Convolutional Networks

Gao Huang(Cornell University), Zhuang Liu(Tsinghua University), Laurens van der Maaten(Facebook AI Research), Kilian Q. Weinberger(Cornell University)



## Abstract

상식적으로 입력에 가까운 계층과 출력에 가까운 계층 사이에 Shortcut이 있다면 컨볼루션 네트워크는 좀 더 깊고 정확하며 효율적으로 훈련시킬 수 있다. 저자들은 이런 상식을 포함시켜서 Dense Convolutional Network(DenseNet)을 발표했는데 역전파시에 각 계층을 모든 다른 계층에 연결하는 구조를 띄고 있다. 전통적인 컨볼루션 계층이 L 계층 사이에 L개의 연결 경로를 가진다면 DenseNet에서는 L(L+1)/2의 연결 경로를 가진다. 각 계층에 대해서는 모든 선행 계층들의 Feature map들은 다음 계층의 입력으로 쓰이고 이 계층의 Feature map들은 모든 후행 계층들의 입력으로 쓰인다. DenseNet은 몇가지 강력한 이점을 가지고 있다. 그래디언트가 역전파 중에 사라지는 문제를 완화하고 Feature 전파력를 강화시키며 Feature를 다시 사용함으로서 전체 파라미터 수를 줄인다. 저자들은 저자들의 아키텍처를 여러 벤치마크에서 검증했다(CIFAR-10, CIFAR-100, SVHN, ImageNet). 저자들에 의하면 DenseNet으로 필요한 연산량을 줄이고 높은 성능으로 모델을 개선했다고 한다. 



## Introduction

CNN이 점점 깊어지면서 새로운 문제가 발생했는데 입력 데이터에 대한 정보나 많은 계층을 거쳐가는 그래디언트가 사라질 수 있다는 것이다. 많은 연구들이 이를 해결하려고 노력했다. ResNet이나 Highway Network는 Identity connection을 통해서 한 계층에서 다음으로 Signal을 우회해서 보낸다. Stochastic depth는 훈련 과정 중에 ResNet에서 랜덤하게 계층을 드롭해서 정보와 그래디언트를 잘 흘려보낼 수 있게 한다. FractalNet은 반복적으로 몇 개의, 다른 수의 컨볼루션 블럭을 가진 병렬 계층 열을 조합하는 방식으로 네트워크에서 많은 Shortcut을 유지한다. 비록 이런 접근법들이 네트워크 토폴로지나 훈련 방법에 따라 다양하긴 하지만 공통적인 특징을 보인다. 초기 계층에서 후기 계층으로의 Shortcut을 만든다는 점이다. 저자들은 다음의 아이디어를 반영한 아키텍처를 제안했다. 네트워크의 각 계층 사이의 정보 흐름을 최대화 하기 위해서 모든 계층을(Feature map 크기가 맞는 계층끼리) 연결한다. 각 계층은 추가적인 입력 데이터를 모든 선행 계층으로부터 받고 그 계층의 모든 Feature map을 모든 후행 계층에 전달한다. 

![](./Figure/Densely_Connected_Convolutional_Networks1.JPG)

ResNet과는 다르게 Summation으로 Feature map을 합치는 것이 아니라 Concatenation으로 Feature map을 합친다. 그러므로 l번째 계층이 l개의 입력을 받는다면 이 입력은 모든 선행 계층의 Feature map으로 구성된다. 그리고 이 계층의 Feature map들은 L-l개의 후행 계층들에 전달된다. 그렇기 때문에 네트워크의 경로는 총 L(L+1)/2가 된다. 

저자들에 의하면 이 아키텍처 구조는 전통적인 컨볼루션 계층보다 더 적은 파라미터만 있으면 된다고 하는데 그 이유는 중복되는 Feature map을 다시 학습할 필요가 없기 때문이라고 한다. 전통적인 순전파 아키텍처를 상태가 계층 간에 전파되는 알고리즘으로서 본다면 각 계층은 바로 이전 계층에서 상태를 읽고 바로 다음 계층에 상태를 쓰는 형태가 된다. 계층이 상태를 변화시키기는 하지만 보존되어야할 정보는 그대로 전달하게 된다. ResNet의 경우 Identity transformation으로 이런 작업을 명시적으로 수행했다. Stochastic depth 연구에서는 사실 많은 계층에서 최종 예측에 기여하는 바가 매우 작기 때문에 훈련 중에 랜덤하게 계층을 드롭해도 된다고 주장했다. 이것이 ResNet을 unrolled RNN과 유사하게 보이게 하는데 ResNet의 파라미터 숫자는 엄청나게 많다. 왜냐하면 각 계층이 그 자신만의 가중치를 보유하기 때문이다. DenseNet에서는 네트워크에 추가되는 정보와 보존되어야할 정보를 분리시켰다. DenseNet에서의 계층들은 보통 얇은데(계층당 12개의 필터 수준) 네트워크의 Collective knowledge에 적은 셋의 Feature map만을 추가하고 남아 있는 Feature map의 상태를 변경하지 않기 때문이다(마지막에서 Classification은 모든 계층의 Feature map으로 수행한다). 

효율적인 파라미터 사용과 더불어 DenseNet의 큰 이점 중 하나는 네트워크를 관통하는 정보나 그래디언트 흐름이 개선되었다는 것이다. 이것이 모델을 훈련시키기 쉽게 만든다. 각 계층은 Loss function과 원본 입력 데이터 Signal로부터의 그래디언트로 직접적인 접근이 가능한데 이는 암시적인 Deep supervision이다. 이것이 깊은 아키텍처의 훈련을 돕는다. 그리고 저자들의 관찰 결과 Dense connection은 규제의 효과도 있어서 적은양의 데이터셋으로 인한 과적합 정도를 줄일 수 있다고 한다. 

저자들의 모델은 그 당시 SOTA 기법들과 정확도는 유사하면서 더 적은 파라미터를 가졌다고 한다. 




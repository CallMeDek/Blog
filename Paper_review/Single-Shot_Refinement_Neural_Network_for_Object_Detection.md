# Single-Shot Refinement Neural Network for Object Detection

Shifeng Zhang(CBSR & NLPR, Institute of Automation, Chinese Academy of Sciences, University of Chinese Academy of Sciences)

Longyin Wen(GE Global Research)

Xiao Bian(GE Global Research)

Zhen Lei(CBSR & NLPR, Institute of Automation, Chinese Academy of Sciences, University of Chinese Academy of Sciences)

Stan Z. Li(CBSR & NLPR, Institute of Automation, Chinese Academy of Sciences, University of Chinese Academy of Sciences)



## Abstract

보통 Two-stage 알고리즘들은 정확도에 강점이 있고 One-stage 알고리즘들은 효율성에 강점이 있다. 저자들은 RefineDet이라고 하는 Single-shot Detector로 정확도는 Two-stage 알고리즘보다 높으면서 One-stage의 효율성을 유지하고자 했다. 이 Detector는 네트워크 내에 두 개의 모듈로 구성되어 있다. 

- Anchor refinement module: Classifier가 답을 찾는 Search space의 크기를 줄이기 위해서 Negative anchor를 필터링하는 것. 다음 모듈이 학습을 시작할때 좀 더 잘 학습할수 있도록 위치와 크기가 어느정도 조정된 Anchor들을 제공하는 것.
- Object detection module: 위의 모듈에서 Anchor를 받아서 Bounding box regression과 Multi-class prediction을 수행한다. 

동시에 저자들은 Transfer connection block이라는 개념을 디자인해서 ARM에서 작업을 수행하고 난 뒤의 Feature들을 ODM으로 Transfer한다. Multi-task Loss function을 정의해서 전체 네트워크가 End-to-End로 학습이 가능하게 했다. 

[코드 이용](https://github.com/sfzhang15/RefineDet)



## Introduction

Object Detection Detector들은 크게 두 가지 범주로 나눌 수 있다.

- Two-stage approach: Sparse set of candidate object 박스가 먼저 생성되고 나서 그 박스들로 분류 혹은 박스 회귀가 수행된다.
- One-stage approach: 이미지 내 객체를 탐지할때 위치, 크기, 종횡비를 고려해서 Regular and dense한 샘플링으로 탐지한다. 계산적으로 효율성이 높다는 장점이 있지만(Two-stage와 다르게 Unified된 네트워크가 End-to-End로 학습됨) Two-stage에 비해서 탐지 정확도가 떨어진다는 단점이 있다. 그 이유 중 하나는 클래스 불균형 문제이다. 

클래스 불균형 문제를 해결하기 위한 시도로 객체 탐지 공간의 크기를 줄이기 위해서 컨볼루션 특징 맵에 Objectness prior constraint를 사용하기도 했고 Standard cross entropy loss를, Hard example에 더 초점을 두고 잘 분류된 example에는 비중을 줄이도록 재정의하는 방법도 있었다. 또 False positive의 수를 줄이도록 하는 Max-out labeling mechanism을 고안하기도 했다. 

저자들이 보기에 Faster R-CNN, R-FCN, FPN과 같은 Two-stage 방법들이 One-stage 방법들과 비교했을때 갖는 장점은 다음과 같다.

- 클래스 불균형 문제를 해결하기 위해서 Sampling heuristics를 Two-stage structure로 구현한다(RPN 같이 Region proposal을 생성하는 네트워크가 훈련이 가능해서 더 좋은 Region을 생성할 수 있다).
- 두 단계로 Box의 파라미터를 조정한다.
- 두 단계로 객체를 분류한다(예를 들어서 Faster R-CNN에서는 먼저 객체가 있는지 없는지를 판단하고 있다면 Multi-class classification을 수행한다).

RefineDet은 두 접근법의 장점은 계승하고 단점은 극복하기 위해서 만들어졌다. RefineDet에는 One-stage의 방법의 네트워크 내에 서로 연결되어 있는 두 가지 모듈이 있다. 하나는 Anchor refinement module(ARM)이고 다른 하나는 Object detection module(ODM)이다. 

![](./Figure/Single-Shot_Refinement_Neural_Network_for_Object_Detection1.JPG)

ARM은 Classifier가 탐색하는 공간을 줄이기 위해서 Negative anchor를 없애고 ODM에게 더 나은 초기 상자를 제공하기 위해서 Anchor의 사이즈와 위치를 정밀하게 조정하는 역할을 한다. ODM은 정제된 Anchor들을 받아서 객체에 대한 Detection을 수행한다. 위의 그림 1을 보면 두 모듈은 Two-stage 구조를 모방했기 때문에 더 정확하게 탐지를 수행하면서도 One-stage 접근법의 높은 효율성을 지닌다. 거기다 Transfer connection block이라는 개념을 고안해서 ARM에서 학습한 특징들을 Transfer learning과 같이 ODM에서 사용할 수 있도록 했다. Multi-task loss 함수를 정의해서 전체 네트워크가 End-to-End로 학습이 가능하게 했다. 


## CSPNET: A NEW BACKBONE THAT CAN ENHANCE LEARNING CAPABILITY OF CNN

Chien-Yao Wang(Institute of Information Science Academia Sinica, Taiwan),

Hong-Yuan Mark Liao(Institute of Information Science Academia Sinica, Taiwan),

I-Hau Yeh(Elan Microelectronics Corporation, Taiwan),

Yueh-Hua Wu(Institute of Information Science Academia Sinica, Taiwan),

Ping-Yang Chen(Department of Computer Science National Chiao Tung University, Taiwan),

Jun-Wei Hsieh(College of Artificial Intelligence and Green Energy National Chiao Tung University, Taiwan)



## Abstract

신경망 기반의 방식은 Object detection과 같은 컴퓨터 비전 Task에서 놀라운 결과를 낼 수 있게 했는데 문제는 컴퓨팅 리소스에 크게 의존한다는 것이다. 이 논문에서 저자들은 Cross Stage Partial Network(CSPNet)이라는 것을 제시해서 이 문제를 완화시키고자 했다. 저자들은 네트워크 단의 무거운 연산량이 네트워크 최적화 과정에서 그래디언트 정보가 복제되기 때문에 발생한다고 보았다. 그래서 네트워크 각 Stage의 시작과 끝의 Feature map을 합치는 방법으로 그래디언트의 변동성을 고려했더니 성능이 더 좋아졌다. 저자들이 말하길 CSPNet은 구현하기 쉽고 ResNet, ResNeXt, DenseNet 계열의 아키텍처에서 구현해도 충분히 일반화 잘 된다고 한다. 



## Introduction

네트워크가 깊고 넓을수록 강력하다는 것은 잘 알려진 사실이지만 네트워크를 확장하는것은 필연적으로 더 많은 연산량을 필요로 한다. 그래서 애초에 Object detection과 같이 연산량을 많이 요구하는 Task에서는 깊고 넓은 네트워크를 사용하기 쉽지 않다. 경량 컴퓨팅 시스템은 주목을 받게 된 이유는 실제 애플리케이션이 주로 작은 디바이스에서의 빠른 추론을 요구하기 때문인데 이는 컴퓨터 비전 알고리즘에게는 도전적인 일이다. 몇몇 연구가 모바일 CPU를 위해 디자인되긴 했으나 Depth-wise separable 컨볼루션 같이, 이런 종류의 연구는 상업적으로 사용되는 IC 디자인과는 맞지 않다. 그래서 저자들은 ResNet, ResNeXt, DenseNet과 같은 방식에서 컴퓨팅 연산의 부담이 얼마나 되는지를 조사했다. 그리고 저자들은 앞서 언급한 네트워크들이, 성능을 저해하지 않으면서도 CPU와 모바일 GPU에 배치 될 수 있도록 하는, 연산적으로 효율적인 요소들을 개발했다. 

저자들은 CSPNet의 주요 목적이, 앞서 언급한 네트워크들을 사용하여, 연산량을 줄이면서 더 풍부한 그래디언트 조합을 달성할 수 있도록 하는 것이라고 했다. 이를 위해서 각 Stage의 Base 계층의 Feature map을 두 부분으로 나누고 난 다음 이 연구에서 제안하는 Cross-stage hierarchy로 다시 합친다. 다시 말해서 주요 컨셉은 그래디언트 흐름을 나눠서 각기 다른 네트워크 내의 경로를 통해서 그래디언트가 전파 될 수 있게 하는 것이다. 이런 방법으로 저자들은 Concatenation과 Transition 단계를 바꿈으로서 전파된 그래디언트 정보가 큰 상관관계의 차이를 보임을 확인했다. 게다가 CSPNet은 아래 그림과 같이 계산량을 크게 줄이면서도 정확도와 추론 속도를 개선시켰다. 

| BFLOPS                                                       | FPS                                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](./Figure/CSPNet_A_New_Backbone_That_Can_Enhance_Learning_Capability_of_CNN1.png) | ![](./Figure/CSPNet_A_New_Backbone_That_Can_Enhance_Learning_Capability_of_CNN2.png) |

CSPNet 기반의 Object detector들은 다음과 같은 세가지 문제를 다룬다. 

- CNN의 학습 능력을 강화시킨다 - 보통 경량화 후에 CNN의 정확도가 낮아지는데 저자들은 CNN의 학습 능력을 강화시켜서 경량화 중에도 정확도를 유지할 수 있기를 원했다. CSPNet을 위에서 언급한 네트워크에 적용하고 나서 10~20퍼센트 연산량이 줄어들 수 있다. 그런데 저자들에 따르면 ImageNet 데이터셋에서 오히려 원본 네트워크의 성능보다 더 좋았다고 한다. 
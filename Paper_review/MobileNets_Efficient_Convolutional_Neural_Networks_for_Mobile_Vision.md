# MobileNets: Efficient Convolutional Neural Networks for Mobile Vision

Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang,

Tobias Weyand, Marco Andreetto, Hartwig Adam(Google Inc.)



## Abstract

저자들은 MobileNet이라고 하는 모바일과 임베디드 비전 애플리케이션을 위한 아키텍처를 발표했다. 특징으로는 Depthwise separable convolution으로 가볍고 능률적인 DNN을 구축한다는 것이다. 또 Latency와 Accuracy간의 Trade off를 설정하는 두 개의 Global hyperparameter가 존재한다. 이 두개의 파라미터로 사용자는 각 도메인에 맞게 모델의 크기를 조절가능하다고 한다. 



## Introduction

CNN은 AlexNet이 ILSVRC 2012에서 우승하면서 컴퓨터 비전 분야에서는 대중화 되었다. 연구들의 경향이, 높은 정확도를 달성하기 위해서 깊고 복잡한 모델을 만드는 쪽으로 이어져왔다. 그러나 이런 방향의 발전이 꼭 네트워크를 효율적이라고 할 수는 없는데 로봇공학, 자율주행차, 증강 현실 같은 분야에서는 제한된 플랫폼 안에서 정해진 시간 안에 작업을 수행해야 하기 때문이다. 

이 연구에서 저자들은 이에 맞는 효율적인 네트워크 아키텍처와 아키텍처를 구축하는데 사용되는 두 개의 Hyper-parameter를 설명했다. 



![](./Figure/MobileNets_Efficient_Convolutional_Neural_Networks_for_Mobile_Vision1.JPG)



## Prior Work

많은 연구에서 작고 효율적인 신경망 네트워크를 만드려는 시도가 있었다. 이런 연구들은 훈련된 네트워크를 압축하거나 직접적으로 작은 네트워크를 훈련시키는, 두 가지 방향으로 나눌 수 있다. MobileNet은 네트워크 Latency를 최적하면서도 작은 모델을 만들어내는데 초점을 두었다. 저자들에 따르면 작은 모델을 만드는데 관심 있는 많은 모델들이 사이즈에만 관심을 두고 속도는 고려하지 않는다고 한다. 

MobileNet은 주로 Depthwise separable convolution으로 만들어지고 처음의 몇개의 계층에 대해서만 연산량을 줄이기 위해서 Inception model을 사용한다.  Flattened network는 Fully factorized convolution으로 구축되었고 극도의 Factorized network의 잠재력을 보여줬다고 한다(J. Jin, A. Dundar, and E. Culurciello. Flattened convolutional neural networks for feedforward acceleration. arXiv preprint arXiv:1412.5474, 2014.). 이와는 별개로 Factorized Network에서는 Factorized convolution 뿐만 아니라 위상적(Topological) 연결 방법도 도입했다고 한다(M. Wang, B. Liu, and H. Foroosh. Factorized convolutional neural networks. arXiv preprint arXiv:1608.04337, 2016.). Xception에서는 Inception V3 네트워크를 능가하기 위해서 어떻게 Depthwise separable filter들의 규모를 확대하는지를 보여줬고 Squeezenet에서는 아주 작은 네트워크를 디자인 위해서 Bottlenetck 접근법을 사용했다. 네트워크 연산량을 줄이기 위한 다른 접근법으로는 Structured transform network와 Deep fried convnet이 있다. 

작은 네트워크를 위한 다른 방법으로는 훈련시킨 네트워크를 축소하고 분해하고 압축하는 것이다. Product quantization에 근거한 Compression, Hashing, Pruning, Vector quantization, Huffman coding이 많은 문헌 연구에서 제안된 바 있다. 거기다 다양한 네트워크 분해 방법이 훈련된 모델의 속도를 높이기 위해서 제안되었다. Distillation은 작은 네트워크를 가르치기 위해서 큰 네트워크를 사용하는 것이다. 또 다른 방법으로는 Low bit network가 있다. 
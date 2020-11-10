# Xception: Deep Learning with Depthwise Separable Convolutions

François Chollet(Google, Inc.)



## Abstract

저자들은 보통의 컨볼루션과 Depthwise separable convolution(Depthwise convolution 후에 Pointwise convolution) 사이쯤에 있는 CNN에서의 Inception module을 해석했다. 저자들이 발표한 용량이 가벼운 Depthwise separable convolution의 경우 극단적인 수의 계층들의 집합이 횡으로 연결되어 있는 Inception module로 이해할 수 있다. 저자들은 이를 통해 DCNN에서 Inception module을 Depthwise separable convolution으로 바꾸는 새로운 형태의 아키텍처를 제시할 수 있었다. 저자들은 이 논문에서 Xception이라고 이름 붙인 이 아키텍처가 ImageNet 데이터셋에서 Inception V3의 성능을 경미하게 웃돌고 17,000 클래스의 350 백만 장의 대용량 데이터셋에서는 크게 앞지르는 것을 확인했다. 그러면서 Xception이 Inception V3와 용량이 같기 때문에 모델의 용량이 증가해서 성능이 좋은 것이 아니라 어떻게 모델 파라미터를 효율적으로 사용했는가 덕분에 성능이 더 좋은 것이라고 한다. 



## Introduction

컴퓨터 비전에서 CNN은 마스터 알고리즘처럼 등장해왔다. 그리고 이 CNN을 어떻게 조직할 것인가로 연구자들이 고심해왔다. 



- LeNet-style: Y. LeCun, L. Jackel, L. Bottou, C. Cortes, J. S. Denker, H. Drucker, I. Guyon, U. Muller, E. Sackinger, P. Simard, et al. Learning algorithms for classification: A comparison on handwritten digit recognition. Neural networks: the statistical mechanics perspective, 261:276, 1995.
- AlexNet: A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information processing systems, pages 1097–1105, 2012.
- Zeiler and Fergus: M. D. Zeiler and R. Fergus. Visualizing and understanding convolutional networks. In Computer Vision–ECCV 2014, pages 818–833. Springer, 2014.
- VGG: K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014.
- Inception V1: C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1–9, 2015.
- Inception V2: S. Ioffe and C. Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In Proceedings of The 32nd International Conference on Machine Learning, pages 448–456, 2015.
- Inception V3: C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the inception architecture for computer vision. arXiv preprint arXiv:1512.00567, 2015.
- Inception-ResNet: C. Szegedy, S. Ioffe, and V. Vanhoucke. Inception-v4, inception-resnet and the impact of residual connections on learning. arXiv preprint arXiv:1602.07261, 2016.
- Network-In-Network: M. Lin, Q. Chen, and S. Yan. Network in network. arXiv preprint arXiv:1312.4400, 2013.

데이터와 관련된 논문

- ImageNet: O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, et al. Imagenet large scale visual recognition challenge. 2014.
- JFT: G. Hinton, O. Vinyals, and J. Dean. Distilling the knowledge in a neural network, 2015.



Inception 스타일의 기본적인 빌딩 블록은 Inception module이고 몇 가지 서로 다른 구현체가 존재한다. 

![](./Figure/Xception_Deep_Learning_with_Depthwise_Separable_Convolutions1.JPG)

 Figure 1은 Inception V3에서 볼 수 있는 전형적인 Inception module의 구조이다. Inception model은 이 module을 쌓은 것으로 볼 수 있다. 이 생각은  VGG에서 단순한 컨볼루션 계층을 쌓아간다는 아이디어에서 착안했다. Inception module이 일반적인 컨볼루션과 비슷하지만(둘 다 이미지의 특징을 뽑아낸다) 전자가 경험적으로 봤을 때 좀 더 적은 파라미터로 더 풍부한 특징을 뽑아 낼 수 있는 것으로 보였다. 



### The Inception hypothesis

컨볼루션 계층은 필터를 3D 공간(2개의 Spatial dimentions(Width, Height)와 Channel dimension)에서 학습시키려고 한다. 그래서 하나의 컨볼루션 커널은 Cross-channel correlations을 출력으로 매핑하는 것과 Spatial correlations을 출력으로 매핑하는 작업을 동시에 진행한다. 

Inception modlue에 내재되어 있는 아이디어는 Cross-channel correlations와 Spatial correlations를 독립적으로 살펴보도록 명시적으로 과정을 나누는 것이다. 구체적으로 보통의 Inception module은 먼저 1x1 컨볼루션의 Pointwise 컨볼루션을 통해 Cross-channel correlations을 살펴본다. 그리고 원래 공간보다 크기가 더 작은 독립적인 공간 그룹으로 관련 있는 입력 데이터 부분을 매핑한다. 그리고 나서 3x3이나 5x5 같은 보통의 컨볼루션 연산으로 이 3D 공간을 매핑한다 (Figure 1 참조).  Inception의 근본적인 가정은 Cross-channel correlations와 Spatial correlations을 동시에 매핑하는 것이 좋지 않기 때문에 Decoupling 하는 것이 좋다는 것이다. 

![](./Figure/Xception_Deep_Learning_with_Depthwise_Separable_Convolutions2.JPG)

저자들에 의하면 위와 같이, 3x3처럼 한 가지 사이즈의 컨볼루션 연산만 진행하고 Pooling 계층을 포함하지 않는 단순한 Inception module의 경우 아래와 같이, 하나의 많은 output channel 수를 가진 1x1 컨볼루션 연산을 수행한 뒤에 각각 겹치는 부분이 없는 Spatial 컨볼루션을 수행하는 모듈로 재구성될수 있다고 한다. 

![](./Figure/Xception_Deep_Learning_with_Depthwise_Separable_Convolutions3.JPG)

저자들이 궁금했던 건 위와 같은 구조가 원래의 Inception module과 달리 Cross-channel correlations와 Spatial correlations를 각각 매핑하여 더 강력한 성능을 내는가였다. 



### The continuum between convolutions and separable convolutions

![](./Figure/Xception_Deep_Learning_with_Depthwise_Separable_Convolutions4.JPG)

극단적으로 저자들의 가설에서 매 Output channel마다 Spatial correlations를 매핑하는 구조가 위 그림의 구조이다 (Extreme version of Inception module). 저자들은 이런 Extreme한 구조가 거의 Depthwise separable convolution(Separable convolution)과 동일하다고 주장한다.  

이런 Separable convolution은 입력 데이터의 각 채널마다 독립적으로 수행되는 Spatial convolution인 Depthwise convolution, 그리고 뒤에,  Depthwise convolution 수행 후에 새로운 Channel 공간으로 데이터를 Projection하는 1x1 Convoltion인 Pointwise convolution으로 구성되어 있다. 

Extreme한 버전의 Incetipn과 Depthwise separable convolution의 두 가지 사소한 차이점은 아래와 같다. 

- 연산 순서: Depthwise separable convolution의 경우 채널 별 Spatial convolution을 수행하고 나서 Pointwise convolution을 수행하는데 반해 Extreme은 Pointwise convolution을 먼저 수행한다.
- 첫 번째 연산 수행 후에 비선형성의 존재 유무: Extreme은 첫 번째 연산 수행 후에 ReLU 비선형성이 추가되지만 Depthwise의 경우 보통 비선형성이 추가되지 않는다.  

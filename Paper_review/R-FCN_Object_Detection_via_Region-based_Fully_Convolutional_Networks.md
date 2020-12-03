# R-FCN: Object Detection via Region-based Fully Convolutional Networks

Jifeng Dai, Kaiming He, Jian Sun(Microsoft Research), Yi Li(Tsinghua University)



## Introduction

Two stage 계열 Object Detection 알고리즘들은 ROI Pooling 계층에 의해서 두 부분의 나눠진다. 

- ROI와 상관 없이 연산 결과를 공유하는 Fully convolutional subnetwork.
- 연산을 공유하지 않고 ROI마다 연산을 수행하는 ROI-wise subnetwork.

Two stage Object Detection 알고리즘 네트워크에서 Classification 서브 네트워크는 Spatial Pooling 계층과 몇 개의 완전 연결 계층으로 구성되었다. 

그런데 점점 네트워크의 모든 계층이 전부 컨볼루션 계층으로 이루어진 모델들이 나타나기 시작했다. 비유자하자면 Object detection 네트워크에서 ROI마다 수행하는 서브 네트워크에 완전 연결계층 같은 Hidden 계층이 없어지고 모든 컨볼루션 계층이 공유되는 그런 네트워크이다. 그렇지만 본 연구에서 조사한 바에 따르면 이런 단순한 해결책은 Classification 정확도에 비해 Detection 정확도가 현저히 떨어진다.  이를 해결하기 위해서 ResNet 논문에서는 Faster R-CNN detector가 두 컨볼루션 계층들의 집합 사이에 삽입되었는데 이게 정확도는 개선하지만 공유되지 않는 ROI 마다의 계산 때문에 속도가 느렸다. 

저자들은 이것이 Classification에서의 Translation invariance와 Object detection에서의 Translation variance의 간극 때문이라고 주장했다. Classification에서는 이미지 내 객체의 위치가 바뀌거나 회전하거나 해도 그 객체라고 분류하지만 Object detection에서는 위치 추정까지 하기 때문에 Translation에 민감하다. 저자들은 깊은 층의 컨볼루션 계층은 이런 Translation에 덜 민감하다고 가정했다. ResNet detection에서는 컨볼루션 계층 사이에 ROI Pooling 계층을 삽입해서 Translation invariance 함을 약화시켯다. 그러나 이런 접근 법은 ROI 마다 연산을 수행하기 때문에 훈련과 테스트 간의 효율성을 저하시킨다. 

![](./Figure/R-FCN_Object_Detection_via_Region-based_Fully_Convolutional_Networks1.JPG)

본 연구에서 저자들은 Region-based Fully Convolutional Network라고 하는 Object detection을 위한 Framework를 개발했다. R-FCN은 FCN와 같이 모든 계층이 컨볼루션 계층이고 연산이 공유된다(한 번에 모든 연산이 수행됨). FCN에서 Translation variance를 고려하게 하기 위해서 저자들은 FCN의 출력으로서 특별한 컨볼루션 계층들을 사용해서 Position-sensitive score map이라고 하는 것을 구축했다. 각 Score map은 객체의 각 부분의 위치 정보를 상대적인 위치로 나타내서 인코딩했다(예를 들어서 객체의 왼쪽). 이런 FCN 최상위층에는 Position-sensitive ROI pooling 계층을 추가해서 Score map들로부터의 정보를, 파라미터 없는 계층으로(컨볼루션, 완전 연결 계층과 같은 계층 말고) 처리했다. 전체적인 아키텍처는 종단 간 학습이 가능하다. 모든 학습 가능한 계층은 전체 이미지에 대해서 연산을 수행하고 연산 결과는 공유되면서 Object detection에 필요한 공간 정보를 인코딩한다. 

![](./Figure/R-FCN_Object_Detection_via_Region-based_Fully_Convolutional_Networks2.JPG)

ResNet-101 아키텍처를 Backbone으로 사용했을때 R-FCN은 PASCAL VOC 2007 데이터셋에서 83.6% mAP의 성능을 보였고 2012 셋에서는 82.0%의 성능을 보였다. 이 모델은 테스트 시에 이미지 한 장을 처리하는데 170ms 라는 시간이 소요되었는데 이는 ResNet-101에 Faster R-CNN을 적용한 모델보다 2.5에서 20배 더 빠른 속도이다. 이 연구의 핵심 아이디어는 Translation invariance/variance간의 딜레마를 해결하고 ResNet 같은 아키텍처가 완전 컨볼루션 계층으로도 Object detection 모델로 사용될 수 있음을 보이는 것이다. 



## Our approach

### Overview

저자들은 R-CNN을 따라서 유명한 Two-stage object detection 알고리즘 전략을 도입했다.

- Region Proposal
- Region classification

저자들의 연구에서도 Region Proposal Network로 지역 후보들을 추출한다. RPN도 Fully convolutional 아키텍처이다. 아래와 같이 RPN과 R-FCN은 Base network에서 추출한 특징을 공유한다. 

![](./Figure/R-FCN_Object_Detection_via_Region-based_Fully_Convolutional_Networks3.JPG)

R-FCN에서는 생성된 지역 후보를 카테고리와 배경으로 분류하는 작업을 수행한다. R-FCN에서 모든 학습 가능한 계층은 컨볼루션 계층이고 전체 이미지에 대해서 연산을 수행한다. 이 아키텍처에서의 마지막 계층은 연달아 있는 k^2 크기의 Position-sensitive score map을 생성하므로, C개의 카테고리와 배경에 대해서 k^2(C+1) 차원의 결과를 출력한다. 연달아 있는 이 k^2 크기의 Score map들은 k x k 격자에서 각 상대적인 위치에 대응된다. 예를 들어서 3x3일 경우 9개의 Score map은 각 카테고리에 대한 {top-left, top-center, ..., bottom-right}에 대한 정보를 인코딩한다. R-FCN은 Position-sensitive ROI Pooling 계층으로 끝난다. 이 계층에서는 마지막 컨볼루션 계층에서의 출력을 집계해서 각 ROI에 대해 점수를 매긴다. Position-sensitive ROI 계층은 Selective pooling 연산을 수행하고 k x k개의 각 bin들은 k x k의 Score map 중에서 오직 하나의 Score map에서의 반응만 집계한다. 종단간 훈련간에 이 ROI 계층은 마지막 컨볼루션 계층이 Specialized position-sensitive score map을 학습하도록 한다. 

[Jonathan Hui - Understanding Region-based Fully Convolutional Networks (R-FCN) for object detection](https://jonathan-hui.medium.com/understanding-region-based-fully-convolutional-networks-r-fcn-for-object-detection-828316f07c99)

![](./Figure/R-FCN_Object_Detection_via_Region-based_Fully_Convolutional_Networks4.JPG)

예를 들어서 이미지 안에 5x5 Feature map 안에 객체가 들어있다고 가정하자. 여기서 이 Feature map을 3x3의 격자로 나눈다. 여기서 객체의 Top-left에만 반응하는 Feature map있다고 할때 오른쪽과 같은 Feature map이 만들어 질 것이다. 3 x 3 = 9이므로 한 개의 5x5 Feature map에서는 다음과 같이 9개의 새로운 Feature map이 만들어질 수 있다. 이를 Position-sensitive score map이라고 한다. 이렇게 부르는 이유는 각 Feature map이 객체의 부분적인 위치를 탐지하기(Score하기) 때문이다.

![](./Figure/R-FCN_Object_Detection_via_Region-based_Fully_Convolutional_Networks5.JPG)

예를 들어 다음의 그림에서 빨간색 점선 박스를 생성된 ROI라고 했을때 이 ROI를 3 x 3의 격자로 나누면 각 Position-sensitive score map과 ROI 격자의 각 셀이 매칭되어 ROI 격자의 각 셀이 원래 객체의 각 부분을 얼마나 포함하고 있는지 알 수 있다. 그리고 그 정도를 3 x 3 vote array에 저장할 수 있다. 

![](./Figure/R-FCN_Object_Detection_via_Region-based_Fully_Convolutional_Networks6.JPG)

이렇게 Score map과 ROI들을 Vote array로 매핑하는 과정을 Position-sensitive ROI-pool이라고 한다. 

![](./Figure/R-FCN_Object_Detection_via_Region-based_Fully_Convolutional_Networks7.JPG)

예를 들어서 위의 그림에서 Top-left score map과 ROI Top-left 부분을 매핑했을때 60%정도의 부분만 포함되므로 Vote array의 Top-left는 0.6이 된다.  모든 Position-sensitive ROI pool의 값들을 계산하고 나서 Vote array의 요소 값들의 평균을 구하면 그 값이 클래스에 대한 Score가 된다. 

![](./Figure/R-FCN_Object_Detection_via_Region-based_Fully_Convolutional_Networks8.JPG)

이를 C개의 클래스와 1개의 배경에 대해서 수행하면 (C+1) x 3 x 3의 Score map이 생성된다. 그리고 나서 class에 대한 class score를 예측하고나서 Softmax를 적용하면 각 클래스에 대한 확률 분포를 출력할 수 있게 된다. 



### Backbone architecture

R-FCN의 구현체는 ResNet-101로 구현했지만 다른 아키텍처도 가능하다고 한다. ResNet-101은 100개의 컨볼루션 계층에 Global average pooling과 1000개 클래스를 분류할 수 있는 완전 연결 계층으로 이루어져 있다. 여기서 저자들은 Global average pooling과 완전 연결계층을 제거하고 Feature map을 계산하기 위해 오직 컨볼루션 계층만 사용했다. ResNet은 ImageNet에서 미리 훈련된 모델을 사용했다. ResNet-101 모델의 마지막 컨볼루션 블럭의 채널 차원이 2048이므로 저자들은 임의의 값으로 초기화된, 1x1 컨볼루션을 수행하는, 1024 차원의 컨볼루션을 붙여서 출력 채널수를 줄였다. 그리고 나서 k^2(C+1) 차원(출력 채널 차원 수)의 컨볼루션 계층을 Score map을 생성하기 위해 더했다.  



### Position-sensitive score maps & Position-sensitive ROI pooling

명시적으로 위치와 관련된 정보를 각 ROI에 인코딩하기 위해서 ROI를 k x k bins로 나눈다. 그러므로 만약에 ROI 상자의 크기가 w x h라면 각 상자의 크기는 거의 w/k x h/k와 같다. 여기서는 마지막 컨볼루션 계층이 각 카테고리에 대해서 k^2 score map을 생성한다. (i, j)째 bin(0 <= i, j <= k-1)에서는 (i, j)번째 Score map에서의 Pool 연산만 진행하는 Position-sensitive ROI pooling 연산을 다음과 같이 수행한다. 

![](./Figure/R-FCN_Object_Detection_via_Region-based_Fully_Convolutional_Networks9.JPG)

r_c(i, j)는 c번째 카테고리에 대한 (i, j)번째 빈에서의 Pooled response이다. z_i,j,c는 k^2(C + 1)개의 score map 중에서 하나의 Score map이고 (x0, y0)는 ROI의 Top-left 코너를 나타낸다. n은 bin 안에 있는 픽셀 수를 나타낸다. 그리고 Θ는 네트워크 안에 있는 모든 학습 가능한 파라미터를 뜻한다. (i, j)번째 bin은 다음과 같은 공간에 걸쳐 있다. 

![](./Figure/R-FCN_Object_Detection_via_Region-based_Fully_Convolutional_Networks10.JPG)

Position-sensitive ROI pooling 연산은 Figure 1에서 묘사되어 있는데 각 색깔은 (i, j)의 쌍을 나타낸다. 여기서는 Average pooling 연산을 수행했지만 Max pooling 연산도 가능하다. 
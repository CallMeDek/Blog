# CornerNet: Detecting Objects as Paired Keypoints

Hei Law, Jia Deng(Princeton University, Princeton, NJ, USA)



## Abstract

저자들이 발표한 CornerNet은 바운딩 박스를 Top-left와 Bottom-right의 두 개의 쌍으로 보고 탐지하는 방법이다. One-stage 알고리즘이고 1개의 Unified network를 사용한다. 저자들이 말하길 이렇게 함으로써 Anchor 박스를 디자인하는 번거로움을 없앴다고 한다. 또 Corner pooling이라고 하는 새로운 Pooling 연산을 소개하면서 말하길 이 연산으로 네트워크가 객체의 위치를 더 잘 찾을 수 있다고 한다. MS COCO에서 42.2% AP를 달성했다. 



## Introduction

ConvNet 계열의 Object detection 알고리즘들이 공통적으로 도입하는 요소는 Anchor box라는 개념이다. 이 박스들은 다양한 크기와 종횡비를 가지는데 Detection을 위한 Candidate의 역할을 한다. 특히 One-stage 알고리즘에서 많은 수의 Anchor box가 사용된다. 

그런데 저자들이 주장하길 이런 방식에는 두 가지 단점이 있다고 한다.

- 첫번째로 매우 많은 양의 Anchor 박스가 있고 그 중에서 몇 가지만 GT에 가깝게 조정된다. 이렇게 되면 실제적으로 정답에 가까운 박스들은 극 소수이고 나머지는 정답이 아닌 박스들이므로 클래스간 심각한 불균형 문제가 발생한다. 
- 두번째로  Anchor 박스를 디자인하기 위한 디자인 문제가 발생한다. 여기에는 박스 모양과 관련된 Size, Aspect ratio 같은 하이퍼 파라미터 요소를 포함한다. 이 요소들은 경험적으로 사용자에 의해서 결정되어야 하는데, 네트워크 디자인에 더 많은 옵션을 고려해야할 경우 더 복잡해진다. 

저자들은 그래서 Anchor 박스 개념 없이 Object detection을 수행하는 방법을 제안했다. 구체적으로 Single convolutional network에서 각 객체들의 Top-left corner를 위한 Heatmap과 Bottom-right corner를 위한 Heatmap 그리고 각 Corner를 각 객체에 맞게 그룹핑하는데 쓰는 Embedding vector들을 출력한다. 

![](./Figure/CornerNet_Detecting_Objects_as_Paired_Keypoints1.JPG)

![](./Figure/CornerNet_Detecting_Objects_as_Paired_Keypoints2.JPG)

![](./Figure/CornerNet_Detecting_Objects_as_Paired_Keypoints3.JPG)

![](./Figure/CornerNet_Detecting_Objects_as_Paired_Keypoints4.JPG)

위 그림과 같이 Heatmap에서 Corner들을 찾고 나서 Embedding vector를 조사해서 각 객체에 맞게 그룹핑한다. Heatmap의 경우 각 클래스마다, 두 가지 Corner가 다른 Channel로써 예측되므로 Channel수가 # of Classes x 2가 들어간다. 

![](./Figure/CornerNet_Detecting_Objects_as_Paired_Keypoints6.JPG)

다음으로 저자들은 Corner pooling이란 개념을 소개했다. 이것은 바운딩 박스의 Corner가 다음과 같이 실제 객체에서 굉장히 떨어져 있는 경우를 보완하기 위한 개념이다. 

![](./Figure/CornerNet_Detecting_Objects_as_Paired_Keypoints5.JPG)

위의 경우 Top-corner가 정확히 객체의 Top-corner 라는 것을 보장하지 못하기 때문에 Top-corner에서 수평적으로 이미지의 오른쪽 끝까지 살펴보고, 수직적으로 이미지의 아래쪽 끝까지 살펴볼 필요가 있다. 그래서 아래 그림과 같이 같은 직선에 있는 값들에 대해서 Max pooling을 수행한다.

![](./Figure/CornerNet_Detecting_Objects_as_Paired_Keypoints7.JPG)

저자들이 말하는 기존의 방법보다 여기서 발표한 방법이 더 나은 이유는 두 가지가 있다. 

- 첫 번째로 Corner와 다르게 객체의 Center와 관련해서는 Center로부터 4가지 방향을 신경써야 한다. 

  ![](./Figure/CornerNet_Detecting_Objects_as_Paired_Keypoints8.JPG)

- 두 번째로 상자의 공간을 조밀하게 이산화하는 효율적인 방법을 제공한다. 아래 그림을 보면 CornerNet의 경우 Heatmap을 표현하는데 O(WH)의 공간이 필요한데 반해 기존의 방법들은 이미지 전체 좌표 상에서 각 박스의 w, h도 표현해야 하므로 O(W^2H^2)의 공간이 필요하다.

  ![](./Figure/CornerNet_Detecting_Objects_as_Paired_Keypoints9.JPG)

  

[princeton-vl - CornerNet](https://github.com/princeton-vl/CornerNet)


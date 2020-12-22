# Objects as Points

Xingyi Zhou(UT Austin), Dequan Wang(UC Berkeley), Philipp Krahenbuhl(UT Austin)



## Abstract

이 논문 전의 Object Detection이 수행되는 방식은 객체를 축에 따라 정렬된 박스로 보고 객체가 있을 법한 많은 위치의 리스트를 열거해놓고 각각을 분류하는 방식이었다. 저자들은 이런 방식이 시간 낭비이고 비효율적이며 별도의 후처리가 필요하다고 주장했다. 저자들이 제안한 방식은 객체를 단일한 점으로 보는 방식이었다. 이때 점은 바운딩 박스의 중심 좌표이다. 저자들은 Keypoint estimation으로 중심 좌표를 찾고 회귀를 각 Task의 Property에 대해서 수행했다. 예를 들어 박스의 크기, 3차원에서의 위치, 방향 그리고 객체의 포즈도 있다. 저자들은 End-to-End 방식의 아키텍처인 CenterNet을 제안했다. CenterNet은 MS COCO 데이터셋에서 28.1% AP에서는 142 FPS, 37.4% AP에서는 52 FPS, 45.1% AP의 Multi-scale test에서는 1.4 FPS의 성능을 보였다. 



## Introduction

앞서 언급했듯이 이 연구 이전의 SOTA Object detection 알고리즘들은 각 객체를, 이 객체를 둘러싸고 있는 축 정렬된 박스로 나타냈다. 그리고 나서 아주 많은 양의 박스 후보들에 대해서 분류 작업을 수행하여 Object detection을 수행했다. 각 박스에 대해서는 어떤 특정 클래스의 객체인지 그냥 배경인지로 분류했다. One stage 알고리즘들은 이미지 내 모든 위치에 Anchor 박스라고 하는 후보 박스를 모두 순회한다. Two stage 알고리즘들은 Backbone에서 추출한 이미지 특징으로 연산을 수행해서 객체가 포함되어 있을 법한 박스를 만들어 내고 그 박스 위주로 분류를 수행한다. 이런 알고리즘들은 NMS라고 하는 후 처리 작업을 수행해서 같은 객체에 연관 되어 있는 박스들의 IOU를 계산해서 상당 수의 겹친 박스를 제거한다. 이런 후처리 작업은 미분 하기 힘들기 때문에 훈련시키기 힘들다. 따라서 End-to-End 방식으로 모델을 훈련시키는 것이 어렵다. 그럼에도 불구하고 Two-stage 방식은 뛰어난 성능을 보였다. Sliding window 방식의 알고리즘들은 가능한 모든 이미지 내의 위치와 차원을 순회해야 한다는 점에서 시간 낭비가 있다. 

저자들은 객체를 바운딩 박스의 중심좌표로 표현하는 방식의 알고리즘을 제안했다. 

![Objects_as_Points1](./Figure/Objects_as_Points1.JPG)

각 Task의 property들, 예를 들어 객체 크기, 차원, 3D 깊이나 차원, 객체의 방향, 포즈 같은 것들은 중심 위치의 이미지 특징으로 부터 직접적으로 회귀를 수행해서 구한다. 이렇게 되면 Object detection을 Standard keypoint estimation 문제로 볼 수 있다. 입력 이미지를 먼저 FCN에 넣으면 하나의 Heatmap을 출력한다. 이 Heatmap의 Peak가 Object의 중심점이 된다. 각 Peak의 이미지 특징들로 객체를 둘러싼 바운딩 박스의 Height와 Weight를 예측한다. 모델은 Standard dense supervised learning으로 학습시킨다. 추론 과정에서는 NMS와 같은 후속 처리 없이 Single-network forward-pass로 수행된다. 

저자들은 이 연구에서 제안한 방법으로 2D Object detection 이외에 다른 작업도 수행할 수 있다고 했다. 

CenterNet으로 매우 빠른 작업을 수행할 수 있다고 한다. 

![](./Figure/Objects_as_Points2.JPG)

[xingyizhou - CenterNet](https://github.com/xingyizhou/CenterNet)


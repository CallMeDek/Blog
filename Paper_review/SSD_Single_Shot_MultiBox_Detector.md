# SSD: Single Shot MultiBox Detector

Wei Liu(UNC Chapel Hill), Dragomir Anguelov(Zoox Inc.), Dumitru Erhan(Google Inc.), Christian Szegedy(Google Inc.), Scott Reed(University of Michigan, Ann-Arbor), Cheng-Yang Fu(UNC Chapel Hill), Alexander C. Berg(UNC Chapel Hill)



## Abstract

저자들이 주장하는 이 연구의 기여의 핵심 점은 단일 딥러닝 네트워크로 다양한 크기의 객체들을 탐지할 수 있는 방법을 제시했다는 것이다. 네트워크의 각 계층에서의 특징 맵을 그리드 형식으로 나누고 각 셀마다 다양한 크기와 종횡비를 갖는 Default box들을 할당해서 이 박스들로 객체에 대한 Predicted Bounding box를 도출해낸다.  그래서 네트워크는 훈련 간에, 각 Default box 안에 각 객체 카테고리에 대한 Presence scoring을 하고, 객체의 위치에 Default box들이 잘 맞도록 조정하게 된다. 특히 YOLOv1과는 다르게 각 계층 마다의 특징맵(당연히 크기가 다 다름.)에 대해서 예측 값을 도출해서 최종적으로 생성되는 결과에 합치게 된다. 이렇게 하면 다양한 크기의 객체를 탐지 하는데 용이하다. Faster R-CNN 같이 지역 후보들을 생성하고 이 지역 후보들에 대해 Scoring을 처리하는 방식이 아니고 종단 간에 특징 추출하고 결과를 도출하는 과정이 통합되어 있기 때문에 속도도 훨씬 빠르다고 한다. 



## Introduction

Two-stage 방식의 Dector들은 기본적으로 이미지 안에 객체가 들어있을만한 패치를 생성하고 그 패치에 대해서 사후 처리를 하는 방식이기 때문에 속도가 느릴수 밖에 없다. 예를 들어서 이쪽 계열에서 가장 뛰어나다고 알려진 Faster R-CNN도 7 FPS의 속도밖에 내지 못한다. 이는 실시간성이 중요한 애플리케이션이나 임베디드 시스템에는 부적절하다. 또 CNN에 적절한 패치를 생성하기 위해서 각 이미지들에 어떤 변형을 가하기도 한다. 

저자들이 제안하는 방식은 이런 과정을 없앤다. CNN의 완전 연결 계층 대신에 작은 사이즈의 필터를 가지는 여러 컨볼루션 계층을 붙인다. 각 계층의 특징 맵에 대해서 다양한 크기와 종횡비를 가지는 Default box들을 이용해서 객체가 있을법한 지역에서 클래스 Score와 바운딩 박스 Offset을 예측한다. 각 계층의 특징 맵의 크기가 다르기 때문에 각 단계에서 담당하는 객체의 크기가 다르다. 그렇기 때문에 YOLO와는 다르게 다양한 크기의 객체를 탐지하는데 더 이점이 있다. CNN을 훈련시키고 SVM을 훈련시키고 Bounding box regressor를 훈련시키는 대신에 하나의 단일 네트워크를 종단간으로 훈련시키면 되기 때문에 속도도 훨씬 빠르다. 



## The Single Shot Detector(SSD)

![](./Figure/SSD_Single_Shot_MultiBox_Detector1.JPG)



### Model

SSD는 기본적으로 네트워크 순전파 시에 정해진 양의 바운딩 박스 예측 값을 생성해내고 그 박스 안에서 클래스 Score를 뽑아낸후 최종 예측 값을 뽑아내기 위해서 NMS를 수행한다. CNN에서 분류를 수행하는 부분을 잘라내고 다음의 특징을 갖는, 위에서 언급한 작업을 수행하는 추가적인 컨볼루션 계층들을 덧붙인다. 



#### Multi-scale feature maps for detection

덧붙여진 컨볼루션 계층들의 특징 맵의 크기가 점점 작아지고 각 특징 맵마다 예측 값이 생성되므로 자연스럽게 다양한 크기에 대한 예측 값이 만들어지게 된다. 



#### Convolutional predictors for detection

Figure2에 나와 있는 네트워크에 추가된 상위의 컨볼루션 계층에 대해서, 각 계층에서의 특징맵의 크기가 m x n이고 p 채널 수를 가지고 있을 때, 기본적으로 적용되는, Detection과 관련된 필터는 3 x 3의 크기를 가진 p 채널의 커널이다. 입력으로 들어온 특징 맵의 각 위치에 이 커널이 적용되고 위에서 언급한 출력 값을 생성해낸다. 바운딩 박스 Offset은 특징 맵의 각 위치와 관련된 Default box와 관련된 값들이 출력되게 된다. 
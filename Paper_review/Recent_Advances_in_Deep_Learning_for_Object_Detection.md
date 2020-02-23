# Recent Advances in Deep Learning for Object Detection



Xiongwei Wu(School of Information System, Singapore Management University),

Doyen Sahoo(School of Information System, Singapore Management University),

Steven C.H. Hoi(School of Information System, Singapore Management University, Salesforce Research Asia)



arXiv: 1908.03673v1







## 0. Abstract

객체 탐지는 주어진 이미지 안의 특정 타겟 클래스의 객체의 정확한 위치를 찾아내어 그에 부합하는 클래스 레이블을 부여하는 것을 목적으로 한다. 최근에 딥러닝 기반 객체 분류가 큰 성공을 거두어 객체 탐지에서도 딥러닝을 사용하는 방법이 활발하게 연구 되었다. 이 연구에서는 현재 존재하는 객체 탐지 프레임워크를 시스템적으로 분석하고 다음과 같은 부분을 조사했다. 1) Detection components 2) Learning strategies 3) Application & benchmarks. 또한 탐지 성능에 영향을 주는 여러 요소들에 대해서도 다루었다(Detector architectures, Feature learning, Proposal generation, Sampling strategies, etc.). 마지막에서는 딥러닝 기반의 객체 탐지 연구가 나아가야할 방향에 대해 다루었다.

Keywords: Object Detection, Deep Learning, Deep Convolutional Neural Networks







## 1. Introduction

![](./Figure/Recent_Advances_in_Deep_Learning_for_Object_Detection_1.JPG)

(a)와 같은 **Image classification **은 주어진 이미지 안에서 객체를 어떤 카테고리로 분류하는 데에 목적이 있다.

(b)와 같은 **Object detection** 은 카테고리 분류 뿐만 아니라 바운딩 박스로 객체의 위치를 예측하는 작업도 포함한다.

(c)와 같은 **Semantic segmentation** 은 Object detection과 같이 객체를 분류하는 것이 아니라 픽셀 단위로 분류를 진행함으로서 같은 카테고리의 다른 객체를 구별하지 않는다. 

(d)왁 같은 **Instance Segmentation** 은 (b)와 (c)의 개념이 합쳐진다. 이미지 안의 객체들에 대해서 픽셀 단위로 마스킹을 하되, 각각 다르게 분류한다.  (b)에서 바운딩 박스로 Localization 하는 것 대신에 픽셀로 Localization을 한다. 

딥러닝 이전의 객체 탐지 파이프 라인은 세가지 단계로 이루어졌다.

1. Proposal generation -  이 단계에서는 이미지 안에서 객체를 탐지하고 있을만한 위치를 찾는다. 이 위치를 **Regions of interest(ROI)** 라고 한다. 가장 직관적인 방법은 전체 이미지를 슬라이딩 윈도우 방식으로 스캔하는 것인데 다양한 크기의, 각기 다른 가로 세로 비율을 가진 객체들의 정보를 획득하기 위해서 입력 이미지들은 각기 다른 크기로 조정되고 다양한 크기의 윈도우가 이미지를 스캔하기 위해 사용된다.
2. Feature vector extraction - 이 단계에서는 이미지의 각 위치에서 슬라이딩 윈도우를 통해 고정된 크기의 특징 벡터들이 얻어진다. 이 벡터들은 각 지역의 특징 정보를 포함하고 있다. 또한 이 벡터들은 조도, 회전, 크기 변화에 큰 영향을 받지 않는다는 것을 보여준 SIFT(Scale Invariant Feature Transform), Haar, HOG(Histogram of Gradients), SURF(Speeded Up Robust Features) 같은 low-level visual descriptors에 의해 인코딩된다.
3. Region classification - 이 단계에서 지역 분류기들은 관심 지역을 레이블링하는 법을 배운다. 여기에는 SVM(Support Vector machines)가 적은 훈련 세트에도 좋은 성능을 보여서 사용되었다. 특별히 bagging, cascade learning, adaboost 같은 분류 기술들이 탐지 정확도를 개선하기 위해서 이 단계에서 사용되었다.

이런 방법들은 Pascal VOC 데이터 셋에서 인상적인 결과를 달성했다. 

 DPMs는 deformable loss로여러 모델들을 통합하고 학습한다. 그리고 각각의 훈련을 위해 그 안에 잠재되어 있는 SVM으로 심각하게 부정적인 포인트들을 캐낸다. 

그러나 위의 방법들은 다음과 같은 이유에 의해서 한계점을 보였다.

- Proposal generation 과정 중에서, 엄청나게 많은 수의 불필요한 제안들이 생성되었고 이는 분류 과정에서 많은 수의 false positive들을 낳았다. 게다가 윈도우 크기는 수동적으로 경험에 의해서 디자인되었기 때문에 실제 객체에 잘 들어 맞지 않았다.
- Feature descriptor들은 low level visual cues 기반으로 사람이 직접 만들어야 했기 때문에 복잡한 문맥에서는 대표격이라고 할 수 있는 정보들을 잡아내는데 어려움이 있었다.
- 탐지 파이프라인의 각 단계들은 각자 따로 디자인되고 최적화 되었기 때문에 전체 시스템 관점에서의 최적화를 달성하기 어려웠다. 

깊은 컨볼루션 신경망이 이미지 분류에서 큰 성공을 일으키고 객체 탐지 영역에서도 딥러닝 기반의 기술들이 주목할만한 진보를 달성했다.  특히 앞선 전통적인 방법들을 월등히 능가했다.  

한가지의 이미지 분류를 위한 계층적이고 공간 불변의(이미지가 변형되어도 그 이미지로 인식하는 것 - e.g. 고양이 사진을 90도 만큼 회전해도 그 고양이 사진으로 인식) 모델을 구축하기 위한 시도는 Fukushima에 의해 제안된 neocognitron이다. 그러나 이 방법은 지도 학습을 위한 효율적인 최적화 기술이 부재했다. 

이 모델을 기반으로 하여 Lecun 등은 CNN을, 역전파를 통한 SGD(Stochastic gradient descent) 으로 최적화했고 숫자 인식에서 경쟁력 있는 성능을 보여줬다. 

그러나 DCNN은 그 후에 SVM에게 주도권을 내줬는데 그 이유로 다음과 같은 이유들이 있었다.

- 주석이 달린(레이블링 된) 데이터의 부족이 과적합을 일으킴.
- 제한적인 컴퓨팅 자원들(성능 부족).
- SVM와 비교했을 때 이론적 뒷받침이 부족.

2009년에 Jia 등이 획득한 ImageNet이라는 1.2M의 고화질의  주석이 달린 데이터를  대량으로 획득한 덕에 딥러닝 모델을 훈련하는 것이 가능해졌고 GPU 클러스터등의 병렬처리 컴퓨팅시스템의 발전했다. 이로 인해 2012년에 Krizhevsky 등은 ImageNet 데이터 셋으로 크고 깊은 CNN 모델을 훈련시킬 수 있었고 ILSVRC(Large Scale Visual Recognition Challenge)에서 다른 방법들과 비교했을때 상당한 개선 사항을 보여줬다. 

딥러닝(CNN) 기반의 기술이 기존의 방법들과 비교했을 때 가지는 장점은 다음과 같다.

- 훈련 데이터의 저차원 픽셀 단위부터 고차원 의미 정보까지의 계층적 특징 표현들을 자동으로 생성해낸다. 그리고 복잡한 문맥에서도 보다 차별적인 표현 역량을 보여준다. 
- 더 많은 데이터가 있을 때 기존의 방법들은 개선의 여지를 보여줄 수 없는데 반해 딥러닝 기반 기술들은 더 나은 특징 표현을 가능하게 한다.
- 시작부터 끝까지 종단간의 최적화를 가능하게 한다.

![](./Figure/Recent_Advances_in_Deep_Learning_for_Object_Detection_2.JPG)

현재의 딥러닝 기반 객체 탐지 프레임워크들은 크게 두 가지 범주로 나눌 수 있다.

- R-CNN(Region-based CNN)과 그 아종들과 같은 Two-stage 그룹 - 2단계 탐지기들은 제안 생성기를 통해 제안 세트들을 만들어 내고 각각의 제안들에서 특징들을 추출해낸다. 그리고 나서 제안된 지역들의 카테고리를 분류기를 통해 예측한다.  2단계 탐지기들은 보통 1단계 탐지기들보다 더 좋은 탐지 성능을 보인다.
- YOLO(You only look once)와 그 아종들과 같은 One-stage 그룹 - 1단계 탐지기들은 지역 분류 단계 없이 특징 맵의 각 지역들의 객체들에 대해 곧바로 분류 작업을 수행한다.  1단계 탐지기들은 시간적인 측면에서 효율적이므로 실시간 객체 탐지에 이용 가능하다.

![](./Figure/Recent_Advances_in_Deep_Learning_for_Object_Detection_3.JPG)

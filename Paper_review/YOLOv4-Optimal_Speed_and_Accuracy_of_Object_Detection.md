# YOLOv4: Optimal Speed and Accuracy of Object Detection

Alexey Bochkovskiy, 

Chien-Yao Wang(Institute of Information Science Academia Sinica, Taiwan),

Hong-Yuan Mark Liao(Institute of Information Science Academia Sinica, Taiwan)



## Abstract

CNN 모델의 정확도를 향상시키기위한 많은 방법들이 있지만, 대용량 데이터에서 이런 방법들의 조합을 테스트 해보는 것과 결과의 이론적 타당성이 필요하다. 실제로 몇가지 방법들은 특정 모델에서만 예외적으로 성능이 잘 나오거나 특정 문제에서만 성능이 잘 나오기도 하고 작은 용량의 데이터셋에서만 성능이 잘 나오기도 한다. 이와 반대로 Batch-nomalization, Residual-connection과 같은 몇가지 방법들은 다수의 모델이나 Task 그리고 데이터셋에서도 잘 적용될 수 있다. 저자들은 이런 기법들 중에 Weighted-Residual-Connection(WRC), Cross-Stage-Partial-connections(CSP), Cross mini-Batch Normalization(CmBN), Self-adversarial-training(SAT), Mish-activation에 관심을 보였다. 이 뿐만 아니라 Mosaic data augmentation, DropBlock regularization, CIoU loss 등에도 관심을 보였다. YOLOv4는 새로운 이론이나 기법을 연구하고 제시하기 보다는 YOLOv3에 여러 기법을 적용해서 성능을 알아본 Technical Report에 가깝다. 



## Introduction

많은 CNN 기반 Object Detection 모델들은 추천 시스템에 적용될 수 있다. 예를 들어서 비어 있는 주차장 자리를 찾는 모델은 느리지만 정확하다. 이에 반해 자동차 충돌을 경고하는 모델은 빠르지만 상대적으로 덜 정확하다. 실시간 Object Detection 모델의 정확도를 개선하게 되면 단지 힌트나 조언을 제공하는 수준에서 벗어나 인간의 개입을 줄이면서 독자적으로 어떤 프로세스를 관리하는 시스템에도 쓰일 수 있다. 통상적으로 GPU에서 실시간 Object Detector를 구동시키면 가용할만한 Cost로 많은 곳에서 모델을 쓸 수 있다. 그러나 많은 모델들이 실시간으로 동작하기는 어렵고 많은 미니 배치 사이즈로 훈련시키려면 많은 수의 GPU를 필요로 한다. 저자들은 GPU 하나에서 훈련시키고 실시간으로 동작할 수 있는 모델을 만들고자 했다. 

저자들이 말하는 이 연구의 주요 목표는 실제 Production system에서 빠르게 동작하는 Object Detection 모델을 디자인하고 병렬 컴퓨팅 연산을 최적화 하는 것이다. Figure 1과 같은 YOLO 모델을 한 대의 GPU로 훈련시킬 수 있는 방법을 고안하는 것이다. 

![](./Figure/YOLOv4-Optimal_Speed_and_Accuracy_of_Object_Detection1.png)

저자들이 말하는 이 연구의 기여하는 바는 다음과 같다. 

- 1080Ti나 2080Ti 하나의 GPU 정도만 사용해도 강력하고 효율적인 Object Detection 모델을 만드는 방법 제시
- Detection 모델을 훈련 시키는 동안 그 동안 알려져 있던 Bag-of-Freebies나 Bag-of-Specials 기법들의 영향력을 조사
- CBN, PAN, SAM과 같은 방법들을 단일 GPU에서 훈련시킬 수 있도록 변경



## Related work

### Object detection models

Object detection 네트워크는 보통 두 부분으로 나뉜다. 한 부분은 ImageNet에서 미리 훈련시킨 Backbone이고 다른 한 부분은 객체를 둘러싼 바운딩 박스와 관련된 값과 객체의 클래스를 예측하는 Head이다. 

- GPU platform backbone: VGG, ResNet, ResNeXt, DenseNet
- CPU platform backbone: SqueezeNet, MobileNet, ShuffleNet 

Head와 관련해서는 One-stage, Two-stage detector로 두 가지 범주로 나눌 수 있다. 가장 많이 알려져 있는 Two-stage detector는 R-CNN 계열로 Fast R-CNN, Faster R-CNN, R-FCN, Libra R-CNN을 포함한다. Anchor 없이 동작하는 Two-stage detector로는 RepPoints가 있다. One-stgae detector에서 유명한 것은 YOLO, SSD, RetinaNet이 있다. 또한 Anchor 없이 동작하는 One-stage detector로는 CenterNet, CornerNet, FCOS이 있다. 

Backbone과 Head 사이에 몇 가지 계층을 집어넣어 Detector를 발전시키기도 했다. 이 계층들의 역할은 주로 각기 다른 Stage에서 Feature map을 모으는 것이다. 저자들은 이 계층들을 Detector의 Neck이라고 부불렀다. Neck들은 주로 몇몇의 Bottom-up 경로와 Top-down 경로로 구성되었다. 이런 매커니즘이 결합된 네트워크로는 Feature Pyramid Network(FPN), Path Aggregation Network(PAN), BiFPN, NAS-FPN이 있다. 

위와 같은 관점 말고도 어떤 연구자들은 아예 Object detection을 위한 새로운 Backbone을 쌓는데 중점을 두거나(DetNet, DetNAS) 모델 전체를 새로 쌓는데 중점을 주었다(SpineNet, HitDector)

요약하자면 보통의 Object detector는 다음과 같이 구성되어 있다. 

![](./Figure/YOLOv4-Optimal_Speed_and_Accuracy_of_Object_Detection2.png)



### Bag of freebies

보통 Object detector 알고리즘은 실질적으로 사용되기 전에 구축되고 훈련된다. 그래서 연구자들은 ';' 훈련 방법으로 추론 Cost는 늘리지 않으면서 정확도가 높은 모델을 개발하려고 한다. 저자들은 이런 훈련 전략이나 훈련 Cost만 영향을 끼치는 방법을 Bag of freebies라고 불렀다. 이 Bog of freebies에 포함되는 기법 중 하나가 Data augmentation이다. Data augmentation은 입력 이미지의 변동성을 증가시켜서 모델이 다양한 환경에서 획득된 이미지에 대해 작업을 잘 수행할 수 있도록 하는 것이 목적이다. 

몇몇 연구자들은 객체의 Occlusion 문제를 다루기 위해서 Data augmentation을 사용했다. 예를 들어서 Random erase, CutOut, hide-and-seek, grid mask, Dropout, DropConnedt, DropBlock, MixUp, CutMix, style transfer GAN 등이 있다. 

또 어떤 연구에서는 Bag of freebies 기법들을 데이터셋에서 Semantic distribution적으로 편향을 보이는 문제에 적용하기도 했다. 예를 들어서 클래스 불균형 문제에서는 Hard negative example mining, Online hard example mining이 Two-stage object detection 계열에 적용되었다. 

그러나 이런 Example mining 기법들은 One-stage 계열에는 쓰일 수 없다. 왜냐하면 Two-stage 계열과는 달리 이미지 내 모든 지역들을 다 살펴보기 때문이다. Lin등은 Focal loss를 제안해서 데이터 클래스 간의 불균형 문제를 해결하고자 했다. 

다른 중요한 이슈는 각 카테고리 간의 연관 정도를 원-핫 방식으로 표현하기 어렵다는 것이다. 특히 레이블링을 수행할때 이런 방식이 자주 쓰인다. Label smoothing이나 Islam 등의 연구가 이를 다룬다. 

저자들이 소개하는 마지막 Bag of freebies는 Bounding box regression의 Objective function이다. 보통의 BBox regression의 경우 BBox의 중심 좌표, 너비, 높이 값을 직접 예측하거나 이 값들의 Offset을 regression으로 알아내는 방식으로 진행되는데 문제는 이 값들을 따로 따로 regression을 수행하기 때문에 객체 자체의 통합성에 대해서는 고려하지 않는다는 점이다. 이를 극복하기 위해서 몇 연구들은 IoU loss라는 개념을 제안했다. GIoU loss, DIoU, CIoU을 참고. 



### Bag of specials

저자들은 추론 Cost를 약간 증가시키면서 모델의 정확도를 상당히 개선시키는 Plugin 모듈이나 후처리 방법을 Bag of spcials라고 불렀다. 여기에는 모델 안의 어떤 속성을 강화시키는 모듈이 포함되는데 예를 들어 Receptive field의 크기를 키우거나 Attention mechanism을 도입하거나 Feature integration capability를 강화시키는 것들이 있다. Post-processing은 모델 예측 결과를 보여주는 것과 관련있는 방법이다. 

Receptive field를 키우는 것과 관련된 모듈은 SPP, ASPP, RFB가 있다. 

Object Detection에서 적용되는 Attention module은 크게 Channel-wise attention, Point-wise attention로 나눌 수 있는데 각각의 대표 기법으로는 Squeeze-and-Excitation(SE), Spatial Attention Module(SAM)이 있다. 

Feature integration에 관해서는 Skip connection, Hyper-column, FPN, SFAM, ASFF, BiFPN이 있다.

딥러닝 분야에서 어떤 연구자들은 좋은 성능을 내는데 도움이 되는 Activation function을 찾는데 초점을 뒀다. 여기에는 ReLU, LReLU, PReLU, ReLU6, Scaled Exponential Linear Unit(SELU), Swish, Hard-Swish, Mish가 있다. 

딥러닝 기반의 Object detection에서 주로 쓰이는 Post-processing 기법은 NMS이다. Girshick등이 제안한 Greedy NMS, Soft NMS, DIoU NMS이 있다. Anchor가 없는 Object detection에서는 Post-processing 기법이 적용되지 않는다.  

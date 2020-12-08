# Mask R-CNN

Kaiming He, Georgia Gkioxari, Piotr Dollar, Ross Girshick (Facebook AI Research (FAIR))



## Abstract

저자들은 Object detection을 통한 박스에서 Instance segmentation을 수행하는 프레임워크를 제안했다. Faster R-CNN을 확장한 개념인 Mask R-CNN은 바운딩 박스 예측 수행하는 브랜치에 병렬적인 브랜치를 하나 더 추가한다. 추가된 브랜치는 Object의 픽셀 mask를 예측하는 작업을 수행한다. 기존의 Faster R-CNN의 완전히 바꾸는 것이 아니라 Classification 브랜치와 Bounding box regression 브랜치를 이용하여 기능을 추가하는 개념이기 때문에 구현하기 쉽고 속도도 Faster R-CNN과 비교해서 큰 차이가 나지 않는다(5 FPS). 저자들은 Mask R-CNN으로 Human pose estimation도 수행하는 모델을 만들기도 했다. 저자들은 COCO 형식의 Challenge에서 Instance segmentation, Object detection, Person keypoint detection 부분에서 결과를 보여줬다. 

코드:  [facebookresearch - Detectron](https://github.com/facebookresearch/Detectron)



## Introduction

서론에서 밝히는 저자들의 연구 목적은 Instance segmentation을 위한 프레임워크를 개발하는 것이었다. Instance segmentation이 어려운 이유는 이미지 내의 객체를 정확하게 탐지해야 하며 동시에 각 객체를 구분할 수 있어야 한다(똑같은 클래스라도 다른 객체인 경우). 이를 위해서 Instance segmentation의 기본적인 접근 방법은 Object detection에서 객체를 분류하고 Bounding box로 위치를 표시하는 특징과 Semantic segmentation에서 각 객체 인스턴스에 상관 없이 카테고리별로 각 픽셀을 분류하는 특징을 결합하는 것이었다. 

![](./Figure/Mask_R-CNN1.JPG)

Mask R-CNN에서는 Fast R-CNN에서 각 ROI마다 기존의 Classification과 Bounding box regression을 수행하는 동시에 픽셀의 Segmentation masks 예측하는 작업을 병렬적으로 수행한다. Mask 브랜치는 FCN이며 Pixel-to-Pixel의 방식으로 Segmentation mask를 예측한다. 이 FCN의 사이즈가 작기 때문에 이 브랜치에서 추가적으로 발생하는 Overhead가 적다. 

원래 Faster R-CNN에서는 네트워크의 입력과 출력 간의 Pixel-to-Pixel alignment를 수행하지 않는다. 이것은 ROI Pooling이 Feature extraction을 위해서 Coarse spatial quantization을 어떻게 수행하지를 보면 명확해진다. Instance segmentation에 맞지 않는 이런 연산 방식을 대체하기 위해서 저자들은 ROIAlign이라고 하는 정확한 Spatial location을 보존하고 Quantization이 없는 연산을 소개했다. ROIAlign으로 인해서 Mask accuracy가 결과적으로 10%에서 50%까지 개선되었다. 또 저자들은 Mask에 대해서 Multi class 중에 하나를 고르는 방식이 아니라 각 Class가 있는지 없는지를 예측하는 Binary mask를 예측했다. 이게 가능한 이유는 ROI의 Classification 브랜치에서 이미 객체에 대한 클래스 예측을 수행하기 때문이다. FCN에서는 Segmentation과 Classification을 동시에 수행했는데 저자들은 브랜치를 나눠서 Segmentation과 Classification을 분리했다. 

![](./Figure/Mask_R-CNN2.JPG)

[Taeoh Kim - PR-057: Mask R-CNN](https://www.youtube.com/watch?v=RtSZALC9DlU)

Mask R-CNN으로 개발한 모델은 GPU에서 하나의 프레임을 처리하는데 200ms가 걸렸고 8-GPU로 COCO 데이터셋으로 훈련시키는데 하루에서 이틀 정도가 걸렸다고 한다. 



## Related Work

### R-CNN

R-CNN 계열 알고리즘은 객체 탐지를 위한 Bounding box를 위해서 지역 후보(ROI)를 생성하고 각 ROI마다 Convolutional network 작업을 수행한다. ROI Pooling 계층을 통해서 속도와 정확도가 개선되었다. Faster R-CNN은 FCN인 RPN을 학습 시켜서 ROI를 생성하는 Attention mechanism으로 Fast R-CNN을 개선시켰다. 



### Instance Segmentation

본문 참조
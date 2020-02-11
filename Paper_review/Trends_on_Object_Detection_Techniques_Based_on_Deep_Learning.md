# 딥러닝 기반 객체 인식 기술 동향

## Trends on Object Detection Techniques Based on Deep Learning



이진수(지식이러닝연구그룹/UST 석사과정), 이상광(지식이러닝연구그룹 책임연구원), 

김대욱(지식이러닝연구그룹 연구원), 홍승진(홍익대학교 게임학부 석사과정)

양성일(지식이러닝연구그룹 책임연구원/PL)



DOI: 10.22648/ETRI. 2018. J. 330403







## 0. 요약









## 1. 서론

최근 객체 인식 분야는 여러 산업 전반에서 핵심기술로 사용되고 있다. 이에 부응하여 PASCAL, ImageNet, MS COCO 등 객체 인식 관련 대회를 개최하고 있다.

과거의 객체 인식 연구(SIFT-Scale Invariant Feature Transform, SURF-Speede-Up Robust Features, Haar, HOG-Histogram of Oriented Gradients)에서는 객체가 가지는 특징을 설계하고 검출해서 객체를 찾아내는 방식으로 진행되었다.

DPM(Deformable Part-based Model)에서는 물체를 여러 부분으로 나누어 특징 정보를 구성하고, 각 부분의 유동적인 구조를 SVM(Support Vector Machine) 등의 방법으로 연결하여 객체 인식 성능을 높였다.

합성곱 신경망(CNN-Convolutional Neural Network)이 ImageNet 2012 대회에서 압도적인 성능을 보여줬고 CNN의 인식률을 향상시키기 위한 연구에서 ZFNet, VGG, ResNet, GoogLeNet, DenseNet 등이 등장하였다.

CNN을 통해 객체 분류 문제는 어느정도 성공을 거뒀으나, 영상에서 객체의 위치를 검출하는 방법에 대한 연구가 후속적으로 등장했다.

R-CNN(Region-based Convolutional Neural Networks)는 딥러닝 회귀(Regression)방법으로 이 문제를 해결했다.

Fast R-CNN은 R-CNN의 느린 검출 속도를 보완했으나 R-CNN과 마찬가지로 딥러닝 망에서 객체의 후보 영역을 찾을 수 없어서 속도가 객체 후보 영역을 찾는 부분에 의존한다는 단점이 있었다.

Faster R-CNN에서는 객체 후보 영역을 딥러닝 네트워크 속에 편입시켜 이 문제를 해결함으로서 검출 속도를 향상시켰다. 

R-FCN은 Faster R-CNN이 영상의 지역적 정보에 의존한다는 단점을 보완하기 위해 등장했다.

이러한 발전으로 객체 인식 속도가 개선되었으나 실시간 처리 작업, 자율 주행등의 요구사항에 적용하는데는 충분하지 못했다.

YOLO(You Only Look Once)는 이러한 속도 문제를 해결하기 위해 객체 인식의 모든 과정을 하나의 딥러닝 네트워크로 구성하는 방법을 제안했다.

SSD(Single Shot MultiBox Detector)은 모바일에서도 동작이 가능한 정도의 빠른 검출 속도를 보인다.

Image Segmentation 분야에서는 단순히 객체의 영역 박스를 찾는 수준이 아니라 객체의 픽셀 영역을 찾는 연구를 진행하고 있다.

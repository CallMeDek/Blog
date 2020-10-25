# Focal Loss for Dense Object Detection

Tsung-Yi Lin (Facebook AI Research (FAIR))

Priya Goyal (Facebook AI Research (FAIR))

Ross Girshick (Facebook AI Research (FAIR))

Kaiming He (Facebook AI Research (FAIR))

Piotr Doll´ar(Facebook AI Research (FAIR))



## Abstract

이 연구가 수행될 당시 가장 높은 정확도를 보이는 Object Detector는 R-CNN에 의해서 유명해진 Two-stage Detector이다. Two-stage Detector의 경우 이미지 안의 Object가 있을법한 위치의 후보 지역을 추려서 Classifier로 프로세스를 수행한다. 이와 반대로 One-stage Detector는 가능한 모든 지역을 꼼꼼하게 살펴보기 때문에 더 빠르고 정확하지만 Two-stage에 비해 정확도가 높지 않았다. 저자들은 이 연구에서 그 이유를 분석했다. 그래서 발견한 주요 이유는 One-stage Detector는 가능한 모든 지역을 살펴보기 때문에 훈련 과정 중에 Foreground-Background간에 클래스 불균형을 피할수 없기 때문이었다. 저자들은 이런 클래스 불균형 문제를 해결하기 위해서 표준 크로스 엔트로피 손실 계산 방법을 바꿔서 이미 잘 분류된 예제들(높은 확률로 Background라고 분류된 대부분의 Negative sample들)이 손실 계산에서 차지하는 비중을 줄일 수 있도록 했다. 즉, Background라고 분류하기 어려운 Negative들에 더 비중을 두고 많은 수를 차지하는 대부분의 쉬운 Negative가 손실에서 다른 손실을 압도하는 것을 막는다. 이 손실 방법을 검증하기 위해서 저자들은 간단한 Dense Detector를 디자인 하고 훈련시켰는데 저자들이 이 Dense Detector를 RetinaNet이라고 부른다. RetinaNet을 Focal Loss로 훈련시키면 One-stage의 속도를 보이면서 기존의 Two-stage Detector들의 정확도를 훨씬 상회한다고 한다. 



## Introduction

Two-stage Detector에서 첫번째 단계에서는 이미지 내에 객체가 있르법한 지역 후보를 추린다. 그리고 두 번째 단계에서는 CNN으로 각 지역 후보가 Foreground 클래스 중 하나인지 Background 클래스인지를 예측한다. Two-stage Detector가 COCO benchmark에서 SOTA를 차지 했었지만 저자들은 이와 반대로 One-stage Detector가 이와 비슷한 정확도를 가질 수 없는지 생각했다. One stage Detector들은 다양한 스케일과 종횡비를 가지는 객체의 각 위치에 적용되었다. 결론적으로 저자들은 One-stage Object Detector면서 FPN, Mask R-CNN, Faster R-CNN 계열의 알고리즘의 COCO AP 수준을 보이는 Detector를 만들어냈다고 주장한다.  저자들이 주목한 난제는 훈련 간에 One stage Detector에 내재되어 있는 클래스 불균형 문제이다. 

Two-stage Detector에서는 클래스 불균형 문제를 Two-stage Cascade와 Sampling Heuristics로 해결하려고 한다. 지역 후보 생성시에는 Selective Search, EdgeBoxes, DeepMask, RPN과 같은 방법을 적용하여 많은 지역 후보를 빠르게 적은 수로 줄인다. 이 과정에서 대부분의 Background Sample들은 거른다. 그 다음 분류 단계에서는 Foreground와 Background의 비율을 1:3으로 하는 등의 Sampling Heuristics나 Online Hard Example Mining(OHEM)이 Foreground와 Background 사이의 균형을 유지하기 위해서 수행된다. 

반대로 One-stage Detector들은 전 이미지에 걸쳐서 객체 후보가 될만한 지역을 처리한다. 실제로는 다양한 스케일과 종횡비를 고려해야 하기 때문에 100k의 지역을 처리해야 하기도 한다. Sampling Heuristics가 적용될 수는 있으나 쉬운 Background 샘플에 압도되어 비효율적이다. 이런 비효율성은 Bootstrapping이나 Hard Example Mining 같은 기술로 처리되기도 한다. 

![](./Figure/Focal_Loss_for_Dense_Object_Detection1.JPG) 

이 연구에서는 새로운 손실 함수를 정의해서 기존과 다른 방식으로 클래스 불균형 문제를 해결하고자 했다. Figure 1을 보면 쉽게 Background로 판단되는 Example들의 확률이 높아질수록 Scaling Factor가 0으로 수렴하여 결국 손실이 거의 0이 되는 것을 확인할 수 있다(γ = 2일때 Well-classified example들의 Loss가 거의 0이 되는 것을 확인할 수 있다). 이렇게 Scaling Factor로 인해 자연스럽게 Background로 판단하기 어려운 Example들에 비중을 더 두게 된다. 저자들이 말하길 저자들이 정의한 Focal Loss가 꼭 위와 같을 필요는 없다고 한다. 

저자들은 Focal Loss의 효율성을 입증하기 위해서 RetinaNet이라고 하는 간단한 One Stage Object Detector를 디자인했는데 네트워크 안에서 Feature Pyramid를 적용했고 Anchor Box 개념을 도입했다. ResNet-101-FPN Backbone을 사용한 RetinaNet 모델은 COCO test-dev 셋에서 AP 39.1, 5 Fps의 성능을 달성했다고 한다. 

 ![](./Figure/Focal_Loss_for_Dense_Object_Detection2.JPG)

# Focal Loss for Dense Object Detection

Tsung-Yi Lin (Facebook AI Research (FAIR))

Priya Goyal (Facebook AI Research (FAIR))

Ross Girshick (Facebook AI Research (FAIR))

Kaiming He (Facebook AI Research (FAIR))

Piotr Doll´ar(Facebook AI Research (FAIR))



## Abstract

이 연구가 수행될 당시 가장 높은 정확도를 보이는 Object Detector는 R-CNN에 의해서 유명해진 Two-stage Detector이다. Two-stage Detector의 경우 이미지 안의 Object가 있을법한 위치의 후보 지역을 추려서 Classifier로 프로세스를 수행한다. 이와 반대로 One-stage Detector는 가능한 모든 지역을 꼼꼼하게 살펴보기 때문에 더 빠르고 정확하지만 Two-stage에 비해 정확도가 높지 않았다. 저자들은 이 연구에서 그 이유를 분석했다. 그래서 발견한 주요 이유는 One-stage Detector는 가능한 모든 지역을 살펴보기 때문에 훈련 과정 중에 Foreground-Background간에 클래스 불균형을 피할수 없기 때문이었다. 저자들은 이런 클래스 불균형 문제를 해결하기 위해서 표준 크로스 엔트로피 손실 계산 방법을 바꿔서 이미 잘 분류된 예제들(높은 확률로 Background라고 분류된 대부분의 Negative sample들)이 손실 계산에서 차지하는 비중을 줄일 수 있도록 했다. 즉, Background라고 분류하기 어려운 Negative들에 더 비중을 두고 많은 수를 차지하는 대부분의 쉬운 Negative가 손실에서 다른 손실을 압도하는 것을 막는다. 이 손실 방법을 검증하기 위해서 저자들은 간단한 Dense Detector를 디자인 하고 훈련시켰는데 저자들이 이 Dense Detector를 RetinaNet이라고 부른다. RetinaNet을 Focal Loss로 훈련시키면 One-stage의 속도를 보이면서 기존의 Two-stage Detector들의 정확도를 훨씬 상회한다고 한다. 
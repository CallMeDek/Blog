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
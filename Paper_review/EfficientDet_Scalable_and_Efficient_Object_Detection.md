# EfficientDet: Scalable and Efficient Object Detection

Mingxing Tan,  Ruoming Pang, Quoc V. Le(Google Research, Brain Team)



## Abstract

저자들은 모델의 효율성을 개선하기 위한 방법을 본 논문에서 제안했다. 

- 첫번째로, 다양한 크기의 Feature들을 합치는 방법인 Bi-directional feature pyramid network(BiFPN)을 제안했다. 
- 두번째로, 네트워크의 Backbone, Feature network, Box/class prediction network의 Resolution, Depth, Width를 동시에 Scaling하는 Compound scaling method를 제안했다. 

이를 바탕으로 저자들은 EfficientDet이라고 하는, 다양한 리소스 환경에서 구동이 가능하면서 비슷한 성능을 낼 수 있는 Detection 알고리즘을 만들었다. 

[google-automl-efficientdet](https://github.com/google/automl/tree/master/efficientdet)



## Introduction

Object detection 분야에서 상당히 많은 발전이 이뤄졌지만 그만큼 Cost가 비싸졌다. 한 예로 AmoebaNet 기반의 NAS-FPN detector의 경우는 167M개의 파라미터와 3045 BFLOPS를 필요로 한다. 그런데 이렇게 크고 비싼 모델은 로보틱스나 자율주행차와 같은, 모델 사이즈와 Latency 때문에 상당히 제한적인 환경의 Application에는 맞지 않는다. 그래서 이런 환경에 맞는, 효율성에 초점을 둔 One-stage, Anchor-free detection 알고리즘을 만들어내려는 노력들이 많았다. 그런데 이 알고리즘들은 대게 효율성은 높지만 정확도는 희생을 감수하는 경우가 많다. 게다가 기존의 연구들은 모델이 돌아가는 환경이 구체적으로 정해져 있다고 보는 경우가 많은데 실제로는 Mobile device부터 Datacenter까지 각기 다른 환경에서 구동되는 경우가 있다. 

저자들은 그래서 각기 다른 환경에서 구동시킬 수 있으면서 높은 정확도와 효율성을 가지는 모델을 구축할 수 있는 방법이 없을까를 고민했다. 그래서 저자들은 Detection 아키텍처의 다양한 디자인 옵션을 알아봤다. 그 결과 두 가지 해결해야할 사항을 확인했다. 

- Efficient multi-scale feature fusion: FPN은 다양한 크기의 Feature들을 합치는데 많이 쓰이고 있다. PANet, NAS-FPN 같은 연구에서는 Cross-scale Feature fusion을 위한 네트워크 구조를 개발하기도 했다. 이런 연구들은 Feature들을 구분 없이 단순히 더해버린다. 하지만 저자들이 관찰한 결과 이 Feature의 Resolution들이 같지 않기 때문에 합쳐진 Output에 기여하는 바가 다르다고 한다. 그래서 저자들은 Top-down과 Bottom-up의 Multi-scale feature fusion이 반복되는 구조에서 각 Feature들의 중요도를 결정하는 Weight들을 학습이 가능하도록 만드는 BiFPN을 제안했다. 
- Model scaling: 보통 많은 연구들이 주로 더 높은 정확도를 위해서 더 큰 Backbone 네트워크에 의존하거나 더 큰 입력 이미지 사이즈에 의존하는데, 저자들이 관찰한 바에 의하면 정확도와 효율성을 동시에 고려했을때 Feature network와 Box/class prediction network를 Scaliing 하는 것도 중요하다고 한다. 그래서 저자들은 Backbone, Feature network, Box/class prediction network의 Resolution, Depth, Width를 동시에 Scaling 하는 Compound scaling method를 고안해냈다. 
- 저자들은 다른 Backbone보다 EfficientNet의 성능이 좋은 것을 확인했다. 그래서 저자들이 제안한 BiFPN과 Compound scaling method를 결합해서 EfficientDet를 만들어냈다. 저자들이 말하는 EfficientDet은 훨씬 적은 모델 파라미터와 FLOPs로 준수한 정확도를 낼 수 있는 알고리즘이다. 

![](./Figure/EfficientDet_Scalable_and_Efficient_Object_Detection1.JPG)



## Related Work

### One-stage Detectors

저자들은 One-stage의 기조를 따르면서 최적화된 Network 아키텍처로 효율성과 정확도를 높일 수 있다고 한다. 



### Multi-Scale Feature Representations

Object detection에서 가장 큰 어려움 중 하나는 효율적으로 다양한 크기의 Feature들을 처리하는 방법이다. 초기의 Detector들은 종종 Backbone network에서 출력된 Pyramidal feature hierarchy에서 예측을 수행하곤 했다. FPN은 이 Feature들을 결합하기 위한 Top-down pathway를 제안하기도 했다. PANet은 추가적인 Bottom-up path aggregation network를 FPN의 위에 더했다.  STDL은 Cross-scale feature들을 이용하기 위한 Scale-transfer module을 제안하기도 했다. M2det은 U모양의 module로 다양한 크기의 Feature들을 Fuse했다. G-FRNet은 Gate unit으로 Feature에 대해서 Information flow를 조절하는 방법을 소개했다. NAS-FPN은 Feature network topology를 자동으로 디자인하는 Neural architecture search에 의존한다. 성능은 좋긴하지만 Search 시간이 매우 길고 Network 모양이 불규칙적이기 때문에 해석하기 어렵다. 



### Model Scaling

보통 더 나은 정확도를 위해서 쓰는 방법은 더 큰 Backbone 네트워크를 사용하는 것이다. 그런데 몇몇 연구에서 채널 크기를 키우는것과 Feature network 구조를 반복하는 것이 정확도를 향상시킨다는 것을  발견했다. 이런 방법들은 하나나 몇가지 안되는 요소를 조절하는 것에 초점을 맞춘다. 그런데 저자들은 EfficientNet 연구에서 이런 요소들을 동시에 Scaling하는 방법이 모델의 효율성을 괄목할만하게 개선한다는 것을 입증했다. 그래서 저자들은 EfficientNet의 Compound scaling method 방법을 본 연구에서도 적용했다. 
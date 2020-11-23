# MobileNetV2: Inverted Residuals and Linear Bottlenecks

Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen(Google Inc.)



## Abstract

저자들은 MobileNetV2라는 모바일에 특화된 새로운 아키텍처를 발표했다. 그리고 Object Detection에서의 성능을 보여주기 위해서 MobileNetV2를 SSD에 적용한 새로운 프레임워크인 SSDLite를 발표했다. 또 모바일용 Semantic segmentation에서 성능을 보여주기 위해서 본래의 DeepLabv3보다 용량이 줄어든 형태인 Mobile DeepLabv3를 발표했다. MobileNetV2는 ResNet에서, Pointwise convolution 후에 Regular convolution 수행하고 다시 Pointwise convolution을 수행하면서 첫 번째 Pointwise convolution의 입력 Feature map과 두 번째 Pointwise convolution의 출력 Feature map을 더하는 Bottleneck 구조에 기초하고 있다(이때 Bottleneck 또한 ResNet에서 Bottleneck이 아니라 MobileNetV1의 Width multiplier로 각 계층의 입력과 출력 채널 수가 줄어든 형태이다). 특히 Bottleneck 구조에서 Regular convolution이 아니라 가벼운 Depthwise convolution을 적용했다. 연구 하면서 저자들이 알아낸 사항 중 하나는 이렇게 얇아진 계층에서는 비선형성을 제거하는 것이 중요한데 그 이유는 이미 얇아진 상태에서 비선형성으로 정보가 손실되면 모델의 표현이 더 줄어들기 때문이다.  저자들의 연구 접근 방식으로 비선형성과 모델 표현력의 상관 관계를 구분해서 살펴볼 수 있게되었다.



## Introduction

신경망 구조가 이미지 인식 분야에서 놀라운 성능을 보여주긴 했지만 아주 많은 양의 리소스나 시간을 필요로 하므로 모바일과 임베디드 시스템 같이 제한적인 환경에서는 적용하기 어려웠다. 본 연구 목적은 이런 환경에서 정확도는 유지하면서 연산량을 줄여서 신경망 네트워크를 적용할 수 있게 하는 것이었다. 이 연구 논문의 핵심은 Inverted residual with linear bottleneck이다. 저차원으로 압축되어 있는 입력 특징을 먼저 높은 차원으로 확장 시키고 Lightweight depthwise convolution 연산을 수행한다. 그런다음에 Linear convolution으로 고차원을 다시 저차원으로 Projection시킨다. 이 연구에서 제안한 모델은 쉽게 구현 가능하고 연산 중간에 쓰는 많은 텐서 양을 줄여서 메모리의 부하를 줄인다. 또 소프트웨어적으로 캐시 메모리를 사용해서 메인 메모리에 많은 접근이 일어나지 않도록한다. 



## Related Work

정확도와 속도 사이의 최적의 균형을 이루기 위한 깊은 신경망 네트워크에 대한 연구는 활발히 이루어지고 있다. 아키텍처 자체에 대한 연구, 훈련 알고리즘 개선에 대한 연구가 이뤄져서 초기 신경망 디자인에 큰 개선을 가져왔다. 대표적으로 AlexNet, VGGNet, GooLeNet, ResNet이 있다. 또 하이퍼 파라미터 최적화 라던지 다양한 네트워크 Pruning 방법들, Connectivity learning에 대한 연구도 이뤄졌다. 또 네트워크 내부의 Convolution 블럭들을 연결하는 구조에 대한 연구도 많이 수행되었다. Genetic 알고리즘, 강화 학습을 아키텍처 연구에 적용하기도 했다. 그런데 결론적으로 성능이 좋은 아키텍처는 굉장히 복잡하다는 단점이 있다. 그래서 이 연구에서는 어떻게 하면 신경망을 간단하게 디자인할 수 있을까에 관심을 두었다. 또, 이 연구는 MobileNetV1의 후행연구이다.  


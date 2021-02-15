# MnasNet: Platform-Aware Neural Architecture Search for Mobile

Mingxing Tan(Google Brain), Bo Chen(Google Inc.), Ruoming Pang(Google Brain), Vijay Vasudevan(Google Brain), Mark Sandler(Google Inc.), Andrew Howard(Google Inc.), Quoc V. Le(Google Brain)



## Abstract

Mobile을 위한 CNN을 디자인하는 일은 어렵다. 왜냐하면 용량이 작으면서 빨라야하고 그러면서 정확해야하기 때문이다. 많은 아키텍처 옵션이 있을때 속도와 정확도 사이의 Tradeoff를 수동적으로 결정하는 것이 어렵다. 이 연구에서 저자들은 자동으로 모바일을 위한 신경망 네트워크 아키텍처 탐색 방법을 제안한다(Mobile Neural Architecture Search, 이하 MNAS). 이때 모델의 Latency를 명시적으로 모델 성능 향상의 주요 목표에 포함시켜서 정확도와 Latency 사이의 좋은 Trade-off를 달성하는 모델을 찾게 된다. 이 연구 이전에는 Latency가 다른 항목으로 간접적으로 대체되어 평가되었다(FLOPs 등). 그러나 여기에서는 모델을 직접 디바이스에서 구동시켜서 실제 Inference Latency를 측정하게 된다. 또, 모델의 유연성과 탐색 공간 사이의 균형점을 찾기 위해서 저자들은 Factorized hierarchical search space라는 개념을 제안한다. 이 개념을 통해서 네트워크의 각 계층마다 탐색을 수행하므로 계층의 다양성을 확보하게 된다. Latency를 측정한 디바이스 모델은 Pixel phone이다. 

[tensorflow-tpu-models-offical-mnasnet](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet)



## Introduction

CNN 모델이 점점 깊어지고 넓어지며 용량이 커지면서 속도가 더 느려지고 많은 계산량을 필요로 한다. 이런 한계점은 Resource가 제한적인 플랫폼, 예를 들어 모바일이나 임베디드 시스템에서 최신 모델을 배포하는 것을 어렵게 한다. 이런 문제점을 극복하기 위해서 네트워크의 깊이를 줄이거나 Depthwise 컨볼루션 혹은 Group 컨볼루션 같이 상대적으로 계산량이 더 적은 연산을 활용하는 방법으로 CNN 모델을 개선시키거나 디자인하는 연구들이 수행되었다. 그러나 Resource 제한적인 모바일 환경에서 모델을 디자인하는 것은 여전히 어려운 일인데 정확도와 Resource 사용 효율성 사이의 Trade-off를 결정하는 과정에서 아주 큰 디자인 공간을 탐색해야 하기 때문이다. 
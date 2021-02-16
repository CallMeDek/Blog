# MnasNet: Platform-Aware Neural Architecture Search for Mobile

Mingxing Tan(Google Brain), Bo Chen(Google Inc.), Ruoming Pang(Google Brain), Vijay Vasudevan(Google Brain), Mark Sandler(Google Inc.), Andrew Howard(Google Inc.), Quoc V. Le(Google Brain)



## Abstract

Mobile을 위한 CNN을 디자인하는 일은 어렵다. 왜냐하면 용량이 작으면서 빨라야하고 그러면서 정확해야하기 때문이다. 많은 아키텍처 옵션이 있을때 속도와 정확도 사이의 Tradeoff를 수동적으로 결정하는 것이 어렵다. 이 연구에서 저자들은 자동으로 모바일을 위한 신경망 네트워크 아키텍처 탐색 방법을 제안한다(Mobile Neural Architecture Search, 이하 MNAS). 이때 모델의 Latency를 명시적으로 모델 성능 향상의 주요 목표에 포함시켜서 정확도와 Latency 사이의 좋은 Trade-off를 달성하는 모델을 찾게 된다. 이 연구 이전에는 Latency가 다른 항목으로 간접적으로 대체되어 평가되었다(FLOPs 등). 그러나 여기에서는 모델을 직접 디바이스에서 구동시켜서 실제 Inference Latency를 측정하게 된다. 또, 모델의 유연성과 탐색 공간 사이의 균형점을 찾기 위해서 저자들은 Factorized hierarchical search space라는 개념을 제안한다. 이 개념을 통해서 네트워크의 각 계층마다 탐색을 수행하므로 계층의 다양성을 확보하게 된다. Latency를 측정한 디바이스 모델은 Pixel phone이다. 

[tensorflow-tpu-models-offical-mnasnet](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet)



## Introduction

CNN 모델이 점점 깊어지고 넓어지며 용량이 커지면서 속도가 더 느려지고 많은 계산량을 필요로 한다. 이런 한계점은 Resource가 제한적인 플랫폼, 예를 들어 모바일이나 임베디드 시스템에서 최신 모델을 배포하는 것을 어렵게 한다. 이런 문제점을 극복하기 위해서 네트워크의 깊이를 줄이거나 Depthwise 컨볼루션 혹은 Group 컨볼루션 같이 상대적으로 계산량이 더 적은 연산을 활용하는 방법으로 CNN 모델을 개선시키거나 디자인하는 연구들이 수행되었다. 그러나 Resource 제한적인 모바일 환경에서 모델을 디자인하는 것은 여전히 어려운 일인데 정확도와 Resource 사용 효율성 사이의 Trade-off를 결정하는 과정에서 아주 큰 디자인 공간을 탐색해야 하기 때문이다. 

초록에서 언급한대로 저자들은 모바일 CNN 모델을 디자인하기 위해 NAS를 활용하는, 자동화된 방법을 제안했다. 

![](./Figure/MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile1.png)

Figure 1은 저자들이 제안한 방법을 나타낸 것이다. 기존 연구와 차이라고 한다면 실제 모바일 Latency를 고려한 Multi-objective reward라는 점과 Search space가 다르다는 점이다. 저자들이 문제를 해결하려는 접근 방식은 두가지 주요 아이디어에 기초한다. 

- 아키텍처 디자인 문제를 CNN 모델의 정확도와 추론 Latency를 모두 고려하는 Multi-objective 최적화 문제로 정형화 한다. FLOPs를 Inference latency의 Proxy로 설정한 기존 연구와는 다르게 저자들은 실제 디바이스에서 모델을 구동했을때 Latency를 측정했다. 이는 FLOPs가 정확한 Proxy가 될 수 없다는 저자들의 관찰에 기반한다(예를 들어 MobileNet, NASNet은 비슷한 FLOPs를 보이지만 Latency는 상당히 다르다. Table 1참고).
-  두번째로 기존 연구와는 Search space가 다르다는 점이다. 기존 연구에서는 몇가지 구조의 Cell을 네트워크 전체에 걸쳐 몇번 반복해서 쌓는 방식으로 Pyramid network를 구성한다. 이는 Search 과정을 단순화 할 수는 있으나, 계산적 효율성을 위해서 중요한 계층의 다양성을 저해한다. 이 문제를 해결하기 위해서 저자들은 Factorized hierachical search space라는 개념을 제안했다. 이 개념으로 각 계층은 구조적으로 다를 수 있지만 네트워크의 유연성과 Search space 크기 간의 균형을 잘 잡을 수 있다. 

저자들은 저자들이 제안한 방식을 ImageNet classification과 COCO object detection에서 적용해봤다. 

![](./Figure/MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile2.png)



저자들이 주장하는 이 연구의 기여점을 요약하자면 다음과 같다. 

- 정확도와 모바일에서의 실제 Latency를 고려하는 Multi-objective NAS 최적화 개념을 도입.
- 네트워크의 유연성과 Search space size 간의 균형점을 잘 조절 가능하게 하도록 계층의 다양성을 보장하는 Factorized hierachical search space를 도입.
- 보통의 디바이스 환경(Pixel phone)에서 ImageNet classification과 COCO object detection의 SOTA 수준 성능을 입증. 



## Related Work

CNN 모델의 Resource 효율성을 개선하기 위한 접근 방식으로는 다음과 같은 것들이 있다. 

- Weight 값들이나/과 CNN 모델의 Baseline의 Activation을 낮은 Bit 표현으로 양자화한다. 
- FLOPs 혹은 Latency 같은 Platform-aware 평가 척도에 근거하여 덜 중요한 필터들을 잘라낸다.

그러나 이런 방법들은 Baseline model에 의존적이고 CNN operation의 Novel composition에 대해서 학습하는 것에 집중하지 않는다. 

다른 접근 방식으로는 직접적으로, 수동으로 모바일에서 효율적인 모바일 아키텍처를 고안해내는 방식이 있다. SqueezeNet은 1x1 컨볼루션을 쓰고 필터 크기를 줄여서 모델의 연산량과 파라미터 수를 줄였다. MobileNet은 연산 밀도를 최소화 하기위해서 집중적으로 Depthwise separable 컨볼루션을 썼다. ShuffleNet에서는 Group 컨볼루션과 Channel shuffle을 활용했다. Condensenet은 계층 간의 Group 컨볼루션을 연결하는 법을 학습시킨다. MobileNetV2는 Inverted residual와 Linear bottleneck을 사용했다. 그러나 아주 방대한 크기의 Search space를 생각한다면 이런 수동적으로 구축하는 모델들은 엄청난 인간의 노동력을 필요로 할 것이다. 

그리고 NAS를 활용해서 모델 디자인 과정을 자동화하는 연구 시도들이 있다. 이런 접근 방식들은 주로 강화학습, Evolutionary search, Differentiable search 등에 근거한다. 이런 방식들이 몇 가지 탐색된 Cell 구조를 반복적으로 쌓아 모바일에 맞는 크기의 모델들을 만들어낼수는 있어도 저자들이 주장하길 이런 방법들은 Mobile platform의 제약 사항들을 Search process 혹은 Search space에 포함시키지는 못한다고 한다. 저자들의 방법들과 상당히 가까운 MONAS, DPP-Net, RNAS, Pareto-NASH도 CNN 아키텍처를 탐색하는  동안 모델 크기나 정확도와 같이 Multiple objective를 최적화 하려는 시도를 했지만 저자들이 말하길 이들의 Search process는 CIFAR 같은 크기가 작은 Task에서나 최적화를 시도했다고 한다. 이와 반대로 저자들이 주장하는 점은 이 연구가 실제 모바일 Latency 제약 사항을 Target으로 하고 비교적 더 큰 Task에 집중했다는 것이다.   





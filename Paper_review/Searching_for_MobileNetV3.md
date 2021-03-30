# Searching for MobileNetV3

Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen (Google AI), Mingxing Tan (Google Brain), Weijun Wang, Yukun Zhu (Google AI), Ruoming Pang, Vijay Vasudevan, Quoc V. Le (Google Brain), Hartwig Adam (Google AI)



## Abstract

이 연구에서 저자들은 Complementary 탐색 기술과 참신한 아키텍처 디자인을 조합해서 새로운 MobileNet 모델을 설계하고자 했다. MobileNetV3는 NetAdapt 알고리즘에 의해 보충되는 하드웨어 인식 NAS에 의해서 모바일 CPU에 맞게 조정되었고 그동안 없었던 아키텍처 디자인을 통해 상당히 개선되었다고 한다. 저자들의 고민은 어떻게 (사람이 하는 것이 아니고) 자동으로 탐색을 하는 알고리즘과 네트워크 디자인을 잘 조합해서, Complementary 접근 방식을 활용해서 좋은 성능을 낼 것인가 하는 데에서 시작했다. 저자들은 그래서 두 가지 버전의 MobileNet 모델을 만들었다. MobileNetV3-Large와 MobileNetV3-Small인데 모델 적용 사례가 리소스를 많이 사용할 수 있는가 없는 가에 따라 각각을 사용한다고 한다. 이 모델들은 Object Detection, Semantic Segmentation 등의 수행하고자 하는 작업에 많게 적용된다. Semantic segmentation(혹은 Dense pixel prediction)의 경우, 저자들은 새로우면서 효율적인 Segmentation decorder인 Lite Reduced Atrous Spatial Pyramid Pooling(LR-ASPP)을 제안했다. 



## Introduction

신경망 기반의 모델이 점차 많이 쓰이면서 데스크탑을 통해 서버와 통신하는 구조가 아닌 모바일을 통해서 언제 어디서든 사용할 수 있는 환경으로 바뀌고 있다. 그러면서 높은 정확도와 낮은 지연율이 중요해지고 연산량을 줄여서 배터리 수명을 늘리는게 중요해졌다. 저자들은 두 가지 버전의 MobileNet 모델을 설계해서 이런 환경에서 좋은 성능을 보이게 하는 것이 목표였다. 이 목표를 이루기 위해서 자동화된 탐색 알고리즘과 아키텍처에서의 진보를 어떻게 결합하여 효율적인 모델을 구축할 것인가를 고민했다. 모바일 환경에서 정확도와 지연율의 Trade off를 최적화하는 컴퓨터 비전 아키텍처를 개발하기 위해서 저자들은 다음 개념을 도입했다. 

- Complementary 탐색 기술
- 모바일 환경에서 실제적으로 활용 가능한 새로운 버전의 비선형성 함수들
- 새로운 네트워크 디자인
- 새로운 Segmentation decoder



## Related Work

이 연구와 관련 있는 여러 선행 연구들은 본문 참고. 



## Efficient Mobile Building Blocks

MobileNetV1은 컨볼루션 대신 Depthwise separable 컨볼루션을 도입했다. Depthwise separable 컨볼루션 Spatial filtering과 Feature generation 매커니즘을 분리했다. 그래서 두 개의 계층으로 나눠지는데 하나는 Spatial filtering을 위한 가벼운 Depthwise 컨볼루션, 다른 하나는 Feature generation을 위한 무거운 1x1 Pointwise 컨볼루션이다. 

MobileNetV2에서는 Linear bottleneck과 Inverted residual 구조를 도입했다. 이는 어떤 문제의 저 랭크(저차원 공간)의 특성을 이용하여 보다 효율적인 계층 구조를 만들기 위함이다. 이 구조는 아래 Figure 3와 같이 1x1 확장 컨볼루션 뒤에 Depthwise 컨볼루션 그리고 1x1 Project 계층이 이어지는 구조이다. 

![](./Figure/Searching_for_MobileNetV3_1.png)

입력과 출력 계층은 만약에 같은 채널 수를 가지면 Residual connection으로 연결된다. 이 구조는 입력과 출력에서의 압축된 정보는 유지한다. 그러면서 Channel transformation 당 비선형 출력의 표현력을 증가시키기 위해서 내부적으로 Feature space를 고차원으로 확장한다. 

MNasNet은 Bottleneck 구조 안에 Squeeze and excitation에 근거한 가벼운 Attention 모듈을 도입하여 MobileNetV2 구조로 구축되었다. 알아둘 것은 Squeeze and excitation 모듈이 ResNet 기반 모듈과는 다른 위치에서 통합된다는 것이다. 이 모듈은 Expansion에서 Depthwise filter들 뒤에 위치한다. 그래서 가장 큰 특징 정보에 Attention이 적용되게 하기 위함이다(아래 Figure 4).

![](./Figure/Searching_for_MobileNetV3_2.png)



## Network Search

MobileNetV3에서 저자들은 Platform-aware NAS를 사용했다. 이는 네트워크의 각 블럭을 최적화해서 전체적으로 최적화된 네트워크를 찾기 위함이다. 그리고 나서 계층마다 최적의 필터 수를 찾기 위해서 NetAdapt 알고리즘을 사용했다. 이런 기술(NetAdapt)은 보완적이며, 주어진 하드웨어 플랫폼에서 최적의 모델을 찾기 위해서 다른 주요 개념들과 결합되어 사용될 수 있다. 



### Platform-Aware NAS for Block-wise Search

저자들은 플랫폼에 따라 신경망 아키텍처를 선택하는 접근 방식을 채택했다. 저자들은 RNN 기반의 컨트롤러를 사용하고 분해한 구조적 탐색 공간 개념을 적용했다. 저자들은 MnasNet-A1 모델을 저자들의 Large mobile model의 기초로 사용하고 NetAdapt을 적용한 뒤에 다른 최적화를 진행했다. 

그런데 저자들은 원래 연구에서의 Reward가 Small mobile 모델을 최적화 하는데는 맞지 않다는 것을 확인했다. 구체적으로 원래의 연구에서는 아래의 다목적 Reward 함수로 Pareto-optimal 솔루션에 가깝게 만드는 전략을 취한다. 

![](./Figure/Searching_for_MobileNetV3_3.png)

즉 대상 Latency TAR에 근거해서 각 모델 m에 대하여 Accuracy ACC(m)과 Latency LAT(m)의 균형점을 찾는 것이다. 저자들이 관측했을때 작은 모델에서는 지연율과 정확도가 급격하게 변경되었다. 그래서 저자들은 원래의 Weight factor(w = -0.07)보다 더 작은 factor w = -0.15를 적용해서 각 Latency들에 따라 크게 변하는 정확도에 어느정도 커버쳐줄 필요가 있었다. 이 Weight factor로 저자들은 처음에 시작할 모델을 찾기 위한 아키텍처 탐색을 처음부터 시작했고 NetAdapt를 적용하고 다른 최적화를 수행하여 최종적으로 MobileNetV3-Small 모델을 찾아냈다. 



### NetAdapt for Layer-wise Search

저자들이 아키텍처 탐색에 도입한 두번째 기법은 NetAdapt이다. 이 기법은 주요 기법은 아니지만 Platform-aware NAS를 보좌하는 기법이다. 이 기법은 계층들 통채로 적용되기 보다는 각 계층마다 별개로 Fine-tuning을 수행한다. 요약하자면 다음과 같은 과정으로 작업이 수행된다. 

![](./Figure/Searching_for_MobileNetV3_4.png)

원래 연구에서 (c)의 Metric은 정확도의 변화 정도를 최소화 하게 하는 것이 목적이었다. 저자들은 이 알고리즘을 수정해서 지연율과 정확도의 변화 정도 사이의 비율을 최소화 하는 것으로 바꿨다. 이 Metric은 각 NetAdapt step 중에 생성되는 모든 Proposal에 해당하며 저자들은 여기서 다음을 최소화하는 Proposal을 고른다. 

![](./Figure/Searching_for_MobileNetV3_5.png)

지연율의 변화량은 2(a)를 만족해야한다. 이렇게 한 이유는 직관적으로 각 Proposal들이 연관되어 있지 않기 때문에 정확도와 지연율의 Trade-off의 Curve의 경사를 최소화 하는 Proposal을(즉 변화량이 최소화하는) 고르기 위함이다. 

이 과정은 Latency가 대상 Latency에 도달할때까지 반복된다. 그리고 나서 찾은 새로운 아키텍처를 처음부터 훈련시킨다. 저자들은 MobileNetV2에 대해서 원래의 연구에서 사용했던 Proposal generator를 그대로 사용했다. 구체적으로 저자들은 다음과 같은 두 개의 종류의 Proposal을 사용했다. 

![](./Figure/Searching_for_MobileNetV3_6.png)

저자들의 실험에서 T=10000였고 초기의 Proposal들의 Fine-tuning의 정확도를 증가시키는 것을 찾았다. 저자들은 위의 2(a)의 값을 다음과 같이 설정했다. 여기서 L은 초기 모델의 Latency이다. 

![](./Figure/Searching_for_MobileNetV3_7.png)



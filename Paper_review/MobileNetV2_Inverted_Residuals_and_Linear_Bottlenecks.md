# MobileNetV2: Inverted Residuals and Linear Bottlenecks

Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen(Google Inc.)



## Abstract

저자들은 MobileNetV2라는 모바일에 특화된 새로운 아키텍처를 발표했다. 그리고 Object Detection에서의 성능을 보여주기 위해서 MobileNetV2를 SSD에 적용한 새로운 프레임워크인 SSDLite를 발표했다. 또 모바일용 Semantic segmentation에서 성능을 보여주기 위해서 본래의 DeepLabv3보다 용량이 줄어든 형태인 Mobile DeepLabv3를 발표했다. MobileNetV2는 ResNet에서, Pointwise convolution 후에 Regular convolution 수행하고 다시 Pointwise convolution을 수행하면서 첫 번째 Pointwise convolution의 입력 Feature map과 두 번째 Pointwise convolution의 출력 Feature map을 더하는 Bottleneck 구조에 기초하고 있다(이때 Bottleneck 또한 ResNet에서 Bottleneck이 아니라 MobileNetV1의 Width multiplier로 각 계층의 입력과 출력 채널 수가 줄어든 형태이다). 특히 Bottleneck 구조에서 Regular convolution이 아니라 가벼운 Depthwise convolution을 적용했다. 연구 하면서 저자들이 알아낸 사항 중 하나는 이렇게 얇아진 계층에서는 비선형성을 제거하는 것이 중요한데 그 이유는 이미 얇아진 상태에서 비선형성으로 정보가 손실되면 모델의 표현이 더 줄어들기 때문이다.  저자들의 연구 접근 방식으로 비선형성과 모델 표현력의 상관 관계를 구분해서 살펴볼 수 있게되었다.



## Introduction

신경망 구조가 이미지 인식 분야에서 놀라운 성능을 보여주긴 했지만 아주 많은 양의 리소스나 시간을 필요로 하므로 모바일과 임베디드 시스템 같이 제한적인 환경에서는 적용하기 어려웠다. 본 연구 목적은 이런 환경에서 정확도는 유지하면서 연산량을 줄여서 신경망 네트워크를 적용할 수 있게 하는 것이었다. 이 연구 논문의 핵심은 Inverted residual with linear bottleneck이다. 저차원으로 압축되어 있는 입력 특징을 먼저 높은 차원으로 확장 시키고 Lightweight depthwise convolution 연산을 수행한다. 그런다음에 Linear convolution으로 고차원을 다시 저차원으로 Projection시킨다. 이 연구에서 제안한 모델은 쉽게 구현 가능하고 연산 중간에 쓰는 많은 텐서 양을 줄여서 메모리의 부하를 줄인다. 또 소프트웨어적으로 캐시 메모리를 사용해서 메인 메모리에 많은 접근이 일어나지 않도록한다. 



## Related Work

정확도와 속도 사이의 최적의 균형을 이루기 위한 깊은 신경망 네트워크에 대한 연구는 활발히 이루어지고 있다. 아키텍처 자체에 대한 연구, 훈련 알고리즘 개선에 대한 연구가 이뤄져서 초기 신경망 디자인에 큰 개선을 가져왔다. 대표적으로 AlexNet, VGGNet, GooLeNet, ResNet이 있다. 또 하이퍼 파라미터 최적화 라던지 다양한 네트워크 Pruning 방법들, Connectivity learning에 대한 연구도 이뤄졌다. 또 네트워크 내부의 Convolution 블럭들을 연결하는 구조에 대한 연구도 많이 수행되었다. Genetic 알고리즘, 강화 학습을 아키텍처 연구에 적용하기도 했다. 그런데 결론적으로 성능이 좋은 아키텍처는 굉장히 복잡하다는 단점이 있다. 그래서 이 연구에서는 어떻게 하면 신경망을 간단하게 디자인할 수 있을까에 관심을 두었다. 또, 이 연구는 MobileNetV1의 후행연구이다.  



## Preliminaries, discussion and intuition

### Depthwise Separable Convolutions

Depthwise separable convolution은 Regular convolution을 두 개의 별개의 Convolution의 분해하는 것이다. 첫 번째 Convolution은 Depthwise convolution으로 입력 채널마다 단일 Convolution 필터를 적용하여 필터링을 수행한다. 두 번째 Convolution은 Pointwise convolution으로 1x1 Convolution을 적용하여 입력 채널들 간의 선형 결합으로 새로운 특징을 출력한다. Standard convolution은 h_i x w_i x d_i의 입력 텐서 L_i를 받아서 Convolution  kernel인 K ∈ R^(kxkxd_ixd_j)를 적용한다. 그리고 나서 h_i x w_i x d_j 차원의 출력 텐서 L_j를 출력한다. 따라서 Standard convolution 계층의 연산량은 아래와 같다.

![](./Figure/MobileNetV2_Inverted_Residuals_and_Linear_Bottlenecks1.JPG)

Depthwise separable convolution은 Standard convolution과 거의 같은 작업을 수행하면서 연산량은 아래와 같다.

![](./Figure/MobileNetV2_Inverted_Residuals_and_Linear_Bottlenecks2.JPG)

(Depthwise convolution에서 입력 채널마다 1개의 커널이 할당되므로 di = 1이라서 k x k x di = k^2이고 Pointwise convolution에서는 k=1이므로 k x k x dj = dj이다.) Standard convolution과 비교했을 때 k^2xdj/(k^2 + dj) 만큼 연산량이 줄어드는데 거의 커널의 사이즈 k에 의해서 좌우된다. MobileNetV2에서 k=3이므로 8배에서 9배 정도 Standard convolution보다 연산량이 적으면서 약간의 정확도 하락이 있다. 





### Linear Bottlenecks

깊은 신경망 네트워크에 L_i 입력 텐서를 받는 n개의 계층이 있다. 저자들은 이 L_i가 h_i x w_i개의 픽셀의 d_i 차원으로 이루어진 Container라고 한다. 저자들은 계층의 출력(활성화 함수를 거친)이 Manifold of interest를 구성한다고 했다(Manifold of interest란 이미지 특성의 차원에서 추출한 이미지 특성이 차지하는 영역을 뜻한다). 

[ markov's Blog - 손실 압축과 매니폴드 학습](https://markov.tistory.com/39)

많은 연구에서, 신경망에서의 Manifold of interest는 저차원 부분공간에 임베딩될 수 있다고 가정되어왔다(고차원에서의 특징 공간을 저차원에서 표현 가능하다는 뜻). 즉, 깊은 Convolution 계층에서 d 채널의 각 픽셀들의 정보는 사실상 어떤 Manifold 안에 인코딩 되어 있고 이를 저차원 부분 공간으로 임베딩 할 수 있다는 말이다(주의할 점은 부분공간의 차원과 선형 변환을 통해서 임베딩 될 수 있는 Manifold의 차원수는 다르다는 것이다). 언뜻보기에는 그냥 차원 수를 줄이면 될 것처럼 보이고 실제로 MobileNetV1에서는 Width multiplier를 통해서 연산량과 정확도의 Trade off를 효율적으로 조절했다. 이런 직관을 따라서 Width multiplier로 Manifold of interest가 전체 공간을 Span할때까지 Activation 공간의 차원을 줄일 수 있다. 

그런데 이런 직관은 깊은 CNN이 ReLU 같은 비선형 활성화 함수를 수행하기 때문에 잘 맞지 않는다. 보통의 경우 만약에 ReLU 연산 결과가 Non-zero volume S를 출력한다면 S로 매핑되는 점들은 입력의 선형 변환 B 연산을 통해서 얻을 수 있으므로 전체 출력 차원에 대응되는 입력 공간의 부분은 선형 변환에 의해서 제한된다는 말이 된다. 즉, 딥러닝 네트워크는 출력 도메인의 0이 아닌 볼륨 부분에서만 선형 분류기의 힘을 갖는다는 말이다.  

만약에 ReLU가 채널을 붕괴시키면 필수불가결로 그 채널에서의 정보는 손실될 수 밖에 없다. 그렇지만 만약에 아주 많은 채널이 있다면 다른 채널에서 정보가 보존될 가능성이 있으므로 Activation manifold가 구축될 수 있다. 

![](./Figure/MobileNetV2_Inverted_Residuals_and_Linear_Bottlenecks3.JPG)

![](./Figure/MobileNetV2_Inverted_Residuals_and_Linear_Bottlenecks4.JPG)

[AiRLab. Research Blog - MobileNetV2](https://blog.airlab.re.kr/2019/07/mobilenetv2)

저자들은 아래와 같이 입력 Manifold가 입력 특징 공간을 ReLU 변환에 의해서 고차원 공간의 저차원 Manifold로 임베딩 하고 나서 원래대로 복원했을 때 채널수가 적으면 정보가 많이 손실되지만 채널수가 많으면 그런대로 잘 복원하는 것을 보여줬다. 

![](./Figure/MobileNetV2_Inverted_Residuals_and_Linear_Bottlenecks5.JPG)

요약하자면 Manifold of interest가 고차원 Activation 공간의 저차원 부분 공간에 있을때의 요구사항은 다음과 같다. 

- Manifold of interest가 ReLU 변환 이후에 Non-zero volume으로 남아있을때 이것은 선형 변환에 해당한다. 
- ReLU가 입력 Manifold에 관한 정보를 완전하게 보존하는 경우는 입력 Manifold가 입력 특징 공간의 저차원 서브 공간에 있을때 이다. 

위 두가지 시사점을 바탕으로 저자들은, 현존하는 신경망 아키텍처를 최적화하는 방향에 대한 통찰을 얻게 되었다. Manifold of interest가 저차원 공간에 있다고 가정하고 Convolution 블럭 안에 Linear bottleneck 계층들을 삽입하는 것이다. Linear 계층이 중요한 이유를 저자들은 실험을 통해서 경험적으로 알아내었는데, 비선형성이 너무 많은 정보를 파괴하는 것을 막는다. 저자들은 Bottlenect 안의 비선형 계층이 성능을 해친다는 가정을 세웠고 실험을 통해 이를 검증했다. 

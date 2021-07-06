# Transformers in Vision: A Survey

Salman Khan, Muzammal Naseer, Munawar Hayat, Syed Waqas Zamir, Fahad Shahbaz Khan, and Mubarak Shah



## Abstract

NLP에서 좋은 성능을 보여준 Transformer를 Vision에 적용하려는 시도가 많이 일어나고 있다. Transformer가 가장 눈에 띄는 장점은 입력 시퀀스 요소 간의 (거리가 멀리 떨어져 있는) 의존성을 모델링 할 수 있다는 점 그리고 RNN 계열 알고리즘과는 다르게 병렬 처리가 가능하다는 점이다. 또 컨볼루션 네트워크와는 디자인 상에서 최소한의 Inductive biase만을 요구한다(특정 Task를 수행하는 모델을 만들기 위해서 미리 모델에 걸어둔 제약사항 혹은 설정). 게다가 Transformer에서는 Multiple modality(이미지, 비디오, 텍스트, 음성 등의 각기 다른 성질을 가진 데이터를 같이 처리하는 방식)를 지원하고 매우 큰 네트워크와 엄청난 양의 데이터셋에서도 좋은 성능을 보여준 바 있다. 저자들이 말하는 이 논문의 목적은 Vision 분야에 적용한 Transformer 모델에 대한 포괄적인 개요를 제공하는 것이다. 먼저 Transformer 모델에서의 필수 개념(Self-attention, Large-scale pre-training, Bidirectional feature encoding 등)을 설명하고 여러 Vision Task(Image classification, Object Detection 등)에 적용한 유명한 Transformer 모델 사례에 대해서 소개한다. 이때 아키텍처 디자인이나 실험 값 분석을 통해서 각 기술의 상대적인 장점과 단점을 비교한다. 마지막으로 미래에 수행할만한 작업에 대한 방향성을 제시한다. 



## Introduction	

Transformer 모델 중 유명한 모델에는 BERT(Bidirectional Encoder Representations from Transformer), GPT(Generative Pre-trained Transformer), RoBERTa(Robustly Optimized BERT Pre-training), T5(Text-to-Text Transformer)가 있다. Transformer 모델의 영향력이 커지는 데에는 아주 큰 용량의 모델까지의 네트워크를 확장할 수 있다는 데에 있다. 

이렇게 NLP에서 Transformer가 하나의 돌파구가 되자 이를 Vision에서도 적용하려는 움직임이 일어났다(Figure 1).

![](./Figure/Transformers_in_Vision_A_Survey1.png) 

그러나 이미지 데이터는 NLP에서의 데이터와 다른 구조를 가지고 있기 때문에(공간적, 시간적 특성) 이를 고려한 새로운 네트워크 디자인이나 훈련 방법이 필요하다. 결과적으로 여러 Vision Task 영역에서 Transformer를 적용한 연구들이 발표되었다. 이 논문 저자들이 말하는 이 논문의 목적은 이런 연구들에 대해서 소개하는 것이다. 

Transformer 아키텍처는 Self-attention이라는 개념에 근거하고 있는데 이 개념은 시퀀스 요소 간의 관계성에 대해서 학습하는 것이다. RNN에서는 각 요소들을 재귀적으로 처리(앞의 데이터 처리 후 뒤의 데이터 처리)하고 짧은 거리의 Context만 고려할 수 있다면 Transformer에서는 이론적으로 전체 거리의 Context를 고려하는 것이 가능하다. 그래서 아주 긴 거리의 요소 간의 관계성을 학습시킬 수 있고 쉽게 병렬 처리가 가능하다. 

Transformer 모델의 중요한 특징 중 하나는 매우 큰 용량의 모델까지 확장 가능하다는 것과 대용량 데이터를 처리할 수 있다는 것이다. Transformer는 CNN이나 RNN과 비교했을때 어떤 문제 구조에 대한 최소한의 지식만을 필요로 하기 때문에 레이블링 되지 않은 대용량 데이터로 Pre-training이 가능하다. 여기서 Pre-training에서는 수동적으로 Anntotaion을 하는 것을 하지 않아도 되서 불필요한 인력 낭비를 하지 않아도 되고, 또 대용량의 데이터로 학습 시키기 때문에 주어진 데이터 요소의 관계성과 관련된 풍부하고, 일반적이며 특징적인 정보를 인코딩 할 수 있게된다. 이런 일련의 작업 후에 타겟 데이터 셋으로 Fine-tuning하여 원하는 결과를 얻을 수 있다. 

저자들은 네트워크 디자인들을 체계적으로 분류하고 각 방법들의 장점과 단점을 표기했다. 



## Foundations

Transformer 모델 계열이 발전 하는데 두 가지 개념이 핵심 역할을 했다. 

- Self-attention: 시퀀스 요소들 간의 의존성이나 긴 거리의 정보를 잡아내는 기법
- Pre-training: (Un)supervised한 방법으로 (Un)labelled된 대용량 데이터로 미리 모델을 학습 시키고 대상 데이터셋으로 Fine-tuning시키는 기법



### Self-Attention

시퀀스의 각 아이템에 대해서 Self-attention 매커니즘으로 각 아이템이 다른 아이템과 얼마나 관련성이 있는지를 측정한다(예를 들어서 한 단어가 문장 안의 다른 단어와 같이 나와야 하는지 여부 등). 기본적으로 Self-attention 계층에서는 전체 입력 시퀀스에서 전역적으로 정보를 모아 시퀀스의 각 요소(가중치 행렬)를 업데이트 한다.

예를 들어서 한 시퀀스의 n개의 Entity들이 있다고 가정하자((x1, x2, ..., xn), 이때 X ∈ R^(n x d)이고 d는 각 Entity를 표현하기 위한 임베딩 차원이다). 이때 Self-attention의 목적은 전역적 컨텍스트 정보 관점에서 모든 n개의 Entity에 대해서 각 Entity를 인코딩해서 상호 간의 어떤 작용을 하는지 캡처하는 것이다. 이때 이 작업은 다음의 세 개의 학습이 가능한 행렬로 Transformation을 수행하여 이뤄진다. 

| Queries                                                      | Keys                                                         | Values                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](./Figure/Transformers_in_Vision_A_Survey2.png) | ![](./Figure/Transformers_in_Vision_A_Survey3.png) | ![](./Figure/Transformers_in_Vision_A_Survey4.png) |

먼저 입력 시퀀스 X를 각 가중치 행렬에 투영시켜서 세 가지 행렬 결과 값을 얻는다. 

![](./Figure/Transformers_in_Vision_A_Survey5.png)

![](./Figure/Transformers_in_Vision_A_Survey6.png)

![](./Figure/Transformers_in_Vision_A_Survey7.png)

그리고 나서 다음과 같은 출력 Z는

![](./Figure/Transformers_in_Vision_A_Survey9.png)

다음과 같이 계산해서 얻는다. 

![](./Figure/Transformers_in_Vision_A_Survey8.png)

과정을 그림으로 표현하여 아래와 같다. 

![](./Figure/Transformers_in_Vision_A_Survey10.png)

![](./Figure/Transformers_in_Vision_A_Survey11.png)

#### Masked Self-Attention

원래의 Self-attetion 계층에서는 모든 Entity에 대해 관련성을 조사한다. 그런데 시퀀스에서 한 Entity의 다음 Entity를 예측하도록 훈련된 경우에는 Decoder에서 기준이 되는 Entity 다음 순서의 모든 Entity를 조사하는 것을 못하도록 마스킹 한다. 이걸 수행하는 방법은 단순히 상삼각행렬 M(M ∈ R^(n x n))과의 요소별 곱셈(Hadamard 곱)을 통해서 수행하면 된다. 

![](./Figure/Transformers_in_Vision_A_Survey12.png)

이렇게 하면 기준 Entity의 다음 Entity를 예측하면서 미래 Entity의 Attention score를 모두 0으로 만들 수 있다. 

#### Multi-Head Attention

시퀀스 안의 각기 다른 요소들 사이의 여러 복잡한 관계성을 캡슐화 하기 위해서 Multi-head attention 이라는 매커니즘을 적용한다. 이 매커니즘은 여러 Self-attention 블럭으로 구성되어 있다. 각 블럭에는 저마다의 Query, Key, Value 가중치 행렬이 있다. 입력 X에 대해서 Multi-head attention의 h개의 Self-attention 블럭의 출력들을 Concatenation 해서 하나의 단일 행렬로 만든다. 

![](./Figure/Transformers_in_Vision_A_Survey13.png)

그리고 나서 다음과 같은 행렬로 투영시킨다. 

![](./Figure/Transformers_in_Vision_A_Survey14.png)

전체 과정은 아래 그림과 같다. 

![](./Figure/Transformers_in_Vision_A_Survey15.png)

Self-attention과 Convolution의 가장 큰 차이점은 Self-attention이 학습이 끝난 모델에 대해서도 동적으로 들어오는 데이터에 따라 가중치가 변함에 반해 Convolution은 학습이 끝난 모델은 가중치가 변하지 않는다는 점이다. 이렇게 되면 어떤 특이성을 가진 입력 데이터에 대해서 Self-attention이 Convolution보다 더 잘 대응할 수 있게 된다. 



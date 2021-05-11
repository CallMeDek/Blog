# Attention Is All You Need

Ashish Vaswani, Noam Shazeer, Łukasz Kaiser(Google Brain), Niki Parmar, Jakob Uszkoreit, Llion Jones(Google Research), Aidan N. Gomez(University of Toronto), Illia Polosukhin



## Abstract

어떤 시계열 데이터를 변환하는 모델 개발에서 지배적인 경향은, Encoder와 Decoder를 포함하는 복잡한 RNN 혹은 CNN를 기반으로 모델을 만든다. 가장 성능이 좋은 모델들을 보면 Attention 매커니즘으로 Encoder와 Decoder를 연결한다. 저자들은 Transformer라는 새로운 네트워크 아키텍처를 제안했다. 특징으로는 RNN이나 CNN에 의존하지 않고 오직 Attention 매커니즘에만 근거한다는 것이다. 이 모델로 두 기계 번역을 실시했을때 성능이 압도적이었으며 동시에 병렬적으로 데이터를 처리할 수 있었고 훈련 하는데 시간도 상당히 짧았다고 한다. 저자들은 Transformer가 저자들이 실험한 작업 말고 다른 작업에서도 잘 동작할 수 있다고 말한다. 



## Introduction

RNN, Long-term memory, Gated RNN은 기계 번역같은 시계열 데이터를 변환하는 문제에서 확고하게 SOTA의 성능을  보여왔고 지배적인 위치를 차지해 왔다. 많은 연구 노력들이 지속적으로 RNN과 Encoder-Decoder 아키텍처 스타일의 모델의 한계를 더 높이려고 해왔다. RNN은 기본적으로 입력과 출력 열의 각 기호의 위치에 따라 계산 과정을 분해한다. 계산 과정의 각 단계에 따라서 t 위치의 입력 값과 바로 전 모델의 가중치 상태 h_t-1을 입력으로 새롭게 모델의 가중치 상태를 h_t로 만든다. 그런데 이런 과정은 입력 데이터를 병렬적으로 처리하는 것을 방해한다. 그리고 이것은 메모리의 한계 때문에 아주 길이가 긴 열 데이터에서는 치명적일 수 있다. 어떤 연구에서는 인수분해 트릭과 조건에 따른 계산 기법을 통해 계산 과정을 최적화 해서 상당히 이런 과정을 개선시켜왔다. 그러나 이런 시계열적 계산 과정(앞의 데이터를 처리하고 나서 뒤의 데이터를 처리하는)은 본질적으로 한계가 있을 수 밖에 없다. 

Attention 매커니즘은 다양한 분야에서 시계열 데이터를 모델링하거나 변환하는 과정을 해야할때 필수적인 부분이 되고 있다. 왜냐하면 입력과 출력 시계열의 거리와 상관 없이 데이터의 의존성을 모델링 할 수 있기 때문이다. 그러나 이런 Attention 매커니즘은 RNN과 같이 사용되고 있다는 점에서 한계가 있다. 

이 연구에서 저자들은 Transformer라는 아키텍처를 제안했다. 이 모델은 RNN 부분을 없애고, 입력과 출력 간의 전역적인 의존성을 모델링 하는데에도 완전히 Attention 매커니즘에만 의존한다. Transformer에서는 데이터를 병렬적으로 처리가 가능하고 그로 인해서 모델 훈련 시간이 빨라진다. 



## Background

연속적인 연산의 연산량을 줄인다는 목적은 Extended Neural GPU, ByteNet 등의 연구의 발단이 되었다. 이 모든 연구들은 CNN을 기본 빌딩 블럭으로 사용했고 모든 입력과 출력 위치에 대해서 병렬적으로 중간 과정의 표현들을 계산했다. 그런데 이런 모델들에서 두 임의의 입력이나 출력 위치 사이의 시그널과 관련된 연산량은 두 위치가 멀수록 증가했다. 예를 들어서ConvS2S의 경우는 선형적으로, ByteNet은 로그함수적으로 늘어났다. 이런 특성은 두 위치 사이의 거리가 멀면서 연관성이 있을때 이를 학습하는 것을 어렵게 했다. Transformer에서는 이런 거리가 멀때 필요한 연산의 숫자가 상수적으로 줄어든다(O(상수)). Attention-weighted 위치에서의 평균을 구하는 것 때문에 유효한 해상도가 줄어들긴하지만 Multi-Head Attention으로 이 악영향에 대응한다. 

Self-attention 혹은 Intra-attention은 한 시퀀스의 표현을 계산하기 위해서 하나의 시퀀스의 각기 다른 위치들을 연관시키는 Attention 매커니즘이다. Self-attention은 그동안 다양한 과업에 사용되었다. 

End-to-End memory 네트워크는 Sequence-aligned recurrence 대신에 Recurrent attention 매커니즘에 근거한다. 이 네트워크는 단순한 질문 및 답변 수행히안 언어 모델링 과업에서 좋은 성능을 보였다. 

저자들이 말하길 Transformer는 Sequence-aligned RNN 혹은 컨볼루션을 사용하는 것 대신에 입력과 출력 사이의 표현을 연산해내기 위해서 온전히 Self-attention에 의존하는 첫번째 Transduction(시퀀스 간 변환) 모델이라고 한다.



## Model Architecture

대부분의 경쟁력 있는 신경망 기반의 Sequence transduction 모델은 Encoder-decoder 구조를 가지고 있다. 이런 구조에서는 Encoder가 (x1, ..., xn)이라는 입력 시퀀스를 연속적인 표현인 z = (z1, .., zn)으로 매핑한다. z에 대해서 Decoder는 출력 시퀀스 (y1, ..., yn)을 하나의 시간에 하나의 심볼씩 만들어낸다. 각 스텝에서 모델은 Auto-regressive하기 때문에 다음 단계에서 심볼 시퀀스를 만들어낼때 바로 전에 만들어진 심볼을 추가적인 입력으로 사용한다. 

Transformer는 Figure 1과 같이 Encoder와 Decoder에 Stacked self-attention과 Point-wise 그리고 완전 연결 계층을 사용해서 위와 같은 구조를 따른다. 

![](./Figure/Attention_Is_All_You_Need1.png)

### Encoder and Decoder Stacks

- Encoder: Encoder는 위의 그림에서 N=6의 Stack으로 구성되어 있다. 각 계층은 두 가지 서브 계층들로 이루어져 있다. 첫 번째는 Multi-head self-attention 매커니즘을 따르고 두번째는 단순하게 Position-wise fully connected feed-forward 네트워크이다. 저자들은 Residual connection을 적용하고 Layer normalization을 적용했다. 그러므로 각 서브 계층의 출력은 LayerNorm(x + Sublayer(x))가 되고 여기서 Sublayer(x)는 각 서브 계층의 기능을 구현한 것이다. Residual connection을 용이하게 하기 위해서 모델에 있는 모든 서브 계층들과 임베딩 계층들은 출력 차원을 512로 고정했다. 
- Decoder: Decoder도 마찬가지로 N=6의 Stack으로 구성되어 있다. Encoder에서의 두 서브 계층들에다 세 번째 서브 계층이 추가되었다. 여기서는 전 단계의 계층의 출력 뿐만 아니라 Encoder stack의 출력 또한 입력으로 받아서 Multi-head attention을 수행한다. Encoder와 유사하게 각 서브 계층들에는 Residual connection과 Layer normalization을 적용했다. 특별히 Decoder stack에서 Self-attention 서브 계층을 수정했는데 어떤 위치에서 다음 미래의 위치의 정보를 참고하는 것을 방지하게 했다. 이런 Masking 작업은 Position i에서의 예측값이 오직 i 이전의 위치에 있는 정보만 참고할 수 있도록 한다. 
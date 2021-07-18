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



### (Un)Supervised Pre-training

Self-attention 기반의 Transformer 모델들은 보통 2단계로 훈련이 진행된다. 먼저 지도 학습 혹은 비지도 학습적인 양상으로 큰 데이터셋에서 Pre-training을 수행한다. 그 후에 타겟 데이터셋으로 Fine-tuning한다. Pre-training 후에 Fine-tuning 하는 방식의 모델 정확도가 더 높은 것으로 실험을 통해 입증 된 바 있다.

그런데 많은 양의 데이터에 대해서 수동적으로 레이블링 하는 것은 번거로우므로 Self-supervised 학습이 Pre-training 단계에 적용된다. Self-supervised 학습 덕분에 매우 큰 네트워크에서 학습하는 것도 가능할 정도로 Self-supervised 기반의 Pre-training 단계가 Transformer 모델의 확장성과 일반화에 큰 영향을 끼친다. SSL의 기본적인 아이디어는 빈칸을 채우는 것이다. 예를 들어, 이미지 내에 어떤 폐색된 객체를 예측하는 시도를 하고자 할 때 Temporal video sequences에서 과거와 현재의 프레임으로 유추하여 예측하는 일이 있고, 또한 입력에 적용된 회전 각도라던지 이미지 패치의 순서를 어떻게 바꿨는지, 무채색 이미지의 색을 유추한다던지 하는 일이 있다. 

Self-supervised 제약을 적용하는 다른 방법은 Constrastive 학습이다. 이 경우에는 어떤 이미지로 두 가지 타입의 이미지를 만들어 내기 위해 Nuisance transformation이 쓰인다. 그 다음에 모델은 Nuisance transformation에 Invariant하도록 훈련되고 Semantic 레이블을 변경하는 변화 정도를 최소화 하도록 모델링 하는 쪽으로 강화된다. 

Self-supervised 학습이 유망한 이뉴는 Annotated 되지 않은 많은 양의 데이터로 학습을 할 수 있기 때문이다. SSL 기반의 Pre-training 단계에서 모델은 Pretext task(입력 데이터에 적용된 어떤 기법들)을 알아내는 것을 통해 데이터에 있는 유의미한 표현을 학습하는 법을 배운다. Pretext task를 위한 가짜 레이블이 데이터 특성과 작업 특성에 따라 자동적으로 부여된다. 그러므로 Pretext task를 잘 정의하는 것이 SSL에서는 매우 중요하다. 현재까지의 SSL은 Pretext task에 따라 다름과 같이 묶을 수 있다고 한다.

- Generative approach - 이미지나 비디오를 합성하는 합성하는 것, 예를 들어서 Masked image modeling, Image colorization, Image super_resolution, Image in-painting, GAN 기반의 방법들이 있다. 
- Context-based method - 이미지 패치나 비디오 프레임 같의 관계성을 이용하는 것, 예를 들어서 이미지 패치들에 대한 Jigsaw 퍼즐 풀기, Masked object classification, Rotation 같은 기하학적 변환 예측하기, Video 프레임의 시간적 순서 확인하기가 있다. 
- Cross-modal method - 여러 데이터 양상에 의존하는 것, 예를 들어서 텍스트와 이미지, 오디오와 비디오, RGB와 Flow 등 두 가지 양상의 데이터의 관련성을 확인 하는 것들이 있다. 



### Transformer Model

![](./Figure/Transformers_in_Vision_A_Survey16.png)

위의 그림은 Transformer 모델의 아키텍처를 보여준다. 아키텍처는 Encoder-decoder 구조로 이뤄져 있다. Encoder는 6개의 동일한 블럭으로 궁성되어 있고 각 블럭은 두 개의 서브 계층으로 이뤄져 있다(Multi-head self-attention 네트워크, Position-wise fully connected feed-forward 네트워크). 각 블럭 뒤에는 Layer normalization과 Residual connection이 있다. 컨볼루션 네트워크에서는 Feature aggregation과 Transformation(컨볼루션 작업 후에 Non-linearity 작업을 수행하므로)이 동시에 수행되지만 Transformer 네트워크에서는 분리되어 있다. Self-attention 계층에서는 Aggregation만 수행하고 Feed-forward 계층에서는 Transformation을 수행한다. Encoder와 유사하게 Decoder도 6개의 동일한 블럭으로 구성되어 있다. 각 Decoder 블럭은 세 개의 서브 계층으로 구성되어 있는데 첫 두 계층은 Encoder와 비슷하고 세 번째 계층은 그 계층과 매칭되는 Encoder 블럭의 출력에 대해 Multi-head attention을 수행한다. 

원래 Transformer 모델은 기계 번역 작어을 위해 훈련되었다. Encoder의 입력은 어떤 언어의 문장 내의 단어들이다. Positional encoding은 문장 안에서 각 단어의 상대적인 위치를 파악하기 위해 입력 시퀀스에 더해진다. Positional encoding도 입력 차원과 같이 512 차원이다. 그리고 미리 정해진 값을 사용하거나 학습 될 수 있다(Sine, cosine function). Auto-regressive 모델이 되기 위해서 Transformer의 Decoder는 문장에서 다음 간어를 출력하 위해서 바로 전에 예측을 사용한다. 그러므로 Decoder는 Encoder에서도 데이터를 입력으로 받고 바로 전의 예측 값을 입력으로 받는다. Residual connection을 적용하기 위해서 모든 계층의 출력 차원은 동일하다. 



### Bidirectional Representations

원래 Transformer에서는 훈련 시에 현재 기준 왼쪽에 있는 단어들만 사용할 수 있었다. 그러나 이것 때문에 성능 상의 제약이 있을 수 밖에 없었는데 대부분의 언어에서는 현재 기준 왼쪽과 오른쪽의 문맥적 정보가 모두 중요하기 때문이다. BERT는 단어의 왼쪽과 오른쪽 문맥 정보를 동시에 인코딩 하는 것을 제안했다. 그래서 텍스트 데이터의 Feature representation을 학습하는 것을 개선시켰다(비지도적인 양상으로). Bidirectional 훈련이 가능하려면 기본적으로 두 가지 Pretext task가 필요하다. 하나는 Masked language model이고 다른 하나는 Next sentence prediction이다. 모델은 비지도적인 양상으로 이 두 가지 Pretext task로 Pre-training하고 나서 Fine-tuning 된다. 이렇게 하기 위해서 Pre-trained된 모델에 추가적인 출력 모듈을 추가해서 End-to-End로 Fine-tuning한다. 

기본 BERT 모델의 아키텍처는 원래 Transformer 모델에 근거하고 GPT와 유사하다. 원래의 Transformer와 비교했을때 차이는 BERT 모델은 Encoder만 사용한다는 것이다. 이에 반해 GPT는 Decoder만 사용한다. BERT가 가장 크게 기여한 바는 Pretext task 정의이다. 이는 비지도적인 방식의 Bidirectional feature encoding을 가능하게 했다. 이를 위해서 BERT는 두 가지 전략을 제안했다. 

- Masked Language Model(MLM) - 미리 설정해 논 비율의 단어를 문장에서 마스킹 해 놓고 모델은 Cross-entropy loss를 고려하여 이 마스킹된 단어를 예측하도록 훈련된다. 마스킹된 단어를 예측하면서 모델은 Bidirectional context를 포함하는 방법을 학습하낟.
- Next Sentence Prediction(NSP) - 주어진 한 쌍의 문장에 대해서 모델은 Binary 레이블을 예측한다. 예를 들어서 그 쌍이 원래의 문서에서 유효한가 아닌가. 이를 위한 훈련 데이터는 그 어떤 Text corpus에서도 쉽게 만들어 낼 수 있다. A와 B라는 문장 쌍이 만들어져서 NSP는 문장 간의 관계를 잡아내는 모델을 생성하는 것을 가능하게 한다.  



## Transformers & Self-attention in Vision

![](./Figure/Transformers_in_Vision_A_Survey17.png)

위의 그림 4는 Transformer가 비전 분야에 쓰인 연구들의 개요를 보여준다. 현존하는 프레임워크들은 Global 혹은 Local attention을 주로 적용하고 모델 디자인 효율성을 위해서 Matrix factorization을 이용하거나 CNN representation의 도움을 받는다. 그리고 Vectorized attention 모델을 사용한다. 



### Transformers for Image Recognition

컨볼루션 연산은 CNN의 원동력이다. 왜냐하면 ImageNet과 같은 고차원의 데이터셋에서 복잡한 Image recognition 작업 같은 것을 위한 해결책을 제시하는데 도움이 되기 때문이다. 그러나 이 연산도 단점이 있다. 고정된 크기의 윈도우로 연산을 수행하기 때문에, 예를 들어서 비디오에서 공간적, 시간적으로 멀리 떨어져 있는 임의의 두 픽셀 간의 연관성을 캡처할 수 없다. 게다가 훈련이 끝나면 필터의 가중치 값이 고정되기 때문에, 입력에 어떤 변형이 가해질때 이를 유동적으로 대처할 수 없다. 이와 같은 단점은 Self-attention 매커니즘과 Transformer 네트워크를 사용하여 완화 할 수 있다. 

Self-attention은 두 개의 주요 접근 방식이 있다. 

- Global self-attention은 입력 특징의 크기에 구애받지 않는다. 예를 들어서 전체 Feature map에 Attention을 적용하는 계층을 사용한다던지 Sparse attention map을 디자인해서 Non-local 연산의 연산적 복잡도를 줄이는 방법이 있다. 
- Local self-attention은 어떤 주어진 가까운 영역에서의 관계성을 모델링 하려고 한다. 예를 들어서 연산적 Overhead를 줄이기 위해서 어떤 주어진 픽셀 주위의 정해진 Window 크기로 Attention을 제안한다던가 입력 데이터/특징의 변동에 Weight aggregation 정도를 동적으로 조정하는 방법이 있다. 

NLP Transformer의 encoder를 직접적으로 이미지 패치에 적용하는 Global self-attention 방식이 성공하기도 했다. Transformer는 본질적으로 많은 데이터를 필요로 한다. ViT를 처음부터 훈련시키기에 ImageNet 같은 데이터 양으로는 부족하다. 그래서 어떤 연구에서는 CNN 모델과 ViT 모델을 Distill knowlege 기법을 적용해 훈련시켜서 Transformer 모델이 추가적인 데이터가 필요 없이 ImageNet 만으로 훈련이 가능하게 하기도 했다. 



#### Non-Local Neural Networks

이 방법은 Non-local means 연산에서 영감을 받았는데 Non-local means 연산은 원래 Image denoising을 위해 디자인되었다. 이 연산은 패치 안의 주어진 픽셀을, 이미지 안에 다른 픽셀 값들의 가중치 합으로 수정한다. 그러나 픽셀 주변의 고정된 크기의 Window의 픽셀들만을 고려하는 것 대신에, 패치 간의 유사성에 근거해서 Filter response에 기여하는 (거리가) 먼 픽셀들을 선택한다. 디자인에 의해서 Non-local 연산은 이미지 공간에서 거리가 먼 특징들의 의존성을 모델링 하게 된다. 이것에 영감을 받아 Wang 등은 DNN을 위한 미분 가능한 Non-local operation을 제안했다. 여기서는 순전파 시에 공간적, 시간적으로 거리가 먼 특징들의 의존성을 캡처한다. 주어진 Feature map에 대해서 이 연산에서는 어떤 위치에 대한 Response를 Feature map 상의 모든 위치에서의 특징들의 가중치 합으로 계산한다. 이 방법으로 Non-local operation은 Feature map 상에서 임의의 두 위치의 거리가 얼마나 먼지와 상관 없이 두 위치가 어떻게 연관되어 있는지를 잡아낼 수 있다. Video classification이 공간적, 시간적으로 떨어져 있는 픽셀들 간의 연관성을 확인할 필요가 있는 Task의 한 예이다.  Kinetics 데이터셋으로 이 방법의 유효성을 보여준 연구가 있다. 



#### Criss-Cross Attention

![](./Figure/Transformers_in_Vision_A_Survey18.png)

Self-attention 매커니즘이 전체 이미지의 Contextual information을 모델링 할 수 있게 했다고 하더라도 이 방법은 필연적으로 메모리 사용량과 연산량이 많이 필요할 수 밖에 없다. 위 그림 a와 같이 주어진 픽셀 위치에 대한 Global context를 인코딩 하기 위해서는 Non-local 블럭은 Dense한 Attention map 연산을 수행해야 한다(그림 a의 초록색 부분). Non-local 블럭은 O(N^2)의 높은 복잡도를 보인다(N은 입력 Feature map의 크기). 이런 연산적 부담을 줄이기 위해서 Huang 등은 Criss-cross attention 모듈을 제안했다. 여기서 각 픽셀 위치는 Sparse attention map이라고 하는, 가로 세로의 십자가 모양의 위치만 계산한다(위 그림 b). 또한 Criss-cross attention을 재귀적으로 적용해서 각 픽셀 위치에서는 모든 다른 픽셀부터 Context를 캡처할 수 있다.  Non-local 블럭과 비교했을때 11배 적은 GPU 메모리를 사용하고 복잡도도 O(2루트N)을 보인다. 



#### Stand-Alone Self-Attention

컨볼루션 계층은 Translation equivariance의 특성을 보이지만 큰 Recptive field로 스케일을 키울 수 없으므로 멀리 떨어져 있는 위치 간의 관련성을 캡처하기 어렵다. 이에 반해 Global attention은 입력의 모든 위치의 연관성에 대해서 캡처 할 수 있지만 연산적으로 부담이 된다. 그래서 Global attention은 Down sampling한 작은 이미지나 이미지 패치들 그리고 Augmented된 컨볼루션 Feature space에 수행한다. Ramachandran 등은 DNN에서 컨볼루션을 계층을 Local self-attention 계층으로 대체하는 것을 제안했다. 이때 계산적 비용을 증가시키지 않고도 큰거나 작은 입력에 수행할 수 있다고 한다. 이 Self attention 계층에서는 주어진 픽셀 주위에 고정된 크기의 Window 만큼의 픽셀을 모두 고려해서 이 픽셀 들에 대한 Query, Key, Value 벡터들을 계산하고 Spatial information을 집계한다. Value 벡터는 Query와 Key의 Softmax score를 Projection 한 뒤 집계된다. 이런 과정이 주어진 모든 픽셀에 대해서 반복되고 그렇게 나온 Response가 Output 픽셀을 만들어 내기 위해서 Concatenated된다. ResNet + Self attention 계층이 컨볼루션 기반의 ResNet 모델과 비교했을 때 더 적은 파라미터로 ImageNet과 COCO Object Detection 문제를 풀 수 있다고 한다. 



#### Local Relation Networks

컨볼루션 연산의 다른 단점은 훈련 뒤에 고정된 가중치로 결과를 내기 때문에 입력에 어떤 변경이 가해져도 유동적으로 대처할 수 없다는 것이다. Hu 등은 Local window 안의 픽셀들을 유동적으로 구성하는 것을 제안했다. 이 저자들은 Local window 안에 있는 픽셀들/특징들 사이의 유사성 혹은 연관성에 근거하여 Weight aggregation을 조정하는 새로운 형태의 미분가능한 계층을 그들의 연구에 도입했다(아래 그림 Fig 6). 

![](./Figure/Transformers_in_Vision_A_Survey19.png)

이런 Adaptive weight aggregation은 네트워크에 Geometric priors를 도입하는데 이는 Recognition task에 매우 중요하다. 컨볼루션은 여러 위치에서 고정되어 있기 때문에 Top-down operator로 보는데 반해 Non-local 연산은 전체 이미지에 대해서 입력 Feature을 집계하기 때문에 Bottom-up 방법으로 본다. Local relation 계층은 Bottom-up 방법으로 분류되기는 하지만 고정된 Window 크기로 제한된다. 



#### Attention Augmented Convolutional Networks

Bello 등은 Self-attention을 컨볼루션 연산 대신으로 사용하는 것의 가능성을 연구했다. 저자들은 2차원에서 Relative position 인코딩을 사용하는 것을 제안했다. 이렇게 해서 Translation equivariance를 유지하는 새로운 Self-attention 매커니즘을 개발하는데 이는 이미지를 다루는데 바람직한 속성이다. 이런 Self-attention이 Stand-alone computational primitive로서 경쟁력 있는 결과를 제공하지만 가장 좋은 성능은 컨볼루션 연산과 같이 쓰일때 나온다. 많은 연구를 통해서 Attention augmentation이 현존하는 다양한 아키텍들에 대해서 Image classification과 Object detection에서 시스템적 성능 개선을 낸 것을 보여줬다. 



#### Vectorized Self-Attention

Zhao 등은 전통적인 컨볼루션 연산이 Feature aggregation과 Transformation을 동시에 수행한다는 것을 강조한다(Filter로 컨볼루션 연산 후에 비선형 연산을 수행). 이와 대조적으로 이 저자들은 Self-attention으로 Feature aggregation을 수행하고 Element-wise Perceptron 계층으로  Transformation을 수행하는 형태를 제안했다(아래 Fig 7).

![](./Figure/Transformers_in_Vision_A_Survey20.png)

이렇게 하기 위해서 저자들은 Feature aggregation을 위한 두 가지 전략을 제안했다. 

- Pairwise self-attention - Permutation, Cardinality 불변 연산 
- Patch-wise self-attention - 위와 같은 불변 속성이 없음(컨볼루션 연산과 유사)

Pairwise와 Patch-wise self-attention 모두 Vector attention으로 구현되었다. Vector attention은 Spatial과 Channel 차원 모두를 위한 가중치를 학습한다. 이 연산은 기존에 수행하는 Scalar weights(내적 연산을 수행하는)을 이용하는 연산에 대한 대안을 제시한다. Pair-wise attention은 주어진 주위 픽셀들에 대해서 주위 픽셀들과 특정한 feature간의 관계성을 고려하면서 Vector attention을 계산하는 Set 연산이다. 반대로 Patch-wise self-attention은 컨볼루션 연산의 일반화된 버전(Set 연산이 아님)이다. 이 연산에서는 Attention 벡터들을 유도해낼때 주위의 모든 Feature 벡터들을 살펴본다. 저자들은 연구를 통해, 훨씬 더 적은 파라미터로 Self-attention 네트워크(SAN)이 ImageNet 데이터셋에서 ResNet 계열의 Baseline에 견줄만하다는 것을 보여줬다. 이 연구 방식의 중요 속성 중 하나는 Adversarial perturbation에 강인하다는 것과 데이터에 잘 모르는 변형이 가해진 것에 Generalization 하다는 것이다. 이는 Attention의 유동적인 특성이 Adversay가 유용한 속임수를 알아내지 못하게끔 하기 때문이다. 



#### Vision Transformer

ViT는 대용량 컴퓨터 비전 데이터셋 Task에서 Transformer가 컨볼루션을 대체한 첫 번째 케이스라고 한다. NLP에서 사용되었던 Transformer를 약간 변경해서 이미지 패치에 적용했다. Transformer 모델은 대용량 데이터셋으로 미리 훈련시키고 타겟 데이터셋으로 Fine-tuning 시켰다. Pre-training 단계가 매우 중요한데 왠만한 대용량 데이터셋이 아니고서야 SOTA의 성능을 내는게 어렵다고 한다. 컨볼루션은 이미지 도메인에 대한 사전 지식(Inductive biases - Translation equivariance)을 인코딩하기 때문에 어느정도 데이터의 필요성을 줄여주는데 반해 Transformer는 이런 지식을 대용량 데이터셋에서 스스로 발견해야 하기 때문이다. 이 연구에서는 300 백만의 JFT 이미지 데이터셋으로 Pre-training 했다고 한다. 

![](./Figure/Transformers_in_Vision_A_Survey21.png)

Figure 8은 ViT의 아키텍처를 보여준다. NLP에서 쓰인 Transformer와 매우 유사한데 차이는 2D짜리 이미지 패치를 벡터 형태로 Flattened시켜서 Transformer에 주입했다는 것이다. 이 벡터화된 패치들은 Linear 계층에서 Patch embedding으로 Projected되고 Position embedding이 위치 정보를 인코딩하기 위해서 추가된다. [cls]라는 토큰이 입력 데이터의 처음 부분에 더해진다. 첫번째 위치에 해당하는 출력 Representation이 Image classification에서 Global image representation으로서 사용된다. 



#### Data-Efficient Image Transformers

Data-efficient image Transformer(DeiT)는 JFT 같은 추가적인 대용량 데이터 없이 Large-scale image classification을 수행한 첫번째 연구 사례이다. 보통은 Transformer 모델은 CNN과 달리 이미지 도메인에 대한 사전 지식을 가정하지 않기 때문에 더 많은 데이터셋과 더 많은 훈련시간을 필요로 한다. 그러나 DeiT는 상대적으로 더 적은 양의 데이터셋(1.2 백만)으로 상대적으로 더 적은 훈련 시간에 Transformer 모델을 훈련시킬 수 있다는 것을 입증했다. DeiT의 주요 기여점은 Transformer 모델을 위한 Novel native disillation 접근법이다. 

Distillation 과정에서 CNN 모델을 Teacher로서 사용하는데 CNN 모델의 출력을 Transformer 모델을 훈련시키는데 사용한다. CNN의 출력은 Transformer가 효율적으로, 입력 이미지의 유용한 Representation을 이해하는데 도움을 준다. Distillation 토큰이 입력 Patch embedding과 Class token 벡터에 추가된다. Self-attention 계층들이 이 토큰들에서 연산을 수행하여 토큰 사이의 관련성을 학습하고 학습된 Class, patch, distillation token들을 출력한다. 네트워크는 출력 Class token으로 정의된 Cross-entropy loss와, 출력된 Distillation token과 Teacher의 출력을 매칭시켜서 얻는 Distillation loss로 훈련된다. Soft와 Hard 레이블이 Distillation을 위한 옵션으로 연구되었는데 Hard distillation이 더 좋은 성능을 냈다. 흥미롭게도 학습된 Class와 Distillation 토큰들은 큰 연관성을 보이지는 않았는데 이는 이들간의 상호보완성을 나타낸다. 



#### CLIP: Contrastive Language-Image Pre-training

CLIP은 텍스트로부터 Image representation을 학습하는 Contrastive 접근법이다. 큰 배치에서 정확한 텍스트-이미지 쌍의 임베딩 간의 유사성을 극대화 하는 것이 목적이다. 구체적으로 N개의 이미지-텍스트 쌍에 대해서 CLIP는 Multi-modal 임베딩 스페이스를 학습한다. 이때 Image-encoder와 Text-encoder를 같이 훈련시켜서 N개의 Image-text 쌍의 Cosine similarity는 극대화하고 남은 N^2 - N개의 남은 쌍의 Cosine similarity는 최소화 한다. 이 연구에서 저자들은 ResNet-50과 ViT를, 이미지를 인코딩 하기 위한 모델로 고려했다. CLIP는 400 백만의 이미지-텍스트 쌍의, 대용량 말뭉치로 훈련시켰고 훌륭한 Zero-shot transfer 역량을 입증했다. 추론 시에는 클래스 이름이 Text-encoder의 입력으로 사용되고 인코딩된 이미지의 유사성이, Image-text 쌍에서 가장 매치가 잘 되는 쌍을 찾기 위해서 모든 인코딩된 텍스트와 함께 계산된다. CLIP는 ImageNet 훈련 셋에서의 Supervision 없이 75%의 Zero-shot classification accuracy를 보여줬다.  알아둬야 할 점은 ResNet의 CLIP가 592개의 V100 GPU에서 훈련시키는데 18일이 걸리고 256 V100 GPU에서 ViT를 훈련시키는데 12일이 걸린다는 것이다. 



### Transformers for Object Detection

Image classification과 유사하게 Transformer 모델은 CNN Backbone에서 추출된 이미지 특징으로 Bounding box와 레이블을 예측할 수 있다. 



#### Detection Transformer - DETR

Transformer 모델을 적용하기 위해서 DETR은 Object detection을 Set prediction 문제로 보았고 Set loss function을 제안했다. 이것이 의미하는 바는 Image feature들의 셋에 대해서 Object bb의 셋을 예측한다는 것이다. Transformer 모델이 (Single shot)으로 객체 셋의 예측을 가능하게 하고 그들의 관계성을 모델링 할 수 있게 한다. Set loss로는 예측과 GT 박스 양자 간의 매칭을 가능하게 한다. DETR의 주요 장점은 RPN, NMS 같은 수작업의 모듈과 연산에 대한 의존을 없앤다는 것이다. 이런 방식으로 Prior knowledge나 Engineering 디자인에 대한 의존성을 줄인다. 

![](./Figure/Transformers_in_Vision_A_Survey22.png)

주어진 CNN backbone에서의 Spatial feature map에 대해서 Encoder는 먼저 Spatial 차원을 하나의 차원으로 Flatten시킨다. 이 과정을 통해서 d x n의 Feature sequence가 만들어진다(d는 Feature 차원이고 n = h x w. h, w는 Spatial feature map의 높이와 너비이다). 이 Feature들은 Multi-head self-attention을 사용해서 인코딩 후에 디코딩한다. 디코딩 단계에서 주요 차이는 모든 박스가 병렬적으로 예측된다는 것이다(RNN을 사용하면 하나씩 만들어짐). 인코더와 디코더가 Permutation invariant(토큰이나 패치의 순서가 바뀌어도 동일한 결과를 도출)이기 때문에 학습된 Positional 인코딩이 각기 다른 박스들을 만들어내기 위해 Decoder에 의해서 Object query로써 사용된다. CNN detector(예를 들어 Faster R-CNN)에서 Spatial structure가 자동으로 Positional information을 인코딩 한다. DETR은 여러 다른 도메인에 사용되었는데 예를 들어서 Cell-DETR은 Biological cell의 Instance segmentation을 위해 사용되었다. Dedicated attention branch가 Instance-wise segementation을 위해서 더해 졌다. 



#### Deformable - DETR

DETR에서는 CNN과 Transformer를 결합해서 RPN이나 NMS 같은 수작업으로 수행하는 디자인 요소를 제거하고 End-to-End로 훈련시킬 수 있는 Object detection 파이프라인을 구축했다. 그러나 크기가 작은 객체를 탐지하는데 어려움을 겪고 성능 수렴이 느리며 상대적으로 높은 연산 비용이 발생했다. DETR은 Relation modeling을 위해서 Transformer를 사용하기 전에 이미지를 Feature space로 매핑한다. 그러므로 Self-attention의 연산적 비용이 Feature map의 Spatial 크기가 커질때 네제곱으로 커진다(O(H^2W^2C)). 이때문에 Multi-scale hierarchical feature를 사용하는 것에 제약이 있다. 그런데 Multi-scale hierarchical feature는 크기가 작은 객체를 탐지하는데 매우 중요하다. 게다가 훈련 시작 부분에서 Attention 모듈은 단순히 Attention을 균등하게 Feature map의 모든 위치에 Projection 시키고 유의미한 위치로 Attention weights가 수렴하도록 Tuning시키려면 많은 기간의 Training 에포크를 필요로 한다. 이 때문에 DETR의 성능이 수렴하는데 시간이 오래 걸린다. 이런 문제점을 경감하기 위해서 Deformable attention 모듈이 제안되었다. Deformable 컨볼루션에 영감을 얻어서 Deformable attention 모듈은 Feature map의 크기와 관계 없이 전체 Feature map에서 Sparse set의 요소들에만 신경 쓴다. 그렇기 때문에 연산적 비용을 크게 증가시키지 않으면서 Multi-scale attention 모듈의 도움으로 Cross-scale의 Feature map aggregation이 가능하다. Deformable DETR은 정확도적인 성능이 더 좋을 뿐만 아니라 훈련 시간 또한 본래의 DETR 모델보다 10배 정도 적다.  



## Open problems & Future directions

Transformer 모델이 각 도메인에서 준수한 성능을 보였지만(본문의 Table 1 참고) 여전히 한계는 존재한다(본문의 Table 2 참고). 가장 중요한 문제에는 훈련시 많은 양의 데이터가 필요하다는 것과 그만큼 많은 계산적 비용이 발생한다는 것이다. 또 Transformer 모델의 학습 상태를 시각화하고 해석화 하는 것에도 어려움이 있다. 



### High Computational Cost

Transformer 모델은 높은 Parametric complexity를 보인다. 그래서 작업을 하는데 필요한 Resource나 연산적 시간 측면에서 훈련시와 추론시에 높은 비용이 발생한다. 예를 들어서 BERT 기본 모델(109 백만의 모델 파라미터)은 1.89의 Peta-flop days만큼 훈련하는데 필요하고 GPT3 모델(175 백만의 모델 파라미터)는 3640의 Peta-flop만큼 훈련하는데 필요하다. 이런 큰 모델은 실제로 써먹으려면 Aggresive compressive technique이 필요하다. 

Language 도메인에서 최근 연구는 Transformer 모델의 높은 복잡도를 낮추는데 초점을 맞췄다(기본적으로 Self-attention 매커니즘에서 이런 복잡도가 발생하는데 Token의 Representation이 바로 전 계층에서의 모든 Token들을 고려하여 업데이트 된다). 예를 들어서 Linformer는 표준 Self-attention operation의 복잡도를 O(n^2)에서 O(n)로 줄인다(시간이나 메모리 요구랑 측면에서). 주요 아이디어는 Low-rank matrix가 Self-attention 매커니즘을 모델링 하는데 충분하다는 것을 보여주는 것이다. Reformer 모델은 Locally-sensitive hashing(LSH)를 도입해서 Self-attention의 복잡도를 O(n^2)에서 O(nlog(n))로 최소화 한다. 이와 유사하게 Lambda Networks는 Self-attention의 복잡도를 줄이는데 도움이 되는 Linear function을 Contex로 모델링하는 것을 제안한다. 

Vyas 등은 거의 본래의 Self-attention에 근접하는 큰 입력 시퀀스를 다루기 위한 효율적인 Cluster attention을 개발했다. 저자들은 Cluster attention에서 쿼리를 클러스터로 묶고 클러스터의 중심 사이의 Attention을 계산한다(네제곱의 복잡도를 보이는 모든 쿼리 사이의 Attention을 계산하는 것 대신에). 주요 아이디어는 Euclidean space 상에서 가까운 쿼리들은 유사한 Attention distribution을 가질 수 밖에 없다는 것이다. 고정된 수의 Cluster로 이런 직관적인 아이디어는 네제곱의 복잡도를 선형의 복잡도로 줄이는데 도움이 될 수 있다(입력 시퀀스 길이 n에 대해서, c는 클러스터의 숫자). 

NLP와 유사하게 Computer vision 모델은 Transformer 모델의 높은 연산 비용으로 고통 받는다. 예를 들어서 Sequence-based Transformer에 기반한 Image generator는(예를 들어서 iGPT) 높은 연산 비용 때문에, 높은 해상도의 입력에 적용하기가 힘들다. 이런 모델은 높은 차원의 입력으로 확장 될 수 있다. 예를 들어서 Multi-scale transformer 디자인과 Local context 모델링을 사용한다던가 하는 방법으로 가능하다.   



### Large Data Requirements

Transformer 아키텍처는 시각 데이터를 다루기 위해서 선천적으로 Inductive biases(Prior knowledge)를 인코딩하지 않기 때문에, 주어진 Task에 대한 Modality-specific rules을 Pre-training하면서 파악한다. 그래서 보통 많은 양의 훈련 데이터를 필요로 한다. 

CNN은 Pooling 연산이나 Multi-scale processing 블럭 덕분에 Patial scale invariance, Weight sharing, Translation invariance 같은 특성이 내재되어 있다. 그러나 Transformer 네트워크는 이런 이미지적인 특성들을 스스로 많은 양의 샘플을 보고 알아내야 한다. 예를 들어서 Video 프레임 간의 관련성을 자동으로 발견하기 위해서 Self-attention 매커니즘은 비디오 시퀀스 데이터 베이스 자료를 아주 많이 본다. 결과적으로 CNN보다 더 긴 훈련시간을 필요로 하고 연산적 부담이 상당히 증가하며 필요로 하는 데이터의 양이 더 많아진다. 

ViT 모델은 ImageNet 데이터셋으로 준수한 성능을 보이는 모델을 만들기 위해서 Pre-training 시에 수억의 이미지 데이터를 필요로 한다. 그래서 최근의 연구 중에는 Transformer 모델을 학습시키는데 데이터를 효율적으로 사용하는 방법이 없을까하는 질문으로 시작하는 연구가 많다. DeiT는 데이터 효율성을 위해서 Distillation 접근법을 사용했고 T2T(Tokens-to-Token) ViT에서는 Spatially 가까운 토큰들을 함께 결합하는 방식으로 Local structure를 모델링 하고 나서 Pre-training 없이 ImageNet 데이터셋으로 처음부터 훈련시켰더니 괄목할만한 성능을 보이는 모델을 만들게 되었다. 



### Vision Tailored Transformer Designs

저자들이 말하길 요즘 연구 추세가 Vision 분야에서는 연구에 Transformer를 적용하려는 경향이 있다고 한다. Image recognition, Video understanding, Multi-modal processing을 위한 아키텍처를 디자인하는데 Transformer 모델을 적용한다. 단순히 Transformer를 적용한 결과가 나쁘지 않고 그래서 연구자들이 좀 더 Self-attention이나 Self-supervised learning에 대해서 살펴보도록 하고 있지만 본질적으로 지금의 Transformer 아키텍처는 NLP에 더 맞게 디자인 되어 있고(시퀀스 구조) 이를 Visual 데이터에 효율적이도록 바꿀 여지가 남아 있다고 한다. 예를 들어서 Vector attention은 이런 아이디어면에서, Channel-wise attention을 통해 시각 데이터를 위한 Tailor self-attention 연산을 구현하려고 시도했다는 것에서 좋은 접근을 하고 있다고 할 수 있다. 유사한 방향성으로 Self-attention loss 기반의 Jigsaw puzzle을 Transformer에서 병렬 브랜치로 사용해서 Person re-identification을 개선하는 시도가 있다. 최근에 어떤 연구는 Spatially 가까운 토큰들을 재정렬해서 Spatially proxiaml location들을 더 좋게 모델링하는 시도를 했다. 



### Interpretability of Transformers

Transformer 모델에서 Classification decision과 관련성 있는 지역을 시각화해서 모델의 상태를 살펴보고 해석하는 것이 유의미할 때가 있다. 그런데 문제는 각 계층에서 계산된 Attention이 그 다음 계층들과 복잡한 방법으로 얽혀 있다는 것이다. 그래서 입력 토큰들이 최종 결정에 어떻게 기여했는가를 시각화 하는 것이 어렵다. 이에 대해 몇몇 연구들이 Transformer를 해석하는 것을 목적으로 해서 성과를 낸 바 있다. Attention roll-out이나 attention flow 방법들은 정확한 Attention을 측정하기 위한 기법이다. 그러나 이런 방법은 Ad-hoc 방식으로 기능하고 단순한 추측만 내릴 수 있다. 예를 들어서 입럭 토큰들이 모든 계층에 대해서 Attention weights를 사용해서 선형적으로 결합된다던지. Chefer 등이 말하길 Self-attention이나 Reassignment를 통해 직접적으로 얻은 Attention score는 최적의 솔루션이 될 수 없다고 한다. 이에 대한 대안으로 이들은 Transformer 네트워크에서 Relevancy score를 할당하고 전파해서 Relevancy의 합이 전체 네트워크에서 상수가 되는 것을 제안했다. 이들의 방법으로 Self-attention 계층에서 경험할 수 있는 Positive / Negative attribution들을 다룰수 있다. 여기서 제안된 프레임워크로 Class-specific 시각화를 할 수 있다는 추가적인 장점이 있다. 이런 연구에도 불구하고 Transformer를 시각화하고 해석하는 것은 풀 수 없는 문제이고 Spatially 정확한 Activation-specific 시각화를 할 수 있는 방법이 필요하다. 이런 방향에서의 추가적인 연구 성과가 있으면 Transformer 모델을 더 잘 이해할 수 있게 되고 모델이 결정을 내리는 과정 중에 Biases나 Erroneous behaviros에 대한 진단을 하는 것이 가능해 질 것이다. 그렇게 되면 이런 부정적인 요소를 배제하는 새로운 모델 아키텍처를 디자인 할 수 있게 된다. 



### Hardware Efficient Designs

큰 규모의 Transformer 네트워크는 강력한 성능의 장비와 많은 연산량을 필요로 하기 때문에 Edge device나 Resource가 제한된 환경에서 모델을 배치하는 것을 저해한다. 몇몇 연구에서는(FPGAs 같은) NLP 모델을 임베디드 환경에서 배치하기 위해서 모델을 압축하고 성능을 가속화 할 수 있도록 하는 노력을 했다. Li 등은 NLP 모델을 압축하기 위해서 Enhanced block-circulant matrix-based representation을 사용했고, 높은 처리량과 낮은 지연율을 위해 효율적으로 Resource를 관리하는 Field Programmable Gate Array(FPGA) 아키텍처 디자인을 제안했다. 이들은 CPU에서의 RoBERTa 모델과 비교했을때 27x, 3x 그리고 81x배 정도 성능 상의 향상(FPS를 통해 측정된 처리량), 완화된 전력 소비 그리고 효율적인 에너지 소비를 달성했다. 어떤 연구에서는 Hardware-Aware Transformer(HAT)를 디자인 하는 것을 제안했다. 여기서는 NAS 전략을 사용한다. 구체적으로 SuperTransformer 모델은, 먼저 모델을 완전하게 훈련하지 않고 성능을 집작하는 방식으로 먼저 훈련된다. 이 모델은 Search space에서 가장한 큰 모델로 구성되고 공통된 부분에서는 Weights를 공유한다. 그리고 최종적으로 Hardware latency 제약사항을 고려하여 Evolutionary search가 수행된다. 이것은 타겟 Hardware 플랫폼(IoT device, GPU, CPU와 같은)에서 돌아갈 수 있는 SubTransformer 모델을 찾기 위함이다. 그러나 이런 Hardware적으로 효율적인 디자인은 Resource적으로 제한이 있는 Device에서 성능 상의 결함 없이 Vision Transformer를 배치하는데  부족한 감이 있다. 



### Leveraging Rich Multi-modeal Annotaions

여러 도메인에서 훈련셋과 함께 Dense label을 사용할 수 있는 경우, 고려해볼 수 있는 질문은 작은 데이터셋에서의 Rich label에 의존하는 Pre-training 과정이 학습의 속도를 높일 수 있는 가이다. Virtex에서 이런 질문에 대한 탐구를 수행했다. 여기서 모델은 Image caption과 같은 Dense textual annotation을 사용해서 강한 Visual representaiton을 학습하는 것을 추구한다. 그래서 Caption은 이미지 내에 있는 객체에 대한 정보, 객체 간의 연관성, 행동이나 특성 정보를 인코딩해서, Generalizable하고 Transferable한 Representation을 학습할 수 있도록 더 나은 Supervision을 제공한다. 특히, Visual backbone과 Caption을 예측하기 위한 Bidirectional language model(Forward and backward transformer)로 훈련시킨 모델이 MS-COCO 데이터셋에서 비지도적인 양식으로 강한 Feature를 학습할 수 있다는 것을 보여줬다. 이렇게 학습된 Feature들이 ImageNet 모델로 Trasfer되었을때, ImageNet에서 직접적으로 학습된 Feature들과 견주어도 뒤지지 않거나 오히려 좋은 성능을 보였다. 그래서 Transformer 모델은 통합된 아키텍처에서 여러 Modality를 처리할 수 있다. 그리고 Densely annotated된 데이터셋이 Transformer 모델을 훈련시키는데 필요한 데이터 필요량을 줄이고 추론 시에 (입력 단에서) 보지 못한 조건에 Feature들이 Transfer가 잘 되는지 탐구 하는 것도 의미가 있다. 



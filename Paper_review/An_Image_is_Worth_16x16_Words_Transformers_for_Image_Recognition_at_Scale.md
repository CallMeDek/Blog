# AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby(Google Research, Brain Team)



## ABSTRACT

Transformer가 NLP 분야에서 사실상 표준이 된 반면에 컴퓨터 비전에 적용에 적용하는 하는 사례는 제한적이었다. 비전에서 Attention은 CNN과 같이 쓰이거나 CNN의 전체적인 구조는 남긴채 특정 부분만 Attention으로 바꾸는 방식으로 활용했다. 저자들은 CNN에 대한 의존이 꼭 필요한 것은 아니며 이미지 패치의 시퀀스에 직접적으로 적용하는 방식의 순수 Transformer가 Classification에서 좋은 성능을 보임을 확인했다. 많은 양의 데이터셋에서 미리 학습시키고 중간이나 적은 양의 데이터셋으로 전이 학습을 적용할때 Vision Transformer(ViT)는 최신 CNN과 비교해서 훌륭한 성능을 달성했을뿐만 아니라 훈련 시키는데 훨씬 적은 양의 계산 리소스만 필요했다. 



## INTRODUCTION

Self-attention 기반의 아키텍처들 중에 Transformer는 NLP에서 대세가 됐다. 주로 많은 양의 텍스트 말뭉치로 모델을 미리 훈련시키고 적은 양의 데이터셋으로 Fine-tuning했다. Transformer의 계산적 효율성과 확장석 덕분에 100B 파라미터 크기 이상의 모델을 훈련시키는 것이 가능해졌다. 모델과 데이터셋의 크기가 커쳐도 성능 상의 수렴은 보이지 않았다. 

그러나 컴퓨터 비전 분야에서는 CNN 아키텍처가 주류인채로 남아있었다. NLP에서의 성공 사례에 영감을 받아서 많은 연구에서 CNN 아키텍처과 Self-attention을 결합하려는 시도를 했다. 어떤 연구에서는 Convolution을 완전히 Attention으로 대체하려고 했다. 후자의 경우 이론적으로는 더 효율적인 모델을 만들 수 있으나 현대의 하드웨어 가속기에서 효율적으로 이런 모델을 적용하지는 못했는데 Specialized attention pattern을 사용하기 때문이었다. 그러므로 CNN 방식의 연구가 여전히 비전 분야에서는 SOTA의 자리를 차지하고 있었다. 

저자들은 표준 Transformer를 직접적으로 이미지에 적용하는 실험을 했다(NLP를 위해 생성한 모델의 구조에서 약간 변경해서). 그렇게 하기 위해서 저자들은 이미지를 패치들로 나누고 이런 패치들의 선형 임베딩의 시퀀스를 Transformer의 입력으로서 집어넣었다. 이미지 패치들은 NLP에서 토큰(단어)와 같이 취급된다. 저자들은 이 모델을 지도 학습 방식으로 Classification을 수행하도록 훈련시켰다. 

저자들이 ImageNet 같은 중간 크기의 데이터셋으로 훈련시켰을때 이런 모델들은 비슷한 크기의 ResNet보다 몇 포인트 더 낮은 정확도를 보였다. 저자들이 말하길 Transformer는 CNN에 내재되어 있는 (Translation equivariance, Locality 같은) Inductive bias가 부족하기 때문에 충분하지 못한 데이터로 훈련시키면 잘 일반화 시킬 수 없다고 한다. 

그런데 더 큰 데이터셋으로 훈련시킬때는 양상이 달라진다고 한다. 저자들의 ViT가 충분히 많은 양의 데이터셋으로 미리 학습시키고 나서 전이 학습을 적용했더니 준수한 성능을 달성했다고 한다. 



## RELATED WORK

Transformer는 원래 기계번역을 위해서 Vaswani등이 제안했고 NLP 분야에서 SOTA의 자리를 차지했다.  크기가 큰 Transformer 기반의 모델은 주로 양이 많은 말뭉치로 미리 학습시키고 수동으로 특정 도메인을 위해 Fine tuning시켰다. (BERT, GPT 등)

이미지에 Self-attention을 적용하는 Native한 방식은 각 픽셀이 모든 다른 픽셀을 참고하도록 했다. 그런데 픽셀 숫자의 제곱에 해당하는 Cost 때문에 실제 크기를 가진 이미지에 적용할 수 없었다. Transformer를 이미지 처리 과정에 적용하기 위해서 이와 근사한 방식의 연구가 수행되었다. Parmer 등은 각 Query에서 픽셀들이 오직 주변의 이웃 픽셀만 참고하도록 하는 Self-attention을 적용했다. 이런 Local multi-head dot-product self attention 블럭은 여러 연구에 따르면Convolution을 대체할 수 있다고 한다. 이와는 다르게 Sparse Transformer의 경우 이미지에 적용가능하도록 Global self-attention을 고안했는데 참고 근사정도를 조절가능하게 했다. Attention의 정도(크기)를 조절하는 다른 방법은 다양한 크기의 블럭에 적용하거나 각 축을 따라 따로 적용하는 방법도 있다. 이런 특화된 Attention 아키텍처가 컴퓨터 비전 분야에서 괜찮은 성능을 보여줬지만 하드웨어 가속기에서 효율적으로 구현하는데에는 복잡한 엔지니어링이 필요하다고 한다. 

CNN을 Self-attention과 결합하는 시도도 많았다. Classification을 위한 Feature map을 Augmenting한다던가 Self-attention을 사용하여 CNN의 출력 결과를 좀 더 처리하다던가 하는 방식이 있다. 

저자들이 말하길 저자들의 연구 이전에 Transformer를 원본 이미지 크기에서 Global self-attention을 적용한 연구 사례는 찾지 못했다고 한다. iGPT와 저자들의 모델이 유사하다고 한다. 여기서는 Transformer를 이미지 해상도와 Color space를 줄이고 난 이미지 픽셀들에 적용했다. 모델은 생성 모델 부분에서는 비지도적인 방식으로 학습했고 Classification 성능을 위해서 출력 Representation을 Fine-tuning하거나 선형적으로 탐사했다고 한다. 

저자들은 표준 ImageNet 데이터셋보다 많은 양의 데이터셋에서 Image recognition을 수행하는 것에 관심이 있었다. 추가적으로 데이터를 사용하면 표준 Benchmark에서 SOTA의 성능을 보인다고 한다. 그밖에 저자들이 참고한 연구 논문으로는 다음과 같은 것들이 있다. Sun 등은 CNN 성능이 데이터셋의 크기에 따라 달라지는 지를 연구했고 Kolesnikov 등과 Djolonga 등은 큰 데이터 셋에서의 CNN 전이 학습과 관련된 연구를 수행했다. 저자들은 ImageNet-21k, JFT-300M 데이터셋으로 실험을 수행했다. 
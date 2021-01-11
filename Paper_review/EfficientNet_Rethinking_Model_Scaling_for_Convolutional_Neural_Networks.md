# EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

Mingxing Tan, Quoc V. Le(1Google Research, Brain Team, Mountain View, CA)



## Abstract

저자들은 이 연구에서 Model scaling 방법에 대해서 연구하고 네트워크의 너비, 깊이, 입력 Resolution을 균형적으로 Scaling하는 것이 임의로 Scaling 하는 것보다 더 좋은 성능 향상으로 이어짐을 확인했다. 이를 바탕으로 Compound coefficient라고 하는, 깊이 너비, Resolution의 차원을 균등하게 Scaling하는 방법을 제시했다. MobileNet 계열과 ResNet에서 이 방법의 효과를 입증했다.  그리고 저자들은 저자들이 제시한 방법을 적용했을때 효과를 늘리기 위한 저자들만의 네트워크 아키텍처인 EfficientNet을 고안했다.

[tpu-models-official-efficientnet/]( https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)



## Introduction

컨볼루션 네트워크의 경우 Scaling up을 통해서 더 좋은 성능을 꾀하기 마련이다. 예를 들어서 ResNet은 깊이를 더 늘려서 Scaling up을 수행한다. GPipe의 경우에는 Baseline 모델보다 4배 더 깊이를 키워서 ImageNet 데이터셋에서 84.3%의 Top-1 accuracy를 달성했다. 그러나 저자들이 연구를 수행할 당시에는 Scaling up 하는 방식에 대한 깊은 연구가 이뤄지지 않았다. 가장 대표적으로 Scaling up을 수행하는 방법은 모델의 깊이나 너비를 넓히는 것이다. 이보다는 덜 일반적이긴 하지만 Image resolution을 키우는 방법도 있다. 이런 요소를 임의로 조정하는 것은 매우 번거로운 일일 뿐더러 가장 최적의 정확도와 효율성을 보장할 수도 없었다. 

저자들의 연구는 이런 요소를 조정하는데 효율적인 방법이 없을까하는 질문에서 기인한다. 저자들은 이런 요소들을 일정한 비율로 균형감 있게 조정해야 좋은 성능을 얻을 수 있다는 것을 관찰했고 이를 바탕으로 Compound scaling method라는 것을 제안했다. 예를 들어서 네트워크의 용량을 2^N만큼 키우고 싶다면 직관적으로 생각했을때 깊이, 너비, 이미지 크기를 각각 α^N, β^N, γ^N(여기서 α, β, γ는 모델의 용량을 키우기 전의 작은 네트워크에서 그리드 서치로 찾아낸 상수 Ratio들)만큼 키우면 된다. 

![](./Figure/EfficientNet_Rethinking_Model_Scaling_for_Convolutional_Neural_Networks1.JPG)

저자들의 방법이 직관적으로 말이 되는 이유는 생각해보면 만약에 입력 이미지가 커지면 이를 위한 Receptive field도 커져야 할 것이고 큰 이미지에 대한 Fine-grained pattern을 좀 더 잘 캐치하기 위해서 채널 수도 커져야 할 것이다. 저자들은 저자들의 방법이 MobileNet, ResNet 계열의 모델에서 잘 적용되는 것을 관찰했다. 그러면서 말하길 Baseline 네트워크가 중요하다고 말했다. 저자들의 연구를 위해서 새로운 모델 아키텍처인 EfficientNet을 고안해냈다. 아래 Figure 1은 ImageNet 데이터셋에 대한 각 아키텍처의 성능을 나타낸 것이다. 

![](./Figure/EfficientNet_Rethinking_Model_Scaling_for_Convolutional_Neural_Networks2.JPG)



## Related Work

### ConvNet Accuracy

관련 연구 내용은 본문 참조. 
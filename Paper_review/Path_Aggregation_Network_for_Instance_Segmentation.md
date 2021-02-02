# Path_Aggregation_Network_for_Instance_Segmentation

Shu Liu(The Chinese University of Hong Kong),

Lu Qi(The Chinese University of Hong Kong),

Haifang Qin(Peking University),

Jianping Shi(SenseTime Research),

YouTu Lab, Tencent



## Abstract

저자들은 Path Aggregation Network(PANet)을 통해서 Proposal 기반의 Instance segmentation 모델에서 정보가 잘 전파될 수 있도록 하는 것을 목표로 했다. 특히 전체적으로 낮은 단계에 있는 비교적 이미지 내 객체 위치 정보가 정확한 특징을 Shortcut을 통해서 가장 높은 단계에 있는 특징에 빠르게 전파할 수 있도록 했다. 그리고 Adaptive feature pooling이라는 개념을 도입하여 모든 단계에 있는 특징들이(Feature extraction 단계를 통해 추출된 해상도가 다른 특징들) Proposal subnetworks에 직접적으로 잘 전파될 수 있도록 했다. 마지막으로 Mask prediction을 위해서 추가적인 브랜치가 추가되었다. 이 브랜치에서는 각 Proposal의 여러 부분을 캡처한다.  

[ShuLiu1993-PANet](https://github.com/ShuLiu1993/PANet)



## Introduction

CNN의 출현과 발전에 힙 입어서 여러 Instance segmentation framework들이 등장하게 되었는데 저자들은 그 중에서 Mask R-CNN에 특히 관심이 있었던 듯 하다. 그뿐만이 아니라 Mask prediction을 위한 FCN이라던지 모델의 높은 정확도를 위해서 네트워크 내에서 여러 해상도의 특징들을 사용하는 FPN에도 관심이 있었다.  

데이터에 관해서는 COCO, Cityscapes, MVD 등에 관심을 가진듯 하다. 

모델 구조와 관련해서는 Residual connection, Dense connection을 통해서 Information flow를 위한 Shortcut이라던지 Propagation을 상대적으로 더 쉽게 한다던지 하는 방법에 관심이 있었다. 또 Split-transform-merge strategy를 통해서 병렬적인 경로를 만들어서 Information flow를 유연하고 다양하게 하는 방법도 유익하다고 했다. 

![](./Figure/Path_Aggregation_Network_for_Instance_Segmentation1.JPG)



### Our Findings

저자들이 주장하길 Mask R-CNN은 좀 더 개선될 여지가 있다고 한다. 특히 저차원과 고차원 사이의 깊이가 깊으면, 객체의 정확한 위치 정보를 담고 있는 저차원의 정보들이 고차원의 특징에 전달되기가 어렵다고 한다. 그리고 각 Proposal의 경우 한 가지 Feature level에서 모아진 Feature grid에 근거해서 예측이 수행되는데 이 Feature들이 할당되는 것도 경험적으로 할당되는 것이기 때문에 어떤 Level에서 버려진 특징들이 최종 예측에는 도움이 될 수 있다는 것이 저자들의 주장이다. 이렇게 되면 Mask prediction의 경우 다양한 정보를 얻게 될 기회를 잃게 된다. 



### Our Contributions

저자들은 Figure 1과 같이 PANet이라고 하는 Instance segmentation을 위한 아키텍처를 발표했다.  

우선 Backbone 때문에 길어질 수 있는 정보 흐름을 짧게 할수 있게 하고, 저차원 단계의 위치 정보를 잘 활용하기 위해서 Bottom-up path augmentation이라는 개념을 창안했다. 

그리고 각 FPN의 각 단계의 Proposal들과 Feature들이 예측 시에 나눠서 할당되기 때문에 발생할 수 있는 정보 손실을 줄이기 위해서 Adaptive feature pooling이라는 것을 만들었다. 여기서는 임의로 Feature들을 나누는 것 대신에 각 Proposal을 위한 모든 단계의 Feature들을 모은다. 

마지막으로 각 Proposal를 여러 관점으로 보고 해석하기 위해서 Mask prediction 브랜치에서, 완전 연결 계층을 추가했다. Mask R-CNN에서 FCN으로만 Mask prediction을 수행한 것과는 대조적으로 FCN 경로와 병렬적으로 완전 연결 계층 경로를 추가했다. 이렇게 두 경로에서의 결과를 Fusing했더니 정보 다양성이 향상되서 Mask prediction 결과가 더 좋아졌다고 한다. 

설명한 요소 중 앞의 두 가지 요소는 Object detection, Instance segmentation 수행시에 공유된다고 한다. 



## Related Work

### Instance Segmentation

Instance segmentation에는 두 가지 계보가 있다고 한다. 첫째는 Proposal 기반의 방법인데 이 방법은 Object detection과 강한 연관성이 있다. 

다른 하나는 Segmentation 기반의 방법이다. 여기에는 특수하게 설계한 Transformation이나 Instance boundaries가 있고 Instance mask는 Predicted transformation 결과에서 디코딩된다. 

그밖의 다른 연구 결과도 있는데 본문 참조. 



### Multi-level Features

여러 계층이나 단계에서의 Feature들을 이용하는 방법과 관련된 여러 연구들은 본문 참조. 여기서 저자들은 FPN을 기본으로 삼고 발전시켜나갔다고 한다. 

ION에서는 여러 계층에서의 Feature grid들을 Concatenation했다고 하는데 이 연산 과정 중에서 Normalization, Concatenation, Dimension reduction이 현실적으로 계산 가능한 Feature들을 만들어내기 위해 필요했다고 한다. 그러나 저자들의 방법에서는 이보다는 더 간단하다고 한다. 

또 아래 연구에서도 각 Proposal을 위한 여러 곳에서의 Feature grid를 Fusing하기도 했는데

- S. Ren, K. He, R. B. Girshick, X. Zhang, and J. Sun. Object detection networks on convolutional feature maps. PAMI, 2017.

이 방법에서는 입력 이미지의 크기를 다르게 한 뒤에 Feature map을 뽑아냈고 Input image pyramid에서 Feature selection을 하는 연산을 개선하기 위해서 Feature fusion을 수행했다고 한다. 그러나 저자들의 방법은 단일 크기의 입력으로 네트워크 내의 Feature hierarachy 안에 모든 Feature level에서의 정보를 활용하는데 목적을 둔 것에 차이점이 잇다고 한다. End-to-End training도 가능하다. 




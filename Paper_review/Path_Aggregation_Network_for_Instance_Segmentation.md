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



### Our Findings

저자들이 주장하길 Mask R-CNN은 좀 더 개선될 여지가 있다고 한다. 특히 저차원과 고차원 사이의 깊이가 깊으면, 객체의 정확한 위치 정보를 담고 있는 저차원의 정보들이 고차원의 특징에 전달되기가 어렵다고 한다. 그리고 각 Proposal의 경우 한 가지 Feature level에서 모아진 Feature grid에 근거해서 예측이 수행되는데 이 Feature들이 할당되는 것도 경험적으로 할당되는 것이기 때문에 어떤 Level에서 버려진 특징들이 최종 예측에는 도움이 될 수 있다는 것이 저자들의 주장이다. 이렇게 되면 Mask prediction의 경우 다양한 정보를 얻게 될 기회를 잃게 된다. ㄴ


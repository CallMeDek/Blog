# Feature Pyramid Networks for Object Detection

Tsung-Yi Lin(Facebook AI Research, Cornell University and Cornell Tech),

Piotr Doll´ar(Facebook AI Research),

Ross Girshick(Facebook AI Research),

Kaiming He(Facebook AI Research),

Bharath Hariharan(Facebook AI Research),

Serge Belongie(Cornell University and Cornell Tech)



## Abstract

객체 인식 시스템에서 다른 크기의 객체를 탐지하기 위해서 크기가 다른 Feature를 사용하는 것은 당연한 일이었다. 그러나 이는 연산 시간이나 메모리를 상당히 잡아먹었다. 그래서 저자들은 Top-down 방식의 Lateral Connection 개념을 적용하여 질 좋은, 각 스케일의  Semantic Feature map을 예측에 사용할 수 있는 아키텍처인 Feature Pyramid Network(FPN)을 제안했다. 



## Introduction

![](./Figure/Feature_Pyramid_Networks_for_Object_Detection1.JPG)

원래는 Figure1의 a와 같이 이미지를 여러 크기로 재조정한 뒤에 예측을 진행하는 방식을 많이 사용했다. 이런 방식은 기본적으로 각 크기의 이미지에서 객체의 크기 변화를 Offset 개념으로 보았기 때문에 Scale-Invariant 했다. 그래서 모델이 넓은 범위의 스케일로 객체를 탐지하는 것을 가능하게 했다. 

CNN이 나오기 이전에 사람이 직접 Feature를 엔지니어링 해야했던 DPM 같은 알고리즘에서는 이런 방식이 많이 쓰였다. 그러다가 ConvNets 같은 DCNN이 등장하게 되면서 객체 인식 분야에서 많이 사용되었다. Figure1의 b와 같이 DCNN 기반의 방식들은 단일 스케일로도 객체의 크기 변화에 어느정도 Robust했다. 그러나 여전히 다양한 스케일에 대해서 더 정확한 결과를 내기 위해서는 a와 같은 방식을 차용할 필요가 있었다. 왜냐하면 a와 같은 방식은 고해상도의 Feature를 포함해서 모든 단계에서 Semantically Strong하기 때문이었다. 그럼에도 불구하고 a와 b의 방식을 섞어서 쓰는 방법은 추론 시에 속도가 느려서 실시간성이 필요한 애플리케이션에는 적용하지 못한다던가, 모델 훈련 간에는 메모리 용량때문에 적용하지 못한다던가 하는 제한사항 때문에 테스트 시에만 적용했는데 이는 훈련과 테스트 시에 모델의 추론 결과를 다르게 만들었다.  

a와 같은 방식 뿐만 아니라 c와 같은 방식으로도 Multi-scale Feature 표현이 가능하다. 계층별로 Feature를 뽑아내서 예측을 하는 방식은 다양한 크기의 객체를 탐지할 수 있긴 하지만 계층의 깊이별 Semantic Information에 대한 격차가 존재한다. 특히 저단계의 고해상도 Feature map 같은 경우에는 객체 인식 성능에 악영향을 끼칠 수 있다. SSD는 이런 방식을 거의 최초로 도입한 알고리즘 중 하나이다. SSD는 순전파때 여러 크기의 Feature map을 예측에 사용하기 때문에 a 방식과 비교했을 때 Cost가 거의 없다 시피한다. 그러나 저단계의 Feature map을 사용하는 것을 피하기 위해서 모든 계층에서 계산된 Feature map을 사용하는 것 대신에 상위 단계 계층에서 Feature Pyramid를 구축하여 Feature map을 사용한다. 그래서 고해상도 Feature map을 다시 사용할 수 없는데 저자들에 따르면 이 고해상도 Feature map이 작은 객체를 탐지하는데 중요한 역할을 한다고 한다.

그래서 저자들이 말하는 이 연구의 목적은 모든 단계에서 Strong Semantic Information을 포함하는 DCNN 기반의 네트워크를 만드는 것이라고 한다. 그림1의 d방식과 같이 Top-down pathway 구조에 Lateral Connection 연산을 통해서, Semantically Strong한 고레벨의 저해상도 Feature map과 Semantically Weak한 저레벨의 고해상도 Feature map을 결합한다. 이렇게 함으로서 모든 단계에서 Semantic이 풍부한 Feature map을 만들어내면서도 단일 이미지 사이즈에서 이 모든 프로세스를 진행하기때문에 빠른 네트워크를 구축할 수 있다고 한다. 이런 방식은 저자들이 말하는 기존 방식의 문제점인 표현력, 속도, 메모리의 어떤 희생 없이 Multi-scale의 탐지를 가능하게 한다. 

![](./Figure/Feature_Pyramid_Networks_for_Object_Detection2.JPG)

물론 저자들의 연구 전에도 비슷한 방식의 Top-down + Skip connection의 구조를 이용한 연구가 많았다. 이들의 목적이 예측을 진행할 단일의 고해상도 Feature map을 만들어 내는 거라면(Figure 2의 위의 방식) 저자들은 이와는 좀 다르다. 저자들은 Figure 2의 아래와 같이 비슷한 구조를 차용하지만 각 단계에서 독립적으로 예측이 진행된다. 

저자들은 이 FPN을 다양한 Detection, Segmentation 시스템에 적용하여 평가했다고 한다. 그 어떤 추가적인 개념을 도입하지 않고, 단순히 FPN과 기본적인 Faster R-CNN Detector를 결합한 네트워크를 사용했을때, 다른 Heavily-engineered Single Model들을 압도하면서 COCO Detection Benchmark에서 제일 좋은 성능을 보였다고 한다. Ablation 실험에서는 Bounding Box Proposal 생성에 대해서 기본 Faster R-CNN + ResNet 네트워크에 비해서 AR은 8 포인트, COCO 스타일의 AP에서는 2.3 포인트, PASCAL 스타일의 AP에서는 3.8 포인트 이상 성능이 더 좋아졌다고 한다. 또 FPN은 Mask Proposal에도 쉽게 적용가능한데 Image Pyramid에 의존하는 방법보다 AR과 속도 면에서 개선이 이루어졌다고 한다. 

FPN을 적용해도 종단 간 학습이 가능하고 메모리를 크게 잡아 먹지 않기 때문에 Image Pyramid와 비교했을 때, 훈련과 테스트 결과가 동일하고, 더 정확하며 속도가 빠르다는 장점이 있다고 한다. 



## Related Work

### Hand-engineered features and early neural networks

- SIFT - D. G. Lowe. Distinctive image features from scale-invariant keypoints. IJCV, 2004.
- HOG - N. Dalal and B. Triggs. Histograms of oriented gradients for human detection. In CVPR, 2005.

SIFT, HOG 같은 알고리즘은 Image Pyramid의 모든 단계에서 Feature를 추출해낸다. 

- P. Doll´ar, R. Appel, S. Belongie, and P. Perona. Fast feature pyramids for object detection. TPAMI, 2014.

위와 같이 빠르게 Image Pyramid에서 특징을 뽑아내려는 시도도 있었다. 

HOG, SIFT 전에 옛날의 ConvNets에서의 Face Detection 같은 경우에는 얇은 Image Pyramid 각 층에서 탐지를 수행했다. 



### Deep ConvNet object detectors

- Overfeat - P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus, and Y. LeCun. Overfeat: Integrated recognition, localization and detection using convolutional networks. In ICLR, 2014.

OverFeat의 경우 기존의 방식과 같이 Image Pyramid에 ConvNet을 Sliding Window Detector로서 적용했다. 

R-CNN에서는 Region Proposal 생성에 Selective search를 적용했고 각 Proposal들은 ConvNet으로 분류 프로세스를 진행하기 전에 Scale-normalized되었다. 

SPPnet에서는 단일 이미지 스케일에서 추출된 Feature map이 Region-based 탐지 방법에 효율적으로 적용될 수 있는 방법을 제안했다. 

Fast R-CNN, Faster R-CNN의 경우 단일 이미지 스케일에 다양한 크기와 종횡비의 Anchor를 사용하여 준수한 성능의 탐지를 가능하게 했다.  그럼에도 불구하고 여러 이미지 스케일에서의 탐지가 더 좋은 성능(특히 작은 객체)을 보였다. 



### Methods using multiple layers

- FCN - J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for semantic segmentation. In CVPR, 2015.

FCN에서는 Semantic Segmentation을 위해서 여러 크기의 Feature map에서 각 카테고리의 부분 점수를 합쳤다. 

- Hypercolumns - B. Hariharan, P. Arbel´aez, R. Girshick, and J. Malik. Hypercolumns for object segmentation and fine-grained localization. In CVPR, 2015.

Hypercolumns도 FCN와 비슷한 접근 방식을 Object Instance Segmentation을 위해서 적용했다고 한다.

- ION - S. Bell, C. L. Zitnick, K. Bala, and R. Girshick. Insideoutside net: Detecting objects in context with skip pooling and recurrent neural networks. In CVPR, 2016.
- HyperNet - T. Kong, A. Yao, Y. Chen, and F. Sun. Hypernet: Towards accurate region proposal generation and joint object detection. In CVPR, 2016.
- ParseNet - W. Liu, A. Rabinovich, and A. C. Berg. ParseNet: Looking wider to see better. In ICLR workshop, 2016.

위 방법들은 예측을 수행하기 전에 여러 계층의 Feature들을 Concatenation했다고한다.

- SSD - W. Liu, D. Anguelov, D. Erhan, C. Szegedy, and S. Reed. SSD: Single shot multibox detector. In ECCV, 2016.
- MS-CNN - Z. Cai, Q. Fan, R. S. Feris, and N. Vasconcelos. A unified multi-scale deep convolutional neural network for fast object detection. In ECCV, 2016.

위의 방법들은 여러 계층의 Feature map에서 예측을 수행하기는 했으나 Feature들이나 Scores를 합치지는 않았다.

- U-Net(Segmentation)  - O. Ronneberger, P. Fischer, and T. Brox. U-Net: Convolutional networks for biomedical image segmentation. In MICCAI, 2015.
- SharpMask(Segmentation) - P. O. Pinheiro, T.-Y. Lin, R. Collobert, and P. Doll´ar. Learning to refine object segments. In ECCV, 2016.
- Recombinator networks(Face detection) - S. Honari, J. Yosinski, P. Vincent, and C. Pal. Recombinator networks: Learning coarse-to-fine feature aggregation. In CVPR, 2016.
- Stacked Hourglass networks(Keypoint estimation) - A. Newell, K. Yang, and J. Deng. Stacked hourglass networks for human pose estimation. In ECCV, 2016.

위의 방법들과 같이 Lateral/Skip Connection을 적용하여 저단계의 Feature map을 여러 해상도와 Semantic 단계에 연관시키는 기법들도 있었다. 

Ghiasi 등은 FCN을 위해, Segmentation 방식을 획기적으로 개선하는, Laplacian pyramid presentation을 발표했다. 이런 방법들이 Pyramid 구조로 아키텍처를 바꾼것처럼 보이긴 하나 그림2와 같이 각 단계에서 예측이 독립적으로 수행되는 것은 아니었다. 오히려 그림2의 위와 같은 구조가 Multiple scale에서 수행되었다.  



## Feature Pyramid Networks

저자들은 여기서 연구 목적을 정확하게 제시했다. 저단계부터 고단계의, Semantic Information이 포함되어 있는 ConveNet의 Pyramidal Feature Hierarchy에 전체적으로 높은 수준의 Semantics을 가진 Feature Pyramid를 구축하는 것이다. FPN은 범용적으로 사용가능하지만 저자들은 이 연구에서는 RPN, Fast R-CNN과 연관시켜 연구를 진행했다고 한다. 또, FPN을 Instance Segmentation Proposal에도 적용했다고 한다. 

FPN은 단일 임의 크기의 이미지를 입력으로 받아, Fully Convolutional 양상 안에서 여러 단계에서 크기가 비례적으로 정해지는 Feature map을 출력한다. Backbone Architecture와 성능은 크게 상관이 없으며 여기서는 ResNet을 사용했다고 한다. 전체적인 구조는 Bottom-up Pathway, Top-down Pathway, Lateral Connection으로 구성되어 있다. 



### Bottom-up pathway

이 구조는 Backbone ConvNetdptj 순전파를 수행한다. 각 Stage에서는 여러 크기의 Feature map을 출력하는데 두 단계의 스케일링 과정을 포함한다. 같은 크기의 Output map을 출력하는 경우 같은 Stage에 있다고 말한다. 후술할 Lateral Connection을 위해서는 각 Stage의 마지막 출력 Feature map을 사용한다. 왜냐하면 각 Stage의 가장 마지막 Feature map에서 각 특징이 가장 잘 드러나기 때문이다. 

특히, ResNet에서는 각 Stage의 마지막 Residual Block에서의 Feature activations 출력을 사용한다. 이런 마지막 Residual Block의 출력을 Conv2, Con3, Conv4, Conv5에 대한 출력이라고 해서 {C2, C3, C4, C5}라고 언급한다. 이들의 Stride는 원본 입력 이미지에 대하여 각각 {4, 8, 16, 32}가 된다. Conv1은 메모리 제약 사항 때문에 Pyramid에 포함시키지 않는다. 



### Top-down pathway and lateral connections

이 구조에서는 가장 상위 Pyramid 단계의 Feature map의 Upsampling하여 Spatially coarser하지만 Sementically Stronger하게 만든다. 그 다음에 Lateral Connection을 통해서 Bottom-up pathway에서의 각 Stage의 Reference feature map을 통해서 보완한다.  Bottom-up Pathway의 Feature map의 경우는 Semantic Information이 적지만 Subsampling을 많이 하지 않아서 좀 더 각 객체의 위치 정보가 정확하다. 

 ![](./Figure/Feature_Pyramid_Networks_for_Object_Detection3.JPG)

[갈아먹는 Object Detection - 갈아먹는 Object Detection  Feature Pyramid Network]( https://yeomko.tistory.com/44)



![](./Figure/Feature_Pyramid_Networks_for_Object_Detection4.JPG)

그림 3에서 보면 Nearest Neighbor Upsampling을 통해서 상위 단계의 Feature map의 해상도를 2배로 늘리고 똑같은 크기의 Bottom-up pathway의 Feature map과 Element-wise Addition을 수행하는 것을 볼 수 있다. 이때,  Bottom-up pathway의 Feature map은 1x1 Convoltuion 연산을 통해서 채널 수를 맞춰준다. 이런 프로세스는 가장 해상도가 높은 Feature map(P2)을 만들어 낼때까지 반복된다. Iteration의 가장 처음에는 단순히 C5에 1x1 컨볼루션 계층을 덧붙여서 Coarsest Resolution map을 만들어낸다(P5). 각 Stage의 합쳐진 Feature map에는 3x3 Convolution 연산을 통해서 Upsampling을 통해서 발생하는 Aliasing Effect을 감소시킨다.  

![](./Figure/Feature_Pyramid_Networks_for_Object_Detection5.JPG)

Pyramid의 모든 단계에서 공통의 Classifier/Regressor를 사용하기 때문에 저자들은 Feature Dimension(# of Channels, d라고 표기)를 고정시켰는데 이 연구에서는 256으로 고정했다고 한다. 따라서 Backbone Architecture를 제외하고 새롭게 추가된 컨볼루션 계층의 모든 차원수는 256으로 고정되고 비선형은 추가되지 않는다. 



## Applications

저자들에 의하면 FPN은 범용적으로 사용이 가능하지만 여기서는 RPN에서 BB proposal 생성과 Fast R-CNN의 객체 탐지를 위해서 적용하는 실험을 했다고 한다. FPN이 단순 명료하면서 효과적임을 입증하기 위해서 원본 시스템을 최대한 변경하지 않으면서 적용했다고 한다. 



### Feature Pyramid Networks for RPN

![](./Figure/Feature_Pyramid_Networks_for_Object_Detection6.JPG)

기존의 RPN의 구조는 위 그림과 같다. 저자들은 여기서 위의 3x3 Convolution과 1x1 Convolution Braches를 묶어서 네트워크의 Head라고 언급했다.  [갈아먹는 Object Detection - 갈아먹는 Object Detection  Feature Pyramid Network]( https://yeomko.tistory.com/44)

![](./Figure/Feature_Pyramid_Networks_for_Object_Detection7.JPG)

[갈아먹는 Object Detection - 갈아먹는 Object Detection  Feature Pyramid Network]( https://yeomko.tistory.com/44)

Backbone Architecture에서 연산을 수행하여 각 Stage별 Feature map을 생성해두고 Lateral Connection을 통해서 Merged Feature map을 만들어 내는데 이를 Intermediate Feature map이라고 한다. 원래의 FPN은 각기 다른 크기의 Anchor들을 Feature map에 적용하여 Head에서 결과를 뽑아내지만 FPN에서는 각 Stage의 Intermediate Feature map의 해상도가 다르므로 다른 크기의 Anchor들을 적용한 것으로 볼 수 있다. 따라서 각 Intermediate Feature map에서 단일 크기의 Anchor box에 대해서 Head를 적용해서 Proposal에 대한 손실을 계산한다. 상위 Feature map은 상대적으로 크기가 큰 물체에 대한 정보를 담고 있을 것으로 예상 가능하고 반대로 하위 Feature map은 작은 물체에 대한 정보를 담고 있을 것으로 예상할 수 있다. 각 단계별 Anchor 박스의 크기는 {P2, P3, P4, P5, P6}에 대해서 {32x32, 64x64, 128x128, 256x256, 512x512}로 하고 종횡비 {1:2, 1:1, 2:1}를 적용해서 총 15개의 Anchor 박스를 사용하게 된다. 

저자들은 BB와 GT 박스의 IoU에 근거하여 각 Anchor에 레이블을 할당했다. 주의할 점은 GT 박스의 크기와 관련해서 각 Stage의 Anchor 박스에 GT를 할당한 것이 아니라 IoU에 근거하여 GT 박스를 각 Anchor에 할당했다는 점이다.  Head의 모든 가중치는 각 Stage별로 공유되는데 저자들이 확인하길 공유하지 않았을 때와 성능 차이가 크지 않았다. 이는 각 Pyramid 단계가 비슷한 Sementic 수준을 보유한다는 것을 반증하다고 주장했다. 이는 Image Pyramid에서 각 이미지 크기에 동일한 Head Classifier를 적용한 것과 맥락을 같이 한다. 
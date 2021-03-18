# YOLOv4

- Densely Connected Convolutional Networks  - 네크워크의 선행 계층이 후행의 모든 계층에 연결되어 있는 구조. 이를 통해서 그래디언트 소실 문제를 경감, 정보(특징) 전파력 강화, 정보(특징) 재사용, 네트워크 파라미터 수 감소의 효과. 

  - CSPNet에서 연구 개념을 설명할때 주요 예시가 DenseNet. Transition Layer, Growth rate, DenseBlock 등의 개념을 이해할 필요성.

- CSPNet: A New Backbone that can Enhance Learning Capability of CNN - Cross Stage Partial Network의 줄임말. 네트워크의 각 Stage에서의 시작과 끝의 Feature map을 Concatenation을 하는데, 시작 부분의 Feature map을 두 부분으로 나눈다. 이렇게 하면 역전파 시의 그래디언트 정보가 복제되는 것을 막아서 네트워크의 용량을 줄이고 최적화 시킬 수 있다. 

  - YOLOv4의 Backbone은 CSPNet의 개념을 적용하므로 CSPNet의 개념을 이해할 필요성. 

- YOLOv4: Optimal Speed and Accuracy of Object Detection - 새로운 개념을 제시하기 보다는 YOLOv3에 여러가지 기법을 적용해서 성능을 끌어올리고 테스트해본 Technical Report. 비싼 GPU 말고 보편적으로 사용되는 (GTX 1080Ti, RTX 2080Ti) GPU 1대로도 누구나 좋은 성능을 내는 Object Detector를 만들어 낼 수 있도록 하는 것이 저자들의 목적. 

  

# Pooling Pyramid Network for Object Detection

-   Pooling Pyramid Network for Object Detection - SSD 계열의 Detector들의 성능은 유지하면서 모델의 크기를 획기적으로 줄이는 방법을 제시. SSD에서 각 Feature map마다 Head가 존재한 것과 달리 하나의 Head로 모든 Feature map에서 Detection 수행. 



# Single-Shot Refinement Neural Network for Object Detection

- Single-Shot Refinement Neural Network for Object Detection - One-stage Detector의 순전파 시에 네트워크를 두 부분의 나눈다. 

  1. Anchor refinement module - 본격적으로 Detection을 수행하기 전 Anchor들의 Search space를 줄이기 위해서 Negative anchor를 필터링하고 Detection 모듈이 작업을 수행할 때 잘 수행할 수 있도록 어느정도 Positive anchor들을 조정하여 제공.
  2. Object detection module - 위의 모듈에서 조정된 Anchor를 받아서 Detection 수행.

  여기서 Transfer connection block이라는 개념을 도입해서 ARM에서 작업을 수행하고 난 뒤의 출력 Feature map으로 ODM이 작업을 수행할 수 있도록 함. 

  One-stage의 효율성은 유지하면서 정확도를 Two-stage와 유사하거나 높게 끌어올리는 것이 목적.



# BottleNeck

- MobileNets: Efficient Convolutional Neural Networks for Mobile Vision - MobileNet이라고 하는 모바일과 임베디드 환경에서 비전 애플리케이션을 위한 아키텍처를 발표. Depthwise separable convolution으로 원래의 Convolution보다 가볍고 능률적인 연산 수행. 모델의 Latency와 Accuracy 간의 Trade off를 결정하는 두 개의 Global parameter인 Width multiplier와 Resolution multiplier가 특징.
- Xception: Deep Learning with Depthwise Separable Convolutions - Inception module을 개념적으로 보통의 Convolution과 Depthwise separable convolution의 사이쯤에 있다고 보고 해석. Inception 모듈을 Depthwise separable convolution으로 바꾸는 과정 제시.
- MobileNetV2: Inverted Residuals and Linear Bottlenecks - Object detection을 위해서 SSD에서 MobileNetV2를 적용한 프레임워크인 SSDLite를 발표. 모바일용 Semantic segmentation을 위한 프레임워크인 Mobile DeepLabv3를 발표. Depthwise separable convolution에서의 Pointwise convolution, Depwise convolution 개념과 ResNet에서의 BottleNeck 개념을 적용해서 Inverted residual block이라는 개념을 고안함. 이를 통해서 연산 중간에 쓰이는 많은 텐서 양을 줄여서 메모리 부하를 줄이는 효과를 봄. 



# EfficientDet

- EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks - 이 연구에서는 Model을 Scaling 하는 방법에 대해서 연구. 네트워크의 너비, 높이, 입력 Resolution을 균등하게 Scaling하는 방법인 Compound coefficient라는 개념 제시. 이를 바탕으로 저자들만의 아키텍처인 EfficientNet 발표.
- EfficientDet: Scalable and Efficient Object Detection - 다음의 두 개념을 적용하여 Object detection 프레임워크인 EfficientDet를 발표.
  1. Network neck에서 다양한 크기의 Feature를 합치는 방법으로 Bi-directional feature pyramid network(BiFPN) 제안.
  2. EfficientNet에서의 Compound scaling 개념을 적용. 



# NetworkNeck + AutoML

- Path Aggregation Network for Instance Segmentation - 저자들이 고안한 PANet을 통해서 Proposal 기반의 Instance segmentation 모델에서 정보가 잘 전파될 수 있도록 하는 것이 목적. 네트워크의 하위 계층의 정보가 상위 계층에 빠르게 전파될 수 있도록 Shortcut을 구축. Adaptive feature pooling이라는 개념을 도입하여 여러 레벨의 Feature들이 Segmentation을 위한 서브 네트워크에 잘 전파될 수 있도록 함. 
- MnasNet: Platform-Aware Neural Architecture Search for Mobile - 모바일을 위한 아키텍처를 디자인할 때 속도와 정확도 사이의 Tradeoff를 자동으로 탐색하는 방법 제시(Mobile Neural Architecture Search, MNAS). 이전 연구에서는 FLOPs 등의 간접 요소로 네트워크의 Latency를 간접적으로 측정했으나 여기에서는 실제 모델을 디바이스에 탑재하고 측정한 Inference Latency를 측정. 모델의 유연성과 탐색 공간 사이의 균형점을 찾기 위해서 Factorized hierarchical search space 개념 제시(네트워크를 블럭으로 나누고 블럭에서의 구성 요소를 자동으로 탐색). 
- NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection - NAS를 도입해서 Object detection을 위한 아키텍처를 만드는 것을 목표로 함. 이때 Feature 크기에 상관 없이 경로를 구축하여 필요하다면 Fusion을 할 수 있도록 함.
- MnasFPN : Learning Latency-aware Pyramid Architecture for Object Detection on Mobile Devices - 저자들은 MnasFPN이라고 하는 Detection head를 자동으로 찾기 위한 모바일 친화적인 Search space 설계 방법을 제시. MnasNet과 같이 모바일에서 측정한 Inference Latency를 고려. 



# Anchor free

- CornerNet: Detecting Objects as Paired Keypoints - 바운딩 박스를 Top-left, Bottom-right의 좌표를 예측하여 만드는 방식. 
- Objects as Points - 객체를 단일 점으로 본다. 이때 점은 바운딩 박스의 중심 좌표가 된다. Keypoint estimation으로 객체의 중심 좌표를 찾고 각 Task에 알맞게 작업을 수행한다(박스의 크기, 3차원에서의 위치 방향, 객체의 포즈 등). 
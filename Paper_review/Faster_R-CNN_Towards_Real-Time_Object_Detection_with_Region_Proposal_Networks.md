# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

Shaoqing Ren(University of Science and Technology of China, Hefei, China),

Kaiming He, Jian Sun(Visual Computing Group, Microsoft Research)

Ross Girshick( Facebook AI Research)



## Abstract

Region proposal 방식의 Object Detection 알고리즘들은 그 동안 많은 기술적 발전을 이뤄왔다. 그 방향성은 네트워크의 각 컴포넌트들을 하나로 통일해서 End-to-End로 학습하는 방향이었다. 그럼에도 불구하고 Region proposal을 생성하는 방식은 통일하지 못해서 학습 단계에서 Bottleneck으로 남아있었다. 이 연구에서는 이 부분까지도 하나로 통일시켜서 이미지 하나의 Feature map으로 Region proposal도 생성하고 이 단계를 바탕으로 Classification과 Localization를 한 번에 진행하는 방법을 제안하고자 했다. 여기서 Region proposal을 하는 컴포넌트를 Region Proposal Network라고 한다. 구조를 보면 전체적인 네트워크 안에 작은 네트워크가 들어 있으므로 Network In Network의 논문을 떠올리게 한다. 최종적으로는 RPN과 선행 연구인 Fast R-CNN을 합친 방식을 고안하게 된다. 여기서 Attention Mechanism이라는 말이 나오는데 다음의 블로그를 참고해서 보자면, 아마 Feature map을 바탕으로 RPN에서 Region proposal을 생성해내고 이를 바탕으로 다시 Feature map에서 어느 위치를 볼껀지 'Attention'한다는 의미 인거 같다. 

[WikiDocs - 어텐션 메커니즘 (Attention Mechanism)](https://wikidocs.net/22893)



## Introduction

이 연구가 수행될때까지 Region Proposal 방식의 Object Detection 알고리즘에서 Bottleneck으로 남아 있는 가장 큰 문제는 역시 Region Proposal을 생성하는 부분이다. Selective search에서는 CPU로 수행하도록 구현되었으며 이미지 하나를 처리하는데 2초 이상이 걸린다. 그것보다는 더 획기적으로 시간을 줄였다고 하는 EdgeBoxes도 이미지 하나를 처리하는데 0.2초가 걸린다고 한다. 그럼에도 불구하고 Region Proposal 방식은 전체 수행 시간에서 꽤 많은 시간을 잡아 먹는다. 게다가 Fast R-CNN은 GPU를 활용하도록 구현되어 있고 Region Proposal 알고리즘들은 CPU로 수행되도록 구현되어 있기 때문에 잘 맞지 않는다. 그래서 저자들은 Fast R-CNN과 Convolution 연산을 공유하면서 Region Proposal을 생성할 수 있는 RPN을 고안해냈다.  RPN 또한 순수하게 Convolution Layer들로 이루어져 있다. 여기서는 그리드의 각 위치에서의 Objectness Score를 계산하고 Region bound의 Regression을 수행한다. 

![](./Figure/Faster_R-CNN_Towards_Real-Time_Object_Detection_with_Region_Proposal_Networks1.JPG)

RPN은 Region proposal들을 다양한 스케일과 종횡비로 보고 예측할수 있도록 고안되어 있다. 기존의 방법들이 위 그림의 a, b처럼 Pyramids of Images나 Pyramids of Filters의 방법을 적용했다면 여기서는 Anchor box들을 사용해서 이를 수행한다. 여기서 Anchor box들은 다양한 스케일과 종횡비의 Reference 역할을 한다. 이 방법은 전자의 방법들과 비교해서 이미지나 필터들을 여러가지 (크기나 종횡비의) 옵션으로 나열할 필요가 없기 때문에 속도면에서 더 효과적이라고 한다. 



## RELATED WORK

### Object Proposals

Object Proposals 생성하는 알고리즘에 대한 많은 문헌이 있는데 대표적으로 다음과 같은 것들이 있다.

- Grouping super-pixels - Selective Search, CPMC, MCG
- Sliding windows - Objectness in windows, EdgeBoxes

이 알고리즘들은 네트워크에 하나로 통합되기 보다는 외부 모듈로서 넣는다. 



### Deep Networks for Object Detection

R-CNN은 End-to-End로 학습이 이뤄지기는 하나, (바운딩 박스 회귀를 제외하고는) Object bounds 예측에는 관여하지 않는다. 그래서 Localization에 대한 정확도는 Region Proposal을 생성하는 알고리즘에 달려 있다. 몇가지 관련 연구들이 DNN 네트워크를 Object bounds를 예측하는데 사용할 수 있도록 하는 방법을 제시했다. 

- OverFeat - P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus,and Y. LeCun, “Overfeat: Integrated recognition, localization and detection using convolutional networks,” in International Conference on Learning Representations (ICLR), 2014.
- MultiBox - (이 알고리즘을 통해 생성된 Region Proposal들이 R-CNN에서 사용된다.)
  - D. Erhan, C. Szegedy, A. Toshev, and D. Anguelov, “Scalable object detection using deep neural networks,” in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014. 
  - C. Szegedy, S. Reed, D. Erhan, and D. Anguelov, “Scalable, high-quality object detection,” arXiv:1412.1441 (v1), 2015.

위와 같이 네트워크를 통해서 Object bounds를 예측하려는 시도를 통해서 Convolution 연산을 공유하려는 시도가 증가 되었다. 왜냐하면 Visual recognition을 위한 Attention mechanism을 효율적이고도 정확하게 적용 가능해지기 때문이다. 



## FASTER R-CNN

![](./Figure/Faster_R-CNN_Towards_Real-Time_Object_Detection_with_Region_Proposal_Networks2.JPG)



여기서 시스템은 크게 두 가지 모듈로 구성된다.

- Deep Fully Convolutional Network - Attention mechanism을 적용하여 탐지기가 이미지의 어디를 봐야 하는지를 알려주는 Region Proposal 생성
- Fast R-CNN Detector - 위에서 생성된 Region Proposal을 보고 Classification + Localization을 수행.



### Region Proposal Networks

RPN은 임의의 크기의 이미지를 입력을 받아서 사각형의 Object Proposal과 각 사각형의 Objectness Score를 출력한다. 저자들은 이 Network를 Fully Convolutional Network로 구축했는데 Fast R-CNN Object Detection Network와 Computation을 공유하기 위함이라고 한다(공통의 CNN 계층에서 출력된 Feature map으로 RPN와 Fast R-CNN Detector Network에서 각각 Task를 수행한다.).

Region Proposal을 생성하기 위해서 RPN에서 먼저 입력으로 들어온 Feature map에 n x n의 Window로 Convolution 연산을 수행한다. 이에 대한 결과로 출력 크기는 같으면서 차원은 다른(ZF - 256d, VGG-512d with ReLU) Feature map을 출력하게 된다. 이 Feature map에  두 브랜치의 1 x 1 Convolutional Layer에서 연산을 수행하고 나서 Fully Connected Layer를 통해서 각각 Box regression과 Box classification과 관련된 산출물을 출력한다. 여기서는 n을 3으로 했지만 Receptive field가 클수록 좋다고 한다(Figure 3 left). 



#### Anchors

각 Sliding-window에서는 동시에 여러 Region Proposal을 생성하는데 각 위치에서 최대로 예측할 수 있는 Region Proposal의 개수는 k개이다.  그러므로 reg Layer에서는 각 Region Proposal의 스케일과 위치와 관련된 4 x k개의 값을 출력할 수 있고 cls Layer에서는 객체가 있는지 없는지 여부와 관련된 2 x k개의 값을 출력할 수 있다. 이 k개의 Proposal들은 k개의 Reference box들과 관련이 있는데 k개의 Reference box들을 Anchor라고 부른다. 즉 이 Anchor들의 Scale과 Coordinate을 조정하여 Region Proposal을 생성하는 것이다. 여기서는 3개의 Scale과 3개의 Aspect Ratio를 고려했으므로 총 9개의 Default Anchor box가 존재한다. 입력 Feature map이 W X H(보통 ~2,400)이라고 하면 W x H x k의 Anchor들을 조정하게 된다. 



#### Translation-Invariant Anchors

저자들이 주장하길 이미지 내에 객체의 위치가 상대적으로 변경되었더라도(Convolution 연산이나 Max pooling 연산을 통해서 객체가 움직인 것 처럼 보일 수 있음.) Detection은 이와 상관 없이 관련 연산을 수행할 수 있다고 한다(MultiBox는 그렇지 못함).

또, 이런 특성 때문에 MultiBox보다 상대적으로 모델의 파라미터 수가 훨씬 적고 이는 PASCAL VOC같은 수가 적은 데이터셋에서 과적합을 일으킬 위험을 줄인다.



#### Multi-Scale Anchors as Regression References

Multi-Scale을 모델 차원에서 다루는 여러가지 방법이 있다(Figure 1). 하나는 Image/Feature pyramid 방식이고(Figure 1-a) 다른 하나는 Filter pyramid 방식이다. 저자들이 이 연구에서 적용한 방식은 여러 크기와 종횡비의 Anchor들을 이용한 방식인데 앞선 두 방식과 비교했을 때, 단일 크기의 Feature map에서 단일 크기의 Filter로 다양한 스케일의 객체를 탐지하는 것을 학습할 수 있으므로 훨씬 비용이 적다. Anchor를 적용하는 방식 덕분에 Region Proposal을 생성하는 부분까지도 네트워크에 하나의 요소로서 통일 시킬 수 있었다.



#### Loss Function

RPN을 훈련시키기 위해서 각 Anchor에는 Positive(객체가 들어 있음)과 Negative(들어 있지 않음)이라는 레이블을 붙인다. 레이블이 붙지 않은 Anchor들은 무시한다. 

- Positive - GT와 가장 높은 IOU를 보인 Anchor, GT와 IOU가 0.7 이상인 Anchor. 첫 번째 조건을 굳이 추가한 이유는 두 번째 조건을 만족하는 샘플이 없는 아주 드문 경우 때문이다.
- Negative - Positive 하지 않은 Anchor 중에서 GT와의 IOU가 0.3보다 작은 Anchor.

Faster R-CNN의 Loss function은 Fast R-CNN과 거의 유사하다. 

![](./Figure/Faster_R-CNN_Towards_Real-Time_Object_Detection_with_Region_Proposal_Networks3.JPG)

여기서 i는 각 미니배치에서 Anchor의 인덱스를 의미한다. pi는 Anchor_i에 객체가 있을법한 확률을 예측한 것이고 p\*\_i는 Anchor가 Positive면 1, 아니면 0이다. ti는 예측된 Bounding box의 4가지 좌표 정보이고 t\*\_i는 해당 Positive box와 관련 있는 GT의 좌표 정보이다. Regression loss 같은 경우에는 예측 박스와 해당 GT의 Smooth L1 Loss를 사용하는데 이때 Positive 박스에 관련된 Smooth L1 Loss만 전체 Loss에 반영한다. λ는 두 Loss 간의 밸런싱을 담당하는데 저자들이 실험을 통해서 10을 기본으로 설정해뒀다. N_cls, N_reg는 두 Loss를 각각 정규화 하기 위함이고(평균 값) N_cls는 미니배치 사이즈, N_reg는 Feature map  크기이다(Feature map의 크기는 W x H이고 주로 under 2400. Feature map의 각 Cell마다 Positive인 Bounding box들의 Location만 신경씀). 저자들은 말하길 정규화는 꼭 필요한 과정은 아니라고 한다. 

t와 관련된 방정식은 다음과 같다.

![](./Figure/Faster_R-CNN_Towards_Real-Time_Object_Detection_with_Region_Proposal_Networks4.JPG)

x, y, w, h는 각 box의 중앙 좌표, 넓이, 높이이고 x, xa, x\*는 각각 예측, 기본 Anchor, GT box를 의미한다. Regression 과정의 경우는 Positive box로 레이블링된 Anchor box들을 GT box로 점점 맞춰가는 과정이라고 볼 수 있다. 

다양한 크기의 ROI를 다루기 위해서 K개의 Bounding box regressor가 존재하는데 각 Regressor는 하나의 스케일과 하나의 종횡비를 담당한다. 각 Regressor들은 학습 가중치를 공유하지 않는다. 이런 디자인 덕분에 Feature들의 크기나 종횡비가 고정되어 있어도 Box들이 다양한 스케일과 종횡비를 처리할 수 있게 되었다. 



#### Training RPNs

RPN은 역전파와 SGD를 통해서 End-to-End로 학습이 가능했다. 하나의 이미지에서 여러 미니배치가 만들어지고(Receptive field 슬라이딩) 각 미니배치에는 Positive나 Negative한 Anchor들이 있다. 이 Anchor들을 모두 고려해서 Loss 줄여 최적화 할 수도 있지만 Negative가 압도적으로 많기 때문에 Negative에 의해서 편향될 가능성이 크다. 따라서 저자들은 256의 Anchor들을 샘플링할때 Negative와 Positive 거의 1:1로 샘플링한다(Positive가 적으면 나머지는 Negative로 채운다).

Localization 계층들은 평균 0, 표준편차 0.01의 가우시안 분포를 따르는 랜덤 값으로 초기화 하고 Classification 계층들은 ImageNet classification Task에서 미리 학습한 가중치로 초기화 한다. ZF net은 모든 계층을 튜닝하고 VGG는 conv3_1부터 위까지 튜닝한다. PASCAL VOC 데이터셋에서 60k 미니 배치는 LR 0.001, 나머지 20k 미니배치는 0.0001로 학습시킨다. 모멘텀은 0.9, Weight decay는 0.0005를 적용한다. 



### Sharing Features for RPN and Fast R-CNN

RPN과 Fast R-CNN은 나름의 목적대로 Convolution 계층의 가중치를 갱신할 것이다. 그런데 각각 훈련을 진행하기보다는 하나의 통합된 네트워크로서 훈련시키기 위해 저자들은 다음과 같은 방법을 생각했다. 

- Alternating Training - 먼저 RPN을 훈련시키고 여기서 생성된 Proposal을 Fast R-CNN을 훈련시키는데 사용한다. Fast R-CNN에 의해서 Tuning된 네트워크는 다음 Iteration에서 RPN을 초기화하는데 쓰이고 이런 과정이 반복된다. 이 연구에서 모든 실험은 이런 방식으로 진행되었다.
- Approximate Joint Training - 훈련 과정부터 RPN과 Fast R-CNN을 합치는 방법. 순전파 시에 RPN에서 생성된 Proposal로 Fast R-CNN에서 Detection을 진행하고 역전파 시에 공유 계층에 대해서는 RPN에서의 Loss와 Fast R-CNN에서의 Loss가 합쳐진다. 그런데 첫번째 결과와 완전히 동일한 결과를 내는 것은 아니고 Proposal Boxes' Coordinates와 관련된 미분값은 무시하게 된다. 저자들이 확인하길 그래도 거의 첫번째 결과와 거의 동일한 결과를 내면서 25\-50% 더 훈련 시간을 줄일 수 있다고 한다.
- Non\-approximate Joint Training - 이상적으로는 역전파 시에 Box coordinates에 대한 경사 하강도 진행되어야 한다. 그렇기 위해서는 RoI Pooling 계층에서도 Box coordinates에 대해서 미분 가능해져야 한다. 이런 문제는 RoI Warping 계층 이라는 것을 통해 해결이 가능하다.



#### 4-Step Alternating Training

저자들은 RPN과 Fast R-CNN의 공유 가중치를 학습시키는 과정을 Alternating Training 방법을 적용해서 사단계로 나타냈다. 

1. RPN을 먼저 학습시킨다. ImageNet 데이터로 미리 학습이 완료된 가중치로 초기화된 모델이 있고  End-to-End로 Region Proposal Task에 맞게 Fine-tuning 시킨다.
2. 첫번째 단계에서 학습한 RPN에서 생성한 Region proposal로 Fast R-CNN Detection network를 훈련시킨다. 이 Fast R-CNN 또한 ImageNet 데이터셋에서 미리 학습되었다. 여기까지는 두 네트워크가 Convolution 계층을 공유하지 않는다.
3. Detector 네트워크를 RPN 훈련 과정에서 초기화하는데 이용한다. 그런데 공유 Convolution 계층은 고정을 시키고 RPN에 들어가는 계층들만 Fine-tuning 시킨다. 여기서부터 두 네트워크가 Convolution 계층을 공유하기 시작한다.
4. 공유 Convolution 계층을 고정시키고 Fast R-CNN에 들어가는 계층들만 Fine-tuning시킨다. 



### Implementation Details

저자들은 훈련할때, 테스트할때의 Region Proposal과 Object Detection에서 모두 이미지의 단일 스케일에서 실험을 진행했다. 짧은쪽의 픽셀이 거의 600 픽셀이 되게 했다. 저자들이 생각하기에 다양한 스케일의 이미지를 사용하면 정확도는 좀 더 올라가겠지만 좋은 트레이드 오프(속도)를 보여주지는 못했다. ZF, VGG net에서 마지막 Convolution 계층에서의 총 Stride가 16이므로 PASCAL 이미지 데이터 셋에서(~500x375)에서 이미지의 크기를 재조정하지 않으면 마지막 Convolution 계층에서는 ~10 픽셀 정도 되었다.

Anchor box의 경우 128^2, 256^2, 512^2의 스케일과 1:1, 1:2, 2:1의 종횡비를 조사했으며 하이퍼 파라미터는 최적화 하지는 않았다. 여기서 주목할 만한 점은 Box에 대한 예측 값이 Receptive field의 크기보다 커질 수 있다는 것이다(예를 들어서 이미지 안의 보이는 객체가 하나이고 정 중앙에 있을 때).

![](./Figure/Faster_R-CNN_Towards_Real-Time_Object_Detection_with_Region_Proposal_Networks5.JPG)

이미지 테두리에 걸치는 Anchor box의 경우 훈련 시에는 무시하기 때문에 Loss에 관여하지 않는다. 1000 x  600의 이미지의 경우에 보통 20000 여개 정도의 Anchor Box가 생성되는데 이 중에서 테두리에 걸치는 Anchor box를 제거하면 이미지 당 6000개 Anchor box 정도만 사용하게 된다. 만약에 훈련 간에 이 Box들을 무시하지 않는다면 이 Box들로 인해서 Loss에서 (올바른 방향으로) 갱신하기 어려운 Box들이 생기기 때문에 훈련 성능이 수렴하지 않게 된다. 테스트 시에는 RPN에서 전체 이미지에 대해 Convolutional 연산만 수행하기 때문에 이미지 테두리에 맞는 Bounding box가 생성될 수 있다. 

RPN에서 몇개의 Proposal들은 서로 매우 겹치는 것들이 있다. 이 숫자를 줄이기 위해서 그 박스들의 cls Score에 근거하여 Non-maximum suppression (NMS)을 수행한다.  NMS의 IoU Threshold 값을 0.7로 고정시키고 이미지당 약 2000개의 Proposal만 남겨둔다. 저자들이 확인한 바로는 NMS로 탐지 성능이 떨어지지 않고 오히려 Proposal의 숫자를 크게 줄인다고 한다.



## EXPERIMENTS

### Experiments on PASCAL VOC

저자들은 PASCAL VOC 2007 Detection Benchmark 데이터셋에서 저자들의 연구 방법을 실험했다. 

![](./Figure/Faster_R-CNN_Towards_Real-Time_Object_Detection_with_Region_Proposal_Networks6.JPG)

Fast R-CNN에서 RPN을 적용하면 SS나 EB보다 시스템 속도는 빠르면서 mAP도 더 높은 것을 확인할 수 있다. 



### Ablation Experiments on RPN

RPN과 Fast R-CNN 사이에 이미지 특징을 추출하는 공유 Convolution Network의 효과를 입증하기 위해서 위에서 언급한 4-Step Training Process에서 두 번째 단계까지 진행하고 각 네트워크를 따로 사용했을 때 mAP가 살짝 줄어든 것을 확인할 수 있었다. 이는 세 번째 단계에서 Detector로 RPN을 Fine tuning할 시에 Proposal의 질이 더 개선되는 것으로 저자들은 결론내렸다. 

다음으로 훈련간에 Fast R-CNN 탐지 네트워크를 훈련시키는데에 대한 RPN의 영향력을 제거하기 위해서 의도적으로 ZF net과 2000개의 SS Proposal로 Fast R-CNN을 훈련시켰다. 그런 다음 테스트 시에 Detector를 고정시키고, 생성되는 Proposal Region들을 바꾸면서 Detection mAP를 측정했다. 여기서 RPN은 Detector와 특징을 공유하지 않는다. 

SS를 300개의 RPN Proposal로 바꿨을 때 mAP는 감소되었는데 이는 훈련과 테스트 사이의 Proposal이 일맥상통하지 않는 이유라고 저자들은 추측했다. 

그럼에도 불구하고 Top-Ranked 100개의 RPN Proposal을 사용했을 때 어느정도 준수한 성능이 나오는데 이는 Top-Ranked RPN Proposal이 정확하다는 것을 가리킨다. 또, 극단적으로 Top-Ranked 6000 RPN Proposal을 사용했을때(NMS 적용하지 않음) 전자와 비슷한 성능이 나오는 것으로 보아 NMS는 Detection mAP에 해를 끼치지 않는 것으로 볼 수 있다.

RPN에서 cls 브랜치 부분을 제거하여 NMS 혹은 Ranking을 사용하지 않았을 때, 랜덤하게 1000개의 Proposal을 샘플링했을 때는 mAP가 거의 변하지 않았는데 100개를 뽑았을때는 심하게 감소했다. 이를 통해서 cls 브랜치가 Highest Ranked Proposal을 정확하게 뽑는데 일조한다는 것을 알 수 있었다.

RPN에서 reg 브랜치 부분을 제거하여 Anchor Box가 곧 Proposal이 될 때 역시 mAP가 감소했다. 이를 통해서 High quality Proposal은 Regressed된 Bounding Box에서 나온다는 것을 알수 있다. 즉, Multiple Scale과 Aspect Ratio로는 불충분하다는 말이 된다. 

RPN의 네트워크를 더 좋은(용량이 큰) 네트워크를 사용할 경우 성능이 더 좋아졌다. 



### Performance of VGG-16

![](./Figure/Faster_R-CNN_Towards_Real-Time_Object_Detection_with_Region_Proposal_Networks7.JPG)



![](./Figure/Faster_R-CNN_Towards_Real-Time_Object_Detection_with_Region_Proposal_Networks8.JPG)



![](./Figure/Faster_R-CNN_Towards_Real-Time_Object_Detection_with_Region_Proposal_Networks9.JPG)



![](./Figure/Faster_R-CNN_Towards_Real-Time_Object_Detection_with_Region_Proposal_Networks10.JPG)



### Sensitivities to Hyper-parameters

![](./Figure/Faster_R-CNN_Towards_Real-Time_Object_Detection_with_Region_Proposal_Networks11.JPG)

위의 테이블은 Anchor Box에 대해서 다른 셋팅을 했을때 결과를 보여준다. 각 위치에서 하나의 Anchor만을 사용하면 3\-4% 정도로 mAP가 크게 떨어진다. 1 Scale + 3 Ratios나 3 Scale + 1 Ratio나 크게 차이가 나지 않으며 성능이 나쁘지 않다. 여기서 이 두 옵션은 탐지 정확도에 있어서 서로 얽매여 있는 옵션은 아니지만 저자들은 시스템의 유연성을 위해서 이 두 옵션 모두 적용한다고 했다.

![](./Figure/Faster_R-CNN_Towards_Real-Time_Object_Detection_with_Region_Proposal_Networks12.JPG)

위의 테이블은 Loss 계산식에서 Lambda의 값에 따른 mAP를 보여준다. Lambda가 10일때 mAP가 가장 높은 것으로 보아 두 Loss 사이의 밸런스를 가장 잘 유지하는 값임을 알 수 있다. 



### Analysis of Recall-to-IoU

저자들은 Proposal과 GT Box의 IoU 값에 따른 Recall 점수를 계산했다. 주의할 점은 이 Metric 같은 경우 탐지 정확도와는 크게 상관이 없고 오히려 Proposal Method를 진단하는 Metric이라는 것이다. 

![Faster_R-CNN_Towards_Real-Time_Object_Detection_with_Region_Proposal_Networks13](./Figure/Faster_R-CNN_Towards_Real-Time_Object_Detection_with_Region_Proposal_Networks13.JPG)

그래프를 보면 Proposal이 2000\->300으로 떨어질때 전체적인 그래프 값이 RPN보다 EB, SS보다 적게 떨어지는 것을 확인할 수 있다. 저자들은 이것이 RPN의 cls Term덕분이라고 했다. 



###  One-Stage Detection vs Two-Stage Proposal + Detection

![](./Figure/Faster_R-CNN_Towards_Real-Time_Object_Detection_with_Region_Proposal_Networks14.JPG)



### Experiments on MS COCO

![](./Figure/Faster_R-CNN_Towards_Real-Time_Object_Detection_with_Region_Proposal_Networks15.JPG)



### Faster R-CNN in ILSVRC & COCO 2015 Competitions

ResNet-101같이 아주 깊은 네트워크에서도 좋은 성능을 낼 수 있다고 한다.



### From MS COCO to PASCAL VOC

![](./Figure/Faster_R-CNN_Towards_Real-Time_Object_Detection_with_Region_Proposal_Networks16.JPG)





## Conclusion

RPN이라고 하는 Fully Convolution Layer로 이루어진 Region Proposal Generation Network 덕분에 Detection Network와 이미지의 특징을 추출하는 Convolution Layer들을 공유할 수 있게 되었고 그 덕분에 Region Proposal Generation Step에서 거의 비용을 줄 일 수 있게 되었다. 이런 특징은 Faster R-CNN을 하나의 통일된 딥러닝 기반의 네트워크로 만들었고 실시간에 가까운 Object Detection을 가능하게 했다. 

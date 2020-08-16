# Rich feature hierarchies for accurate object detection and semantic segmentation

Ross Girshick(UC Berkeley), Jeff Donahue(UC Berkeley), Trevor Darrell(UC Berkeley), Jitendra Malik(UC Berkeley)



## Abstract

저자들이 주장하길 PASCAL VOC 데이터셋을 측정되는 객체 탐지 알고리즘의 성능은 몇 년간 발전 없이 정체 상태였다고 한다. 당시에 좋은 성능을 내는 방법들은 여러 낮은 차원의 이미지 특징들을 고차원 수준의 문맥과 연결시키는 아주 복잡한 앙상블 시스템이었다고 한다.  그래서 저자들은 보다 간단하면서도 좀 더 규모면으로 확장 가능한 알고리즘을 제안했다. VOC2012 데이터셋에서 mAP 53.3%을 달성했는데 이는 저자들이 제안한 방법 이전의 최고 성능을 보인 방법보다 30%이상 개선된 결과라고 한다. 저자들이 말하는 이 연구의 두 가지 키 포인트는 다음과 같다.

- 대용량의 CNN을, 이미지 안의 객체의 위치 추정을 하거나 픽셀 단위의 세그멘테이션을 위한 상향식의 지역 후보 생성에 적용할 수 있다는 점.
- 레이블링된 데이터가 부족할 때, 다른 작업을 위해서 훈련된 CNN을 도메인 타겟에 맞춰 Fine-tuning하고 나서 객체 탐지나 세그멘테이션 작업을 위해서 사용하면 큰 성과를 보일 수 있다는 점.

저자들은 이 연구에서 지역 후보 생성과 CNN을 결합했기 때문에 이 방법론을 R-CNN이라고 불렀다(Regions with CNN features). 



## Introduction

이 연구가 제안되었을 당시에, 시각적인 인식과 관련된 연구들은 SIFT나 HOG에 의존적이었다. 그런데 PASCAL VOC 객체 탐지 과제를 놓고 보면 2010-2012년에 눈에 띄는 성과가 없었다. 이때의 방식은 복작합 앙상블 시스템을 만들거나 이미 성공한 방법들에 약간의 변형을 가한 방법들뿐이었다. 

- SIFT - D. Lowe. Distinctive image features from scale-invariant keypoints. IJCV, 2004
- HOG - N. Dalal and B. Triggs. Histograms of oriented gradients for human detection. In CVPR, 2005
- Neocognitron - K. Fukushima. Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position. Biological cybernetics, 36(4):193–202, 1980

그러다가 다음의 연구에 기반하여

- D. E. Rumelhart, G. E. Hinton, and R. J. Williams. Learning internal representations by error propagation. Parallel Distributed Processing, 1:318–362, 1986

LeCun 등은 역전파를 통해서 SGD를 수행하는 것이 CNN을 훈련 시키는 것에 효율적이라는 것을 보여줬다. 그래서 1990년대에 많이 이용되다가 SVM이 출현하면서 이용 빈도가 쇠퇴되었다. 2012년에 Krizhevsky 등이 ILSVRC에서 매우 높은 정확도를 보이는 CNN 모델을 선보이면서 역전파를 통해 CNN을 훈련시키는 방법이 재주목을 받게 되었다(CNN에 추가적으로 ReLU, Dropout 등을 적용). 저자들은 이미지 분류 문제를 해결한 이 방법론을 어떻게 객체 탐지 문제에 적용할 것인가하는 질문에 주목했다. 특히 HOG 스러운 방법들보다 CNN을 적용한 방법들이 PASCAL VOC 객체 탐지 과제에서 더 높은 성능을 보일 수 있음을 목표로 했다. 그렇기 때문에 다음의 두 가지 문제에 집중했다고 한다.

- 딥 러닝 네트워크를 통한 이미지 안의 객체들의 위치 추정
- 적은양의 Annotated된 탐지용 데이터를 대용량의 모델에서 훈련시키는 방법

이미지 분류 문제와는 다르게 객체 탐지는 이미지 안의 객체들의 위치 추정도 해야 한다. 어떤 연구에서는 위치 추정을 회귀 문제로 보기도 했는데 Szegedy 등에 의하면 이런 접근법이 실제로는 잘 맞지 않을 수도 있다고 했다. 다른 접근 방식으로는 슬라이딩 윈도우 방식의 탐지기를 만드는 것인데 적어도 20년동안 CNN은 이런식으로 객체 탐지 분야에 적용되었다(얼굴이나 보행자 탐지에 한해서). 저자들도 원래 슬라이딩 윈도우 방식의 접근법을 고려했었는데 네트워크의 용량이 커지고 입력 이미지의 크기가 커지면서 정확한 위치 추정을 하는 것은 기술적으로 문제가 돼었다고 한다. 대신에 저자들은 CNN의 위치 추정 문제를 이미지 상의 지역을 인식하는 문제로 해결해서 객체 탐지나 시멘틱 세트멘테이션 과제에서 좋은 성과를 거두었다. 

![](./Figure/Rich_feature_hierarchies_for_accurate_object_detection_and_semantic_segmentation1.JPG)

저자들은 ILSVRC2013의 클래스 200개까지 탐지 데이터셋으로 다음의 연구와 성능을 비교하여 제시했다.

- OverFeat - P. Sermanet, D. Eigen, X. Zhang, M. Mathieu, R. Fergus, and Y. LeCun. OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks. In ICLR, 2014

저자들이 집중했던 두 번째 문제는, 객체 탐지 분야에서 대용량의 CNN을 학습시킬만한 훈련 데이터가 매우 부족하다는 것이었다. 당시에 이를 해결하기 위한 전통적인 방법으로는 비지도 학습으로 일단 미리 훈련을 시키고 지도 학습 기반의 Fine-tuning을 하는 것이었다. 이 연구에서 저자들은 다른 목적으로 학습시킨(지도 학습 기반, 예를 들어서 ILSVRC 이미지 분류 데이터셋) CNN에 특정 도메인 목적에 맞는 적은 양의 데이터(PASCAL 이미지 탐지 데이터셋)로 Fine-tuning 하는 방법이, 대용량의 CNN을 상대적으로 적은 양의 데이터로 학습시키는 방법으로서 매우 효율적이라는 것을 보여줬다. 

저자들이 말하길, 이 시스템에서 클래스에 특정한 계산은 적은 양의 행렬, 벡터 곱 연산과 Greedy NMS(Non-maxtimum suppression) 뿐이라고 한다. 

저자들은 또, 시스템을 개선하기 위해서 실패한 시도들을 분석하는 것이 중요하다고 생각하여 탐지 성능을 Hoiem 등이 만들어낸 도구로 분석했다고 한다. 이 분석 방법으로 분석한 결과, 단순한 바운딩 박스 회귀 방법이, 저자들의 실패에서 주요 원인으로 여겨지던 잘못된 위치 추정을 상당하게 줄인다는 것을 확인했다. 

R-CNN이 이미지 상의 지역에 기반하여 작동하기 때문에 시멘틱 세그멘테이션 과제에도 적용될 수 있다고 하며 약간 시스템을 변형하여 PASCAL VOC 세그멘테이션 과제에서 좋은 성적을 냈다고 한다. 



## Object detection with R-CNN

이 연구에서의 객체 탐지 시스템은 크게 세 부분으로 이루어져 있다. 

- 카테고리에 상관 없이 지역 후보를 생산하는 부분. 이 후보들은 탐지기가  목적으로 하는 탐지 후보들의 집합이다.
- 각 지역마다 고정된 길이의 특징 벡터를 추출하기 위한 CNN
- 특정 클래스를 담당하는 선형 SVM들의 집합



### Module design

#### Region proposals

많은 연구에서 카테고리와 상관없는 지역 후보들을 생성하는 방법을 제안한다.



#### Feature extraction

저자들은 Krizhevsky 등에 의해 구현된 CNN의 Caffe 구현체를 사용해서 각 지역 후보에 대해 4096차원짜리 특징 벡터를 추출했다. 각 입력 이미지는 5층짜리 컨볼루션 계층과 2층짜리 완전 연결 계층으로 구성된 CNN을 227x227의 크기로 RGB 채널 별로 평균을 빼는 전처리 과정을 거치고 나서 순전파된다.  

각 지역 후보 별로 특징 벡터들을 뽑아내기 위해서 가장 먼저 할 일은 입력 이미지들을 CNN의 입력에 맞게 변환하는 것이다(CNN 아키텍처의 입력 크기는 227x227로 고정되어 있다). 임의의 모양의 지역들을 변환하기 위한 많은 변환 방법 중에서 저자들은 가장 간단한 방법을 골랐다고 한다. 지역들의 크기나 종횡비에 상관없이 후보들의 모든 픽셀은 원하는 크기에 딱 맞는 바운딩 박스로 워프(Warp)된다. 워프 하기 전에 이 딱 맞는 바운딩 박스를 팽창시켜서 워프된 입력 이미지의 문맥 정보가 담긴 p개의 픽셀이(저자들은 p=16을 선택) 더 존재하도록 했다. 

 ![](./Figure/Rich_feature_hierarchies_for_accurate_object_detection_and_semantic_segmentation2.JPG)



위 그림은 워프된 훈련 지역들의 예시를 보여준다. 



#### Object proposal transformations

![](./Figure/Rich_feature_hierarchies_for_accurate_object_detection_and_semantic_segmentation3.JPG)

워프 방법에는 여러 가지 방법이 존재한다. 첫 번째 방법은 (B)열처럼 각 지역 후보들을 정사각 상자로 감싸고 CNN의 입력 크기에 맞게 종횡비를 유지하면서 크기를 맞추는 것이다. 또 하나의 방법은 B에서 객체의 주변을 잘라내는 방법으로 (C)열에 예시가 나와 있다. 다음으로 (D)열에는 종횡비를 유지하지 않으면서 워프 하는 방법의 예시가 나와 있다. 

각 방법에서 저자들은 원래의 지역 후보 부변에 추가적인 문맥 정보를 더하는 방법을 고려했다. 문맥 패딩 p의 양은 변환된 입력 프레임 주변의 가장 자리의 크기로 볼 수 있다. 위 그림의 예시들에서 위쪽 행은 p = 0이고 아래의 행은 p = 16일때이다. 만약에 원본 직사각형이 입력 이미지의 크기를 넘어선다면 나머지는 입력 이미지의 픽셀 평균 값으로 채워진다(CNN으로 들어가기전에는 이 값들이 빠지게 된다). p = 16일때가 p = 0 일때보다 성능이 좋았다고 한다. 



### Test-time detection

테스트 시에는 테스트 이미지에 대하여 Selective search를 수행해 약 2000개의 지역 후보들을 추출한다.  

[Arc Lab -논문 요약11 Selective Search for Object Recognition](https://arclab.tistory.com/166)

그리고 이 후보들을 워프 시킨 다음에 CNN에서 순전파 시켜서 고정된 길이의 특징 벡터를 추출해낸다. 그런 다음 각 벡터를 특정 클래스인지 아닌지 판단하도록 훈련시킨 SVM을 통해서 Scoring을 한다. 도출된 Scored 지역들에 대해서 Greedy NMS를 수행한다. 이때 학습된 한계값보다 높은 선택 영역과 IoU가 높아 겹치는 경우 이 지역들을 제거한다. 



#### Run-time analysis

이 연구에서 두 가지 특성이 탐지를 효율적으로 만든다. 

- 모든 CNN이 모든 카테고리에 공유되어 사용된다. 
- CNN을 통해 추출된 특징 벡터들이 다른 연구에 비하면 저차원이다. 

첫 번째 특성 덕분에 각 지역 후보들과 그 특징을 계산하는데 들어가는 시간이 각 클래스로 분할된다. 클래스 별로 들어가는 계산은 특징 벡터와 SVM 가중치 사이의 내적(Dot product)과 NMS에 들어가는 계산뿐이다. 실제로 한 이미지에 수행되는 모든 내적은 단일 매트릭스의 곱의 배치들로 나눠진다.  특징 행렬이 주로 2000x4096이고 SVM 가중치 행렬은 4096xN이다(여기서 n은 클래스의 숫자).

 이런 분석 결과는 R-CNN이 수천의 클래스로 확장될 수 있다는 것을 보여준다. 계산 방법이 간단하기 때문에 멀티 코어 CPU에서 이런 행렬 곱 계산이 용이하다. 또, UVA 같은 방법보다 저차원의 특징 벡터를 사용한다는 것도 이런 효율성을 만든다. 



### Training

#### Supervised pre-training

저자들은 CNN을 ILSVRC2012 이미지 분류 데이터셋으로 Pre-training했다고 한다. 왜냐하면 바운딩 박스 레이블이 이 데이터에는 없기 때문이다. Pre-training은 Caffe CNN 라이브러리를 이용해 수행되었고 Krizhevsky 등의 연구와 거의 유사한 성능을 보였다고 한다. 



#### Domain-specific fine-tuning

CNN을 탐지 목적에 맞게 조정하고 워프된 입력 이미지에 맞게 하기 위해서 CNN을 SGD으로 또 훈련시켰다. 이때 CNN을 ImageNet 1000개의 레이블 분류 문제에서 (N+1)의 분류 문제(여기서 1은 Background)로 바꾸고 무작위로 초기화 하는 것 빼고는 아키텍처 자체는 변경 시키지 않았다. VOC의 경우 N=20, ILSVRC2013의 경우 N=200이다. 지역 후보의 경우 Ground-truth 박스와 0.5 IoU인 지역 후보들을 그 클래스의 Positive, 나머지는 Negative로 설정한다. SGD는 0.001의 학습률로 시작해서(Pre-training할때보다 1/10) 미리 학습한 가중치의 정보를 최대한 손상시키지 않으면서 학습이 일어날 수 있도록 Fine-tuning 하는 것을 가능하게 한다. 각 Iteration마다 모든 클래스에 균등하게 32개의 Positive 윈도우를 샘플링하고 96개의 Background 윈도우를 샘플링해서 128의 배치 사이즈를 구성했다. 특별히 Positive 윈도우에는 좀 비중을 두었다. 왜냐하면 Background에 비해서 그 수가 극도로 적기 때문이다. 



#### Object category classifiers

만약에 차를 탐지하는 과제가 있다고 할 때, 차에 딱맞는 지역은 Positive, 차와 관련된 것이 아무것도 없는 지역은 Negative로 하는 것은 분명하다. 그런데 차가 일부 포함된 지역을 어떻게 처리할 것인가는 생각해봐야 할 문제이다. 저자들은 IoU 특정 한계값 이하의 지역들은 Negative로 설정하는 것으로 이 문제를 해결하고자 했다. 한계값은 {0, 0.1, ... 0.5}까지의 값들을 그리드 서치로 탐색해서 최적 값을 찾았는데 0.3이 최적 값으로 선정되었다. 이런 방법으로 Positive로 설정된 샘플들은 그 클래스의 Groud-truth 바운딩 박스가 된다. 

이런 과정을 거치고 나서 클래스별 SVM을 최적화 시키는데 훈련 셋이 메모리에 모두 로드 되기에는 너무 크기 때문에 저자들은 Standard hard negative mining 방법을 적용했다. Hard negative mining은 빠르게 수렴되며 모든 이미지에 한번 수행되면 mAP가 증가되는 것이 멈춘다. 

![](./Figure/Rich_feature_hierarchies_for_accurate_object_detection_and_semantic_segmentation4.JPG)



#### Positive vs Negative examples and softmax

이 연구에서는 CNN을 Fine-tuning할때 샘플들을 Positive와 Negative로 설정하는 방법과 객체 탐지용 SVM들을 훈련시킬때 설정하는 방법이 다르다. 미세 조정을 위해서는 각 지역 후보를 최대의  IoU를 가지고 있는 Ground-truth과 매칭해서 만약에 IoU가 0.5 이상이라면  그 Groud-truth 클래스의 Positive로 레이블링한다. 나머지 다른 후보들은 Background로 레이블링한다. SVM을 훈련시킬 때는 반대로 SVM을 훈련시킬 때는 Groud-truth 상자들만을 각 클래스에 대한 Positive example로 취하고 클래스의 모든 지역들과 IoU가 0.3 미만인 지역 후보들을 그 클래스에 대한 Negative로 설정한다. 0.3 IoU 이상이지만 Groud truth가 아닌 것들은 무시된다.

원래 저자들은 SVM을 ImageNet으로 Pre-trained한 CNN에서 추출한 특징들로 훈련 시키기 시작했고 그  당시에는 Fine-tuning은 고려 대상이 아니었다. 이때 저자들은 SVM을 훈련시키기위한 특정한 레이블 설정법이 저자들이 평가했던 옵션 셋 안에서(Fine-tuning을 할 때 적용했던 설정법 포함) 최적이라는 것을 발견하여 적용했다(즉 저자들이 이 연구를 할 당시에 SVM에 적용한 레이블 설정법이 Fine-tuning을 적용하기 전에는 SVM을 훈련시키위한 레이블 설정법 중 최적의 설정법이었다는 소리). Fine-tuning을 적용하고 나서 SVM을 훈련시키는데 적용한 레이블 설정법을 그대로 사용했지만 저자들이 CNN을 미세조정 할 때 적용한 레이블 설정법보다 성능이 나빴다. 

저자들이 추측하길 레이블 설정법의 이런 차이가  Fine-tuning할 때 데이터가 제한적이기 때문에 발생하는 것이었다. 저자들이 적용하는 방법은 많은 jittered 샘플(IoU 0.5이상 1이하지만 Ground truth는 아닌)을 만들어 내는데 이 샘플들이 Positive 샘플의 수를 거의 30배 만큼 확장시킨다. 이 샘플들은 전체 네트워크의 과적합을 피하는데 도움이 될 수 있으나 정확한 위치 추정을 위해 Fine-tuned 되는데는 최적은 아니었다. 

그리고 Fine-tuning 후에 SVM을 따로 훈련시키는 이유는 다음과 같다. 저자들은 Fine-tuned한 CNN의 마지막 계층에 21-way 소프트맥스 회귀 분류기를 객체 탐지기로서 사용해보는 옵션을 시도해봤다. VOC2007 데이터셋에서 시도했을때 mAP가 54.2%에서 50.9%로 떨어졌다. 성능 저하에는 여러 가지 이유가 있겠지만 저자들은 Fine-tuning 시에 적용했던 Positive example 설정법이 정확한 위치 추정을 하는데는 도움이 되지 않았고 소프트맥스 분류기가, SVM을 훈련시키는데 사용되었던 Hard negatives 의 부분 셋이 아니라 랜덤하게 선택된 Negative 샘플에서 훈련되었다는 사실을 꼽았다. 



### Results on PASCAL VOC 2010-12

저자들은 모든 선택 옵션(하이퍼 파라미터 설정 등)을 VOC 2007 데이터셋에 대해서 검증했다. VOC 2010-12 데이터셋에 대한 최종 결과에서는 CNN을 VOC 2012 훈련 셋으로 최적화 시키고 SVM 탐지기를 VOC2012 훈련+검증 셋으로 최적화시켰다. 검증 서버에는 바운딩 박스 회귀가 있는 버전과 없는 버전 두 가지 버전을 제출했다. 

 ![](./Figure/Rich_feature_hierarchies_for_accurate_object_detection_and_semantic_segmentation5.JPG)



### Results on ILSVRC 2013 detection

저자들은 R-CNN을 ILSVRC2013 200 클래스 탐지 데이터셋에서 수행했는데 이때 PASCAL VOC에 사용한 하이퍼파라미터와 같은 시스템으로 수행했다. 검증 서버에 보낼 때는 위에서와 마찬가지로 바운딩 박스 회귀가 있을때와 없을 때 두 가지 버전으로 제출했다. 

![](./Figure/Rich_feature_hierarchies_for_accurate_object_detection_and_semantic_segmentation6.JPG)



## Visualization, ablation, and modes of error

### Visualizing learned features

첫 번째 계층의 필터들을 시각화 하고 이해하는 것은 쉽다. 그러나 그 다음 계층들의 필터들을 이해하는 것은 도전적인 일이다. 저자들은 여기서 간단하면서 non-parametric하게 네트워크가 무엇을 학습하는 지를 보여주는 방법을 제시했다. 

아이디어는 네트워크의 특정 한 유닛(특징)을 골라서 그 자체가 하나의 객체 탐지기 인것처럼 사용하는 것이다. 대략 10 million 정도 되는 검증 지역 후보의 셋에 대한 그 유닛의 activation을 계산해서 지역 후보들 activation의 하향식으로 정렬하고 NMS를 수행하고 난 다음 가장 높은 지역들을 보여주는 것이다. 이 방법은 선택된 유닛이 어떤 입력에 대해 활성화 되는지 보여준다. 결과의 평균을 구하지 않는데 유닛의 각기 다른 시각 정보를 보고 불변 하는 특징에 대한 통찰을 얻기 위함이다. 

![](./Figure/Rich_feature_hierarchies_for_accurate_object_detection_and_semantic_segmentation7.JPG)

저자들은 마지막이자 5번째 컨볼루션 계층의 출력에 Max 풀링을 적용한 pool5의 유닛을 시각화 했다. pool5 계층의 특징 맵은 6x6x256=9216 차원이고 원래의 227x227 픽셀 입력에 대해서 195x195 픽셀 크기의 수용체 영역(입력단 영역) 크기를 가진다. 중앙에 있는 유닛은 영역의 거의 전체적인 부분을 보지만 모서리에 가까운 유닛은 좀 더 작으면서 clipped된(가장자리라서 영역이 잘린) 영역을 본다. 

위 그림의 각 행은, VOC 2007 훈련+검증 셋으로 Fine-tuning한 CNN의 pool5 유닛 중에 Top 16 activation을 보여준다. 이 유닛들은 네트워크가 샘플로부터 무엇을 배우는지 설명하기위해 선택되었다. 예를 들어서 두번째 행에서는 강아지 얼굴과 점들의 배열에 반응했다. 세번째 행은 붉은색 덩어리를 탐지하고 어떤 탐지기는 사람 얼굴을, 또 다른 탐지기는 글자, 창문이 달린 삼각형 형태의 구조물 같은 추상적인 패턴을 탐지하기도 한다.  네트워크는 약간의 클래스에 특화된 특징과 함께 모양, 질감, 색, 물체 특성 면에서 넓게 분포하고 있는 특징들을 결합한 것을 학습하는 것처럼 보인다. 다음 계층인 완전 연결 계층 fc5는 이런 풍부한 특징들의 복합체를 모델링할 수 있는 능력을 가진다. 



![](./Figure/Rich_feature_hierarchies_for_accurate_object_detection_and_semantic_segmentation8.JPG)

### Ablation studies

#### Performance layer-by-layer, without fine-tuning

저자들은 어떤 계층이 탐지 성능에 중요한지 알아보기 위해서 CNN의 마지막 3 계층에 대해서 VOC2007 데이터셋을 넣었을때 출력을 분석했다.완전 연결 계층 6과 7을 분석한 결과는 다음과 같다( pool5는 위에서 이미 설명). 

- fc6는 pool5에 완전하게(Dense) 연결되어 있다. 특징을 계산해내기 위해서 4096 x 9216의 가중치 행렬와 pool5의 특징맵(9216차원의 벡터로 Reshape됨)을 곱한다. 그리고 편향 벡터를 더한다.  벡터의 요소에는 ReLU가 적용된다.
- fc7은 네트워크의 마지막 계층이고 fc6의 출력과 4096 x 4096짜리 가중치 행렬을 곱하고 편향 벡터를 더해서 ReLU를 적용한다.  

먼저 Fine-tuning을 하지 않았을 때 결과를 분석했다. CNN은 PASCAL 데이터에 대한 activation이며 ILSVRC 2012에서 Pre-trained되었다. 위의 테이블2를 보면 fc7의 특징이 fc6의 특징보다 일반화 성능이 떨어지는 것을 확인할 수 있다. 이것은 CNN 파라미터의 29%에 해당하는 대략 16.8 million의 파라미터들을 mAP의 하락 걱정 없이 제거할 수 있다는 것을 의미한다. 더 놀라운 것은 fc7과 fc6을 제거하고 pool5까지만 분석했을때의 결과가 꽤 괜찮다는 것이다(pool5의 파라미터는 CNN 파라미터의 6%). 이것으로 CNN의 특징력(mAP에 영향을 주는 특징을 발견해내는 능력)은 아주 큰 완전 연결 계층들보다는 컨볼루션 계층들에서 도출된다는 것을 확인할 수 있다. 



#### Performance layer-by-layer, with fine-tuning

이번에는 Fine-tuning을 하고 난 뒤의 CNN activation의 분석 결과이다. Fine-tuning을 했을 때 mAP는 8%가량 상승했다. 이런 개선 정도는 pool5보다 fc6와 fc7에서 크게 나타났다. 이것은 ImageNet에 Pre-trained된 pool5의 특징들은 좀 더 일반적이며 Fine-tuning을 통해서는 분류기의 상위 요소들이 Domain-specific하고 non-linear한 특징들을 학습한다는 것이다. 
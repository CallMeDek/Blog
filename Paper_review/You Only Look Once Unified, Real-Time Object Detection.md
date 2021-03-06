# You Only Look Once: Unified, Real-Time Object Detection

Joseph Redmon(University of Washington), 

Santosh Divvala(University of Washington, Allen Institute for AI),

Ross Girshick(Facebook AI Research)

Ali Farhadi(University of Washington, Allen Institute for AI)



## 0. Abstract

YOLO는 기존의 연구들과는 다르게 분류기가 다른 목적에 맞는 일(Detection)을 수행하게 만드는 것 대신에 <u>객체 탐지 문제를 공간적으로 분리(이미지를 여러개의 격자로)시켜 바운딩 박스의 회귀(위치 조정)와 관련된 객체의 분류 문제로 접근</u>했다. 하나의 신경망이 단 한번의 평가에서 전체 이미지들로부터 직접적으로 바운딩 박스의 위치와 객체 분류 확률을 예측한다. <u>전체적인 파이프라인이 하나의 네트워크로 구성되어 있기 때문에 탐지 작업을 종단간으로 최적화</u> 할 수 있게 한다. 

특히 <u>속도에 강점이 있는데</u> YOLO 기반의 모델은 45 FPS(frames per second - 1초당 처리하는 이미지 수)의 성능을 보였다. Fast YOLO라고 하는 네트워크의 크기가 작은 모델은 155 FPS 의 <u>속도를 보이면서 다른 실시간 탐지 알고리즘보다 mAP(mean Average Precision)가 두 배 더 높다</u>. 또, 다른 시스템과 비교했을 때, 비록 <u>더 많은 Localization 에러를 만들어 내기는 하나, 배경에 대한 거짓 양성을 더 적게 예측</u>하는 것을 보인다. 그리고 <u>객체에 대한 좀 더 일반적인 특징들을 잘 잡아낸다</u>. 실사 이미지부터 미술품 같은 만들어진 이미지의 일반적인 특징을 DPM 이나 R-CNN 같은 다른 방법들보다 더 잘 잡아낸다. 



## 1. Introduction

빠르고 정확한 객체 탐지 알고리즘은 컴퓨터가 특별한 장치 없이 차를 주행하고, 관련된 장비들이 유저에게 실시간으로 정보를 전달 가능하게 하고, 대화 가능한 로봇 시스템 등 범용적인 목적에 쓰일 수 있다.

어떤 알고리즘은 객체를 탐지하기 위해서 분류기가 한 테스트 이미지를 여러 부분과 다양한 크기의 시야로 보게 한다. 특히 DPM 같은 시스템은 Sliding windows 접근 법으로 분류기가 전체 이미지를 균등하게 분리시켜 탐색하도록 한다. 

R-CNN은 이 작업을 위해서 지역 후보 기반 방법을 사용한다. 먼저, 이미지에서 객체가 있을법한 부분 후보를 만들어 내고 그 후보에 분류기를 돌린다. 분류 작업 후에는 바운딩 박스를 조정하고 중복 탐지(한 객체에 너무 많은 바운딩 박스가 예측된 것 - Occlusion )를 제거하고 이미지 안에 다른 객체들에 근거하여 박스의 점수를 다시 매긴다. 이런 과정들은 하나 하나가 개별적으로 훈련되므로 느리고 최적화 되기 어렵다(종단 간 최적화 불가). 

YOLO는 직접 이미지 픽셀 정보에서 바운딩 박스 좌표 조정과 클래스 확률을 뽑아 내는 것으로, 객체 탐지를 단일한 회귀 문제로 재정의 했다. <u>YOLO를 사용하면 이미지를 한번만 보고도 어떤 객체들이 어디에 있는지 예측할 수 있게 된다</u>. 

![](./Figure/You_Only_Look_Once_Unified_Real-Time_Object_Detection1.JPG)

YOLO는 간단하다. <u>하나의 컨볼루션 네트워크가 동시에 여러 바운딩 박스를 예측하고 그 박스에 대한 클래스 확률을 계산</u>한다. 훈련에는 전체 이미지를 사용하고 직접적으로 탐지 성능을 최적화하는데 이는 기존의 전통적인 객체 탐지 방법들에 비해 다음과 같은 장점이 있다. 

- YOLO는 복잡한 파이프라인이 필요 없으므로 <u>빠르다</u>. 기본 버전에서 Titan X GPU에서 <u>배치 처리(전체 이미지를 부분 부분 나눠서 진행)없이</u> 모델을 돌렸을 때 45 FPS의 결과가 나왔고 속도가 더 빠른 버전에서는 150 FPS 이상의 결과가 나왔다. 그러면서 다른 실시간 시스템보다 두 배 이상의 mAP를 달성했다.

  [Comparison to Other Detectors](https://pjreddie.com/darknet/yolo/)

- YOLO는 이미지에서 <u>예측을 진행할 때 전역적으로 추론</u>한다. 훈련과 테스트 진행 간에 전체 이미지를 보기 때문에 이미지 상에 나타나는 객체들의 <u>외형적인 특징</u>(공이 하나 있다)은 물론 <u>문맥적인 특징</u>(공 옆에 방망이와 야구 선수들이 있는 것으로 보아 야구공이다) <u>같이 암시적으로 인코딩</u> 한다. Fast R-CNN의 경우 거국적 문맥을 보지 못하기 때문에 이미지 상의 배경 부분을 잘못 판단하는 경우가 많은데 YOLO는 이와 비교하여 절반 이상으로 <u>배경에 관한 에러를 적게</u> 만든다. 
- YOLO는 <u>객체의 특징을 일반화시키는 방법을 학습</u>한다. 예를 들어서 실사 이미지로 학습하고 미술 작품을 테스트할 때 DPM, R-CNN을 크게 상회하는 성능으로 탐지를 수행한다. YOLO의 이런 특징 덕분에 새로운 도메인이나 예상하지 않은 입력에도 비교적 잘 적용할 수 있다. 

그럼에도 불구하고 YOLO는 정확도 측면에서 뒤쳐지는데, 특히 <u>빠르게 객체를 찾아내는 것은 가능하나 정확하게 어떤 객체(특히 크기가 작은)의 위치를 찾아내는 것에는 어려움을 겪는다</u>. 



## 2. Unified Detection

앞서 언급한 것처럼, YOLO에서는 객체 탐지에 필요한 각 과정을 하나의 단일한 신경망으로 구성했다. 이 네트워크에서는 각 바운딩 박스를 예측하기 위해서 전체 이미지에서의 특징을 사용한다. 바운딩 박스 예측 뿐만이 아니라 각 바운딩 박스안의 객체에 대한 클래스의 확률까지 동시에 예측한다. 이것은 곧 한 이미지 안의 전체 이미지(문맥적 정보)와 모든 객체들을 전역적으로 추론한다는 것을 의미한다. 그래서 YOLO는 훈련과 테스트 간에 높은 평균 정밀도를 유지하면서 실시가간으로 종단 간의 훈련을 가능하게 한다. 

YOLO에서는 입력 이미지를 <u>S x S의 격자</u>로 나눈다. 만약에 <u>객체의 중심점이 어떤 격자 요소안에 있다면 그 격자 요소가 객체를 탐지하는 데 책임</u>을 진다. 

각 격자 요소는 B개의 바운딩 박스와 그 박스에 대한 Confidence 점수를 예측한다. <u>Confidence 점수는 모델이 봤을 때 그 박스 안에 객체가 있음직할 확률이 얼마나 되는지, 예측한 박스가 실제 정답 박스와 비교해서 얼마나 정확한지</u>를 나타낸다. Confidence 점수는 다음과 같이 나타낸다. 

![](./Figure/You_Only_Look_Once_Unified_Real-Time_Object_Detection2.JPG)

객체가 격자 요소 안에 없다면 점수는 0이 되어야 하고, 만약에 있다고 하면 예측 박스와 실제 박스 사이의 IOU(Intersection over Union - 두 박스 사이의 겹치는 영역의 넓이와 두 박스를 합친 영역에 대한 넓이의 비율)과 같아야 한다.

<u>각 바운딩 박스는 5개의 예측 값</u>으로 구성되어 있다. (x, y)는 격자 요소의 테두리에서 부터의 박스의 중심점 좌표를 나타낸다. (w, h)는 전체 이미지에 대한 박스의 넓이와 높이의 비율을 나타낸다. Confidence 예측은 예측 상자와 실제 상자 사이의 IOU를 나타낸다. 

![](./Figure/You_Only_Look_Once_Unified_Real-Time_Object_Detection3.JPG)



각 격자 요소는 다음과 같은 C개의 클래스에 대한 조건부 확률을 예측한다. 

![](./Figure/You_Only_Look_Once_Unified_Real-Time_Object_Detection4.JPG)

<u>이 조건부 확률은 격자 요소 안에 객체가 있을 때, 특정 객체일 확률을 계산</u>한다. 한 격자 요소 안에 B개의 바운딩 박스가 있지만, 격자 요소당 한 개의 클래스 확률 집합을 예측한다. 

<u>테스트 수행 시에는 클래스에 대한 조건부 확률과 각 상자의 Confidence 예측 값을 곱한다</u>. 이것은 각 박스에 대해 특정 클래스에 대한 Confidence 점수로 볼 수 있다. 이 점수는 <u>곧, 어떤 클래스가 박스 안에 보일 확률과 박스가 얼마나 객체에 잘 맞춰져 있는가를 내포</u>한다.

![](./Figure/You_Only_Look_Once_Unified_Real-Time_Object_Detection5.JPG)

![](./Figure/You_Only_Look_Once_Unified_Real-Time_Object_Detection6.JPG)

PASCAL VOC 데이터셋에서 테스트 할 때는, 7 x 7 격자에, 각 격자마다 2개의 바운딩 박스를 예측했고 이 데이터 셋에는 20개의 레이블링된 클래스가 있었다. 그렇기 때문에 최종 예측은 7 x 7 x (2x5 + 20) 크기의 Tensor(다차원 배열)가 되었다. 



### 2.1 Network Design

네트워크의 초반부의 컨볼루션 계층에서는 이미지에서 특징을 추출하고 완전 연결 계층에서는 (클래스)확률과 (바운딩 박스)좌표를 출력한다. 

<u>네트워크 구조</u>는 이미지 분류를 위한 GoogLeNet에서 영감을 받았다. 네트워크는 <u>24개의 컨볼루션 계층과 2개의 완전 연결 계층이 연결</u>되어 있다. GooLeNet에서 사용된 Inception 모듈 대신에(가로 방향으로 여러 개의 브랜치를 쌓는것) Lin 등의 연구에서와 같이 <u>필터 크기 1x1의 축소 계층(Unit의 개수를 줄여 feature map의 크기는 유지하되 탐색 공간은 줄인다)과 필터 크기 3X3의 컨볼루션 계층을 연결</u>하여 사용한다. 전체 적인 구조는 다음과 같다.

![](./Figure/You_Only_Look_Once_Unified_Real-Time_Object_Detection7.JPG)

<u>빠른 버전의 YOLO는 속도에 초점을 맞추기 위해서 9개의 컨볼루션 계층을 사용하고 각 계층마다 더 적은 수의 필터 수를 사용</u>했다. 원래의 YOLO와 빠른 버전의 YOLO는 네트워크의 크기를 제외하고는 훈련과 테스트 간의 <u>모든 하이퍼 파라미터는 동일</u>하다. <u>네트워크의 최종 출력은 7 x 7 x 30 크기의 예측 값이 들어 있는 Tensor</u>이다. 



### 2.2 Training

<u>컨볼루션 계층은 ImageNet의 1000개의 클래스를 가진 데이터셋에서 Pre-train</u> 했다. 이를 위해서 위의 그림3에서의 처음 20개의 컨볼루션 계층과 Average Pooling 계층 그리고 하나의 완전 연결 계층을 연결했다. 이 네트워크를 훈련 시키는데 거의 한 주가 걸렸고 Caffe's Model Zoo 안의 GooLeNet 모델과 견줄 수 있을 만한 성능을, ImageNet 2012 검증 셋에서 single crop top-5 정확도 88%정도로 달성했다. 훈련과 추론 과정 간에는 Darknet 프레임워크를 사용했다. 

그런 다음 이 모델을 탐지 작업을 수행할 수 있도록 변경했다. Ren 등의 연구에 따르면 Pre-trained된 네트워크에 컨볼루션, 완전 연결 계층을 더하는 것이 성능을 개선시키는 데 도움이 되는데, 이들을 따라서 YOLO에서도 <u>4개의 컨볼루션 계층과 2개의 완전 연결 계층을, 랜덤한 숫자로 가중치를 초기화 시켜 Pre-trained된 모델에 연결</u>했다. <u>탐지 과정에서는 화질이 좋은 정보가 필요하기 때문에 입력 해상도를 224 x 224에서 448 x 448로 늘렸다</u>. 

최종 출력 계층에서는 클래스 확률 값과 바운딩 박스 좌표 값을 예측한다. <u>바운딩 박스의 넓이와 높이를 이미지의 넓이와 높이로 정규화하여 값이 0부터 1사이가 되게 하고 바운딩 박스의 중심 좌표 (x, y)를 격자 요소의 테두리로부터의 오프셋으로 만들어 역시 값이 0과 1 사이가 되게 한다</u>. 

<u>최종 출력 계층에 선형 활성화 함수를 사용하고 다른 모든 계층에는 다음의 Leaky rectified linear 활성화 함수를 사용</u>했다. 

![](./Figure/You_Only_Look_Once_Unified_Real-Time_Object_Detection8.JPG)

<u>모델의 최적화 과정에서는 Sum-squared error를 통해 출력의 에러를 평가하고 최적화를 진행</u>했다. Sum-squared error가 최적화를 진행하기 쉬워서 채택했으나 <u>Average precision을 극대화하는 목적에는 부합하지 않는다. 왜냐하면 Localization error(바운딩 박스 좌표 에러)와 Classification error(클래스 분류 에러)를 동일한 가중치로 두기 때문</u>이다. 대부분의 격자 요소에는 어떤 객체도 들어있지 않고 이때 Confidence 점수는 이 요소들의 점수를 0으로 만든다. 이는 객체를 포함하고 있는 요소들의 그래디언트를 억제한다(역전파 시에 객체를 포함하고 있지 않은 요소들이 너무 많아서, 객체를 포함하고 있는 요소들의 그래디언트가 영향을 끼치는 것에 부정적인 영향을 끼친다). 이는 모델은 불안정하게 만들고 학습이 조기에 종료되도록 만든다. 

<u>이를 해결하기 위해서 바운딩 박스의 예측에서 발생하는 loss는 가중치를 높이고 객체가 포함되지 않는 박스에서의 Confidence 예측에서 발생하는 loss는 가중치를 낮춘다</u>. 이를 위해서 λcoord  λnoobj를 각각 5, 0.5로 설정했다. 

또, <u>Sum-squared error는 큰 박스에서의 에러와 작은 박스에서의 에러를 동등하게 취급</u>하는데, 큰 박스에서의 작은 편차가 작은 박스에서의 편차보다 문제가 덜 됨을 반영시킬 필요가 있다. 이것을 부분적으로나마 다루기 위해서 <u>바운딩 박스의 넓이와 높이를 그대로 예측하기 보다는 제곱근을 예측</u>했다. 

YOLO에서는 격자 요소당 여러개의 바운딩 박스를 예측한다. 훈련 과정에서 각 객체에, 하나의 바운딩 박스 예측기만 예측을 만들어내게 하기 위해서, <u>각 예측과 실제 값 사이에 가장 높은 IOU를 보이는 예측기만이 그 객체에 대한 바운딩 박스를 예측</u>하도록 했다. 이에 대한 결과로 예측기들의 분업화가 생겼는데 <u>각 예측기는 특정 사이즈, 횡단비, 클래스에 특화된 예측</u>을 하게 되었고 이는 전체적으로 향상된 Recall을 이끌어 내었다. 

훈련 과정에서 다음의 여러 부분으로 구성된 손실 함수를 최적화 시켰다.

![](./Figure/You_Only_Look_Once_Unified_Real-Time_Object_Detection9.JPG)

λcoord는 좌표 값 예측에 대한 손실, λnoobj는  객체가 없는 분류에 대한 가중치, <u>1^obj_i는 요소 i에 객체가 나타났는지 여부, 1^obj_i j는 요소 i의 j번째 바운딩 박스 예측기가 바운딩 박스 예측을 한다는 것을 나타낸다</u>. 주목할 점은 각 격자 요소에 실제로 객체가 그 격자 안에 있을 때, 그 객체의 클래스에 대한 분류 에러에 패널티를 부과하고 바운딩 박스 예측기가 바운딩 박스 좌표 예측을 담당할 때만 실제 값에 대한 좌표 에러에 대해 패널티를 부과한다는 점이다. 

PASCAL VOC 2007과 2012의 데이터를 훈련 셋과 검증 셋으로 나누어 네트워크를 훈련시켰고 2012 데이터셋에서 테스트 할 때는 훈련 셋에 사용된 2007 데이터도 포함시켰다. 배치 사이즈는 64, 모멘텀은 0.8, decay 값은 0.0005로 설정했다. 

학습률은 다음과 같이 설정 했다. 처음 에폭에는 10^-3에서 10^-2로 천천히 증가시켰다. 처음에 높은 학습률로 시작하면 불안정한 그래디언트 때문에 학습이 조기 종료되기 때문이다. 그 다음 75 에폭동안 10^-2로 훈련을 계속하고 다음 30에폭 동안 10^-3으로, 마지막 30에폭에는 10^-4으로 진행했다. 

<u>과적합을 방지하기 위해서 드롭아웃과 많은 데이터 어그멘테이션을 사용</u>했다. 첫번째 완전 연결 계층 뒤의, 0.5 비율의 드롭아웃 계층은 계층 간의 협력을 방지한다.  데이터 어그멘테이션에서는 원본 데이터의 20% 갯수만큼 Random scaling과 Translation을 수행했다. 또, HSV (Hue saturation value의 약어. 색상(H), 채도(S), 명도(V)로 색을 지정하는 방법을 이용한 디자인 용도에 적합한 프로그램)에서는 1.5까지 임의로 이미지의 채도 등을 조정했다. 



### 2.3 Inference

테스트 이미지로 탐지에 대한 예측을 할 때도 한 번의 네트워크 평가만 있으면 되기 때문에 Classifier-based 방법보다 빠르다. PASCAL VOC 데이터셋에서는 이미지당 98개의 바운딩 박스와 그에 따른 클래스 확률들을 예측했다.

격자 형태의 설계는 바운딩 박스 예측에 공간적인 다양성을 강요한다. 보통은 한 객체가 어떤 격자 요소 안에 들어 있는게 확실하고 네트워크는 그 객체에 대해 하나의 바운딩 박스를 예측한다. 하지만 <u>어떤 크기가 큰 객체나 여러 요소 사이의 테두리 가까이에 있는 객체들은 여러 격자 요소에 의해 좀 더 잘 위치 파악이 잘 되기도 한다. Non-maximal suppression(IOU가 가장 높은 박스를 제외하고는 모두 없애는 것)이 이런 다중 탐지 문제를 해결하는 사용될 수 있다</u>. R-CNN이나 DPM에서처럼 성능에 그다지 중요하지 않지만 mAP를 2-3% 정도 더 올린다. 



### 2.4 Limitations of YOLO

YOLO는 <u>격자 요소 하나당 두 개의 바운딩 박스와 하나의 클래스만 예측할 수 있기 때문에</u> 심각한 공간적 제약이 존재한다. 이런 공간적 제약 조건이, <u>YOLO 모델에서 예측할 수 있는, 서로 붙어 있는 객체들의 수를 제한한다</u>. 특히 새뗴 같이 작은 객체들이 몰려 있는 경우에 모델이 예측을 하는데 어려움을 겪는다. 

YOLO 모델은 데이터로부터 바운딩 박스를 예측하는 법을 학습하기 때문에 <u>새롭거나 특이한 종횡비 혹은 배치 형태의 객체들을 일반화하는데 어려움을 겪는다</u>. 그리고 이 모델 구조가 입력 이미지로부터 <u>여러번의 다운 샘플링 계층 과정</u>을 거치기 때문에 <u>바운딩 박스 예측을 하는데 있어서 자세하지 않은(Coarse), 뭉쳐진 특징들을 사용</u>한다. 

마지막으로 탐지 성능을 계산하는 손실 함수를 통해 <u>훈련하는 과정에서 작은 박스들의 에러와 큰 박스들의 에러를 동등하게 취급</u>한다. <u>큰 박스에서의 작은 에러는 감당할만하지만 작은 박스에서의 작은 에러는 IOU에 훨씬 큰 영향을 준다</u>. YOLO 모델의 주된 에러 원인은 잘못된 위치 추정이다. 



## 3. Comparison to Other Detection Systems

객체 탐지는 컴퓨터 비전에서 핵심적인 문제 중 하나이며 파이프라인은 다음과 같다. 

1. 입력 이미지에서 좋은 특징들을 추출한다(Haar, SIFT, HOG, Convolutional features).
2. 분류기나 위치추정기가 특징 공간 안의 객체들을 확인하는데 사용된다.
3. 분류기나 위치 추정기는 이미지의 전체 영역이나 부분 영역들을 Sliding window로 순회한다. 



- Deformable pars models - DPM은 객체 탐지에 Sliding window를 사용하는 접근법을 시도한다. DPM은 특징을 추출하고 지역 영역을 분류하고 높은 점수대의 지역의 바운딩 박스를 예측하는 일련의 과정을 통합적으로 하지 않고 개별적으로 진행한다. <u>YOLO에서는 이런 따로 떨어져 있는 과정들을 하나의 컨볼루션 신경망 네트워크로 대체</u>한다. 이 네트워크가 특징을 추출하고 바운딩 박스 예측을 하고 Non-maximal suppression을 하고 문맥적 추론을 하는 <u>과정을 동시에 진행</u>한다. 

- R-CNN - R-CNN과 그 아종은 Sliding window 대신에 이미지 안에서 객체를 찾아내기 위해서 지역 후보를 찾아낸다. Selective search로 잠재적으로 바운딩 박스를 만들어 내고, 컨볼루션 네트워크에서 특징을 찾아내며 SVM으로 박스에 점수를 매긴다. 선형 모델로 바운딩 박스를 조정하고 Non-max suppression은 중복 탐지를 제거한다. 이 복잡한 파이프라인의 각 단계는 독립적으로 정확하게 조정되기 때문에 결과를 출력하는데 한 장당 40초가 걸릴만큼 속도가 느리다.

  YOLO는 R-CNN과 어느정도 유사성을 가진다. 각 격자 요소가 잠재적으로 바운딩 박스를 제안하고 컨볼루션 특징들로 이 박스에 점수를 매긴다. 그러나 YOLO에서는 <u>각 격자 요소가 제안하는 방식에 공간적인 제약을 두어 같은 객체에 다중 탐지가 도출되는 것을 완화</u>한다. Selective search에서 한 장단 2,000개의 바운딩 박스를 제안하는 것과 비교해서 YOLO에서는 98개의 바운딩 박스를 제안한다. 그리고 이 모든 과정이 유기적으로 연결되어 최적화가 진행된다. 

- Other Fast Detectors - Fast, Faster R-CNN은 계층마다 사이의 계산을 공유하고 Selective search 대신에 신경망으로 지역 후보들을 제안하는 방식으로 R-CNN의 속도를 높이는 데 주력했다. R-CNN보다 나은 속도와 정확도를 보이지만 실시간이라고 하는 성능에까지는 미치지 못한다. 

  많은 연구들이 DPM 파이프라인의 속도를 올리거나, Cascade를 사용하여 HOG 계산의 속도를 높이고 GPU에 계산을 할당하는 등의 연구로 속도 증가에 초점을 맞췄으나 30Hz DPM 만이 실시간성을 달성했다. 

  얼굴이나 사람 탐지 같은 단일 클래스를 위한 탐지기의 경우 훨씬 적은 경우의 변수 덕분에 고도로 최적화될 있다. <u>YOLO는 다양한 객체들을 동시에 찾아내는 법을 학습하는 범용 목적의 탐지기</u>이다.

- Deep MultiBox - R-CNN과는 다르게 Szegedy 등은 Selective Search 대신에 ROI를 예측하는 컨볼루션 신경망 네트워크를 훈련시켰다. MultiBox 모델도 Confidence 예측 방법에서 단일 클래스 예측 방법으로 방법을 교체하면서 단일 객체 탐지를 수행할 수 있다. 그러나 MultiBox 모델은 범용 객체 탐지 작업을 수행할 수 없고 단지 좀 더 큰 탐지 파이프라인의 한 부분일 뿐이며 더 많은 이미지 조각 분류 작업을 필요로 한다. <u>YOLO는 MultiBox와 다르게 하나의 완전한 객체 탐지 시스템</u>이다. 

- OverFeat - Sermanet 등은 위치 추정을 수행하는 컨볼루션 신경망 네트워크를 훈련시키고 이런 객체 탐지를 수행하는 위치 추정기를 조정한다. OverFeat은 효율적으로 Sliding window 방식의 탐지를 수행하지만 여전히 파이프라인의 각 과정이 독립적이다. OverFeat은 객체 탐지 성능이 아니라 위치 추정(객체의 클래스 추정 추가x)을 최적화 한다. DPM과 같이 위치 추정기는 예측을 할 때 지엽적인 부분을 본다. 전역적인 문맥을 추론 하지 않기 때문에 일관적인 탐지를 위해서 엄청난 사후 처리를 필요로 한다. 

- MultiGrasp - YOLO에서 바운딩 박스를 예측하는 과정은 Redmon이 제시한 MultiGrasp 시스템의 gra-sp detection에 기초한다. 그러나 grasp detection는 객체 탐지에 비하면 훨씬 간단한 작업이다.  Mul-tiGrasp은 하나의 객체를 포함하는 하나의 이미지의 single graspable region를 예측한다. 객체의 크기, 위치, 테두리, 클래스 등을 추정할 필요가 없고 grasping하기 적절한 지역만을 찾는다. 



## 4. Experiments

### 4.1 Comparison to Other Real-Time Systems

![](./Figure/You_Only_Look_Once_Unified_Real-Time_Object_Detection10.JPG)



### 4.2 VOC 2007 Error Analysis

![](./Figure/You_Only_Look_Once_Unified_Real-Time_Object_Detection11.JPG)

![](./Figure/You_Only_Look_Once_Unified_Real-Time_Object_Detection12.JPG)



### 4.3 Combining Fast R-CNN and YOLO

![](./Figure/You_Only_Look_Once_Unified_Real-Time_Object_Detection13.JPG)



### 4.4 VOC 2012 Results

![](./Figure/You_Only_Look_Once_Unified_Real-Time_Object_Detection14.JPG)



### 4.5 Generalizability: Person Detection in Artwork

![](./Figure/You_Only_Look_Once_Unified_Real-Time_Object_Detection15.JPG)

![](./Figure/You_Only_Look_Once_Unified_Real-Time_Object_Detection16.JPG)



## 5. Real-Time Detection In The Wild

[Demo of the YOLO system connected to a webcam](https://pjreddie.com/darknet/yolo/)



## 6. Conclusion

YOLO는 객체 탐지를 위한 (파이프라인의 각 과정들이) 통합된 모델이다. 구축하기 쉽고 이미지 전체에서 정보를 추출해 학습한다. 또, 탐지 성능을 측정하는 손실 함수르 통해 전체 모델이 유기적으로 학습될 수 있다. 

Fast YOLO는 말 그대로 객체 탐지를 위한 범용 목적의 탐지기 중에서 가장 빠르며 실시간 객체 탐지에서 최고의 성능을 보인다. 새로운 도메인에서 문제를 해결하기 위한 애플리케이션 구축에 맞게 일반화가 가능하다.
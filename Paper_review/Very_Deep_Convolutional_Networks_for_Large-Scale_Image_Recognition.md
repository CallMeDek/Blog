# Very Deep Convolutional Networks for Large-Scale Image Recognition

Karen Simonyan(Visual Geometry Group, Department of Engineering Science, University of Oxford),

Andrew Zisserman(Visual Geometry Group, Department of Engineering Science, University of Oxford)



## Abstract

저자들이 말하는 연구 목적은 큰 크기의 이미지 인식에서의 정확도와 CNN의 깊이의 연관성이다. 이를 위해서 16-19층의 네트워크와 작은 크기의 컨볼루션 필터(3x3)를 사용했다고 한다. 



## Introduction

당시에 여러 CNN 관련 알고리즘이나 연구들이 좋은 성과를 거둔 이유로 저자들은 ImageNet과 같은 대용량 데이터셋과 GPU, 분산 클러스터링같은 높은 성능의 컴퓨팅 시스템을 들었다. 특히 ILSVRC에서는 층이 깊은 비전 관련 아키텍처들이 활약을 했다고 한다. 

Krizhevsky 등의 CNN이 주목을 받으면서 원본 아키텍처를 더 발전시키려는 노력이 있었다. 2013 ILSVRC에서는 첫번째 컨볼루션 계층의 필터 사이즈와 스트라이드를 작게 하는 시도가 최고의 성능을 보여줬다. 또 다른 종류의 시도로는, 여러 크기의 이미지 전체를 훈련과 테스트 시에 다루기는 방법도 있었다. 이 연구에서 저자들은 깊이에 주목했는데 작은 크기의 필터 사이즈 덕분에 깊이를 늘리는 것이 가능했다고 한다. 



## Convnet Configurations

CNN의 깊이에 따른 효과를 보기 위해서 Ciresan 등의 연구와 Krizhevsky 등의 연구에서 적용했던 설정을 그대로 차용했다고 한다. 

### Architecture

훈련간에 입력은 224x224 크기의 RGB이미지이고 데이터 전처리는 각 RGB 픽셀 값의 평균을 뺀 것 밖에 없다. 필터 사이즈는 3x3이다. 1x1 필터도 사용했는데 비선형을 추가하고 입력 이미지의 채널 수를 바꾸기 위한 용도로 사용했다(입력 이미지에 대한 선형 변환으로 볼 수 있음). 스트라이드는 컨볼루션 연산이 끝나고도 해상도가 유지될 수 있도록 설정되었다(3x3에서는 1). 어떤 컨볼루션 계층 뒤에는 2x2 윈도우 사이즈의 스트라이드 2의 Max 풀링 계층이 추가되었다. 

컨볼루션 계층 뒤에는 3개의 Fc가 붙는다. 처음 2개 계층은 4096의 채널을 갖고 마지막에는 ILSVRC 분류 레이블이 1000개이므로 1000개의 채널을 갖는다. 최종 계층은 Soft max 계층이다. 

모든 히든 계층에는 ReLU 활성화 함수가 적용되었고 LRN(Local Response Normalization)은 사용되지 않았다. 왜냐하면 저자들이 관찰했을때, LRN은 성능을 개선시키기는 커녕, 메모리나 시간 등의 리소스만 소비했기 때문이다. 이때 LRN의 설정은 Krizhevsky 등의 연구를 따랐다. 

### Configurations

이 연구에서 평가된 CNN 아키텍처와 설정은 아래와 같다. 기본적인 설정은 위에서 설명한 설정을 따르되, 차이점은 계층의 숫자이다. 컨볼루션 계층의 채널 수는 64부터 시작해서 512가 될 때까지 Max 풀링 계층을 지날때마다 2배씩 증가한다.

![](./Figure/Very_Deep_Convolutional_Networks_for_Large-Scale_Image_Recognition1.JPG)

아래 테이블은 각 설정의 모델 파라미터 숫자를 나타낸 것인데 깊이가 깊어짐에도 불구하고 다른 연구에서의 얇고 큰 CNN보다 파라미터 숫자가 적은 것을 확인할 수 있다. 

![](./Figure/Very_Deep_Convolutional_Networks_for_Large-Scale_Image_Recognition2.JPG)


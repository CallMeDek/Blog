# MnasFPN : Learning Latency-aware Pyramid Architecture for Object Detection on Mobile Devices

Bo Chen, Golnaz Ghiasi, Hanxiao Liu, Tsung-Yi Lin, Dmitry Kalenichenko, Hartwig Adam, Quoc V. Le(Google Research)



## Abstract

저자들이 주장하길 리소스에 대한 제약사항이 있는 환경 하에서 괜찮은 모델의 아키텍처를 탐색하는 연구가 많이 이뤄졌지만 온디바이스 환경에서의 Object detection을 위한 디자인은 대부분 수동적으로 이뤄졌다고 한다. 이에 대한 몇가지 시도는 수행되었으나 모바일 친화적이지 않은 Search space 위주이거나 온디바이스 Latency가 반영되지 않거나 하는 문제가 있었다. 저자들은 MnasFPN이라고 하는, Detection head를 위한 모바일 친화적인 Search space를 제안했다. 그리고 이를 Latency를 인식하는 아키텍처 Search와 합쳐서 Object detection 모델을 만들었다. Ablation study에서는 성능 향상의 주요 원인이 저자들의 Search space의 혁신성에서 비롯된다는 것을 보여줬다. 



## Introduction

모바일 환경에 배치할 모델을 위한 아키텍처를 디자인 하는 것은 쉽지 않다. 왜냐하면 모델의 용량과 정확도 사이에 면밀한 Trade-off를 잘 조절해야 하고 디바이스가 모델이 사용하는 연산을 지원하는가 혹은 연산이 디바이스 친화적(연산을 빠르게 잘 수행)인가를 고려해야하기 때문이다. Neural architecture search는 이런 디자인 과정을 자동화하는 프레임워크를 제공했다. 여기서 강화학습 기반의 Controller는 유저가 지정한 Search space 하에 빠르고 정확한 모델을 만들어 내는 법을 학습한다. 그런데 NAS 연구는 Search 알고리즘을 개선하는데 집중하는데 반해 Search space를 디자인하는 작업은 성능에 큰 영향을 끼침에도 불구하고 연구가 비교적 덜 수행되었다고 한다. 

모바일과 서버 기반의 Image classification을 위한 NAS 관련 연구는 큰 진보를 이뤄낸 반면 Object detection 관련해서는 상대적으로 적은 시도가 이뤄졌다고 한다. 그 이유 중 어느정도는 Backbone과 관련해서 Detection head의 Search space에서의 복잡성 때문이다. Backbone에서는 연속적으로 이미지의 특징을 추출하게 되는데 이때 특징이 점점 미세해진다. Object detection과 Image classification에서의 이미지 추출 과정은 유사하다. 그러므로 원래 NAS 방식의 연구들은 Classification feature extractor를 Detection에 맞게 재조정하거나 Detection head는 고정시켜놓고 Backbone을 탐색하거나 하는 방식이었다. Backbone은 계층의 연속이기 때문에 이때의 Search space는 연속적일수 밖에 없다. 이에 반해 Detection head는 연속적이지 않을 가능성이 크다. 여기서는 특징들을 합치거나 여러 크기에 맞춰 다시 만들어내거나 할 수 있다. 이렇게 하는 이유는 Class 예측이나 객체의 위치 추정을 더 잘 하게 하기 위해서이다. 그러므로 이때의 Search space는 어떤 특징을 합칠 것인가(Fuse) 얼마나 자주, 어떤 순서로 특징들을 합칠 것인가 하는 과정을 포함한다. 이것을 NAS 방식 아키텍처로 해결하는 것은 어렵다는 것을 몇몇 연구가 입증했다.

한가지 예외는 NAS-FPN인데 Detection head의 비연속적인 Search space 문제를 다룬 첫 번째 NAS 논문이다. 이 연구에서는 정확도 만을 위해 최적화를 수행했고 SOTA 성능을 보였다. 그리고 수동적으로 NAS-FPNLite라고 하는 원본과 다른 변경 버전을 디자인해서 모바일에서 괜찮은 성능을 보여줬다. 그러나 NAS-FPNLite는 세 부분에서 한계를 가진다.

- 아키텍처를 만들어내는 탐색 과정이 연산 복잡성이나 온디바이스에서의 지연율에 의해서 주도되지 않았다. 
- 모바일에 맞게 수동적으로 아키텍처가 조정되었기 때문에 더 최적화가 가능할 여지가 남아있다.
- 원본 NAS-FPN search space는 모바일에서의 사용을 위한 모델을 만들어 내는 목적이 아니었다. 

저자들이 말하길 MnasFPN에서는 위의 이슈들을 다룬다고 한다. 구체적으로 이 Search space에서는 Depthwise 컨볼루션이 최적화 되어 있어서 모바일에 알맞다고 하고, Inverted residual block 개념을 Search space에 다시 도입해서 모바일 CPU 환경에서 Detection head에 효율적이라는 것을 증명했다고 주장한다. 저자들은 온디바이스에서 지연율이 주도하는 Search space에서 NAS를 수행했다. 

그래서 저자들이 주장하는 저자들이 기여하는 바는 다음과 같다. 

- 모바일 친화적인 Detection head를 위한 Search space
- Object detection을 위한 지연율 인식 탐색을 첫번째로 수행
- SSDLite, NAS-FPNLite를 능가한 Detection head 아키텍처



## Related Work

### Mobile Object Detection Models

대부분의, 모바일에서의 Detection 모델은 전문가에 의해서 수동적으로 구축된다. SSDLite는 Light-weight detection head 아키텍처로 유명한 모델 중 하나이다. 이 모델에서는 모바일에서의 연산 부담을 줄이기 위해서 SSD head의 3x3 컨볼루션을 Separable 컨볼루션으로 바꿨다. 이 방법은 NAS-FPN을 모바일에 탑재하기 위해 만든 NAS-FPNLite에도 적용되었다. SSDLite와 NAS-FPNLite는 MobileNetV3와 같은 Backbone과 쌍을 이룰수 있다. 



### Architecture Search for Mobile Models

저자들의 NAS 탐색 방법은 온디바이스에서 측정되는 Latency signal에 의해서 작업이 진행된다. Latency 인식 NAS 방법은 이 연구 이전에 NetAdapt와 AMC에서 Pre-trained된 모델의 채널 차원을 학습하기 위해 사용되었다. 이때 Look-up table(LUT)이 네트워크의 각 부분들의 지연율의 합에 근거하여 종단간의 지연율을 효율적으로 측정하기 위해 사용되었다. 이 아이디어는 MnasNet에서 NAS 프레임워크를 사용해서 Generic 아키텍처 파라미터를 탐색하기 위한 용도로 확장되었다. 이때 강화학습 기반의 Controller는 각종 아키텍처의 지연율과 정확도를 관찰하고나서 효율적인 아키텍처를 만들어내는 법을 학습한다. 이 프레임워크는 MobileNetV3와도 결합되었다. 

MnasNet 스타일의 탐색방법은 리소스 사용이 제한되는 연구자들에게는 그림의 떡이었다. 그러므로 NAS 연구 결과들의 비대한 몸집에도 불구하고 이 문제를 어느정도 해소하고자 탐색 효율성을 개선하는 방법쪽으로 초점이 맞춰졌다. 이 방법들은 하이퍼네트워크와 가중치 공유를 사용해서 탐색 효율성을 올리는 아이디어를 이용했다. 모바일에서의 Classification의 성공에도 불구하고 이런 탐색 방법들은 리소스가 제약되어 있는 상황에서 연속적이지 않은 Search space로까지 크게 확장되어 본적이 없다. 이런 이유로 저자들이 말하길 Mobile에서 Object detection 모델은 보기 힘들다고 한다. 



### Architecture Search for Object Detection

Object detection에서의 아키텍처 탐색의 비연속성이라는 특성때문에 Object detection에서의 NAS는 상당히 제한적이었다. 

NAS-FPN은 Detection head Search 문제를 다루는 선구자격 연구이다. 이 연구에서는 FPN에 근거해서 아주 중요한 Search space를 제안한 바 있다. 저자들의 연구도 NAS-FPN에 영감을 받았지만 목적이 좀 더 모바일 친화적인 Search space에 있다는 차이점은 있다. 

다른 선구자격 연구는 Auto-Deeplab이 있다. 여기서는 NAS search를 Semantic segmentation 문제에서 다뤘다. 저자들의 연구도 서로 다른 해상도의 연결 패턴을 학습하는, 유사한 문제에 직면했다. 

DetNAS는 Detection body를 탐색하는 것의 효율성을 개선시키는데 초점을 맞췄다. 이 연구에서는 탐색 동안 모든 샘플링된 아키첵처의 ImageNet pretraining의 필요성에 의해서 촉발된 관리하기 힘든 연산량 문제를 다뤘다. 저자들의 연구는 Body보다는 Head에만 관심을 뒀다. 

NAS-FCOS에서는 Object detection을 위한 탐색 과정의 속도를 높이기 위해서 Detection head에까지 가중치 공유 개념을 확장한 바 있다. NAS-FPN과 유사하게 Detection head를 위한 Search space는 모두 컨볼루션으로 이루어져 있으나, 모바일이 대상이 아니었다. 저자들의 연구는 이들의 연구에 보충적으로, 저자들의 모바일 친화적인 Search space에 근거한 Latency-aware 탐색은 그들의 가중치 공유 탐색 전략을 더 가속화 할 수 있다. 

모바일쪽에서 Object detection 아키텍처는 거의 주요 타겟으로서 최적화되지는 않는다. 그보다는 Classification을 목적으로 한 가벼운 Backbone과 미리 훈련시킨 Detection head로 구성된다. 저자들은 모바일에서의 모델 배치를 위한 Object detection용 아키텍처 최적화를 직접적으로 수행한다고 한다.  



## MnasFPN

저자들이 말하는 MnasFPN은 저자들이 제안하는 Search space와 NAS를 통해서 찾은 아키텍처 군을 의미한다. NAS-FPN(Lite)와 MnasFPN 모두 Feature extractor backbone과 반복적으로 쓰일수 있어서, 존재하는 Feature들의 쌍을 병합해서 새로운 Feature를 재귀적으로 만들어낼수 있는 Cell 구조로부터 Detection 네트워크를 구축한다. 각 Cell은 각기 다른 해상도의 Feature들을 써서 입력과 같은 해상도 수준의 Feature들을 출력한다. 그렇기 때문에 Cell 구조는 반복적으로 적용될 수 있다. 하나의 Cell은 Block들의 집합이다. 각 Block은 해상도가 다를 수 있는 두 Feature map을 Intermediate feature로 병합시키는데 Separable 컨볼루션으로 작업이 수행된다. MnasFPN은 NAS-FPN(Lite)와 Block 수준에서 차이 점이 있다. 



### Generalized Inverted Residual Block(IRB)

Inverted Residual block(IRB)는 NAS search space 관련 연구에서 흔히 사용되는 Block 아키텍처로 알려져 있다. IRB의 핵심 안점은 메모리에 부담을 덜어내기 위해서 낮은 차원의(채널) Feature로 연산을 수행하고 나서 Depthwise 컨볼루션으로 차원을 다시 확장 시키는 것이다. 저자들은 이에 착안해서 NAS-FPN search space에 IRB 같은 디자인을 적용하는 것을 고려했다. 여기서 주요 문제이자 혁신이라고 할 수 있는 점은 NAS-FPN 블럭 안에 비선형성 구조를 개선시키는데 있다. 



### Expandable intermediate features

NAS-FPN에서는 모든 Feature map이 의도적으로 C개의 같은 채널 수를 공유한다. 이와 비교해서 MnasFPN은 Intermediate feature size F라는, 좀 더 융통성을 부여했다. 이 F는 탐색하는 동안 값이 바뀌면서 C와는 독립적이다. F와 C를 조절하면서 Intermediate feature는 확장의 역할을 하거나(채널 수가 많아지거나) 병목 현상(채널 수가 적어지거나)의 역할을 한다. 1x1 컨볼루션이 필요에 의해서 입력 Feature 채널 수를 C에서 F로 바꿀때 적용될 수 있다. 



### Learn-able block count

NAS-FPN에서 각 셀당 블럭의 수는 미리 정의되어 있다. 이것은 Feature 재활용 매카니즘 때문이다. 만약에 블럭이 Cell의 출력에 사용되지 않으며 이 블럭의 Intermediate feature는, 같은 해상도와 크기를 가지는 출력 Feature에 더해지게 될 것이다. MnasFPN에서는 이와 반대로 Intermediate feature가 출력 Feature와 채널 사이즈가 다른 경우가 많다(F와 C). 결과적으로 사용되지 않는 블럭은 자주 버려지게 되고 Latency-accuracy trade-off를 결정하는데 유연성을 더할 수 있다. 



### Cell-wide residuals

Feature 간의 연결이 점점 더 얇아지면서 저자들이 알아 낸 사실은 같은 해상도의 모든 입력과 출력 쌍 사이에 Residual을 추가하면 정보의 흐름을 증강시키는데 도움이 된다는 것이다 IRB와 비슷하게 저자들은 Intermediate feature에 ReLU 비성형을 더했지만 출력 Feature에는 적용하지 않았다. 왜냐하면 Input/output feature 채널 사이즈 C는 의도적으로 메모리에 부담을 덜기 위해서 작게 설정되었기 때문이다. 정보를 잃을 수도 있는 비선형성 연산은 불필요하게 정보 흐름을 더 죽이는 결과를 가져올 수도 있을 것이다. 

Figure 1을 보면 IRB와 비슷한, 입력과 출력 Feature 사이의 연결된 경로를 확인할 수 있다.

![](./Figure/MnasFPN_Learning_Latency-aware_Pyramid_Architecture_for_Object_Detection_on_Mobile_Devices1.JPG)



### Size Dependent Ordering(SDO)

저자들이 말하는 MnasFPN의 또다른 혁신은 입력과 출력 해상도에 근거해서 컨볼루션 연산 종류와 네트워크의 구조를 동적으로 바꿀수 있다는 점이다. 저자들은 이를 Size Dependent Ordering(SDO)라고 부른다. 구체적으로 만약에 입력 Feature가 다운 샘플링될 필요가 있다면 다운 샘플링은 1x1 컨볼루션 전에 수행된다. 반대로 업 샘플링될 필요가 있다면 1x1 컨볼루션이 먼저 수행된다. 

이 디자인은 계산을 최소화한다. 표기상의 편의를 위해서 Feature map이 정사각형이라고 가정하고, R을 높이와 너비를 나타낸다고 가정한다. Feature map을 병합할때, 저자들은 초기 입력 해상도 R0와 채널 차원 C가 중간 해상도 R과 채널 차원 F와 맞지 않을때, Reshaping과 1x1 컨볼루션을 수행해서 이를 맞춘다. 

만약에 R0 > R(다운 샘플링이 필요한 경우)일때, R0 = kR이라고 한다면(k >= 2) 다운 샘플링은 Stride k의 kxk 컨볼루션으로 수행되어야 하고, 다운 샘플링 후에 1x1 컨볼루션을 수행할때 Cost는 아래와 같은 것이다.

![](./Figure/MnasFPN_Learning_Latency-aware_Pyramid_Architecture_for_Object_Detection_on_Mobile_Devices2.JPG)

이에 반해 1x1 컨볼루션 후에 다운 샘플링을 수행할 경우는 아래와 같다.

![](./Figure/MnasFPN_Learning_Latency-aware_Pyramid_Architecture_for_Object_Detection_on_Mobile_Devices3.JPG)

F >= 2인 조건이 합리적이라고 한다면 아래의 조건을 만족하므로 

![](./Figure/MnasFPN_Learning_Latency-aware_Pyramid_Architecture_for_Object_Detection_on_Mobile_Devices4.JPG)

Cost2와 Cost1의 차이는 아래와 같게 된다.

![](./Figure/MnasFPN_Learning_Latency-aware_Pyramid_Architecture_for_Object_Detection_on_Mobile_Devices5.JPG)

따라서 다운 샘플링 후에 1x1 컨볼루션을 적용하는 것이 더 실속있다는 것을 알 수 있다. 반대의 경우도 비슷하게 증명할수 있다. 



### MnasFPN Search

MnasFPN의 Feature 생성 과정과 모든 탐색가능한 요소는 Figure 1에 묘사되어 있다. 각 Feature 생성 블럭에 대해서 저자들은 병합할 두 입력 Feature, 합쳐진 Feature의 해상도 R과 채널 차원 F, 병합 연산(Addition 혹은 SE), 병합 후에 Depthwise 컨볼루션의 커널 크기를 탐색한다. 전체 네트워크 차원에서는 입력과 출력 그리고 생성된 Feature가 같은 채널 차원인 C를 공유하는데 이는 탐색 대상 값이다. 

저자들은 MnasNet의 아키텍처 탐색 프레임워크를 도입해서 Latency 측정 값을 탐색 Objective에 포함시켰다. 저자들은 강화 학습 기반 컨트롤러를 훈련시켜서 다음과 같이 정의되는 Reward function을 극대화 하는 네트워크 아키텍처를 Controller가 제안하도록 했다. 샘플링된 아키텍처 m이 훈련되고 평가되는 과정은 Proxy task로 진행된다. Proxy task는 실제 Task의 축소 버전이다. Proxy task 성능은 mAP(m), 네트워크의 온디바이스에서의 지연율인 LAT(m)이 결합된 상태로 측정된다. 

![](./Figure/MnasFPN_Learning_Latency-aware_Pyramid_Architecture_for_Object_Detection_on_Mobile_Devices6.JPG)

w < 0일때는 Latency와 Accuracy의 Trade off point를 조절한다. 이론적으로 w는 원하는 Latency에서 성능 Trade-off 곡선을 딱 자르는 접선의 기울기이다. 실제로는 저자들이 관찰한 바에 따르면 원하는 Latency 주변의 성능을 보이는 아키텍처 또한 최적화된 아키텍처가 될 수 있고 Search space의 성능 경계가 비슷한 곡률을 보이기 때문에 w는 한 번만 설정될 필요가 있다고 한다. 

Controller는 반복적으로 후보 아키텍처 m을 제안하고, Proximal Policy Optimization을 적용해서 Reward feedback인 Reward(m)에 근거해서 Controller 자신을 훈련시킨다. 모든 탐색 실험 후에 모든 샘플링된 아키텍처는 Figure 6와 같이 성능 경계를 추적하게 된다. 

 ![](./Figure/MnasFPN_Learning_Latency-aware_Pyramid_Architecture_for_Object_Detection_on_Mobile_Devices7.JPG)

그런 다음 가장 성능이 유력한 아키텍처를 성능 경계 곡선에 따라서 실제 Task에 배치할 수 있다. 

#### Connectivity-based LUT

저자들은 탐지에 맞도록 Latency look-up table을 조정해서 적용해서 LAT(m)를 측정했다. 이 연구 이전의 LUT 방식은 MnasFPN과 맞지 않았다. 왜냐하면 블럭의 숫자와 Head의 연결 패턴이 동적이기 때문이었다. 그 대신에 저자들은 런 타임에 각 모델의 계층 연결성을 계산해서 계층이 look-up table에 포함될지 안될지를 결정하도록 했다. 이런 Connectivity-based LUT는 온디바이스 측정과 높은 친밀성을 보였다(R^2 > 0.97)



### Connectivity Search

저자들이 말하길 MnasFPN search space의 디자인은 간결하다고 한다. 저자들이 이렇게 의도한 이유는 당시 Search 알고리즘들이 완벽하지 않고 큰 Search space가 항상 좋은 모델을 제안하는 것은 아니기 때문이라고 한다. 그러므로 Search space 디자인은 무엇을 포함하고 포함하지 않을 것인지와 관련된 문제가 된다. 

저자들이 포함하지 않은 디자인 요소는 일반적인 연결 패턴을 탐색하는 것이다. 이 방법은 MNAS Controller에게는 부담이 많이 되지만 Search 알고리즘에게는 개선될 여지를 남겨둔다. 저자들은 탐색의 퀄리티가 탐색 효율성 뿐만 아니라 네트워크 연결 부분의 디자인 편향성에 의해서 질이 떨어질 수 있다는 연구 결과(Randomly-wired network)를 보고 병합 시에 두 개의 Feature만 선택되도록 했다. 저자들은 Conn-Search라고 하는 새로운 Search space를 디자인 했는데 D >= 2 이상의 서로 다른 Feature map들이 병합되는 것을 허락했다(저자들의 연구에서는 D=4).




## Experiments

저자들은 COCO object detection으로 실험을 실시했고 Latency-aware search와 Search space의 각 요소들의 효율성을 알아보기 위한 Ablation study를 수행했다. 

### Search Experiments and Models

저자들은 아래와 같이 실험을 실시했는데 여기서 모든 Search space에서는 Cell 마다 5개의 블럭을 허용했다. 

- MnasFPN: Figure 1에 나와 있음.
- NAS-FPNLite: NAS-FPN을 가볍게 만든 모델로서 Head의 모든 컨볼루션을 Separable 컨볼루션으로 대체했다. 이 모델들은 Latency-sensitive NAS를 통해 탐색되지 않은 유일한 모델들이다(3.3절 참고). 
- NAS-FPNLite-S: 모든 컨볼루션을 Separable 컨볼루션으로 대체한, NAS-FPN search space를 탐색한다. NAS-FPNLite와 차이점은 모델 구축 후에 이런 대체를 한 것이 아니고 애초에 Search space에서 대체를 한 뒤에 모델 구축 과정을 탐색하는 것이다. 
- No-Exand: 중간 과정에서의 모든 Feature들에 대해서 F = C를 강제해서 MnasFPN에서 Expansion 부분만 제거한 실험이다. 이 실험으로 알 수 있는 것은 IRB에서 Expansion의 영향력을 알아볼수 있다는 것이다. 
- Conn-Search: 블럭마다의 2개에서 D>= 2 사이의 서로 다른 입력(Feature map)을 합병하는 것을 용인한다.  단, 합병 연산은 덧셈만 가능하다. 

![](./Figure/MnasFPN_Learning_Latency-aware_Pyramid_Architecture_for_Object_Detection_on_Mobile_Devices8.png)

![](./Figure/MnasFPN_Learning_Latency-aware_Pyramid_Architecture_for_Object_Detection_on_Mobile_Devices9.png)



### Experimental Setup

저자들은 여기서 모든 Detection model을 같은 설정과 하이퍼 파라미터로 훈련시켰고 Ablation study는 5k COCO val2017 데이터셋에서 수행되었다. 최종 비교 결과는 COCO test-dev 데이터셋으로 평가되었다. 

#### Training setup

- COCO val2017: 각 모델은 150 epochs 혹은 277k steps 동안 훈련되었고 배치 사이즈는 64로 COCO train2017 데이터셋으로 훈련시켰다. 훈련은 8 Replicas로 동기화되었다. LR은 다음과 같은 스케쥴을 따랐다. 0부터 0.04까지 첫번째 에폭동안 선형적으로 증가하다가 이후부터는 값을 고정시켰다. 120 epoch과 140 epoch에서 각각 0.1까지 급격하게 줄었다. 훈련의 안정성을 위해서 Gradient-norm clipping을 10 epoch에서 적용했고 Ablation study에서는 MobileNetV2를 Backbone으로 사용한 모델을 ImageNet 데이터셋으로 Pre-training시켰다. 
- COCO test-dev: 각 모델은 100k steps동안 처음부터(Pre training 없이) 훈련시켰고 32의 동기화된 Replicas로 배치 사이즈 1024로 훈련시켰다. LR은 4부터 0까지 Decayed되는 Cosine 스케쥴링을 적용했다. 단 처음 2k steps 동안은 Linear warmup phase를 적용했다. 경쟁력을 보장하기 위해서 COCO train2017과 val2017 데이터셋을 훈련 셋으로 병합했다. 

모든 훈련과 평가는 320x320 크기의 입력 이미지를 사용했다. 또 Drop-block이나 Auto-augmentation이나 하이퍼 파라미터 튜닝을 적용하지 않았는데 이는 비교 연구에서 특정 유형의 모델을 선호하는 것을 피해서, 각종 문헌의 기존 결과와 공평하게 비교하기 위함이다. 

#### Timing setup

모든 Timing은 Pixel 1 디바이스에서 TensorflowLite의 Latency benchmarker를 사용해서 배치 사이즈 1의 Single-thread로 수행되었다. MobileNetV2의 관행을 따라서 각 Detection model은 TensorflowLite flatbuffer format으로 변경되어 NMS를 수행하기 바로 직전의 Box와 Class prediction 값을 출력한다. 

#### Architecture Search Setup

저자들은 MNASNet에서 사용된 Controller setup을 따랐다. Controller는 10K의 자식 모델을 샘플링하는데 TPUv2 디바이스에서 각각 1시간까지 걸리기도 한다. 모델을 훈련시키기 위해서 저자들은 COCO train2017 데이터셋을 임의로 111k-search-train 데이터셋과 7k-seach-val 데이터셋으로 나눴다. 저자들은 20 epochs동안은 배치 사이즈 64로 search-train 셋으로 훈련시키고 search-val 셋으로 mAP 척도로 모델을 평가했다. LR은 첫 번째 Epoch동안 0부터 0.4로 선형적으로 증가하고 Step-wise precedure를 따라 Epoch 16에서 0.1로 Decay시킨다. 저자들은 320x320 해상도의 이미지로 Proxy task로 훈련을 수행해서 Proxy task와 Main task의 측정된 Latency가 동일하도록 했다. Reward objective에 대해서는(Eq 4) w = -0.3으로 설정했는데 이는 약간의 실험을 미리 실시했을때 측정된 결과이다.

훈련이 끝나고 모든 샘플링된 모델에 대해서 성능 곡선을 측정했고 166ms, 173ms, 180m의 Latency에서 가장 성능이 좋은 모델을 뽑았다. 그런 다음 반복을 3에서 5로 증가시켜 총 3x3=9의 모델을 생성했다. 이 중에서 저자들은 Latency나 mAP에 편향되지 않는 모델만을 남겨뒀다. 



### Discovered Architectures

저자들은 최고의 성능을 보이는 MnasFPN 아키텍처와 NAS-FPNLite-S 아키텍처를 Figure 2와 3에 각각 나타냈다. 두 모델 모두 NAS-FPNLite 모델과 유사한 Latency를 보였다. 

| MnasFPN                                                      | NAS-FPNLite-S                                                |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](./Figure/MnasFPN_Learning_Latency-aware_Pyramid_Architecture_for_Object_Detection_on_Mobile_Devices10.png) | ![](./Figure/MnasFPN_Learning_Latency-aware_Pyramid_Architecture_for_Object_Detection_on_Mobile_Devices11.png) |

두 모델을 비교했을때 알 수 있는 점은 다음과 같다.

- MnasFPN은 대부분 간결하다. 5개의 내부 블럭이 주어짐에도 불구하고 MnasFPN은 하나의 블럭만 사용한다. 이에 반해 NAS-FPNLite-S는 5개를 사용하고 이 블럭들을 모두 같은 해상도에 배치한다. MnasFPN의 간결함은 저자들이 추정하길 다음과 같은 요소들때문인것으로 추정한다. 사용되지 않는 블럭을 잘라낼수 있는 능력, 각 블럭의 용량을 증가시킬수 있는 IRB에서의 확장력
- 두 Feature을 병합하기 위한 Squeeze-and-excite(SE) 옵션은 사용되지 않는다. 
- MnasFPN과 NAS-FPNLite-S 모두 20x20 해상도를 Intermediate Feature로서 선호한다. 이런 경향은 여러번 탐색을 수행하고 여러 Search space의 변경체들로 탐색을 진행해도 같은 경향을 보였다. 

![](./Figure/MnasFPN_Learning_Latency-aware_Pyramid_Architecture_for_Object_Detection_on_Mobile_Devices12.png)

Figure 4는 D=4의 Conn-search 아키텍처를 보여준다. 먼저 MnasFPN과 유사하게 Intermediate feature의 해상도는 모두 20x20과 가까운 것을 확인할 수 있다. 두번째로 거의 모든 경우가 2개 아니면 3개의 특징들이 병합되었다. 그러므로 4개의 입력을 연결하는 것은 과도하거나 저자들 연구에서의 Search space가 Search 알고리즘이 다룰 수 있는 어떤 한계에 있음을 확인할 수 있다. 



### Latency Breakdowns

저자들은 MnasFPN 아키텍처를 Feature extractor backbone, Detection head, Predictor로 나눴다. Predictor는 여러 컨볼루션과 Class predictor 그리고 Box decoder로 이뤄져 있다. 이 컨볼루션들은 CxC 크기이고 C는 MnasFPN의 출력 채널 차원수와 동일한 파라미터이다. 그러므로 저자들은 탐색은 Head와 네트워크의 Predictor의 부분적으로 영향을 끼치게 된다. 

MnasFPN detection head의 개선 정도를 살펴보기 위해서 저자들은 두 200ms 모델의 Latency breakdown을 그렸다(MnasFPN with 5 repeats(25.5 mAP), NAS-FPNLite with 6 repeats(24.4 mAP)). 

![](./Figure/MnasFPN_Learning_Latency-aware_Pyramid_Architecture_for_Object_Detection_on_Mobile_Devices13.png)

Figure 5에서와 같이 저자들의 Search는 거의 80ms(49ms + 25ms) 혹은 40%의 시간에 영향을 끼친다. MnasFPN(C = 48)은 거의 2배정도되는 리소스를 Predictor보다는 Head에 할당했고 NAS-FPNLite(C=is 64)는 Predictor에 좀 더 리소스를 할당했다. 이는 MnasFPN에서 Detection head의 Early feature fusion과 상당한 연관이 있는 현상임을 암시한다. 

또 위의 Figure 5는 Backbone 모델이 더 성능 Bottleneck이 되었다는 것을 알려준다(Right보다 파란색의 비율이 큼). 즉, NAS search 과정에서 MnasFPN이 더 효율적이라는 것을 알 수 있다. 



### Ablation on IRB

저자들이 IRB를 Detection head에 도입 한 것의 효과를 알아보기 위해서 Figure 7에 MnasFPN, NAS-FPNLite-S 그리고 No-Expand(Intermediate 모든 채널의 차원수를 F=C로 강제) 했을때 성능 곡선을 보여준다. 

![](./Figure/MnasFPN_Learning_Latency-aware_Pyramid_Architecture_for_Object_Detection_on_Mobile_Devices14.png) 

MnasFPN과 NAS-FPNLite-S는 Latency-aware 탐색을 똑같이 적용하지만 Search space가 다르다. MnasFPN seach space의 디자인이 NAS-FPNLite모델의 성능보다 뛰어난 모델을 생성해내는데 기여했음을 알 수 있다. MnasFPN과 No-Expand의 차이는 MnasFPN 블럭에서 Expansion의 적용 여부이다. No-Expand 모델의 성능은 MnasFPN 모델보다 훨씬 떨어진다. 학습된 모델 아키텍처를 살펴봤을때, 모델이 채널 차원을 C에서 16으로 줄였을때 Intermediate 노드의 수는 늘린 것을 확인할 수 있다. 이것은 NAS controller가 계속해서 빠지는(정체되는) 차선의 디자인 전략이다. 결과적으로 Figure 6를 봐도 알 수 있듯이 No-Expand의 성능 곡선이 다른 방법들에 비해 차선의 결과를 냄을 알 수 있다. 

![](./Figure/MnasFPN_Learning_Latency-aware_Pyramid_Architecture_for_Object_Detection_on_Mobile_Devices15.png)



### Ablation on Latency-aware Search

저자들이 주장하길 저자들의 방법이 객체 탐지를 위한 아키텍처 탐지에 Latency-aware 훈련 방식을 도입한 첫 사례라고 한다. Latency signal을 이용하는 것의 이점을 조사하기 위해서 저자들은 MnasFPN과 NAS-FPNLite 그리고 NAS-FPNLite-S를 비교했다. 

Figure 7에 의하면 MnasFPN은 NAS-FPNLite보다 성능 상의 확실한 우위를 보인다. NAS-FPNLite-S도 NAS-FPNLite보다는 우위지만 그 정도가 MnasFPN보다는 덜 하다고 한다. 저자들이 말하길 MNASNet 스타일의 Latency-aware 탐색이 전체적으로 효과적이긴 하지만 MnasFPN의 성능 우위의 주요 요인은 Seach space 디자인에 있다고 한다. 



### Connectivity Search

MnasFPN search space가 충분히 큰 지 측정하기 위해서 저자들은 Conn-Search(D=4) 모델과 비교했다. Figure 7을 보면 MnasFPN을 포함하는 충분히 큰 Search space를 가짐에도 불구하고 Conn-Search 모델은 차선의 Latency-accuracy tradeoff를 보였다. Figure 6에서 저자들은 Conn-Search의 성능 곡선이 MnasFPN보다 경미하게 낮다는 것을 확인했는데 이를 통해서 Controller가 Search space를 충분히 탐색할 수 없다는 것을 확인했다. Table 2를 보면 Conn-Search의 Cardinality가 MnasNet과 NAS-FPN의 Cardinality보다 훨씬 큰 것을 확인할 수 있다. 이 결과는 Search space와 Search 알고리즘의 공동 조정이 중요함을 말해준다. NAS가 수동으로 튜닝하는 일을 없애주고 모든 Search space를 포함하는 충분히 강력한 Search space만 디자인하면 된다고 생각할 수 있지만 현실은 Search 알고리즘이 임의의 큰 Search space를 다룰만큼 충분히 강력하지 못하는 것이다. 그러므로 NAS paper에서 했던 것처럼 Search space의 공동 조정과 반복적인 축소 작업은 여전히 유효하다. 



### Ablation on SDO

SDO의 영향을 알아보기 위해서 저자들은 MnasFPN의 아키텍처에서 SDO를 적용하지 않고 4번이나 5번 반복했다. SDO 개념이 적용되지 않는 모델들은 Feature map 크기를 조정하기 전에 1x1 컨볼루션을 수행하는데 이는 Down-sampling의 경우 비실속적이다. 그리고 샘플링된 MnasFPN 아키텍처는 Down-sampling 연산에 의해 지배당하게 된다. 

![](./Figure/MnasFPN_Learning_Latency-aware_Pyramid_Architecture_for_Object_Detection_on_Mobile_Devices16.png)

Table 3를 보면 SDO를 적용하지 않은 것이 mAP에 영향을 끼치지는 않지만 Latency를 8에서 11ms 더 느리게 증가시킨다. Backbone을 고려하지 않은 나머지 부분으로만 봤을때 이 양은 12%~14%에 해당하는 Latency 부분이 최적화 여지가 남게 되는 것이다. 



### Performance Comparison on COCO Test-dev

저자들은 MnasFPN의 다양한 Backbone들과 여러 온디바이스 Detection head 모델들과 성능을 비교했다.  

![](./Figure/MnasFPN_Learning_Latency-aware_Pyramid_Architecture_for_Object_Detection_on_Mobile_Devices17.png)



## Conclusion

이 연구에서 저자들은 기존의 연구들이 Classification에서의 학습된 Backbone을 Trasfer 하는 것 대신에 Object detection에 맞는 아키텍처를 직접적으로 찾는 접근 방식을 보여줬다. 추가적으로 Search 프로세스를 디자인하고 타겟 플랫폼에 대한 지식을 포함하기 위한 Search space를 디자인했다. 저자들이 말하는 MnasFPN의 혁신점은 다음과 같다. 

- MnasFPN은 IRB을 Detection head에 포함시켰고 Mobile CPU 환경에서 효과적임을 보였다.
- MnasFPN은 Head에 있는 Reshaping과 컨볼루션 연산을 다시 구축해서 여러 크기의 정보를 효과적으로 병합할 수 있도록 했다. 

Ablation study를 통해서 저자들은 위의 Search space 디자인에서의 두 혁신점이 성능을 끌어올리는데 필요하다고 생각했다. 반면에 Feature map 연결 부분으로 Search space를 확장시키는 개념은 NAS 프레임워크의 능력 밖으로 봤다. 결과적으로 저자들은 MnasFPN search space가 NAS controller의 용량과 연관이 있다고 봤다. Controller가 좀 더 강력해질수록 MnasFPN과 Connectivity search도 가능할 것이라고 봤다. 

COCO test-dev 데이터로 MnasFPN은 NAS-FPNLite보다, Backbone이 없는 나머지 부분에서 25%의 성능 향상을 가져왔다고 한다. 이 개선 정도가 상당하기 때문에 개선이 일어나지 않은 나머지 부분이 성능 향상에 있어서 Bottleneck이 된다고 주장한다. 예를 들어서 Backbone의 경우 전체 Latency의 60%를 차지 하고 있는데 MnasFPN head를 조건으로 하거나, 같이 탐색될 수 있다. 이 작업이 유용할 수 있다고 보는 이유는 Table1에 나와 있는 것처럼 MnasFPN는 MobileNetV3 혹은 Depth-multiplied MobileNetV2 Backbone과 상성이 맞기 때문이다. Backbone과 Head를 함께 탐색하는 작업은 Cardinality 면에서 한계가 있지만 최근에 One-shot NAS 방법이 활로를 보여줬다고 한다. 



## Appendix

### Search space cardinality comparison

본문 참고.

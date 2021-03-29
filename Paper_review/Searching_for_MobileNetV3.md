# Searching for MobileNetV3

Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen (Google AI), Mingxing Tan (Google Brain), Weijun Wang, Yukun Zhu (Google AI), Ruoming Pang, Vijay Vasudevan, Quoc V. Le (Google Brain), Hartwig Adam (Google AI)



## Abstract

이 연구에서 저자들은 Complementary 탐색 기술과 참신한 아키텍처 디자인을 조합해서 새로운 MobileNet 모델을 설계하고자 했다. MobileNetV3는 NetAdapt 알고리즘에 의해 보충되는 하드웨어 인식 NAS에 의해서 모바일 CPU에 맞게 조정되었고 그동안 없었던 아키텍처 디자인을 통해 상당히 개선되었다고 한다. 저자들의 고민은 어떻게 (사람이 하는 것이 아니고) 자동으로 탐색을 하는 알고리즘과 네트워크 디자인을 잘 조합해서, Complementary 접근 방식을 활용해서 좋은 성능을 낼 것인가 하는 데에서 시작했다. 저자들은 그래서 두 가지 버전의 MobileNet 모델을 만들었다. MobileNetV3-Large와 MobileNetV3-Small인데 모델 적용 사례가 리소스를 많이 사용할 수 있는가 없는 가에 따라 각각을 사용한다고 한다. 이 모델들은 Object Detection, Semantic Segmentation 등의 수행하고자 하는 작업에 많게 적용된다. Semantic segmentation(혹은 Dense pixel prediction)의 경우, 저자들은 새로우면서 효율적인 Segmentation decorder인 Lite Reduced Atrous Spatial Pyramid Pooling(LR-ASPP)을 제안했다. 



## Introduction

신경망 기반의 모델이 점차 많이 쓰이면서 데스크탑을 통해 서버와 통신하는 구조가 아닌 모바일을 통해서 언제 어디서든 사용할 수 있는 환경으로 바뀌고 있다. 그러면서 높은 정확도와 낮은 지연율이 중요해지고 연산량을 줄여서 배터리 수명을 늘리는게 중요해졌다. 저자들은 두 가지 버전의 MobileNet 모델을 설계해서 이런 환경에서 좋은 성능을 보이게 하는 것이 목표였다. 이 목표를 이루기 위해서 자동화된 탐색 알고리즘과 아키텍처에서의 진보를 어떻게 결합하여 효율적인 모델을 구축할 것인가를 고민했다. 모바일 환경에서 정확도와 지연율의 Trade off를 최적화하는 컴퓨터 비전 아키텍처를 개발하기 위해서 저자들은 다음 개념을 도입했다. 

- Complementary 탐색 기술
- 모바일 환경에서 실제적으로 활용 가능한 새로운 버전의 비선형성 함수들
- 새로운 네트워크 디자인
- 새로운 Segmentation decoder



## Related Work

이 연구와 관련 있는 여러 선행 연구들은 본문 참고. 
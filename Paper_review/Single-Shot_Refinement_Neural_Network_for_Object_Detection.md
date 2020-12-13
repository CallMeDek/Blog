# Single-Shot Refinement Neural Network for Object Detection

Shifeng Zhang(CBSR & NLPR, Institute of Automation, Chinese Academy of Sciences, University of Chinese Academy of Sciences)

Longyin Wen(GE Global Research)

Xiao Bian(GE Global Research)

Zhen Lei(CBSR & NLPR, Institute of Automation, Chinese Academy of Sciences, University of Chinese Academy of Sciences)

Stan Z. Li(CBSR & NLPR, Institute of Automation, Chinese Academy of Sciences, University of Chinese Academy of Sciences)



## Abstract

보통 Two-stage 알고리즘들은 정확도에 강점이 있고 One-stage 알고리즘들은 효율성에 강점이 있다. 저자들은 RefineDet이라고 하는 Single-shot Detector로 정확도는 Two-stage 알고리즘보다 높으면서 One-stage의 효율성을 유지하고자 했다. 이 Detector는 네트워크 내에 두 개의 모듈로 구성되어 있다. 

- Anchor refinement module: Classifier가 답을 찾는 Search space의 크기를 줄이기 위해서 Negative anchor를 필터링하는 것. 다음 모듈이 학습을 시작할때 좀 더 잘 학습할수 있도록 위치와 크기가 어느정도 조정된 Anchor들을 제공하는 것.
- Object detection module: 위의 모듈에서 Anchor를 받아서 Bounding box regression과 Multi-class prediction을 수행한다. 

동시에 저자들은 Transfer connection block이라는 개념을 디자인해서 ARM에서 작업을 수행하고 난 뒤의 Feature들을 ODM으로 Transfer한다. Multi-task Loss function을 정의해서 전체 네트워크가 End-to-End로 학습이 가능하게 했다. 

[코드 이용](https://github.com/sfzhang15/RefineDet)




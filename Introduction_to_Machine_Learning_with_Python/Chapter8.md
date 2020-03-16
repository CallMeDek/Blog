# 8. 마무리

### 8.1 머신러닝 문제 접근 방법

- 전체 데이터 분석과 의사 결정 과정에서 머신러닝 알고리즘이 차지하는 부분은 보통 작다.

- 먼저 답을 얻고자 하는 질문이 어떤 종류인지 생각해본다.

  - 탐색적 분석으로 데이터를 분석한다.
  - 이미 정한 목표가 있는지 확인한다.
  - 목표가 있다면 시스템을 구축하기 전에 성공을 어떻게 정의하고 측정할 것인지 정한다.
  - 최종 솔루션이 비즈니스와 연구의 목표에 어떤 영향을 줄지 생각해 본다. 

- 데이터를 모으고 작동하는 프로토 타입을 만든다. 

  - 모델은 데이터 과학 워크 플로의 일부이다.
  - 모델 구축은 데이터를 모으고, 정제하고, 만들고, 평가하는 순환 사이클의 한 부분이다.
  - 모델이 만든 오류를 분석하면 빠진 데이터나 추가로 수집할 데이터, 더 효과적인 모델을 위해 어떻게 작업을 재구성할지에 대한 정보를 얻을 수 있다.
  - 더 많은 데이터를 수집하거나 작업 흐름을 바꾸는 것이 매개변수 튜닝이나 그리드 서치를 돌리는 것보다 이득일 수 있다.

  

##### 8.1.1 의사 결정 참여

의사 결정에 사람이 개입해야 할지 고려해야 한다. 자율 주행 자동차의 보행자 탐지 같은 작업은 즉각적인 결정이 필요할 수 있다. 의료 애플리케이션은 정확도가 아주 높아야 해서 머신러닝 알고리즘 단독으로는 달성하기가 불가능하다. 



### 8.2 프로토타입에서 제품까지

복잡한 머신러닝 시스템 구축 시에 참고.

[Machine Learning: The High Interest Credit Card of Technical Debt]: https://research.google/pubs/pub43146/



### 8.3 제품 시스템 테스트

사전에 수집한 테스트 세트를 기초로 하여 알고리즘이 만든 예측을 평가하는 방법을 **오프라인 평가(Offline evaluation)** 라고 한다. 전체 시스템에 알고리즘이 적용된 이후에 평가하는 방법을 **온라인 테스트(Online test)** 혹은 **라이브 테스트(Live test)** 라고 한다. **A/B 테스트** 에서는 사용자 중 일부가 자신도 모르게 알고리즘 A를 사용한 웹사이트나 서비스를 이용하게 된다. 반면 나머지 사용자는 알고리즘 B에 노출된다. 두 그룹에 대해 적절한 성공 지표를 일정 기간 기록한다. 그런 다음 A와 B의 측정 값을 비교해서 두 방식 중 하나를 선택한다. A/B 테스트를 사용하면 실전에서 알고리즘을 평가해볼 수 있고 사용자들에게 모델이 노출됐을 때 예상치 못한 결과를 발견할 수도 있다. 보통 A가 새 모델이고 B는 기존 시스템이다. A/B 테스트 이외에도 **밴디트 알고리즘(Bandit algorithms)** 같이 온라인 테스트를 위한 정교한 방법들이 있다. 



### 8.4 나만의 추정기 만들기

Scikit-learn 인터페이스와 호환되는 추정기를 만드는 방법은 다음을 참조.

[Scikit-learn 인터페이스 형식의 추정기 만들기]: https://goo.gl/fkQWsN

가장 쉬운 방법은 BaseEstimator와 TransformerMixin을 상속해서 init, fit, transform 메소드를 구현하는 것이다.  분류와 회귀 모델을 만들 떄는 TransformerMixin 대신 ClassifierMixin이나 RegressorMixin을 상속하고, transfor 대신 predict를 구현한다. 

```python 
from sklearn.base import BaseEstimator, TransformerMixin

class MyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, first_paramter=1, second_parameter=2):
        # __init__ 메소드에 필요한 모든 매개변수를 나열.
        self.first_paramter = 1
        self.second_parameter = 2
        
    def fit(self, X, y=None):
        # fit 메소드는 X와 y 매개변수만을 갖는다.
        # 비지도 학습 모델이더라도 y 매개변수를 받도록 해야한다.
        
        # 모델 학습 시작
        print("모델 학습을 시작합니다")
        # 객체 자신인 self를 반환.
        return self
    
    def transform(self, X):
        # transform 메소드는 X 매개변수만을 받는다.
        
        # X를 변환합니다
        X_transformed = X + 1
        return X_transformed
```



### 8.5 더 배울 것들

##### 8.5.1 이론

- 헤이스티, 팁시라니, 프리드먼 - "The Elements of Statistical Learning"
- 스티븐 마스랜드 - "Machine Learning: An Algorithmic Persperctive"(Chapman and Hall/CRC)
- 크리스토퍼 비숍 - "Pattern Recognition and Machine Learning"(Springer)
- 캐빈 머피 - "Machine Learning: A Probabilistic Perspective"(MIT Press)



##### 8.5.2 다른 머신러닝 프레임워크와 패키지

- 확률 모델링과 추론에 더 관심이 있을 때 statsmodels를 고려해 볼 수 있다. 이 패키지는 조금 더 확률적 입장에서 인터페이스를 구축한 여러 선형 모델을 제공한다.

  [statsmodels]: http://statsmodels.sourceforge.net/

- R은 통계 분석을 위해 특별히 설계된 어어이고 훌륭한 시각화 기능과 많은 통계 모델 패키지를 활용할 수 있다. 

  [R-project]: https://www.r-project.org/

- 머신러닝에서 유명한 또 다른 패키지로 명령어 인터페이스를 제공하고 C++로 작성된 vowpal wabiit이 있다. vw는 특히 대량의 데이터셋과 스트리밍 데이터에 유용하다. 

  [vowpal wabbit]: https://github.com/VowpalWabbit/vowpal_wabbit/wiki

- 클러스터에 머신러닝 알고리즘을 분산해서 실행하는 기능으로 유명한 프레임워크로, spark 분산 컴퓨팅 환경에서 구축된 스칼라 라이브러리인 MLlib도 있다. 



##### 8.5.3 랭킹, 추천 시스템과 그 외 다른 알고리즘

- 랭킹 시스템 - 어떤 질문의 대답을 관련도 순으로 추출한다. 검색어를 입력하면 관련도순으로 순위가 매겨져 정렬된 목록을 얻는다. 

  매닝, 라가반, 쉬체 - "Introduction to Information Retrieval"

- 추천 시스템 - 사용자의 기호에 맞게 제안을 한다. 

- 시계열 예측



##### 8.5.4 확률 모델링, 추론, 확률적 프로그래밍

- [PyMC]: http://pymc-devs.github.io/pymc/

- [Stan]: https://mc-stan.org/



##### 8.5.5 신경망

- 이안 굿펠로, 조슈아 벤지오, 아론 쿠르빌 - "Deep Learning"(MIT Press, 2016)



##### 8.5.6 대규모 데이터셋으로 확장

- **외부 메모리 학습(Out-of-core)** 은 메모리에 저장할 수 없는 데이터로 학습하는 것을 말하며, 학습이 아나의 컴퓨터(하나의 프로세서)에서 수행된다. 데이터는 하드 디스크 같은 저장소나 네트워크로부터 한 번에 샘플 하나씩 또는 메모리 용량에 맞는 크기의 덩어리로 읽어 들인다. 데이터가 처리되면 데이터로부터 학습된 것이 반영되록 모델을 갱신한다. 그런 다름 이 데이터 덩어리는 버리고 다음 덩어리를 읽는다. 외부 메모리 학습에서는 컴퓨터 한 대에서 모든 데이터를 처리해야 하므로 큰 데이터셋을 처리하려면 시간이 오래 걸린다. 또 모든 머신러닝 알고리즘이 이 방식을 지원하지는 않는다. 
- **클러스터 병렬화(Parallelization overa cluster)**는 여러 컴퓨터로 데이터를 분산해서 각 컴퓨터가 해당하는 데이터를 처리하는 것이다. 일부 모델에서 처리 속도가 빨라지며 처리할 수 있는 데이터 크기는 클러스터 크기에 의해 제한된다. 가장 인기 있는 분산 컴퓨팅 플랫폼 중 하나는 하둡(Hadoop) 위에 구축된 spark이다. spark는 MLlib 패키지에 일부 머신러닝 기능을 포함하고 있다. 데이터가 이미 하둡 파일 시스템에 저장되어 있거나 데이터 전처리를 위해 spark를 쓰고 있다면 가장 쉽다. 인프라가 준비되어 있지 않을 경우 spark 클러스터를 구축하고 통합하려면 많은 노력이 필요하다. vw 패키지가 분산 기능을 조금 지원하므로 더 나은 대안이 될 수 있다. 



##### 8.5.7 실력 기르기

[캐글(Kaggle)]: https://www.kaggle.com/
[OpenML]: https://www.openml.org/


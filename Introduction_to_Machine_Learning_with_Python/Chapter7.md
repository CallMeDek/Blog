# 7. 텍스트 데이터 다루기

텍스트 데이터는 글자가 연결된 문자열로 표현된다. 텍스트 데이터의 길이는 서로 같은 경우가 거의 없다. 이런 특성은 이제까지 본 수치형 특성과 매우 다르므로 머신러닝 알고리즘에 적용하기 전에 전처리를 해야 한다(이제까지 본 데이터는 - 정량적인 연속형 특성, 고정된 목록에서 값이 정해지는 범주형 특성 - 데이터 포인트의 속성이 고정된, 즉 특성의 개수가 같았다. 텍스트 데이터는 내용의 길이가 달라지므로 전처리 과정이 없다면 샘플마다 특성의 수가 달라진다)



### 7.1 문자열 데이터 타입

- 범주형 데이터 - 범주형 데이터는 고정된 목록으로 구성된다. 드롭 다운 메뉴에서 "빨강", "녹색", "파랑" 등의 옵션이 있으면 데이터셋에 3개의 값 중 하나가 들어가면 당연히 범주형 변수로 인코딩 된다. 만약 오타가 있으면 데이터셋에는 같은 의미를 나타내는 값으로 합쳐야한다("뻘건"과 "빨강").
- 범주에 의미를 연결시킬 수 있는 임의 문자열 - "치과 오랜지색", "녹색과 빨강 줄무늬" 같이 사람들의 의견을 텍스트 필드로 받을 때의 응답은 범주에 의미를 연결시킬 수 있는 임의의 문자열이다. 이런 데이터는 범주형 변수로 인코딩하려면 가장 보편적인 값을 선택하든지, 애플리케이션에 맞게 이런 응답을 포용할 수 있는 점주를 정의하는 게 최선이다. 
- 구조화된 문자열 데이터 - 미리 정의된 범주에 속하지 않지만 직접 입력한 값들이 주소나 장소, 사람 이름, 날짜, 전화 번호, 식별 변호처럼 일정한 구조를 가지기도 한다. 이런 종류의 문자열은 분석하기 매우 어렵고, 처리 방법이 문맥이나 분야에 따라 매우 다르다.
- 텍스트 데이터 - 자유로운 형태의 절과 문장으로 구성되어 있다. 이런 데이터는 대부분 단어로 구성된 문장에 정보를 담고 있다. 텍스트 분석에서 데이터 셋을 말뭉치(Corpus), 하나의 텍스트를 의미하는 각 데이터 포인트를 문서(Document)라고 한다. 이런 용어는 텍스트 데이터를 주로 다루는 정보 검색(Information retrieval(IR))과 자연어 처리(Natural language processing(NLP))에서 유래했다. 



### 7.2 예제 애플리케이션: 영화 리뷰 감성 분석

다음은 스탠퍼드 대학교 연구원인 Andrew Mass가 IMDb(Internet Movie Database) 웹사이트에서 수집한 영화 리뷰 데이터셋이다. 이 데이터셋은 리뷰 텍스트와 "양성" 혹은 "음성"을 나타내는 레이블을 포함하고 있다. 1에서 10점의 평점중에서 7점 이상은 "양성", 4점 이하는 "음성"인 이진 분류 데이터셋이다(중간 포함x). 

train 폴더에 unsup 폴더는 레이블이 없는 데이터를 담고 있으므로 삭제한다.  하위 폴더가 레이블로 구분된 폴더 구조라면 scikit-learn의 load_files 함수를 사용해서 파일을 읽을 수 있다. 이 때 폴더의 알파벳 순서에 따라 0부터 부여된다. 

```python 
In:
!tar -xvzf ./introduction_to_ml_with_python/data/aclImdb_v1.tar.gz
!find ./aclImdb/ -type d
```

```python 
Out:
./aclImdb/
./aclImdb/train
./aclImdb/train/neg
./aclImdb/train/pos
./aclImdb/train/unsup
./aclImdb/test
./aclImdb/test/neg
./aclImdb/test/pos
```

```python 
!rm -r ./aclImdb/train/unsup
```

```python 
In:
from sklearn.datasets import load_files
import pprint

reviews_train = load_files("./aclImdb/train/")

text_train, y_train = reviews_train.data, reviews_train.target
print(f"text_train의 타입: {type(text_train)}")
print(f"text_train의 길이: {len(text_train)}")
print("text_train[6]:")
pprint.pprint(text_train[6], width=140, compact=True)
```

```python 
Out:
text_train의 타입: <class 'list'>
text_train의 길이: 25000
text_train[6]:
(b'This movie has a special way of telling the story, at first i found it rather odd as it jumped through time and I had no idea whats happ'
 b'ening.<br /><br />Anyway the story line was although simple, but still very real and touching. You met someone the first time, you fell '
 b"in love completely, but broke up at last and promoted a deadly agony. Who hasn't go through this? but we will never forget this kind of "
 b'pain in our life. <br /><br />I would say i am rather touched as two actor has shown great performance in showing the love between the c'
 b'haracters. I just wish that the story could be a happy ending.')
```

```python 
In:
import numpy as np

text_train = [doc.replace(b"<br />", b" ") for doc in text_train]
print(f"클래스별 샘플 수 (훈련 데이터): {np.bincount(y_train)}")
```

```python 
Out:
클래스별 샘플 수 (훈련 데이터): [12500 12500]
```

text \_train의 항목의 타입은 파이썬 버전에 따라 다른데 파이썬 3에서는 문자열 데이터의 바이너리 인코딩인 bytes 타입이다. 파이썬 2에서는 text_train의 타입이 문자열이다. 아스키(ASCII) 코드에 대응하는 파이썬 2의 문자열(str)이 파이썬 3에서 사라지고, 파이썬 2의 unicode 문자열이 파이썬 3의 기본 str이 되었다. 그리고 파이썬 3에서는 str의 바이너리 표현인 bytes가 추가되었다. 파이썬 3에서 문자열 "한글"으 ㅣ길이는 2지만, "한글".encode('utf8')과 같이 bytes 타입으로 변환하면 길이가 6이 된다. load_files 함수는 open 함수에 "rb" 옵션을 주어 파일을 바이너리로 읽기 때문에 파이썬 3에서 text_train의 타입이 bytes가 된다. 

```python 
In:
reviews_test = load_files("./aclImdb/test/")
text_test, y_test = reviews_test.data, reviews_test.target
print(f"테스트 데이터의 문서 수: {len(text_test)}")
print(f"클래스별 샘플 수 (테스트 데이터): {np.bincount(y_test)}")
text_test = [doc.replace(b"<br />", b" ") for doc in text_test]
```

```python 
Out:
테스트 데이터의 문서 수: 25000
클래스별 샘플 수 (테스트 데이터): [12500 12500]
```

텍스트 데이터는 머신러닝 모델이 다룰 수 있는 형태가 아니다. 따라서 텍스트의 문자열 표현을 머신러닝 알고리즘에 적용할 수 있도록 수치 표현으로 바꿔줘야 한다. 
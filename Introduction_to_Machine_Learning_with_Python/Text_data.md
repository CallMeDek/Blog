# 텍스트 데이터 다루기

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

text \_train의 항목의 타입은 파이썬 버전에 따라 다른데 파이썬 3에서는 문자열 데이터의 바이너리 인코딩인 bytes 타입이다. 파이썬 2에서는 text_train의 타입이 문자열이다. 아스키(ASCII) 코드에 대응하는 파이썬 2의 문자열(str)이 파이썬 3에서 사라지고, 파이썬 2의 unicode 문자열이 파이썬 3의 기본 str이 되었다. 그리고 파이썬 3에서는 str의 바이너리 표현인 bytes가 추가되었다. 파이썬 3에서 문자열 "한글"의 길이는 2지만, "한글".encode('utf8')과 같이 bytes 타입으로 변환하면 길이가 6이 된다. load_files 함수는 open 함수에 "rb" 옵션을 주어 파일을 바이너리로 읽기 때문에 파이썬 3에서 text_train의 타입이 bytes가 된다. 

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



### 7.3 텍스트 데이터를 BOW로 표현하기

BOW(Bag of words)를 쓰면 장, 문단, 문장, 서식 같은 입력 테스트의 구조 대부분을 잃고, 각 단어가 이 말뭉치에 있는 텍스트에 얼마나 많이 나타나는지만 헤아린다. 구조와 상관 없이 단어의 출현 횟수만 세기 떄문에 텍스트를 담는 '가방(bag)'을 생각할 수 있다. 

BOW 표현의 3단계는 다음과 같다.

1. **토큰화(Tokenization)** - 각 문서를 문서에 포함된 단어(토큰)으로 나눈다. 예를 들어 공백이나 구두점을 기준으로 분리한다.
2. **어휘 사전 구축** - 모든 문서에 나타난 모든 단어의 어휘를 모으고 번호를 매긴다(알파벳 순서).
3. **인코딩** - 어휘 사전의 단어가 문서마다 몇 번이나 나타나는지를 헤아린다. 

![](./Figure/7_3_1.JPG)

출력은 각 문서에서 나타난 단어의 횟수가 담긴 하나의 벡터이다. 이를 위해 사전에 있는 각 단어가 문서마다 얼마나 자주 나타나는지 세야 한다. 즉, 이 수치 표현은 전체 데이터셋에서 고유한 각 단어를 특성으로 가진다. 원본 문자열에 있는 단어의 순서는 BOW 특성 표현에서는 완전히 무시된다. 



##### 7.3.1 샘플 데이터에 BOW 적용하기

BOW 표현은 CountVectorizer에 변환기 인터페이스로 구현되어 있다. 

```python 
In:
from sklearn.feature_extraction.text import CountVectorizer

bards_words = ["The fool doth think he is wise,",
               "but the wise man knows himself to be a fool"]

vect = CountVectorizer()
vect.fit(bards_words)
print(f"어휘 사전의 크기: {len(vect.vocabulary_)}")
print(f"어휘 사전의 내용: {vect.vocabulary_}")
```

```python 
Out:
어휘 사전의 크기: 13
어휘 사전의 내용: {'the': 9, 'fool': 3, 'doth': 2, 'think': 10, 'he': 4, 'is': 6, 'wise': 12, 'but': 1, 'man': 8, 'knows': 7, 'himself': 5, 'to': 11, 'be': 0}
```

훈련 데이터에 대해 BOW 표현을 만들려면 transform 메소드를 호출한다.

```python 
In:
bag_of_words = vect.transform(bards_words)
print(f"BOW: {repr(bag_of_words)}")
```

```python 
Out:
BOW: <2x13 sparse matrix of type '<class 'numpy.int64'>'
	with 16 stored elements in Compressed Sparse Row format>
```

BOW 표현은 0이 아닌 값만 저장하는 SciPy 희소행렬로 저장되어  있다. 각각의 행은 하나의 데이터 포인트를 나타내고, 각 특성은 어휘 사전에 있는 각 단어에 대응한다. 대부분의 문서는 어휘 사전에 있는 단어 중 일부만 포함하므로, 즉 특성 배열의 대부분의 원소가 0이라서 희소 행렬을 사용한다. 어휘 사전에 있는 전체 영어 단어 수에 비해 한 편의 영화 리뷰에 들어 있는 단어의 수가 적기 때문에 값이 0인 원소를 모두 저장하는 것은 메모리 낭비이다. 희소 행렬의 실제 내용을 보려면 toarray 메소드를 사용하여(0인 원소도 모두 저장되도록) 밀집된 NumPy 배열로 바꿔야 한다.

```python 
In:
print(f"BOW의 밀집 표현:\n{bag_of_words.toarray()}")
```

```python 
Out:
BOW의 밀집 표현:
[[0 0 1 1 1 0 1 0 0 1 1 0 1]
 [1 1 0 1 0 1 0 1 1 1 0 1 1]]
```

각 행은 앞에서의 문자열 리스트의 원소 ["The fool doth think he is wise,",  "but the wise man knows himself to be a fool"] 이고 어휘 사전을 만들 때 알파벳 순서대로 단어를 정렬하면 BOW는 "be", "but", "doth" 등의 순으로 나타난다(결과의 열). 결과의 각 원소는 이 열이 각 문자열에 대해서 얼만큼 나타났는지를 의미한다.  



##### 7.3.2 영화 리뷰에 대한 BOW

이전 영화 리뷰에 대한 감성 분석(IMDb)을 적용해보면 다음과 같다.

```python 
In:
vect = CountVectorizer().fit(text_train)
X_train = vect.transform(text_train)
print(f"X_train:\n{repr(X_train)}")
```

```python 
Out:
X_train:
<25000x74849 sparse matrix of type '<class 'numpy.int64'>'
	with 3431196 stored elements in Compressed Sparse Row format>
```

훈련 데이터의 BOW 표현은 25,000 X 74,849의 크기를 가지고 있는데, 이 어휘 사전은 74,849의 단어로 이루어진 SciPy 희소 행렬로 저장되어 있다. 

```python 
In:
feature_names = vect.get_feature_names()
print(f"특성 개수: {len(feature_names)}")
print(f"처음 20개 특성:\n{feature_names[:20]}")
print(f"20,010에서 20,030까지 특성:\n{feature_names[20010:20030]}")
print(f"매 2,000번째 특성:\n{feature_names[::2000]}")
```

```python 
Out:
특성 개수: 74849
처음 20개 특성:
['00', '000', '0000000000001', '00001', '00015', '000s', '001', '003830', '006', '007', '0079', '0080', '0083', '0093638', '00am', '00pm', '00s', '01', '01pm', '02']
20,010에서 20,030까지 특성:
['dratted', 'draub', 'draught', 'draughts', 'draughtswoman', 'draw', 'drawback', 'drawbacks', 'drawer', 'drawers', 'drawing', 'drawings', 'drawl', 'drawled', 'drawling', 'drawn', 'draws', 'draza', 'dre', 'drea']
매 2,000번째 특성:
['00', 'aesir', 'aquarian', 'barking', 'blustering', 'bête', 'chicanery', 'condensing', 'cunning', 'detox', 'draper', 'enshrined', 'favorit', 'freezer', 'goldman', 'hasan', 'huitieme', 'intelligible', 'kantrowitz', 'lawful', 'maars', 'megalunged', 'mostey', 'norrland', 'padilla', 'pincher', 'promisingly', 'receptionist', 'rivals', 'schnaas', 'shunning', 'sparse', 'subset', 'temptations', 'treatises', 'unproven', 'walkman', 'xylophonist']
```

영화 리뷰라는 것을 고려하면 숫자들 대부분은 그 자체로 의미가 있어 보이지 않는다. "dra"로 시작하는 영어 단어의 목록을 보면 "draught", "drawback", "drawer" 모두 단수와 복수형이 서로 다른 단어로 어휘 사전에 포함되어 있는데 이런 단어들은 의미가 매우 비슷하므로 다른 특성으로 간주하여 개별적으로 기록하는 것이 바람직 하지 않다. 

이런 희소 행렬의 고차원 데이터셋에서는 LogisticRegression 같은 선형 모델의 성능이 뛰어나다. 이 알고리즘의 분류기를 만들어 교차 검증 성능 수치를 확인하면 다음과 같다(CountVectorizer의 min_df, max_df 같은  매개변수의 기본 값을 바꾸면 교차 검증의 결과에 영향을 주므로 Pipeline을 사용하는 것이 좋다).

```python 
In:
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)
print(f"교차 검증 평균 점수: {np.mean(scores):.2f}")
```

```python 
Out:
교차 검증 평균 점수: 0.88
```

LogisticRegression의 규제 매개변수 C를 그리드 서치를 사용해 조정해보면 다음과 같다.

```python 
In:
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [10**i for i in range(-3, 3)]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print(f"최상의 교차 검증 점수: {grid.best_score_:.2f}")
print(f"최적의 매개변수: ", grid.best_params_)
```

```python 
Out:
최상의 교차 검증 점수: 0.88
최적의 매개변수:  {'C': 0.1}
```

```python 
In:
X_test = vect.transform(text_test)
print(f"테스트 덤수: {grid.score(X_test, y_test):.2f}")
```

```python 
Out:
테스트 덤수: 0.88
```

CountVectorizer는 정규 표현식을 사용해 토큰을 추출한다. 기본적으로 사용하는 정규표현식은 "\b\w\w+\b"이다. 이 식으로 경계(\b)가 구분되고 적어도 둘 이상의 문자나 숫자(\w)가 연속된 단어를 찾는다. 한 글자로 된 단어는 찾지 않으며, "doesn't" 같은 축양형이나 "bit.ly" 같은 단어는 분리되고, "h8ter"는 한단어로 매칭된다. CountVectorizer는 모든 단어를 소문자로 바꾸므로 "soon", "Soon", "sOon"이 모두 가은 토큰(즉 특성)이 된다. 이런 특성은 대체로 잘 작동하나 의미 없는 특성(숫자 같은)을 많이 생성한다. 이를 줄이는 방법은 적어도 두 개의 문서(또는 다섯 개의 문서 등)에 나타난 토큰만을 사용하는 것이다. 하나의 문서에서만 나타난 토큰은 테스트 세트에 나타날 가능성이 적으므로 큰 도움이 되지 않는다. min_df 매개변수로 토큰이 나타날 최소 문서 개수를 지정할 수 있다. 

```python 
In:
vect = CountVectorizer(min_df=5).fit(text_train)
X_train = vect.transform(text_train)
print(f"min_df로 제한한 X_train: {repr(X_train)}")
```

```python 
Out:
min_df로 제한한 X_train: <25000x27271 sparse matrix of type '<class 'numpy.int64'>'
	with 3354014 stored elements in Compressed Sparse Row format>
```

토큰이 적어도 다섯 번 이상 나타나야 하므로,  결과에서 보면 특성의 수가 원래의 74,849에서 1/3 정도로 줄어든 것을 확인할 수 있다.

```python 
In:
feature_names = vect.get_feature_names()

print(f"처음 50개 특성:\n{feature_names[:50]}")
print(f"20,010에서 20,030까지 특성:\n{feature_names[20010:20030]}")
print(f"매 700번째 특성:\n{feature_names[::700]}")
```

```python 
Out:
처음 50개 특성:
['00', '000', '007', '00s', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '100', '1000', '100th', '101', '102', '103', '104', '105', '107', '108', '10s', '10th', '11', '110', '112', '116', '117', '11th', '12', '120', '12th', '13', '135', '13th', '14', '140', '14th', '15', '150', '15th', '16', '160', '1600', '16mm', '16s', '16th']
20,010에서 20,030까지 특성:
['repentance', 'repercussions', 'repertoire', 'repetition', 'repetitions', 'repetitious', 'repetitive', 'rephrase', 'replace', 'replaced', 'replacement', 'replaces', 'replacing', 'replay', 'replayable', 'replayed', 'replaying', 'replays', 'replete', 'replica']
매 700번째 특성:
['00', 'affections', 'appropriately', 'barbra', 'blurbs', 'butchered', 'cheese', 'commitment', 'courts', 'deconstructed', 'disgraceful', 'dvds', 'eschews', 'fell', 'freezer', 'goriest', 'hauser', 'hungary', 'insinuate', 'juggle', 'leering', 'maelstrom', 'messiah', 'music', 'occasional', 'parking', 'pleasantville', 'pronunciation', 'recipient', 'reviews', 'sas', 'shea', 'sneers', 'steiger', 'swastika', 'thrusting', 'tvs', 'vampyre', 'westerns']
```

```python 
In:
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print(f"최상의 교차 검증 점수: {grid.best_score_:.2f}")
```

```python 
Out:
최상의 교차 검증 점수: 0.89
```

그리드 서치의 교차 검증 점수가 크게 개선되지는 않았지만, 특성의 개수가 줄어서 처리 속도가 빨리지고, 불필요한 특성이 없어져 모델을 이해하기가 더 쉬워졌다. 

(CountVectorizer의 transform 메소드를 훈련 데이터에 없던 단어가 포함된 문서에 적용하면, 어휘 사전에 없기 때문에 그 단어를 무시한다. 훈련 데이터에 없는 단어에 대해 무언가 학습한다는 것이  불가능하므로, 분류 작업에서 보통 문제가 되지는 않는다.)



### 7.4 불용어

의미 없는 단어를 제거하는 또 다른 방법은 너무 빈번하여 유용하지 않은 단어를 제외하는 것이다. 두 가지 방식이 있는 언어별 불용어(Stopword) 목록을 사용하는 것과 max_df 옵션을 사용하여 너무 자주 나타나는 단어를 제외하는 것이다. 

```python 
In:
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

print(f"불용어 개수: {len(ENGLISH_STOP_WORDS)}")
print(f"매 10번째 불용어:\n{list(ENGLISH_STOP_WORDS)[::10]}")
```

```python 
Out:
불용어 개수: 318
매 10번째 불용어:
['twelve', 'thus', 'cannot', 'latterly', 'whom', 'few', 'whenever', 'show', 'they', 'throughout', 'was', 'you', 'are', 'ltd', 'two', 'what', 'eleven', 'together', 'in', 'towards', 'one', 'since', 'and', 'none', 'your', 'meanwhile', 'being', 'whose', 'fifty', 'ever', 'becoming', 'as']
```

```python 
In:
vect = CountVectorizer(min_df=5, stop_words="english").fit(text_train)
X_train = vect.transform(text_train)
print(f"불용어가 제거된 X_train:\n{repr(X_train)}")
```

```python 
Out:
불용어가 제거된 X_train:
<25000x26966 sparse matrix of type '<class 'numpy.int64'>'
	with 2149958 stored elements in Compressed Sparse Row format>
```

원래의 특성 27,271에서 305개가 줄어든 26,966개가 남아있음을 확인할 수 있다.

```python 
In:
gird = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print(f"최상의 교차 검증 점수: {grid.best_score_:.2f}")
```

```python 
Out:
최상의 교차 검증 점수: 0.88
```

고정된 불용어 목록은 모델이 데이터셋만 보고 불용어를 골라내기 어려운 작은 데이터셋에서나 도움이 될 수 있다. 다른 방식으로는 CountVectorizer의 max_df 옵션을 지정하여 자주 나타나는 단어를 제거하는 방법이 있다. 



### 7.5 tf-idf로 데이터 스케일 변경하기

중요하지 않아 보이는 특성을 제외하는 대신, 얼마나 의미 있는 특성인지를 계산해서 스케일을 조정하는 방식 중 하나가 **tf-idf(Term frequency-inverse document frequency)** 이다. tf-idf는 말뭉치의 다른 문서보다 특정 문서에 자주 나타나는 단어에 높은 가중치를 주는 방법이다. 한 단어가 특정 문서에 자주 나타나고 다른 여러 문서에서는 그렇지 않다면, 그 문서의 내용을 아주 잘 설명하는 단어라고 볼 수 있다. TfidfTransformer는 CountVectorizer가 만든 희소 행렬을 입력받아 변환한다. TfidfVectorizer는 텍스트 데이터를 입력 받아 BOW 특성 추출과 tf-idf 변환을 수행한다(TfidfVectorizer는 CountVectorizer의 서브 클래스로 CountVectorizer를 이용해 BOW를 만들고 TfidfTransformer를 사용해 tf-idf 변환을 한다).

 문서 d에 있는 단어 w에 대한 tf-df 점수는 TfidfTransformer와 TfidfVectorizer에 다음과 같이 저장되어 있다. 단 smooth_idf 매개변수가 기본값(True)일 때 아래와 같은 공식이 사용된다. 

![](./Figure/7_5_1.JPG)

N은 훈련 세트에 있는 문서의 개수고, N_w는 단어 w가 나타난 훈련 세트 문서의 개수이며, tf(단어 빈도수)는 간어 w가 대상 문서 d(변환 또는 인코딩 하려는 문서)에 나타난 횟수이다. 로그 안의 분모와 분자에 1을 더해 모든 단어가 포함된 가상의 문서가 있는 거 같은 효과를 내거 분모가 0이 되는 것을 막아준다. 또 모든 문서에 포함된 단어가 있으면 로그 값이 0이 되므로, 전체 tf-idf 값이 0이 되는 것을 막기 위해 idf 공식 마지막에 1을 더한다. 

smooth_idf 매개변수가 False일 때는 위의 공식에 분모와 분자에 1을 더하지 않는 표준 idf 공식이 사용된다.

![](./Figure/7_5_2.JPG)

두 파이썬 클래스 모두 tf-idf 계산을 한 후에 L2 정규화(L2 normalization)를 적용한다. 즉, 유클리디안 놈(Euclidean norm)이 1이 되도록 각 문서 벡터의 스케일을 바꾼다. 이렇게 스케일이 바뀐 벡터는 문서의 길이(단어의 수)에 영향을 받지 않는다(벡터의 원소를 유클리디안 놈으로 나누면 L2 놈이 1인 단위 벡터가 된다. norm 매개변수의 기본 값은 L2 정규화를 의미하는 "l2"이다). 

```python 
In:
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(TfidfVectorizer(min_df=5), LogisticRegression())
param_grid = {'logisticregression__C': [10**i for i in range(-3, 3)]}

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(text_train, y_train)
print(f"최상의 교차 검증 점수: {grid.best_score_:.2f}")
```

```python 
Out:
최상의 교차 검증 점수: 0.89
```

여기서는 tf-idf가 성능에 큰 영향을 주지 못했다. tf-idf는 어떤 단어가 가장 중요한지도 알려준다. tf-idf 변환은 문서를 구별하는 단어를 찾는 방법이지만 완전히 비지도 학습이다. 그래서 우리의 관심사인 '긍정적인 리뷰'와 '부정적인 리뷰' 레이블과 꼭 관계있지 않다. 

```python 
In:
vectorizer = grid.best_estimator_.named_steps['tfidfvectorizer']
X_train = vectorizer.transform(text_train)
#특성 별로 가장 큰 값을 찾는다.
max_value = X_train.max(axis=0).toarray().ravel()
sorted_by_tfidf = max_value.argsort()
#특성 이름을 구한다.
feature_names = np.array(vectorizer.get_feature_names())

print(f"tfidf가 가장 낮은 특성:\n{feature_names[sorted_by_tfidf[:20]]}")
print(f"tfidf가 가장 높은 특성:\n{feature_names[sorted_by_tfidf[-20:]]}")
```

```python 
Out:
tfidf가 가장 낮은 특성:
['suplexes' 'gauche' 'hypocrites' 'oncoming' 'songwriting' 'galadriel'
 'emerald' 'mclaughlin' 'sylvain' 'oversee' 'cataclysmic' 'pressuring'
 'uphold' 'thieving' 'inconsiderate' 'ware' 'denim' 'reverting' 'booed'
 'spacious']
tfidf가 가장 높은 특성:
['gadget' 'sucks' 'zatoichi' 'demons' 'lennon' 'bye' 'dev' 'weller'
 'sasquatch' 'botched' 'xica' 'darkman' 'woo' 'casper' 'doodlebops'
 'smallville' 'wei' 'scanners' 'steve' 'pokemon']
```

tf-idf가 낮은 특성은 전체 문서에 걸쳐 매우 많이 나타나거나, 조금씩만 사용되거나, 매우 긴 문서에서만 사용된다(전체 문서에 걸쳐 많이 나타나면 idf 값이 1에 가깝게 되고, 조금씩 사용괴거나 매우 긴 문서에서만 사용되면 L2 정규화 때문에 tf-idf 값이 작아진다). 

```python 
In:
sorted_by_idf = np.argsort(vectorizer.idf_)
print(f"idf가 가장 낮은 특성:\n{feature_names[sorted_by_tfidf[:100]]}")
```

```python 
Out:
idf가 가장 낮은 특성:
['suplexes' 'gauche' 'hypocrites' 'oncoming' 'songwriting' 'galadriel'
 'emerald' 'mclaughlin' 'sylvain' 'oversee' 'cataclysmic' 'pressuring'
 'uphold' 'thieving' 'inconsiderate' 'ware' 'denim' 'reverting' 'booed'
 'spacious' 'gliding' 'orientated' 'attained' 'coaxing' 'auspicious'
 'sharpshooter' 'basking' 'southampton' 'manically' 'livelier' 'vertical'
 'backfire' 'negotiate' 'slyly' 'roughing' 'patrolman' 'distort'
 'temperamental' 'immunity' 'inciting' 'ralli' 'swells' 'melchior'
 'stifled' 'confessing' 'subtracted' 'nyree' 'deportation' 'alloy'
 'unspectacular' 'usefulness' 'mays' 'annihilated' 'scuffle' 'unsavoury'
 'ancestral' 'labourer' 'administered' 'shoehorn' 'père' 'plunder'
 'plunging' 'empowered' '800' 'perched' 'nursed' 'decayed' 'dubs'
 'assassinates' 'leverage' 'mistreatment' 'bibles' 'confiscated' 'collier'
 'cineplex' 'lifeboats' 'jurisdiction' 'radiating' '1890s' 'confidante'
 'negotiating' 'tahoe' 'tanovic' 'levene' 'malignant' 'unlocked' 'trots'
 'extravaganzas' 'lenin' 'quip' 'outcry' 'foetus' 'sagas' 'squash'
 'disclose' 'peculiarly' 'tellingly' 'brainchild' 'britons'
 'dimensionality']
```



### 7.6 모델 계수 조사

아래의 그래프는 로지스틱 회귀의 가장 큰 계수 25개와 가장 작은 계수 25개를 보여준다. 막대의 크기는 계수의 크기이다. 

```python 
mglearn.tools.visualize_coefficients(
    grid.best_estimator_.named_steps['logisticregression'].coef_[0],
    feature_names, n_top_features=40)
```

![](./Figure/7_6_1.JPG)

왼쪽의 음수 계수는 모델에서 부정적인 리뷰를 의미하는 단어에 속하고, 오른쪽 양수 계수는 긍정적인 리뷰의 단어에 해당하는 것을 확인할 수 있다.



### 7.7 여러 단어로 만든 BOW(n-그램)

BOW 표현 방식은 단어의 순서가 완전히 무시된다는 큰 단점이 있다. 그렇기 때문에 의미가 완전히 반대인 두 문자열 "it's bad, not good at all"과 "it's good, not bad at all"이 완전히 동일하게 변환된다. BOW 표현 방식을 사용할 때 문맥을 고려하는 방법이 있는데 토큰 하나의 횟수만 고려하지 않고 옆에 있는 두 세개의 토큰을 함께 고려하는 방식이다.  토큰 2개를 **바이그램(Bigram)**, 3개를 **트라이그램(Trigram)** 이라고 하며일반적으로 연속된 토큰을 **n-그램(n-gram)** 이라고 한다. CountVectorizer와 TfidfVectorizer는 ngram_range 매개변수에 특성으로 고려할 토큰의 범위를 지정할 수 있다. ngram_range 매개변수의 입력 값은 튜플이며 연속된 토큰의 최소 길이와 최대 길이이다.

```python 
In:
print(f"bards_words:\n{bards_words}")
```

```python 
Out:
bards_words:
['The fool doth think he is wise,', 'but the wise man knows himself to be a fool']
```



토큰 1개는 **유니 그램(Unigram)** 이라고 한다.

```python 
In:
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(ngram_range=(1, 1)).fit(bards_words)
print(f"어휘 사전 크기: {len(cv.vocabulary_)}")
print(f"어휘 사전:\n{cv.get_feature_names()}")
```

```python 
Out:
어휘 사전 크기: 13
어휘 사전:
['be', 'but', 'doth', 'fool', 'he', 'himself', 'is', 'knows', 'man', 'the', 'think', 'to', 'wise']
```



토큰 두 개가 연속된 바이그램만 만들려면 ngram_range에 (2, 2)를 지정한다.

```python 
In:
cv = CountVectorizer(ngram_range=(2, 2)).fit(bards_words)
print(f"어휘 사전 크기:{len(cv.vocabulary_)}")
print(f"어휘 사전:\n{cv.get_feature_names()}")
```

```python 
Out:
어휘 사전 크기:14
어휘 사전:
['be fool', 'but the', 'doth think', 'fool doth', 'he is', 'himself to', 'is wise', 'knows himself', 'man knows', 'the fool', 'the wise', 'think he', 'to be', 'wise man']
```



연속된 토큰의 수가 커지면 보통 특성이 더 구체적이고 많이 만들어진다. bard_words에 있는 두 문장 사이에는 공통된 바이그램이 없다. 

```python 
In:
print(f"변환된 데이터 (밀집 배열):\n{cv.transform(bards_words).toarray()}")
```

```python 
Out:
변환된 데이터 (밀집 배열):
[[0 0 1 1 1 0 1 0 0 1 0 1 0 0]
 [1 1 0 0 0 1 0 1 1 0 1 0 1 1]]
```

단어 하나가 큰 의미를 가진 경우가 많으므로 대부분의 애플리케이션에서 토큰의 최소 길이는 1이다. 많은 경우에 바이그램을 추가하면 도움이 된다. 더 길게 5-그램까지는 도움이 되지만 특성의 갯수가 많아질수 있으며 구체적인 특성이 많아지기 때문에 과대적합할 가능이 있다. 이론상 바이그램의 수는 유니그램 수의 제곱이 되고, 트라이그램의 수는 유니그램의 세제곱이 되므로 특성의 갯수가 많이 늘어난다. (영어) 언어의 구조상 실제로 데이터에 나타나는 높은 n-그램의 횟수가 많기는 하지만 이보다는 훨씬 적다.

```python 
In:
cv = CountVectorizer(ngram_range=(1, 3)).fit(bards_words)
print(f"어휘 사전 크기: {len(cv.vocabulary_)}")
print(f"어휘 사전:\n{cv.get_feature_names()}")
```

```python 
Out:
어휘 사전 크기: 39
어휘 사전:
['be', 'be fool', 'but', 'but the', 'but the wise', 'doth', 'doth think', 'doth think he', 'fool', 'fool doth', 'fool doth think', 'he', 'he is', 'he is wise', 'himself', 'himself to', 'himself to be', 'is', 'is wise', 'knows', 'knows himself', 'knows himself to', 'man', 'man knows', 'man knows himself', 'the', 'the fool', 'the fool doth', 'the wise', 'the wise man', 'think', 'think he', 'think he is', 'to', 'to be', 'to be fool', 'wise', 'wise man', 'wise man knows']
```



다음은 IMDb 영화 리뷰 데이터에 TfidfVectorizer를 적용하고 그리드 서치로 최적의 n-그램 범위를 찾는 예제이다. 

```python 
In:
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

pipe = make_pipeline(TfidfVectorizer(min_df=5), LogisticRegression())
param_grid = {"logisticregression__C": [10**i for i in range(-3, 3)],
              "tfidfvectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)]}

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(text_train, y_train)
print(f"최상의 교차 검증 점수: {grid.best_score_:.2f}")
print(f"최적의 매개변수:\n{grid.best_params_}")
```

```python 
Out:
최상의 교차 검증 점수: 0.91
최적의 매개변수:
{'logisticregression__C': 100, 'tfidfvectorizer__ngram_range': (1, 3)}
```

```python 
#그리드 서치에서 테스트 점수를 추출한다.
scores = grid.cv_results_['mean_test_score'].reshape(-1, 3).T
heatmap = mglearn.tools.heatmap(
    scores, xlabel="C", ylabel="ngram_range", cmap="viridis", fmt="%.3f",
    xticklabels=param_grid['logisticregression__C'],
    yticklabels=param_grid['tfidfvectorizer__ngram_range'])
plt.colorbar(heatmap)
```

![](./Figure/7_7_1.JPG)

```python 
vect = grid.best_estimator_.named_steps['tfidfvectorizer']
feature_names = np.array(vect.get_feature_names())
coef = grid.best_estimator_.named_steps['logisticregression'].coef_
mglearn.tools.visualize_coefficients(coef, feature_names, n_top_features=40)
```

![](./Figure/7_7_2.JPG)

```python 
#트라이그램 특성을 찾음.
mask = np.array([len(feature.split(" ")) for feature in feature_names]) == 3
#트라이그램 특성만 그래프로 나타냄.
mglearn.tools.visualize_coefficients(coef.ravel()[mask],
                                     feature_names[mask], n_top_features=40)
```

![](./Figure/7_7_3.JPG)



### 7.8 고급 토큰화, 어간 추출, 표제어 추출

"drawback", "drawbacks", "drawer" 등 BOW 모델에서 의미가 매우 가까워서 이를 구분하면 과대적합되기 쉽고, 모델이 훈련 데이터를 완전하게 활용하지 못하는 경우가 있다. 이 문제를 해결하려면 각 단어를 그 단어의 어간(Stem)을 표현해서 같은 어간을 가진 모든 단어를 구분해야(또는 합쳐야) 한다. 일일이 어미를 찾아 제외하는 규칙 기반 방식을 **어간 추출(Stemming)** 이라고 한다. 대신 알려진 단어의 형태 사전(명시적이고 사람이 구축한 시스템)을 사용하고 문장에서 단어의 역할을 고려하는 처리 방식을 **표제어 추출(Lemmatization)** 이라고 하며 단어의 표준 형태를 표제어라고 한다. 두 처리 방식은 단어의 일반 형태를 추출하는 **정규화(Normalization)** 의 한 형태로 볼 수 있다. 

```python 
In:
import spacy
import nltk

en_nlp = spacy.load('en')
stemmer = nltk.stem.PorterStemmer()

def compare_normalization(doc):
  doc_spacy = en_nlp(doc)
  print("표제어:")
  print([token.lemma_ for token in doc_spacy])
  print("어간:")
  print([stemmer.stem(token.norm_.lower()) for token in doc_spacy])

compare_normalization(u"Our meeting today was worse than yesterday, I'm scared of meeting the clients tomorrow.")
```

```python 
Out:
표제어:
['-PRON-', 'meeting', 'today', 'be', 'bad', 'than', 'yesterday', ',', '-PRON-', 'be', 'scared', 'of', 'meet', 'the', 'client', 'tomorrow', '.']
어간:
['our', 'meet', 'today', 'wa', 'wors', 'than', 'yesterday', ',', 'i', 'am', 'scare', 'of', 'meet', 'the', 'client', 'tomorrow', '.']
```

어간 추출이 단어에서 어간만 남겨놓고 제거하므로 "was"는 "wa"가 되지만, 표제어 추출은 올바른 동사형인 "be"를 추출했다. 어간 추출이 두 번의 "meeting"을 "meet"로 바꿨지만 표제어 추출은 첫 번쨰 "meeting"은 명사로 인식해서 그대로 두고 두 번째는 동사로 인식 "meet"로 바꿨다. 일반적으로 표제어 추출은 어간 추출보다 훨씬 복잡한 처리를 거친다. 하지만 머신러닝을 위해 토큰 정규화를 할 때는 어간 추출보다 좋은 결과를 낸다고 알려져 있다. 

```python 
In:
import re

regexp = re.compile('(?u)\\b\\w\\w+\\b')

en_nlp = spacy.load('en')
old_tokenizer = en_nlp.tokenizer
en_nlp.tokenizer = lambda string: old_tokenizer.tokens_from_list(regexp.findall(string))

def custom_tokenizer(document):
  doc_spacy = en_nlp(document)
  return [token.lemma_ for token in doc_spacy]

lemma_vect = CountVectorizer(tokenizer=custom_tokenizer, min_df=5)

X_train_lemma = lemma_vect.fit_transform(text_train)
print(f"X_train_lemma.shape: {X_train_lemma.shape}")

vect = CountVectorizer(min_df=5).fit(text_train)
X_train = vect.transform(text_train)
print(f"X_train.shape: {X_train.shape}")
```

```python 
Out:
X_train_lemma.shape: (25000, 21571)
X_train.shape: (25000, 27271)
```

결과를 보면 알 수 있듯이, 표제어 추출은 특성 개수를 (표준 CountVectorizer에서 얻은) 27,271개에서 21,571개로 줄여준다. 표제어 추출은 일부 특성들을 합치기 때문에 일종의 규제로 볼 수 있다. 그래서 데이터셋이 작을 때도 표제어 추출이 성능을 높여줄 수 있다. 

다음은 표제어가 어떻게 효과가 있는 지 보기 위해 StratifiedShuffleSplit을 사용해 훈련 세트의 1%만 훈련 폴드로 하고, 나머지는 테스트 폴드로 하여 교차 검증을 수행하는 예제이다.

```python 
In:
from sklearn.model_selection import StratifiedShuffledSplit

param_grid = {'C': [10**i for i in range(-3, 2)]}
cv = StratifiedShuffledSplit(n_split=5, test_size=0.99, train_size=0.01, random_state=0)
grid = GrodSearchCV(LogisticRegression(), param_grid, cv=cv)
grid.fit(X_train, y_train)
print(f"최상의 교차 검증 점수: (기본 CountVectorizer): {grid.best_score_:.3f}")
grid.fit(X_train_lemma, y_train)
print(f"최상의 교차 검증 점수: (표제어): {grid.best_score_:.3f}")
```

```python 
Out:
최상의 교차 검증 점수: (기본 CountVectorizer): 0.719    
최상의 교차 검증 점수: (표제어): 0.731
```

다른 특성 추출 기법들과 마찬가지로 데이터셋에 따라 결과에 차이가 있다. 표제어 추출과 어간 추출은 모델을 더 낫게(또는 더 간단하게) 만들어 주기 때문에, 어떤 작업에서 마지막 성능까지 짜내야 할 떄 시도해보면 좋다.



##### 7.8.1 KoNlPy를 사용한 영화 리뷰 분석

여기서 사용할 한글 영화 리뷰 데이터셋은 다음과 같다.

[Naver sentiment movie corpus]: https://github.com/e9t/nsmc/

이 말뭉치는 네이버 영화 사이트의 리뷰 20만 개를 묶은 데이터이다. 

KoNLPy는 여러 언어로 만들어진 형태소 분석기를 파이썬에서 손쉽게 사용할 수 있ㄷ로고 도와주는 도구다. 

[KoNLPy 설치 방법]: http://konlpy.org/en/latest/install/



KoNLPy는 5개의 형태소 분석기를 각각 하나의 태그 클래스로 지원한다. 먼저 트위터에서 만든 한국어 처리기인 twitter-korean-text를 사용한다.

```python 
import pandas as pd

df_train = pd.read_csv('./ratings_train.txt', delimiter='\t', keep_default_na=False)
df_train.head(n=3)
```

![](./Figure/7_8_1_1.JPG)

```python 
In:
text_train = df_train['document'].as_matrix()
y_train = df_train['label'].as_matrix()

df_test = pd.read_csv('./ratings_test.txt', delimiter='\t', keep_default_na=False)
text_test = df_test['document'].as_matrix()
y_test = df_test['label'].as_matrix()

print(len(text_train), np.bincount(y_train))
print(len(text_test), np.bincount(y_test))
```

```python 
Out:
150000 [75173 74827]
50000 [24827 25173]
```

KoNLPy의 Twitter 클래스의 객체를 만들고 TfidfVectorizer의 tokenizer 매개변수에 주입할 함수를 만든다. 이 함수는 텍스트 하나를 입력 받아서 Twitter의 형태소 분석 메소드인 morphs에서 받은 문자열의 리스트를 그대로 반환한다.

```python 
from konlpy.tag import Twitter

twitter_tag = Twitter()

def twitter_tokenizer(text):
  return twitter_tag.morphs(text)
```

```python 
In:
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

twit_param_grid = {'tfidfvectorizer__min_df': [3, 5, 7],
                   'tfidfvectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
                   'logisticregression__C': [0.1, 1, 10]}
twit_pipe = make_pipeline(TfidfVectorizer(tokenizer=twitter_tokenizer), LogisticRegression())
twit_grid = GridSearchCV(twit_pipe, twit_param_grid)

twit_grid.fit(text_train[0:1000], y_train[0:1000])
print(f"최상의 교차 검증 점수: {twit_grid.best_score_:.3f}")
print(f"최적의 교차 검증 매개변수: ", twit_grid.best_params_)
```

```python 
Out:
최상의 교차 검증 점수: 0.718
최적의 교차 검증 매개변수:  {'logisticregression__C': 1, 'tfidfvectorizer__min_df': 3, 'tfidfvectorizer__ngram_range': (1, 3)}
```

```python 
In:
X_test_konlpy = twit_grid.best_estimator_.named_steps['tfidfvectorizer'].transform(text_test)
score = twit_grid.best_estimator_.named_steps['logisticregression'].score(X_test_konlpy, y_test)
print(f"테스트 세트 점수: {score:.3f}")
```

```python 
Out:
테스트 세트 점수: 0.707
```

전체 훈련 데이터를 사용해 그리드 서치를 진행하려면 테스트할 매개변수 조합이 많아 매우 오랜 시간이 걸린다. 여기서는 C++ 기반의 Mecab 태그 클래스를 사용한다. 

Mecab 형태소 분석을 사용하는 토큰 분할 함수를 만든다.

```python 
from konlpy.tag import Mecab
mecab = Mecab()

def mecab_tokenizer(text):
  return mecab.morphs(text)
```

특별히 PC에 장착된 CPU 코어를 최대한 활용하기 위해 n_jobs=-1로 지정한다. 

```python 
In:
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

mecab_param_grid = {'tfidfvectorizer__min_df': [3, 5, 7],
                    'tfidfvectorizer__ngram_range': [(1, i) for i in [1, 2, 3]],
                    'logisticregression__C': [10**i for i in range(-1, 3)]}
mecab_pipe = make_pipeline(TfidfVectorizer(tokenizer=mecab_tokenizer),
                           LogisticRegression())
mecab_grid = GridSearchCV(mecab_pipe, mecab_param_grid, n_jobs=-1)
mecab_grid.fit(text_train, y_train)
print(f"최상의 교차 검증 점수: {mecab_grid.best_score_:.3f}")
print(f"최적의 교차 검증 매개변수: ", mecab_grid.best_params_)
```

```python 
Out:
최상의 교차 검증 점수: 0.870
최적의 교차 검증 매개변수:  {'logisticregression__C': 10, 'tfidfvectorizer__min_df': 3, 'tfidfvectorizer__ngram_range': (1, 3)}
```

```python 
In:
X_test_mecab = mecab_grid.best_estimator_.named_steps['tfidfvectorizer'].transform(text_test)
score = mecab_grid.best_estimator_.named_step['logisticregression'].score(X_test_mecab, y_test)
print(f"테스트 세트 점수: {score:.3f}")
```

```python 
Out:
테스트 세트 점수: 0.875
```



### 7.9 토픽 모델링과 문서 군집화

텍스트 데이터에 자주 적용하는 기법으로 **토픽 모델링(Topic modeling)**이 있다. 이 용어는 비지도 학습으로 문서를 하나 또는 그 이상의 토픽으로 할당하는 작업을 통칭한다. '정치', '금융', '스포츠' 등의 토픽으로 묶을 수 있는 뉴스 데이터가 좋은 예이다. 한 문서가 하나의 토픽에 할당되면 문서를 군집시키는 문제가 된다. 문서가 둘 이상의 토픽을 가질 수 있다면 학습된 각 성분은 하나의 토픽에 해당하며 문서를 표현한 성분의 계수는 문서가 어떤 토픽에 얼마만큼 연관되어 있는지를 말해준다. 보통 토픽 모델링에 대해 이야기할 때 **잠재 디리클레 할당(Latent Dirichlet Allocation(LDA))** 라고 하는 특정한 성분 분해 방법을 이야기한다. 



##### 7.9.1 LDA

LDA 모델은 함께 자주 나타나는 단어의 그룹(토픽)을 찾는 것이다. 또 LDA는 각 문서에 토픽의 일부가 혼합되어 있다고 간주한다. 머신러닝에서 토픽은 일상 대화에서 말하는 '주제'가 아니고 의미가 있든 없든 PCA나 NMF로 추출한 성분에 가까운 것이다. LDA의 토픽에 의미가 있다고 하더라도, 이것은 보통 말하는 주제는 아니다. 정치 기사에서는 '주지사', '선거', '정당' 등의 단어를 예상할 수 있고 스포츠 기사에서는 '팀', '점수', '시즌' 같은 단어를 예상할 수 있다. 이런 그룹들의 단어는 함께 나타나는 경우가 많으며 '팀'과 '주지사'는 함께 나타나는 경우가 드물다. 그러나 동시에 나타날 것 같은 단어의 그룹만 있는 것은 아니다. 예를 들어 두 기자는 다른 문장이나 다른 종류의 단어를 좋아할 수있다. 한 명은 '구별'이란 단어를 즐겨 쓰고 다른 한 명은 '분리'란 말을 좋아할 수 있다. 이때 토픽은 전자의 기자가 즐겨쓰는 단어와 후자의 기자가 즐겨 쓰는 단어가 될 수 있다. 

다음은 영화 리뷰 데이터셋에 LDA를 적용하는 예제이다.

```python 
from sklearn.feature_extraction.text import CountVectorizer

#적어도 15%의 문서에서 빈번히 나타나는 단어를 삭제한 후 
#가장 많이 등장하는 단어 10,000개에 대한 BOW
vect = CountVectorizer(max_features=10000, max_df=.15)
X = vect.fit_transform(text_train)
```

10개의 토픽으로 토픽 모델을 학습시킨다. NMF 성분과 비슷하게 토픽은 어떤 순서를 가지고 있지 않으며, 토픽의 수를 바꾸면 모든 토픽이 바뀌게 된다(NMF과 LDA는 유사한 문제를 풀 수 있어서 토픽 추출에 NMF를 사용할 수도 있다). 기본 학습 방법인 "online" 대신 조금 느리지만 성능이 나은 "batch" 방법을 사용하고 모델 성능을 위해 "max_iter" 값을 증가시킨다(LDA는 온라인 변분 베이즈 알고리즘(Online varational Bayes algorithm)을 사용하여 max_iter 매개변수의 기본값이 10이다).

```python 
In:
from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=10, learning_method="batch",
                                max_iter=25, random_state=0)
document_topics = lda.fit_transform(X)
#LDA에는 토픽 마다 각 단어의 중요도를 저장한 components_ 속성이 있다.
#components_의 크기는 (n_topics, n_words)
print(f"lda.components_.shape: {lda.components_.shape}")
```

```python 
Out:
lda.components_.shape: (10, 10000)
```

```python 
In:
#토픽마다(components_의 행) 특성을 오름차순으로 정렬.
#내림차순이 되도록 [:, ::-1]을 사용해 행의 정렬을 반대로 바꿈.
sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
feature_names = np.array(vect.get_feature_names())
mglearn.tools.print_topics(topics=range(10), feature_names=feature_names, 
                           sorting=sorting, topics_per_chunk=5, n_words=10)
```

```python 
Out:
topic 0       topic 1       topic 2       topic 3       topic 4       
--------      --------      --------      --------      --------      
between       war           funny         show          didn          
young         world         worst         series        saw           
family        us            comedy        episode       am            
real          our           thing         tv            thought       
performance   american      guy           episodes      years         
beautiful     documentary   re            shows         book          
work          history       stupid        season        watched       
each          new           actually      new           now           
both          own           nothing       television    dvd           
director      point         want          years         got           


topic 5       topic 6       topic 7       topic 8       topic 9       
--------      --------      --------      --------      --------      
horror        kids          cast          performance   house         
action        action        role          role          woman         
effects       animation     john          john          gets          
budget        game          version       actor         killer        
nothing       fun           novel         oscar         girl          
original      disney        both          cast          wife          
director      children      director      plays         horror        
minutes       10            played        jack          young         
pretty        kid           performance   joe           goes          
doesn         old           mr            performances  around        
```

다음은 100개의 토픽으로 새로운 모델을 학습시킨 예이다. 많은 토픽을 사용하면 분석은 더 어려워지지만 데이터에서 특이한 부분을 잘 잡아낼 수 있다.

```python 
In:
lda100 = LatentDirichletAllocation(n_components=100, learning_method="batch",
                                   max_iter=25, random_state=0)
document_topics100 = lda100.fit_transform(X)
topics = np.array([7, 16, 24, 25, 28, 36, 37, 41, 45, 51, 53, 54, 63, 89, 97])
sorting = np.argsort(lda100.components_, axis=1)[:, ::-1]
feature_names = np.array(vect.get_feature_names())
mglearn.tools.print_topics(topics=topics, feature_names=feature_names,
                           sorting=sorting, topics_per_chunk=5, n_words=20)
```

```python 
Out:
topic 7       topic 16      topic 24      topic 25      topic 28      
--------      --------      --------      --------      --------      
thriller      worst         german        car           beautiful     
suspense      awful         hitler        gets          young         
horror        boring        nazi          guy           old           
atmosphere    horrible      midnight      around        romantic      
mystery       stupid        joe           down          between       
house         thing         germany       kill          romance       
director      terrible      years         goes          wonderful     
quite         script        history       killed        heart         
bit           nothing       new           going         feel          
de            worse         modesty       house         year          
performances  waste         cowboy        away          each          
dark          pretty        jewish        head          french        
twist         minutes       past          take          sweet         
hitchcock     didn          kirk          another       boy           
tension       actors        young         getting       loved         
interesting   actually      spanish       doesn         girl          
mysterious    re            enterprise    now           relationship  
murder        supposed      von           night         saw           
ending        mean          nazis         right         both          
creepy        want          spock         woman         simple        


topic 36      topic 37      topic 41      topic 45      topic 51      
--------      --------      --------      --------      --------      
performance   excellent     war           music         earth         
role          highly        american      song          space         
actor         amazing       world         songs         planet        
cast          wonderful     soldiers      rock          superman      
play          truly         military      band          alien         
actors        superb        army          soundtrack    world         
performances  actors        tarzan        singing       evil          
played        brilliant     soldier       voice         humans        
supporting    recommend     america       singer        aliens        
director      quite         country       sing          human         
oscar         performance   americans     musical       creatures     
roles         performances  during        roll          miike         
actress       perfect       men           fan           monsters      
excellent     drama         us            metal         apes          
screen        without       government    concert       clark         
plays         beautiful     jungle        playing       burton        
award         human         vietnam       hear          tim           
work          moving        ii            fans          outer         
playing       world         political     prince        men           
gives         recommended   against       especially    moon          


topic 53      topic 54      topic 63      topic 89      topic 97      
--------      --------      --------      --------      --------      
scott         money         funny         dead          didn          
gary          budget        comedy        zombie        thought       
streisand     actors        laugh         gore          wasn          
star          low           jokes         zombies       ending        
hart          worst         humor         blood         minutes       
lundgren      waste         hilarious     horror        got           
dolph         10            laughs        flesh         felt          
career        give          fun           minutes       part          
sabrina       want          re            body          going         
role          nothing       funniest      living        seemed        
temple        terrible      laughing      eating        bit           
phantom       crap          joke          flick         found         
judy          must          few           budget        though        
melissa       reviews       moments       head          nothing       
zorro         imdb          guy           gory          lot           
gets          director      unfunny       evil          saw           
barbra        thing         times         shot          long          
cast          believe       laughed       low           interesting   
short         am            comedies      fulci         few           
serial        actually      isn           re            half          
```

토픽을 이용해 추론을 더 잘 하려면 토픽에 할당된 문서를 보고 가장 높은 순위에 있는 단어의 의미를 확인해야 한다. 토픽 45는 음악에 관한 것으로 보이는데 이 토픽에 할당된 리뷰는 다음과 같다.

```python 
In:
# 음악적인 토픽 45를 가중치로 정렬한다.
music = np.argsort(document_topics100[:, 45])[::-1]
# 이 토픽이 가장 비중이 큰 문서 다섯개를 출력한다.
for i in music[:10]:
    # 첫 두 문장을 출력한다.
    print(b".".join(text_train[i].split(b".")[:2]) + b".\n")
```

```python 
Out:
b'I love this movie and never get tired of watching. The music in it is great.\n'
b"I enjoyed Still Crazy more than any film I have seen in years. A successful band from the 70's decide to give it another try.\n"
b'Hollywood Hotel was the last movie musical that Busby Berkeley directed for Warner Bros. His directing style had changed or evolved to the point that this film does not contain his signature overhead shots or huge production numbers with thousands of extras.\n'
b"What happens to washed up rock-n-roll stars in the late 1990's? They launch a comeback / reunion tour. At least, that's what the members of Strange Fruit, a (fictional) 70's stadium rock group do.\n"
b'As a big-time Prince fan of the last three to four years, I really can\'t believe I\'ve only just got round to watching "Purple Rain". The brand new 2-disc anniversary Special Edition led me to buy it.\n'
b"This film is worth seeing alone for Jared Harris' outstanding portrayal of John Lennon. It doesn't matter that Harris doesn't exactly resemble Lennon; his mannerisms, expressions, posture, accent and attitude are pure Lennon.\n"
b"The funky, yet strictly second-tier British glam-rock band Strange Fruit breaks up at the end of the wild'n'wacky excess-ridden 70's. The individual band members go their separate ways and uncomfortably settle into lackluster middle age in the dull and uneventful 90's: morose keyboardist Stephen Rea winds up penniless and down on his luck, vain, neurotic, pretentious lead singer Bill Nighy tries (and fails) to pursue a floundering solo career, paranoid drummer Timothy Spall resides in obscurity on a remote farm so he can avoid paying a hefty back taxes debt, and surly bass player Jimmy Nail installs roofs for a living.\n"
b"I just finished reading a book on Anita Loos' work and the photo in TCM Magazine of MacDonald in her angel costume looked great (impressive wings), so I thought I'd watch this movie. I'd never heard of the film before, so I had no preconceived notions about it whatsoever.\n"
b'I love this movie!!! Purple Rain came out the year I was born and it has had my heart since I can remember. Prince is so tight in this movie.\n'
b"This movie is sort of a Carrie meets Heavy Metal. It's about a highschool guy who gets picked on alot and he totally gets revenge with the help of a Heavy Metal ghost.\n"
```

토픽을 조사하는 다른 방법은 각 토픽의 가중치가 얼마인지 모든 리뷰에 걸쳐 document_topics 값을 합해서 보는 것이다. 다음은 각 토픽을 대표하는 두 단어로 토픽 이름을 붙인, 학습된 토픽의 가중치이다.

```python 
fig, ax = plt.subplots(1, 2, figsize=(10, 10))
topic_names = ["{:>2} ".format(i) + " ".join(words)
               for i, words in enumerate(feature_names[sorting[:, :2]])]
# 두 개의 열이 있는 막대 그래프
for col in [0, 1]:
    start = col * 50
    end = (col + 1) * 50
    ax[col].barh(np.arange(50), np.sum(document_topics100, axis=0)[start:end])
    ax[col].set_yticks(np.arange(50))
    ax[col].set_yticklabels(topic_names[start:end], ha="left", va="top")
    ax[col].invert_yaxis()
    ax[col].set_xlim(0, 2000)
    yax = ax[col].get_yaxis()
    yax.set_tick_params(pad=130)
plt.tight_layout()
```

![](./Figure/7_9_1_1.JPG)

중요도가 높은 토픽 중 97번은 거의 불용어에 가깝고 약간 부정적 경향의 단어이다. 토픽 16은 부정적이고, 장르에 관련된 토픽들이 이어지고 36, 37번은 칭찬하는 단어를 포함한다. 이렇게 LDA는 장르와 점수라는 두 종류의 큰 토픽과 어디에도 속하지 않는 토픽을 찾을 수 있다. 

LDA와 같은 토픽 모델은 레이블이 없거나, 레이블이 있더라도 큰 규모의 큰 텍스트 말뭉치를 해석하는 데 좋은 방법이다. LDA는 확률적 알고리즘이기 때문에 random_state를 바꾸면 결과가 많이 달라진다. 토픽으로 구별하는 게 도움이 되더라도 비지도 학습에서 내린 결론은 보수적으로 평가해야 하므로 각 토픽에 해당하는 문서를 직접 보고 직관을 검증하는 것이 좋다. LDA.transform 메소드에서 만든 토픽이 지도 학습을 위한 압축된 표현으로 사용될 수도 있다. 특히 훈련 샘플이 적을 떄 유용하다(훈련 데이터가 적고 특성의 수가 많을 때 과대적합하기 쉬우므로 특성의 수를 줄이는 데 LDA를 사용할 수 있다).



### 7.10 요약 및 정리

텍스트 데이터를 처리 할 때, 특히 스팸이나 부정거래 탐지, 감성 분석 같은  텍스트 분류 작업에서 BOW 표현은 간단하고 강력한 해법이 될 수 있다. 데이터의 표현이 자연어 처리 애플리케이션의 해심이고 추출된 토큰과 n-그램을 분석하면 모델링 과정에서 많은 통찰을 얻게 된다. 텍스트 처리 애플리케이션에서는 지도 학습이나 비지도 학습 작업을 위해 모델을 자체적으로 분석해 의미를 찾을 수 있을 때가 많다. 실전에서 자연어 처리 기반의 방법을 사용할 때 이 장점을 최대한 활용해야 한다.

참고 사항

- [Natural Language Processing with Python]: http://www.nltk.org/book/

- [Introduction to Information Retrieval]: https://nlp.stanford.edu/IR-book/

고수준의 텍스트 처리를 위한 패키지

- spacy - 비교적 최근에 나왔고 효율적이며 잘 설계된 패키지
- nltk - 매우 잘 구축되어 있고 기능이 풍부하지만 조금 오래된 라이브러리
- gensim - 토픽 모델링 강점인 자연어 처리 패키지

신경망과 관련된 몇 가지 연구가 텍스트 처리 분야에서 두각을 보였다.

- word2vec 라이브러리에 구현된 단어 벡터(Word vector) 또는 분산 단어 표현(Distributed word representations)의 연속적인 벡터 표현 

  [Distributed Representations of Words and Phrases and Their Compositionality]: https://goo.gl/V3mTpj

- 텍스트 처리와 순환 신경망(Recurrent neural networks, RNN). RNN은 신경망의 한 종류로, 클래스 레이블을 할당하는 분류 모데로가 달리 텍스트를 출력할 수 있다. 텍스트 출력을 만들 수 있기 때문에 자동 번역이나 자동 요약에 잘 맞는다.  

  [Sequence to Sequence Learning with Neural Networks]: https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf

  
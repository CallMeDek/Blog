# 1. 소개

### 1.1 왜 머신러닝인가?

머신러닝이 없던 초창기에는 규칙 기반 전문가 시스템(Rule-based expert system)을 사용하여 지능형 애플리케이션을 개발했다고 한다.  예를 들어,  이메일이 하나 왔을 때, 이 메일이 스팸 메일인지 아닌지를 판단하기 위해서 다음과 같은 과정이 필요했을 것이다.

```python
if "싼" in an_email.text:
    an_email.check = "스팸"

elif "지금 바로" in an_email.text:
    an_email.check = "스팸"
   
elif "클릭" in an_email.text:
    an_email.check = "스팸"
```

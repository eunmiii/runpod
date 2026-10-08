# PART 1. 멀티모달 AI의 기초

# 멀티모달 AI 이해

## 1. 멀티모달을 배우기 전에

ChatGPT와 같은 초기 LLM은 주로 텍스트를 입력받아 텍스트를 생성하였다.

예를 들어 사용자가 다음과 같이 질문한다고 생각해보자.

```text
사용자
"딥러닝이 무엇인가요?"

        ↓

텍스트

        ↓

언어 모델

        ↓

텍스트

        ↓

"딥러닝은 인공신경망을 이용하여..."
```

여기에서는 입력과 출력이 모두 **Text**이다.

하지만 사람이 정보를 받아들이는 방식은 텍스트 하나에 국한되지 않는다.

사람은 동시에 다음 정보를 이용한다.

```text
글을 읽는다.
사진을 본다.
사람의 말을 듣는다.
영상을 본다.
표와 그래프를 해석한다.
```

AI 역시 이러한 여러 종류의 정보를 함께 처리하도록 발전하고 있다.

이를 **멀티모달 AI(Multimodal AI)**라고 한다.

---

# 2. Modality란 무엇인가

멀티모달을 이해하려면 먼저 **Modality**라는 용어를 알아야 한다.

Modality는 쉽게 말하면

> **정보가 표현되는 형태**

라고 이해하면 된다.

예를 들어 다음은 서로 다른 modality이다.

| Modality | 데이터 예 |
|---|---|
| Text | 문장, 뉴스, 문서 |
| Image | 사진, 그림 |
| Audio | 음성, 음악 |
| Video | 동영상 |
| Sensor | 온도, 위치, 가속도 |
| Table | 구조화된 데이터 |

따라서

```text
Text + Image
```

를 동시에 처리한다면 두 개의 modality를 사용하는 것이다.

---

# 3. Single Modal과 Multimodal

기존 이미지 분류 모델을 생각해보자.

```text
강아지 이미지
      ↓
     CNN
      ↓
   Feature
      ↓
Classifier
      ↓
     Dog
```

입력이 Image 하나이므로 **Single Modal Model**이라고 볼 수 있다.

텍스트 감성분석도 마찬가지이다.

```text
"이 영화 정말 재미있다."
          ↓
       Text Model
          ↓
       Positive
```

입력이 Text 하나이다.

반면 다음 문제는 다르다.

```text
              이미지
                ↓
          Vision Encoder
                ↓
                │
                │
                ▼
             AI Model
                ▲
                │
                │
          Text Encoder
                ↑
                질문
```

예를 들어 사용자가 강아지 사진을 올리고 질문한다.

```text
이미지 : 강아지 사진

질문 :
"사진 속 동물은 무엇인가?"
```

모델이 답한다.

```text
"강아지입니다."
```

여기에서는

```text
Image + Text
```

두 종류의 정보를 함께 처리한다.

이것이 멀티모달의 기본 개념이다.

---

# 4. 대표적인 멀티모달 문제

멀티모달 모델을 이용하면 다양한 문제를 해결할 수 있다.

## Image Classification

```text
Image
 ↓
Model
 ↓
Class
```

예:

```text
강아지 사진 → Dog
고양이 사진 → Cat
```

---

## Image-Text Matching

이미지와 문장이 얼마나 관련 있는지를 판단한다.

```text
강아지 이미지

       ↕

"a photo of a dog"
```

결과:

```text
Similarity = 높음
```

반대로

```text
강아지 이미지

       ↕

"a photo of an airplane"
```

결과:

```text
Similarity = 낮음
```

이 기능이 이후 배우는 **CLIP**의 핵심이다.

---

# 5. Image Retrieval

텍스트를 입력하여 관련 이미지를 찾을 수도 있다.

```text
사용자

"a dog running on grass"

        ↓

Text Encoder

        ↓

Text Embedding

        ↓

이미지 Embedding들과 비교

        ↓

가장 비슷한 이미지 검색
```

검색 엔진을 다음처럼 생각하면 된다.

```text
검색어

"바닷가에서 뛰는 강아지"

        ↓

CLIP

        ↓

┌────────┐
│ image1 │  0.21
├────────┤
│ image2 │  0.91 ← 가장 유사
├────────┤
│ image3 │  0.35
└────────┘
```

---

# 6. Image Captioning

이미지를 보고 문장을 생성하는 문제이다.

```text
Image

 ↓

Vision Encoder

 ↓

Language Model

 ↓

Text
```

예:

```text
[강아지가 잔디밭을 뛰는 사진]

↓

"A dog is running on the grass."
```

이후 배우게 될 **BLIP**의 대표적인 기능이다.

---

# 7. Visual Question Answering

Visual Question Answering은 일반적으로 **VQA**라고 부른다.

입력은 다음 두 가지이다.

```text
Image
+
Question
```

예를 들어

```text
Image

[강아지가 공을 물고 있는 사진]

Question

"What is the dog holding?"
```

모델은 이미지와 질문을 함께 이해해야 한다.

```text
Answer

"a ball"
```

따라서 단순 이미지 분류보다 훨씬 복잡한 문제이다.

---

# 8. 멀티모달의 가장 중요한 문제


> 컴퓨터는 이미지와 텍스트를 어떻게 비교할 수 있는가?

사람에게는

```text
강아지 사진
```

과

```text
"a photo of a dog"
```

이 같은 의미라는 것이 당연하다.

하지만 컴퓨터 입장에서는 전혀 다르다.

이미지는

```text
Pixel
```

이고 텍스트는

```text
Token
```

이다.

즉,

```text
Image

[Pixel, Pixel, Pixel ...]

Text

[Token, Token, Token ...]
```

형태 자체가 완전히 다르다.

따라서 둘을 바로 비교할 수 없다.

이 문제를 해결하기 위해 등장하는 개념이

**Embedding**이다.

---

# 9. Embedding이란 무엇인가

Embedding은 데이터를 **숫자로 이루어진 벡터 공간으로 변환한 표현**이다.

예를 들어 컴퓨터가 다음 단어를 처리한다고 생각해보자.

```text
dog
cat
car
```

컴퓨터는 이를 다음과 같은 숫자 벡터로 표현할 수 있다.

```text
dog

[0.91, 0.82, 0.12]


cat

[0.88, 0.79, 0.15]


car

[0.10, 0.15, 0.93]
```

실제 임베딩은 3개의 숫자가 아니라 수백 또는 수천 개의 숫자로 구성될 수 있다.

여기서는 설명을 위해 3차원으로 단순화한 것이다.

---

# 10. 왜 벡터로 표현하는가

벡터로 표현하면 데이터 사이의 관계를 수학적으로 계산할 수 있기 때문이다.

예를 들어

```text
dog
[0.91, 0.82, 0.12]

cat
[0.88, 0.79, 0.15]
```

는 상당히 비슷하다.

반면

```text
car
[0.10, 0.15, 0.93]
```

는 상당히 다르다.

따라서 벡터 사이의 거리를 계산하면

```text
dog ↔ cat
```

이

```text
dog ↔ car
```

보다 의미적으로 가깝다는 것을 표현할 수 있다.

---

# 11. PyTorch Tensor로 직접 확인하기

먼저 실제 모델을 사용하지 않고 간단한 벡터를 직접 만들어보자.

```python
import torch

# ---------------------------------------------------------
# 의미를 설명하기 위해 임의로 만든 3차원 벡터이다.
# 실제 딥러닝 모델의 임베딩은 훨씬 높은 차원을 사용한다.
# ---------------------------------------------------------

dog = torch.tensor(
    [0.90, 0.80, 0.10],
    dtype=torch.float32
)

cat = torch.tensor(
    [0.85, 0.75, 0.15],
    dtype=torch.float32
)

car = torch.tensor(
    [0.10, 0.20, 0.95],
    dtype=torch.float32
)

print("dog :", dog)
print("cat :", cat)
print("car :", car)

print()

print("dog shape :", dog.shape)
print("cat shape :", cat.shape)
print("car shape :", car.shape)
```

예상되는 형태는 다음과 같다.

```text
dog : tensor([0.9000, 0.8000, 0.1000])
cat : tensor([0.8500, 0.7500, 0.1500])
car : tensor([0.1000, 0.2000, 0.9500])

dog shape : torch.Size([3])
```

여기서

```text
torch.Size([3])
```

은 숫자가 3개 있는 **3차원 벡터**라는 의미이다.

---

# 12. 벡터의 유사도를 어떻게 계산하는가

멀티모달 모델에서는 두 벡터가 얼마나 비슷한지 판단해야 한다.

대표적인 방법이

**Cosine Similarity**

이다.

한국어로는 **코사인 유사도**라고 한다.

공식은 다음과 같다.

\[
\text{cosine similarity}(A,B)
=
\frac{A\cdot B}
{\|A\|\|B\|}
\]

처음 배우는 단계에서는 수식 자체보다 의미가 중요하다.

두 벡터의 **방향이 얼마나 비슷한지 측정한다**고 이해하면 된다.

---

# 13. Cosine Similarity 직접 계산하기

PyTorch에서는 다음 함수를 사용할 수 있다.

```python
torch.nn.functional.cosine_similarity()
```

전체 코드를 실행한다.

```python
import torch
import torch.nn.functional as F

# 비교할 벡터를 생성한다.
dog = torch.tensor(
    [0.90, 0.80, 0.10],
    dtype=torch.float32
)

cat = torch.tensor(
    [0.85, 0.75, 0.15],
    dtype=torch.float32
)

car = torch.tensor(
    [0.10, 0.20, 0.95],
    dtype=torch.float32
)

# ---------------------------------------------------------
# cosine_similarity()에 두 벡터를 전달하여
# 두 벡터의 방향이 얼마나 비슷한지 계산한다.
#
# dim=0
# 현재 벡터가 [3] 형태이므로
# 0번 차원을 기준으로 유사도를 계산한다.
# ---------------------------------------------------------

dog_cat_similarity = F.cosine_similarity(
    dog,
    cat,
    dim=0
)

dog_car_similarity = F.cosine_similarity(
    dog,
    car,
    dim=0
)

print(
    "dog ↔ cat :",
    dog_cat_similarity.item()
)

print(
    "dog ↔ car :",
    dog_car_similarity.item()
)
```

여기에서 `.item()`은

```text
Tensor
```

안에 들어 있는 하나의 값을 Python 숫자로 가져오는 역할을 한다.

예를 들어

```python
print(dog_cat_similarity)
```

는

```text
tensor(0.998...)
```

형태가 될 수 있다.

반면

```python
print(dog_cat_similarity.item())
```

은

```text
0.998...
```

처럼 출력된다.

---

# 14. Cosine Similarity 결과 해석

일반적으로 cosine similarity는 다음처럼 해석할 수 있다.

| 값 | 의미 |
|---:|---|
| 1 | 방향이 동일함 |
| 0에 가까움 | 관련성이 낮음 |
| -1 | 반대 방향 |

다만 실제 임베딩에서는 단순히 특정 숫자 이상이면 무조건 같은 의미라고 판단해서는 안 된다.

중요한 것은 **상대적인 크기**이다.

예를 들어

```text
dog ↔ cat = 0.99

dog ↔ car = 0.30
```

이라면

```text
dog는 car보다 cat과 더 가깝다.
```

라고 해석할 수 있다.

---

# 15. Dot Product도 이해해야 하는 이유

CLIP 코드를 보면 다음과 같은 코드가 자주 등장한다.

```python
similarity = image_features @ text_features.T
```

처음 보면 cosine similarity 함수가 없어서 의문이 생길 수 있다.

이를 이해하려면 **벡터 정규화**를 알아야 한다.

---

# 16. 벡터 정규화

벡터의 크기를 1로 만드는 과정을 생각해보자.

PyTorch에서는 다음처럼 한다.

```python
F.normalize(vector, dim=0)
```

실제로 확인한다.

```python
import torch
import torch.nn.functional as F

dog = torch.tensor(
    [0.90, 0.80, 0.10],
    dtype=torch.float32
)

# 원래 벡터의 크기를 계산한다.
original_norm = torch.linalg.vector_norm(dog)

# L2 정규화를 수행한다.
normalized_dog = F.normalize(
    dog,
    dim=0
)

# 정규화된 벡터의 크기를 다시 계산한다.
normalized_norm = torch.linalg.vector_norm(
    normalized_dog
)

print("원래 벡터")
print(dog)

print()

print("원래 벡터 크기")
print(original_norm.item())

print()

print("정규화된 벡터")
print(normalized_dog)

print()

print("정규화 후 벡터 크기")
print(normalized_norm.item())
```

마지막 값은 거의

```text
1.0
```

이 된다.

---

# 17. 정규화와 Cosine Similarity의 관계

두 벡터를 먼저 정규화하면

\[
\frac{A}{||A||}
\]

와

\[
\frac{B}{||B||}
\]

가 된다.

이 두 벡터의 내적은 cosine similarity와 같다.

즉 다음 두 방법이 같은 결과를 낸다.

### 방법 1

```python
F.cosine_similarity(a, b)
```

### 방법 2

```python
a = F.normalize(a)
b = F.normalize(b)

similarity = a @ b
```

직접 확인한다.

```python
import torch
import torch.nn.functional as F

a = torch.tensor(
    [0.90, 0.80, 0.10],
    dtype=torch.float32
)

b = torch.tensor(
    [0.85, 0.75, 0.15],
    dtype=torch.float32
)

# 방법 1
cosine = F.cosine_similarity(
    a,
    b,
    dim=0
)

# 방법 2
a_norm = F.normalize(
    a,
    dim=0
)

b_norm = F.normalize(
    b,
    dim=0
)

dot_product = a_norm @ b_norm

print("Cosine Similarity :", cosine.item())
print("Normalized Dot Product :", dot_product.item())
```

두 결과는 거의 동일하게 나온다.

이 원리가 나중에 CLIP에서 매우 중요하다.

---

# 18. 하나의 벡터가 아니라 여러 벡터 비교하기

CLIP에서는 이미지 하나와 문장 하나만 비교하지 않는다.

예를 들어 이미지 3장과 문장 3개가 있다고 생각해보자.

```text
Image

Dog
Cat
Car


Text

"a dog"
"a cat"
"a car"
```

각각 임베딩하면 다음과 같은 행렬이 된다.

```text
Image Embeddings

[
 [dog vector],
 [cat vector],
 [car vector]
]
```

shape은

```text
[3, embedding_dimension]
```

이 된다.

---

# 19. 행렬 연산으로 모든 조합 비교하기

다음 코드를 실행한다.

```python
import torch
import torch.nn.functional as F

# ---------------------------------------------------------
# 이미지 3장의 임베딩이라고 가정한다.
# 각 이미지는 3차원 벡터로 표현하였다.
# ---------------------------------------------------------

image_embeddings = torch.tensor(
    [
        [0.90, 0.80, 0.10],  # dog image
        [0.80, 0.90, 0.10],  # cat image
        [0.10, 0.20, 0.95],  # car image
    ],
    dtype=torch.float32
)

# ---------------------------------------------------------
# 텍스트 3개의 임베딩이라고 가정한다.
# ---------------------------------------------------------

text_embeddings = torch.tensor(
    [
        [0.88, 0.82, 0.10],  # "a dog"
        [0.78, 0.92, 0.10],  # "a cat"
        [0.10, 0.15, 0.98],  # "a car"
    ],
    dtype=torch.float32
)

print(
    "Image Embedding Shape :",
    image_embeddings.shape
)

print(
    "Text Embedding Shape :",
    text_embeddings.shape
)

# ---------------------------------------------------------
# 각 벡터의 길이를 1로 정규화한다.
#
# dim=-1
# 마지막 차원, 즉 embedding 차원을 기준으로 정규화한다.
# ---------------------------------------------------------

image_embeddings = F.normalize(
    image_embeddings,
    dim=-1
)

text_embeddings = F.normalize(
    text_embeddings,
    dim=-1
)

# ---------------------------------------------------------
# 이미지 임베딩
#
# [3, 3]
#
# 텍스트 임베딩
#
# [3, 3]
#
# text_embeddings.T를 하면
#
# [3, 3]
#
# 이미지 × 텍스트 전치 행렬을 곱하면
# 모든 이미지와 모든 텍스트 사이의 유사도가 계산된다.
# ---------------------------------------------------------

similarity_matrix = (
    image_embeddings
    @
    text_embeddings.T
)

print()
print("Similarity Matrix")
print(similarity_matrix)
```

결과 구조는 다음과 같다.

```text
                    Text

                dog   cat   car

Image   dog      높음   높음   낮음

        cat      높음   높음   낮음

        car      낮음   낮음   높음
```

이것이 CLIP을 이해하는 데 매우 중요한 연산이다.

---

# 20. 이미지도 Embedding으로 만들 수 있는가

가능하다.

CNN이나 Vision Transformer는 이미지를 특징 벡터로 변환할 수 있다.

예를 들어 ResNet50을 생각해보자.

원래 ResNet50은 다음과 같은 구조를 가진다.

```text
Image

 ↓

Convolution

 ↓

Feature Extraction

 ↓

AdaptiveAvgPool

 ↓

Flatten

 ↓

FC Layer

 ↓

1000 Classes
```

여기서 FC Layer 직전의 feature를 가져오면 이미지의 특징을 나타내는 벡터로 사용할 수 있다.

```text
Image

 ↓

ResNet50

 ↓

Feature

 ↓

2048-dimensional Vector
```

---

# 21. 실습 환경 구성

재현 가능한 실습을 위해 버전을 고정하는 편이 안전하다.

예를 들어 `uv` 환경에서는 다음처럼 구성할 수 있다.

```bash
uv add "torch==2.8.0" "torchvision==0.23.0" "transformers==4.57.2" "pillow>=10,<12" "requests>=2.32,<3" "matplotlib>=3.9,<4"
```

Jupyter를 사용한다면 추가한다.

```bash
uv add "jupyter>=1,<2" "ipywidgets>=8.1,<9"
```

이후

```bash
uv run jupyter lab
```

로 실행할 수 있다.

---

# 22. CPU / CUDA / Apple MPS 자동 선택

앞으로 모든 실습에서 동일한 코드를 사용한다.

```python
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")

elif (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
):
    device = torch.device("mps")

else:
    device = torch.device("cpu")

print("사용 장치 :", device)
```

의미는 다음과 같다.

```text
NVIDIA GPU 존재

        ↓ Yes

      CUDA


        No
        ↓

Apple Silicon MPS 사용 가능

        ↓ Yes

       MPS


        No
        ↓

       CPU
```

---

# 23. ResNet50으로 실제 이미지 Embedding 만들기

외부 URL에 의존하지 않고도 바로 실행할 수 있도록 `torchvision`이 제공하는 CIFAR10 이미지를 사용한다.

처음 실행할 때 CIFAR10 데이터셋을 다운로드한다.

```python
import torch
from torchvision.datasets import CIFAR10
from torchvision.models import (
    resnet50,
    ResNet50_Weights
)

# ---------------------------------------------------------
# 실행 장치를 선택한다.
# ---------------------------------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda")
elif (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
):
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("사용 장치 :", device)

# ---------------------------------------------------------
# ImageNet으로 사전학습된 ResNet50의 가중치를 선택한다.
# ---------------------------------------------------------

weights = ResNet50_Weights.DEFAULT

# ---------------------------------------------------------
# 사전학습된 ResNet50 모델을 불러온다.
# ---------------------------------------------------------

model = resnet50(
    weights=weights
)

model = model.to(device)

# 추론 모드로 변경한다.
model.eval()

# ---------------------------------------------------------
# ResNet50 학습에 사용된 이미지 전처리 방법을 가져온다.
#
# resize
# crop
# tensor 변환
# normalize
#
# 등의 과정이 포함되어 있다.
# ---------------------------------------------------------

preprocess = weights.transforms()

# ---------------------------------------------------------
# CIFAR10 테스트 데이터셋을 다운로드한다.
#
# transform=None으로 설정하여
# 원본 PIL Image를 가져온다.
# ---------------------------------------------------------

dataset = CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=None
)

print("전체 이미지 수 :", len(dataset))
```

---

# 24. CIFAR10 데이터 확인

CIFAR10에는 다음 10개 클래스가 있다.

```python
print(dataset.classes)
```

대표적으로

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

이 포함되어 있다.

이미지 하나를 확인한다.

```python
import matplotlib.pyplot as plt

image, label = dataset[0]

print("Label Index :", label)
print("Class Name :", dataset.classes[label])
print("Image Size :", image.size)

plt.figure(figsize=(4, 4))
plt.imshow(image)
plt.title(dataset.classes[label])
plt.axis("off")
plt.show()
```

---

# 25. ResNet50의 마지막 FC Layer 확인

```python
print(model.fc)
```

ResNet50의 마지막에는 다음과 같은 계층이 있다.

```text
Linear(
    in_features=2048,
    out_features=1000
)
```

즉,

```text
2048차원 Feature

        ↓

FC

        ↓

1000개 ImageNet Class
```

이다.

우리는 분류 결과가 아니라 **2048차원 feature**를 가져오려고 한다.

---

# 26. FC Layer 제거하기

다음처럼 마지막 FC를 제거할 수 있다.

```python
import torch.nn as nn

feature_extractor = nn.Sequential(
    *list(model.children())[:-1]
)

feature_extractor = feature_extractor.to(device)

feature_extractor.eval()

print(feature_extractor)
```

구조는 다음과 같이 변한다.

```text
원래

Image
 ↓
ResNet
 ↓
2048 Features
 ↓
FC
 ↓
1000 Classes


변경

Image
 ↓
ResNet
 ↓
2048 Features
```

---

# 27. 이미지 하나를 Embedding으로 변환

```python
import torch

# CIFAR10 첫 번째 이미지를 가져온다.
image, label = dataset[0]

print(
    "원본 클래스 :",
    dataset.classes[label]
)

# ---------------------------------------------------------
# ResNet50이 요구하는 형태로 이미지를 전처리한다.
#
# 결과:
#
# [3, 224, 224]
#
# 형태의 Tensor가 만들어진다.
# ---------------------------------------------------------

image_tensor = preprocess(image)

print(
    "전처리 후 Shape :",
    image_tensor.shape
)

# ---------------------------------------------------------
# 모델은 Batch 단위로 입력을 받는다.
#
# 현재:
# [3, 224, 224]
#
# 모델 입력:
# [Batch, Channel, Height, Width]
#
# 따라서 batch 차원을 하나 추가한다.
# ---------------------------------------------------------

image_tensor = image_tensor.unsqueeze(0)

print(
    "Batch 추가 후 Shape :",
    image_tensor.shape
)

# GPU/MPS/CPU로 이동한다.
image_tensor = image_tensor.to(device)

# ---------------------------------------------------------
# 학습이 아니라 특징 추출이므로
# gradient 계산이 필요하지 않다.
# ---------------------------------------------------------

with torch.inference_mode():

    features = feature_extractor(
        image_tensor
    )

print(
    "Feature Shape :",
    features.shape
)
```

결과는

```text
[1, 2048, 1, 1]
```

형태이다.

---

# 28. 왜 [1, 2048, 1, 1]인가

각 숫자의 의미는 다음과 같다.

```text
[
 Batch,
 Channel,
 Height,
 Width
]
```

즉,

```text
[1, 2048, 1, 1]
```

은

```text
이미지 수 = 1

Feature = 2048개

공간 크기 = 1 × 1
```

이라는 의미이다.

이를 일반적인 벡터로 바꾼다.

```python
features = features.flatten(1)

print(
    "Flatten 후 Shape :",
    features.shape
)
```

결과:

```text
torch.Size([1, 2048])
```

이제 하나의 이미지를

```text
2048개의 숫자
```

로 표현하였다.

이것이 **Image Embedding 또는 Image Feature Vector**의 기본 개념이다.

엄밀히 말하면 여기서는 ResNet50의 feature representation이며, CLIP의 공동 의미 공간에 투영된 embedding과는 구분할 필요가 있다.

---

# 29. 여러 이미지를 Embedding으로 변환하기

함수로 만든다.

```python
import torch
import torch.nn.functional as F

def get_resnet_embedding(
    image,
    feature_extractor,
    preprocess,
    device
):
    """
    PIL Image 한 장을 입력받아
    ResNet50의 2048차원 특징 벡터를 반환한다.

    Parameters
    ----------
    image:
        PIL.Image.Image 형식의 이미지

    feature_extractor:
        마지막 FC Layer를 제거한 ResNet50

    preprocess:
        ResNet50 전용 이미지 전처리 함수

    device:
        cuda, mps 또는 cpu

    Returns
    -------
    embedding:
        shape = [1, 2048]인
        L2 정규화된 이미지 특징 벡터
    """

    # PIL Image를 모델 입력 Tensor로 변환한다.
    image_tensor = preprocess(image)

    # Batch 차원을 추가한다.
    image_tensor = image_tensor.unsqueeze(0)

    # 모델이 있는 장치로 이동한다.
    image_tensor = image_tensor.to(device)

    # 추론이므로 gradient 계산을 비활성화한다.
    with torch.inference_mode():

        features = feature_extractor(
            image_tensor
        )

    # [1, 2048, 1, 1]
    # ↓
    # [1, 2048]
    features = features.flatten(1)

    # cosine similarity 계산을 쉽게 하기 위해
    # 벡터 길이를 1로 정규화한다.
    embedding = F.normalize(
        features,
        p=2,
        dim=-1
    )

    return embedding
```

---

# 30. 같은 클래스와 다른 클래스 이미지 비교

CIFAR10에서 dog 이미지 두 장과 automobile 이미지 한 장을 찾아 비교해보자.

```python
# ---------------------------------------------------------
# 특정 클래스의 이미지를 찾는 함수
# ---------------------------------------------------------

def find_images_by_class(
    dataset,
    class_name,
    count=2
):
    """
    CIFAR10 데이터셋에서
    특정 클래스 이미지를 count개 찾는다.
    """

    class_index = dataset.classes.index(
        class_name
    )

    images = []

    for image, label in dataset:

        if label == class_index:
            images.append(image)

        if len(images) == count:
            break

    return images


# dog 이미지 2장
dog_images = find_images_by_class(
    dataset,
    "dog",
    count=2
)

# automobile 이미지 1장
car_images = find_images_by_class(
    dataset,
    "automobile",
    count=1
)

dog1 = dog_images[0]
dog2 = dog_images[1]
car1 = car_images[0]

print(
    "dog images :",
    len(dog_images)
)

print(
    "car images :",
    len(car_images)
)
```

---

# 31. 세 이미지 시각화

```python
import matplotlib.pyplot as plt

images = [
    dog1,
    dog2,
    car1
]

titles = [
    "dog 1",
    "dog 2",
    "automobile"
]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(9, 3)
)

for ax, image, title in zip(
    axes,
    images,
    titles
):
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()
```

---

# 32. 실제 이미지 Embedding 비교

```python
dog1_embedding = get_resnet_embedding(
    dog1,
    feature_extractor,
    preprocess,
    device
)

dog2_embedding = get_resnet_embedding(
    dog2,
    feature_extractor,
    preprocess,
    device
)

car_embedding = get_resnet_embedding(
    car1,
    feature_extractor,
    preprocess,
    device
)

print(
    "dog1 :",
    dog1_embedding.shape
)

print(
    "dog2 :",
    dog2_embedding.shape
)

print(
    "car :",
    car_embedding.shape
)
```

모두

```text
[1, 2048]
```

형태이다.

---

# 33. 실제 이미지 유사도 계산

```python
# ---------------------------------------------------------
# 앞에서 이미 L2 정규화를 수행하였다.
#
# 따라서 행렬곱으로 cosine similarity를 계산할 수 있다.
# ---------------------------------------------------------

dog_dog_similarity = (
    dog1_embedding
    @
    dog2_embedding.T
)

dog_car_similarity = (
    dog1_embedding
    @
    car_embedding.T
)

print(
    "dog ↔ dog similarity :",
    dog_dog_similarity.item()
)

print(
    "dog ↔ automobile similarity :",
    dog_car_similarity.item()
)
```

여기서 중요한 교육 포인트가 있다.

**같은 클래스라고 해서 항상 높은 유사도가 보장되는 것은 아니다.**

ResNet은 CLIP처럼 이미지와 텍스트를 공동 의미 공간에 맞추도록 학습된 모델이 아니다.

또한 CIFAR10 이미지는 32×32 크기로 매우 작다.

따라서 이 실습의 목적은

> 이미지를 숫자 벡터로 변환하고 벡터끼리 비교할 수 있다는 원리를 확인하는 것

이다.

---

# 34. 여기까지의 전체 과정

지금까지 수행한 과정을 정리하면 다음과 같다.

```text
              원본 이미지

                   ↓

               Resize
               Normalize

                   ↓

                Tensor

                   ↓

               ResNet50

                   ↓

          Feature Extraction

                   ↓

            [1, 2048, 1, 1]

                   ↓

                Flatten

                   ↓

              [1, 2048]

                   ↓

             L2 Normalize

                   ↓

          Image Embedding

                   ↓

          Cosine Similarity
```

이 과정이 매우 중요하다.

CLIP에서도 기본 아이디어는 동일하다.

다만 결정적인 차이가 하나 있다.

---

# 35. 이미지와 텍스트를 비교하려면 발생하는 문제

이미지는 ResNet으로 다음처럼 만들었다.

```text
Image

 ↓

ResNet50

 ↓

2048-dimensional vector
```

텍스트를 BERT와 같은 모델로 처리한다고 생각하면

```text
Text

 ↓

Tokenizer

 ↓

BERT

 ↓

768-dimensional vector
```

가 될 수 있다.

문제가 발생한다.

```text
Image

[2048 dimensions]


Text

[768 dimensions]
```

차원이 다르다.

따라서 직접

```python
image_embedding @ text_embedding.T
```

을 계산할 수 없다.

하지만 단순히 차원만 같게 만든다고 문제가 해결되는 것도 아니다.

---

# 36. 차원이 같다고 같은 공간은 아니다

이 부분이 멀티모달에서 매우 중요하다.

다음 두 모델이 있다고 가정하자.

```text
Image Encoder

Image
 ↓
512-dimensional Vector


Text Encoder

Text
 ↓
512-dimensional Vector
```

둘 다 512차원이므로 계산 자체는 가능하다.

하지만

```text
Image Vector의 1번 값
```

과

```text
Text Vector의 1번 값
```

이 같은 의미를 가진다는 보장이 없다.

즉,

> **차원이 같다는 것과 같은 의미 공간에 있다는 것은 전혀 다른 문제이다.**

그래서 필요한 것이 **Shared Embedding Space**이다.

---

# 37. Shared Embedding Space

한국어로는 보통

**공유 임베딩 공간**

이라고 표현한다.

목표는 이미지와 텍스트를 동일한 의미 공간에 배치하는 것이다.

```text
Image

 ↓

Image Encoder

 ↓

Image Projection

 ↓

        Shared Embedding Space


Text

 ↓

Text Encoder

 ↓

Text Projection

 ↓

        Shared Embedding Space
```

예를 들어

```text
강아지 이미지
```

와

```text
"a photo of a dog"
```

가 같은 의미라면 공유 공간에서 서로 가까워야 한다.

```text
Shared Embedding Space


       ● Dog Image
      /
     /
    ● "a photo of a dog"


                         ● "an airplane"
```

---

# 38. Projection Layer가 필요한 이유

Image Encoder와 Text Encoder가 서로 다른 차원의 feature를 만들 수 있다.

예를 들어

```text
Image Encoder

768 dimensions


Text Encoder

512 dimensions
```

이라고 하자.

각각 projection을 적용한다.

```text
Image

 ↓

Image Encoder

 ↓

768

 ↓

Image Projection

 ↓

512
```

텍스트도

```text
Text

 ↓

Text Encoder

 ↓

512

 ↓

Text Projection

 ↓

512
```

가 된다.

최종적으로

```text
Image Embedding → 512

Text Embedding  → 512
```

로 맞춘다.

하지만 다시 강조하면 projection의 목적은 단순히 숫자의 개수만 맞추는 것이 아니다.

**학습을 통해 이미지와 텍스트의 의미가 대응되는 공간을 만드는 것**이 핵심이다.

---

# 39. Positive Pair

공유 임베딩 공간을 학습하려면 이미지와 해당 이미지의 설명이 필요하다.

예를 들어

```text
Image 1
강아지 사진

Text 1
"a photo of a dog"
```

은 서로 맞는 쌍이다.

이를

**Positive Pair**

라고 한다.

```text
Dog Image
     ↕
"a photo of a dog"

Positive Pair
```

---

# 40. Negative Pair

반대로 서로 맞지 않는 조합은 Negative Pair이다.

```text
Dog Image
     ↕
"a photo of a car"

Negative Pair
```

또는

```text
Dog Image
     ↕
"a photo of an airplane"

Negative Pair
```

이다.

---

# 41. Contrastive Learning

이제 학습 목표를 매우 간단하게 정의할 수 있다.

```text
Positive Pair

Similarity ↑


Negative Pair

Similarity ↓
```

즉,

```text
맞는 것 → 가까이

틀린 것 → 멀리
```

학습하는 것이다.

이러한 학습 방식을 **Contrastive Learning**이라고 한다.

---

# 42. Batch를 이용한 Contrastive Learning

이미지 3장과 해당 설명이 있다고 생각해보자.

```text
Image 1 = Dog
Text 1  = "a dog"

Image 2 = Cat
Text 2  = "a cat"

Image 3 = Car
Text 3  = "a car"
```

모든 조합을 비교하면

```text
             Text

             dog    cat    car

Image dog     O      X      X

      cat     X      O      X

      car     X      X      O
```

대각선이 Positive Pair이다.

나머지는 Negative Pair이다.

따라서 모델은

```text
대각선 similarity → 크게

나머지 similarity → 작게
```

학습하면 된다.

---

# 43. Contrastive Learning의 핵심을 코드로 확인

```python
import torch
import torch.nn.functional as F

# ---------------------------------------------------------
# 이미지 3개의 embedding이라고 가정한다.
# ---------------------------------------------------------

image_embeddings = torch.tensor(
    [
        [0.95, 0.05, 0.00],
        [0.05, 0.95, 0.00],
        [0.00, 0.05, 0.95],
    ],
    dtype=torch.float32
)

# ---------------------------------------------------------
# 각각의 이미지와 대응하는 텍스트 embedding이다.
# ---------------------------------------------------------

text_embeddings = torch.tensor(
    [
        [0.90, 0.10, 0.00],
        [0.10, 0.90, 0.00],
        [0.00, 0.10, 0.90],
    ],
    dtype=torch.float32
)

# L2 정규화
image_embeddings = F.normalize(
    image_embeddings,
    dim=-1
)

text_embeddings = F.normalize(
    text_embeddings,
    dim=-1
)

# ---------------------------------------------------------
# 모든 이미지와 모든 텍스트의 유사도를 계산한다.
#
# 결과:
#
# [3, 3]
# ---------------------------------------------------------

logits = (
    image_embeddings
    @
    text_embeddings.T
)

print(logits)
```

결과에서 대각선이 가장 크게 나타난다.

---

# 44. Temperature 적용

실제 contrastive learning에서는 similarity를 그대로 사용하지 않고 temperature를 적용하는 경우가 많다.

예를 들어

\[
logits =
\frac{similarity}{temperature}
\]

이다.

코드로 확인한다.

```python
temperature = 0.07

logits = (
    image_embeddings
    @
    text_embeddings.T
) / temperature

print(logits)
```

temperature가 작으면 유사도 차이가 더 크게 확대된다.

CLIP에서는 학습 가능한 logit scale을 사용한다.

---

# 45. 정답 Label 만들기

현재 batch에서

```text
Image 0 ↔ Text 0

Image 1 ↔ Text 1

Image 2 ↔ Text 2
```

가 정답이다.

따라서

```python
labels = torch.arange(3)

print(labels)
```

결과:

```text
tensor([0, 1, 2])
```

이다.

---

# 46. Image → Text Loss

각 이미지가 올바른 텍스트를 선택하도록 학습한다.

```python
loss_image_to_text = F.cross_entropy(
    logits,
    labels
)

print(
    "Image → Text Loss :",
    loss_image_to_text.item()
)
```

예를 들어 첫 번째 행은

```text
Dog Image

↓

Text 0
Text 1
Text 2

↓

정답 = Text 0
```

라는 분류 문제처럼 볼 수 있다.

---

# 47. Text → Image Loss

반대 방향도 학습한다.

```python
loss_text_to_image = F.cross_entropy(
    logits.T,
    labels
)

print(
    "Text → Image Loss :",
    loss_text_to_image.item()
)
```

이번에는

```text
"a dog"

↓

Image 0
Image 1
Image 2

↓

정답 = Image 0
```

이 된다.

---

# 48. 최종 Contrastive Loss

두 방향의 loss를 평균낸다.

```python
loss = (
    loss_image_to_text
    +
    loss_text_to_image
) / 2

print(
    "Contrastive Loss :",
    loss.item()
)
```

전체 구조는 다음과 같다.

```text
                  Similarity Matrix

                       ↓

              ┌─────────────────┐
              │                 │
              ▼                 ▼

         Image → Text      Text → Image

              │                 │

              ▼                 ▼

           CE Loss           CE Loss

              │                 │

              └────────┬────────┘
                       │
                       ▼

                    Average

                       ↓

               Contrastive Loss
```

이 개념이 CLIP 학습의 핵심이다.

---

# 49. CLIP으로 연결하기

지금까지 배운 내용을 연결하면 CLIP의 구조가 자연스럽게 나온다.

```text
                    CLIP

     Image                       Text
       │                           │
       ▼                           ▼

Vision Encoder               Text Encoder

       │                           │
       ▼                           ▼

Image Feature                Text Feature

       │                           │
       ▼                           ▼

Image Projection            Text Projection

       │                           │
       ▼                           ▼

Image Embedding             Text Embedding

       │                           │
       └───────────┬───────────────┘
                   │
                   ▼

          Shared Embedding Space

                   │
                   ▼

            Cosine Similarity

                   │
                   ▼

           Contrastive Learning
```

CLIP의 공식 구현에서도 이미지와 텍스트 특징을 각각 추출하는 인터페이스를 제공한다. 현재 Transformers 문서에서는 `get_image_features()`와 `get_text_features()`가 해당 역할을 담당한다. [Hugging Face](https://huggingface.co/docs/transformers/model_doc/clip?utm_source=chatgpt.com)

---

# 50. CLIP이 기존 이미지 분류와 다른 점

기존 CNN 분류를 다시 생각해보자.

```text
Image

 ↓

CNN

 ↓

Feature

 ↓

FC Layer

 ↓

Dog / Cat / Car
```

분류 클래스가 FC Layer에 고정되어 있다.

예를 들어 10개 클래스로 학습했다면 기본적으로 그 10개 클래스 중 하나를 예측한다.

CLIP은 다르다.

```text
Image
  ↓
Image Encoder
  ↓
Image Embedding
          │
          │ 비교
          ▼
Text Embeddings

"a dog"

"a cat"

"a car"

"a tiger"

"a bus"
```

텍스트 후보를 바꾸면 새로운 분류 문제를 만들 수 있다.

이것이 CLIP의 **Zero-shot Classification**을 이해하는 핵심이다.

---

# 51. Zero-shot Classification

Zero-shot은 해당 분류 문제에 대해 별도의 지도학습을 하지 않고 새로운 클래스를 판단하는 방식이다.

예를 들어 CLIP에게

```text
"a photo of a dog"

"a photo of a cat"

"a photo of a car"
```

를 제공한다.

각 텍스트를 embedding으로 만든다.

```text
Dog Text Embedding

Cat Text Embedding

Car Text Embedding
```

이미지도 embedding으로 만든다.

```text
Image

 ↓

Image Embedding
```

그다음 비교한다.

```text
Image ↔ dog = 0.82

Image ↔ cat = 0.23

Image ↔ car = 0.08
```

가장 높은 것이

```text
dog
```

이므로

```text
Prediction = dog
```

으로 판단한다.

---

# 52. 지금까지 반드시 이해해야 하는 연결

여기까지가 이후 CLIP과 BLIP 실습을 제대로 이해하기 위한 기반이다.

```text
Multimodal
    │
    ▼
서로 다른 데이터 형태
    │
    ├──────── Image
    │
    └──────── Text
              │
              ▼
           Encoder
              │
              ▼
           Feature
              │
              ▼
          Projection
              │
              ▼
          Embedding
              │
              ▼
     Shared Embedding Space
              │
              ▼
       Cosine Similarity
              │
              ▼
      Contrastive Learning
              │
              ▼
             CLIP
```


| 개념 | 의미 |
|---|---|
| Modality | 정보의 형태 |
| Encoder | 데이터를 특징으로 변환하는 모델 |
| Feature | 모델이 추출한 특징 |
| Embedding | 벡터 공간에서 표현된 데이터 |
| Projection | 서로 다른 표현을 목표 차원/공간으로 변환 |
| Shared Embedding Space | 이미지와 텍스트가 함께 배치되는 의미 공간 |
| Cosine Similarity | 두 벡터의 방향적 유사성 측정 |
| Positive Pair | 서로 의미가 맞는 이미지-텍스트 쌍 |
| Negative Pair | 서로 의미가 맞지 않는 쌍 |
| Contrastive Learning | Positive는 가깝게, Negative는 멀게 학습 |
| Zero-shot | 해당 분류 작업을 별도로 재학습하지 않고 예측 |

---

# 다음 단계: CLIP → BLIP


다음 「멀티모달 모델 - CLIP, BLIP」 단순히 모델 호출 코드를 나열하는 방식보다 다음 순서로 연결하는 것이 적절하다.

```text
멀티모달 기초
      ↓
Shared Embedding Space
      ↓
Contrastive Learning
      ↓
CLIP 구조
      ↓
CLIP Processor
      ↓
Image Encoder
      ↓
Text Encoder
      ↓
get_image_features()
      ↓
get_text_features()
      ↓
Zero-shot Classification
      ↓
Prompt Engineering
      ↓
CIFAR10 실제 성능 평가
      ↓
Text → Image Retrieval
      ↓
Image → Image Retrieval
      ↓
Embedding 저장/재사용
      ↓
BLIP 등장 배경
      ↓
CLIP과 BLIP 구조 차이
      ↓
Vision Encoder
      ↓
Text Decoder
      ↓
Image Captioning
      ↓
Conditional Captioning
      ↓
Visual Question Answering
      ↓
CLIP + BLIP 통합 프로젝트
```

특히 **BLIP을 가르칠 때 CLIP과 구조적 차이를 명확히 연결해야 한다.** CLIP은 이미지와 텍스트의 대응 관계를 임베딩 공간에서 비교하는 데 강점이 있는 반면, BLIP의 VQA 모델은 vision encoder, text encoder, text decoder를 사용하여 이미지와 질문을 처리하고 답변을 생성한다. [Hugging Face](https://huggingface.co/docs/transformers/model_doc/blip?utm_source=chatgpt.com)


[Hugging Face CLIP 공식 문서](https://huggingface.co/docs/transformers/model_doc/clip?utm_source=chatgpt.com)  
[Hugging Face BLIP 공식 문서](https://huggingface.co/docs/transformers/model_doc/blip?utm_source=chatgpt.com)


---

# 핵심 정리

멀티모달 AI의 출발점은 서로 다른 데이터 형식을 하나의 계산 가능한 표현으로 연결하는 것이다. 이미지는 Pixel, 텍스트는 Token으로 표현되므로 원래 상태에서는 직접 비교하기 어렵다. Encoder는 각 입력을 Feature로 변환하고, Projection을 거쳐 비교 가능한 Embedding 공간으로 보낸다. 이때 의미가 비슷한 데이터가 가까워지도록 학습된 공간을 Shared Embedding Space라고 이해할 수 있다.

Cosine Similarity는 두 Embedding의 방향이 얼마나 비슷한지를 측정한다. L2 정규화를 수행한 두 Vector의 Dot Product는 Cosine Similarity와 같은 의미를 갖기 때문에 CLIP과 같은 모델의 코드에서 행렬 곱으로 대량의 유사도를 효율적으로 계산할 수 있다.

Contrastive Learning은 Positive Pair의 유사도는 높이고 Negative Pair의 유사도는 낮추도록 학습한다. 이 원리가 이미지와 텍스트를 같은 의미 공간에 정렬하는 핵심이다.

# 질문과 답변

### Q1. Modality란 무엇인가?

**답변:** 정보가 표현되는 형태이다. Text, Image, Audio, Video, Sensor, Table 등이 서로 다른 Modality에 해당한다.

### Q2. 이미지와 텍스트를 바로 비교할 수 없는 이유는 무엇인가?

**답변:** 이미지는 Pixel 값으로, 텍스트는 Token으로 표현되기 때문에 데이터 구조와 의미 표현 방식이 서로 다르다. 각각을 Encoder와 Projection을 통해 Embedding으로 변환해야 비교할 수 있다.

### Q3. Embedding은 단순히 차원을 줄이는 기술인가?

**답변:** 아니다. Embedding의 핵심은 입력의 특징이나 의미를 Vector로 표현하는 것이다. 차원 축소가 동반될 수 있지만, 차원 축소 자체가 Embedding의 목적은 아니다.

### Q4. 두 Vector의 차원이 같으면 Shared Embedding Space인가?

**답변:** 아니다. 차원이 같다는 것은 행렬 연산이 가능하다는 뜻일 뿐이다. 이미지와 텍스트의 의미가 같은 방향으로 정렬되도록 학습되어야 Shared Embedding Space라고 할 수 있다.

### Q5. Cosine Similarity에서 1에 가까우면 항상 같은 의미인가?

**답변:** 일반적으로 방향이 매우 유사하다는 뜻이지만 절대적인 임계값으로 해석해서는 안 된다. 사용한 모델, 데이터 분포, Task에 따라 점수 범위가 달라질 수 있으므로 후보 간 상대 비교와 검증이 중요하다.

### Q6. L2 Normalize 후 Dot Product를 사용하는 이유는 무엇인가?

**답변:** Vector의 길이를 1로 정규화하면 두 Vector의 Dot Product가 Cosine Similarity와 동일해진다. 여러 Image/Text Embedding을 행렬 곱으로 한 번에 비교할 수 있어 효율적이다.

### Q7. Positive Pair와 Negative Pair는 무엇인가?

**답변:** Positive Pair는 의미상 서로 대응하는 이미지와 텍스트 쌍이고, Negative Pair는 서로 대응하지 않는 조합이다. Contrastive Learning은 Positive Pair는 가깝게, Negative Pair는 멀어지도록 학습한다.

# 확인문제

1. Single Modal과 Multimodal의 차이를 설명하라.
2. Image Encoder와 Text Encoder가 필요한 이유를 설명하라.
3. Feature, Projection, Embedding의 관계를 설명하라.
4. Cosine Similarity와 정규화된 Dot Product의 관계를 설명하라.
5. Contrastive Learning이 Shared Embedding Space를 만드는 과정을 설명하라.

## 확인문제 해설

1. Single Modal은 하나의 정보 형태를 처리하고, Multimodal은 둘 이상의 정보 형태를 연결하거나 함께 처리한다.
2. Pixel과 Token은 직접 비교할 수 없으므로 각 Modality를 의미 있는 Feature로 변환하기 위해 별도의 Encoder가 필요하다.
3. Encoder가 Feature를 만들고 Projection이 이를 공통 차원으로 변환하며, 최종 Vector 표현을 Embedding으로 사용한다.
4. 두 Vector를 L2 정규화하면 Vector의 길이가 1이 되므로 Dot Product가 Cosine Similarity와 같아진다.
5. 대응하는 Image-Text Pair의 유사도는 높이고 다른 Pair의 유사도는 낮추는 학습을 반복하면서 의미가 정렬된다.

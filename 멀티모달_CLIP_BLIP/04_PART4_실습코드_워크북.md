# PART 4. 실습 코드 워크북

이 문서는 제공된 Notebook의 설명과 코드를 Markdown 교재 형태로 옮긴 보충 자료이다. 본문 PART 1~3의 개념을 읽은 뒤 코드를 직접 실행하며 구조를 확인하는 용도로 사용할 수 있다.

> 코드 실행 전에 사용하는 Dataset의 **원본 해상도**를 확인한다. Processor가 만드는 모델 입력 크기는 원본 화질이 향상되었다는 의미가 아니다.

---

# CLIP·BLIP 멀티모달 AI 

개념 설명 → 구조와 작동 원리 → 코드 실습 → 실행 결과 해석 → 학습 확인 순서로 진행한다.

수업의 핵심은 CLIP과 BLIP의 차이를 명확하게 이해하는 데 있다.

* CLIP: 이미지와 텍스트를 벡터로 변환하고, 두 데이터가 얼마나 의미적으로 유사한지 계산한다.

* BLIP: 이미지 정보를 바탕으로 설명 문장을 생성하거나 이미지에 관한 질문에 답한다.

# 1. 멀티모달 AI와 CLIP 기초


# 이미지와 텍스트를 연결하는 원리 이해

학습 목표: 임베딩, 인코더, 공유 임베딩 공간, 코사인 유사도를 이해하고 CLIP으로 직접 계산한다.

## 1. 멀티모달 AI 개요 
### 1.1 모달리티란 무엇인가?

모달리티(Modality)는 정보가 표현되는 형태 또는 데이터 유형을 의미한다.

| 모달리티   | 데이터 예시        | AI의 주요 처리 대상   |
| ------ | ------------- | -------------- |
| 텍스트    | 문장, 문서, 질문    | 단어와 문장의 의미     |
| 이미지    | 사진, 그림, 의료 영상 | 객체, 색상, 형태, 장면 |
| 오디오    | 음성, 음악, 환경음   | 음성 내용, 소리의 특징  |
| 비디오    | 동영상, CCTV 영상  | 객체와 시간에 따른 행동  |
| 센서 데이터 | 온도, 압력, 위치    | 수치 패턴과 상태 변화   |

하나의 데이터 유형만 처리하면 단일 모달 AI이고, 서로 다른 유형의 정보를 함께 활용하면 멀티모달 AI라고 할 수 있다.

예를 들어 이미지 분류 모델은 사진을 입력받아 `고양이`라는 클래스를 출력한다. 멀티모달 모델은 사진뿐 아니라 텍스트 설명이나 질문까지 함께 활용할 수 있다.

### 1.2 단일 모달 AI와 멀티모달 AI 비교

![image.png](attachment:image.png)

### 1.3 멀티모달 AI의 실제 활용 사례

* 이미지 검색: “눈 덮인 산과 호수가 있는 사진”이라는 문장으로 사진을 검색한다.

* 의료 영상 보조: 영상과 판독 관련 텍스트의 연관성을 분석한다. 실제 진단에는 별도의 임상 검증이 필요하다.

* 상품 검색: “검은색 가죽 가방”처럼 자연어로 상품 이미지를 검색한다.

* 이미지 설명 생성: 사진을 입력하면 “소파 위에 고양이 두 마리가 앉아 있다”와 같은 설명을 생성한다.

* 시각적 질의응답: 사진을 보여 주고 “테이블 위에 무엇이 있는가?”라고 질문한다.

여기서 첫 세 가지는 CLIP의 활용 방식과 밀접하고, 이미지 설명과 질의응답은 BLIP의 대표적인 활용 방식이다.

### 수업 확인 질문

1. 이미지 분류 AI와 멀티모달 AI의 차이는 무엇인가?

2. 이미지와 텍스트를 함께 처리하면 어떤 문제를 해결할 수 있는가?

3. 자연어로 사진을 검색하려면 이미지와 문장이 어떤 방식으로 연결되어야 하는가?

<details>
<summary>정답 보기</summary>


</details>

## 2. 임베딩 기초 

### 2.1 임베딩이란 무엇인가?

임베딩(Embedding)은 텍스트나 이미지 같은 데이터를 숫자로 구성된 벡터로 표현한 것이다.

컴퓨터는 이미지의 의미를 사람처럼 직접 이해하지 못한다. 모델은 학습 과정에서 데이터의 특징을 추출해 수치 벡터로 표현한다.

예를 들어 설명을 위한 가상의 임베딩을 다음과 같이 생각할 수 있다.

| 데이터     | 가상 임베딩            |
| ------- | ----------------- |
| 고양이 사진  | `[0.8, 0.2, 0.1]` |
| 고양이 설명문 | `[0.7, 0.3, 0.1]` |
| 자동차 설명문 | `[0.1, 0.2, 0.9]` |

위 숫자는 원리를 설명하기 위한 예시이며 실제 CLIP의 출력값은 아니다.

고양이 사진과 고양이 설명문은 벡터 방향이 비슷하고, 자동차 설명문은 상대적으로 다른 방향을 갖는다고 가정할 수 있다.

### 2.2 벡터의 차원

벡터는 여러 숫자를 순서대로 나열한 자료 구조다.

* 2차원 벡터: `[0.8, 0.2]`

* 3차원 벡터: `[0.8, 0.2, 0.1]`

* 고차원 임베딩: 수백 개 이상의 숫자로 데이터의 특징을 표현한다.

CLIP의 `openai/clip-vit-base-patch32` 모델은 이미지와 텍스트를 각각 512차원 임베딩으로 표현한다. 즉, 이미지 하나는 512개의 숫자로 표현되는 벡터로 변환될 수 있다.

### 2.3 임베딩 공간 시각화

![고양이 사진](./image/1_2_임베딩공간시각화.png)

중요한 점은 벡터가 비슷하다는 의미는 사람이 정한 규칙으로 숫자를 직접 배정했다는 뜻이 아니라, 모델이 데이터에서 특징을 학습했다는 뜻이라는 것이다.

### 2.4 임베딩과 원본 데이터의 차이

| 구분    | 원본 데이터        | 임베딩                    |
| ----- | ------------- | ---------------------- |
| 이미지   | 픽셀로 구성된 사진    | 이미지 특징을 표현하는 숫자 벡터     |
| 텍스트   | 사람이 읽는 문장     | 텍스트 의미와 특징을 표현하는 숫자 벡터 |
| 주요 용도 | 사람이 내용을 직접 확인 | 검색, 유사도 계산, 분류 등에 활용   |

### 2.5 실습: 벡터 유사도 계산

```python

import torch
import torch.nn.functional as F

# 원리를 이해하기 위한 가상의 임베딩 벡터다.
# 실제 CLIP 모델이 생성한 값은 아니다.
cat_image = torch.tensor([0.8, 0.2, 0.1])
cat_text = torch.tensor([0.7, 0.3, 0.1])
car_text = torch.tensor([0.1, 0.2, 0.9])

# 두 벡터의 코사인 유사도를 계산한다.
# 값이 1에 가까울수록 두 벡터의 방향이 비슷하다.
similarity_cat = F.cosine_similarity(
    cat_image.unsqueeze(0),
    cat_text.unsqueeze(0)
)

similarity_car = F.cosine_similarity(
    cat_image.unsqueeze(0),
    car_text.unsqueeze(0)
)

print("고양이 이미지와 고양이 문장:", similarity_cat.item())
print("고양이 이미지와 자동차 문장:", similarity_car.item())

#  이 코드는 임베딩과 유사도의 원리를 학습하기 위한 간단한 예제다. 실제 CLIP에서는 사람이 벡터를 지정하는 대신 이미지 인코더와 텍스트 인코더가 입력 데이터에서 벡터를 생성한다.
```

## 3. CLIP의 모델 구조 

### 3.1 CLIP이란 무엇인가?

CLIP(Contrastive Language–Image Pre-training)은 이미지와 텍스트 사이의 의미적 관계를 학습한 비전-언어 모델이다.

일반적인 이미지 분류 모델은 학습 데이터에 정의된 클래스에 맞춰 예측하도록 학습한다. 반면 CLIP은 이미지와 텍스트를 연결하는 방식으로 학습하므로, 후보 텍스트를 바꾸어 새로운 분류 문제에 적용할 수 있다.

CLIP의 핵심 구성 요소는 다음과 같다.

![1_3_CLIP의핵심구성요소](./image/1_3_CLIP의핵심구성요소.png)

### 3.2 이미지 인코더

이미지 인코더는 사진을 입력받아 시각적 특징을 추출한다.

CLIP의 대표적인 이미지 인코더는 Vision Transformer(ViT) 또는 ResNet 계열을 사용할 수 있다.  `openai/clip-vit-base-patch32`는 ViT 기반 모델이다.

이미지 인코더가 수행하는 주요 과정은 다음과 같다.

1. 이미지를 모델이 요구하는 크기와 픽셀 범위로 전처리한다.

2. 이미지를 패치 단위로 나누어 시각적 정보를 표현한다.

3. Transformer가 패치 간 관계를 처리한다.

4. 시각적 특징을 추출한다.

5. 투영 계층을 통해 텍스트와 비교할 수 있는 임베딩을 생성한다.

### 3.3 텍스트 인코더

텍스트 인코더는 문장을 토큰으로 변환하고 문맥적 특징을 추출한다.

예를 들어 `A cat is sitting on a sofa.`라는 문장은 토큰화 과정을 거쳐 숫자 ID로 표현된다. Transformer 기반 텍스트 인코더가 이 토큰의 문맥을 처리한 후, 최종적으로 텍스트 임베딩을 생성한다.

여기서 토큰 ID 자체가 임베딩은 아니다. 토큰 ID는 사전에 등록된 토큰을 가리키는 번호이며, 인코더는 이를 이용해 문맥적 특징을 계산한다.

### 3.4 공유 임베딩 공간

공유 임베딩 공간(Shared Embedding Space)은 이미지와 텍스트를 같은 벡터 공간에서 비교할 수 있도록 표현한 공간이다.

예를 들어 이미지 인코더가 강아지 사진을 벡터로 표현하고, 텍스트 인코더가 `A dog playing in a park.`를 벡터로 표현한다고 가정한다. CLIP은 두 표현이 의미적으로 관련되도록 학습되어 있다.

다만 서로 다른 인코더의 출력이 자동으로 비교 가능한 것은 아니다. 학습 과정에서 두 표현을 정렬하도록 학습했다는 점이 중요하다.

### 3.5 대조학습(Contrastive Learning)

CLIP은 이미지와 텍스트의 쌍을 활용한 대조학습을 통해 시각적 표현과 언어적 표현을 연결한다.

* 긍정 쌍: 사진과 그 사진에 대응하는 설명문

* 부정 쌍: 해당 사진과 의미적으로 관련이 적은 다른 설명문

학습 과정에서는 긍정 쌍의 관련성 점수를 높이고, 배치 안의 다른 이미지·텍스트 쌍과는 구별할 수 있도록 학습한다.

이 교재에서는 다음과 같이 설명하면 이해하기 쉽다.

> CLIP은 사진에 정답 레이블만 붙이는 방식이 아니라, 사진과 설명문이 서로 잘 맞는지를 학습한다. 그 결과 학습에 직접 사용하지 않은 새로운 후보 문장도 이미지와 비교할 수 있다.

### 수업 확인 질문

1. 이미지 인코더와 텍스트 인코더는 각각 어떤 역할을 하는가?

2. 공유 임베딩 공간이 필요한 이유는 무엇인가?

3. 긍정 쌍과 부정 쌍은 대조학습에서 어떤 역할을 하는가?

4. 토큰 ID와 임베딩 벡터는 어떻게 다른가?

<details>
<summary>정답 보기</summary>


</details>

## 4. 이미지 전처리와 텍스트 토큰화
### 4.1 이미지 전처리가 필요한 이유

딥러닝 모델은 입력 이미지의 크기와 채널 구성, 픽셀값 범위가 일정해야 한다.

원본 사진의 크기가 각각 800 X 600, 1024 X 768, 400 X 400 이라면 이를 그대로 한 배치에 넣기 어렵다. 따라서 모델이 기대하는 입력 형식에 맞춰 전처리해야 한다.

CLIP의 전처리에는 일반적으로 다음 작업이 포함된다.

* 이미지의 색상 모드를 RGB로 맞춘다.

* 모델의 학습 설정에 맞춰 이미지 크기를 조정하고 필요한 영역을 자른다.

* 픽셀값을 텐서로 변환한다.

* 사전학습 때 사용한 평균과 표준편차를 기준으로 정규화한다.

`CLIPProcessor`를 사용하면 모델에 맞는 전처리를 수행할 수 있다. 따라서 직접 전처리 함수를 작성하기 전에는 Processor가 수행하는 작업을 먼저 확인하는 것이 좋다.

### 4.2 텍스트 토큰화

텍스트는 모델이 처리할 수 있는 숫자 ID의 시퀀스로 변환된다.

예를 들어 다음 문장을 입력한다고 가정한다.

`A photo of a cat.`

토큰화 과정에서는 문장이 여러 토큰으로 분리되고, 각 토큰이 어휘 사전에 정의된 ID로 변환된다. 실제 토큰 분할 방식은 토크나이저에 따라 달라질 수 있다.

### 4.3 코드 실습: Processor의 출력 확인

```python
# ============================================================
# 4.3 Processor의 출력 확인
# 실습 모델: CLIP
# ============================================================

import torch
from PIL import Image
from IPython.display import display
from urllib.request import urlopen
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel


# ------------------------------------------------------------
# 1. 모델 설정
# ------------------------------------------------------------

# Hugging Face에 공개된 사전 학습 CLIP 모델이다.
# 처음 실행할 때 모델과 Processor 파일을 다운로드한다.
MODEL_NAME = "openai/clip-vit-base-patch32"

# CPU와 GPU 중 사용할 장치를 자동으로 선택한다.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

print("=" * 60)
print("1. 실행 환경")
print("=" * 60)
print("PyTorch 버전:", torch.__version__)
print("사용 장치:", device)


# ------------------------------------------------------------
# 2. CLIP Processor와 Model 불러오기
# ------------------------------------------------------------

# CLIPProcessor:
# 이미지 전처리와 텍스트 토큰화를 담당한다.
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# CLIPModel:
# Processor가 변환한 입력을 받아 이미지와 텍스트를 분석한다.
model = CLIPModel.from_pretrained(MODEL_NAME)

# 모델을 CPU 또는 GPU에 배치한다.
model = model.to(device)

# 학습이 아닌 추론 모드로 설정한다.
model.eval()

print("\nProcessor와 Model을 정상적으로 불러왔다.")


# ------------------------------------------------------------
# 3. 테스트 이미지 생성
# ------------------------------------------------------------

# 실제 강아지 사진을 인터넷에서 가져온다.
image_url = "https://images.dog.ceo/breeds/maltese/n02085936_100.jpg"

# URL에서 이미지 데이터를 읽어 PIL Image 객체로 변환한다.
with urlopen(image_url, timeout=30) as response:
    image = Image.open(BytesIO(response.read())).convert("RGB")

# 이미지의 기본 정보를 확인한다.
print("\n" + "=" * 60)
print("2. 원본 이미지 정보")
print("=" * 60)
print("자료형:", type(image))
print("이미지 크기:", image.size)
print("색상 모드:", image.mode)

# 실제 강아지 사진을 화면에 표시한다.
display(image)


# ------------------------------------------------------------
# 4. 텍스트 후보 준비
# ------------------------------------------------------------

texts = [
    "a photo of a dog",
    "a photo of a cat",
    "a photo of a car"
]

print("\n" + "=" * 60)
print("3. 입력 텍스트")
print("=" * 60)

for index, text in enumerate(texts):
    print(f"{index}: {text}")


# ------------------------------------------------------------
# 5. Processor 실행
# ------------------------------------------------------------

# Processor는 이미지와 텍스트를 각각 모델 입력 형식으로 변환한다.
#
# text:
#   토큰화할 텍스트 목록
#
# images:
#   전처리할 PIL 이미지
#
# return_tensors="pt":
#   결과를 PyTorch Tensor 형태로 반환
#
# padding=True:
#   텍스트 길이가 서로 다를 때 길이를 맞춘다.
inputs = processor(
    text=texts,
    images=image,
    return_tensors="pt",
    padding=True
)


# ------------------------------------------------------------
# 6. Processor 출력의 자료형과 키 확인
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("4. Processor 출력 확인")
print("=" * 60)

print("전체 자료형:", type(inputs))
print("출력 키:", list(inputs.keys()))

# 일반적으로 CLIPProcessor는 다음 항목을 반환한다.
# input_ids: 텍스트 토큰 ID
# attention_mask: 실제 토큰과 패딩 토큰을 구분하는 마스크
# pixel_values: 전처리된 이미지 픽셀 텐서

for key, value in inputs.items():
    print(f"\n항목 이름: {key}")
    print("자료형:", type(value))
    print("텐서 크기:", tuple(value.shape))
    print("데이터 자료형:", value.dtype)
    print("장치:", value.device)


# ------------------------------------------------------------
# 7. input_ids 확인
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("5. input_ids 확인")
print("=" * 60)

# input_ids는 텍스트를 토큰 ID로 변환한 결과다.
print(inputs["input_ids"])

# 토큰 ID를 다시 사람이 읽을 수 있는 토큰으로 변환한다.
tokens = processor.tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0].tolist()
)

print("\n첫 번째 문장의 토큰:")
print(tokens)


# ------------------------------------------------------------
# 8. attention_mask 확인
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("6. attention_mask 확인")
print("=" * 60)

# 1: 실제 입력 토큰
# 0: 패딩 토큰
print(inputs["attention_mask"])


# ------------------------------------------------------------
# 9. pixel_values 확인
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("7. pixel_values 확인")
print("=" * 60)

pixel_values = inputs["pixel_values"]

print("이미지 텐서 크기:", tuple(pixel_values.shape))
print("최솟값:", pixel_values.min().item())
print("최댓값:", pixel_values.max().item())

# pixel_values에는 원본 이미지가 아니라
# CLIP이 처리할 수 있도록 전처리된 픽셀 값이 저장된다.
print("\n첫 번째 이미지 텐서의 일부 값:")
print(pixel_values[0, :, :3, :3])


# ------------------------------------------------------------
# 10. 모델 입력 장치로 이동
# ------------------------------------------------------------

# Processor는 일반적으로 CPU에 있는 텐서를 반환한다.
# 모델이 GPU에 있다면 입력도 GPU로 옮겨야 한다.
inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


# ------------------------------------------------------------
# 11. CLIP 모델 실행
# ------------------------------------------------------------

# 추론 단계에서는 역전파를 위한 그래디언트를 계산하지 않는다.
with torch.no_grad():
    outputs = model(**inputs)

print("\n" + "=" * 60)
print("8. CLIP 모델 실행 결과")
print("=" * 60)

# logits_per_image:
# 이미지 하나와 각 텍스트 후보 사이의 비교 점수
print("이미지-텍스트 점수 텐서 크기:",
      tuple(outputs.logits_per_image.shape))

# logits_per_text:
# 각 텍스트와 이미지 사이의 비교 점수
print("텍스트-이미지 점수 텐서 크기:",
      tuple(outputs.logits_per_text.shape))

print("\n이미지-텍스트 점수:")
print(outputs.logits_per_image)


# ------------------------------------------------------------
# 12. 후보별 상대 점수와 순위 확인
# ------------------------------------------------------------

# softmax는 현재 후보 목록 안에서 상대 점수를 계산한다.
scores = outputs.logits_per_image[0]
probabilities = scores.softmax(dim=0)

print("\n" + "=" * 60)
print("9. 텍스트 후보별 상대 점수")
print("=" * 60)

# 점수가 높은 후보부터 정렬한다.
ranking = sorted(
    zip(texts, probabilities.tolist()),
    key=lambda item: item[1],
    reverse=True
)

for rank, (text, score) in enumerate(ranking, start=1):
    print(f"{rank}위 | {text} | 상대 점수: {score:.4f}")

print("\n실습 완료")
```

예상되는 출력 항목은 `pixel_values`, `input_ids`, `attention_mask` 등이다. 이미지 크기와 텍스트 길이에 따라 실제 텐서 크기는 달라진다.

| 입력 항목            | 의미                     |
| ---------------- | ---------------------- |
| `pixel_values`   | 전처리된 이미지 픽셀 텐서         |
| `input_ids`      | 텍스트 토큰의 ID             |
| `attention_mask` | 실제 토큰과 패딩 토큰을 구분하는 마스크 |

### 4.4 배치 처리의 의미

배치(Batch)는 여러 입력을 묶어 한 번에 처리하는 방식이다.

이미지 1장과 문장 5개를 비교할 때는 이미지 배치 크기가 1이고 텍스트 배치 크기는 5가 될 수 있다. 이미지 10장과 문장 5개를 함께 처리하면 결과적으로 이미지와 텍스트 간 모든 조합의 점수를 계산할 수도 있다.

배치 처리는 GPU 활용도를 높일 수 있지만, 배치가 커지면 메모리 사용량도 증가한다.

### 실습 과제

* 크기가 서로 다른 이미지 3장을 준비한다.

* Processor를 이용해 입력 텐서의 크기를 확인한다.

* 텍스트 후보를 2개에서 5개로 늘려 `input_ids`의 형태가 어떻게 달라지는지 확인한다.

* 패딩과 어텐션 마스크가 필요한 이유를 설명한다.

## 5. 사전학습 CLIP 모델 불러오기

### 5.1 사전학습 모델이란?

사전학습 모델(Pre-trained Model)은 대규모 데이터와 학습 과정을 통해 이미 파라미터가 학습된 모델이다.

처음부터 모델을 학습하려면 대량의 데이터, 연산 자원, 학습 시간이 필요하다. 사전학습 모델을 활용하면 학습된 표현을 이용해 다양한 작업을 빠르게 실험할 수 있다.

이번 이 교재에서는 다음 두 클래스를 사용한다.

* `CLIPProcessor`: 이미지와 텍스트를 모델 입력 형식으로 변환한다.

* `CLIPModel`: 이미지와 텍스트의 특징을 추출하고 관련성 점수를 계산한다.

### 5.2 모델 로딩 코드

```python

import torch
from transformers import CLIPModel, CLIPProcessor

# GPU를 사용할 수 있으면 GPU를 선택하고, 아니면 CPU를 사용한다.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hugging Face에 공개된 CLIP 체크포인트 이름이다.
model_name = "openai/clip-vit-base-patch32"

# 모델이 요구하는 이미지 전처리와 텍스트 토큰화를 담당한다.
processor = CLIPProcessor.from_pretrained(model_name)

# 사전학습 가중치를 불러온 뒤 실행 장치로 이동한다.
model = CLIPModel.from_pretrained(model_name).to(device)

# 추론 모드로 설정한다.
# Dropout과 같은 학습·추론 동작 차이가 있는 계층을 평가 방식으로 전환한다.
model.eval()

print("실행 장치:", device)
print("모델 로드 완료")
```

### 5.3 `model.eval()`과 `torch.no_grad()`의 차이

두 기능은 서로 다르다.

| 기능                | 역할                         |
| ----------------- | -------------------------- |
| `model.eval()`    | 모델을 평가 모드로 전환한다.           |
| `torch.no_grad()` | 해당 코드 블록에서 기울기 계산을 비활성화한다. |

추론 시에는 일반적으로 두 기능을 함께 사용한다. 다만 `eval()`을 호출했다고 해서 기울기 계산이 자동으로 비활성화되는 것은 아니다.

### 5.4 사전학습과 미세조정 구분

* 추론: 이미 학습된 파라미터를 이용해 결과를 계산한다.

* 미세조정(Fine-tuning): 특정 데이터셋을 사용해 기존 모델의 파라미터 일부 또는 전체를 추가로 학습한다.

* 처음부터 학습: 모델 파라미터를 초기화하고 학습 데이터를 이용해 새로 학습한다.


## 6. 이미지·텍스트 임베딩 추출 

### 6.1 임베딩을 직접 추출하는 이유

모델 내부의 임베딩을 직접 확인하면 유사도 계산이 어떻게 이루어지는지 이해할 수 있다.

CLIP은 이미지와 텍스트를 각각 인코더에 전달하고, 투영된 특징 벡터를 반환한다. 이번 예제에서 사용하는 체크포인트의 최종 임베딩 차원은 512다.

### 6.2 코드 실습

```python

import torch
import torch.nn.functional as F
from PIL import Image

# 이미지와 비교할 텍스트 후보를 준비한다.
image = Image.open('./image/dog.jpg').convert("RGB")

texts = [
    "A photo of a cat.",
    "A photo of a dog.",
    "A photo of a car."
]

# 이미지와 텍스트를 각각 모델 입력으로 변환한다.
inputs = processor(
    images=image,
    text=texts,
    return_tensors="pt",
    padding=True
)

# 입력 텐서를 모델이 실행되는 장치로 이동한다.
inputs = {key: value.to(device) for key, value in inputs.items()}

# 추론 단계이므로 기울기 계산을 비활성화한다.
import torch
import torch.nn.functional as F

with torch.no_grad():

    # 1. 이미지 인코더를 실행한다.
    image_outputs = model.vision_model(
        pixel_values=inputs["pixel_values"]
    )

    # 2. 이미지의 pooled output을 추출한다.
    image_pooled = image_outputs.pooler_output

    # 3. 이미지 투영층을 적용해 CLIP 임베딩 공간으로 변환한다.
    image_features = model.visual_projection(image_pooled)

    # 4. 텍스트 인코더를 실행한다.
    text_outputs = model.text_model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )

    # 5. 텍스트의 pooled output을 추출한다.
    text_pooled = text_outputs.pooler_output

    # 6. 텍스트 투영층을 적용해 CLIP 임베딩 공간으로 변환한다.
    text_features = model.text_projection(text_pooled)

# 7. # 각 임베딩 벡터의 L2 노름이 1이 되도록 정규화한다.
# 이렇게 하면 이후 내적을 코사인 유사도로 사용할 수 있다.
image_features = F.normalize(image_features, p=2, dim=-1)
text_features = F.normalize(text_features, p=2, dim=-1)

print("이미지 임베딩:", image_features.shape)
print("텍스트 임베딩:", text_features.shape)
```

예상되는 형태는 다음과 같다.

* 이미지 임베딩: `(1, 512)`

* 텍스트 임베딩: `(3, 512)`

이 크기는 해당 체크포인트와 입력 개수에 대한 예시다. 다른 모델을 사용하면 임베딩 차원이 달라질 수 있다.

### 6.3 코드 해석

`image_features`는 이미지 1장을 표현하는 512차원 벡터다. `text_features`는 텍스트 후보 3개를 각각 표현하는 512차원 벡터 3개를 담는다.

두 벡터의 차원이 같은 이유는 두 인코더의 출력을 비교할 수 있는 공통 임베딩 공간으로 투영하기 때문이다.

### 학습 확인 질문

1. 이미지 임베딩의 첫 번째 차원은 무엇을 의미하는가?

2. 텍스트 후보가 3개라면 텍스트 임베딩의 형태는 어떻게 되는가?

3. 임베딩을 정규화하면 어떤 장점이 있는가?

## 7. 코사인 유사도와 유사도 행렬 

### 7.1 코사인 유사도

코사인 유사도(Cosine Similarity)는 두 벡터의 방향이 얼마나 비슷한지 측정하는 방법이다.


$$\text{Cosine Similarity}(I, T) = \cos(\theta) = \frac{I \cdot T}{\Vert{}I\Vert{} \Vert{}T\Vert{}} = \frac{\sum_{i=1}^{n} I_i T_i}{\sqrt{\sum_{i=1}^{n} I_i^2} \sqrt{\sum_{i=1}^{n} T_i^2}}$$

여기서 I는 이미지 임베딩, T는 텍스트 임베딩을 의미한다.

코사인 유사도는 두 벡터의 방향에 초점을 둔다. 벡터의 크기가 달라도 방향이 같으면 코사인 유사도는 1이 될 수 있다.

### 7.2 CLIP에서 유사도 계산

```python

# 이미지 임베딩: (이미지 수, 임베딩 차원)
# 텍스트 임베딩: (텍스트 수, 임베딩 차원)

# 텍스트 임베딩의 전치 행렬을 사용해 행렬 곱을 수행한다.
# 결과: (이미지 수, 텍스트 수)
similarity_matrix = image_features @ text_features.T

print("유사도 행렬 크기:", similarity_matrix.shape)
print(similarity_matrix)
```

이미지 1장과 텍스트 3개를 비교하면 결과 행렬은 `(1, 3)` 형태가 된다.

### 7.3 여러 이미지와 여러 텍스트의 관계

이미지 3장과 텍스트 4개를 비교하면 유사도 행렬은 `(3, 4)` 형태가 된다.

### 유사도 행렬 예시

아래 값은 원리를 설명하기 위한 가상 데이터다.

|       |      |      |      |
| ----- | ---- | ---- | ---- |
| 이미지   | 고양이  | 강아지  | 자동차  |
| 이미지 1 | 0.91 | 0.18 | 0.06 |
| 이미지 2 | 0.22 | 0.86 | 0.11 |
| 이미지 3 | 0.04 | 0.13 | 0.94 |

각 행에서 가장 높은 값은 해당 이미지와 가장 관련성이 높은 후보 문장을 나타낸다. 그러나 실제 점수는 모델, 프롬프트, 이미지 내용에 따라 달라지며 점수만으로 의미의 완전한 일치나 정답 확률을 보장하지 않는다.

### 7.4 유사도 결과 정렬

```python

# 이미지 한 장에 대한 유사도 점수를 1차원 텐서로 선택한다.
scores = similarity_matrix[0]

# 점수가 높은 순서대로 텍스트 인덱스를 정렬한다.
ranked_indices = torch.argsort(scores, descending=True)

# 각 후보 문장과 유사도 점수를 출력한다.
for rank, index in enumerate(ranked_indices, start=1):
    idx = int(index)
    print(
        f"{rank}위: {texts[idx]} "
        f"(유사도: {scores[idx].item():.4f})"
    )
```

### 수업 확인 질문

* 유사도 행렬에서 행과 열은 각각 무엇을 의미하는가?

* 가장 높은 유사도 점수를 받은 문장을 예측 결과로 선택하는 이유는 무엇인가?

* 유사도가 높아도 이미지 설명이 틀릴 수 있는 이유는 무엇인가?

## 8. CLIP 결과 분석 및 실습 평가

### 실습 절차

1. 이미지 1장을 불러온다.

2. 서로 다른 설명문 5개를 작성한다.

3. CLIP으로 이미지와 텍스트 임베딩을 추출한다.

4. 코사인 유사도를 계산한다.

5. 유사도가 높은 순서로 문장을 정렬한다.

6. 이미지와 결과를 비교하고 오분류 원인을 기록한다.

### 결과 분석 보고서 예시

| 항목     | 작성 내용                     |
| ------ | ------------------------- |
| 이미지 내용 | 이미지에 실제로 포함된 객체와 상황       |
| 후보 문장  | 비교에 사용한 5개 문장             |
| 최상위 결과 | 유사도 점수가 가장 높은 문장          |
| 결과 적절성 | 이미지 내용과 일치하는지 여부          |
| 오차 원인  | 후보 문장 구성, 세부 속성, 배경 등의 영향 |
| 개선 방안  | 후보 문장 수정 및 추가 실험 계획       |

### 평가 과제

과제: 이미지와 텍스트 간 유사도를 계산하는 프로그램을 작성하고 결과를 분석한다.

평가 기준은 다음과 같다.

* 모델 및 Processor를 정상적으로 불러오는가?

* 이미지와 텍스트의 임베딩을 추출하는가?

* 유사도 계산과 순위 정렬이 올바른가?

* 실제 이미지와 결과를 비교해 한계점을 설명하는가?

# 2. CLIP 응용과 BLIP 기초


# Zero-shot 분류와 이미지 설명 생성

학습 목표: CLIP으로 이미지 분류·검색을 구현하고, BLIP으로 이미지 설명을 생성한다.

## 1. CLIP Zero-shot 분류 
### 1.1 Zero-shot이란 무엇인가?

Zero-shot 분류는 새로운 분류 대상에 대해 별도의 클래스별 분류기를 학습하지 않고, 입력 이미지와 후보 텍스트의 관련성을 비교해 클래스를 선택하는 방식이다.

일반적인 이미지 분류에서는 강아지, 고양이, 자동차 등을 구분하기 위해 해당 클래스가 포함된 데이터로 분류 모델을 학습한다.

CLIP에서는 다음과 같이 후보 문장을 준비할 수 있다.

* `A photo of a cat.`

* `A photo of a dog.`

* `A photo of a car.`

입력 이미지와 각 문장 사이의 점수를 계산한 뒤 가장 높은 점수를 받은 문장을 예측 결과로 선택한다.

### 1.2 실행 구조

![image.png](./image/2_1_입력이미지.png)

### 1.3 코드 실습

```python

import torch

# 분류할 클래스의 이름을 정의한다.
candidate_labels = ["cat", "dog", "car", "airplane"]

# 각 클래스 이름을 자연어 문장으로 바꾼다.
# CLIP은 이미지와 텍스트 간 관련성을 비교한다.
candidate_texts = [
    f"A photo of a {label}."
    for label in candidate_labels
]

# 이미지와 후보 문장을 모델 입력으로 전처리한다.
inputs = processor(
    images=image,
    text=candidate_texts,
    return_tensors="pt",
    padding=True
)

# 모델의 실행 장치와 입력 장치를 일치시킨다.
inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

# 모델을 추론 모드로 실행한다.
with torch.no_grad():
    outputs = model(**inputs)

    # 각 후보 텍스트와 이미지 간 관련성 로짓을 가져온다.
    logits = outputs.logits_per_image[0]

    # 후보 클래스 사이의 상대적 점수 분포를 계산한다.
    # 이 값은 후보 집합에 조건부인 점수이지, 보정된 실제 정답 확률은 아니다.
    scores = torch.softmax(logits, dim=0)

# 점수가 높은 후보부터 출력한다.
ranked_indices = torch.argsort(scores, descending=True)

for idx in ranked_indices:
    i = int(idx)
    print(
        f"{candidate_labels[i]:10s} "
        f"점수={scores[i].item():.4f}"
    )
```

### 1.4 결과 해석 시 주의할 점

Zero-shot 분류는 별도의 클래스별 학습 없이 활용할 수 있다는 장점이 있지만, 항상 정확한 결과를 보장하지 않는다.

* 후보 클래스에 정답이 없으면 가장 덜 부적절한 후보를 선택할 수 있다.

* 문장 표현을 바꾸면 점수 순위가 달라질 수 있다.

* 학습 데이터와 실제 이미지의 분포가 다르면 성능이 저하될 수 있다.

* 세부적인 객체 수량이나 공간 관계를 정확히 판별하지 못할 수 있다.

따라서 분류 결과를 평가할 때는 정답 레이블이 있는 데이터셋을 이용해 전체 정확도와 클래스별 정확도를 계산해야 한다.

### 수업 확인 질문

1. Zero-shot 분류에서 후보 텍스트는 어떤 역할을 하는가?

2. 후보 문장에 실제 정답 클래스가 없으면 어떤 문제가 발생하는가?

3. 소프트맥스 점수를 실제 정답 확률로 단정할 수 없는 이유는 무엇인가?

## 2. 프롬프트 설계 

### 2.1 프롬프트가 중요한 이유

CLIP은 이미지와 텍스트를 비교한다. 따라서 후보 문장을 어떻게 구성하느냐에 따라 유사도 점수가 달라질 수 있다.

예를 들어 고양이 사진에 대해 다음 후보를 비교할 수 있다.

* `cat`

* `a photo of a cat`

* `a close-up photo of a cat`

* `a cat sitting on a sofa`

이 문장들은 같은 객체를 언급하더라도 표현하는 상황과 세부 정보가 다르다.

### 2.2 프롬프트 비교 실험

```python

# 같은 이미지에 대해 서로 다른 프롬프트를 준비한다.
prompts = [
    "cat",
    "a photo of a cat",
    "a close-up photo of a cat",
    "a cat sitting on a sofa",
    "a dog sitting on a sofa"
]

# 동일한 이미지와 프롬프트를 함께 전처리한다.
inputs = processor(
    images=image,
    text=prompts,
    return_tensors="pt",
    padding=True
)
inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

# 이미지-텍스트 관련성 점수를 계산한다.
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits_per_image[0]

# 점수를 후보 문장 간 비교가 쉬운 상대적 분포로 변환한다.
scores = torch.softmax(logits, dim=0)

# 결과를 점수가 높은 순서대로 출력한다.
order = torch.argsort(scores, descending=True)

for idx in order:
    i = int(idx)
    print(f"{scores[i].item():.4f} | {prompts[i]}")
```

### 2.3 프롬프트 실험 설계

수강생은 프롬프트를 무작정 바꾸는 대신 하나의 조건만 바꾸어 실험하는 것이 좋다.

| 실험   | 변경 요소    | 확인할 내용             |
| ---- | -------- | ------------------ |
| 실험 1 | 단어와 문장   | 문장 형태가 점수에 미치는 영향  |
| 실험 2 | 객체와 행동   | 행동 정보가 순위에 미치는 영향  |
| 실험 3 | 배경 정보    | 장면 설명이 관련성에 미치는 영향 |
| 실험 4 | 후보 클래스 수 | 후보 집합 변화에 따른 결과 차이 |

실험 결과는 프롬프트별 점수와 순위를 함께 기록한다. 특정 프롬프트가 높은 점수를 받았다는 사실만으로 실제 정확도가 높아졌다고 판단하지 않는다.

## 3. CLIP을 활용한 이미지 검색

### 3.1 텍스트 기반 이미지 검색

이미지 검색 시스템에서는 미리 여러 이미지의 임베딩을 계산해 저장하고, 사용자가 입력한 검색 문장의 임베딩과 비교할 수 있다.

예를 들어 이미지 데이터베이스에 다음 사진들이 있다고 가정한다.

* 강아지가 공원에서 뛰는 사진

* 고양이가 소파에 앉아 있는 사진

* 자동차가 도로를 달리는 사진

사용자가 `a dog playing in a park`라는 문장을 입력하면 CLIP은 검색 문장과 각 이미지의 유사도를 계산해 순위를 정할 수 있다.

### 3.2 검색 시스템의 구조

![2_2_검색시스템의구조](./image/2_2_검색시스템의구조.png)

### 3.3 이미지 검색과 이미지 분류의 차이

이미지 분류는 일반적으로 각 이미지에 대해 하나의 클래스를 선택한다. 이미지 검색은 데이터베이스의 여러 이미지 가운데 검색 문장과 관련성이 높은 항목을 찾는다.

| 구분    | 이미지 분류    | 이미지 검색         |
| ----- | --------- | -------------- |
| 입력    | 이미지       | 이미지 또는 텍스트 질의  |
| 비교 대상 | 후보 클래스    | 저장된 이미지 집합     |
| 결과    | 예측 클래스    | 순위가 매겨진 이미지 목록 |
| 활용    | 이미지 자동 분류 | 자연어 기반 이미지 검색  |

### 실습 과제

이미지 10장 이상을 준비하고 검색 문장 3개를 작성한다. 검색 결과 상위 3개를 확인한 뒤, 기대한 결과와 다른 이미지가 나온 이유를 분석한다.

## 4. CLIP 결과 평가 

### 4.1 왜 평가가 필요한가?

유사도 계산이 정상적으로 실행된다는 사실과 모델이 정확한 결과를 제공한다는 사실은 다르다.

예를 들어 강아지 사진에 대해 `a photo of a dog`라는 문장이 가장 높은 점수를 받았더라도, 데이터셋 전체에서 강아지를 잘 구분한다고 단정할 수는 없다.

### 4.2 주요 평가 지표

정확도(Accuracy)

Accuracy=정답 수전체 예측 수\text{Accuracy}= \frac{\text{정답 수}}{\text{전체 예측 수}}Accuracy=전체 예측 수정답 수

전체 이미지 가운데 정답 클래스를 올바르게 예측한 비율이다.

클래스별 정확도

특정 클래스에 속한 이미지 가운데 올바르게 예측한 비율이다. 클래스별 정확도를 확인하면 전체 정확도만으로 드러나지 않는 성능 차이를 발견할 수 있다.

예를 들어 강아지 사진은 잘 구분하지만 자동차 사진은 잘 구분하지 못하는 경우가 있을 수 있다.

### 4.3 평가 시 주의사항

* 실제 정답 레이블을 기준으로 평가한다.

* 평가 이미지와 프롬프트 구성을 기록한다.

* 전체 정확도와 클래스별 정확도를 함께 확인한다.

* 오분류 사례를 이미지와 함께 검토한다.

* 같은 평가 데이터에서 프롬프트를 반복적으로 조정했다면, 별도의 검증 데이터로 최종 성능을 확인한다.

### 실습 과제

1. 클래스 3개 이상의 이미지 데이터셋을 준비한다.

2. 각 이미지에 대해 CLIP 예측을 수행한다.

3. 전체 정확도와 클래스별 정확도를 계산한다.

4. 오분류 이미지 3개를 선택해 원인을 분석한다.

## 5. BLIP의 개념과 구조ㄴ

### 5.1 BLIP이란 무엇인가?

BLIP(Bootstrapping Language-Image Pre-training)은 이미지와 텍스트의 관계를 학습해 이미지 이해와 텍스트 생성 작업에 활용할 수 있도록 설계된 비전-언어 모델 프레임워크다.

CLIP과 BLIP 모두 이미지와 텍스트를 다루지만, 이 교재에서는 다음과 같이 구분하면 이해하기 쉽다.

* CLIP: 이미지와 후보 문장이 얼마나 관련 있는지 비교한다.

* BLIP: 이미지 내용을 바탕으로 설명 문장을 생성하거나 질문에 답한다.

BLIP은 모델 구성과 학습 방식에 따라 이미지-텍스트 검색, 이미지 캡셔닝, 시각적 질의응답 등의 작업에 활용할 수 있다.

### 5.2 BLIP의 작동 구조

  ![2_3_BLIP의작동구조](./image/2_3_BLIP의작동구조.png)

위 구조는 이미지 캡셔닝을 설명하기 위한 개념도다. 실제 BLIP은 작업과 체크포인트에 따라 이미지-텍스트 인코더, 이미지 조건부 텍스트 디코더 등의 구성과 연결 방식이 달라질 수 있다.

### 5.3 CLIP과 BLIP을 구분하는 질문

* CLIP에 “이 사진과 가장 관련 있는 문장은 무엇인가?”라고 묻는다면 후보 문장들의 관련성 점수를 비교한다.

* BLIP에 “이 사진에는 무엇이 있는가?”라는 작업을 맡기면 이미지 내용을 바탕으로 설명 문장을 생성할 수 있다.

* BLIP의 질문응답 모델은 이미지와 질문을 함께 입력받아 질문에 대한 답변을 생성한다.

이 차이는 모델을 선택할 때 중요한 기준이 된다.

## 6. BLIP 모델 로드와 이미지 캡셔닝

### 6.1 이미지 캡셔닝이란?

이미지 캡셔닝(Image Captioning)은 이미지의 주요 객체와 장면을 자연어 문장으로 설명하는 작업이다.

입력은 이미지이고 출력은 문장이다. 이미지 분류처럼 미리 정의된 클래스 하나만 선택하는 것이 아니라, 모델이 학습한 언어 표현을 이용해 문장을 생성한다.

### 6.2 코드 실습

```python

from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# 실행 환경에 GPU가 있으면 GPU를 선택한다.
device = "cuda" if torch.cuda.is_available() else "cpu"

# 이미지 캡셔닝용 사전학습 체크포인트를 지정한다.
model_name = "Salesforce/blip-image-captioning-base"

# Processor는 이미지 입력을 모델이 요구하는 형태로 변환한다.
processor = BlipProcessor.from_pretrained(model_name)

# 이미지 캡셔닝 모델과 사전학습 가중치를 불러온다.
model = BlipForConditionalGeneration.from_pretrained(
    model_name
).to(device)

# 추론 모드로 설정한다.
model.eval()

# 실제 이미지 파일을 불러온다.
image = Image.open("./image/cat.jpg").convert("RGB")

# 이미지 픽셀을 모델 입력 텐서로 변환한다.
inputs = processor(
    images=image,
    return_tensors="pt"
)

# 입력 텐서를 모델과 같은 장치로 이동한다.
inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

# 모델이 이미지에 대한 설명 문장을 생성한다.
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=40
    )

# 생성된 토큰 ID를 사람이 읽을 수 있는 문장으로 변환한다.
caption = processor.decode(
    output_ids[0],
    skip_special_tokens=True
)

print("생성된 설명:", caption)
```

### 6.3 코드 해석

`BlipProcessor`는 이미지 입력을 전처리하고, `BlipForConditionalGeneration`은 이미지 특징을 바탕으로 텍스트를 생성한다.

`generate()`는 토큰을 순차적으로 생성하는 함수다. `max_new_tokens=40`은 새롭게 생성할 토큰 수의 상한을 설정한다.

생성된 설명은 이미지의 내용을 정확하게 반영할 수도 있지만, 잘못된 객체나 상황을 포함할 수도 있다. 따라서 모델 출력은 반드시 원본 이미지와 비교해 검증해야 한다.

## 7. 이미지 캡셔닝 결과 분석

### 7.1 생성 결과는 어떻게 평가하는가?

이미지 설명 생성의 결과를 평가할 때는 문장의 자연스러움만 확인해서는 안 된다. 이미지의 핵심 내용이 제대로 반영되었는지 확인해야 한다.

| 평가 항목 | 확인 질문                    |
| ----- | ------------------------ |
| 객체    | 이미지에 실제로 존재하는 객체를 설명했는가? |
| 행동    | 사람이나 동물의 행동을 정확히 설명했는가?  |
| 속성    | 색상, 수량, 크기 등의 묘사가 적절한가?  |
| 배경    | 장면과 배경을 적절하게 설명했는가?      |
| 사실성   | 이미지에 없는 정보를 생성하지 않았는가?   |

### 7.2 이미지별 비교 실습

이미지 5장을 준비하고 같은 캡셔닝 모델로 각각 설명을 생성한다. 이미지 종류는 동물, 음식, 거리, 실내, 자연 풍경 등으로 다양하게 구성한다.

수강생은 생성 문장과 실제 이미지 내용을 비교해 다음을 기록한다.

* 정확하게 설명한 객체

* 누락된 핵심 정보

* 잘못된 객체 또는 속성

* 개선할 수 있는 데이터나 모델 설정

### 7.3 생성 모델의 한계

BLIP은 이미지 특징과 학습된 언어 패턴을 활용해 문장을 생성한다. 하지만 출력 문장이 자연스럽다고 해서 모든 내용이 사실이라는 뜻은 아니다.

예를 들어 사진에 컵이 있지만 컵 안의 내용물이 명확하지 않은 경우, 모델이 실제로 확인하기 어려운 세부 내용을 잘못 설명할 수 있다.

이러한 현상은 시각적 환각(Visual Hallucination)의 사례로 분석할 수 있다.

## 8. CLIP과 BLIP 비교 및  평가 

### 8.1 모델 비교 실습

동일한 이미지를 두 모델에 입력해 결과를 비교한다.

* CLIP: 이미지와 후보 설명문 5개 사이의 관련성 점수를 계산한다.

* BLIP: 이미지에 대한 설명 문장을 생성한다.

예를 들어 CLIP은 다음 후보 중 가장 관련성이 높은 문장을 선택할 수 있다.

* A cat is sleeping on a sofa.

* A dog is running in a park.

* A car is parked on the road.

BLIP은 입력 이미지에 대해 `A cat is sleeping on a sofa.`와 같은 문장을 생성할 수 있다.

두 결과가 같을 수도 있지만, CLIP은 제공된 후보 가운데 하나를 선택하고 BLIP은 학습된 언어 표현을 이용해 문장을 생성한다는 차이가 있다.

### 8.2 비교 결과 보고서

| 항목    | CLIP            | BLIP                |
| ----- | --------------- | ------------------- |
| 입력    | 이미지와 비교할 텍스트    | 이미지                 |
| 출력    | 후보별 관련성 점수      | 생성된 설명 문장           |
| 평가    | 순위, 정확도, 검색 적합성 | 설명의 사실성, 객체·행동의 정확성 |
| 주요 한계 | 후보 문장에 의존       | 생성 오류와 시각적 환각 가능성   |

### 평가 과제

1. CLIP으로 이미지 Zero-shot 분류를 구현한다.

2. 프롬프트를 바꾸어 결과 차이를 기록한다.

3. BLIP으로 이미지 5장의 설명을 생성한다.

4. 두 모델의 입력과 출력, 적용 목적을 비교한다.

5. 잘못된 결과 사례를 최소 2개 분석한다.

# 3. BLIP 응용과 통합 프로젝트


# 이미지 질의응답과 멀티모달 AI 서비스 구현

학습 목표: BLIP VQA를 구현하고 CLIP과 BLIP을 결합해 이미지 검색·설명 서비스를 완성한다.

## 1. BLIP 시각적 질의응답(VQA) 원리 

### 1.1 VQA란 무엇인가?

VQA(Visual Question Answering)는 이미지와 질문을 함께 입력받아 이미지 내용을 바탕으로 답변을 생성하는 작업이다.

이미지 캡셔닝과 VQA의 차이는 질문의 유무와 출력 목적에 있다.

* 이미지 캡셔닝: 이미지 전체를 설명한다.

* VQA: 사용자가 질문한 내용에 초점을 맞춰 답변한다.

예를 들어 이미지가 고양이 두 마리가 소파에 앉아 있는 사진이라면 다음과 같은 질문을 만들 수 있다.

| 질문                               | 기대하는 답변 유형 |
| -------------------------------- | ---------- |
| What animals are in the picture? | 객체         |
| Where are the cats sitting?      | 장소·공간 관계   |
| How many cats are visible?       | 수량         |
| What color is the sofa?          | 속성         |

위 답변은 예시이며, 실제 모델이 반드시 정확하게 생성한다는 의미는 아니다.

### 1.2 VQA의 데이터 흐름
 
 ![3_1_VQA의데이터흐름](./image/3_1_VQA의데이터흐름.png)


### 1.3 캡셔닝과 VQA를 구분해야 하는 이유

캡셔닝 모델은 이미지 전체를 요약하는 설명을 생성하도록 사용한다. VQA 모델은 이미지와 질문을 함께 받아 질문에 대한 답변을 생성하도록 사용한다.

따라서 이미지 설명 생성에 적합한 체크포인트와 VQA에 적합한 체크포인트를 구분해야 한다. 같은 BLIP 계열이라도 모델 클래스와 학습 목적이 다를 수 있다.

### 수업 확인 질문

1. 이미지 캡셔닝과 VQA의 입력 차이는 무엇인가?

2. 수량을 묻는 질문에서 모델이 오류를 낼 수 있는 이유는 무엇인가?

3. VQA 결과를 검증하려면 어떤 정보와 비교해야 하는가?

## 2. BLIP VQA 코드 실습 

### 2.1 모델 준비

VQA 실습에서는 `Salesforce/blip-vqa-base` 체크포인트와 `BlipForQuestionAnswering` 클래스를 사용한다.

```python

import torch
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForQuestionAnswering
)

# GPU가 사용 가능하면 GPU, 아니면 CPU를 선택한다.
device = "cuda" if torch.cuda.is_available() else "cpu"

# 이미지와 질문을 함께 처리하도록 학습된 VQA 체크포인트다.
model_name = "Salesforce/blip-vqa-base"

# 이미지 전처리 및 질문 텍스트 처리를 담당한다.
processor = BlipProcessor.from_pretrained(model_name)

# VQA 작업용 사전학습 모델을 불러온다.
model = BlipForQuestionAnswering.from_pretrained(
    model_name
).to(device)

# 추론 모드로 전환한다.
model.eval()

# 본인의 이미지 경로로 변경한다.
image = Image.open("cat.jpg").convert("RGB")

print("VQA 모델 준비 완료")
```

### 2.2 이미지와 질문을 함께 입력하기

```python


def answer_image_question(image, question):
    """
    이미지와 질문을 입력받아 BLIP VQA 모델의 답변을 생성한다.

    Parameters
    ----------
    image : PIL.Image
        질문 대상 이미지
    question : str
        이미지에 대해 묻고 싶은 자연어 질문

    Returns
    -------
    str
        모델이 생성한 답변
    """

    # 이미지의 색상 모드를 RGB로 맞춘다.
    image = image.convert("RGB")

    # Processor가 이미지와 질문을 모델 입력 텐서로 변환한다.
    inputs = processor(
        images=image,
        text=question,
        return_tensors="pt"
    )

    # 입력을 모델과 동일한 실행 장치로 이동한다.
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # 추론 단계에서는 기울기 계산이 필요하지 않다.
    with torch.no_grad():

        # 모델이 질문에 대한 답변 토큰을 생성한다.
        answer_ids = model.generate(
            **inputs,
            max_new_tokens=20
        )

    # 생성된 토큰 ID를 읽을 수 있는 문장으로 변환한다.
    answer = processor.decode(
        answer_ids[0],
        skip_special_tokens=True
    )

    return answer


# 같은 이미지에 여러 질문을 적용한다.
questions = [
    "What animals are in the picture?",
    "Where are the animals sitting?",
    "How many cats are visible?"
]

for question in questions:
    answer = answer_image_question(image, question)

    print("질문:", question)
    print("답변:", answer)
    print("-" * 40)
```

### 2.3 결과 해석

같은 이미지라도 질문이 달라지면 모델이 생성해야 할 답변도 달라진다.

* 객체 질문은 이미지 안의 대상을 식별하는 데 초점을 둔다.

* 공간 관계 질문은 대상과 배경의 관계를 파악해야 한다.

* 수량 질문은 객체의 개수를 정확히 파악해야 한다.

특히 수량과 세부 공간 관계는 오류가 발생할 수 있다. 결과가 그럴듯하더라도 이미지와 대조해 확인하는 절차가 필요하다.

### 실습 과제

이미지 3장에 대해 질문을 각각 5개씩 작성한다. 답변을 사람이 확인하고 오류를 다음 범주로 구분한다.

* 객체 식별 오류

* 수량 오류

* 속성 오류

* 공간 관계 오류

* 질문 해석 오류

## 3. 생성 결과 평가와 시각적 환각

### 3.1 시각적 환각이란?

시각적 환각은 모델이 이미지에서 확인할 수 없는 객체나 상황을 사실처럼 생성하는 현상을 말한다.

예를 들어 이미지에는 컵만 있는데 모델이 컵 안에 커피가 있다고 설명하거나, 실제로 확인하기 어려운 객체의 색상을 단정할 수 있다.

### 3.2 오류 분석 기준

| 오류 유형    | 예시                  | 분석 관점   |
| -------- | ------------------- | ------- |
| 객체 오류    | 고양이를 강아지라고 설명       | 객체 식별   |
| 수량 오류    | 객체 2개를 3개라고 답변      | 개수 인식   |
| 속성 오류    | 검은색 물체를 흰색이라고 설명    | 시각적 속성  |
| 공간 오류    | 책상 위 물체를 바닥에 있다고 설명 | 객체 간 관계 |
| 근거 없는 생성 | 보이지 않는 행동을 설명       | 사실성 검증  |

### 3.3 생성 결과 평가 실습

수강생은 BLIP이 생성한 캡션과 VQA 답변을 표로 정리한다.

| 이미지   | 작업  | 모델 결과  | 실제 이미지와의 차이 | 오류 유형   |
| ----- | --- | ------ | ----------- | ------- |
| 이미지 A | 캡셔닝 | 생성된 설명 | 검토 후 기록     | 객체·속성 등 |
| 이미지 A | VQA | 생성된 답변 | 검토 후 기록     | 수량·공간 등 |
| 이미지 B | 캡셔닝 | 생성된 설명 | 검토 후 기록     | 해당 유형   |
| 이미지 B | VQA | 생성된 답변 | 검토 후 기록     | 해당 유형   |

평가 시에는 문장의 자연스러움과 사실성을 구분해야 한다. 자연스러운 문장도 이미지의 사실과 일치하지 않을 수 있다.

### 3.4 개선 방향

* 입력 이미지의 품질을 확인한다.

* 질문을 명확하고 구체적으로 작성한다.

* 여러 종류의 이미지에서 반복적으로 평가한다.

* 생성 결과를 사람이 확인할 수 있는 검증 절차를 마련한다.

* 실제 서비스에서는 모델의 한계와 오류 가능성을 사용자에게 알린다.

## 4. CLIP과 BLIP 통합 설계

### 4.1 통합 서비스가 필요한 이유

CLIP과 BLIP은 서로 다른 역할을 담당한다. 두 모델을 연결하면 검색과 설명 생성을 한 서비스에서 제공할 수 있다.

예를 들어 사용자가 사진을 업로드하면 다음 작업을 수행한다.

1. CLIP이 이미지와 후보 문장의 유사도를 계산한다.

2. 유사도가 높은 문장을 검색 결과로 표시한다.

3. BLIP이 이미지에 대한 설명을 생성한다.

4. 사용자에게 검색 결과와 생성 설명을 함께 제공한다.

### 4.2 시스템 구조도

![3_2_시스템구조도](./image/3_2_시스템구조도.png)


### 4.3 기능을 분리해야 하는 이유

CLIP의 유사도 계산과 BLIP의 텍스트 생성은 서로 다른 작업이다. 한 함수에 모든 코드를 넣기보다 다음과 같이 기능을 분리하는 편이 유지보수에 유리하다.

* `clip_similarity_for_image()`: 이미지와 후보 텍스트의 유사도를 계산한다.

* `blip_caption()`: 이미지 설명을 생성한다.

* `main()`: 입력과 결과 표시를 담당한다.

이렇게 분리하면 CLIP의 후보 문장 구성이나 BLIP의 생성 설정을 독립적으로 변경할 수 있다.

## 5. 통합 서비스 구현

### 5.1 모델 로드

통합 프로젝트에서는 CLIP과 BLIP을 각각 불러온다. 모델을 한 번만 로드하고 여러 입력에서 재사용해야 불필요한 로딩 시간을 줄일 수 있다.

```python

import torch
import torch.nn.functional as F

from transformers import (
    CLIPModel,
    CLIPProcessor,
    BlipProcessor,
    BlipForConditionalGeneration
)

# GPU가 있으면 GPU를 사용한다.
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP: 이미지와 텍스트의 관련성을 비교한다.
clip_name = "openai/clip-vit-base-patch32"

clip_processor = CLIPProcessor.from_pretrained(clip_name)
clip_model = CLIPModel.from_pretrained(
    clip_name
).to(device)

clip_model.eval()

# BLIP: 이미지 설명을 생성한다.
blip_name = "Salesforce/blip-image-captioning-base"

blip_processor = BlipProcessor.from_pretrained(blip_name)
blip_model = BlipForConditionalGeneration.from_pretrained(
    blip_name
).to(device)

blip_model.eval()

print("CLIP·BLIP 모델 로드 완료")
```

### 5.2 CLIP 검색 함수

```python

def clip_similarity_for_image(image, candidate_texts):
    """
    이미지와 여러 후보 텍스트의 관련성을 비교한다.

    반환값은 점수가 높은 순서대로 정렬한
    (후보 문장, 상대적 점수) 목록이다.
    """

    # 이미지와 텍스트를 CLIP 입력 형식으로 전처리한다.
    inputs = clip_processor(
        images=image.convert("RGB"),
        text=candidate_texts,
        return_tensors="pt",
        padding=True
    )

    # 입력 텐서를 모델 실행 장치로 옮긴다.
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        # 이미지와 텍스트의 관련성 로짓을 계산한다.
        outputs = clip_model(**inputs)
        logits = outputs.logits_per_image[0]

        # 후보 문장 집합 안에서 상대적인 점수 분포를 계산한다.
        scores = torch.softmax(logits, dim=0)

    # 문장과 점수를 연결하고 높은 점수부터 정렬한다.
    ranked_results = sorted(
        zip(
            candidate_texts,
            scores.detach().cpu().tolist()
        ),
        key=lambda item: item[1],
        reverse=True
    )

    return ranked_results
```

### 5.3 BLIP 설명 생성 함수

```python

def blip_caption(image, max_new_tokens=40):
    """
    입력 이미지의 설명 문장을 생성한다.

    max_new_tokens는 새로 생성할 토큰 수의 상한이다.
    """

    # 이미지를 BLIP의 입력 형식으로 전처리한다.
    inputs = blip_processor(
        images=image.convert("RGB"),
        return_tensors="pt"
    )

    # 모델과 같은 장치로 이동한다.
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # 추론을 실행한다.
    with torch.no_grad():
        output_ids = blip_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

    # 토큰 ID를 자연어 설명으로 복원한다.
    caption = blip_processor.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    return caption
```

### 5.4 두 기능 연결

```python

from PIL import Image

# 실제 이미지 파일을 불러온다.
image = Image.open("cat.jpg").convert("RGB")

# CLIP이 비교할 후보 문장을 준비한다.
candidate_texts = [
    "A photo of a cat.",
    "A photo of a dog.",
    "A photo of a car.",
    "A photo of a bird."
]

# BLIP으로 이미지 설명을 생성한다.
caption = blip_caption(image)

# CLIP으로 이미지와 후보 문장의 관련성을 계산한다.
ranked_results = clip_similarity_for_image(
    image,
    candidate_texts
)

print("=== BLIP 이미지 설명 ===")
print(caption)

print("\\n=== CLIP 검색 결과 ===")
for rank, (text, score) in enumerate(
    ranked_results,
    start=1
):
    print(f"{rank}위 | 점수={score:.4f} | {text}")


# 이 코드는 통합 서비스의 핵심 추론 부분이다. 사용자 인터페이스를 연결하기 전에 각 함수가 정상적으로 실행되는지, 서로 다른 이미지에서도 적절한 결과를 반환하는지 확인한다.
```

## 6. 사용자 인터페이스 구현

### 6.1 Streamlit을 사용하는 이유

Streamlit은 Python 코드로 간단한 웹 인터페이스를 만들 수 있는 라이브러리다. 별도의 프런트엔드 개발을 최소화하면서 이미지 업로드, 텍스트 입력, 결과 표시 기능을 구현할 수 있다.

### 6.2 화면 구성

서비스 화면에는 다음 요소를 배치한다.

1. 이미지 업로드 버튼

2. CLIP 후보 문장 입력란

3. 이미지 미리보기

4. BLIP 생성 설명

5. CLIP 유사도 순위

### 6.3 Streamlit 앱 예시

다음 코드는 앞서 정의한 CLIP·BLIP 모델 로딩 및 추론 함수를 `app.py` 한 파일로 구성한 예시다. 이 교재에서는 모델 로딩 코드를 파일 상단에 한 번만 배치하고, 사용자가 입력을 변경할 때 추론 함수를 재사용하도록 설명한다.

# app.py
# 실행 명령: streamlit run app.py

import streamlit as st
import torch
from PIL import Image
from transformers import (
    CLIPModel,
    CLIPProcessor,
    BlipProcessor,
    BlipForConditionalGeneration
)

# --------------------------------------------------
# 1. 실행 장치 설정
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# 2. 모델 로딩
# --------------------------------------------------

# Streamlit은 입력이 변경될 때 앱을 다시 실행할 수 있다.
# cache_resource를 사용하면 같은 세션 환경에서 모델 로딩을 재사용할 수 있다.
@st.cache_resource
def load_models():

    clip_name = "openai/clip-vit-base-patch32"
    clip_processor = CLIPProcessor.from_pretrained(clip_name)
    clip_model = CLIPModel.from_pretrained(
        clip_name
    ).to(device)
    clip_model.eval()

    blip_name = "Salesforce/blip-image-captioning-base"
    blip_processor = BlipProcessor.from_pretrained(blip_name)
    blip_model = BlipForConditionalGeneration.from_pretrained(
        blip_name
    ).to(device)
    blip_model.eval()

    return (
        clip_processor,
        clip_model,
        blip_processor,
        blip_model
    )


# --------------------------------------------------
# 3. BLIP 캡셔닝 함수
# --------------------------------------------------

def make_caption(image, processor, model):

    inputs = processor(
        images=image.convert("RGB"),
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=40
        )

    return processor.decode(
        output_ids[0],
        skip_special_tokens=True
    )


# --------------------------------------------------
# 4. CLIP 유사도 함수
# --------------------------------------------------

def rank_texts(image, texts, processor, model):

    inputs = processor(
        images=image.convert("RGB"),
        text=texts,
        return_tensors="pt",
        padding=True
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

        # 이미지 한 장과 후보 문장 사이의 로짓을 가져온다.
        logits = outputs.logits_per_image[0]

        # 후보 집합 안에서 상대적인 점수 분포를 계산한다.
        scores = torch.softmax(logits, dim=0)

    ranked = sorted(
        zip(texts, scores.cpu().tolist()),
        key=lambda item: item[1],
        reverse=True
    )

    return ranked


# --------------------------------------------------
# 5. 웹 화면 구성
# --------------------------------------------------

st.title("CLIP · BLIP 이미지 분석 서비스")
st.write(
    "CLIP으로 이미지와 후보 문장의 관련성을 비교하고, "
    "BLIP으로 이미지 설명을 생성한다."
)

uploaded_file = st.file_uploader(
    "분석할 이미지를 업로드한다.",
    type=["jpg", "jpeg", "png", "webp"]
)

text_input = st.text_area(
    "CLIP 후보 문장을 입력한다. 문장마다 한 줄씩 작성한다.",
    value=(
        "A photo of a cat.\\n"
        "A photo of a dog.\\n"
        "A photo of a car."
    )
)

if uploaded_file is not None:

    # 업로드된 파일을 RGB 이미지로 변환한다.
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="업로드한 이미지",
        use_container_width=True
    )

    # 빈 줄을 제거해 후보 문장 목록을 만든다.
    candidate_texts = [
        line.strip()
        for line in text_input.splitlines()
        if line.strip()
    ]

    if st.button("이미지 분석 실행"):

        if not candidate_texts:
            st.warning("비교할 후보 문장을 한 개 이상 입력한다.")

        else:
            with st.spinner("모델을 실행하고 있다..."):

                (
                    clip_processor,
                    clip_model,
                    blip_processor,
                    blip_model
                ) = load_models()

                # BLIP으로 이미지 설명을 생성한다.
                caption = make_caption(
                    image,
                    blip_processor,
                    blip_model
                )

                # CLIP으로 후보 문장의 순위를 계산한다.
                results = rank_texts(
                    image,
                    candidate_texts,
                    clip_processor,
                    clip_model
                )

            st.subheader("BLIP 이미지 설명")
            st.write(caption)

            st.subheader("CLIP 관련성 순위")

            for rank, (text, score) in enumerate(
                results,
                start=1
            ):
                st.write(
                    f"{rank}위 | 점수 {score:.4f} | {text}"
                )

### 실행 방법

터미널에서 파일이 있는 디렉터리로 이동한 뒤 다음 명령을 실행한다.

Bash

```
uv add streamlit torch torchvision transformers pillow
streamlit run app.py
```

최초 실행 시 모델 다운로드가 필요하다. GPU 메모리가 부족하거나 다운로드 시간이 길면 모델 로딩 및 추론에 실패할 수 있으므로 수업 전에 환경을 점검해야 한다.

### 실습 과제

* 이미지 파일을 업로드할 수 있도록 구현한다.

* 후보 문장을 수정한 뒤 CLIP 순위가 바뀌는지 확인한다.

* BLIP 생성 설명을 화면에 표시한다.

* 입력 이미지가 없거나 후보 문장이 비어 있는 경우를 처리한다.

## 7. 통합 서비스 테스트와 개선

### 7.1 기능 테스트

서비스를 구현한 뒤에는 기능이 정상적으로 실행되는지 확인한다.

| 테스트 항목   | 검증 방법       | 기대 결과           |
| -------- | ----------- | --------------- |
| 이미지 업로드  | JPG 파일 선택   | 이미지 미리보기 표시     |
| CLIP 분석  | 후보 문장 3개 입력 | 관련성 점수와 순위 출력   |
| BLIP 캡셔닝 | 이미지 분석 실행   | 설명 문장 생성        |
| 후보 변경    | 문장 표현 수정    | 관련성 순위 재계산      |
| 입력 오류    | 빈 후보 문장     | 안내 메시지 출력       |
| 다양한 이미지  | 동물·풍경·사물 입력 | 각 이미지에 대한 결과 표시 |

### 7.2 성능과 정확도 평가

실제 프로젝트에서는 모델이 정상적으로 실행되는지만 평가하지 않는다.

* CLIP은 정답 레이블이 있는 데이터셋을 사용해 분류 정확도와 클래스별 성능을 확인한다.

* 이미지 검색은 관련 이미지가 상위 순위에 나타나는지 확인한다.

* BLIP은 설명에 포함된 객체, 행동, 속성이 실제 이미지와 일치하는지 검토한다.

* 추론 시간과 메모리 사용량을 측정해 서비스 환경에 적합한지 판단한다.

### 7.3 개선 아이디어

1. 이미지 임베딩을 미리 계산해 저장하고 검색 속도를 높인다.

2. CLIP 후보 문장을 사용자 입력으로 받도록 확장한다.

3. BLIP 설명 생성 결과를 검색 문장 후보로 활용하는 실험을 수행한다.

4. 이미지 설명과 VQA 답변을 한 화면에서 비교한다.

5. 생성 결과의 오류 가능성을 표시하고 사람이 검토할 수 있도록 한다.

## 8. 프로젝트 발표와 최종 평가 

### 8.1 프로젝트 발표 구성

발표 자료는 다음 순서로 작성하도록 지도한다.

1. 프로젝트 목적 및 해결하려는 문제

2. CLIP과 BLIP을 선택한 이유

3. 전체 시스템 구조도

4. 이미지 입력과 전처리 방식

5. CLIP 검색 결과 및 유사도 해석

6. BLIP 생성 설명과 오류 분석

7. 테스트 결과 및 한계점

8. 향후 개선 방향

### 8.2 최종 평가 기준

| 평가 항목   | 배점   | 세부 기준                   |
| ------- | ---- | ----------------------- |
| 개념 이해   | 20점  | CLIP과 BLIP의 구조·기능 차이 설명 |
| 코드 구현   | 25점  | 모델 로딩, 추론, 결과 출력        |
| 결과 분석   | 20점  | 유사도 및 생성 결과의 정확성 분석     |
| 통합 프로젝트 | 25점  | 검색과 설명 생성 기능 연결         |
| 발표·문서화  | 10점  | 실행 방법, 한계, 개선 방향 정리     |
| 합계      | 100점 |                         |

### 최종 학습 확인

수강생이 다음 질문에 자신의 말로 답할 수 있다면 핵심 학습 목표를 달성한 것이다.

* CLIP은 이미지와 텍스트를 어떻게 비교하는가?

* 공유 임베딩 공간은 왜 필요한가?

* 코사인 유사도와 소프트맥스 점수는 어떻게 다른가?

* BLIP의 이미지 캡셔닝과 VQA는 무엇이 다른가?

* 두 모델을 하나의 서비스에 결합하면 어떤 기능을 제공할 수 있는가?

* 모델이 생성한 설명이 틀릴 수 있다는 점을 서비스에서 어떻게 다룰 것인가?

---

# CLIP·BLIP 멀티모달 AI 

개념 설명 → 구조와 작동 원리 → 코드 실습 → 실행 결과 해석 → 학습 확인 순서로 진행한다.

수업의 핵심은 CLIP과 BLIP의 차이를 명확하게 이해하는 데 있다.

* CLIP: 이미지와 텍스트를 벡터로 변환하고, 두 데이터가 얼마나 의미적으로 유사한지 계산한다.

* BLIP: 이미지 정보를 바탕으로 설명 문장을 생성하거나 이미지에 관한 질문에 답한다.

# 1. 멀티모달 AI와 CLIP 기초


# 이미지와 텍스트를 연결하는 원리 이해

학습 목표: 임베딩, 인코더, 공유 임베딩 공간, 코사인 유사도를 이해하고 CLIP으로 직접 계산한다.

## 1. 멀티모달 AI 개요 
### 1.1 모달리티란 무엇인가?

모달리티(Modality)는 정보가 표현되는 형태 또는 데이터 유형을 의미한다.

| 모달리티   | 데이터 예시        | AI의 주요 처리 대상   |
| ------ | ------------- | -------------- |
| 텍스트    | 문장, 문서, 질문    | 단어와 문장의 의미     |
| 이미지    | 사진, 그림, 의료 영상 | 객체, 색상, 형태, 장면 |
| 오디오    | 음성, 음악, 환경음   | 음성 내용, 소리의 특징  |
| 비디오    | 동영상, CCTV 영상  | 객체와 시간에 따른 행동  |
| 센서 데이터 | 온도, 압력, 위치    | 수치 패턴과 상태 변화   |

하나의 데이터 유형만 처리하면 단일 모달 AI이고, 서로 다른 유형의 정보를 함께 활용하면 멀티모달 AI라고 할 수 있다.

예를 들어 이미지 분류 모델은 사진을 입력받아 `고양이`라는 클래스를 출력한다. 멀티모달 모델은 사진뿐 아니라 텍스트 설명이나 질문까지 함께 활용할 수 있다.

### 1.2 단일 모달 AI와 멀티모달 AI 비교

![image.png](attachment:image.png)

### 1.3 멀티모달 AI의 실제 활용 사례

* 이미지 검색: “눈 덮인 산과 호수가 있는 사진”이라는 문장으로 사진을 검색한다.

* 의료 영상 보조: 영상과 판독 관련 텍스트의 연관성을 분석한다. 실제 진단에는 별도의 임상 검증이 필요하다.

* 상품 검색: “검은색 가죽 가방”처럼 자연어로 상품 이미지를 검색한다.

* 이미지 설명 생성: 사진을 입력하면 “소파 위에 고양이 두 마리가 앉아 있다”와 같은 설명을 생성한다.

* 시각적 질의응답: 사진을 보여 주고 “테이블 위에 무엇이 있는가?”라고 질문한다.

여기서 첫 세 가지는 CLIP의 활용 방식과 밀접하고, 이미지 설명과 질의응답은 BLIP의 대표적인 활용 방식이다.

### 수업 확인 질문

1. 이미지 분류 AI와 멀티모달 AI의 차이는 무엇인가?

2. 이미지와 텍스트를 함께 처리하면 어떤 문제를 해결할 수 있는가?

3. 자연어로 사진을 검색하려면 이미지와 문장이 어떤 방식으로 연결되어야 하는가?

<details>
<summary>정답 보기</summary>


</details>

## 2. 임베딩 기초 

### 2.1 임베딩이란 무엇인가?

임베딩(Embedding)은 텍스트나 이미지 같은 데이터를 숫자로 구성된 벡터로 표현한 것이다.

컴퓨터는 이미지의 의미를 사람처럼 직접 이해하지 못한다. 모델은 학습 과정에서 데이터의 특징을 추출해 수치 벡터로 표현한다.

예를 들어 설명을 위한 가상의 임베딩을 다음과 같이 생각할 수 있다.

| 데이터     | 가상 임베딩            |
| ------- | ----------------- |
| 고양이 사진  | `[0.8, 0.2, 0.1]` |
| 고양이 설명문 | `[0.7, 0.3, 0.1]` |
| 자동차 설명문 | `[0.1, 0.2, 0.9]` |

위 숫자는 원리를 설명하기 위한 예시이며 실제 CLIP의 출력값은 아니다.

고양이 사진과 고양이 설명문은 벡터 방향이 비슷하고, 자동차 설명문은 상대적으로 다른 방향을 갖는다고 가정할 수 있다.

### 2.2 벡터의 차원

벡터는 여러 숫자를 순서대로 나열한 자료 구조다.

* 2차원 벡터: `[0.8, 0.2]`

* 3차원 벡터: `[0.8, 0.2, 0.1]`

* 고차원 임베딩: 수백 개 이상의 숫자로 데이터의 특징을 표현한다.

CLIP의 `openai/clip-vit-base-patch32` 모델은 이미지와 텍스트를 각각 512차원 임베딩으로 표현한다. 즉, 이미지 하나는 512개의 숫자로 표현되는 벡터로 변환될 수 있다.

### 2.3 임베딩 공간 시각화

![고양이 사진](./image/1_2_임베딩공간시각화.png)

중요한 점은 벡터가 비슷하다는 의미는 사람이 정한 규칙으로 숫자를 직접 배정했다는 뜻이 아니라, 모델이 데이터에서 특징을 학습했다는 뜻이라는 것이다.

### 2.4 임베딩과 원본 데이터의 차이

| 구분    | 원본 데이터        | 임베딩                    |
| ----- | ------------- | ---------------------- |
| 이미지   | 픽셀로 구성된 사진    | 이미지 특징을 표현하는 숫자 벡터     |
| 텍스트   | 사람이 읽는 문장     | 텍스트 의미와 특징을 표현하는 숫자 벡터 |
| 주요 용도 | 사람이 내용을 직접 확인 | 검색, 유사도 계산, 분류 등에 활용   |

### 2.5 실습: 벡터 유사도 계산

```python

import torch
import torch.nn.functional as F

# 원리를 이해하기 위한 가상의 임베딩 벡터다.
# 실제 CLIP 모델이 생성한 값은 아니다.
cat_image = torch.tensor([0.8, 0.2, 0.1])
cat_text = torch.tensor([0.7, 0.3, 0.1])
car_text = torch.tensor([0.1, 0.2, 0.9])

# 두 벡터의 코사인 유사도를 계산한다.
# 값이 1에 가까울수록 두 벡터의 방향이 비슷하다.
similarity_cat = F.cosine_similarity(
    cat_image.unsqueeze(0),
    cat_text.unsqueeze(0)
)

similarity_car = F.cosine_similarity(
    cat_image.unsqueeze(0),
    car_text.unsqueeze(0)
)

print("고양이 이미지와 고양이 문장:", similarity_cat.item())
print("고양이 이미지와 자동차 문장:", similarity_car.item())

#  이 코드는 임베딩과 유사도의 원리를 학습하기 위한 간단한 예제다. 실제 CLIP에서는 사람이 벡터를 지정하는 대신 이미지 인코더와 텍스트 인코더가 입력 데이터에서 벡터를 생성한다.
```

## 3. CLIP의 모델 구조 

### 3.1 CLIP이란 무엇인가?

CLIP(Contrastive Language–Image Pre-training)은 이미지와 텍스트 사이의 의미적 관계를 학습한 비전-언어 모델이다.

일반적인 이미지 분류 모델은 학습 데이터에 정의된 클래스에 맞춰 예측하도록 학습한다. 반면 CLIP은 이미지와 텍스트를 연결하는 방식으로 학습하므로, 후보 텍스트를 바꾸어 새로운 분류 문제에 적용할 수 있다.

CLIP의 핵심 구성 요소는 다음과 같다.

![1_3_CLIP의핵심구성요소](./image/1_3_CLIP의핵심구성요소.png)

### 3.2 이미지 인코더

이미지 인코더는 사진을 입력받아 시각적 특징을 추출한다.

CLIP의 대표적인 이미지 인코더는 Vision Transformer(ViT) 또는 ResNet 계열을 사용할 수 있다.  `openai/clip-vit-base-patch32`는 ViT 기반 모델이다.

이미지 인코더가 수행하는 주요 과정은 다음과 같다.

1. 이미지를 모델이 요구하는 크기와 픽셀 범위로 전처리한다.

2. 이미지를 패치 단위로 나누어 시각적 정보를 표현한다.

3. Transformer가 패치 간 관계를 처리한다.

4. 시각적 특징을 추출한다.

5. 투영 계층을 통해 텍스트와 비교할 수 있는 임베딩을 생성한다.

### 3.3 텍스트 인코더

텍스트 인코더는 문장을 토큰으로 변환하고 문맥적 특징을 추출한다.

예를 들어 `A cat is sitting on a sofa.`라는 문장은 토큰화 과정을 거쳐 숫자 ID로 표현된다. Transformer 기반 텍스트 인코더가 이 토큰의 문맥을 처리한 후, 최종적으로 텍스트 임베딩을 생성한다.

여기서 토큰 ID 자체가 임베딩은 아니다. 토큰 ID는 사전에 등록된 토큰을 가리키는 번호이며, 인코더는 이를 이용해 문맥적 특징을 계산한다.

### 3.4 공유 임베딩 공간

공유 임베딩 공간(Shared Embedding Space)은 이미지와 텍스트를 같은 벡터 공간에서 비교할 수 있도록 표현한 공간이다.

예를 들어 이미지 인코더가 강아지 사진을 벡터로 표현하고, 텍스트 인코더가 `A dog playing in a park.`를 벡터로 표현한다고 가정한다. CLIP은 두 표현이 의미적으로 관련되도록 학습되어 있다.

다만 서로 다른 인코더의 출력이 자동으로 비교 가능한 것은 아니다. 학습 과정에서 두 표현을 정렬하도록 학습했다는 점이 중요하다.

### 3.5 대조학습(Contrastive Learning)

CLIP은 이미지와 텍스트의 쌍을 활용한 대조학습을 통해 시각적 표현과 언어적 표현을 연결한다.

* 긍정 쌍: 사진과 그 사진에 대응하는 설명문

* 부정 쌍: 해당 사진과 의미적으로 관련이 적은 다른 설명문

학습 과정에서는 긍정 쌍의 관련성 점수를 높이고, 배치 안의 다른 이미지·텍스트 쌍과는 구별할 수 있도록 학습한다.

이 교재에서는 다음과 같이 설명하면 이해하기 쉽다.

> CLIP은 사진에 정답 레이블만 붙이는 방식이 아니라, 사진과 설명문이 서로 잘 맞는지를 학습한다. 그 결과 학습에 직접 사용하지 않은 새로운 후보 문장도 이미지와 비교할 수 있다.

### 수업 확인 질문

1. 이미지 인코더와 텍스트 인코더는 각각 어떤 역할을 하는가?

2. 공유 임베딩 공간이 필요한 이유는 무엇인가?

3. 긍정 쌍과 부정 쌍은 대조학습에서 어떤 역할을 하는가?

4. 토큰 ID와 임베딩 벡터는 어떻게 다른가?

<details>
<summary>정답 보기</summary>


</details>

## 4. 이미지 전처리와 텍스트 토큰화
### 4.1 이미지 전처리가 필요한 이유

딥러닝 모델은 입력 이미지의 크기와 채널 구성, 픽셀값 범위가 일정해야 한다.

원본 사진의 크기가 각각 800 X 600, 1024 X 768, 400 X 400 이라면 이를 그대로 한 배치에 넣기 어렵다. 따라서 모델이 기대하는 입력 형식에 맞춰 전처리해야 한다.

CLIP의 전처리에는 일반적으로 다음 작업이 포함된다.

* 이미지의 색상 모드를 RGB로 맞춘다.

* 모델의 학습 설정에 맞춰 이미지 크기를 조정하고 필요한 영역을 자른다.

* 픽셀값을 텐서로 변환한다.

* 사전학습 때 사용한 평균과 표준편차를 기준으로 정규화한다.

`CLIPProcessor`를 사용하면 모델에 맞는 전처리를 수행할 수 있다. 따라서 직접 전처리 함수를 작성하기 전에는 Processor가 수행하는 작업을 먼저 확인하는 것이 좋다.

### 4.2 텍스트 토큰화

텍스트는 모델이 처리할 수 있는 숫자 ID의 시퀀스로 변환된다.

예를 들어 다음 문장을 입력한다고 가정한다.

`A photo of a cat.`

토큰화 과정에서는 문장이 여러 토큰으로 분리되고, 각 토큰이 어휘 사전에 정의된 ID로 변환된다. 실제 토큰 분할 방식은 토크나이저에 따라 달라질 수 있다.

### 4.3 코드 실습: Processor의 출력 확인

```python
# ============================================================
# 4.3 Processor의 출력 확인
# 실습 모델: CLIP
# ============================================================

import torch
from PIL import Image
from IPython.display import display
from urllib.request import urlopen
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel


# ------------------------------------------------------------
# 1. 모델 설정
# ------------------------------------------------------------

# Hugging Face에 공개된 사전 학습 CLIP 모델이다.
# 처음 실행할 때 모델과 Processor 파일을 다운로드한다.
MODEL_NAME = "openai/clip-vit-base-patch32"

# CPU와 GPU 중 사용할 장치를 자동으로 선택한다.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

print("=" * 60)
print("1. 실행 환경")
print("=" * 60)
print("PyTorch 버전:", torch.__version__)
print("사용 장치:", device)


# ------------------------------------------------------------
# 2. CLIP Processor와 Model 불러오기
# ------------------------------------------------------------

# CLIPProcessor:
# 이미지 전처리와 텍스트 토큰화를 담당한다.
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# CLIPModel:
# Processor가 변환한 입력을 받아 이미지와 텍스트를 분석한다.
model = CLIPModel.from_pretrained(MODEL_NAME)

# 모델을 CPU 또는 GPU에 배치한다.
model = model.to(device)

# 학습이 아닌 추론 모드로 설정한다.
model.eval()

print("\nProcessor와 Model을 정상적으로 불러왔다.")


# ------------------------------------------------------------
# 3. 테스트 이미지 생성
# ------------------------------------------------------------

# 실제 강아지 사진을 인터넷에서 가져온다.
image_url = "https://images.dog.ceo/breeds/maltese/n02085936_100.jpg"

# URL에서 이미지 데이터를 읽어 PIL Image 객체로 변환한다.
with urlopen(image_url, timeout=30) as response:
    image = Image.open(BytesIO(response.read())).convert("RGB")

# 이미지의 기본 정보를 확인한다.
print("\n" + "=" * 60)
print("2. 원본 이미지 정보")
print("=" * 60)
print("자료형:", type(image))
print("이미지 크기:", image.size)
print("색상 모드:", image.mode)

# 실제 강아지 사진을 화면에 표시한다.
display(image)


# ------------------------------------------------------------
# 4. 텍스트 후보 준비
# ------------------------------------------------------------

texts = [
    "a photo of a dog",
    "a photo of a cat",
    "a photo of a car"
]

print("\n" + "=" * 60)
print("3. 입력 텍스트")
print("=" * 60)

for index, text in enumerate(texts):
    print(f"{index}: {text}")


# ------------------------------------------------------------
# 5. Processor 실행
# ------------------------------------------------------------

# Processor는 이미지와 텍스트를 각각 모델 입력 형식으로 변환한다.
#
# text:
#   토큰화할 텍스트 목록
#
# images:
#   전처리할 PIL 이미지
#
# return_tensors="pt":
#   결과를 PyTorch Tensor 형태로 반환
#
# padding=True:
#   텍스트 길이가 서로 다를 때 길이를 맞춘다.
inputs = processor(
    text=texts,
    images=image,
    return_tensors="pt",
    padding=True
)


# ------------------------------------------------------------
# 6. Processor 출력의 자료형과 키 확인
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("4. Processor 출력 확인")
print("=" * 60)

print("전체 자료형:", type(inputs))
print("출력 키:", list(inputs.keys()))

# 일반적으로 CLIPProcessor는 다음 항목을 반환한다.
# input_ids: 텍스트 토큰 ID
# attention_mask: 실제 토큰과 패딩 토큰을 구분하는 마스크
# pixel_values: 전처리된 이미지 픽셀 텐서

for key, value in inputs.items():
    print(f"\n항목 이름: {key}")
    print("자료형:", type(value))
    print("텐서 크기:", tuple(value.shape))
    print("데이터 자료형:", value.dtype)
    print("장치:", value.device)


# ------------------------------------------------------------
# 7. input_ids 확인
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("5. input_ids 확인")
print("=" * 60)

# input_ids는 텍스트를 토큰 ID로 변환한 결과다.
print(inputs["input_ids"])

# 토큰 ID를 다시 사람이 읽을 수 있는 토큰으로 변환한다.
tokens = processor.tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0].tolist()
)

print("\n첫 번째 문장의 토큰:")
print(tokens)


# ------------------------------------------------------------
# 8. attention_mask 확인
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("6. attention_mask 확인")
print("=" * 60)

# 1: 실제 입력 토큰
# 0: 패딩 토큰
print(inputs["attention_mask"])


# ------------------------------------------------------------
# 9. pixel_values 확인
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("7. pixel_values 확인")
print("=" * 60)

pixel_values = inputs["pixel_values"]

print("이미지 텐서 크기:", tuple(pixel_values.shape))
print("최솟값:", pixel_values.min().item())
print("최댓값:", pixel_values.max().item())

# pixel_values에는 원본 이미지가 아니라
# CLIP이 처리할 수 있도록 전처리된 픽셀 값이 저장된다.
print("\n첫 번째 이미지 텐서의 일부 값:")
print(pixel_values[0, :, :3, :3])


# ------------------------------------------------------------
# 10. 모델 입력 장치로 이동
# ------------------------------------------------------------

# Processor는 일반적으로 CPU에 있는 텐서를 반환한다.
# 모델이 GPU에 있다면 입력도 GPU로 옮겨야 한다.
inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


# ------------------------------------------------------------
# 11. CLIP 모델 실행
# ------------------------------------------------------------

# 추론 단계에서는 역전파를 위한 그래디언트를 계산하지 않는다.
with torch.no_grad():
    outputs = model(**inputs)

print("\n" + "=" * 60)
print("8. CLIP 모델 실행 결과")
print("=" * 60)

# logits_per_image:
# 이미지 하나와 각 텍스트 후보 사이의 비교 점수
print("이미지-텍스트 점수 텐서 크기:",
      tuple(outputs.logits_per_image.shape))

# logits_per_text:
# 각 텍스트와 이미지 사이의 비교 점수
print("텍스트-이미지 점수 텐서 크기:",
      tuple(outputs.logits_per_text.shape))

print("\n이미지-텍스트 점수:")
print(outputs.logits_per_image)


# ------------------------------------------------------------
# 12. 후보별 상대 점수와 순위 확인
# ------------------------------------------------------------

# softmax는 현재 후보 목록 안에서 상대 점수를 계산한다.
scores = outputs.logits_per_image[0]
probabilities = scores.softmax(dim=0)

print("\n" + "=" * 60)
print("9. 텍스트 후보별 상대 점수")
print("=" * 60)

# 점수가 높은 후보부터 정렬한다.
ranking = sorted(
    zip(texts, probabilities.tolist()),
    key=lambda item: item[1],
    reverse=True
)

for rank, (text, score) in enumerate(ranking, start=1):
    print(f"{rank}위 | {text} | 상대 점수: {score:.4f}")

print("\n실습 완료")
```

예상되는 출력 항목은 `pixel_values`, `input_ids`, `attention_mask` 등이다. 이미지 크기와 텍스트 길이에 따라 실제 텐서 크기는 달라진다.

| 입력 항목            | 의미                     |
| ---------------- | ---------------------- |
| `pixel_values`   | 전처리된 이미지 픽셀 텐서         |
| `input_ids`      | 텍스트 토큰의 ID             |
| `attention_mask` | 실제 토큰과 패딩 토큰을 구분하는 마스크 |

### 4.4 배치 처리의 의미

배치(Batch)는 여러 입력을 묶어 한 번에 처리하는 방식이다.

이미지 1장과 문장 5개를 비교할 때는 이미지 배치 크기가 1이고 텍스트 배치 크기는 5가 될 수 있다. 이미지 10장과 문장 5개를 함께 처리하면 결과적으로 이미지와 텍스트 간 모든 조합의 점수를 계산할 수도 있다.

배치 처리는 GPU 활용도를 높일 수 있지만, 배치가 커지면 메모리 사용량도 증가한다.

### 실습 과제

* 크기가 서로 다른 이미지 3장을 준비한다.

* Processor를 이용해 입력 텐서의 크기를 확인한다.

* 텍스트 후보를 2개에서 5개로 늘려 `input_ids`의 형태가 어떻게 달라지는지 확인한다.

* 패딩과 어텐션 마스크가 필요한 이유를 설명한다.

## 5. 사전학습 CLIP 모델 불러오기

### 5.1 사전학습 모델이란?

사전학습 모델(Pre-trained Model)은 대규모 데이터와 학습 과정을 통해 이미 파라미터가 학습된 모델이다.

처음부터 모델을 학습하려면 대량의 데이터, 연산 자원, 학습 시간이 필요하다. 사전학습 모델을 활용하면 학습된 표현을 이용해 다양한 작업을 빠르게 실험할 수 있다.

이번 이 교재에서는 다음 두 클래스를 사용한다.

* `CLIPProcessor`: 이미지와 텍스트를 모델 입력 형식으로 변환한다.

* `CLIPModel`: 이미지와 텍스트의 특징을 추출하고 관련성 점수를 계산한다.

### 5.2 모델 로딩 코드

```python

import torch
from transformers import CLIPModel, CLIPProcessor

# GPU를 사용할 수 있으면 GPU를 선택하고, 아니면 CPU를 사용한다.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hugging Face에 공개된 CLIP 체크포인트 이름이다.
model_name = "openai/clip-vit-base-patch32"

# 모델이 요구하는 이미지 전처리와 텍스트 토큰화를 담당한다.
processor = CLIPProcessor.from_pretrained(model_name)

# 사전학습 가중치를 불러온 뒤 실행 장치로 이동한다.
model = CLIPModel.from_pretrained(model_name).to(device)

# 추론 모드로 설정한다.
# Dropout과 같은 학습·추론 동작 차이가 있는 계층을 평가 방식으로 전환한다.
model.eval()

print("실행 장치:", device)
print("모델 로드 완료")
```

### 5.3 `model.eval()`과 `torch.no_grad()`의 차이

두 기능은 서로 다르다.

| 기능                | 역할                         |
| ----------------- | -------------------------- |
| `model.eval()`    | 모델을 평가 모드로 전환한다.           |
| `torch.no_grad()` | 해당 코드 블록에서 기울기 계산을 비활성화한다. |

추론 시에는 일반적으로 두 기능을 함께 사용한다. 다만 `eval()`을 호출했다고 해서 기울기 계산이 자동으로 비활성화되는 것은 아니다.

### 5.4 사전학습과 미세조정 구분

* 추론: 이미 학습된 파라미터를 이용해 결과를 계산한다.

* 미세조정(Fine-tuning): 특정 데이터셋을 사용해 기존 모델의 파라미터 일부 또는 전체를 추가로 학습한다.

* 처음부터 학습: 모델 파라미터를 초기화하고 학습 데이터를 이용해 새로 학습한다.


## 6. 이미지·텍스트 임베딩 추출 

### 6.1 임베딩을 직접 추출하는 이유

모델 내부의 임베딩을 직접 확인하면 유사도 계산이 어떻게 이루어지는지 이해할 수 있다.

CLIP은 이미지와 텍스트를 각각 인코더에 전달하고, 투영된 특징 벡터를 반환한다. 이번 예제에서 사용하는 체크포인트의 최종 임베딩 차원은 512다.

### 6.2 코드 실습

```python

import torch
import torch.nn.functional as F
from PIL import Image

# 이미지와 비교할 텍스트 후보를 준비한다.
image = Image.open('./image/dog.jpg').convert("RGB")

texts = [
    "A photo of a cat.",
    "A photo of a dog.",
    "A photo of a car."
]

# 이미지와 텍스트를 각각 모델 입력으로 변환한다.
inputs = processor(
    images=image,
    text=texts,
    return_tensors="pt",
    padding=True
)

# 입력 텐서를 모델이 실행되는 장치로 이동한다.
inputs = {key: value.to(device) for key, value in inputs.items()}

# 추론 단계이므로 기울기 계산을 비활성화한다.
import torch
import torch.nn.functional as F

with torch.no_grad():

    # 1. 이미지 인코더를 실행한다.
    image_outputs = model.vision_model(
        pixel_values=inputs["pixel_values"]
    )

    # 2. 이미지의 pooled output을 추출한다.
    image_pooled = image_outputs.pooler_output

    # 3. 이미지 투영층을 적용해 CLIP 임베딩 공간으로 변환한다.
    image_features = model.visual_projection(image_pooled)

    # 4. 텍스트 인코더를 실행한다.
    text_outputs = model.text_model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )

    # 5. 텍스트의 pooled output을 추출한다.
    text_pooled = text_outputs.pooler_output

    # 6. 텍스트 투영층을 적용해 CLIP 임베딩 공간으로 변환한다.
    text_features = model.text_projection(text_pooled)

# 7. # 각 임베딩 벡터의 L2 노름이 1이 되도록 정규화한다.
# 이렇게 하면 이후 내적을 코사인 유사도로 사용할 수 있다.
image_features = F.normalize(image_features, p=2, dim=-1)
text_features = F.normalize(text_features, p=2, dim=-1)

print("이미지 임베딩:", image_features.shape)
print("텍스트 임베딩:", text_features.shape)
```

예상되는 형태는 다음과 같다.

* 이미지 임베딩: `(1, 512)`

* 텍스트 임베딩: `(3, 512)`

이 크기는 해당 체크포인트와 입력 개수에 대한 예시다. 다른 모델을 사용하면 임베딩 차원이 달라질 수 있다.

### 6.3 코드 해석

`image_features`는 이미지 1장을 표현하는 512차원 벡터다. `text_features`는 텍스트 후보 3개를 각각 표현하는 512차원 벡터 3개를 담는다.

두 벡터의 차원이 같은 이유는 두 인코더의 출력을 비교할 수 있는 공통 임베딩 공간으로 투영하기 때문이다.

### 학습 확인 질문

1. 이미지 임베딩의 첫 번째 차원은 무엇을 의미하는가?

2. 텍스트 후보가 3개라면 텍스트 임베딩의 형태는 어떻게 되는가?

3. 임베딩을 정규화하면 어떤 장점이 있는가?

## 7. 코사인 유사도와 유사도 행렬 

### 7.1 코사인 유사도

코사인 유사도(Cosine Similarity)는 두 벡터의 방향이 얼마나 비슷한지 측정하는 방법이다.


$$\text{Cosine Similarity}(I, T) = \cos(\theta) = \frac{I \cdot T}{\Vert{}I\Vert{} \Vert{}T\Vert{}} = \frac{\sum_{i=1}^{n} I_i T_i}{\sqrt{\sum_{i=1}^{n} I_i^2} \sqrt{\sum_{i=1}^{n} T_i^2}}$$

여기서 I는 이미지 임베딩, T는 텍스트 임베딩을 의미한다.

코사인 유사도는 두 벡터의 방향에 초점을 둔다. 벡터의 크기가 달라도 방향이 같으면 코사인 유사도는 1이 될 수 있다.

### 7.2 CLIP에서 유사도 계산

```python

# 이미지 임베딩: (이미지 수, 임베딩 차원)
# 텍스트 임베딩: (텍스트 수, 임베딩 차원)

# 텍스트 임베딩의 전치 행렬을 사용해 행렬 곱을 수행한다.
# 결과: (이미지 수, 텍스트 수)
similarity_matrix = image_features @ text_features.T

print("유사도 행렬 크기:", similarity_matrix.shape)
print(similarity_matrix)
```

이미지 1장과 텍스트 3개를 비교하면 결과 행렬은 `(1, 3)` 형태가 된다.

### 7.3 여러 이미지와 여러 텍스트의 관계

이미지 3장과 텍스트 4개를 비교하면 유사도 행렬은 `(3, 4)` 형태가 된다.

### 유사도 행렬 예시

아래 값은 원리를 설명하기 위한 가상 데이터다.

|       |      |      |      |
| ----- | ---- | ---- | ---- |
| 이미지   | 고양이  | 강아지  | 자동차  |
| 이미지 1 | 0.91 | 0.18 | 0.06 |
| 이미지 2 | 0.22 | 0.86 | 0.11 |
| 이미지 3 | 0.04 | 0.13 | 0.94 |

각 행에서 가장 높은 값은 해당 이미지와 가장 관련성이 높은 후보 문장을 나타낸다. 그러나 실제 점수는 모델, 프롬프트, 이미지 내용에 따라 달라지며 점수만으로 의미의 완전한 일치나 정답 확률을 보장하지 않는다.

### 7.4 유사도 결과 정렬

```python

# 이미지 한 장에 대한 유사도 점수를 1차원 텐서로 선택한다.
scores = similarity_matrix[0]

# 점수가 높은 순서대로 텍스트 인덱스를 정렬한다.
ranked_indices = torch.argsort(scores, descending=True)

# 각 후보 문장과 유사도 점수를 출력한다.
for rank, index in enumerate(ranked_indices, start=1):
    idx = int(index)
    print(
        f"{rank}위: {texts[idx]} "
        f"(유사도: {scores[idx].item():.4f})"
    )
```

### 수업 확인 질문

* 유사도 행렬에서 행과 열은 각각 무엇을 의미하는가?

* 가장 높은 유사도 점수를 받은 문장을 예측 결과로 선택하는 이유는 무엇인가?

* 유사도가 높아도 이미지 설명이 틀릴 수 있는 이유는 무엇인가?

## 8. CLIP 결과 분석 및 실습 평가

### 실습 절차

1. 이미지 1장을 불러온다.

2. 서로 다른 설명문 5개를 작성한다.

3. CLIP으로 이미지와 텍스트 임베딩을 추출한다.

4. 코사인 유사도를 계산한다.

5. 유사도가 높은 순서로 문장을 정렬한다.

6. 이미지와 결과를 비교하고 오분류 원인을 기록한다.

### 결과 분석 보고서 예시

| 항목     | 작성 내용                     |
| ------ | ------------------------- |
| 이미지 내용 | 이미지에 실제로 포함된 객체와 상황       |
| 후보 문장  | 비교에 사용한 5개 문장             |
| 최상위 결과 | 유사도 점수가 가장 높은 문장          |
| 결과 적절성 | 이미지 내용과 일치하는지 여부          |
| 오차 원인  | 후보 문장 구성, 세부 속성, 배경 등의 영향 |
| 개선 방안  | 후보 문장 수정 및 추가 실험 계획       |

### 평가 과제

과제: 이미지와 텍스트 간 유사도를 계산하는 프로그램을 작성하고 결과를 분석한다.

평가 기준은 다음과 같다.

* 모델 및 Processor를 정상적으로 불러오는가?

* 이미지와 텍스트의 임베딩을 추출하는가?

* 유사도 계산과 순위 정렬이 올바른가?

* 실제 이미지와 결과를 비교해 한계점을 설명하는가?

# 2. CLIP 응용과 BLIP 기초

 · 8시간

# Zero-shot 분류와 이미지 설명 생성

학습 목표: CLIP으로 이미지 분류·검색을 구현하고, BLIP으로 이미지 설명을 생성한다.

## . CLIP Zero-shot 분류 (1시간)

### 1.1 Zero-shot이란 무엇인가?

Zero-shot 분류는 새로운 분류 대상에 대해 별도의 클래스별 분류기를 학습하지 않고, 입력 이미지와 후보 텍스트의 관련성을 비교해 클래스를 선택하는 방식이다.

일반적인 이미지 분류에서는 강아지, 고양이, 자동차 등을 구분하기 위해 해당 클래스가 포함된 데이터로 분류 모델을 학습한다.

CLIP에서는 다음과 같이 후보 문장을 준비할 수 있다.

* `A photo of a cat.`

* `A photo of a dog.`

* `A photo of a car.`

입력 이미지와 각 문장 사이의 점수를 계산한 뒤 가장 높은 점수를 받은 문장을 예측 결과로 선택한다.

### 1.2 실행 구조

입력 이미지

고양이 사진

후보 문장 구성

고양이 · 강아지 · 자동차

CLIP 이미지-텍스트 점수 계산

후보 문장별 관련성 비교

최고 점수 선택

예측: 고양이

### 1.3 코드 실습

Python

실행됨

```
import torch

# 분류할 클래스의 이름을 정의한다.
candidate_labels = ["cat", "dog", "car", "airplane"]

# 각 클래스 이름을 자연어 문장으로 바꾼다.
# CLIP은 이미지와 텍스트 간 관련성을 비교한다.
candidate_texts = [
    f"A photo of a {label}."
    for label in candidate_labels
]

# 이미지와 후보 문장을 모델 입력으로 전처리한다.
inputs = processor(
    images=image,
    text=candidate_texts,
    return_tensors="pt",
    padding=True
)

# 모델의 실행 장치와 입력 장치를 일치시킨다.
inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

# 모델을 추론 모드로 실행한다.
with torch.no_grad():
    outputs = model(**inputs)

    # 각 후보 텍스트와 이미지 간 관련성 로짓을 가져온다.
    logits = outputs.logits_per_image[0]

    # 후보 클래스 사이의 상대적 점수 분포를 계산한다.
    # 이 값은 후보 집합에 조건부인 점수이지, 보정된 실제 정답 확률은 아니다.
    scores = torch.softmax(logits, dim=0)

# 점수가 높은 후보부터 출력한다.
ranked_indices = torch.argsort(scores, descending=True)

for idx in ranked_indices:
    i = int(idx)
    print(
        f"{candidate_labels[i]:10s} "
        f"점수={scores[i].item():.4f}"
    )
```

### 1.4 결과 해석 시 주의할 점

Zero-shot 분류는 별도의 클래스별 학습 없이 활용할 수 있다는 장점이 있지만, 항상 정확한 결과를 보장하지 않는다.

* 후보 클래스에 정답이 없으면 가장 덜 부적절한 후보를 선택할 수 있다.

* 문장 표현을 바꾸면 점수 순위가 달라질 수 있다.

* 학습 데이터와 실제 이미지의 분포가 다르면 성능이 저하될 수 있다.

* 세부적인 객체 수량이나 공간 관계를 정확히 판별하지 못할 수 있다.

따라서 분류 결과를 평가할 때는 정답 레이블이 있는 데이터셋을 이용해 전체 정확도와 클래스별 정확도를 계산해야 한다.

### 수업 확인 질문

1. Zero-shot 분류에서 후보 텍스트는 어떤 역할을 하는가?

2. 후보 문장에 실제 정답 클래스가 없으면 어떤 문제가 발생하는가?

3. 소프트맥스 점수를 실제 정답 확률로 단정할 수 없는 이유는 무엇인가?

## . 프롬프트 설계 (1시간)

### 2.1 프롬프트가 중요한 이유

CLIP은 이미지와 텍스트를 비교한다. 따라서 후보 문장을 어떻게 구성하느냐에 따라 유사도 점수가 달라질 수 있다.

예를 들어 고양이 사진에 대해 다음 후보를 비교할 수 있다.

* `cat`

* `a photo of a cat`

* `a close-up photo of a cat`

* `a cat sitting on a sofa`

이 문장들은 같은 객체를 언급하더라도 표현하는 상황과 세부 정보가 다르다.

### 2.2 프롬프트 비교 실험

Python

실행됨

```
# 같은 이미지에 대해 서로 다른 프롬프트를 준비한다.
prompts = [
    "cat",
    "a photo of a cat",
    "a close-up photo of a cat",
    "a cat sitting on a sofa",
    "a dog sitting on a sofa"
]

# 동일한 이미지와 프롬프트를 함께 전처리한다.
inputs = processor(
    images=image,
    text=prompts,
    return_tensors="pt",
    padding=True
)
inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

# 이미지-텍스트 관련성 점수를 계산한다.
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits_per_image[0]

# 점수를 후보 문장 간 비교가 쉬운 상대적 분포로 변환한다.
scores = torch.softmax(logits, dim=0)

# 결과를 점수가 높은 순서대로 출력한다.
order = torch.argsort(scores, descending=True)

for idx in order:
    i = int(idx)
    print(f"{scores[i].item():.4f} | {prompts[i]}")
```

### 2.3 프롬프트 실험 설계

수강생은 프롬프트를 무작정 바꾸는 대신 하나의 조건만 바꾸어 실험하는 것이 좋다.

| 실험   | 변경 요소    | 확인할 내용             |
| ---- | -------- | ------------------ |
| 실험 1 | 단어와 문장   | 문장 형태가 점수에 미치는 영향  |
| 실험 2 | 객체와 행동   | 행동 정보가 순위에 미치는 영향  |
| 실험 3 | 배경 정보    | 장면 설명이 관련성에 미치는 영향 |
| 실험 4 | 후보 클래스 수 | 후보 집합 변화에 따른 결과 차이 |

실험 결과는 프롬프트별 점수와 순위를 함께 기록한다. 특정 프롬프트가 높은 점수를 받았다는 사실만으로 실제 정확도가 높아졌다고 판단하지 않는다.

## . CLIP을 활용한 이미지 검색 (1시간)

### 3.1 텍스트 기반 이미지 검색

이미지 검색 시스템에서는 미리 여러 이미지의 임베딩을 계산해 저장하고, 사용자가 입력한 검색 문장의 임베딩과 비교할 수 있다.

예를 들어 이미지 데이터베이스에 다음 사진들이 있다고 가정한다.

* 강아지가 공원에서 뛰는 사진

* 고양이가 소파에 앉아 있는 사진

* 자동차가 도로를 달리는 사진

사용자가 `a dog playing in a park`라는 문장을 입력하면 CLIP은 검색 문장과 각 이미지의 유사도를 계산해 순위를 정할 수 있다.

### 3.2 검색 시스템의 구조

이미지 데이터베이스

이미지 파일 여러 장

이미지 인코더

이미지별 임베딩 생성 및 저장

사용자 검색 문장

텍스트 인코더 → 검색 임베딩

벡터 유사도 계산

저장된 이미지 임베딩과 검색 임베딩 비교

검색 결과

유사도가 높은 이미지부터 표시

### 3.3 이미지 검색과 이미지 분류의 차이

이미지 분류는 일반적으로 각 이미지에 대해 하나의 클래스를 선택한다. 이미지 검색은 데이터베이스의 여러 이미지 가운데 검색 문장과 관련성이 높은 항목을 찾는다.

| 구분    | 이미지 분류    | 이미지 검색         |
| ----- | --------- | -------------- |
| 입력    | 이미지       | 이미지 또는 텍스트 질의  |
| 비교 대상 | 후보 클래스    | 저장된 이미지 집합     |
| 결과    | 예측 클래스    | 순위가 매겨진 이미지 목록 |
| 활용    | 이미지 자동 분류 | 자연어 기반 이미지 검색  |

### 실습 과제

이미지 10장 이상을 준비하고 검색 문장 3개를 작성한다. 검색 결과 상위 3개를 확인한 뒤, 기대한 결과와 다른 이미지가 나온 이유를 분석한다.

## . CLIP 결과 평가 (1시간)

### 4.1 왜 평가가 필요한가?

유사도 계산이 정상적으로 실행된다는 사실과 모델이 정확한 결과를 제공한다는 사실은 다르다.

예를 들어 강아지 사진에 대해 `a photo of a dog`라는 문장이 가장 높은 점수를 받았더라도, 데이터셋 전체에서 강아지를 잘 구분한다고 단정할 수는 없다.

### 4.2 주요 평가 지표

정확도(Accuracy)

Accuracy=정답 수전체 예측 수\text{Accuracy}= \frac{\text{정답 수}}{\text{전체 예측 수}}Accuracy=전체 예측 수정답 수

전체 이미지 가운데 정답 클래스를 올바르게 예측한 비율이다.

클래스별 정확도

특정 클래스에 속한 이미지 가운데 올바르게 예측한 비율이다. 클래스별 정확도를 확인하면 전체 정확도만으로 드러나지 않는 성능 차이를 발견할 수 있다.

예를 들어 강아지 사진은 잘 구분하지만 자동차 사진은 잘 구분하지 못하는 경우가 있을 수 있다.

### 4.3 평가 시 주의사항

* 실제 정답 레이블을 기준으로 평가한다.

* 평가 이미지와 프롬프트 구성을 기록한다.

* 전체 정확도와 클래스별 정확도를 함께 확인한다.

* 오분류 사례를 이미지와 함께 검토한다.

* 같은 평가 데이터에서 프롬프트를 반복적으로 조정했다면, 별도의 검증 데이터로 최종 성능을 확인한다.

### 실습 과제

1. 클래스 3개 이상의 이미지 데이터셋을 준비한다.

2. 각 이미지에 대해 CLIP 예측을 수행한다.

3. 전체 정확도와 클래스별 정확도를 계산한다.

4. 오분류 이미지 3개를 선택해 원인을 분석한다.

## 5교시. BLIP의 개념과 구조 (1시간)

### 5.1 BLIP이란 무엇인가?

BLIP(Bootstrapping Language-Image Pre-training)은 이미지와 텍스트의 관계를 학습해 이미지 이해와 텍스트 생성 작업에 활용할 수 있도록 설계된 비전-언어 모델 프레임워크다.

CLIP과 BLIP 모두 이미지와 텍스트를 다루지만, 이 교재에서는 다음과 같이 구분하면 이해하기 쉽다.

* CLIP: 이미지와 후보 문장이 얼마나 관련 있는지 비교한다.

* BLIP: 이미지 내용을 바탕으로 설명 문장을 생성하거나 질문에 답한다.

BLIP은 모델 구성과 학습 방식에 따라 이미지-텍스트 검색, 이미지 캡셔닝, 시각적 질의응답 등의 작업에 활용할 수 있다.

### 5.2 BLIP의 작동 구조

이미지 입력

사진 픽셀

비전 인코더

이미지 특징 추출

이미지 특징을 활용한 텍스트 생성

디코더가 토큰을 순차적으로 생성

자연어 설명

예: A cat is sitting on a sofa.

위 구조는 이미지 캡셔닝을 설명하기 위한 개념도다. 실제 BLIP은 작업과 체크포인트에 따라 이미지-텍스트 인코더, 이미지 조건부 텍스트 디코더 등의 구성과 연결 방식이 달라질 수 있다.

### 5.3 CLIP과 BLIP을 구분하는 질문

* CLIP에 “이 사진과 가장 관련 있는 문장은 무엇인가?”라고 묻는다면 후보 문장들의 관련성 점수를 비교한다.

* BLIP에 “이 사진에는 무엇이 있는가?”라는 작업을 맡기면 이미지 내용을 바탕으로 설명 문장을 생성할 수 있다.

* BLIP의 질문응답 모델은 이미지와 질문을 함께 입력받아 질문에 대한 답변을 생성한다.

이 차이는 모델을 선택할 때 중요한 기준이 된다.

## 6교시. BLIP 모델 로드와 이미지 캡셔닝 (1시간)

### 6.1 이미지 캡셔닝이란?

이미지 캡셔닝(Image Captioning)은 이미지의 주요 객체와 장면을 자연어 문장으로 설명하는 작업이다.

입력은 이미지이고 출력은 문장이다. 이미지 분류처럼 미리 정의된 클래스 하나만 선택하는 것이 아니라, 모델이 학습한 언어 표현을 이용해 문장을 생성한다.

### 6.2 코드 실습

Python

실행됨

```
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# 실행 환경에 GPU가 있으면 GPU를 선택한다.
device = "cuda" if torch.cuda.is_available() else "cpu"

# 이미지 캡셔닝용 사전학습 체크포인트를 지정한다.
model_name = "Salesforce/blip-image-captioning-base"

# Processor는 이미지 입력을 모델이 요구하는 형태로 변환한다.
processor = BlipProcessor.from_pretrained(model_name)

# 이미지 캡셔닝 모델과 사전학습 가중치를 불러온다.
model = BlipForConditionalGeneration.from_pretrained(
    model_name
).to(device)

# 추론 모드로 설정한다.
model.eval()

# 실제 이미지 파일을 불러온다.
image = Image.open("cat.jpg").convert("RGB")

# 이미지 픽셀을 모델 입력 텐서로 변환한다.
inputs = processor(
    images=image,
    return_tensors="pt"
)

# 입력 텐서를 모델과 같은 장치로 이동한다.
inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

# 모델이 이미지에 대한 설명 문장을 생성한다.
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=40
    )

# 생성된 토큰 ID를 사람이 읽을 수 있는 문장으로 변환한다.
caption = processor.decode(
    output_ids[0],
    skip_special_tokens=True
)

print("생성된 설명:", caption)
```

### 6.3 코드 해석

`BlipProcessor`는 이미지 입력을 전처리하고, `BlipForConditionalGeneration`은 이미지 특징을 바탕으로 텍스트를 생성한다.

`generate()`는 토큰을 순차적으로 생성하는 함수다. `max_new_tokens=40`은 새롭게 생성할 토큰 수의 상한을 설정한다.

생성된 설명은 이미지의 내용을 정확하게 반영할 수도 있지만, 잘못된 객체나 상황을 포함할 수도 있다. 따라서 모델 출력은 반드시 원본 이미지와 비교해 검증해야 한다.

## 7교시. 이미지 캡셔닝 결과 분석 (1시간)

### 7.1 생성 결과는 어떻게 평가하는가?

이미지 설명 생성의 결과를 평가할 때는 문장의 자연스러움만 확인해서는 안 된다. 이미지의 핵심 내용이 제대로 반영되었는지 확인해야 한다.

| 평가 항목 | 확인 질문                    |
| ----- | ------------------------ |
| 객체    | 이미지에 실제로 존재하는 객체를 설명했는가? |
| 행동    | 사람이나 동물의 행동을 정확히 설명했는가?  |
| 속성    | 색상, 수량, 크기 등의 묘사가 적절한가?  |
| 배경    | 장면과 배경을 적절하게 설명했는가?      |
| 사실성   | 이미지에 없는 정보를 생성하지 않았는가?   |

### 7.2 이미지별 비교 실습

이미지 5장을 준비하고 같은 캡셔닝 모델로 각각 설명을 생성한다. 이미지 종류는 동물, 음식, 거리, 실내, 자연 풍경 등으로 다양하게 구성한다.

수강생은 생성 문장과 실제 이미지 내용을 비교해 다음을 기록한다.

* 정확하게 설명한 객체

* 누락된 핵심 정보

* 잘못된 객체 또는 속성

* 개선할 수 있는 데이터나 모델 설정

### 7.3 생성 모델의 한계

BLIP은 이미지 특징과 학습된 언어 패턴을 활용해 문장을 생성한다. 하지만 출력 문장이 자연스럽다고 해서 모든 내용이 사실이라는 뜻은 아니다.

예를 들어 사진에 컵이 있지만 컵 안의 내용물이 명확하지 않은 경우, 모델이 실제로 확인하기 어려운 세부 내용을 잘못 설명할 수 있다.

이러한 현상은 시각적 환각(Visual Hallucination)의 사례로 분석할 수 있다.

## 8교시. CLIP과 BLIP 비교 및  평가 (1시간)

### 8.1 모델 비교 실습

동일한 이미지를 두 모델에 입력해 결과를 비교한다.

* CLIP: 이미지와 후보 설명문 5개 사이의 관련성 점수를 계산한다.

* BLIP: 이미지에 대한 설명 문장을 생성한다.

예를 들어 CLIP은 다음 후보 중 가장 관련성이 높은 문장을 선택할 수 있다.

* A cat is sleeping on a sofa.

* A dog is running in a park.

* A car is parked on the road.

BLIP은 입력 이미지에 대해 `A cat is sleeping on a sofa.`와 같은 문장을 생성할 수 있다.

두 결과가 같을 수도 있지만, CLIP은 제공된 후보 가운데 하나를 선택하고 BLIP은 학습된 언어 표현을 이용해 문장을 생성한다는 차이가 있다.

### 8.2 비교 결과 보고서

| 항목    | CLIP            | BLIP                |
| ----- | --------------- | ------------------- |
| 입력    | 이미지와 비교할 텍스트    | 이미지                 |
| 출력    | 후보별 관련성 점수      | 생성된 설명 문장           |
| 평가    | 순위, 정확도, 검색 적합성 | 설명의 사실성, 객체·행동의 정확성 |
| 주요 한계 | 후보 문장에 의존       | 생성 오류와 시각적 환각 가능성   |

###  평가 과제

1. CLIP으로 이미지 Zero-shot 분류를 구현한다.

2. 프롬프트를 바꾸어 결과 차이를 기록한다.

3. BLIP으로 이미지 5장의 설명을 생성한다.

4. 두 모델의 입력과 출력, 적용 목적을 비교한다.

5. 잘못된 결과 사례를 최소 2개 분석한다.

가 끝나면 수강생은 CLIP을 활용한 분류·검색과 BLIP을 활용한 설명 생성의 차이를 실제 코드로 설명할 수 있어야 한다.


# Day 3. BLIP 응용과 통합 프로젝트

3일차 · 8시간

# 이미지 질의응답과 멀티모달 AI 서비스 구현

학습 목표: BLIP VQA를 구현하고 CLIP과 BLIP을 결합해 이미지 검색·설명 서비스를 완성한다.

## . BLIP 시각적 질의응답(VQA) 원리 (1시간)

### 1.1 VQA란 무엇인가?

VQA(Visual Question Answering)는 이미지와 질문을 함께 입력받아 이미지 내용을 바탕으로 답변을 생성하는 작업이다.

이미지 캡셔닝과 VQA의 차이는 질문의 유무와 출력 목적에 있다.

* 이미지 캡셔닝: 이미지 전체를 설명한다.

* VQA: 사용자가 질문한 내용에 초점을 맞춰 답변한다.

예를 들어 이미지가 고양이 두 마리가 소파에 앉아 있는 사진이라면 다음과 같은 질문을 만들 수 있다.

| 질문                               | 기대하는 답변 유형 |
| -------------------------------- | ---------- |
| What animals are in the picture? | 객체         |
| Where are the cats sitting?      | 장소·공간 관계   |
| How many cats are visible?       | 수량         |
| What color is the sofa?          | 속성         |

위 답변은 예시이며, 실제 모델이 반드시 정확하게 생성한다는 의미는 아니다.

### 1.2 VQA의 데이터 흐름

이미지

시각적 정보

질문

언어적 정보

BLIP VQA 모델

이미지 특징과 질문을 활용해 답변 생성

답변 텍스트

예: Two cats.

### 1.3 캡셔닝과 VQA를 구분해야 하는 이유

캡셔닝 모델은 이미지 전체를 요약하는 설명을 생성하도록 사용한다. VQA 모델은 이미지와 질문을 함께 받아 질문에 대한 답변을 생성하도록 사용한다.

따라서 이미지 설명 생성에 적합한 체크포인트와 VQA에 적합한 체크포인트를 구분해야 한다. 같은 BLIP 계열이라도 모델 클래스와 학습 목적이 다를 수 있다.

### 수업 확인 질문

1. 이미지 캡셔닝과 VQA의 입력 차이는 무엇인가?

2. 수량을 묻는 질문에서 모델이 오류를 낼 수 있는 이유는 무엇인가?

3. VQA 결과를 검증하려면 어떤 정보와 비교해야 하는가?

## . BLIP VQA 코드 실습 (1시간)

### 2.1 모델 준비

VQA 실습에서는 `Salesforce/blip-vqa-base` 체크포인트와 `BlipForQuestionAnswering` 클래스를 사용한다.

Python

실행됨

```
import torch
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForQuestionAnswering
)

# GPU가 사용 가능하면 GPU, 아니면 CPU를 선택한다.
device = "cuda" if torch.cuda.is_available() else "cpu"

# 이미지와 질문을 함께 처리하도록 학습된 VQA 체크포인트다.
model_name = "Salesforce/blip-vqa-base"

# 이미지 전처리 및 질문 텍스트 처리를 담당한다.
processor = BlipProcessor.from_pretrained(model_name)

# VQA 작업용 사전학습 모델을 불러온다.
model = BlipForQuestionAnswering.from_pretrained(
    model_name
).to(device)

# 추론 모드로 전환한다.
model.eval()

# 본인의 이미지 경로로 변경한다.
image = Image.open("cat.jpg").convert("RGB")

print("VQA 모델 준비 완료")
```

### 2.2 이미지와 질문을 함께 입력하기

Python

실행됨

```
def answer_image_question(image, question):
    """
    이미지와 질문을 입력받아 BLIP VQA 모델의 답변을 생성한다.

    Parameters
    ----------
    image : PIL.Image
        질문 대상 이미지
    question : str
        이미지에 대해 묻고 싶은 자연어 질문

    Returns
    -------
    str
        모델이 생성한 답변
    """

    # 이미지의 색상 모드를 RGB로 맞춘다.
    image = image.convert("RGB")

    # Processor가 이미지와 질문을 모델 입력 텐서로 변환한다.
    inputs = processor(
        images=image,
        text=question,
        return_tensors="pt"
    )

    # 입력을 모델과 동일한 실행 장치로 이동한다.
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # 추론 단계에서는 기울기 계산이 필요하지 않다.
    with torch.no_grad():

        # 모델이 질문에 대한 답변 토큰을 생성한다.
        answer_ids = model.generate(
            **inputs,
            max_new_tokens=20
        )

    # 생성된 토큰 ID를 읽을 수 있는 문장으로 변환한다.
    answer = processor.decode(
        answer_ids[0],
        skip_special_tokens=True
    )

    return answer


# 같은 이미지에 여러 질문을 적용한다.
questions = [
    "What animals are in the picture?",
    "Where are the animals sitting?",
    "How many cats are visible?"
]

for question in questions:
    answer = answer_image_question(image, question)

    print("질문:", question)
    print("답변:", answer)
    print("-" * 40)
```

### 2.3 결과 해석

같은 이미지라도 질문이 달라지면 모델이 생성해야 할 답변도 달라진다.

* 객체 질문은 이미지 안의 대상을 식별하는 데 초점을 둔다.

* 공간 관계 질문은 대상과 배경의 관계를 파악해야 한다.

* 수량 질문은 객체의 개수를 정확히 파악해야 한다.

특히 수량과 세부 공간 관계는 오류가 발생할 수 있다. 결과가 그럴듯하더라도 이미지와 대조해 확인하는 절차가 필요하다.

### 실습 과제

이미지 3장에 대해 질문을 각각 5개씩 작성한다. 답변을 사람이 확인하고 오류를 다음 범주로 구분한다.

* 객체 식별 오류

* 수량 오류

* 속성 오류

* 공간 관계 오류

* 질문 해석 오류

## . 생성 결과 평가와 시각적 환각 (1시간)

### 3.1 시각적 환각이란?

시각적 환각은 모델이 이미지에서 확인할 수 없는 객체나 상황을 사실처럼 생성하는 현상을 말한다.

예를 들어 이미지에는 컵만 있는데 모델이 컵 안에 커피가 있다고 설명하거나, 실제로 확인하기 어려운 객체의 색상을 단정할 수 있다.

### 3.2 오류 분석 기준

| 오류 유형    | 예시                  | 분석 관점   |
| -------- | ------------------- | ------- |
| 객체 오류    | 고양이를 강아지라고 설명       | 객체 식별   |
| 수량 오류    | 객체 2개를 3개라고 답변      | 개수 인식   |
| 속성 오류    | 검은색 물체를 흰색이라고 설명    | 시각적 속성  |
| 공간 오류    | 책상 위 물체를 바닥에 있다고 설명 | 객체 간 관계 |
| 근거 없는 생성 | 보이지 않는 행동을 설명       | 사실성 검증  |

### 3.3 생성 결과 평가 실습

수강생은 BLIP이 생성한 캡션과 VQA 답변을 표로 정리한다.

| 이미지   | 작업  | 모델 결과  | 실제 이미지와의 차이 | 오류 유형   |
| ----- | --- | ------ | ----------- | ------- |
| 이미지 A | 캡셔닝 | 생성된 설명 | 검토 후 기록     | 객체·속성 등 |
| 이미지 A | VQA | 생성된 답변 | 검토 후 기록     | 수량·공간 등 |
| 이미지 B | 캡셔닝 | 생성된 설명 | 검토 후 기록     | 해당 유형   |
| 이미지 B | VQA | 생성된 답변 | 검토 후 기록     | 해당 유형   |

평가 시에는 문장의 자연스러움과 사실성을 구분해야 한다. 자연스러운 문장도 이미지의 사실과 일치하지 않을 수 있다.

### 3.4 개선 방향

* 입력 이미지의 품질을 확인한다.

* 질문을 명확하고 구체적으로 작성한다.

* 여러 종류의 이미지에서 반복적으로 평가한다.

* 생성 결과를 사람이 확인할 수 있는 검증 절차를 마련한다.

* 실제 서비스에서는 모델의 한계와 오류 가능성을 사용자에게 알린다.

## . CLIP과 BLIP 통합 설계 (1시간)

### 4.1 통합 서비스가 필요한 이유

CLIP과 BLIP은 서로 다른 역할을 담당한다. 두 모델을 연결하면 검색과 설명 생성을 한 서비스에서 제공할 수 있다.

예를 들어 사용자가 사진을 업로드하면 다음 작업을 수행한다.

1. CLIP이 이미지와 후보 문장의 유사도를 계산한다.

2. 유사도가 높은 문장을 검색 결과로 표시한다.

3. BLIP이 이미지에 대한 설명을 생성한다.

4. 사용자에게 검색 결과와 생성 설명을 함께 제공한다.

### 4.2 시스템 구조도

사용자

이미지 업로드 및 검색어 입력

CLIP

이미지·텍스트 유사도

검색 순위 및 점수

BLIP

이미지 캡셔닝

자연어 설명 생성

결과 통합 화면

이미지 · 검색 결과 · 설명 문장

### 4.3 기능을 분리해야 하는 이유

CLIP의 유사도 계산과 BLIP의 텍스트 생성은 서로 다른 작업이다. 한 함수에 모든 코드를 넣기보다 다음과 같이 기능을 분리하는 편이 유지보수에 유리하다.

* `clip_similarity_for_image()`: 이미지와 후보 텍스트의 유사도를 계산한다.

* `blip_caption()`: 이미지 설명을 생성한다.

* `main()`: 입력과 결과 표시를 담당한다.

이렇게 분리하면 CLIP의 후보 문장 구성이나 BLIP의 생성 설정을 독립적으로 변경할 수 있다.

## 5교시. 통합 서비스 구현 (1시간)

### 5.1 모델 로드

통합 프로젝트에서는 CLIP과 BLIP을 각각 불러온다. 모델을 한 번만 로드하고 여러 입력에서 재사용해야 불필요한 로딩 시간을 줄일 수 있다.

Python

실행됨

```
import torch
import torch.nn.functional as F

from transformers import (
    CLIPModel,
    CLIPProcessor,
    BlipProcessor,
    BlipForConditionalGeneration
)

# GPU가 있으면 GPU를 사용한다.
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP: 이미지와 텍스트의 관련성을 비교한다.
clip_name = "openai/clip-vit-base-patch32"

clip_processor = CLIPProcessor.from_pretrained(clip_name)
clip_model = CLIPModel.from_pretrained(
    clip_name
).to(device)

clip_model.eval()

# BLIP: 이미지 설명을 생성한다.
blip_name = "Salesforce/blip-image-captioning-base"

blip_processor = BlipProcessor.from_pretrained(blip_name)
blip_model = BlipForConditionalGeneration.from_pretrained(
    blip_name
).to(device)

blip_model.eval()

print("CLIP·BLIP 모델 로드 완료")
```

### 5.2 CLIP 검색 함수

Python

실행됨

```
def clip_similarity_for_image(image, candidate_texts):
    """
    이미지와 여러 후보 텍스트의 관련성을 비교한다.

    반환값은 점수가 높은 순서대로 정렬한
    (후보 문장, 상대적 점수) 목록이다.
    """

    # 이미지와 텍스트를 CLIP 입력 형식으로 전처리한다.
    inputs = clip_processor(
        images=image.convert("RGB"),
        text=candidate_texts,
        return_tensors="pt",
        padding=True
    )

    # 입력 텐서를 모델 실행 장치로 옮긴다.
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        # 이미지와 텍스트의 관련성 로짓을 계산한다.
        outputs = clip_model(**inputs)
        logits = outputs.logits_per_image[0]

        # 후보 문장 집합 안에서 상대적인 점수 분포를 계산한다.
        scores = torch.softmax(logits, dim=0)

    # 문장과 점수를 연결하고 높은 점수부터 정렬한다.
    ranked_results = sorted(
        zip(
            candidate_texts,
            scores.detach().cpu().tolist()
        ),
        key=lambda item: item[1],
        reverse=True
    )

    return ranked_results
```

### 5.3 BLIP 설명 생성 함수

Python

실행됨

```
def blip_caption(image, max_new_tokens=40):
    """
    입력 이미지의 설명 문장을 생성한다.

    max_new_tokens는 새로 생성할 토큰 수의 상한이다.
    """

    # 이미지를 BLIP의 입력 형식으로 전처리한다.
    inputs = blip_processor(
        images=image.convert("RGB"),
        return_tensors="pt"
    )

    # 모델과 같은 장치로 이동한다.
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # 추론을 실행한다.
    with torch.no_grad():
        output_ids = blip_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

    # 토큰 ID를 자연어 설명으로 복원한다.
    caption = blip_processor.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    return caption
```

### 5.4 두 기능 연결

Python

실행됨

```
from PIL import Image

# 실제 이미지 파일을 불러온다.
image = Image.open("cat.jpg").convert("RGB")

# CLIP이 비교할 후보 문장을 준비한다.
candidate_texts = [
    "A photo of a cat.",
    "A photo of a dog.",
    "A photo of a car.",
    "A photo of a bird."
]

# BLIP으로 이미지 설명을 생성한다.
caption = blip_caption(image)

# CLIP으로 이미지와 후보 문장의 관련성을 계산한다.
ranked_results = clip_similarity_for_image(
    image,
    candidate_texts
)

print("=== BLIP 이미지 설명 ===")
print(caption)

print("\\n=== CLIP 검색 결과 ===")
for rank, (text, score) in enumerate(
    ranked_results,
    start=1
):
    print(f"{rank}위 | 점수={score:.4f} | {text}")
```

이 코드는 통합 서비스의 핵심 추론 부분이다. 사용자 인터페이스를 연결하기 전에 각 함수가 정상적으로 실행되는지, 서로 다른 이미지에서도 적절한 결과를 반환하는지 확인한다.

## 6교시. 사용자 인터페이스 구현 (1시간)

### 6.1 Streamlit을 사용하는 이유

Streamlit은 Python 코드로 간단한 웹 인터페이스를 만들 수 있는 라이브러리다. 별도의 프런트엔드 개발을 최소화하면서 이미지 업로드, 텍스트 입력, 결과 표시 기능을 구현할 수 있다.

### 6.2 화면 구성

서비스 화면에는 다음 요소를 배치한다.

1. 이미지 업로드 버튼

2. CLIP 후보 문장 입력란

3. 이미지 미리보기

4. BLIP 생성 설명

5. CLIP 유사도 순위

### 6.3 Streamlit 앱 예시

다음 코드는 앞서 정의한 CLIP·BLIP 모델 로딩 및 추론 함수를 `app.py` 한 파일로 구성한 예시다. 이 교재에서는 모델 로딩 코드를 파일 상단에 한 번만 배치하고, 사용자가 입력을 변경할 때 추론 함수를 재사용하도록 설명한다.

Python

실행됨

```
# app.py
# 실행 명령: streamlit run app.py

import streamlit as st
import torch
from PIL import Image
from transformers import (
    CLIPModel,
    CLIPProcessor,
    BlipProcessor,
    BlipForConditionalGeneration
)

# --------------------------------------------------
# 1. 실행 장치 설정
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# 2. 모델 로딩
# --------------------------------------------------

# Streamlit은 입력이 변경될 때 앱을 다시 실행할 수 있다.
# cache_resource를 사용하면 같은 세션 환경에서 모델 로딩을 재사용할 수 있다.
@st.cache_resource
def load_models():

    clip_name = "openai/clip-vit-base-patch32"
    clip_processor = CLIPProcessor.from_pretrained(clip_name)
    clip_model = CLIPModel.from_pretrained(
        clip_name
    ).to(device)
    clip_model.eval()

    blip_name = "Salesforce/blip-image-captioning-base"
    blip_processor = BlipProcessor.from_pretrained(blip_name)
    blip_model = BlipForConditionalGeneration.from_pretrained(
        blip_name
    ).to(device)
    blip_model.eval()

    return (
        clip_processor,
        clip_model,
        blip_processor,
        blip_model
    )


# --------------------------------------------------
# 3. BLIP 캡셔닝 함수
# --------------------------------------------------

def make_caption(image, processor, model):

    inputs = processor(
        images=image.convert("RGB"),
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=40
        )

    return processor.decode(
        output_ids[0],
        skip_special_tokens=True
    )


# --------------------------------------------------
# 4. CLIP 유사도 함수
# --------------------------------------------------

def rank_texts(image, texts, processor, model):

    inputs = processor(
        images=image.convert("RGB"),
        text=texts,
        return_tensors="pt",
        padding=True
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

        # 이미지 한 장과 후보 문장 사이의 로짓을 가져온다.
        logits = outputs.logits_per_image[0]

        # 후보 집합 안에서 상대적인 점수 분포를 계산한다.
        scores = torch.softmax(logits, dim=0)

    ranked = sorted(
        zip(texts, scores.cpu().tolist()),
        key=lambda item: item[1],
        reverse=True
    )

    return ranked


# --------------------------------------------------
# 5. 웹 화면 구성
# --------------------------------------------------

st.title("CLIP · BLIP 이미지 분석 서비스")
st.write(
    "CLIP으로 이미지와 후보 문장의 관련성을 비교하고, "
    "BLIP으로 이미지 설명을 생성한다."
)

uploaded_file = st.file_uploader(
    "분석할 이미지를 업로드한다.",
    type=["jpg", "jpeg", "png", "webp"]
)

text_input = st.text_area(
    "CLIP 후보 문장을 입력한다. 문장마다 한 줄씩 작성한다.",
    value=(
        "A photo of a cat.\\n"
        "A photo of a dog.\\n"
        "A photo of a car."
    )
)

if uploaded_file is not None:

    # 업로드된 파일을 RGB 이미지로 변환한다.
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="업로드한 이미지",
        use_container_width=True
    )

    # 빈 줄을 제거해 후보 문장 목록을 만든다.
    candidate_texts = [
        line.strip()
        for line in text_input.splitlines()
        if line.strip()
    ]

    if st.button("이미지 분석 실행"):

        if not candidate_texts:
            st.warning("비교할 후보 문장을 한 개 이상 입력한다.")

        else:
            with st.spinner("모델을 실행하고 있다..."):

                (
                    clip_processor,
                    clip_model,
                    blip_processor,
                    blip_model
                ) = load_models()

                # BLIP으로 이미지 설명을 생성한다.
                caption = make_caption(
                    image,
                    blip_processor,
                    blip_model
                )

                # CLIP으로 후보 문장의 순위를 계산한다.
                results = rank_texts(
                    image,
                    candidate_texts,
                    clip_processor,
                    clip_model
                )

            st.subheader("BLIP 이미지 설명")
            st.write(caption)

            st.subheader("CLIP 관련성 순위")

            for rank, (text, score) in enumerate(
                results,
                start=1
            ):
                st.write(
                    f"{rank}위 | 점수 {score:.4f} | {text}"
                )
```

### 실행 방법

터미널에서 파일이 있는 디렉터리로 이동한 뒤 다음 명령을 실행한다.

Bash

```
pip install streamlit torch torchvision transformers pillow
streamlit run app.py
```

최초 실행 시 모델 다운로드가 필요하다. GPU 메모리가 부족하거나 다운로드 시간이 길면 모델 로딩 및 추론에 실패할 수 있으므로 수업 전에 환경을 점검해야 한다.

### 실습 과제

* 이미지 파일을 업로드할 수 있도록 구현한다.

* 후보 문장을 수정한 뒤 CLIP 순위가 바뀌는지 확인한다.

* BLIP 생성 설명을 화면에 표시한다.

* 입력 이미지가 없거나 후보 문장이 비어 있는 경우를 처리한다.

## 7교시. 통합 서비스 테스트와 개선 (1시간)

### 7.1 기능 테스트

서비스를 구현한 뒤에는 기능이 정상적으로 실행되는지 확인한다.

| 테스트 항목   | 검증 방법       | 기대 결과           |
| -------- | ----------- | --------------- |
| 이미지 업로드  | JPG 파일 선택   | 이미지 미리보기 표시     |
| CLIP 분석  | 후보 문장 3개 입력 | 관련성 점수와 순위 출력   |
| BLIP 캡셔닝 | 이미지 분석 실행   | 설명 문장 생성        |
| 후보 변경    | 문장 표현 수정    | 관련성 순위 재계산      |
| 입력 오류    | 빈 후보 문장     | 안내 메시지 출력       |
| 다양한 이미지  | 동물·풍경·사물 입력 | 각 이미지에 대한 결과 표시 |

### 7.2 성능과 정확도 평가

실제 프로젝트에서는 모델이 정상적으로 실행되는지만 평가하지 않는다.

* CLIP은 정답 레이블이 있는 데이터셋을 사용해 분류 정확도와 클래스별 성능을 확인한다.

* 이미지 검색은 관련 이미지가 상위 순위에 나타나는지 확인한다.

* BLIP은 설명에 포함된 객체, 행동, 속성이 실제 이미지와 일치하는지 검토한다.

* 추론 시간과 메모리 사용량을 측정해 서비스 환경에 적합한지 판단한다.

### 7.3 개선 아이디어

1. 이미지 임베딩을 미리 계산해 저장하고 검색 속도를 높인다.

2. CLIP 후보 문장을 사용자 입력으로 받도록 확장한다.

3. BLIP 설명 생성 결과를 검색 문장 후보로 활용하는 실험을 수행한다.

4. 이미지 설명과 VQA 답변을 한 화면에서 비교한다.

5. 생성 결과의 오류 가능성을 표시하고 사람이 검토할 수 있도록 한다.

## 8교시. 프로젝트 발표와 최종 평가 (1시간)

### 8.1 프로젝트 발표 구성

발표 자료는 다음 순서로 작성하도록 지도한다.

1. 프로젝트 목적 및 해결하려는 문제

2. CLIP과 BLIP을 선택한 이유

3. 전체 시스템 구조도

4. 이미지 입력과 전처리 방식

5. CLIP 검색 결과 및 유사도 해석

6. BLIP 생성 설명과 오류 분석

7. 테스트 결과 및 한계점

8. 향후 개선 방향

### 8.2 최종 평가 기준

| 평가 항목   | 배점   | 세부 기준                   |
| ------- | ---- | ----------------------- |
| 개념 이해   | 20점  | CLIP과 BLIP의 구조·기능 차이 설명 |
| 코드 구현   | 25점  | 모델 로딩, 추론, 결과 출력        |
| 결과 분석   | 20점  | 유사도 및 생성 결과의 정확성 분석     |
| 통합 프로젝트 | 25점  | 검색과 설명 생성 기능 연결         |
| 발표·문서화  | 10점  | 실행 방법, 한계, 개선 방향 정리     |
| 합계      | 100점 |                         |

### 최종 학습 확인

수강생이 다음 질문에 자신의 말로 답할 수 있다면 핵심 학습 목표를 달성한 것이다.

* CLIP은 이미지와 텍스트를 어떻게 비교하는가?

* 공유 임베딩 공간은 왜 필요한가?

* 코사인 유사도와 소프트맥스 점수는 어떻게 다른가?

* BLIP의 이미지 캡셔닝과 VQA는 무엇이 다른가?

* 두 모델을 하나의 서비스에 결합하면 어떤 기능을 제공할 수 있는가?

* 모델이 생성한 설명이 틀릴 수 있다는 점을 서비스에서 어떻게 다룰 것인가?

# 강사용 최종 정리

## 1. 3일 교육의 핵심 연결

Day 1 — 표현 학습 이해

임베딩 → 인코더 → 공유 공간 → 유사도

Day 2 — 사전학습 모델 활용

CLIP Zero-shot 분류·검색 → BLIP 캡셔닝

Day 3 — 응용 및 서비스 구현

BLIP VQA → CLIP·BLIP 통합 → 평가 및 발표

## 2. 수업 진행 시 주의할 사항

* 코드보다 데이터 흐름을 먼저 설명한다. 이미지와 텍스트가 어떤 형태로 변환되고 어떤 결과를 만드는지 이해한 뒤 코드를 실행한다.

* CLIP의 유사도 점수와 정답 확률을 구분한다. 점수가 높다고 해서 반드시 사실적으로 정확한 설명이라는 의미는 아니다.

* BLIP의 생성 결과를 검증한다. 자연스러운 문장도 이미지와 일치하지 않을 수 있다.

* 사전학습 모델과 직접 학습한 모델을 구분한다. 본 과정은 기본적으로 사전학습 모델의 추론과 응용에 초점을 둔다.

* 실습 환경을 사전 점검한다. 모델 체크포인트 다운로드, 라이브러리 버전, GPU 메모리, 이미지 파일 경로를 수업 전에 확인한다.

이 교안은 앞서 작성한 6개 Notebook과 연결해 활용할 수 있다. 이 교재에서는 각 교시의 이론을 설명한 후 해당 Notebook을 실행하고, 수강생에게 결과 분석표를 작성하도록 하면 이론과 코딩을 일관된 흐름으로 연결할 수 있다.

---

# 워크북 점검 질문과 답변

### Q1. 코드를 실행하기 전에 가장 먼저 확인할 것은 무엇인가?

**답변:** 라이브러리 버전, Device, Dataset 로딩 여부, 원본 이미지 해상도, 모델 가중치 다운로드 가능 여부를 확인한다.

### Q2. 모델과 입력 Tensor의 Device가 다르면 어떻게 되는가?

**답변:** CPU Tensor와 CUDA/MPS 모델을 직접 연산할 수 없으므로 Device mismatch 오류가 발생한다. 모델과 입력 Tensor를 같은 Device로 이동해야 한다.

### Q3. `torch.inference_mode()`를 사용하는 이유는 무엇인가?

**답변:** 추론에서는 Gradient 계산이 필요하지 않으므로 Autograd 관련 비용을 줄여 메모리 사용과 연산 부담을 줄일 수 있다.

### Q4. 모델 출력이 예상과 다를 때 무엇부터 확인해야 하는가?

**답변:** 원본 데이터 품질, Processor 입력, Tensor Shape, Prompt, 후보 클래스, 정규화 여부, 모델의 Task 적합성을 순서대로 확인한다.

### Q5. 실행 결과가 자연스러운 문장이라면 정확한 결과라고 볼 수 있는가?

**답변:** 아니다. 특히 Captioning과 VQA는 자연스럽지만 이미지 근거와 다른 문장을 생성할 수 있다. 원본 이미지와 Ground Truth를 기준으로 검증해야 한다.

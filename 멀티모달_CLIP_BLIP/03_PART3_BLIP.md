# PART 3. BLIP: 이미지를 이해하고 언어를 생성하기

앞에서는 CLIP을 이용하여 다음 과정을 학습하였다.

```text
Image                           Text
  ↓                              ↓
Vision Encoder               Text Encoder
  ↓                              ↓
Image Embedding             Text Embedding
  └──────────────┬───────────────┘
                 ↓
        Shared Embedding Space
                 ↓
          Similarity 계산
                 ↓
       Zero-shot / Retrieval
```

이번에는 여기서 한 단계 더 나아간다.

이미지와 문장의 유사성을 계산하는 것에서 끝나는 것이 아니라 **이미지를 보고 새로운 문장을 생성하고, 이미지에 관한 질문에 답하는 방법**을 학습한다.

대표적인 모델이 **BLIP(Bootstrapping Language-Image Pre-training)**이다.

BLIP은 vision-language understanding과 generation을 하나의 프레임워크에서 다루기 위해 제안되었으며, 원 논문에서는 이미지-텍스트 검색, 이미지 캡셔닝, VQA 등의 작업을 함께 다룬다. [arXiv](https://arxiv.org/abs/2201.12086?utm_source=chatgpt.com)

---

# 1. CLIP만으로 해결하기 어려운 문제

먼저 다음 강아지 이미지가 있다고 가정한다.

CLIP에게 다음 세 문장을 제공할 수 있다.

```text
"a photo of a dog"

"a photo of a cat"

"a photo of a car"
```

CLIP은 이미지와 각각의 문장을 비교한다.

```text
                 Dog Image
                     ↓
              Image Encoder
                     ↓
              Image Embedding
                     │
         ┌───────────┼───────────┐
         ↓           ↓           ↓
       0.91        0.23        0.05
         ↑           ↑           ↑
       dog         cat         car
```

따라서 다음과 같은 문제를 해결할 수 있다.

```text
이 이미지와 가장 잘 맞는 문장은 무엇인가?

→ "a photo of a dog"
```

하지만 다음 요청은 성격이 다르다.

```text
"이 사진을 설명해줘."
```

우리가 원하는 답은 예를 들어 다음과 같다.

```text
"a dog running through a grassy field"
```

이 문장을 미리 후보로 제공한 것이 아니다.

모델이 **새로운 token을 순차적으로 생성**해야 한다.

---

# 2. 판별과 생성의 차이

CLIP과 BLIP을 처음 배우는 단계에서는 이 차이를 명확히 구분해야 한다.

### CLIP

```text
후보 Text A
후보 Text B
후보 Text C
      +
    Image
      ↓
   비교
      ↓
가장 적합한 Text 선택
```

### BLIP Captioning

```text
Image
  ↓
이미지 이해
  ↓
Text 생성
  ↓
"a dog is running on the grass"
```

즉,

> **CLIP은 이미지와 텍스트의 관계를 비교하는 데 강하고, BLIP은 이미지 정보를 기반으로 언어를 생성하는 기능까지 제공한다.**

다만 BLIP 자체도 retrieval 등의 understanding task를 지원하도록 설계된 모델이다. 따라서 "CLIP=이해, BLIP=생성"으로 완전히 이분법적으로 구분하면 정확하지 않다. BLIP의 핵심은 **understanding과 generation을 모두 지원하는 통합적인 Vision-Language Pre-training**에 있다. [arXiv](https://arxiv.org/abs/2201.12086?utm_source=chatgpt.com)

---

# 3. Image Captioning이란

Image Captioning은

> **이미지를 입력받아 그 이미지의 내용을 설명하는 자연어 문장을 생성하는 작업**

이다.

예를 들어

```text
Input

[고양이 두 마리가 소파에 있는 사진]

              ↓

        Vision Model

              ↓

       Language Model

              ↓

Output

"two cats sitting on a couch"
```

이다.

이 문제에는 두 가지 능력이 필요하다.

```text
이미지를 이해하는 능력

        +

문장을 생성하는 능력
```

따라서 일반적인 이미지 분류보다 복잡하다.

---

# 4. BLIP Captioning의 기본 구조

Hugging Face의 `BlipForConditionalGeneration`은 이미지 캡셔닝을 위해 **Vision Encoder + Text Decoder** 구조를 제공한다. 텍스트 prompt를 선택적으로 제공할 수도 있다. [Hugging Face](https://huggingface.co/docs/transformers/v4.57.2/en/model_doc/blip?utm_source=chatgpt.com)

개념적으로 보면 다음과 같다.

```text
                Image
                  ↓
            Image Processor
                  ↓
            pixel_values
                  ↓
           Vision Encoder
                  ↓
          Visual Features
                  ↓
            Text Decoder
                  ↓
        다음 Token 예측
                  ↓
        다음 Token 예측
                  ↓
                ...
                  ↓
              Caption
```

예를 들어

```text
Image

 ↓

Vision Encoder

 ↓

Visual Features

 ↓

Text Decoder

 ↓

"a"

 ↓

"a dog"

 ↓

"a dog running"

 ↓

"a dog running on"

 ↓

"a dog running on grass"
```

처럼 문장이 순차적으로 만들어진다고 이해할 수 있다.

---

# 5. BLIP에서 Decoder가 중요한 이유

CLIP에는 기본적인 caption 생성용 Text Decoder가 없다.

반면 BLIP의 Captioning 모델에는 **Text Decoder**가 있다.

Decoder의 핵심 역할은

> **이전까지 생성된 token과 이미지 정보를 이용하여 다음 token을 예측하는 것**

이다.

예를 들어

```text
Image Feature
      +
"a dog is"

      ↓

Text Decoder

      ↓

running
```

다음에는

```text
Image Feature
      +
"a dog is running"

      ↓

Text Decoder

      ↓

on
```

다음에는

```text
Image Feature
      +
"a dog is running on"

      ↓

Text Decoder

      ↓

the
```

처럼 진행된다.

최종적으로

```text
a dog is running on the grass
```

라는 문장이 만들어진다.

---

# 6. BLIP 실습 환경

앞에서 사용한 환경을 그대로 사용한다.

```bash
uv add "torch==2.8.0" "torchvision==0.23.0" "transformers==4.57.2" "pillow>=10,<12" "matplotlib>=3.9,<4"
```

Jupyter 환경이 없다면 추가한다.

```bash
uv add "jupyter>=1,<2" "ipywidgets>=8.1,<9"
```

실행한다.

```bash
uv run jupyter lab
```

---

# 7. 라이브러리와 Device 설정

새로운 Notebook에서 시작해도 실행되도록 처음부터 작성한다.

```python
import torch
import transformers

print("PyTorch     :", torch.__version__)
print("Transformers:", transformers.__version__)

# ---------------------------------------------------------
# 실행 장치를 자동으로 선택한다.
#
# NVIDIA GPU → CUDA
# Apple Silicon → MPS
# 그 외 → CPU
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

print("사용 장치:", device)
```

---

# 8. BLIP Captioning 모델 선택

이미지 캡셔닝에는 다음 모델을 사용한다.

```text
Salesforce/blip-image-captioning-base
```

필요한 클래스는 두 개이다.

```python
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration
)
```

역할은 다음과 같다.

| 클래스 | 역할 |
|---|---|
| `BlipProcessor` | 이미지와 텍스트 전처리 |
| `BlipForConditionalGeneration` | 이미지 기반 텍스트 생성 |

---

# 9. BLIP 모델 로딩

```python
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration
)

# ---------------------------------------------------------
# 사용할 BLIP Image Captioning 모델
# ---------------------------------------------------------

CAPTION_MODEL_NAME = (
    "Salesforce/blip-image-captioning-base"
)

# ---------------------------------------------------------
# Processor를 불러온다.
#
# 이미지 전처리와 텍스트 토큰화를 담당한다.
# ---------------------------------------------------------

caption_processor = (
    BlipProcessor.from_pretrained(
        CAPTION_MODEL_NAME
    )
)

# ---------------------------------------------------------
# Image Captioning 모델을 불러온다.
#
# 주요 구조:
#
# Image
#   ↓
# Vision Encoder
#   ↓
# Text Decoder
#   ↓
# Caption
# ---------------------------------------------------------

caption_model = (
    BlipForConditionalGeneration
    .from_pretrained(
        CAPTION_MODEL_NAME
    )
)

# 모델을 GPU / MPS / CPU로 이동한다.
caption_model = caption_model.to(
    device
)

# 추론 모드
caption_model.eval()

print("BLIP Captioning 모델 로딩 완료")
```

---

# 10. Processor 구조 확인

```python
print(
    type(caption_processor)
)

print()

print(
    caption_processor
)
```

Processor 안에는 이미지 처리와 tokenizer 기능이 포함되어 있다.

개념적으로

```text
BlipProcessor
│
├── Image Processor
│
└── Tokenizer
```

이다.

---

# 11. BLIP 구조 확인

```python
print(
    caption_model
)
```

출력이 상당히 길게 나타난다.

따라서 핵심 부분만 확인한다.

```python
print(
    "Vision Model:"
)

print(
    type(caption_model.vision_model)
)

print()

print(
    "Text Decoder:"
)

print(
    type(caption_model.text_decoder)
)
```

이 구조를 다음과 연결한다.

```text
caption_model
│
├── vision_model
│
│       ↓
│   Image 이해
│
└── text_decoder
        ↓
    Caption 생성
```

---

# 12. 실습 이미지 준비

앞선 CLIP 수업과 동일하게 CIFAR10을 사용할 수 있다.

```python
from torchvision.datasets import CIFAR10

dataset = CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=None
)

print(
    "이미지 수:",
    len(dataset)
)

print(
    "클래스:",
    dataset.classes
)
```

---

# 13. 이미지 확인

```python
import matplotlib.pyplot as plt

image, label = dataset[0]

class_name = dataset.classes[
    label
]

print(
    "정답 클래스:",
    class_name
)

print(
    "원본 이미지 크기:",
    image.size
)

plt.figure(
    figsize=(4, 4)
)

plt.imshow(image)

plt.title(
    class_name
)

plt.axis("off")

plt.show()
```

주의할 점이 있다.

CIFAR10은

```text
32 × 32
```

크기의 저해상도 이미지이다.

BLIP Captioning은 일반적인 자연 이미지에 대한 설명을 생성하는 모델이므로 CIFAR10은 **기능 확인에는 사용할 수 있지만 caption 품질을 평가하기에는 적합하지 않을 수 있다.**

따라서 뒤에서는 보다 큰 샘플 이미지를 함께 사용한다.

---

# 14. BLIP Processor에 이미지 넣기

```python
inputs = caption_processor(
    images=image,
    return_tensors="pt"
)

print(
    inputs.keys()
)
```

텍스트를 제공하지 않았으므로 핵심 입력은

```text
pixel_values
```

이다.

확인한다.

```python
print(
    inputs["pixel_values"].shape
)

print(
    inputs["pixel_values"].dtype
)
```

구조는

```text
[Batch, Channel, Height, Width]
```

이다.

---

# 15. 입력 Tensor의 의미

처리 흐름은 다음과 같다.

```text
PIL Image

    ↓

BlipProcessor

    ↓

Resize / Rescale / Normalize

    ↓

pixel_values

    ↓

Vision Encoder
```

즉 Processor는 모델이 직접 처리할 수 있는 Tensor 형태로 이미지를 변환한다.

---

# 16. 첫 번째 Image Caption 생성

가장 기본적인 BLIP 코드이다.

```python
# ---------------------------------------------------------
# 이미지 전처리
# ---------------------------------------------------------

inputs = caption_processor(
    images=image,
    return_tensors="pt"
)

# ---------------------------------------------------------
# 입력 Tensor를 모델과 같은 장치로 이동한다.
# ---------------------------------------------------------

inputs = {
    key: value.to(device)
    for key, value
    in inputs.items()
}

# ---------------------------------------------------------
# Caption 생성
#
# generate()는 Text Decoder를 이용하여
# token을 순차적으로 생성한다.
# ---------------------------------------------------------

with torch.inference_mode():

    generated_ids = (
        caption_model.generate(
            **inputs,
            max_new_tokens=30
        )
    )

print(
    generated_ids
)
```

아직 결과가 숫자이다.

---

# 17. generated_ids란 무엇인가

Language Model은 문자열을 직접 출력하지 않는다.

먼저 token ID를 생성한다.

```text
Image

 ↓

Vision Encoder

 ↓

Text Decoder

 ↓

Token ID

 ↓

Token ID

 ↓

Token ID

 ↓

...
```

따라서

```python
print(
    generated_ids.shape
)
```

을 실행하면 생성된 token sequence의 형태를 확인할 수 있다.

---

# 18. Token ID를 문장으로 변환

`decode()`를 사용한다.

```python
caption = (
    caption_processor.decode(
        generated_ids[0],
        skip_special_tokens=True
    )
)

print(
    "생성 Caption:"
)

print(
    caption
)
```

처리 과정은 다음과 같다.

```text
generated_ids

[101, ..., 102]

      ↓

Tokenizer Decode

      ↓

"a ship in the water"
```

`skip_special_tokens=True`는 `[CLS]`, `[SEP]`와 같은 특별 token을 최종 문장에서 제거하기 위해 사용한다.

---

# 19. Caption 생성 함수 만들기

앞으로 여러 이미지에서 반복 사용할 수 있도록 함수화한다.

```python
def generate_caption(
    image,
    model,
    processor,
    device,
    max_new_tokens=30
):
    """
    PIL 이미지를 입력받아
    BLIP으로 영문 Caption을 생성한다.

    Parameters
    ----------
    image:
        PIL.Image.Image

    model:
        BlipForConditionalGeneration

    processor:
        BlipProcessor

    device:
        cuda / mps / cpu

    max_new_tokens:
        새롭게 생성할 수 있는
        최대 token 수

    Returns
    -------
    caption:
        생성된 문자열
    """

    # ---------------------------------------------
    # 이미지 → 모델 입력 Tensor
    # ---------------------------------------------

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    # ---------------------------------------------
    # 모든 입력 Tensor를 모델과
    # 동일한 장치로 이동한다.
    # ---------------------------------------------

    inputs = {
        key: value.to(device)
        for key, value
        in inputs.items()
    }

    # ---------------------------------------------
    # Text 생성
    # ---------------------------------------------

    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

    # ---------------------------------------------
    # Token ID → 문자열
    # ---------------------------------------------

    caption = processor.decode(
        generated_ids[0],
        skip_special_tokens=True
    )

    return caption.strip()
```

사용한다.

```python
caption = generate_caption(
    image=image,
    model=caption_model,
    processor=caption_processor,
    device=device
)

print(
    "정답 클래스:",
    class_name
)

print(
    "BLIP Caption:",
    caption
)
```

---

# 20. 여러 이미지에서 Caption 생성하기

```python
NUM_IMAGES = 10

results = []

for index in range(
    NUM_IMAGES
):

    image, label = dataset[
        index
    ]

    class_name = dataset.classes[
        label
    ]

    caption = generate_caption(
        image=image,
        model=caption_model,
        processor=caption_processor,
        device=device
    )

    results.append(
        {
            "index": index,
            "class": class_name,
            "caption": caption
        }
    )

    print(
        f"{index:02d} | "
        f"{class_name:10s} | "
        f"{caption}"
    )
```

여기서 다음 질문을 통해 결과를 점검해보자.

```text
Caption 안에 실제 객체가 올바르게 포함되어 있는가?

색상은 맞는가?

수량은 맞는가?

배경은 맞는가?

모델이 이미지에 없는 내용을 생성하지 않았는가?
```

이러한 관찰이 이후 **Hallucination** 개념으로 연결된다.

---

# 21. Caption 결과 시각화

```python
import matplotlib.pyplot as plt

num_show = min(
    8,
    len(results)
)

fig, axes = plt.subplots(
    2,
    4,
    figsize=(14, 7)
)

axes = axes.flatten()

for ax in axes:
    ax.axis("off")

for ax, result in zip(
    axes,
    results[:num_show]
):

    image, _ = dataset[
        result["index"]
    ]

    ax.imshow(image)

    ax.set_title(
        f"True: {result['class']}\n"
        f"{result['caption']}",
        fontsize=9
    )

    ax.axis("off")

plt.tight_layout()

plt.show()
```

---

# 22. CIFAR10만으로 Caption을 평가하면 안 되는 이유

CIFAR10 이미지는 매우 작다.

```text
32 × 32
```

BLIP Processor가 이를 모델 입력 크기로 확대한다고 해서 원래 이미지에 존재하지 않았던 세부 정보가 복원되는 것은 아니다.

예를 들어

```text
32 × 32

   ↓

Resize

   ↓

384 × 384
```

가 되더라도

```text
정보량이 증가한 것은 아니다.
```

따라서 실제 Captioning 이 교재에서는 고해상도 자연 이미지를 함께 사용하는 것이 좋다.

---

# 23. 고해상도 샘플 이미지 사용하기

BLIP 공식 예제 계열에서 사용되는 Salesforce 데모 이미지를 활용할 수 있다. 다만 외부 URL은 네트워크가 필요하므로 CIFAR10 예제와 병행하는 것이 안전하다.

```python
import requests

from PIL import Image

from io import BytesIO


IMAGE_URL = (
    "https://storage.googleapis.com/"
    "sfr-vision-language-research/"
    "BLIP/demo.jpg"
)

response = requests.get(
    IMAGE_URL,
    timeout=30
)

response.raise_for_status()

demo_image = Image.open(
    BytesIO(
        response.content
    )
).convert("RGB")

print(
    "이미지 크기:",
    demo_image.size
)

plt.figure(
    figsize=(6, 6)
)

plt.imshow(
    demo_image
)

plt.axis("off")

plt.show()
```

네트워크가 없는 환경에서는 이 셀만 건너뛰고 CIFAR10 또는 로컬 이미지를 사용하면 된다.

---

# 24. 고해상도 이미지 Caption 생성

```python
demo_caption = generate_caption(
    image=demo_image,
    model=caption_model,
    processor=caption_processor,
    device=device
)

print(
    "BLIP Caption:"
)

print(
    demo_caption
)
```

이 결과와 CIFAR10 결과를 비교한다.

학습자는 다음 사실을 직접 확인할 수 있다.

```text
모델 성능

=

모델 자체의 성능

+

입력 데이터 품질

+

학습 데이터와 입력 데이터의 관계

+

추론 설정
```

---

# 25. Unconditional Captioning

지금까지는 텍스트를 전혀 제공하지 않았다.

```python
inputs = caption_processor(
    images=demo_image,
    return_tensors="pt"
)
```

즉,

```text
Image

 ↓

BLIP

 ↓

Caption
```

이다.

Hugging Face 문서에서는 텍스트 prompt가 없는 경우 decoder가 시작 token에서 caption 생성을 시작한다고 설명한다. [Hugging Face](https://huggingface.co/docs/transformers/v4.57.2/en/model_doc/blip?utm_source=chatgpt.com)

이를 **Unconditional Image Captioning** 관점으로 이해할 수 있다.

---

# 26. Conditional Captioning

BLIP은 이미지와 함께 텍스트 prompt를 제공할 수도 있다.

예를 들어

```text
"a picture of"
```

를 먼저 제공한다.

```python
prompt = "a picture of"

inputs = caption_processor(
    images=demo_image,
    text=prompt,
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value
    in inputs.items()
}

with torch.inference_mode():

    generated_ids = (
        caption_model.generate(
            **inputs,
            max_new_tokens=30
        )
    )

conditional_caption = (
    caption_processor.decode(
        generated_ids[0],
        skip_special_tokens=True
    )
)

print(
    conditional_caption
)
```

BLIP Captioning 모델은 선택적으로 `input_ids`를 받아 decoder가 주어진 text prompt를 이어서 생성할 수 있다. [Hugging Face](https://huggingface.co/docs/transformers/v4.57.2/en/model_doc/blip?utm_source=chatgpt.com)

---

# 27. Conditional Captioning 구조

```text
             Image
               ↓
        Vision Encoder
               ↓
        Visual Features
               │
               │
               ▼
Text Prompt → Decoder
               ↓
         Text Generation
```

예를 들어

```text
Image
+
"a picture of"

        ↓

BLIP

        ↓

"a picture of a woman sitting with a dog"
```

이다.

---

# 28. Processor 출력 비교

텍스트를 넣지 않았을 때 확인한다.

```python
inputs_without_text = (
    caption_processor(
        images=demo_image,
        return_tensors="pt"
    )
)

print(
    inputs_without_text.keys()
)
```

이번에는 prompt를 제공한다.

```python
inputs_with_text = (
    caption_processor(
        images=demo_image,
        text="a picture of",
        return_tensors="pt"
    )
)

print(
    inputs_with_text.keys()
)
```

텍스트가 포함되면 다음과 같은 입력이 추가된다.

```text
input_ids
attention_mask
```

즉,

```text
Image Only

pixel_values


Image + Text

pixel_values
input_ids
attention_mask
```

라는 차이가 있다.

---

# 29. Prompt의 Token 확인

```python
prompt = "a picture of"

encoded = (
    caption_processor.tokenizer(
        prompt,
        return_tensors="pt"
    )
)

print(
    "Input IDs:"
)

print(
    encoded["input_ids"]
)

tokens = (
    caption_processor.tokenizer
    .convert_ids_to_tokens(
        encoded["input_ids"][0]
    )
)

print()

print(
    "Tokens:"
)

print(
    tokens
)
```

이를 통해 CLIP에서 배웠던 Tokenization이 BLIP에서도 그대로 연결된다는 것을 확인할 수 있다.

---

# 30. `generate()`가 하는 일

학습자들이 자주 오해하는 부분이다.

다음 코드는

```python
caption_model.generate(...)
```

단순히 모델의 `forward()`를 한 번 실행하는 것이 아니다.

개념적으로

```text
Image Feature
    +
Start Token

      ↓

Decoder

      ↓

Token 1

      ↓

Decoder

      ↓

Token 2

      ↓

Decoder

      ↓

Token 3

      ↓

...

      ↓

End Token
```

처럼 반복적인 생성 과정을 관리한다.

즉 `generate()`는 autoregressive generation을 수행하는 고수준 API이다.

---

# 31. Greedy Search 개념

가장 단순한 생성 방법은 매 단계에서 가장 확률이 높은 token을 선택하는 것이다.

```text
현재 출력

"a dog is"

       ↓

다음 Token 확률

running     0.62
sitting     0.18
standing    0.12
sleeping    0.08

       ↓

running 선택
```

이를 반복한다.

```text
a

↓

a dog

↓

a dog is

↓

a dog is running
```

이것이 Greedy 방식의 기본 개념이다.

---

# 32. Beam Search

한 경로만 따라가는 대신 여러 후보 sequence를 유지하면서 탐색할 수도 있다.

```python
inputs = caption_processor(
    images=demo_image,
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value
    in inputs.items()
}

with torch.inference_mode():

    generated_ids = (
        caption_model.generate(
            **inputs,
            max_new_tokens=30,

            # 여러 후보 경로를 유지한다.
            num_beams=5
        )
    )

beam_caption = (
    caption_processor.decode(
        generated_ids[0],
        skip_special_tokens=True
    )
)

print(
    beam_caption
)
```

`num_beams=5`는 생성 과정에서 5개의 후보 경로를 고려한다는 의미이다.

---

# 33. 생성 설정 비교하기

```python
def generate_caption_with_beams(
    image,
    num_beams
):

    inputs = caption_processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value
        in inputs.items()
    }

    with torch.inference_mode():

        output_ids = (
            caption_model.generate(
                **inputs,
                max_new_tokens=30,
                num_beams=num_beams
            )
        )

    return (
        caption_processor.decode(
            output_ids[0],
            skip_special_tokens=True
        ).strip()
    )
```

비교한다.

```python
for beams in [
    1,
    3,
    5
]:

    caption = (
        generate_caption_with_beams(
            demo_image,
            beams
        )
    )

    print(
        f"num_beams={beams}"
    )

    print(
        caption
    )

    print("-" * 60)
```

실습에서는 결과가 항상 더 좋아진다고 단정하지 않는다.

이미지와 생성 조건에 따라 확인이 필요하다.

---

# 34. 이제 VQA로 넘어가기

Captioning에서는

```text
Image
 ↓
"이 이미지를 설명해라"
 ↓
Caption
```

이었다.

VQA에서는 사용자가 구체적인 질문을 제공한다.

```text
Image
+
Question

"What is the woman holding?"

        ↓

BLIP VQA

        ↓

Answer
```

즉 입력 modality는

```text
Image + Text
```

이고 출력은

```text
Text
```

이다.

---

# 35. VQA란 무엇인가

VQA는

**Visual Question Answering**

의 약자이다.

한국어로는 **시각 질의응답** 정도로 이해할 수 있다.

다음 문제를 생각해보자.

```text
Image

[고양이 두 마리]


Question

"How many cats are there?"


Answer

"two"
```

모델은 질문만 읽어서는 답할 수 없다.

이미지만 보아도 무엇을 답해야 하는지 알 수 없다.

두 정보를 함께 이해해야 한다.

```text
Image
   +
Question

   ↓

Multimodal Understanding

   ↓

Answer
```

---

# 36. BLIP VQA 구조

BLIP의 VQA 구조는 Captioning보다 조금 더 복잡하다.

Hugging Face의 `BlipForQuestionAnswering`은 공식적으로 **Vision Encoder + Text Encoder + Text Decoder** 구조로 설명된다. Vision Encoder가 이미지를 인코딩하고, Text Encoder가 질문과 이미지 표현을 함께 처리하며, Text Decoder가 답을 생성한다. [Hugging Face](https://huggingface.co/docs/transformers/v4.57.2/en/model_doc/blip?utm_source=chatgpt.com)

```text
                   Image
                     ↓
               Vision Encoder
                     ↓
              Visual Features
                     │
                     │
Question             │
   ↓                 │
Text Encoder ←───────┘
   ↓
Multimodal Representation
   ↓
Text Decoder
   ↓
Answer
```

---

# 37. Captioning과 VQA 구조 비교

### Captioning

```text
Image
 ↓
Vision Encoder
 ↓
Visual Features
 ↓
Text Decoder
 ↓
Caption
```

### VQA

```text
Image
 ↓
Vision Encoder
 ↓
Visual Features
      │
      │
Question
 ↓    │
Text Encoder
      ↓
Multimodal Representation
      ↓
Text Decoder
      ↓
Answer
```

이 차이를 반드시 설명해야 한다.

---

# 38. VQA 모델 로딩

Captioning 모델과 VQA 모델은 별도로 로딩한다.

```python
from transformers import (
    BlipProcessor,
    BlipForQuestionAnswering
)

VQA_MODEL_NAME = (
    "Salesforce/blip-vqa-base"
)

# ---------------------------------------------------------
# VQA 전용 Processor
# ---------------------------------------------------------

vqa_processor = (
    BlipProcessor.from_pretrained(
        VQA_MODEL_NAME
    )
)

# ---------------------------------------------------------
# VQA 모델
#
# Vision Encoder
# +
# Text Encoder
# +
# Text Decoder
# ---------------------------------------------------------

vqa_model = (
    BlipForQuestionAnswering
    .from_pretrained(
        VQA_MODEL_NAME
    )
)

vqa_model = vqa_model.to(
    device
)

vqa_model.eval()

print(
    "BLIP VQA 모델 로딩 완료"
)
```

---

# 39. VQA 모델 구조 확인

```python
print(
    vqa_model
)
```

핵심 객체를 확인한다.

```python
print(
    "Vision Model:"
)

print(
    type(vqa_model.vision_model)
)

print()

print(
    "Text Encoder:"
)

print(
    type(vqa_model.text_encoder)
)

print()

print(
    "Text Decoder:"
)

print(
    type(vqa_model.text_decoder)
)
```

이를 코드와 구조도로 연결한다.

```text
vqa_model
│
├── vision_model
│
├── text_encoder
└── text_decoder
```

---

# 40. 첫 번째 VQA 실행

데모 이미지를 사용한다.

```python
question = (
    "What is in the image?"
)

inputs = vqa_processor(
    images=demo_image,
    text=question,
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value
    in inputs.items()
}

with torch.inference_mode():

    generated_ids = (
        vqa_model.generate(
            **inputs,
            max_new_tokens=20
        )
    )

answer = (
    vqa_processor.decode(
        generated_ids[0],
        skip_special_tokens=True
    )
)

print(
    "Question:",
    question
)

print(
    "Answer:",
    answer
)
```

Hugging Face 4.57.2 문서 역시 inference 시 이미지와 질문을 Processor에 전달한 뒤 `model.generate(**inputs)`를 이용하고 decode하여 답을 얻는 방법을 제공한다. [Hugging Face](https://huggingface.co/docs/transformers/v4.57.2/en/model_doc/blip?utm_source=chatgpt.com)

---

# 41. VQA Processor 출력 확인

```python
question = (
    "What is the woman doing?"
)

inputs = vqa_processor(
    images=demo_image,
    text=question,
    return_tensors="pt"
)

for key, value in (
    inputs.items()
):

    print(
        key,
        value.shape,
        value.dtype
    )
```

주요 입력은

```text
pixel_values
input_ids
attention_mask
```

이다.

이를 다음처럼 연결한다.

```text
pixel_values
     ↓
Vision Encoder


input_ids
attention_mask
     ↓
Text Encoder
```

---

# 42. 질문 Token 확인

```python
question = (
    "What is the woman doing?"
)

encoded_question = (
    vqa_processor.tokenizer(
        question,
        return_tensors="pt"
    )
)

print(
    encoded_question[
        "input_ids"
    ]
)

tokens = (
    vqa_processor.tokenizer
    .convert_ids_to_tokens(
        encoded_question[
            "input_ids"
        ][0]
    )
)

print(
    tokens
)
```

따라서 VQA에서도 텍스트는

```text
Question
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Text Encoder
```

순서로 처리된다.

---

# 43. VQA 함수로 만들기

```python
def ask_image(
    image,
    question,
    model,
    processor,
    device,
    max_new_tokens=20
):
    """
    이미지와 질문을 입력받아
    BLIP VQA 답변을 생성한다.

    Parameters
    ----------
    image:
        PIL.Image.Image

    question:
        이미지에 대한 영문 질문

    model:
        BlipForQuestionAnswering

    processor:
        BlipProcessor

    device:
        cuda / mps / cpu

    max_new_tokens:
        생성할 최대 token 수

    Returns
    -------
    answer:
        생성된 답변 문자열
    """

    # ---------------------------------------------
    # Image + Question 전처리
    # ---------------------------------------------

    inputs = processor(
        images=image,
        text=question,
        return_tensors="pt"
    )

    # 모델 장치로 이동
    inputs = {
        key: value.to(device)
        for key, value
        in inputs.items()
    }

    # 답변 생성
    with torch.inference_mode():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

    # Token ID → 문자열
    answer = processor.decode(
        generated_ids[0],
        skip_special_tokens=True
    )

    return answer.strip()
```

---

# 44. 하나의 이미지에 여러 질문하기

```python
questions = [
    "What is in the image?",
    "How many people are in the image?",
    "Is there a dog in the image?",
    "Where is the person?",
    "What is the person doing?"
]

for question in questions:

    answer = ask_image(
        image=demo_image,
        question=question,
        model=vqa_model,
        processor=vqa_processor,
        device=device
    )

    print(
        "Q:",
        question
    )

    print(
        "A:",
        answer
    )

    print(
        "-" * 60
    )
```

이 실습에서 중요한 것은 모든 답변을 맞았다고 간주하지 않는 것이다.

이미지를 직접 보면서 사람이 결과를 평가한다.

---

# 45. VQA 질문 유형 분류

학습자들에게 질문을 임의로 만들게 하기 전에 유형을 나누는 것이 좋다.

| 유형 | 질문 예 |
|---|---|
| 객체 | What animal is in the image? |
| 수량 | How many dogs are there? |
| 색상 | What color is the dog? |
| 행동 | What is the person doing? |
| 위치 | Where is the dog? |
| Yes/No | Is there a dog? |
| 관계 | What is next to the person? |

이렇게 하면 VQA 성능을 체계적으로 분석할 수 있다.

---

# 46. VQA 평가 데이터 만들기

간단한 평가 구조를 직접 만든다.

```python
evaluation_questions = [
    {
        "question":
            "Is there a dog in the image?",

        "category":
            "yes_no"
    },
    {
        "question":
            "How many people are in the image?",

        "category":
            "count"
    },
    {
        "question":
            "What animal is in the image?",

        "category":
            "object"
    },
    {
        "question":
            "Where is the person?",

        "category":
            "location"
    },
    {
        "question":
            "What is the person doing?",

        "category":
            "action"
    }
]
```

실행한다.

```python
vqa_results = []

for item in evaluation_questions:

    answer = ask_image(
        image=demo_image,
        question=item["question"],
        model=vqa_model,
        processor=vqa_processor,
        device=device
    )

    result = {
        "category":
            item["category"],

        "question":
            item["question"],

        "answer":
            answer
    }

    vqa_results.append(
        result
    )

    print(
        f"[{item['category']}]"
    )

    print(
        "Q:",
        item["question"]
    )

    print(
        "A:",
        answer
    )

    print(
        "-" * 60
    )
```

---

# 47. 사람 평가를 추가하는 이유

생성형 모델은 단순 정확도만으로 평가하기 어려운 경우가 많다.

예를 들어 정답이

```text
dog
```

인데 모델이

```text
a dog
```

이라고 답했다면 의미적으로 맞다.

문자열 비교를 하면

```python
"dog" == "a dog"
```

은

```text
False
```

이다.

따라서 VQA 교육에서는 다음과 같은 평가도 유용하다.

```text
Correct

Partially Correct

Incorrect
```

또는

```text
2점 = 정확

1점 = 부분적으로 정확

0점 = 오답
```

으로 사람이 평가할 수 있다.

---

# 48. BLIP Captioning 오류 분석

Caption이 자연스럽다고 해서 반드시 이미지에 충실한 것은 아니다.

예를 들어 실제 이미지가

```text
강아지 한 마리가 잔디 위에 있다.
```

인데 모델이

```text
"two dogs playing on the grass"
```

라고 생성했다고 가정하자.

문법은 자연스럽지만

```text
two dogs
```

가 잘못되었다.

이러한 오류는 생성형 Vision-Language Model에서 중요한 문제이다.

---

# 49. Caption 오류 유형

이 교재에서는 다음 기준으로 분석할 수 있다.

| 오류 | 설명 |
|---|---|
| Object Error | 존재하지 않는 객체를 언급 |
| Count Error | 객체 수량 오류 |
| Attribute Error | 색상·크기·상태 오류 |
| Action Error | 행동을 잘못 설명 |
| Spatial Error | 위치·공간관계 오류 |
| Hallucination | 이미지에 근거하지 않은 내용 생성 |

학습 과정에서 단순히

```text
맞았다 / 틀렸다
```

를 판단하게 하는 것보다

```text
왜 틀렸는가?
```

를 분석하게 하는 것이 중요하다.

---

# 50. CLIP과 BLIP을 다시 비교하기

| 항목 | CLIP | BLIP |
|---|---|---|
| 핵심 목적 | 이미지-텍스트 정렬·비교 | 이해 + 생성 |
| Vision Encoder | 있음 | 있음 |
| Text Encoder | 있음 | 작업별 사용 |
| Text Decoder | Caption 생성용 기본 구조에는 없음 | 있음 |
| Shared Embedding | 핵심 | 다양한 학습 목적 중 하나 |
| Zero-shot 분류 | 매우 적합 | 주목적 아님 |
| Text→Image 검색 | 매우 적합 | 가능하지만 이번 실습의 초점 아님 |
| Captioning | 기본 CLIP만으로 직접 생성하지 않음 | 가능 |
| VQA | 기본 CLIP만으로 직접 답변 생성하지 않음 | 가능 |
| 대표 출력 | similarity | generated text |

---

# 51. 같은 이미지를 CLIP과 BLIP에 넣으면

이 부분이 두 모델을 연결하는 핵심이다.

### CLIP

```text
Image

 ↓

Image Embedding

 ↓

Text Embeddings와 비교

 ↓

"a photo of a dog"

Similarity = 0.87
```

### BLIP

```text
Image

 ↓

Vision Encoder

 ↓

Text Decoder

 ↓

"a brown dog sitting on the grass"
```

즉,

```text
CLIP
"어떤 설명과 가장 가까운가?"


BLIP
"어떤 설명을 생성할 것인가?"
```

라는 관점으로 설명할 수 있다.

---

# 52. CLIP + BLIP을 결합하면 무엇을 만들 수 있는가

이제 두 모델을 하나의 시스템으로 결합할 수 있다.

예를 들어 이미지 Gallery가 있다고 하자.

```text
gallery/

dog.jpg
cat.jpg
car.jpg
beach.jpg
food.jpg
street.jpg
```

사용자가

```text
"a dog on grass"
```

라고 검색한다.

CLIP이 관련 이미지를 찾는다.

```text
사용자 Query
     ↓
CLIP Text Encoder
     ↓
Text Embedding
     ↓
Gallery Embedding 비교
     ↓
Top-K Image
```

선택된 이미지에 BLIP을 적용한다.

```text
Top-1 Image
     ↓
BLIP
     ↓
Caption

"a dog running through a field"
```

또 질문도 가능하다.

```text
Top-1 Image

+

"What color is the dog?"

     ↓

BLIP VQA

     ↓

"brown"
```

---

# 53. 통합 시스템 구조

전체 구조를 하나로 연결한다.

```text
                   사용자

                     │
          ┌──────────┴───────────┐
          │                      │

     Text Search             Image Upload

          │                      │
          ▼                      │

    CLIP Text Encoder             │
          │                      │
          ▼                      │

    Query Embedding               │
          │                      │
          ▼                      │

    Image Embeddings              │
          │                      │
          ▼                      │

       Top-K Images ──────────────┘
              │
              │
        ┌─────┴─────┐
        │           │
        ▼           ▼

      BLIP        BLIP VQA

        │           │
        ▼           ▼

     Caption      Answer
```

---

# 54. CLIP + BLIP 통합 환경

새 Notebook에서도 실행할 수 있도록 필요한 모델을 모두 로딩한다.

```python
import torch

from transformers import (
    CLIPModel,
    CLIPProcessor,
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering
)

# ---------------------------------------------------------
# Device
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

print(
    "사용 장치:",
    device
)

# ---------------------------------------------------------
# CLIP
# ---------------------------------------------------------

CLIP_MODEL_NAME = (
    "openai/clip-vit-base-patch32"
)

clip_processor = (
    CLIPProcessor.from_pretrained(
        CLIP_MODEL_NAME
    )
)

clip_model = (
    CLIPModel.from_pretrained(
        CLIP_MODEL_NAME
    )
    .to(device)
)

clip_model.eval()

# ---------------------------------------------------------
# BLIP Captioning
# ---------------------------------------------------------

CAPTION_MODEL_NAME = (
    "Salesforce/blip-image-captioning-base"
)

caption_processor = (
    BlipProcessor.from_pretrained(
        CAPTION_MODEL_NAME
    )
)

caption_model = (
    BlipForConditionalGeneration
    .from_pretrained(
        CAPTION_MODEL_NAME
    )
    .to(device)
)

caption_model.eval()

# ---------------------------------------------------------
# BLIP VQA
# ---------------------------------------------------------

VQA_MODEL_NAME = (
    "Salesforce/blip-vqa-base"
)

vqa_processor = (
    BlipProcessor.from_pretrained(
        VQA_MODEL_NAME
    )
)

vqa_model = (
    BlipForQuestionAnswering
    .from_pretrained(
        VQA_MODEL_NAME
    )
    .to(device)
)

vqa_model.eval()

print(
    "모든 모델 로딩 완료"
)
```

---

# 55. CLIP Gallery Embedding 생성

```python
import torch.nn.functional as F

from torchvision.datasets import CIFAR10


dataset = CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=None
)

class_names = dataset.classes


GALLERY_SIZE = 100


gallery_images = []

gallery_labels = []


for index in range(
    GALLERY_SIZE
):

    image, label = dataset[
        index
    ]

    gallery_images.append(
        image
    )

    gallery_labels.append(
        class_names[label]
    )


# ---------------------------------------------------------
# 이미지 100장을 한 번에 Processor에 전달한다.
# ---------------------------------------------------------

gallery_inputs = (
    clip_processor(
        images=gallery_images,
        return_tensors="pt"
    )
    .to(device)
)


# ---------------------------------------------------------
# Gallery Image Embedding 생성
# ---------------------------------------------------------

with torch.inference_mode():

    gallery_embeddings = (
        clip_model.get_image_features(
            **gallery_inputs
        )
    )


# L2 Normalization
gallery_embeddings = F.normalize(
    gallery_embeddings,
    dim=-1
)


print(
    gallery_embeddings.shape
)
```

---

# 56. CLIP 검색 함수

```python
def clip_search(
    query,
    gallery_embeddings,
    gallery_images,
    gallery_labels,
    model,
    processor,
    device,
    top_k=5
):
    """
    자연어 Query를 이용하여
    가장 유사한 이미지를 검색한다.
    """

    # Text → Tensor
    text_inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True
    ).to(device)

    # Text → Embedding
    with torch.inference_mode():

        text_embedding = (
            model.get_text_features(
                **text_inputs
            )
        )

    # L2 normalization
    text_embedding = F.normalize(
        text_embedding,
        dim=-1
    )

    # Text ↔ Image Similarity
    similarities = (
        text_embedding
        @
        gallery_embeddings.T
    )

    k = min(
        top_k,
        len(gallery_images)
    )

    scores, indices = torch.topk(
        similarities[0],
        k=k
    )

    results = []

    for rank, (
        index,
        score
    ) in enumerate(
        zip(
            indices.tolist(),
            scores.tolist()
        ),
        start=1
    ):

        results.append(
            {
                "rank": rank,
                "index": index,
                "label":
                    gallery_labels[index],
                "score": score,
                "image":
                    gallery_images[index]
            }
        )

    return results
```

---

# 57. BLIP Caption 함수

```python
def blip_caption(
    image,
    model,
    processor,
    device
):
    """
    이미지에 대한 Caption을 생성한다.
    """

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value
        in inputs.items()
    }

    with torch.inference_mode():

        generated_ids = (
            model.generate(
                **inputs,
                max_new_tokens=30
            )
        )

    caption = processor.decode(
        generated_ids[0],
        skip_special_tokens=True
    )

    return caption.strip()
```

---

# 58. BLIP VQA 함수

```python
def blip_vqa(
    image,
    question,
    model,
    processor,
    device
):
    """
    이미지와 질문을 입력받아
    VQA 답변을 생성한다.
    """

    inputs = processor(
        images=image,
        text=question,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value
        in inputs.items()
    }

    with torch.inference_mode():

        generated_ids = (
            model.generate(
                **inputs,
                max_new_tokens=20
            )
        )

    answer = processor.decode(
        generated_ids[0],
        skip_special_tokens=True
    )

    return answer.strip()
```

---

# 59. 통합 실행

먼저 검색한다.

```python
query = "a photo of a dog"

search_results = clip_search(
    query=query,
    gallery_embeddings=gallery_embeddings,
    gallery_images=gallery_images,
    gallery_labels=gallery_labels,
    model=clip_model,
    processor=clip_processor,
    device=device,
    top_k=5
)
```

검색 결과를 출력한다.

```python
print(
    "검색어:",
    query
)

print()

for result in search_results:

    print(
        f"{result['rank']}위 | "
        f"Label: {result['label']:10s} | "
        f"Similarity: "
        f"{result['score']:.4f}"
    )
```

---

# 60. Top-1 이미지에 BLIP 적용

```python
top_image = (
    search_results[0][
        "image"
    ]
)

top_label = (
    search_results[0][
        "label"
    ]
)

caption = blip_caption(
    image=top_image,
    model=caption_model,
    processor=caption_processor,
    device=device
)

print(
    "Dataset Label:",
    top_label
)

print(
    "BLIP Caption:",
    caption
)
```

---

# 61. Top-1 이미지에 질문하기

```python
question = (
    "What animal is in the image?"
)

answer = blip_vqa(
    image=top_image,
    question=question,
    model=vqa_model,
    processor=vqa_processor,
    device=device
)

print(
    "Question:",
    question
)

print(
    "Answer:",
    answer
)
```

---

# 62. 최종 결과 시각화

```python
import matplotlib.pyplot as plt

plt.figure(
    figsize=(6, 6)
)

plt.imshow(
    top_image
)

plt.title(
    f"Query: {query}\n"
    f"Label: {top_label}\n"
    f"Caption: {caption}\n"
    f"Q: {question}\n"
    f"A: {answer}",
    fontsize=10
)

plt.axis("off")

plt.tight_layout()

plt.show()
```

이 코드 하나의 결과 안에 세 가지 멀티모달 기능이 들어 있다.

```text
CLIP
 ↓
Text → Image Retrieval


BLIP Captioning
 ↓
Image → Text


BLIP VQA
 ↓
Image + Question → Answer
```

---

# 63. 최종 실습 과제

수업의 마지막에는 최소 **15장 이상의 이미지**를 사용하도록 하는 것이 적절하다.

폴더 구조는 다음처럼 구성한다.

```text
multimodal_project/
│
├── gallery/
│   ├── dog_01.jpg
│   ├── dog_02.jpg
│   ├── dog_03.jpg
│   ├── cat_01.jpg
│   ├── cat_02.jpg
│   ├── cat_03.jpg
│   ├── car_01.jpg
│   ├── car_02.jpg
│   ├── car_03.jpg
│   └── ...
│
└── multimodal.ipynb
```

최소 세 개 이상의 의미 범주를 사용한다.

예를 들어

```text
Animal

Vehicle

Food

Landscape

Person
```

등을 사용할 수 있다.

---

# 64. 최종 과제에서 구현할 기능

학습자는 하나의 Notebook에서 다음 전체 흐름을 구현한다.

```text
이미지 15장 이상
      ↓
CLIP Image Embedding 생성
      ↓
Embedding 저장
      ↓
자연어 Query 입력
      ↓
CLIP Text Embedding
      ↓
Cosine Similarity
      ↓
Top-5 Image Retrieval
      ↓
검색 이미지 시각화
      ↓
Top-1 Image
      ↓
BLIP Caption 생성
      ↓
질문 5개 입력
      ↓
BLIP VQA
      ↓
결과 평가
      ↓
오류 분석
```

---

# 65. VQA 평가 과제

이미지 **3장 이상**을 선정한다.

각 이미지에 대해 질문 **5개 이상**을 만든다.

따라서 최소

\[
3 \times 5 = 15
\]

개의 VQA 결과를 분석한다.

질문은 한 종류만 사용하지 않는다.

```text
객체 질문

수량 질문

색상 질문

행동 질문

공간 관계 질문
```

을 포함한다.

평가표는 다음처럼 작성할 수 있다.

| Image | 유형 | Question | Answer | 평가 |
|---|---|---|---|---|
| 1 | 객체 | What animal...? | dog | 2 |
| 1 | 수량 | How many...? | two | 1 |
| 1 | 색상 | What color...? | brown | 2 |

평가 기준은

```text
2 = 정확
1 = 부분적으로 정확
0 = 오답
```

으로 한다.

---

# 66. Caption 오류 분석 과제

BLIP이 생성한 Caption 중 잘못된 사례를 최소 **2개 이상** 찾는다.

단순히

```text
Caption이 틀렸다.
```

라고 작성하지 않는다.

다음 형태로 분석한다.

```text
원본 이미지

강아지 한 마리가 잔디밭에 있음


생성 Caption

"two dogs playing in a field"


오류

Count Error


원인 분석

모델이 강아지 객체는 올바르게 인식했지만
객체 수를 잘못 생성하였다.
```

---

# 67. 최종적으로 학습자가 설명할 수 있어야 하는 것

전체 32시간 과정을 마친 후 학습자는 다음 흐름을 코드 없이 설명할 수 있어야 한다.

```text
                    Multimodal AI

                         │
              ┌──────────┴──────────┐
              │                     │
            Image                  Text
              │                     │
              ▼                     ▼
           Encoder                Encoder
              │                     │
              ▼                     ▼
           Feature                Feature
              │                     │
              ▼                     ▼
          Projection             Projection
              │                     │
              └─────────┬───────────┘
                        │
                        ▼
             Shared Embedding Space
                        │
                        ▼
                       CLIP
                   ┌────┴────┐
                   │         │
                   ▼         ▼
              Zero-shot   Retrieval
                            │
                            ▼
                         Image
                            │
                            ▼
                           BLIP
                       ┌────┴────┐
                       │         │
                       ▼         ▼
                    Caption     VQA
```

그리고 코드 수준에서는 다음 관계를 이해해야 한다.

```text
CLIPProcessor
    ↓
pixel_values / input_ids / attention_mask
    ↓
CLIPModel
    ↓
get_image_features()
get_text_features()
    ↓
Embedding
    ↓
F.normalize()
    ↓
Matrix Multiplication
    ↓
Similarity


BlipProcessor
    ↓
pixel_values
    ↓
BlipForConditionalGeneration
    ↓
generate()
    ↓
Caption


BlipProcessor
    ↓
pixel_values + input_ids
    ↓
BlipForQuestionAnswering
    ↓
generate()
    ↓
Answer
```

## 반드시 짚어야 할 BLIP 핵심 개념

| 개념 | 의미 |
|---|---|
| BLIP | Vision-Language Understanding과 Generation을 통합적으로 다루는 VLP 모델 |
| `BlipProcessor` | 이미지와 텍스트를 모델 입력 Tensor로 변환 |
| `pixel_values` | Vision Encoder에 들어가는 이미지 Tensor |
| `input_ids` | 텍스트를 token ID로 변환한 Tensor |
| Vision Encoder | 이미지의 시각적 특징 추출 |
| Text Encoder | VQA에서 질문과 시각 정보를 처리 |
| Text Decoder | Caption이나 Answer 생성 |
| `generate()` | Autoregressive token generation 관리 |
| Image Captioning | Image → Text |
| Conditional Captioning | Image + Prompt → Text |
| VQA | Image + Question → Answer |
| Hallucination | 이미지에 근거하지 않은 내용을 생성하는 현상 |


[Hugging Face Transformers 4.57.2 BLIP 공식 문서](https://huggingface.co/docs/transformers/v4.57.2/en/model_doc/blip?utm_source=chatgpt.com)


---

# 핵심 정리

BLIP은 이미지와 텍스트의 관계를 이해하는 것뿐 아니라 이미지 정보를 바탕으로 새로운 언어를 생성하는 Task까지 다룬다. Image Captioning에서는 Vision Encoder가 이미지의 시각적 특징을 추출하고 Text Decoder가 이를 조건으로 Token을 순차적으로 생성한다.

VQA에서는 Image와 Question을 함께 사용한다. 같은 이미지라도 질문이 달라지면 모델이 집중해야 할 정보가 달라진다. 따라서 VQA 평가는 객체 인식뿐 아니라 속성, 수량, 행동, 공간관계, 질문 이해, Hallucination을 함께 살펴보아야 한다.

Caption이나 VQA 결과는 문자열이 자연스럽다는 이유만으로 정확하다고 판단할 수 없다. 이미지에 실제로 존재하지 않는 객체나 속성을 생성했다면 Hallucination이다. 따라서 사람이 원본 이미지를 확인하는 정성 평가와 가능한 경우 Ground Truth 기반 정량 평가를 함께 사용해야 한다.

# 질문과 답변

### Q1. CLIP과 BLIP의 가장 큰 차이는 무엇인가?

**답변:** CLIP은 이미지와 텍스트를 공통 Embedding 공간에서 비교하는 데 초점이 있고, BLIP은 Vision-Language Understanding과 함께 Captioning처럼 새로운 Text를 생성하는 기능도 제공한다.

### Q2. Image Captioning이 Classification보다 복잡한 이유는 무엇인가?

**답변:** Classification은 미리 정해진 클래스 중 하나를 선택하지만 Captioning은 이미지 내용을 이해한 뒤 여러 Token으로 이루어진 새로운 문장을 순차적으로 생성해야 한다.

### Q3. Text Decoder의 역할은 무엇인가?

**답변:** 이미지의 Visual Feature와 이전까지 생성된 Token을 참고하여 다음 Token을 예측한다. 이 과정을 반복하여 Caption을 완성한다.

### Q4. `generate()`는 무엇을 하는가?

**답변:** 모델의 생성 과정을 반복하여 다음 Token을 선택하고, 종료 조건이나 최대 Token 수에 도달할 때까지 Sequence를 만든다.

### Q5. `generated_ids`를 바로 읽을 수 없는 이유는 무엇인가?

**답변:** 모델은 문자열이 아니라 Token ID Sequence를 생성한다. Processor 또는 Tokenizer의 `decode()`를 이용해 Token ID를 문자열로 변환해야 한다.

### Q6. VQA의 입력은 무엇인가?

**답변:** 기본적으로 Image와 Question이다. 모델은 이미지 정보와 질문의 의미를 함께 이용하여 Answer를 생성한다.

### Q7. Captioning과 VQA의 차이는 무엇인가?

**답변:** Captioning은 이미지 전체를 설명하는 문장을 생성하는 것이 목적이고, VQA는 이미지에 대한 특정 질문에 필요한 정보를 찾아 답하는 것이 목적이다.

### Q8. Hallucination이란 무엇인가?

**답변:** 이미지나 제공된 근거에서 확인할 수 없는 객체, 속성, 관계 등을 모델이 사실처럼 생성하는 현상이다.

### Q9. 저해상도 이미지가 Caption/VQA 평가에 불리한 이유는 무엇인가?

**답변:** 객체의 세부 형태, 색상, 수량, 배경, 공간관계 같은 정보가 원본에서 이미 손실되어 있을 수 있기 때문이다. 단순 Resize로 손실된 정보를 복원할 수 없다.

# 확인문제

1. BLIP Captioning의 전체 흐름을 작성하라.
2. Caption 생성에서 Vision Encoder와 Text Decoder의 역할을 각각 설명하라.
3. VQA에서 Question이 필요한 이유를 설명하라.
4. Caption 평가 항목을 다섯 가지 이상 제시하라.
5. Hallucination 사례를 어떻게 찾아낼 수 있는지 설명하라.

## 확인문제 해설

1. Original Image → Processor → pixel_values → Vision Encoder → Visual Features → Text Decoder → generated_ids → decode → Caption 순서이다.
2. Vision Encoder는 이미지 특징을 추출하고 Text Decoder는 그 특징을 조건으로 다음 Token을 반복적으로 생성한다.
3. 질문은 이미지에서 어떤 정보를 찾아야 하는지 모델에 조건을 제공한다.
4. Object, Attribute, Count, Action, Spatial Relation, Fluency, Hallucination 등을 사용할 수 있다.
5. 원본 이미지와 생성 문장을 직접 비교하고 이미지에 존재하지 않는 객체·속성·수량·관계가 포함되었는지 확인한다.

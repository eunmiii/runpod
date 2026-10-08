# PART 2. CLIP: 이미지와 텍스트를 하나의 의미 공간에서 연결하기

앞에서 다음 흐름까지 학습하였다.

```text
Image / Text
     ↓
Encoder
     ↓
Feature
     ↓
Projection
     ↓
Embedding
     ↓
Shared Embedding Space
     ↓
Cosine Similarity
     ↓
Contrastive Learning
```

이제 이 개념을 실제로 구현한 대표적인 멀티모달 모델인 **CLIP(Contrastive Language-Image Pre-training)**을 다룬다.

CLIP은 이미지와 자연어 설명의 대응 관계를 대규모로 학습하고, 학습 후에는 별도의 분류기를 새로 학습하지 않고도 자연어를 이용해 Zero-shot 분류를 수행할 수 있도록 설계된 모델이다. 원 논문은 인터넷에서 수집한 약 4억 개의 이미지-텍스트 쌍을 사용해 학습했다고 설명한다. [arXiv](https://arxiv.org/abs/2103.00020?utm_source=chatgpt.com)

---

# 1. 기존 이미지 분류 모델부터 다시 생각하기

CNN이나 Vision Transformer를 이용한 일반적인 이미지 분류는 다음과 같다.

```text
                Image
                  ↓
             CNN / ViT
                  ↓
               Feature
                  ↓
             Classifier
                  ↓
        ┌─────────┼─────────┐
       Dog       Cat       Car
```

예를 들어 CIFAR10 분류기를 학습하였다면 출력 뉴런은 10개이다.

```python
nn.Linear(
    in_features=512,
    out_features=10
)
```

여기서 `out_features=10`은 분류할 클래스가 10개라는 뜻이다.

문제는 새로운 클래스를 추가하려면 일반적으로 분류 구조나 학습 데이터 측면에서 추가 작업이 필요하다는 점이다.

CLIP은 접근 방법을 바꾼다.

```text
                    Image
                      ↓
                Image Encoder
                      ↓
               Image Embedding
                      │
                      │ similarity
                      ↓
          ┌───────────┼────────────┐
          ↓           ↓            ↓

   "a photo of   "a photo of   "a photo of
      a dog"        a cat"        a car"

          ↓           ↓            ↓
              Text Encoder
```

CLIP은 `Dog`, `Cat`, `Car`라는 고정된 출력 뉴런을 선택하는 대신 이미지와 각 텍스트의 **유사도**를 계산한다.

---

# 2. CLIP이라는 이름의 의미

CLIP은

**Contrastive Language-Image Pre-training**

의 약자이다.

각 단어를 나누어 보면 구조가 잘 드러난다.

| 단어 | 의미 |
|---|---|
| Contrastive | 서로 비교하며 학습 |
| Language | 텍스트 |
| Image | 이미지 |
| Pre-training | 대규모 데이터로 사전학습 |

즉,

> **이미지와 언어의 관계를 Contrastive Learning으로 사전학습한 모델**

이라고 이해할 수 있다.

---

# 3. CLIP 전체 구조

CLIP에는 크게 두 개의 Encoder가 있다.

```text
                  CLIP

       Image                 Text
         │                     │
         ▼                     ▼
   Image Encoder          Text Encoder
         │                     │
         ▼                     ▼
   Image Feature          Text Feature
         │                     │
         ▼                     ▼
Image Projection        Text Projection
         │                     │
         ▼                     ▼
 Image Embedding        Text Embedding
         │                     │
         └──────────┬──────────┘
                    │
                    ▼
          Shared Embedding Space
                    │
                    ▼
               Similarity
```

우리가 사용할 모델은

```text
openai/clip-vit-base-patch32
```

이다.

이 모델에서는 이미지 Encoder로 Vision Transformer 계열 구조를 사용한다.

---

# 4. CLIP을 이용한 이미지 분류 과정

강아지 사진 한 장이 있다고 가정한다.

후보 클래스는 다음과 같다.

```text
dog
cat
car
```

이를 바로 사용하는 것보다 CLIP에서는 자연어 prompt로 만드는 방법을 많이 사용한다.

```text
"a photo of a dog"
"a photo of a cat"
"a photo of a car"
```

처리 과정은 다음과 같다.

```text
                     Dog Image
                         ↓
                  Vision Encoder
                         ↓
                  Image Embedding
                         │
           ┌─────────────┼─────────────┐
           │             │             │
           ▼             ▼             ▼
         0.91          0.27          0.08
           ▲             ▲             ▲
           │             │             │
      dog text       cat text      car text
      embedding      embedding     embedding
```

가장 높은 점수가 `dog`이면

```text
Prediction = dog
```

으로 판단한다.

---

# 5. 실습 환경 구성

이 이 교재에서는 다음 버전을 기준으로 한다.

```bash
uv add "torch==2.8.0" "torchvision==0.23.0" "transformers==4.57.2" "pillow>=10,<12" "matplotlib>=3.9,<4"
```

Jupyter Notebook을 사용한다면 추가한다.

```bash
uv add "jupyter>=1,<2" "ipywidgets>=8.1,<9"
```

실행한다.

```bash
uv run jupyter lab
```

Hugging Face Transformers 4.57.2 공식 문서에서 `CLIPProcessor`는 이미지 프로세서와 tokenizer를 함께 감싸며, `CLIPModel`은 `pixel_values`, `input_ids`, `attention_mask` 등을 입력받는다. [Hugging Face](https://huggingface.co/docs/transformers/v4.57.2/model_doc/clip?utm_source=chatgpt.com)

---

# 6. 라이브러리 확인

첫 번째 Notebook 셀에서 실행한다.

```python
import torch
import torchvision
import transformers

print("PyTorch     :", torch.__version__)
print("TorchVision :", torchvision.__version__)
print("Transformers:", transformers.__version__)
```

기준 환경에서는 다음 버전을 확인할 수 있다.

```text
PyTorch     : 2.8.0
TorchVision : 0.23.0
Transformers: 4.57.2
```

버전이 다르면 코드 실행 자체가 반드시 실패한다는 의미는 아니지만, 강의 환경을 동일하게 맞추는 것이 오류 재현과 해결에 유리하다.

---

# 7. Device 설정

앞으로 모든 코드에서 동일한 device를 사용한다.

```python
import torch

# NVIDIA GPU가 있으면 CUDA를 사용한다.
if torch.cuda.is_available():
    device = torch.device("cuda")

# Apple Silicon Mac에서는 MPS를 사용할 수 있다.
elif (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
):
    device = torch.device("mps")

# GPU를 사용할 수 없다면 CPU를 사용한다.
else:
    device = torch.device("cpu")

print("사용 장치:", device)
```

또는 간단하게 다음처럼 작성해도 된다.

```python
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(device)
```

---

# 8. CLIP 모델 불러오기

```python
import torch
from transformers import CLIPModel, CLIPProcessor

# Hugging Face Hub에 공개되어 있는 CLIP 모델 ID이다.
MODEL_NAME = "openai/clip-vit-base-patch32"

# ---------------------------------------------------------
# CLIPProcessor
#
# 이미지:
#     resize, crop, rescale, normalize 등을 수행한다.
#
# 텍스트:
#     tokenizer를 이용하여 문자열을 token ID로 변환한다.
#
# 따라서 이미지와 텍스트의 전처리를 하나의 객체가 담당한다.
# ---------------------------------------------------------

processor = CLIPProcessor.from_pretrained(
    MODEL_NAME
)

# ---------------------------------------------------------
# 사전학습된 CLIP 모델을 불러온다.
# ---------------------------------------------------------

model = CLIPModel.from_pretrained(
    MODEL_NAME
)

# 선택한 연산 장치로 모델을 이동한다.
model = model.to(device)

# 학습이 아니라 추론에 사용할 것이므로
# evaluation mode로 변경한다.
model.eval()

print("모델 로딩 완료")
print("사용 장치:", device)
```

---

# 9. CLIPProcessor는 왜 필요한가

모델에 다음과 같이 직접 문자열을 전달할 수는 없다.

```python
# 이렇게 사용할 수 없다.
model("a photo of a dog")
```

이미지도 PIL Image 상태 그대로 모델 내부 Transformer에 전달되는 것이 아니다.

CLIP 모델은 최종적으로 **Tensor**를 입력받는다.

```text
Image
 ↓
CLIPProcessor
 ↓
pixel_values
 ↓
CLIP Vision Model


Text
 ↓
CLIPProcessor
 ↓
input_ids
attention_mask
 ↓
CLIP Text Model
```

따라서 Processor는 모델과 실제 데이터를 연결하는 전처리 계층이다.

---

# 10. Processor 내부 구성 확인

```python
print(processor)
```

좀 더 직접적으로 확인한다.

```python
print(
    "Image Processor:"
)

print(
    type(processor.image_processor)
)

print()

print(
    "Tokenizer:"
)

print(
    type(processor.tokenizer)
)
```

즉 `CLIPProcessor` 하나 안에 크게 다음 두 기능이 들어 있다.

```text
CLIPProcessor
│
├── Image Processor
│
└── Tokenizer
```

Hugging Face 4.57.2 문서도 `CLIPProcessor`를 CLIP image processor와 tokenizer를 결합한 객체로 정의한다. [Hugging Face](https://huggingface.co/docs/transformers/v4.57.2/model_doc/clip?utm_source=chatgpt.com)

---

# 11. CIFAR10을 이용해 실습 이미지 준비하기

외부 이미지 URL은 네트워크 상태나 원본 사이트 변경에 따라 강의 중 실패할 수 있다.

따라서 기본 실습에서는 CIFAR10을 사용한다.

```python
from torchvision.datasets import CIFAR10

# ---------------------------------------------------------
# CIFAR10 테스트 데이터를 다운로드한다.
#
# transform=None:
# 원본 PIL Image 형태로 가져오기 위함이다.
#
# 이미지 전처리는 이후 CLIPProcessor가 수행한다.
# ---------------------------------------------------------

dataset = CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=None
)

print(
    "전체 테스트 이미지:",
    len(dataset)
)

print(
    "클래스:",
    dataset.classes
)
```

CIFAR10에는 다음 클래스가 있다.

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

---

# 12. 이미지 한 장 확인하기

```python
import matplotlib.pyplot as plt

image, label = dataset[0]

class_name = dataset.classes[label]

print("Label Index:", label)
print("Class Name :", class_name)
print("Image Size :", image.size)

plt.figure(figsize=(4, 4))
plt.imshow(image)
plt.title(class_name)
plt.axis("off")
plt.show()
```

CIFAR10 원본 이미지는

```text
32 × 32
```

크기이다.

CLIPProcessor가 모델 입력에 맞도록 이미지를 전처리한다.

---

# 13. 이미지 Processor 동작 확인

이미지만 Processor에 넣어보자.

```python
image_inputs = processor(
    images=image,
    return_tensors="pt"
)

print(image_inputs)
```

Dictionary와 비슷한 객체가 반환된다.

키를 확인한다.

```python
print(
    image_inputs.keys()
)
```

핵심 결과는

```text
pixel_values
```

이다.

shape을 확인한다.

```python
print(
    image_inputs["pixel_values"].shape
)
```

`clip-vit-base-patch32` 기준으로 일반적으로 다음과 같다.

```text
torch.Size([1, 3, 224, 224])
```

각 차원의 의미는

```text
[
 Batch,
 Channel,
 Height,
 Width
]
```

이다.

따라서

```text
[1, 3, 224, 224]
```

는

```text
이미지 1장
RGB 3채널
높이 224
너비 224
```

라는 의미이다.

공식 문서에서도 CLIP의 `pixel_values`는 `(batch_size, num_channels, image_size, image_size)` 구조의 Tensor로 설명된다. [Hugging Face](https://huggingface.co/docs/transformers/v4.57.2/model_doc/clip?utm_source=chatgpt.com)

---

# 14. Processor가 이미지에 하는 일

개념적으로 다음 과정이 수행된다.

```text
PIL Image

    ↓

Resize

    ↓

Center Crop

    ↓

Rescale

    ↓

Normalize

    ↓

Tensor

    ↓

[1, 3, 224, 224]
```

사전학습 모델을 사용할 때는 **모델이 학습될 당시 사용한 전처리 규칙과 동일한 규칙을 적용하는 것이 중요하다.**

따라서 임의로

```python
transforms.Resize(...)
transforms.Normalize(...)
```

를 작성하는 것보다 해당 모델의 Processor를 사용하는 것이 안전하다.

Hugging Face의 이미지 processor는 pretrained 모델의 설정에 따라 resize, center crop, rescale, normalize 등을 수행한다. [Hugging Face](https://huggingface.co/docs/transformers/main/image_processors?utm_source=chatgpt.com)

---

# 15. 텍스트 Processor 확인

이번에는 텍스트를 넣는다.

```python
texts = [
    "a photo of a dog",
    "a photo of a cat",
    "a photo of a car"
]

text_inputs = processor(
    text=texts,
    return_tensors="pt",
    padding=True
)

print(
    text_inputs.keys()
)
```

주요 결과는 다음 두 가지이다.

```text
input_ids
attention_mask
```

확인한다.

```python
print(
    "input_ids shape:",
    text_inputs["input_ids"].shape
)

print(
    "attention_mask shape:",
    text_inputs["attention_mask"].shape
)
```

---

# 16. input_ids란 무엇인가

모델은

```text
"a photo of a dog"
```

이라는 문자열 자체를 계산하지 않는다.

Tokenizer가 문장을 token으로 나누고 각 token을 숫자로 변환한다.

개념적으로

```text
"a photo of a dog"

        ↓

Tokenizer

        ↓

token
token
token
token

        ↓

Token ID

        ↓

[49406, ..., 49407]
```

가 된다.

실제 값을 확인한다.

```python
for i, text in enumerate(texts):

    print(
        "원문:",
        text
    )

    print(
        "input_ids:",
        text_inputs["input_ids"][i]
    )

    print()
```

---

# 17. Token을 다시 확인하기

숫자만 보면 의미를 알기 어렵다.

각 ID를 token으로 다시 변환해보자.

```python
for text in texts:

    encoded = processor.tokenizer(
        text,
        return_tensors="pt"
    )

    token_ids = encoded[
        "input_ids"
    ][0]

    tokens = (
        processor.tokenizer
        .convert_ids_to_tokens(
            token_ids
        )
    )

    print("문장:", text)
    print("Token IDs:", token_ids.tolist())
    print("Tokens:", tokens)
    print()
```

이를 통해 다음 과정을 직접 확인할 수 있다.

```text
문장
 ↓
Tokenizer
 ↓
Token
 ↓
Token ID
 ↓
Text Transformer
```

---

# 18. attention_mask는 무엇인가

여러 문장의 길이는 서로 다를 수 있다.

예를 들어

```text
"a dog"

"a beautiful dog running on grass"
```

는 길이가 다르다.

Batch 처리를 위해 길이를 맞출 때 padding이 필요하다.

```text
문장 1

token token PAD PAD


문장 2

token token token token
```

`attention_mask`는 실제 token과 padding 영역을 구분한다.

개념적으로

```text
input_ids

[10, 20, 30, PAD, PAD]

attention_mask

[ 1,  1,  1,   0,   0]
```

가 된다.

공식 CLIP API에서도 attention mask의 `1`은 masking하지 않는 token, `0`은 masking되는 위치를 나타낸다. [Hugging Face](https://huggingface.co/docs/transformers/v4.57.2/model_doc/clip?utm_source=chatgpt.com)

---

# 19. 이미지와 텍스트를 동시에 Processor에 전달하기

이제 실제 CLIP 방식으로 입력을 만든다.

```python
texts = [
    "a photo of an airplane",
    "a photo of an automobile",
    "a photo of a bird",
    "a photo of a cat",
    "a photo of a deer",
    "a photo of a dog",
    "a photo of a frog",
    "a photo of a horse",
    "a photo of a ship",
    "a photo of a truck"
]

inputs = processor(
    text=texts,
    images=image,
    return_tensors="pt",
    padding=True
)

print(
    inputs.keys()
)
```

출력에는 대체로

```text
input_ids
attention_mask
pixel_values
```

가 들어 있다.

각 shape을 확인한다.

```python
for key, value in inputs.items():

    print(
        key,
        value.shape
    )
```

---

# 20. Tensor를 Device로 이동하기

모델이 GPU에 있는데 데이터가 CPU에 있으면 계산할 수 없다.

따라서 입력도 같은 device로 이동한다.

```python
inputs = {
    key: value.to(device)
    for key, value
    in inputs.items()
}
```

Transformers의 `BatchEncoding`은 `.to()`도 지원하므로 다음처럼 사용할 수도 있다.

```python
inputs = processor(
    text=texts,
    images=image,
    return_tensors="pt",
    padding=True
).to(device)
```

이 교재에서는 두 번째 방법이 간결하다.

---

# 21. CLIP 추론 실행

```python
inputs = processor(
    text=texts,
    images=image,
    return_tensors="pt",
    padding=True
).to(device)

with torch.inference_mode():

    outputs = model(
        **inputs
    )

print(
    type(outputs)
)
```

`model(**inputs)`는 개념적으로 다음을 실행한다.

```text
pixel_values
      ↓
Vision Encoder
      ↓
Image Projection
      ↓
Image Embedding
      │
      │
      │
      │ similarity
      │
      │
Text Embedding
      ↑
Text Projection
      ↑
Text Encoder
      ↑
input_ids
attention_mask
```

---

# 22. CLIP 출력 확인

```python
print(
    outputs.keys()
)
```

중요한 출력에는 다음이 있다.

```text
logits_per_image
logits_per_text
text_embeds
image_embeds
```

각각 확인한다.

```python
print(
    "Image Embedding:",
    outputs.image_embeds.shape
)

print(
    "Text Embedding:",
    outputs.text_embeds.shape
)

print(
    "logits_per_image:",
    outputs.logits_per_image.shape
)

print(
    "logits_per_text:",
    outputs.logits_per_text.shape
)
```

이미지 1장과 텍스트 10개라면 개념적으로

```text
Image Embedding
[1, embedding_dim]

Text Embedding
[10, embedding_dim]

logits_per_image
[1, 10]

logits_per_text
[10, 1]
```

이 된다.

---

# 23. `logits_per_image` 이해하기

가장 중요한 값이다.

```python
print(
    outputs.logits_per_image
)
```

이미지 하나와 텍스트 10개의 대응 점수를 담고 있다.

```text
Image

  │
  ├── airplane     score
  ├── automobile   score
  ├── bird         score
  ├── cat          score
  ├── deer         score
  ├── dog          score
  ├── frog         score
  ├── horse        score
  ├── ship         score
  └── truck        score
```

Hugging Face 공식 예제 역시 `outputs.logits_per_image`를 image-text similarity score로 사용한다. [Hugging Face](https://huggingface.co/docs/transformers/v4.57.2/model_doc/clip?utm_source=chatgpt.com)

---

# 24. Softmax 적용하기

logit을 사람이 해석하기 쉬운 상대적인 확률 형태로 변환한다.

```python
probs = (
    outputs
    .logits_per_image
    .softmax(dim=1)
)

print(probs)
```

확률 합계를 확인한다.

```python
print(
    probs.sum(dim=1)
)
```

결과는 거의

```text
tensor([1.])
```

이 된다.

중요한 점은 이것을 **절대적인 현실 세계의 확률**로 해석하지 않는 것이다.

현재 제공한 후보 텍스트들 사이에서 softmax를 계산한 상대적 점수이다.

---

# 25. Zero-shot 분류 결과 출력

```python
probs_cpu = (
    probs[0]
    .detach()
    .cpu()
)

for text, prob in zip(
    texts,
    probs_cpu
):

    print(
        f"{text:30s}"
        f"{prob.item():.4f}"
    )
```

가장 높은 클래스를 찾는다.

```python
pred_index = (
    probs_cpu
    .argmax()
    .item()
)

pred_text = texts[
    pred_index
]

print()
print(
    "정답:",
    dataset.classes[label]
)

print(
    "예측 Prompt:",
    pred_text
)
```

---

# 26. 분류용 함수로 정리하기

실제 이 교재에서는 반복 실행할 수 있도록 함수화한다.

```python
def classify_image(
    image,
    class_names,
    model,
    processor,
    device
):
    """
    CLIP을 이용하여 이미지 한 장을
    Zero-shot 방식으로 분류한다.

    Parameters
    ----------
    image:
        PIL.Image.Image

    class_names:
        분류할 클래스 이름 목록

    model:
        CLIPModel

    processor:
        CLIPProcessor

    device:
        cuda / mps / cpu

    Returns
    -------
    pred_class:
        가장 높은 점수를 받은 클래스

    probabilities:
        각 클래스의 상대적인 softmax 점수
    """

    # ---------------------------------------------
    # 클래스 이름을 자연어 prompt로 변환한다.
    #
    # 예:
    # dog
    # ↓
    # a photo of a dog
    # ---------------------------------------------

    prompts = [
        f"a photo of a {name}"
        for name in class_names
    ]

    # 이미지와 모든 텍스트를 전처리한다.
    inputs = processor(
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    # 추론만 수행하므로
    # gradient 계산을 하지 않는다.
    with torch.inference_mode():

        outputs = model(
            **inputs
        )

    # 이미지와 각 텍스트의 logit을
    # softmax로 변환한다.
    probabilities = (
        outputs
        .logits_per_image
        .softmax(dim=1)[0]
    )

    # 가장 높은 확률의 index를 구한다.
    pred_index = (
        probabilities
        .argmax()
        .item()
    )

    # index에 해당하는 클래스 이름을 가져온다.
    pred_class = class_names[
        pred_index
    ]

    return (
        pred_class,
        probabilities.detach().cpu()
    )
```

사용한다.

```python
pred_class, probabilities = (
    classify_image(
        image=image,
        class_names=dataset.classes,
        model=model,
        processor=processor,
        device=device
    )
)

print("정답:", class_name)
print("예측:", pred_class)
```

---

# 27. 여기서 Zero-shot이라는 말의 정확한 의미

중요하다.

우리는 지금 CIFAR10의 학습 데이터로 CLIP을 학습하지 않았다.

다음 작업도 하지 않았다.

```python
optimizer.zero_grad()

loss.backward()

optimizer.step()
```

즉,

```text
CIFAR10 Training

없음
```

이다.

대신 클래스 이름을 자연어로 제공하였다.

```text
"a photo of an airplane"
"a photo of an automobile"
...
"a photo of a truck"
```

CLIP이 사전학습 과정에서 얻은 이미지-언어 관계를 이용해 새로운 분류 작업을 수행한다.

이를 **Zero-shot Classification**이라고 한다.

---

# 28. CLIP 내부 Embedding 직접 가져오기

이번에는 `model(**inputs)`에 맡기지 않고 이미지와 텍스트 embedding을 각각 가져온다.

먼저 이미지이다.

```python
image_inputs = processor(
    images=image,
    return_tensors="pt"
).to(device)

with torch.inference_mode():

    image_features = (
        model.get_image_features(
            **image_inputs
        )
    )

print(
    "Image Features Shape:",
    image_features.shape
)
```

Transformers 4.57.2 공식 API에서 `get_image_features()`는 CLIP Vision Model의 pooled output에 projection layer를 적용한 이미지 embedding을 반환한다. [Hugging Face](https://huggingface.co/docs/transformers/v4.57.2/model_doc/clip?utm_source=chatgpt.com)

---

# 29. Text Embedding 가져오기

```python
prompts = [
    f"a photo of a {name}"
    for name in dataset.classes
]

text_inputs = processor(
    text=prompts,
    return_tensors="pt",
    padding=True
).to(device)

with torch.inference_mode():

    text_features = (
        model.get_text_features(
            **text_inputs
        )
    )

print(
    "Text Features Shape:",
    text_features.shape
)
```

`get_text_features()` 역시 Text Model의 pooled output에 projection layer를 적용한 embedding을 반환한다. [Hugging Face](https://huggingface.co/docs/transformers/v4.57.2/model_doc/clip?utm_source=chatgpt.com)

---

# 30. Image와 Text의 차원이 같아졌다

확인한다.

```python
print(
    image_features.shape
)

print(
    text_features.shape
)
```

이것이 앞에서 배웠던

**Shared Embedding Space**

와 연결되는 지점이다.

```text
Image
 ↓
Vision Transformer
 ↓
Image Feature
 ↓
Projection
 ↓
Image Embedding
        │
        │ 같은 차원
        │
Text Embedding
 ↑
Projection
 ↑
Text Transformer
 ↑
Text
```

---

# 31. Embedding 정규화

CLIP의 embedding을 직접 비교할 때는 L2 normalization을 적용한다.

```python
import torch.nn.functional as F

image_features = F.normalize(
    image_features,
    p=2,
    dim=-1
)

text_features = F.normalize(
    text_features,
    p=2,
    dim=-1
)
```

벡터 크기를 확인한다.

```python
print(
    torch.linalg.vector_norm(
        image_features,
        dim=-1
    )
)

print(
    torch.linalg.vector_norm(
        text_features,
        dim=-1
    )
)
```

각 embedding의 길이가 거의 1이 된다.

---

# 32. 직접 Cosine Similarity 계산

정규화했으므로 행렬곱을 이용한다.

```python
similarities = (
    image_features
    @
    text_features.T
)

print(
    similarities.shape
)

print(
    similarities
)
```

구조는

```text
               Text

Image       airplane
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

이다.

---

# 33. 가장 유사한 Text 찾기

```python
best_index = (
    similarities[0]
    .argmax()
    .item()
)

print(
    "가장 유사한 Prompt:"
)

print(
    prompts[best_index]
)

print(
    "Cosine Similarity:"
)

print(
    similarities[
        0,
        best_index
    ].item()
)
```

이 코드는 CLIP의 Zero-shot classification 원리를 가장 직접적으로 보여준다.

---

# 34. `model(**inputs)`과 직접 계산의 차이

처음에는

```python
outputs = model(**inputs)

outputs.logits_per_image
```

를 사용했다.

지금은

```python
image_features = model.get_image_features(...)

text_features = model.get_text_features(...)

image_features = F.normalize(...)

text_features = F.normalize(...)

similarity = (
    image_features
    @
    text_features.T
)
```

를 사용했다.

둘은 같은 개념을 서로 다른 수준에서 확인하는 것이다.

```text
간편한 방법

model(**inputs)
      ↓
logits_per_image


원리를 확인하는 방법

get_image_features()
get_text_features()
      ↓
Normalize
      ↓
Matrix Multiplication
      ↓
Similarity
```

단, `logits_per_image`는 단순 cosine similarity 값 그 자체가 아니라 CLIP의 logit scale이 적용된 similarity score라는 점을 구분해야 한다.

---

# 35. CLIP의 `logit_scale`

모델 내부를 확인해보자.

```python
print(
    model.logit_scale
)
```

실제 scale을 확인하려면 exponent를 적용한다.

```python
scale = (
    model.logit_scale
    .exp()
    .item()
)

print(
    "Logit Scale:",
    scale
)
```

개념적으로 CLIP의 logit은

\[
logit =
scale \times cosine\ similarity
\]

형태이다.

즉,

```text
Normalized Image Embedding
             │
             │
             ▼
        Cosine Similarity
             │
             ▼
         Logit Scale
             │
             ▼
            Logit
```

이 된다.

---

# 36. CLIP 계산을 직접 재현하기

```python
with torch.inference_mode():

    image_features = (
        model.get_image_features(
            **image_inputs
        )
    )

    text_features = (
        model.get_text_features(
            **text_inputs
        )
    )

image_features = F.normalize(
    image_features,
    dim=-1
)

text_features = F.normalize(
    text_features,
    dim=-1
)

cosine_similarity = (
    image_features
    @
    text_features.T
)

logit_scale = (
    model.logit_scale.exp()
)

manual_logits = (
    logit_scale
    *
    cosine_similarity
)

print(
    manual_logits
)
```

이 과정은 다음과 같다.

```text
Image
 ↓
Image Embedding
 ↓
Normalize
        │
        │
        ├── Dot Product
        │
Normalize
 ↑
Text Embedding
 ↑
Text

        ↓

Cosine Similarity

        ↓

× exp(logit_scale)

        ↓

logits_per_image
```

---

# 37. Prompt가 왜 중요한가

CLIP은 단순한 클래스 번호를 학습한 모델이 아니다.

**자연어와 이미지의 관계**를 학습하였다.

따라서

```text
dog
```

만 주는 것과

```text
a photo of a dog
```

를 주는 것이 동일한 결과를 보장하지 않는다.

비교해보자.

```python
prompt_sets = {
    "class_only": [
        name
        for name in dataset.classes
    ],

    "photo": [
        f"a photo of a {name}"
        for name in dataset.classes
    ],

    "image": [
        f"an image of a {name}"
        for name in dataset.classes
    ]
}
```

---

# 38. Prompt별 결과 비교 함수

```python
def predict_with_prompts(
    image,
    prompts,
    class_names,
    model,
    processor,
    device
):
    """
    사용자가 직접 만든 prompt 목록으로
    CLIP Zero-shot 분류를 수행한다.
    """

    inputs = processor(
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.inference_mode():

        outputs = model(
            **inputs
        )

    probabilities = (
        outputs
        .logits_per_image
        .softmax(dim=1)[0]
        .detach()
        .cpu()
    )

    pred_index = (
        probabilities
        .argmax()
        .item()
    )

    return (
        class_names[pred_index],
        probabilities
    )
```

비교한다.

```python
for prompt_name, prompts in (
    prompt_sets.items()
):

    prediction, probabilities = (
        predict_with_prompts(
            image=image,
            prompts=prompts,
            class_names=dataset.classes,
            model=model,
            processor=processor,
            device=device
        )
    )

    print(
        f"{prompt_name:12s}: "
        f"{prediction}"
    )
```

---

# 39. Prompt Engineering의 의미

Prompt Engineering은 생성형 AI에서만 사용하는 개념이 아니다.

CLIP의 Zero-shot classification에서도 텍스트 표현 방식에 따라 결과가 달라질 수 있다.

예를 들어 자동차에 대해

```text
automobile
```

보다

```text
a photo of an automobile
```

또는 데이터 특성에 따라 다른 자연어 설명이 더 적절할 수 있다.

따라서 CLIP에서는 클래스 이름도 **모델에게 제공되는 의미 정보**이다.

---

# 40. CIFAR10 전체 평가로 확장하기

이미지 한 장만 분류해서는 모델 성능을 판단할 수 없다.

이번에는 여러 장을 평가한다.

처음에는 100장을 사용한다.

```python
NUM_SAMPLES = 100

correct = 0

results = []

for index in range(NUM_SAMPLES):

    image, label = dataset[index]

    true_class = (
        dataset.classes[label]
    )

    pred_class, probabilities = (
        classify_image(
            image=image,
            class_names=dataset.classes,
            model=model,
            processor=processor,
            device=device
        )
    )

    is_correct = (
        pred_class == true_class
    )

    if is_correct:
        correct += 1

    results.append(
        {
            "index": index,
            "true": true_class,
            "pred": pred_class,
            "correct": is_correct
        }
    )

accuracy = (
    correct
    /
    NUM_SAMPLES
)

print(
    f"평가 이미지: {NUM_SAMPLES}"
)

print(
    f"정답: {correct}"
)

print(
    f"오답: {NUM_SAMPLES - correct}"
)

print(
    f"정확도: {accuracy * 100:.2f}%"
)
```

이 코드는 교육적으로 이해하기 쉽지만 이미지마다 텍스트 embedding을 반복 계산하므로 효율적이지 않다.

다음 단계에서 개선한다.

---

# 41. Text Embedding을 한 번만 계산하기

클래스 prompt는 모든 이미지에서 동일하다.

따라서 매번

```text
"a photo of an airplane"
...
"a photo of a truck"
```

을 다시 embedding할 필요가 없다.

```python
class_names = dataset.classes

prompts = [
    f"a photo of a {name}"
    for name in class_names
]

text_inputs = processor(
    text=prompts,
    return_tensors="pt",
    padding=True
).to(device)

with torch.inference_mode():

    text_embeddings = (
        model.get_text_features(
            **text_inputs
        )
    )

text_embeddings = F.normalize(
    text_embeddings,
    dim=-1
)

print(
    text_embeddings.shape
)
```

이제 Text Embedding은 한 번만 만든다.

---

# 42. 효율적인 Zero-shot 분류 함수

```python
def predict_from_text_embeddings(
    image,
    text_embeddings,
    class_names,
    model,
    processor,
    device
):
    """
    미리 계산된 Text Embedding을 이용하여
    이미지 한 장을 분류한다.

    동일한 클래스들을 반복 평가할 때
    Text Encoder를 반복 실행하지 않아도 된다.
    """

    # 이미지만 전처리한다.
    image_inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    # 이미지 embedding을 계산한다.
    with torch.inference_mode():

        image_embedding = (
            model.get_image_features(
                **image_inputs
            )
        )

    # L2 normalization
    image_embedding = F.normalize(
        image_embedding,
        dim=-1
    )

    # 이미지와 모든 텍스트의
    # cosine similarity를 계산한다.
    similarities = (
        image_embedding
        @
        text_embeddings.T
    )

    # 가장 높은 similarity를 가진
    # 클래스 index를 찾는다.
    pred_index = (
        similarities[0]
        .argmax()
        .item()
    )

    pred_class = class_names[
        pred_index
    ]

    return (
        pred_class,
        similarities[0]
        .detach()
        .cpu()
    )
```

---

# 43. CIFAR10 1,000장 평가

```python
NUM_SAMPLES = 1000

results = []

correct = 0

for index in range(NUM_SAMPLES):

    image, label = dataset[index]

    true_class = (
        class_names[label]
    )

    pred_class, similarities = (
        predict_from_text_embeddings(
            image=image,
            text_embeddings=text_embeddings,
            class_names=class_names,
            model=model,
            processor=processor,
            device=device
        )
    )

    is_correct = (
        pred_class == true_class
    )

    if is_correct:
        correct += 1

    results.append(
        {
            "index": index,
            "true": true_class,
            "pred": pred_class,
            "correct": is_correct
        }
    )

accuracy = (
    correct
    /
    NUM_SAMPLES
)

print("=" * 60)

print(
    "CIFAR10 CLIP Zero-shot 분류 결과"
)

print("=" * 60)

print(
    "전체 평가 이미지 수:",
    NUM_SAMPLES
)

print(
    "정답 이미지 수:",
    correct
)

print(
    "오답 이미지 수:",
    NUM_SAMPLES - correct
)

print(
    f"정확도: {accuracy * 100:.2f}%"
)

print("=" * 60)
```

실제 정확도는 실행 환경, prompt 선택, 평가 샘플 등에 따라 확인해야 한다. 특정 결과값을 미리 고정해서 제시하지 않는 것이 정확하다.

---

# 44. 클래스별 정확도 계산

전체 정확도만 보면 어떤 클래스를 잘 분류하고 어떤 클래스를 어려워하는지 알 수 없다.

```python
from collections import defaultdict

class_total = defaultdict(int)
class_correct = defaultdict(int)

for result in results:

    true_class = result["true"]

    class_total[
        true_class
    ] += 1

    if result["correct"]:

        class_correct[
            true_class
        ] += 1

print(
    "클래스별 정확도"
)

print("-" * 60)

for class_name in class_names:

    total = class_total[
        class_name
    ]

    correct_count = class_correct[
        class_name
    ]

    if total > 0:

        class_accuracy = (
            correct_count
            /
            total
            *
            100
        )

    else:

        class_accuracy = 0.0

    print(
        f"{class_name:12s}: "
        f"{correct_count:3d}/"
        f"{total:3d} "
        f"({class_accuracy:6.2f}%)"
    )
```

---

# 45. 오분류 데이터 추출

```python
wrong_results = [
    result
    for result in results
    if not result["correct"]
]

print(
    "오분류 이미지 수:",
    len(wrong_results)
)
```

앞의 10개를 확인한다.

```python
for result in wrong_results[:10]:

    print(
        f"이미지 {result['index']:04d} | "
        f"정답: {result['true']:10s} | "
        f"예측: {result['pred']:10s}"
    )
```

---

# 46. 오분류 이미지 시각화

```python
import matplotlib.pyplot as plt

num_show = min(
    12,
    len(wrong_results)
)

if num_show > 0:

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(12, 9)
    )

    axes = axes.flatten()

    for ax in axes:
        ax.axis("off")

    for ax, result in zip(
        axes,
        wrong_results[:num_show]
    ):

        image, _ = dataset[
            result["index"]
        ]

        ax.imshow(image)

        ax.set_title(
            f"True: {result['true']}\n"
            f"Pred: {result['pred']}"
        )

        ax.axis("off")

    plt.tight_layout()
    plt.show()

else:

    print(
        "오분류 이미지가 없습니다."
    )
```

이 단계가 매우 중요하다.

성능 평가는

```text
Accuracy 계산
```

에서 끝나는 것이 아니다.

```text
성능 평가
    ↓
오류 발견
    ↓
오분류 사례 관찰
    ↓
오류 패턴 분석
    ↓
원인 가설
    ↓
Prompt / 모델 / 데이터 개선
```

까지 이어져야 한다.

---

# 47. CLIP이 틀리는 이유 분석하기

오분류 이미지를 보면서 다음을 분석한다.

첫째, **이미지가 너무 작은가?**

CIFAR10은 32×32 이미지이다.

이를 CLIP 입력 크기로 확대한다고 해서 원래 없던 세부 정보가 생기는 것은 아니다.

둘째, **클래스가 시각적으로 비슷한가?**

예:

```text
automobile ↔ truck

cat ↔ dog

airplane ↔ bird
```

셋째, **Prompt가 적절한가?**

```text
automobile
```

과

```text
a photo of an automobile
```

은 CLIP에게 완전히 동일한 입력이 아니다.

넷째, **CLIP이 해당 데이터셋 전용으로 학습되었는가?**

아니다.

지금 수행하는 것은 CIFAR10에 대해 별도로 fine-tuning하지 않은 **Zero-shot 평가**이다.

---

# 48. 이제 CLIP을 검색 모델로 바꿔보기

CLIP의 중요한 장점은 같은 구조를 이미지 검색에도 사용할 수 있다는 것이다.

분류에서는

```text
Image
 ↓
Image Embedding
 ↓
Text Embeddings와 비교
 ↓
Class 선택
```

하였다.

검색에서는 방향을 바꾼다.

```text
Text Query
 ↓
Text Embedding
 ↓
Image Embeddings와 비교
 ↓
Top-K Image
```

즉, **같은 Shared Embedding Space를 사용한다.**

---

# 49. 이미지 Gallery 만들기

CIFAR10에서 100장의 이미지를 Gallery로 사용한다.

```python
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

print(
    "Gallery 이미지 수:",
    len(gallery_images)
)
```

---

# 50. Gallery 이미지 Embedding 만들기

이번에는 batch 처리를 사용한다.

```python
gallery_inputs = processor(
    images=gallery_images,
    return_tensors="pt"
).to(device)

with torch.inference_mode():

    gallery_embeddings = (
        model.get_image_features(
            **gallery_inputs
        )
    )

gallery_embeddings = F.normalize(
    gallery_embeddings,
    dim=-1
)

print(
    "Gallery Embeddings:",
    gallery_embeddings.shape
)
```

이미지 100장이면

```text
[100, embedding_dimension]
```

형태가 된다.

---

# 51. 검색 문장을 Embedding으로 만들기

사용자가 다음과 같이 검색한다고 가정한다.

```text
"a photo of a dog"
```

```python
query = "a photo of a dog"

query_inputs = processor(
    text=[query],
    return_tensors="pt",
    padding=True
).to(device)

with torch.inference_mode():

    query_embedding = (
        model.get_text_features(
            **query_inputs
        )
    )

query_embedding = F.normalize(
    query_embedding,
    dim=-1
)

print(
    query_embedding.shape
)
```

---

# 52. 모든 이미지와 검색 문장 비교

```python
search_scores = (
    query_embedding
    @
    gallery_embeddings.T
)

print(
    search_scores.shape
)
```

결과는

```text
[1, 100]
```

이다.

즉,

```text
Query

   ↓

Image 0     score
Image 1     score
Image 2     score
...
Image 99    score
```

가 만들어졌다.

---

# 53. Top-K 이미지 찾기

```python
TOP_K = 5

top_scores, top_indices = (
    torch.topk(
        search_scores[0],
        k=TOP_K
    )
)

print(
    "검색어:",
    query
)

print()

for rank, (
    index,
    score
) in enumerate(
    zip(
        top_indices.tolist(),
        top_scores.tolist()
    ),
    start=1
):

    print(
        f"{rank}위 | "
        f"Index: {index:3d} | "
        f"Label: "
        f"{gallery_labels[index]:10s} | "
        f"Similarity: {score:.4f}"
    )
```

---

# 54. 검색 결과 이미지로 확인하기

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(
    1,
    TOP_K,
    figsize=(15, 3)
)

if TOP_K == 1:
    axes = [axes]

for rank, (
    ax,
    index,
    score
) in enumerate(
    zip(
        axes,
        top_indices.tolist(),
        top_scores.tolist()
    ),
    start=1
):

    ax.imshow(
        gallery_images[index]
    )

    ax.set_title(
        f"#{rank}\n"
        f"{gallery_labels[index]}\n"
        f"{score:.3f}"
    )

    ax.axis("off")

plt.suptitle(
    f"Query: {query}"
)

plt.tight_layout()
plt.show()
```

---

# 55. 검색 함수로 완성하기

```python
def search_images(
    query,
    gallery_images,
    gallery_labels,
    gallery_embeddings,
    model,
    processor,
    device,
    top_k=5
):
    """
    자연어 검색어를 입력받아
    CLIP Shared Embedding Space에서
    가장 유사한 이미지 Top-K를 반환한다.
    """

    # ---------------------------------------------
    # 검색 문장을 CLIP 입력으로 변환한다.
    # ---------------------------------------------

    query_inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True
    ).to(device)

    # ---------------------------------------------
    # Text Embedding을 생성한다.
    # ---------------------------------------------

    with torch.inference_mode():

        query_embedding = (
            model.get_text_features(
                **query_inputs
            )
        )

    # ---------------------------------------------
    # Cosine Similarity를 계산하기 위해
    # L2 normalization을 수행한다.
    # ---------------------------------------------

    query_embedding = F.normalize(
        query_embedding,
        dim=-1
    )

    # ---------------------------------------------
    # Query와 모든 Gallery Image를 비교한다.
    #
    # [1, D] @ [D, N]
    #
    # 결과:
    # [1, N]
    # ---------------------------------------------

    scores = (
        query_embedding
        @
        gallery_embeddings.T
    )

    # 실제 gallery 크기보다
    # top_k가 클 경우 오류가 발생하지 않도록 한다.
    k = min(
        top_k,
        len(gallery_images)
    )

    top_scores, top_indices = (
        torch.topk(
            scores[0],
            k=k
        )
    )

    results = []

    for rank, (
        index,
        score
    ) in enumerate(
        zip(
            top_indices.tolist(),
            top_scores.tolist()
        ),
        start=1
    ):

        results.append(
            {
                "rank": rank,
                "index": index,
                "label": gallery_labels[
                    index
                ],
                "score": score,
                "image": gallery_images[
                    index
                ]
            }
        )

    return results
```

사용한다.

```python
results = search_images(
    query="a photo of a dog",
    gallery_images=gallery_images,
    gallery_labels=gallery_labels,
    gallery_embeddings=gallery_embeddings,
    model=model,
    processor=processor,
    device=device,
    top_k=5
)

for result in results:

    print(
        result["rank"],
        result["label"],
        f"{result['score']:.4f}"
    )
```

---

# 56.  분류와 검색의 관계

CLIP에서는 두 작업이 본질적으로 같은 원리를 사용한다.

### Zero-shot Classification

```text
Image

 ↓

Image Embedding

 ↓

Text Embeddings와 비교

 ↓

가장 유사한 Text

 ↓

Class
```

### Text-to-Image Retrieval

```text
Text

 ↓

Text Embedding

 ↓

Image Embeddings와 비교

 ↓

가장 유사한 Image

 ↓

Top-K Images
```

따라서 CLIP을

> **이미지 분류 모델**

이라고만 이해하면 부족하다.

더 본질적으로는

> **이미지와 텍스트를 비교 가능한 공동 임베딩 공간으로 변환하는 모델**

이라고 이해하는 것이 적절하다.

---

# 57. Image-to-Image 검색도 가능하다

CLIP의 이미지 Encoder를 이용하면 이미지끼리도 비교할 수 있다.

```text
Query Image
     ↓
Image Encoder
     ↓
Query Image Embedding
     ↓
Gallery Image Embeddings와 비교
     ↓
Top-K Similar Images
```

구현한다.

```python
def search_similar_images(
    query_image,
    gallery_images,
    gallery_labels,
    gallery_embeddings,
    model,
    processor,
    device,
    top_k=5
):
    """
    Query 이미지와 의미적으로 유사한
    Gallery 이미지를 검색한다.
    """

    query_inputs = processor(
        images=query_image,
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():

        query_embedding = (
            model.get_image_features(
                **query_inputs
            )
        )

    query_embedding = F.normalize(
        query_embedding,
        dim=-1
    )

    scores = (
        query_embedding
        @
        gallery_embeddings.T
    )

    k = min(
        top_k,
        len(gallery_images)
    )

    top_scores, top_indices = (
        torch.topk(
            scores[0],
            k=k
        )
    )

    results = []

    for rank, (
        index,
        score
    ) in enumerate(
        zip(
            top_indices.tolist(),
            top_scores.tolist()
        ),
        start=1
    ):

        results.append(
            {
                "rank": rank,
                "index": index,
                "label": gallery_labels[
                    index
                ],
                "score": score,
                "image": gallery_images[
                    index
                ]
            }
        )

    return results
```

---

# 58. Embedding을 저장해야 하는 이유

실제 검색 시스템에서 사용자가 검색할 때마다 Gallery의 모든 이미지를 다시 CLIP에 넣는 것은 비효율적이다.

잘못된 구조는 다음과 같다.

```text
사용자 검색
   ↓
이미지 100,000장
   ↓
CLIP Encoder 실행
   ↓
Embedding 생성
   ↓
검색
```

매 검색마다 이미지 Encoder를 다시 실행한다.

대신 다음처럼 해야 한다.

```text
최초 1회

Images
 ↓
CLIP
 ↓
Embeddings
 ↓
저장


사용자 검색

Text
 ↓
CLIP
 ↓
Query Embedding
 ↓
저장된 Image Embeddings와 비교
 ↓
검색 결과
```

---

# 59. Image Embedding 저장하기

PyTorch 파일로 간단하게 저장할 수 있다.

```python
import torch

torch.save(
    {
        "embeddings":
            gallery_embeddings.cpu(),

        "labels":
            gallery_labels
    },
    "clip_gallery_embeddings.pt"
)

print(
    "Embedding 저장 완료"
)
```

---

# 60. Embedding 다시 불러오기

```python
saved_data = torch.load(
    "clip_gallery_embeddings.pt",
    map_location="cpu",
    weights_only=False
)

loaded_embeddings = (
    saved_data["embeddings"]
    .to(device)
)

loaded_labels = (
    saved_data["labels"]
)

print(
    loaded_embeddings.shape
)

print(
    len(loaded_labels)
)
```

이제 이미지를 다시 Encoder에 넣지 않고 저장된 embedding을 검색에 사용할 수 있다.

---

# 61. 실제 서비스로 확장하면

이미지 수가 적으면 PyTorch Tensor만으로도 충분하다.

하지만 이미지가 수십만~수백만 장이 되면 Vector Database나 ANN(Approximate Nearest Neighbor) 검색 구조를 고려하게 된다.

```text
Images
   ↓
CLIP Image Encoder
   ↓
Image Embeddings
   ↓
Vector Database
   │
   │
   │ similarity search
   │
Query Embedding
   ↑
CLIP Text Encoder
   ↑
사용자 검색
```

이 때문에 CLIP 학습은 이후 다음 기술과 자연스럽게 연결된다.

```text
CLIP
 ↓
Embedding
 ↓
Vector Search
 ↓
Vector DB
 ↓
Multimodal Retrieval
 ↓
Multimodal RAG
```

---

# 62. CLIP 전체 흐름 다시 정리하기


```text
                    CLIP

        Image                     Text
          │                         │
          ▼                         ▼

  CLIPProcessor               CLIPProcessor
          │                         │
          ▼                         ▼

   pixel_values        input_ids / attention_mask
          │                         │
          ▼                         ▼

   Vision Encoder              Text Encoder
          │                         │
          ▼                         ▼

    Image Feature              Text Feature
          │                         │
          ▼                         ▼

  Image Projection            Text Projection
          │                         │
          ▼                         ▼

  Image Embedding             Text Embedding
          │                         │
          └──────────┬──────────────┘
                     │
                     ▼

             L2 Normalization

                     ↓

             Cosine Similarity

                     ↓

          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼

    Zero-shot             Image Retrieval
  Classification
```

---

# 63. CLIP에서 반드시 구분해야 하는 개념

| 개념 | 역할 |
|---|---|
| `CLIPProcessor` | 이미지·텍스트 전처리 |
| `pixel_values` | Vision Encoder 입력 |
| `input_ids` | Text Encoder의 token ID |
| `attention_mask` | padding 등 attention 제외 영역 표시 |
| Vision Encoder | 이미지 특징 추출 |
| Text Encoder | 텍스트 특징 추출 |
| Projection | 두 표현을 공동 embedding 공간으로 변환 |
| `image_embeds` | 이미지 embedding |
| `text_embeds` | 텍스트 embedding |
| `get_image_features()` | 이미지 projected feature 추출 |
| `get_text_features()` | 텍스트 projected feature 추출 |
| L2 Normalize | 벡터 크기를 1로 정규화 |
| Cosine Similarity | 이미지와 텍스트의 유사성 계산 |
| `logit_scale` | similarity의 scale 조정 |
| `logits_per_image` | 이미지→텍스트 대응 점수 |
| Softmax | 후보들 사이의 상대적인 점수 분포 |
| Zero-shot | 대상 데이터셋에서 별도 학습 없이 자연어로 분류 |
| Retrieval | embedding 유사도로 관련 데이터 검색 |

---

# 64. CLIP의 한계에서 BLIP이 필요한 이유

강아지 이미지를 CLIP에 입력하고

```text
"a photo of a dog"
"a photo of a cat"
"a photo of a car"
```

를 비교하는 것은 가능하다.

그렇다면 이미지에 아무 설명도 주지 않고

> **"이 사진을 설명해줘."**

라고 요청하면 CLIP이 새로운 문장을 만들어낼 수 있을까?

기본 CLIP의 목적은 그렇지 않다.

CLIP은 본질적으로

```text
Image
+
Text Candidates

       ↓

Similarity

       ↓

어떤 Text가 Image와 가장 가까운가?
```

를 판단하는 구조이다.

새로운 문장을 생성하는 **Text Decoder**가 기본 CLIP 구조에 포함된 것은 아니다.

따라서

```text
Image

 ↓

"잔디밭에서 갈색 강아지가
 공을 가지고 뛰고 있습니다."
```

처럼 새로운 텍스트를 생성하는 문제에는 다른 구조가 필요하다.

여기에서 **BLIP**으로 넘어간다.

```text
              CLIP

Image ───────── Text
        비교

Similarity / Retrieval
Zero-shot Classification


                ↓


              BLIP

Image
 ↓
Vision Encoder
 ↓
Language Component
 ↓
Text 생성

Image Captioning
Visual Question Answering
```


[Hugging Face Transformers 4.57.2 CLIP 공식 문서](https://huggingface.co/docs/transformers/v4.57.2/model_doc/clip?utm_source=chatgpt.com)


---

# 핵심 정리

CLIP은 Image Encoder와 Text Encoder를 이용하여 이미지와 자연어를 Shared Embedding Space에 배치한다. 고정된 Classifier의 출력 뉴런을 선택하는 대신, 이미지와 자연어 Prompt의 유사도를 비교한다. 이 구조 덕분에 별도의 분류 Head를 새로 학습하지 않고도 Zero-shot Classification을 수행할 수 있다.

`CLIPProcessor`는 이미지와 텍스트의 전처리를 함께 담당한다. 이미지는 Resize, Crop, Rescale, Normalize 과정을 거쳐 `pixel_values`가 되고, 텍스트는 Tokenizer를 거쳐 `input_ids`와 `attention_mask`가 된다. Processor가 만드는 입력 Tensor와 모델이 위치한 Device는 일치해야 한다.

CLIP Image Retrieval에서는 Gallery 이미지의 Embedding을 미리 계산해 저장하고, 검색 시 Text Query의 Embedding과 비교한다. 실제 서비스 규모에서는 이 Embedding을 Vector DB나 ANN Index에 저장하여 검색 비용을 줄인다.

# 질문과 답변

### Q1. CLIP은 무엇의 약자인가?

**답변:** Contrastive Language-Image Pre-training의 약자이다. 이미지와 언어의 대응 관계를 Contrastive Learning으로 사전학습한다.

### Q2. 기존 이미지 분류기와 CLIP Zero-shot Classification의 핵심 차이는 무엇인가?

**답변:** 기존 분류기는 고정된 Classifier의 출력 클래스 중 하나를 선택한다. CLIP은 이미지와 후보 클래스 Prompt를 각각 Embedding으로 만들고 유사도가 가장 높은 텍스트를 선택한다.

### Q3. `CLIPProcessor`가 필요한 이유는 무엇인가?

**답변:** 모델은 PIL Image나 문자열을 직접 계산하지 않는다. Processor가 이미지를 `pixel_values`로, 문자열을 `input_ids`와 `attention_mask`로 변환하여 모델이 처리할 수 있는 Tensor 입력을 만든다.

### Q4. `pixel_values`가 `[1, 3, 224, 224]`이면 무엇을 의미하는가?

**답변:** Batch 1개, RGB Channel 3개, Height 224, Width 224인 모델 입력 Tensor라는 뜻이다. 이 크기는 Processor가 만든 모델 입력 크기이지 원본 이미지의 실제 해상도가 향상되었다는 뜻은 아니다.

### Q5. Zero-shot Classification에서 Prompt가 중요한 이유는 무엇인가?

**답변:** CLIP은 텍스트의 의미 표현과 이미지의 의미 표현을 비교한다. 같은 클래스 이름이라도 문장 구성에 따라 Text Embedding이 달라질 수 있으므로 Prompt Template이 결과에 영향을 줄 수 있다.

### Q6. `get_image_features()`와 `get_text_features()`는 무엇을 반환하는가?

**답변:** 각각 이미지와 텍스트의 Projection 이후 Feature를 반환한다. Cosine Similarity 계산을 직접 수행할 때는 일반적으로 L2 Normalize를 적용한 뒤 비교한다.

### Q7. Image Retrieval에서 Gallery Embedding을 매번 다시 계산해야 하는가?

**답변:** Gallery가 변경되지 않았다면 매번 계산할 필요가 없다. 한 번 계산하여 파일이나 Vector DB에 저장하고 Query Embedding만 새로 계산하는 방식이 효율적이다.

### Q8. 32×32 이미지를 224×224로 Resize하면 화질이 좋아지는가?

**답변:** 아니다. 모델 입력 크기는 커지지만 원본에 없던 세부 정보가 새로 생성되는 것은 아니다. CLIP, BLIP, VLM의 시각적 품질을 평가하려면 충분한 원본 해상도의 이미지를 사용하는 것이 바람직하다.

# 확인문제

1. CLIP의 두 Encoder와 두 Projection의 역할을 설명하라.
2. Zero-shot Classification 처리 순서를 작성하라.
3. `attention_mask`의 역할을 설명하라.
4. Prompt Engineering이 CLIP 성능에 영향을 줄 수 있는 이유를 설명하라.
5. Text-to-Image Retrieval에서 Embedding Cache가 필요한 이유를 설명하라.

## 확인문제 해설

1. Image/Text Encoder는 각 Modality의 Feature를 추출하고 Projection은 이를 비교 가능한 공통 Embedding 공간으로 보낸다.
2. Image 입력 → Image Embedding, 후보 Prompt → Text Embedding → 정규화 → 유사도 계산 → 가장 높은 후보 선택 순서이다.
3. Padding된 위치와 실제 Token 위치를 구분하여 Attention 계산에서 불필요한 Padding 영역을 처리하기 위해 사용한다.
4. Prompt 문장이 달라지면 Text Embedding도 달라지므로 이미지와의 유사도 순위가 달라질 수 있다.
5. Gallery가 클수록 매 Query마다 모든 Image Embedding을 다시 계산하는 비용이 크므로 미리 계산해 재사용하는 것이 효율적이다.

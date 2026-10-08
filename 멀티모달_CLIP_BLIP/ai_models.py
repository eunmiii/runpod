
# ============================================================
# ai_models.py
#
# 멀티모달 AI 시스템에서 사용하는 모델과
# 추론 함수를 정의한다.
#
# 주요 기능
# ------------------------------------------------------------
# 1. Faster R-CNN 객체 탐지
# 2. CLIP 자연어 이미지 검색
# 3. NLLB 한국어 → 영어 번역
# 4. BLIP 이미지 설명 생성
# 5. NLLB 영어 → 한국어 번역
#
# 전체 구조
# ------------------------------------------------------------
#
# [객체 탐지]
#
# 이미지
#   ↓
# Faster R-CNN
#   ↓
# 객체 + Bounding Box
#
#
# [한국어 이미지 검색]
#
# 한국어 검색어
#   ↓
# NLLB : 한국어 → 영어
#   ↓
# CLIP Text Encoder
#   ↓
# Text Embedding
#                      이미지
#                        ↓
#                 CLIP Image Encoder
#                        ↓
#                 Image Embedding
#                        ↓
#               Cosine Similarity
#                        ↓
#                      Top-K
#
#
# [한국어 이미지 설명]
#
# 이미지
#   ↓
# BLIP
#   ↓
# 영어 Caption
#   ↓
# NLLB : 영어 → 한국어
#   ↓
# 한국어 Caption
#
# ============================================================


# ============================================================
# 1. 라이브러리
# ============================================================

from time import perf_counter

import torch
import torch.nn.functional as F

from PIL import ImageDraw

# ------------------------------------------------------------
# Faster R-CNN
# ------------------------------------------------------------

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)

# ------------------------------------------------------------
# Hugging Face Transformers
# ------------------------------------------------------------

from transformers import (
    # CLIP
    CLIPModel,
    CLIPProcessor,

    # BLIP
    BlipProcessor,
    BlipForConditionalGeneration,

    # NLLB 번역
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)


# ============================================================
# 2. 실행 장치 설정
# ============================================================
#
# 우선순위
#
# NVIDIA GPU
#     ↓
# Apple MPS
#     ↓
# CPU
#
# ============================================================

if torch.cuda.is_available():

    device = torch.device("cuda")

elif (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
):

    device = torch.device("mps")

else:

    device = torch.device("cpu")


# ============================================================
# 3. Faster R-CNN 모델 로딩
# ============================================================

def load_detector():
    """
    COCO 데이터셋으로 사전 학습된
    Faster R-CNN 객체 탐지 모델을 로딩한다.

    반환
    ----------------------------------------------------------
    model
        Faster R-CNN 모델

    weights
        전처리 방법과 COCO 클래스 이름 등을 포함하는
        사전 학습 가중치 정보
    """

    # --------------------------------------------------------
    # 사전 학습 가중치
    # --------------------------------------------------------

    weights = (
        FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )

    # --------------------------------------------------------
    # 모델 생성
    # --------------------------------------------------------

    model = fasterrcnn_resnet50_fpn(
        weights=weights
    )

    # --------------------------------------------------------
    # 실행 장치로 이동
    # --------------------------------------------------------

    model = model.to(device)

    # --------------------------------------------------------
    # 평가 모드
    # --------------------------------------------------------

    model.eval()

    return model, weights


# ============================================================
# 4. Faster R-CNN 객체 탐지
# ============================================================

def detect_objects(
    model,
    weights,
    image,
    threshold=0.5,
):
    """
    이미지에서 객체를 탐지한다.

    처리 과정
    ----------------------------------------------------------

    PIL Image
        ↓
    Tensor 변환
        ↓
    Faster R-CNN
        ↓
    boxes / labels / scores
        ↓
    Threshold 필터링
        ↓
    Bounding Box 표시
    """

    # --------------------------------------------------------
    # 이미지 형식을 RGB로 통일
    # --------------------------------------------------------

    image = image.convert("RGB")


    # --------------------------------------------------------
    # Faster R-CNN 기본 전처리 함수
    # --------------------------------------------------------

    preprocess = weights.transforms()


    # --------------------------------------------------------
    # PIL Image → Tensor
    # --------------------------------------------------------

    image_tensor = preprocess(image)


    # --------------------------------------------------------
    # 모델이 위치한 장치 확인
    # --------------------------------------------------------

    model_device = next(
        model.parameters()
    ).device


    # --------------------------------------------------------
    # 입력 Tensor 이동
    # --------------------------------------------------------

    image_tensor = image_tensor.to(
        model_device
    )


    # --------------------------------------------------------
    # 추론 시작
    # --------------------------------------------------------

    start_time = perf_counter()


    # --------------------------------------------------------
    # 추론 과정에서는 Gradient 계산이 필요하지 않으므로
    # inference_mode()를 사용한다.
    # --------------------------------------------------------

    with torch.inference_mode():

        prediction = model(
            [image_tensor]
        )[0]


    elapsed = (
        perf_counter()
        - start_time
    )


    # --------------------------------------------------------
    # COCO 클래스 이름
    # --------------------------------------------------------

    categories = (
        weights.meta["categories"]
    )


    # --------------------------------------------------------
    # Bounding Box를 표시할 이미지
    # --------------------------------------------------------

    result_image = image.copy()

    draw = ImageDraw.Draw(
        result_image
    )


    # --------------------------------------------------------
    # 결과 Tensor를 CPU로 이동
    # --------------------------------------------------------

    boxes = (
        prediction["boxes"]
        .detach()
        .cpu()
    )

    labels = (
        prediction["labels"]
        .detach()
        .cpu()
    )

    scores = (
        prediction["scores"]
        .detach()
        .cpu()
    )


    detections = []


    # --------------------------------------------------------
    # 탐지된 객체 반복
    # --------------------------------------------------------

    for box, label, score in zip(
        boxes,
        labels,
        scores,
    ):

        score_value = float(
            score.item()
        )


        # ----------------------------------------------------
        # Threshold보다 낮으면 제외
        # ----------------------------------------------------

        if score_value < threshold:
            continue


        label_index = int(
            label.item()
        )


        # ----------------------------------------------------
        # 클래스 이름
        # ----------------------------------------------------

        if (
            0 <= label_index
            < len(categories)
        ):

            label_name = (
                categories[label_index]
            )

        else:

            label_name = str(
                label_index
            )


        # ----------------------------------------------------
        # Bounding Box 좌표
        # ----------------------------------------------------

        x1, y1, x2, y2 = (
            box.tolist()
        )


        # ----------------------------------------------------
        # Bounding Box 그리기
        # ----------------------------------------------------

        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            outline="red",
            width=3,
        )


        # ----------------------------------------------------
        # 클래스 + Confidence 표시
        # ----------------------------------------------------

        draw.text(
            (
                x1 + 4,
                y1 + 4,
            ),
            (
                f"{label_name} "
                f"{score_value:.2f}"
            ),
            fill="red",
        )


        # ----------------------------------------------------
        # 표에 표시할 결과 저장
        # ----------------------------------------------------

        detections.append(
            {
                "객체": label_name,

                "점수": round(
                    score_value,
                    4,
                ),

                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
            }
        )


    return (
        result_image,
        detections,
        elapsed,
    )


# ============================================================
# 5. CLIP 모델 로딩
# ============================================================

def load_clip():
    """
    OpenAI CLIP ViT-B/32 모델을 로딩한다.

    CLIP은 이미지와 텍스트를 각각 임베딩한 후
    동일한 Shared Embedding Space에서 비교한다.
    """

    model_name = (
        "openai/clip-vit-base-patch32"
    )


    # --------------------------------------------------------
    # Processor
    #
    # 이미지:
    # PIL Image → pixel_values
    #
    # 텍스트:
    # 문자열 → input_ids, attention_mask
    # --------------------------------------------------------

    processor = (
        CLIPProcessor.from_pretrained(
            model_name
        )
    )


    # --------------------------------------------------------
    # CLIP 모델
    # --------------------------------------------------------

    model = (
        CLIPModel.from_pretrained(
            model_name
        )
    )


    model = model.to(device)

    model.eval()


    return (
        model,
        processor,
    )


# ============================================================
# 6. NLLB 번역 모델 로딩
# ============================================================
#
# 하나의 NLLB 모델을 다음 두 방향에서 공통 사용한다.
#
# 한국어 → 영어
# 영어 → 한국어
#
# 따라서 번역 모델을 두 개 메모리에 올릴 필요가 없다.
#
# ============================================================

def load_translator():

    model_name = (
        "facebook/"
        "nllb-200-distilled-600M"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            model_name
        )
    )

    model = (
        AutoModelForSeq2SeqLM
        .from_pretrained(
            model_name
        )
    )

    model = model.to(device)

    model.eval()

    return model, tokenizer

    model.eval()


    return (
        model,
        tokenizer,
    )


# ============================================================
# 7. 한글 포함 여부 확인
# ============================================================

def contains_korean(text):
    """
    문자열에 한글 완성형 문자가 포함되어 있는지 검사한다.

    예
    ----------------------------------------------------------

    contains_korean("고양이")
        → True

    contains_korean("cat")
        → False
    """

    return any(
        "\uac00" <= char <= "\ud7a3"
        for char in text
    )


# ============================================================
# 8. 공통 NLLB 번역 함수
# ============================================================

def translate_text(
    text,
    translator_model,
    translator_tokenizer,
    source_language,
    target_language,
):
    """
    NLLB를 이용하여 문장을 번역한다.

    Parameters
    ----------------------------------------------------------
    text
        번역할 문자열

    translator_model
        NLLB 모델

    translator_tokenizer
        NLLB Tokenizer

    source_language
        입력 언어 코드

    target_language
        출력 언어 코드


    예
    ----------------------------------------------------------

    한국어 → 영어

    source_language = "kor_Hang"
    target_language = "eng_Latn"


    영어 → 한국어

    source_language = "eng_Latn"
    target_language = "kor_Hang"
    """

    # --------------------------------------------------------
    # 1. 공백 제거
    # --------------------------------------------------------

    text = text.strip()


    if not text:
        return ""


    # --------------------------------------------------------
    # 2. 입력 언어 설정
    # --------------------------------------------------------
    #
    # NLLB Tokenizer는 현재 입력 문장이
    # 어떤 언어인지 알아야 한다.
    # --------------------------------------------------------

    translator_tokenizer.src_lang = (
        source_language
    )


    # --------------------------------------------------------
    # 3. Tokenization
    # --------------------------------------------------------

    inputs = translator_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )


    # --------------------------------------------------------
    # 4. 모델 장치 확인
    # --------------------------------------------------------

    model_device = next(
        translator_model.parameters()
    ).device


    # --------------------------------------------------------
    # 5. 입력 Tensor 이동
    # --------------------------------------------------------

    inputs = {
        key: value.to(
            model_device
        )
        for key, value
        in inputs.items()
    }


    # --------------------------------------------------------
    # 6. 출력 언어 Token ID
    # --------------------------------------------------------
    #
    # 예:
    #
    # eng_Latn
    # kor_Hang
    #
    # --------------------------------------------------------

    target_language_id = (
        translator_tokenizer
        .convert_tokens_to_ids(
            target_language
        )
    )


    # --------------------------------------------------------
    # 7. 번역
    # --------------------------------------------------------

    with torch.inference_mode():

        generated_ids = (
            translator_model.generate(
                **inputs,

                # --------------------------------------------
                # 첫 번째 출력 언어를 지정한다.
                # --------------------------------------------
                forced_bos_token_id=(
                    target_language_id
                ),

                max_new_tokens=64,

                # --------------------------------------------
                # Beam Search
                # --------------------------------------------
                num_beams=4,
            )
        )


    # --------------------------------------------------------
    # 8. Token → 문자열
    # --------------------------------------------------------

    translated_text = (
        translator_tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )
    )


    return translated_text.strip()


# ============================================================
# 9. 한국어 → 영어
# ============================================================

def translate_ko_to_en(
    text,
    translator_model,
    translator_tokenizer,
):
    """
    한국어 검색어를 영어로 번역한다.

    예
    ----------------------------------------------------------

    공원에서 뛰는 강아지

              ↓

    a dog running in a park
    """

    return translate_text(
        text=text,

        translator_model=(
            translator_model
        ),

        translator_tokenizer=(
            translator_tokenizer
        ),

        source_language="kor_Hang",

        target_language="eng_Latn",
    )


# ============================================================
# 10. 영어 → 한국어
# ============================================================

def translate_en_to_ko(
    text,
    translator_model,
    translator_tokenizer,
):
    """
    BLIP의 영어 Caption을 한국어로 번역한다.

    예
    ----------------------------------------------------------

    two women doing yoga poses

              ↓

    요가 자세를 취하고 있는 두 여성
    """

    return translate_text(
        text=text,

        translator_model=(
            translator_model
        ),

        translator_tokenizer=(
            translator_tokenizer
        ),

        source_language="eng_Latn",

        target_language="kor_Hang",
    )


# ============================================================
# 11. CLIP 자연어 이미지 검색
# ============================================================

def search_images(
    model,
    processor,
    query,
    image_items,
    top_k=5,
    translator_model=None,
    translator_tokenizer=None,
):
    """
    자연어 검색어와 이미지의 코사인 유사도를 계산한다.

    한국어 검색의 경우
    ----------------------------------------------------------

    한국어
      ↓
    NLLB
      ↓
    영어
      ↓
    CLIP Text Encoder
      ↓
    Text Embedding


    이미지
      ↓
    CLIP Image Encoder
      ↓
    Image Embedding


    두 Embedding
      ↓
    L2 Normalize
      ↓
    Cosine Similarity
      ↓
    Top-K
    """

    start_time = perf_counter()


    # --------------------------------------------------------
    # 이미지가 없는 경우
    # --------------------------------------------------------

    if not image_items:

        return (
            [],
            0.0,
            query,
        )


    # ========================================================
    # 1. 검색 문장
    # ========================================================

    original_query = (
        query.strip()
    )


    # ========================================================
    # 2. 한국어 검색어 처리
    # ========================================================

    if contains_korean(
        original_query
    ):

        if (
            translator_model is None
            or translator_tokenizer is None
        ):

            raise ValueError(
                "한국어 검색을 위한 "
                "번역 모델이 없습니다."
            )


        # ----------------------------------------------------
        # 한국어 → 영어
        # ----------------------------------------------------

        clip_query = (
            translate_ko_to_en(
                original_query,
                translator_model,
                translator_tokenizer,
            )
        )


    else:

        # ----------------------------------------------------
        # 영어는 번역하지 않는다.
        # ----------------------------------------------------

        clip_query = (
            original_query
        )


    # ========================================================
    # 3. 이미지 이름
    # ========================================================

    image_names = [
        item[0]
        for item in image_items
    ]


    # ========================================================
    # 4. PIL Image
    # ========================================================

    images = [
        item[1].convert("RGB")
        for item in image_items
    ]


    # ========================================================
    # 5. CLIP Processor
    # ========================================================
    #
    # 텍스트
    #   ↓
    # input_ids
    # attention_mask
    #
    # 이미지
    #   ↓
    # pixel_values
    #
    # ========================================================

    inputs = processor(
        text=[clip_query],
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )


    # ========================================================
    # 6. CLIP 장치 확인
    # ========================================================

    model_device = next(
        model.parameters()
    ).device


    # ========================================================
    # 7. 입력 Tensor 이동
    # ========================================================

    inputs = {
        key: value.to(
            model_device
        )
        for key, value
        in inputs.items()
    }


    # ========================================================
    # 8. CLIP 전체 Forward
    # ========================================================
    #
    # 여기서는
    #
    # model.get_image_features()
    # model.get_text_features()
    #
    # 를 사용하지 않는다.
    #
    # 사용자 환경에서 이전에
    #
    # BaseModelOutputWithPooling
    #
    # 객체가 반환되어
    #
    # .float()
    #
    # 오류가 발생했기 때문이다.
    #
    # 대신 전체 CLIP Forward 결과의
    #
    # outputs.image_embeds
    # outputs.text_embeds
    #
    # 를 직접 사용한다.
    #
    # ========================================================

    with torch.inference_mode():

        outputs = model(
            **inputs,
            return_dict=True,
        )


    # ========================================================
    # 9. Shared Embedding 추출
    # ========================================================

    image_features = (
        outputs.image_embeds
    )

    text_features = (
        outputs.text_embeds
    )


    # ========================================================
    # 10. Tensor 검사
    # ========================================================

    if not torch.is_tensor(
        image_features
    ):

        raise TypeError(
            "outputs.image_embeds가 "
            "Tensor가 아닙니다. "
            f"실제 자료형: "
            f"{type(image_features)}"
        )


    if not torch.is_tensor(
        text_features
    ):

        raise TypeError(
            "outputs.text_embeds가 "
            "Tensor가 아닙니다. "
            f"실제 자료형: "
            f"{type(text_features)}"
        )


    # ========================================================
    # 11. float32 변환
    # ========================================================

    image_features = (
        image_features.to(
            dtype=torch.float32
        )
    )


    text_features = (
        text_features.to(
            dtype=torch.float32
        )
    )


    # ========================================================
    # 12. L2 Normalize
    # ========================================================
    #
    # x_hat = x / ||x||
    #
    # ========================================================

    image_features = F.normalize(
        image_features,
        p=2,
        dim=-1,
    )


    text_features = F.normalize(
        text_features,
        p=2,
        dim=-1,
    )


    # ========================================================
    # 13. Cosine Similarity
    # ========================================================
    #
    # text_features
    # [1, D]
    #
    # image_features
    # [N, D]
    #
    # image_features.T
    # [D, N]
    #
    # 결과
    # [N]
    #
    # ========================================================

    similarities = (
        text_features
        @ image_features.T
    ).squeeze(0)


    # ========================================================
    # 14. Top-K
    # ========================================================

    k = min(
        max(
            int(top_k),
            1,
        ),
        len(images),
    )


    scores, indices = (
        torch.topk(
            similarities,
            k=k,
        )
    )


    # ========================================================
    # 15. 결과 구성
    # ========================================================

    results = []


    for score, index in zip(
        scores.detach().cpu().tolist(),
        indices.detach().cpu().tolist(),
    ):

        results.append(
            {
                "name": (
                    image_names[index]
                ),

                "image": (
                    images[index]
                ),

                "similarity": float(
                    score
                ),
            }
        )


    elapsed = (
        perf_counter()
        - start_time
    )


    return (
        results,
        elapsed,
        clip_query,
    )


# ============================================================
# 12. BLIP 모델 로딩
# ============================================================

def load_blip():
    """
    BLIP Image Captioning 모델을 로딩한다.

    입력
    ----------------------------------------------------------
    이미지

    출력
    ----------------------------------------------------------
    영어 Caption
    """

    model_name = (
        "Salesforce/"
        "blip-image-captioning-base"
    )


    # --------------------------------------------------------
    # BLIP Processor
    # --------------------------------------------------------

    processor = (
        BlipProcessor.from_pretrained(
            model_name
        )
    )


    # --------------------------------------------------------
    # BLIP 모델
    # --------------------------------------------------------

    model = (
        BlipForConditionalGeneration
        .from_pretrained(
            model_name
        )
    )


    model = model.to(device)

    model.eval()


    return (
        model,
        processor,
    )


# ============================================================
# 13. BLIP Caption 생성
# ============================================================

def generate_caption(
    model,
    processor,
    image,
    max_new_tokens=30,
):
    """
    BLIP으로 이미지 설명을 생성한다.

    중요
    ----------------------------------------------------------

    이 함수가 반환하는 Caption은 영어이다.

    이미지
      ↓
    BLIP
      ↓
    영어 Caption

    한국어 변환은 translate_en_to_ko()에서 수행한다.
    """

    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    image = image.convert(
        "RGB"
    )


    # --------------------------------------------------------
    # PIL Image → Tensor
    # --------------------------------------------------------

    inputs = processor(
        images=image,
        return_tensors="pt",
    )


    # --------------------------------------------------------
    # BLIP 모델 장치
    # --------------------------------------------------------

    model_device = next(
        model.parameters()
    ).device


    # --------------------------------------------------------
    # Tensor 이동
    # --------------------------------------------------------

    inputs = {
        key: value.to(
            model_device
        )
        for key, value
        in inputs.items()
    }


    # --------------------------------------------------------
    # 추론 시간 측정
    # --------------------------------------------------------

    start_time = perf_counter()


    # --------------------------------------------------------
    # Caption 생성
    # --------------------------------------------------------

    with torch.inference_mode():

        generated_ids = (
            model.generate(
                **inputs,
                max_new_tokens=(
                    max_new_tokens
                ),
            )
        )


    elapsed = (
        perf_counter()
        - start_time
    )


    # --------------------------------------------------------
    # Token → 영어 문장
    # --------------------------------------------------------

    caption = (
        processor.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )
    )


    return (
        caption.strip(),
        elapsed,
    )

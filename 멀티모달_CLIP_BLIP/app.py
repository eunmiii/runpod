
# ============================================================
# app.py
#
# Streamlit 멀티모달 AI 이미지 분석 시스템
#
# 기능
# ------------------------------------------------------------
#
# [1] Faster R-CNN 객체 탐지
#
# [2] CLIP 한국어 이미지 검색
#
#     한국어
#       ↓
#     NLLB 한국어 → 영어
#       ↓
#     CLIP
#       ↓
#     Cosine Similarity
#       ↓
#     Top-K
#
# [3] BLIP 한국어 이미지 설명
#
#     이미지
#       ↓
#     BLIP
#       ↓
#     영어 Caption
#       ↓
#     NLLB 영어 → 한국어
#       ↓
#     한국어 Caption
#
# ============================================================


# ============================================================
# 1. 라이브러리
# ============================================================

from pathlib import Path
from time import perf_counter
import io

import streamlit as st

from PIL import (
    Image,
    ImageOps,
    UnidentifiedImageError,
)


# ============================================================
# 2. AI 모델
# ============================================================

from ai_models import (
    # 실행 장치
    device,

    # Faster R-CNN
    load_detector,
    detect_objects,

    # CLIP
    load_clip,
    search_images,

    # NLLB
    load_translator,
    translate_en_to_ko,

    # BLIP
    load_blip,
    generate_caption,
)


# ============================================================
# 3. Streamlit 페이지 설정
# ============================================================

st.set_page_config(
    page_title=(
        "멀티모달 AI 이미지 분석"
    ),
    page_icon="🔎",
    layout="wide",
)


# ============================================================
# 4. 제목
# ============================================================

st.title(
    "멀티모달 AI 이미지 분석 시스템"
)


st.write(
    "Faster R-CNN 객체 탐지, "
    "CLIP 한국어 이미지 검색, "
    "BLIP 한국어 이미지 설명 생성을 "
    "하나의 웹 애플리케이션에서 실행한다."
)


st.caption(
    f"PyTorch 실행 장치: {device}"
)


# ============================================================
# 5. 모델 Cache
# ============================================================
#
# Streamlit은 버튼 클릭, 입력 변경 등이 발생하면
# Python 파일을 다시 실행한다.
#
# 모델을 매번 다시 다운로드하거나 로딩하면
# 실행 시간이 매우 길어진다.
#
# 따라서 st.cache_resource를 사용하여
# 한 번 로딩된 모델을 재사용한다.
#
# ============================================================


# ------------------------------------------------------------
# Faster R-CNN
# ------------------------------------------------------------

@st.cache_resource(
    show_spinner=(
        "Faster R-CNN 모델을 "
        "불러오는 중..."
    )
)
def get_detector():

    return load_detector()


# ------------------------------------------------------------
# CLIP
# ------------------------------------------------------------

@st.cache_resource(
    show_spinner=(
        "CLIP 모델을 "
        "불러오는 중..."
    )
)
def get_clip():

    return load_clip()


# ------------------------------------------------------------
# NLLB
#
# 한국어 → 영어
# 영어 → 한국어
#
# 두 작업에 동일한 모델을 사용한다.
# ------------------------------------------------------------

@st.cache_resource(
    show_spinner=(
        "NLLB 번역 모델을 "
        "불러오는 중..."
    )
)
def get_translator():

    return load_translator()


# ------------------------------------------------------------
# BLIP
# ------------------------------------------------------------

@st.cache_resource(
    show_spinner=(
        "BLIP 모델을 "
        "불러오는 중..."
    )
)
def get_blip():

    return load_blip()


# ============================================================
# 6. 업로드 이미지 읽기
# ============================================================

def read_uploaded_image(
    uploaded_file,
):
    """
    Streamlit에서 업로드한 이미지를
    PIL RGB 이미지로 변환한다.
    """

    image = Image.open(
        uploaded_file
    )


    # --------------------------------------------------------
    # 스마트폰 사진 등의 EXIF 방향 정보를 반영한다.
    # --------------------------------------------------------

    image = ImageOps.exif_transpose(
        image
    )


    # --------------------------------------------------------
    # 모델 입력 형식을 RGB로 통일한다.
    # --------------------------------------------------------

    image = image.convert(
        "RGB"
    )


    return image


# ============================================================
# 7. CLIP 검색용 이미지 로딩
# ============================================================

def load_gallery(
    folder="./image",
):
    """
    ./image 폴더의 이미지를 읽는다.

    반환
    ----------------------------------------------------------

    [
        ("cat.jpg", PIL.Image),
        ("dog.jpg", PIL.Image),
        ("car.jpg", PIL.Image)
    ]
    """

    folder = Path(
        folder
    )


    # --------------------------------------------------------
    # 폴더가 없으면 생성한다.
    # --------------------------------------------------------

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )


    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
    }


    image_items = []


    # --------------------------------------------------------
    # 폴더의 모든 파일 확인
    # --------------------------------------------------------

    for path in sorted(
        folder.iterdir()
    ):

        # ----------------------------------------------------
        # 이미지 확장자가 아니면 제외한다.
        # ----------------------------------------------------

        if (
            path.suffix.lower()
            not in extensions
        ):

            continue


        try:

            with Image.open(
                path
            ) as source:

                # --------------------------------------------
                # EXIF 방향 보정
                # --------------------------------------------

                image = (
                    ImageOps
                    .exif_transpose(
                        source
                    )
                )


                # --------------------------------------------
                # RGB 변환
                # --------------------------------------------

                image = image.convert(
                    "RGB"
                )


                # --------------------------------------------
                # 파일을 닫은 이후에도 사용할 수 있도록
                # 이미지 객체를 복사한다.
                # --------------------------------------------

                image_items.append(
                    (
                        path.name,
                        image.copy(),
                    )
                )


        except (
            OSError,
            UnidentifiedImageError,
        ):

            # 읽을 수 없는 파일은 제외한다.
            continue


    return image_items


# ============================================================
# 8. Sidebar
# ============================================================

st.sidebar.header(
    "실행 설정"
)


# ------------------------------------------------------------
# 객체 탐지 Confidence Threshold
# ------------------------------------------------------------

threshold = st.sidebar.slider(
    "객체 탐지 최소 점수",
    min_value=0.10,
    max_value=0.95,
    value=0.50,
    step=0.05,
)


# ------------------------------------------------------------
# CLIP Top-K
# ------------------------------------------------------------

top_k = st.sidebar.slider(
    "CLIP 검색 결과 수",
    min_value=1,
    max_value=10,
    value=5,
    step=1,
)


st.sidebar.divider()


st.sidebar.subheader(
    "사용 모델"
)


st.sidebar.markdown(
    """
- **Faster R-CNN** : 객체 탐지
- **NLLB-200** : 한국어 ↔ 영어 번역
- **CLIP ViT-B/32** : 이미지 검색
- **BLIP** : 이미지 설명 생성
"""
)


# ============================================================
# 9. 분석 이미지 업로드
# ============================================================

st.subheader(
    "분석 이미지"
)


uploaded_file = st.file_uploader(
    "분석할 이미지를 업로드한다.",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
)


uploaded_image = None


if uploaded_file is not None:

    try:

        uploaded_image = (
            read_uploaded_image(
                uploaded_file
            )
        )


        # ----------------------------------------------------
        # 너무 큰 이미지는 화면/메모리를 위해 축소한다.
        # ----------------------------------------------------

        max_side = 1600


        if (
            max(uploaded_image.size)
            > max_side
        ):

            resized = (
                uploaded_image.copy()
            )


            resized.thumbnail(
                (
                    max_side,
                    max_side,
                )
            )


            uploaded_image = resized


        # ----------------------------------------------------
        # 화면 표시
        # ----------------------------------------------------

        col_image, col_info = (
            st.columns(
                [3, 1]
            )
        )


        with col_image:

            st.image(
                uploaded_image,
                caption="업로드 이미지",
                use_container_width=True,
            )


        with col_info:

            st.markdown(
                "#### 이미지 정보"
            )


            st.write(
                f"Width: "
                f"{uploaded_image.width}"
            )


            st.write(
                f"Height: "
                f"{uploaded_image.height}"
            )


            st.write(
                f"Mode: "
                f"{uploaded_image.mode}"
            )


    except Exception as exc:

        st.error(
            "이미지를 읽는 중 오류가 발생했다: "
            f"{exc}"
        )

        st.exception(
            exc
        )

        uploaded_image = None


# ============================================================
# 10. Tab 생성
# ============================================================
#
# 이전에 발생했던
#
# NameError:
# name 'tab_search' is not defined
#
# 오류를 방지하기 위해
# 반드시 with tab_search보다 먼저 탭을 생성한다.
#
# ============================================================

(
    tab_detection,
    tab_search,
    tab_caption,
) = st.tabs(
    [
        "객체 탐지",
        "한국어 이미지 검색",
        "한국어 이미지 설명",
    ]
)


# ============================================================
# 11. 객체 탐지
# ============================================================

with tab_detection:

    st.header(
        "Faster R-CNN 객체 탐지"
    )


    st.write(
        "업로드된 이미지에서 객체를 탐지하고 "
        "Bounding Box와 객체 종류를 표시한다."
    )


    if uploaded_image is None:

        st.info(
            "먼저 상단에서 이미지를 업로드한다."
        )


    if st.button(
        "객체 탐지 실행",
        disabled=(
            uploaded_image is None
        ),
        key="run_detection",
    ):

        try:

            with st.spinner(
                "Faster R-CNN 객체 탐지 중..."
            ):

                # --------------------------------------------
                # 모델 로딩
                # --------------------------------------------

                (
                    detector_model,
                    detector_weights,
                ) = get_detector()


                # --------------------------------------------
                # 객체 탐지
                # --------------------------------------------

                (
                    result_image,
                    detections,
                    elapsed,
                ) = detect_objects(
                    model=(
                        detector_model
                    ),
                    weights=(
                        detector_weights
                    ),
                    image=(
                        uploaded_image
                    ),
                    threshold=(
                        threshold
                    ),
                )


            # =================================================
            # 결과
            # =================================================

            col1, col2 = (
                st.columns(2)
            )


            with col1:

                st.subheader(
                    "탐지 결과 이미지"
                )


                st.image(
                    result_image,
                    use_container_width=True,
                )


                # --------------------------------------------
                # 결과 이미지 다운로드
                # --------------------------------------------

                buffer = (
                    io.BytesIO()
                )


                result_image.save(
                    buffer,
                    format="PNG",
                )


                st.download_button(
                    label=(
                        "탐지 결과 이미지 다운로드"
                    ),
                    data=(
                        buffer.getvalue()
                    ),
                    file_name=(
                        "detection_result.png"
                    ),
                    mime="image/png",
                )


            with col2:

                st.subheader(
                    "탐지 정보"
                )


                st.metric(
                    "탐지 객체 수",
                    len(detections),
                )


                st.metric(
                    "추론 시간",
                    f"{elapsed:.3f}초",
                )


                if detections:

                    st.dataframe(
                        detections,
                        use_container_width=True,
                    )


                else:

                    st.warning(
                        "설정한 점수 이상의 "
                        "객체가 탐지되지 않았다."
                    )


        except Exception as exc:

            st.error(
                "객체 탐지 중 오류가 발생했다: "
                f"{exc}"
            )

            st.exception(
                exc
            )


# ============================================================
# 12. 한국어 CLIP 이미지 검색
# ============================================================

with tab_search:

    st.header(
        "CLIP 한국어 자연어 이미지 검색"
    )


    st.write(
        "한국어 검색어를 NLLB로 영어로 번역한 후 "
        "CLIP 공유 임베딩 공간에서 이미지와 "
        "텍스트의 코사인 유사도를 계산한다."
    )


    # --------------------------------------------------------
    # 검색 구조
    # --------------------------------------------------------

    st.code(
        """
한국어 검색어
      │
      ▼
NLLB 한국어 → 영어
      │
      ▼
영어 검색어
      │
      ▼
CLIP Text Encoder
      │
      ▼
Text Embedding
      │
      │               이미지
      │                 │
      │                 ▼
      │          CLIP Image Encoder
      │                 │
      │                 ▼
      │          Image Embedding
      │                 │
      └─────────┬───────┘
                ▼
       Shared Embedding Space
                │
                ▼
        Cosine Similarity
                │
                ▼
              Top-K
""",
        language=None,
    )


    # --------------------------------------------------------
    # 이미지 갤러리
    # --------------------------------------------------------

    gallery_preview = (
        load_gallery(
            "./image"
        )
    )


    st.metric(
        "검색 가능한 이미지",
        len(gallery_preview),
    )


    st.caption(
        "검색 대상 이미지는 "
        "./image 폴더에서 읽는다."
    )


    # --------------------------------------------------------
    # 검색어
    # --------------------------------------------------------

    query = st.text_input(
        "검색 문장",
        placeholder=(
            "예: 고양이, 꽃, 빨간 자동차, "
            "공원에서 뛰는 강아지"
        ),
        key="clip_query",
    )


    # --------------------------------------------------------
    # 예제
    # --------------------------------------------------------

    with st.expander(
        "검색 문장 예제"
    ):

        st.code(
            """
고양이
강아지
자동차
꽃
빨간 자동차
공원에서 뛰는 강아지
소파에 앉아 있는 고양이
바다 위에 있는 배
""",
            language=None,
        )


    # --------------------------------------------------------
    # 검색 실행
    # --------------------------------------------------------

    if st.button(
        "이미지 검색 실행",
        disabled=(
            not query.strip()
        ),
        key="run_search",
    ):

        try:

            gallery = (
                load_gallery(
                    "./image"
                )
            )


            if not gallery:

                st.warning(
                    "./image 폴더에 "
                    "검색할 이미지가 없다."
                )


            else:

                with st.spinner(
                    "NLLB 번역 및 "
                    "CLIP 이미지 검색 중..."
                ):

                    # ----------------------------------------
                    # CLIP
                    # ----------------------------------------

                    (
                        clip_model,
                        clip_processor,
                    ) = get_clip()


                    # ----------------------------------------
                    # NLLB
                    # ----------------------------------------

                    (
                        translator_model,
                        translator_tokenizer,
                    ) = get_translator()


                    # ----------------------------------------
                    # 검색
                    # ----------------------------------------

                    (
                        results,
                        elapsed,
                        clip_query,
                    ) = search_images(
                        model=(
                            clip_model
                        ),
                        processor=(
                            clip_processor
                        ),
                        query=(
                            query
                        ),
                        image_items=(
                            gallery
                        ),
                        top_k=(
                            top_k
                        ),
                        translator_model=(
                            translator_model
                        ),
                        translator_tokenizer=(
                            translator_tokenizer
                        ),
                    )


                # =================================================
                # 검색어 처리 결과
                # =================================================

                st.subheader(
                    "검색어 처리 결과"
                )


                query_col1, query_col2 = (
                    st.columns(2)
                )


                with query_col1:

                    st.markdown(
                        "#### 사용자 입력"
                    )

                    st.info(
                        query
                    )


                with query_col2:

                    st.markdown(
                        "#### CLIP 실제 입력"
                    )

                    st.info(
                        clip_query
                    )


                # =================================================
                # 검색 정보
                # =================================================

                metric1, metric2 = (
                    st.columns(2)
                )


                with metric1:

                    st.metric(
                        "검색 대상 이미지",
                        len(gallery),
                    )


                with metric2:

                    st.metric(
                        "검색 시간",
                        f"{elapsed:.3f}초",
                    )


                # =================================================
                # 검색 결과
                # =================================================

                st.subheader(
                    "검색 결과"
                )


                for start in range(
                    0,
                    len(results),
                    2,
                ):

                    columns = (
                        st.columns(2)
                    )


                    for (
                        offset,
                        column,
                    ) in enumerate(
                        columns
                    ):

                        index = (
                            start
                            + offset
                        )


                        if (
                            index
                            >= len(results)
                        ):

                            continue


                        item = (
                            results[index]
                        )


                        similarity = float(
                            item["similarity"]
                        )


                        with column:

                            st.image(
                                item["image"],
                                caption=(
                                    item["name"]
                                ),
                                use_container_width=True,
                            )


                            st.markdown(
                                f"### {index + 1}위"
                            )


                            st.write(
                                "파일: "
                                f"`{item['name']}`"
                            )


                            st.metric(
                                "코사인 유사도",
                                f"{similarity:.4f}",
                            )


                            # --------------------------------
                            # 코사인 유사도 범위는 -1 ~ +1이다.
                            #
                            # Streamlit progress는
                            # 0 ~ 1 범위를 사용하므로
                            # 화면 표시용 값으로 변환한다.
                            # --------------------------------

                            progress_value = (
                                similarity
                                + 1.0
                            ) / 2.0


                            progress_value = max(
                                0.0,
                                min(
                                    1.0,
                                    progress_value,
                                ),
                            )


                            st.progress(
                                progress_value
                            )


                st.caption(
                    "코사인 유사도는 이미지와 검색 문장의 "
                    "의미적 유사성을 나타내는 값이다. "
                    "정답 확률을 의미하지 않는다."
                )


        except Exception as exc:

            st.error(
                "이미지 검색 중 오류가 발생했다: "
                f"{exc}"
            )

            st.exception(
                exc
            )


# ============================================================
# 13. BLIP 한국어 이미지 설명
# ============================================================

with tab_caption:

    st.header(
        "BLIP 한국어 이미지 설명 생성"
    )


    st.write(
        "BLIP이 이미지 설명을 영어로 생성한 후 "
        "NLLB가 영어 설명을 한국어로 번역한다."
    )


    # --------------------------------------------------------
    # 처리 구조
    # --------------------------------------------------------

    st.code(
        """
이미지
  │
  ▼
BLIP
  │
  ▼
영어 이미지 설명
  │
  ▼
NLLB
영어 → 한국어
  │
  ▼
한국어 이미지 설명
""",
        language=None,
    )


    if uploaded_image is None:

        st.info(
            "먼저 상단에서 이미지를 업로드한다."
        )


    # --------------------------------------------------------
    # 실행 버튼
    # --------------------------------------------------------

    if st.button(
        "이미지 설명 생성",
        disabled=(
            uploaded_image is None
        ),
        key="run_caption",
    ):

        try:

            # =================================================
            # 1. BLIP Caption 생성
            # =================================================

            with st.spinner(
                "BLIP 이미지 설명을 생성하는 중..."
            ):

                (
                    blip_model,
                    blip_processor,
                ) = get_blip()


                (
                    english_caption,
                    blip_elapsed,
                ) = generate_caption(
                    model=(
                        blip_model
                    ),
                    processor=(
                        blip_processor
                    ),
                    image=(
                        uploaded_image
                    ),
                )


            # =================================================
            # 2. NLLB 영어 → 한국어
            # =================================================

            with st.spinner(
                "영어 설명을 한국어로 번역하는 중..."
            ):

                (
                    translator_model,
                    translator_tokenizer,
                ) = get_translator()


                translation_start = (
                    perf_counter()
                )


                korean_caption = (
                    translate_en_to_ko(
                        text=(
                            english_caption
                        ),
                        translator_model=(
                            translator_model
                        ),
                        translator_tokenizer=(
                            translator_tokenizer
                        ),
                    )
                )


                translation_elapsed = (
                    perf_counter()
                    - translation_start
                )


            # =================================================
            # 3. 한국어 설명
            # =================================================

            st.subheader(
                "생성된 한국어 설명"
            )


            st.success(
                korean_caption
            )


            # =================================================
            # 4. 영어 / 한국어 비교
            # =================================================

            col1, col2 = (
                st.columns(2)
            )


            # -------------------------------------------------
            # BLIP 원본
            # -------------------------------------------------

            with col1:

                st.markdown(
                    "#### BLIP 원본 영어 설명"
                )


                st.info(
                    english_caption
                )


            # -------------------------------------------------
            # NLLB 번역
            # -------------------------------------------------

            with col2:

                st.markdown(
                    "#### 한국어 번역"
                )


                st.info(
                    korean_caption
                )


            # =================================================
            # 5. 실행 시간
            # =================================================

            time_col1, time_col2 = (
                st.columns(2)
            )


            with time_col1:

                st.metric(
                    "BLIP 추론 시간",
                    f"{blip_elapsed:.3f}초",
                )


            with time_col2:

                st.metric(
                    "NLLB 번역 시간",
                    f"{translation_elapsed:.3f}초",
                )


            # =================================================
            # 6. 교육용 설명
            # =================================================

            st.caption(
                "BLIP이 한국어를 직접 생성하는 구조가 아니다. "
                "BLIP은 영어 Caption을 생성하고, "
                "NLLB가 영어 Caption을 한국어로 번역한다."
            )


        except Exception as exc:

            st.error(
                "이미지 설명 생성 중 오류가 발생했다: "
                f"{exc}"
            )

            st.exception(
                exc
            )


# ============================================================
# 14. 모델별 역할
# ============================================================

st.divider()


st.subheader(
    "모델별 역할"
)


st.markdown(
    """
| 모델 | 입력 | 출력 | 역할 |
|---|---|---|---|
| Faster R-CNN | 이미지 | 객체 + Bounding Box | 객체 탐지 |
| NLLB-200 | 한국어 | 영어 | CLIP 검색어 번역 |
| CLIP | 이미지 + 텍스트 | 임베딩 | 이미지 검색 |
| BLIP | 이미지 | 영어 텍스트 | 이미지 설명 생성 |
| NLLB-200 | 영어 | 한국어 | BLIP 설명 번역 |
"""
)


# ============================================================
# 15. CLIP 검색 원리
# ============================================================

with st.expander(
    "CLIP 이미지 검색 원리"
):

    st.markdown(
        "### 처리 과정"
    )


    st.code(
        """
한국어 검색 문장
       │
       ▼
      NLLB
한국어 → 영어
       │
       ▼
영어 검색 문장                    이미지
       │                            │
       ▼                            ▼
 Text Encoder                 Image Encoder
       │                            │
       ▼                            ▼
Text Embedding              Image Embedding
       │                            │
       └────────────┬───────────────┘
                    │
                    ▼
           Shared Embedding Space
                    │
                    ▼
            Cosine Similarity
                    │
                    ▼
                  Top-K
""",
        language=None,
    )


    st.markdown(
        "### 코사인 유사도"
    )


    st.latex(
        r"""
\operatorname{cosine}(\mathbf{x},\mathbf{y})
=
\frac{
\mathbf{x}\cdot\mathbf{y}
}{
\|\mathbf{x}\|_2
\|\mathbf{y}\|_2
}
"""
    )


    st.markdown(
        "### L2 정규화"
    )


    st.latex(
        r"""
\hat{\mathbf{x}}
=
\frac{
\mathbf{x}
}{
\|\mathbf{x}\|_2
}
"""
    )


    st.latex(
        r"""
\hat{\mathbf{y}}
=
\frac{
\mathbf{y}
}{
\|\mathbf{y}\|_2
}
"""
    )


    st.markdown(
        "L2 정규화를 수행한 경우 두 벡터의 "
        "내적으로 코사인 유사도를 계산할 수 있다."
    )


    st.latex(
        r"""
\operatorname{cosine}
=
\hat{\mathbf{x}}
\cdot
\hat{\mathbf{y}}
"""
    )


# ============================================================
# 16. BLIP 한국어 설명 원리
# ============================================================

with st.expander(
    "BLIP 한국어 이미지 설명 원리"
):

    st.code(
        """
이미지
  │
  ▼
BLIP Image Encoder
  │
  ▼
이미지 특징
  │
  ▼
BLIP Text Decoder
  │
  ▼
영어 Caption
  │
  │  예:
  │  "two women doing yoga poses"
  │
  ▼
NLLB-200
eng_Latn → kor_Hang
  │
  ▼
한국어 Caption
  │
  │  예:
  │  "요가 자세를 취하고 있는 두 여성"
  │
  ▼
화면 출력
""",
        language=None,
    )


    st.markdown(
        """
**중요한 구분**

BLIP이 한국어 설명을 직접 생성하는 것이 아니다.

BLIP은 이미지에서 **영어 Caption**을 생성하고,
NLLB-200이 해당 영어 문장을 **한국어로 번역**한다.
"""
    )


# ============================================================
# 17. 전체 시스템 구조
# ============================================================

with st.expander(
    "전체 시스템 구조"
):

    st.code(
        """
                         사용자
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
        이미지          한국어 검색어        이미지
          │                │                │
          ▼                ▼                ▼
    Faster R-CNN          NLLB             BLIP
          │                │                │
          ▼                ▼                ▼
       객체 탐지          영어          영어 Caption
          │                │                │
          ▼                ▼                ▼
   Bounding Box           CLIP             NLLB
                           │                │
                           ▼                ▼
                   Shared Embedding     한국어 Caption
                           │
                           ▼
                  Cosine Similarity
                           │
                           ▼
                         Top-K
                           │
                           ▼
                       이미지 검색
""",
        language=None,
    )


# ============================================================
# 18. 하단
# ============================================================

st.divider()


st.caption(
    "Multimodal AI Demo | "
    "Faster R-CNN + NLLB-200 + CLIP + BLIP"
)

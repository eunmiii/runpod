# Qwen2.5-VL 로컬 CPU 멀티모달 챗 (Ollama + Streamlit)
# 설치: pip install streamlit requests pillow
# Ollama 모델 준비: ollama pull qwen2.5vl:3b
# 실행: python -m streamlit run app.py
import base64
import hashlib
import io
import time
import requests
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

st.set_page_config(page_title="로컬 Qwen2.5-VL 멀티모달 챗", layout="wide")
st.title("Qwen2.5-VL 한국어 멀티모달 챗 (CPU)")
st.caption("Ollama 로컬 양자화 모델 사용 · 외부 API 불필요 · 최초 모델 다운로드 필요")

# Ollama는 별도의 프로세스에서 모델을 로드하고 CPU 추론을 수행한다.
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "qwen2.5vl:3b"
MAX_IMAGES = 3
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# Session State에는 화면용 대화와 실제 모델에 보낼 메시지를 분리하여 저장한다.
def init_state():
    defaults = {"messages": [], "image_signature": "", "images": [], "image_bytes": []}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def clear_history():
    st.session_state.messages = []

def encode_image(uploaded_file, max_side):
    # 이미지 형식 검사와 방향 보정을 수행한 뒤 RGB JPEG로 압축한다.
    raw = uploaded_file.getvalue()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError(f"{uploaded_file.name}: 파일 크기가 10MB를 초과합니다.")
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"{uploaded_file.name}: 지원되지 않는 이미지입니다.") from exc
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=82, optimize=True)
    return image, base64.b64encode(output.getvalue()).decode("ascii")

def create_signature(files, max_side):
    # 이미지 크기 설정을 바꾸어도 새로 처리하도록 설정값을 포함한다.
    digest = hashlib.sha256(str(max_side).encode("utf-8"))
    for file in files:
        digest.update(file.name.encode("utf-8"))
        digest.update(file.getvalue())
    return digest.hexdigest()

def check_ollama():
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        response.raise_for_status()
        models = [item.get("name", "") for item in response.json().get("models", [])]
        available = any(name == MODEL_NAME or name.startswith(MODEL_NAME + "-") for name in models)
        return True, available, models
    except requests.RequestException:
        return False, False, []

def build_messages(history, prompt, images_b64, max_turns):
    # 이미지는 첫 질문에만 포함하고 후속 질문에는 텍스트 대화를 이어 붙인다.
    # 최근 N개의 완성된 질문/답변 쌍만 유지해 CPU 처리량을 제한한다.
    recent = history[-2 * max_turns:]
    messages = [{"role": "system", "content": "이미지를 정확하게 관찰하고 한국어로 간결하게 답하세요. 확실하지 않으면 모른다고 답하세요."}]
    first_prompt = next((m["content"] for m in history if m["role"] == "user"), prompt)
    # 이미지 문맥을 잃지 않도록 최근 대화와 별도로 이미지 메시지를 항상 유지한다.
    messages.append({"role": "user", "content": first_prompt, "images": images_b64})
    if history:
        # 원래 첫 질문을 중복 추가하지 않도록 제외한다.
        rest = history[1:]
        messages.extend(rest[-2 * max_turns:])
        messages.append({"role": "user", "content": prompt})
    return messages

def ask_ollama(messages, num_predict, timeout_seconds):
    # 로컬 서버만 호출하며 외부 인터넷 API를 호출하지 않는다.
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {"num_predict": num_predict, "temperature": 0},
        "keep_alive": "10m",
    }
    started = time.perf_counter()
    response = requests.post(OLLAMA_URL, json=payload, timeout=(10, timeout_seconds))
    response.raise_for_status()
    result = response.json()
    answer = result.get("message", {}).get("content", "").strip()
    return answer or "모델이 빈 응답을 반환했습니다.", time.perf_counter() - started

init_state()
with st.sidebar:
    st.header("CPU 설정")
    st.write("모델:", MODEL_NAME)
    st.write("추론 장치: CPU (Ollama 설치/실행 설정 확인)")
    max_side = st.select_slider("이미지 최대 변(px)", options=[224, 336, 448, 560, 672], value=336)
    max_new_tokens = st.slider("최대 생성 토큰", 16, 128, 32, step=16)
    max_turns = st.slider("유지할 최근 대화 쌍", 1, 5, 2)
    timeout_seconds = st.slider("요청 대기 제한(초)", 60, 900, 300, step=60)
    if st.button("대화 초기화"):
        clear_history()
        st.rerun()
    if st.button("Ollama 연결 확인"):
        connected, model_ready, installed = check_ollama()
        if not connected:
            st.error("Ollama 서버에 연결할 수 없습니다. Ollama 실행 여부를 확인하세요.")
        elif not model_ready:
            st.warning(f"모델이 없습니다. 터미널에서 ollama pull {MODEL_NAME} 실행")
            st.write("설치 모델:", installed)
        else:
            st.success("Ollama와 모델을 확인했습니다.")

st.subheader("이미지 업로드")
uploaded_files = st.file_uploader("PNG/JPG/WEBP 이미지 (최대 3장, 각 10MB)", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
files = uploaded_files or []
if len(files) > MAX_IMAGES:
    st.warning(f"CPU 실습에서는 최대 {MAX_IMAGES}장만 사용합니다. 처음 {MAX_IMAGES}장을 선택합니다.")
    files = files[:MAX_IMAGES]
signature = create_signature(files, max_side)
if signature != st.session_state.image_signature:
    try:
        images, image_bytes = [], []
        for file in files:
            image, encoded = encode_image(file, max_side)
            images.append(image)
            image_bytes.append(encoded)
        st.session_state.images = images
        st.session_state.image_bytes = image_bytes
        st.session_state.image_signature = signature
        clear_history()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

if st.session_state.images:
    cols = st.columns(min(3, len(st.session_state.images)))
    for idx, img in enumerate(st.session_state.images):
        with cols[idx % len(cols)]:
            st.image(img, caption=f"Picture {idx + 1}: {img.width}×{img.height}", width="stretch")
else:
    st.info("먼저 이미지를 업로드하세요.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("이미지에 대해 한국어로 질문하세요.")
if prompt:
    if not st.session_state.image_bytes:
        st.warning("이미지를 먼저 업로드하세요.")
    else:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("로컬 CPU에서 추론 중입니다. 첫 요청은 모델 로딩으로 오래 걸릴 수 있습니다..."):
                try:
                    model_messages = build_messages(st.session_state.messages, prompt, st.session_state.image_bytes, max_turns)
                    answer, elapsed = ask_ollama(model_messages, max_new_tokens, timeout_seconds)
                    st.markdown(answer)
                    st.caption(f"추론 시간: {elapsed:.1f}초 | 이미지: {len(st.session_state.images)}장 | 최대 생성: {max_new_tokens}토큰")
                except requests.Timeout:
                    st.error("응답 제한 시간을 초과했습니다. 이미지를 1장/224px, 출력 16토큰으로 줄여 재시도하세요.")
                    st.stop()
                except requests.RequestException as exc:
                    st.error(f"Ollama 연결 또는 추론 오류: {exc}")
                    st.info(f"Ollama 실행과 모델 설치를 확인하세요: ollama pull {MODEL_NAME}")
                    st.stop()
        st.session_state.messages.extend([{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}])

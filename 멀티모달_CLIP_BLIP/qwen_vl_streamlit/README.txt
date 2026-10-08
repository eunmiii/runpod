Qwen2.5-VL Streamlit Multimodal Chat

1. 패키지 설치

uv add "torch==2.8.0" "torchvision==0.23.0" "transformers==4.57.2" "accelerate>=1.5,<2" "pillow>=10,<12" "streamlit>=1.40,<2"

2. 실행

uv run streamlit run qwen_vl_streamlit/app.py

3. 기능

- 이미지 한 장 또는 여러 장 업로드
- 한국어 질문
- 멀티이미지 비교
- 멀티턴 대화
- 대화 초기화
- CUDA / MPS / CPU 선택

4. 모델

Qwen/Qwen2.5-VL-3B-Instruct

주의:
최초 실행 시 사전학습 모델 가중치 다운로드가 필요합니다.
CPU에서는 3B급 멀티모달 모델 추론이 매우 느릴 수 있습니다.

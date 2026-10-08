# 다국어 멀티모달 AI 이미지 분석 시스템

한국어, 영어, 일본어, 중국어 UI를 지원하는 교육용 Streamlit 프로젝트이다.

## 제공 기능

- Faster R-CNN 기반 객체 탐지
- 다국어 CLIP 기반 텍스트-이미지 검색
- BLIP 기반 영어 이미지 캡션 생성
- NLLB 기반 캡션 번역
- 다국어 화면 문구와 일부 COCO 객체 클래스 이름 번역

## 요구 환경

- Windows 10/11
- Python 3.11 또는 3.12 권장
- CPU 실행 가능, NVIDIA GPU가 있으면 추론 속도가 향상될 수 있음
- 최초 실행 시 모델 가중치 다운로드를 위해 인터넷 연결 필요

## 설치

### 1. 가상환경 생성 및 활성화

PowerShell에서 프로젝트 폴더로 이동한 다음 실행한다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. PyTorch와 TorchVision 설치

CPU 또는 NVIDIA GPU 환경에 맞는 명령어를 공식 페이지에서 확인한다.

https://pytorch.org/get-started/locally/

### 3. 나머지 라이브러리 설치

```powershell
python -m pip install -r requirements.txt
```

## 실행

```powershell
streamlit run app.py
```

브라우저에서 `http://localhost:8501`을 연다.

## 이미지 검색 갤러리

`data/search_images/` 폴더에 `.jpg`, `.jpeg`, `.png`, `.webp` 파일을 추가한다.
검색 기능은 이 폴더의 이미지들을 대상으로 작동한다.

## 모델 안내

- 객체 탐지: `FasterRCNN_ResNet50_FPN_Weights.DEFAULT`
- 다국어 이미지 검색: `sentence-transformers/clip-ViT-B-32-multilingual-v1`
- 이미지 캡션: `Salesforce/blip-image-captioning-base`
- 캡션 번역: `facebook/nllb-200-distilled-600M`

## 주의사항

- 이 프로젝트는 교육용 예제이며, 배포 전 성능·보안·메모리 사용량 테스트가 필요하다.
- 최초 실행 시 모델 다운로드가 오래 걸릴 수 있다.
- NLLB 번역 모델은 크기가 크며 CPU에서는 번역이 느릴 수 있다.
- 다국어 CLIP의 언어별 검색 성능은 동일하지 않다.
- 객체 이름 번역은 코드에 정의한 사전 기반이며, 모든 COCO 클래스 번역을 포함하지는 않는다.
- PyTorch와 TorchVision의 버전 및 CUDA 호환성을 맞춰야 한다.



https://devocean.sk.com/blog/techBoardDetail.do?ID=167689

https://velog.io/@3un0ia/VisualMultimodal-AI-3-Instance-Segmentation-YOLO

https://railly-linker.tistory.com/220


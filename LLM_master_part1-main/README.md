# LLM 마스터 과정 특별 교육 시리즈 — Part 1

> **자연어 처리 기초부터 LLM 파인튜닝까지** — LLM을 제대로 이해하고 직접 학습시키기 위한 실습 중심 과정


NLP의 기본 개념(인코딩·토큰화·임베딩)에서 출발해 트랜스포머 구조, HuggingFace/LangChain 생태계, 그리고 LoRA·PEFT·SFT·Instruction Tuning 등 파인튜닝 실전까지 과정으로 다룹니다.

---


## 학습 목표

- 자연어를 컴퓨터가 다루는 방식(인코딩·토큰화·임베딩)을 이해한다.
- 트랜스포머 아키텍처와 Attention, BERT/GPT 계열의 구조적 차이를 설명할 수 있다.
- 생성 AI와 LLM의 전체 지형, sLLM의 위치를 파악한다.
- HuggingFace Transformers·Hub와 LangChain으로 LLM 애플리케이션을 구성할 수 있다.
- 파인튜닝 개념(FFT/PEFT, SFT/CPT/IT)을 구분하고, LoRA·PEFT로 실제 모델을 학습시킬 수 있다.
- SFT, 지속 사전학습, Instruction Tuning을 직접 수행하고 LLM-as-a-Judge로 결과를 평가한다.

---

## 커리큘럼

### 📅   NLP 기초와 LLM 생태계

| # | 세션 | 노트북 |
|---|---|---|
| 00 | 실습 환경 점검 | [setup_check.ipynb](setup_check.ipynb) |
| 01 | 자연어 처리 기초 — 인코딩, 토큰화, 임베딩(Word2Vec) | [01_nlp_encoding_tokenization.ipynb](01_nlp_encoding_tokenization.ipynb) |
| 02 | 트랜스포머 아키텍처 — Attention, BERT, GPT 구조 비교 | [02_transformer_bert_gpt.ipynb](02_transformer_bert_gpt.ipynb) |
| 03 | 생성 AI와 LLM 개요 (sLLM 포함) | [03_llm_overview_sllm.ipynb](03_llm_overview_sllm.ipynb) |
| 04 | Transformers 라이브러리 & HuggingFace Hub | [04_huggingface_ecosystem.ipynb](04_huggingface_ecosystem.ipynb) |
| 05 | LangChain 소개 및 기본 실습 | [05_langchain_practice.ipynb](05_langchain_practice.ipynb) |

### 📅  LLM 파인튜닝 실전

| # | 세션 | 노트북 |
|---|---|---|
| 06 | 파인튜닝 개념 정리 — SFT/CPT/IT, FFT/PEFT 한눈에 보기 | [06_finetuning_overview.ipynb](06_finetuning_overview.ipynb) |
| 07 | HuggingFace Transformers & TRL 기본 사용법 | [07_transformers_trl_basics.ipynb](07_transformers_trl_basics.ipynb) |
| 08 | LoRA vs Full Fine-tuning 이론 비교 | [08_lora_peft_theory.ipynb](08_lora_peft_theory.ipynb) |
| 09 | LoRA vs FFT 실전 비교 실습 | [09_lora_peft_practice.ipynb](09_lora_peft_practice.ipynb) |
| 10 | Next Token Prediction 기반 SFT 실습 | [10_sft_huggingface_trl.ipynb](10_sft_huggingface_trl.ipynb) |
| 11 | Continuous Pretraining (지속 사전학습) — Qwen2.5-1.5B (LoRA) | [11_continuous_learning.ipynb](11_continuous_learning.ipynb) |
| 12 | Instruction Tuning — Qwen2.5-1.5B (LoRA) | [12_instruction_tuning.ipynb](12_instruction_tuning.ipynb) |
| 13 | LLM-as-a-Judge — 자동 평가 메트릭과 GPT-4 평가 | [13_llm_as_judge.ipynb](13_llm_as_judge.ipynb) |

---

## 실습 환경 설정

먼저 저장소를 복제합니다.

```bash
git clone https://github.com/choki0715/LLM_master_part1.git
cd LLM_master_part1
```

이후 깡통(clean) Ubuntu 상태에서 다음 스크립트로 실습 환경을 자동 구성합니다. `sudo` 없이 `uv`로 독립형 **Python 3.11**을 설치하고 가상환경을 생성합니다.

```bash
bash setup.sh
```

생성되는 가상환경:

| 환경 | 용도 |
|---|---|
| `venv` | 메인 환경 (torch 2.11 / transformers 4.57 / vllm / langchain / gensim 등) |
| `venv-quant` | 양자화 전용 (torch 2.2 / transformers 4.46 / auto-gptq / autoawq) — 양자화 데모에서만 사용 |

> Python 3.11을 고정하는 이유: 최신 Python(3.14)에서는 gensim·auto-gptq·autoawq 등 다수 ML 패키지의 사전 빌드 휠이 없어 설치가 깨지기 때문입니다.

### 설치 확인

환경 구성 후 [setup_check.ipynb](setup_check.ipynb)를 실행해 주요 패키지와 GPU/CUDA 인식 여부를 점검하세요.

```bash
source venv/bin/activate
jupyter notebook   # 또는 VS Code / JupyterLab 사용
```

---


# DeepKNLP

Transformer-based Korean Natural Language Processing

## 이번 과정 (2026-09-03 ~ 09-04)

[AI아카데미] A4021 언어지능: 언어모델 기반 자연어처리 실습 기초

이 과정은 **같은 태스크를 세 가지 구조로 풀어 보고 무엇이 다른지 확인하는 것**이 뼈대입니다.

| 구조 | 하는 일 |
|---|---|
| **BERT** (인코더) | 읽고 **라벨이나 위치를 고른다**. 입력에 없는 말은 지어내지 못한다 |
| **T5** (인코더-디코더) | 읽고 **답을 글자로 써낸다**. 읽기 전용 인코더가 따로 있다 |
| **GPT 계열** (디코더) | 앞말에 이어 **계속 써낸다**. 지시문만 바꾸면 다른 일을 한다 |

구조의 이름은 인코더·인코더-디코더·디코더이고, 각 자리를 대표하는 모델이 BERT·T5·GPT입니다.
**요즘 이 디코더 계열을 아주 크게 키운 것을 LLM**이라고 부릅니다 — LLM은 구조 이름이 아니라
규모·쓰임을 가리키는 말입니다. 이 문서에서 구조를 나란히 놓을 때는 `GPT 계열`, 큰 모델을
가리킬 때는 `LLM` 이라고 씁니다.

| 일차 | 실습 | 파일 | 구조 | 내용 |
|---|---|---|---|---|
| 1일차 | 실습1 | `day1/01_주제분류.ipynb` | BERT → T5 | 주제분류 (KLUE-YNAT, 7개 주제) — 분류 head |
| 1일차 | 실습2 | `day1/02_문장유사도.ipynb` | BERT → T5 | 문장유사도 (KLUE-STS) — 회귀 head |
| 1일차 | 실습3 | `day1/03_개체명인식.ipynb` | BERT → T5 | 개체명인식 (KLUE-NER) — 토큰분류 head |
| 2일차 | 실습4A | `day2/04_기계독해_BERT.ipynb` | BERT (인코더) | 기계독해 (KorQuAD) — 지문에서 답의 시작·끝 위치를 고른다 |
| 2일차 | 실습4B | `day2/05_기계독해_T5.ipynb` | T5 (인코더-디코더) | 생성형 기계독해 (KorQuAD) — pko-T5가 답을 글자로 써낸다 |
| 2일차 | **실습5** | **`day2/06_LLM파인튜닝.ipynb`** (코드 `task5-llm-ft/`) | **GPT 계열 (디코더)** | **LLM 파인튜닝 — 하나의 LLM + LoRA 어댑터 하나로 여섯 태스크를 통합 학습하고, 같은 데이터·같은 평가셋으로 BERT·T5와 정면 비교** |

1일차 실습 세 개는 **같은 태스크를 BERT로 한 번, T5로 한 번** 풉니다. 앞부분(BERT)에는 여러분이
직접 채우는 **미션 셀**이 하나씩 있습니다 — 태스크마다 **객관식 두 문항**을 풀고 그 답을
빈칸(`____`) 두 곳에 옮겨 적는 식입니다. 뒷부분(T5)은 강사 시연을 따라 그대로 실행하면 됩니다.
2일차는 기계독해를 두 방식으로 푼 뒤, 실습5에서 모델을 그대로 두고 **지시문만 바꿔서** 여섯 태스크를
하나의 모델로 처리하고 **세 방식을 같은 평가셋·같은 채점 코드로 나란히 놓습니다.**

### 여섯 태스크 (5+1)

| # | 태스크 | 데이터 | 주 지표 | BERT | T5 | GPT 계열 |
|---|---|---|---|---|---|---|
| 1 | 주제분류 | KLUE-YNAT | 정확도 | ○ | ○ | ○ |
| 2 | 문장유사도 | KLUE-STS | 피어슨 상관 | ○ | ○ | ○ |
| 3 | 개체명인식 | KLUE-NER | 개체 F1 | ○ | ○ | ○ |
| 4 | 기계독해 | KorQuAD v1 | EM | ○ | ○ | ○ |
| 5 | SQL생성 | Spider-ko | 정규화 EM | × | ○ | ○ |
| +1 | 수학추론 (보너스) | GSM8K-ko | EM | × | ○ | ○ |

앞의 네 개는 세 방식이 모두 풀 수 있고, SQL생성·수학추론은 답이 라벨도 지문 속 위치도 아니어서
생성 모델(T5·LLM)만 풀 수 있습니다. **수학추론은 시간이 남을 때 다루는 보너스**입니다 —
같은 방식으로 파인튜닝했을 때 SQL생성은 9개 모델 모두 좋아졌지만 수학추론은 3개만 좋아졌고,
"파인튜닝이 항상 이기지는 않는다"를 보여 주는 사례로 씁니다. 수치는 `docs/LECTURE-FACTS.md` 를 보세요.

### 미션 정답과 퀴즈 해답 — 노트북 안에서 객관식으로

미션과 퀴즈의 정답은 **노트북 안에서** 봅니다. 터미널 명령은 없습니다. **미션 셀 바로 위**의
**객관식 셀**을 실행하면 보기 네 개가 버튼으로 나오고, 하나를 고르면 **정답이면 바로**, 오답이면 **5초 뒤에** 알려 줍니다(눌러 맞히기를 막으려는 것입니다. 힌트 버튼도 5초).

- 틀리면 왜 틀렸는지 한 줄 설명과 함께 **힌트**가 단계별로 열립니다 — 개념 → 참고 자료·웹 검색 링크 →
  "AI에게 이렇게 물어보세요" 예시 질문. 스스로 찾아가며 답을 완성하는 것이 목표입니다.
- 힌트로도 안 되면 **[정답 바로 보기]** 를 누릅니다. "정말 보시겠습니까? 더 생각해 보시겠습니까?"를
  두 번 확인하고 10초를 기다린 뒤 정답 코드가 열립니다. 복사 버튼으로 미션 셀에 붙여 넣으면 됩니다.
- 강의 페이지(`lecture-site/index.html`)의 확인문제도 같은 화면·같은 방식입니다.

정답은 저장소에 평문으로 들어 있지 않습니다(`quiz/public.json` 은 고른 보기가 맞을 때만 풀리는 형태).

**1일차가 끝나면** 1일차 항목의 정답을 [`docs/DAY1-ANSWERS.md`](docs/DAY1-ANSWERS.md) 에 공개합니다. 2일차(기계독해·실습5) 정답은 2일차가 끝난 뒤 같은 방식으로 공개합니다.

## 강의자료

모두 **인터넷 없이** 브라우저로 바로 열립니다(파일 하나에 글꼴·그림까지 들어 있습니다).

| 자료 | 파일 | 무엇 |
|---|---|---|
| **1일차 이론 슬라이드** (인터랙티브 HTML) | [`lectures/2026-09/deck/index.html`](lectures/2026-09/deck/index.html) | 80장. `S` 키로 발표자 노트, `?print-pdf` 로 인쇄용 배치, 화살표·`ESC`(전체 보기)로 이동 |
| 같은 내용 PDF | [`lectures/2026-09/deck/A4021-1일차.pdf`](lectures/2026-09/deck/A4021-1일차.pdf) | 인쇄·배포용 80쪽 |
| 발표자 노트 | [`lectures/2026-09/deck/A4021-1일차-노트.html`](lectures/2026-09/deck/A4021-1일차-노트.html) | 슬라이드별 설명을 한 문서로 — 복습용 |
| **강의 페이지** (이론·실습 해설·확인문제 24개) | [`lecture-site/index.html`](lecture-site/index.html) | 슬라이드보다 깊은 해설과 객관식 확인문제 |



## Code Reference

* ratsgo nlpbook: https://ratsgo.github.io/nlpbook | https://github.com/ratsgo/ratsnlp | https://ratsgo.github.io/nlpbook/docs/tutorial_links
* Pytorch Lightning: https://github.com/Lightning-AI/pytorch-lightning | https://lightning.ai/docs/fabric/stable
* HF(🤗) Datasets: https://huggingface.co/docs/datasets/index
* HF(🤗) Accelerate: https://huggingface.co/docs/accelerate/index
* HF(🤗) Transformers: https://github.com/huggingface/transformers | https://github.com/huggingface/transformers/tree/main/examples/pytorch

## Data Reference

이번 과정에서 쓰는 데이터만 적었습니다.

* KLUE(Korean Language Understanding Evaluation) — 주제분류(YNAT) · 문장유사도(STS) · 개체명인식(NER):
  https://huggingface.co/datasets/klue/klue | https://klue-benchmark.com
* KorQuAD 1.0(한국어 기계독해): https://huggingface.co/datasets/KorQuAD/squad_kor_v1 | https://korquad.github.io/category/1.0_KOR.html
* Spider-ko(Text-to-SQL, 한국어 질문): https://huggingface.co/datasets/huggingface-KREW/spider-ko | 스키마 https://huggingface.co/datasets/richardr1126/spider-schema
* GSM8K-ko(초등 수준 수학 문장제, 한국어판): https://huggingface.co/datasets/kuotient/gsm8k-ko | 원본 https://huggingface.co/datasets/openai/gsm8k

## Model Reference

세 구조를 대표하는 모델입니다. **굵은 것이 이번 과정에서 실제로 쓰는 모델**입니다.

* 인코더(BERT 계열): https://huggingface.co/docs/transformers/main/en/model_summary#nlp-encoder
    - **KLUE-RoBERTa**: https://huggingface.co/klue/roberta-base | https://github.com/KLUE-benchmark/KLUE
    - KLUE-BERT: https://huggingface.co/klue/bert-base
    - KcBERT(구어·댓글): https://huggingface.co/beomi/kcbert-base | https://github.com/Beomi/KcBERT
    - KoELECTRA: https://huggingface.co/monologg/koelectra-base-v3-discriminator | https://github.com/monologg/KoELECTRA
* 인코더-디코더(T5 계열): https://huggingface.co/docs/transformers/main/en/model_summary#nlp-encoder-decoder
    - **pko-T5**: https://huggingface.co/paust/pko-t5-base | https://github.com/paust-team/pko-t5
    - KE-T5: https://huggingface.co/KETI-AIR/ke-t5-base | https://github.com/airc-keti/ke-t5
    - KoT5: https://huggingface.co/wisenut-nlp-team/KoT5-base
* 디코더(GPT 계열): https://huggingface.co/docs/transformers/main/en/model_summary#nlp-decoder
    - **A.X-4.0-Light**(2일차 실습5 기본, SKT 7.3B, Apache-2.0): https://huggingface.co/skt/A.X-4.0-Light
    - **EXAONE-4.0-1.2B**(라인업에서 가장 가벼운 대안): https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B
    - kanana-1.5-2.1B: https://huggingface.co/kakaocorp/kanana-1.5-2.1b-instruct-2505
    - Qwen3.5-2B: https://huggingface.co/Qwen/Qwen3.5-2B
    - 초기 한국어 GPT(참고): KoGPT2 https://huggingface.co/skt/kogpt2-base-v2 · Polyglot-Ko-1.3B https://huggingface.co/EleutherAI/polyglot-ko-1.3b
* 국내 연구기관이 공개한 모델
    - **ETRI 언어지능연구실**: https://huggingface.co/etri-lirs — EAGLE 계열(한국어 디코더),
      KEByT5 계열(글자가 아니라 바이트를 직접 다루는 토큰 프리 인코더-디코더).
      이 실습 코드로 그대로 돌려 본 실측은 [`docs/ETRI-MODELS.md`](docs/ETRI-MODELS.md)

## Installation

### 수강생 PC — 1일차 (이 순서대로 한 줄씩)

강의장 PC와 같은 환경(Ubuntu + NVIDIA GPU)을 기준으로 합니다. **conda 는 쓰지 않습니다.**
CUDA 툴킷도 따로 설치하지 않습니다 — NVIDIA 드라이버만 있으면 PyTorch 휠이 CUDA 런타임을 함께 가져옵니다.

1. 저장소를 받습니다 (약 136MB, 내려받는 양은 약 25MB, 2~5분)
    ```bash
    git clone https://github.com/chrisjihee/DeepKNLP-26.09.git
    cd DeepKNLP-26.09
    ```
2. 1일차 환경을 만듭니다 — uv 설치 → Python 3.12 가상환경(`.venv`) → 패키지 → **1일차 모델만**(1GB 미만)
    ```bash
    bash setup_classroom.sh day1
    ```
3. 잘 됐는지 확인합니다
    ```bash
    bash setup_classroom.sh check
    ```
    `cuda True` 와 `OK  [1일차] klue/roberta-base` 가 보이면 됩니다.
    `없음  [2일차] ...` 는 정상입니다 — 2일차 모델은 아직 받지 않습니다.
4. 노트북을 엽니다
    ```bash
    .venv/bin/jupyter lab
    ```
    `day1/01_주제분류.ipynb` 부터 시작합니다. 미션 셀의 빈칸(`____`)을 채우지 않고 내려가면
    **'확인' 셀에서** `NameError: name '____' is not defined` 로 멈추는 것이 정상입니다 —
    그 빈칸을 채우는 것이 실습입니다.

### 수강생 PC — 2일차 준비 (1일차 오후 실습이 끝난 뒤)

실습5 기본 모델 `skt/A.X-4.0-Light`(7.3B)가 약 15GB이고, 1일차 모델까지 합쳐 **약 16GB · 7~15분** 입니다.
가벼운 대안 두 개(약 8GB)는 선택입니다 — `bash setup_classroom.sh models-alt`. 여러 대가 동시에 받으면 느려지므로 **1일차 실습을 마친 뒤** 받습니다.

```bash
bash setup_classroom.sh models     # 2일차 LLM까지 전부 (내려받는 동안 자리를 비워도 됩니다)
bash setup_classroom.sh smoke      # 2분짜리 학습 연기 테스트 (선택)
```

설치 중 문제가 생기면 [`docs/STUDENT-QUICKSTART.md`](docs/STUDENT-QUICKSTART.md) 의 마지막 절을 보세요.

### 한 번에 하기 (강사 PC·GPU 서버)

```bash
git clone https://github.com/chrisjihee/DeepKNLP-26.09.git && cd DeepKNLP-26.09
bash setup_classroom.sh all      # env → models → smoke → check (약 15~25분, 대부분 다운로드)
```

`setup_classroom.sh` 는 단계별로 나눠 실행할 수도 있습니다
(`check` / `env` / `models-day1` / `models` / `smoke` / `day1`).

2026-09-02 강의장 PC(Ubuntu 24.04, RTX 4500 Ada 24GB, 드라이버 570/CUDA 12.8, uv 0.12.5)에서 확인한 환경은
torch 2.11.0+cu128 · transformers 5.16.1 · trl 1.12.0 · peft 0.20.0 (GPU 서버와 동일). torch 와 나머지 패키지를
한 번에 섞어 설치하면 uv 0.12 가 PyTorch 인덱스의 옛 `requests` 에 걸려 실패하므로, 스크립트는 torch 를 먼저 받습니다.
같은 PC에서 실습 5의 실제 수업 학습(**A.X-4.0-Light**, `train_main.jsonl` 4,800건, 1 epoch)을 그대로 돌려 보니
**학습 22.1분(300스텝) · GPU 최대 17.08 GB**(`nvidia-smi` 기준 18.7GB)입니다.
학습 전·후 평가도 한 번에 여러 예제를 생성하므로 그만큼 씁니다 — 24GB 카드에 여유가 넉넉하지 않아
**앞 노트북(실습 4A·4B)의 커널을 먼저 종료해야** 합니다. 남겨 두면 모델을 올리는 단계에서 바로
`OutOfMemoryError` 가 나거나 학습을 다 돌린 뒤 평가에서 멈춥니다. 평가 시간은 다시 재고 있습니다.
학습 전→후 점수도 같은 PC에서 쟀습니다(태스크당 300건) — **6태스크 평균 51.9 → 69.2**,
주제분류 58.7 → 81.0 · 개체명 43.7 → 72.9 · 기계독해 EM 61.7 → 87.0 · 문장유사도 78.8 → 87.7 ·
SQL 17.0 → 34.3 · **수학 51.7 → 52.0**. 옛 기본 모델 EXAONE-4.0-1.2B 는 같은 PC에서
학습 9.7분 · GPU 4.3GB 였고, 수학추론은 파인튜닝 뒤 **떨어졌습니다** — 모델을 바꾼 이유가 그것입니다.

GPU 서버(A6000)에서 같은 설정으로 돌린 값과 열한 모델 라인업 비교는 `docs/LECTURE-FACTS.md` §4 가
정본입니다. 두 기계의 값은 조금 다르지만 방향은 같습니다.

**맥(Apple Silicon)** 에서도 됩니다 — `uv venv .venv --python 3.12 && uv pip install -e .` 두 줄이면 끝이고(PyTorch를 따로 받지 않음),
실습 5는 명령 그대로, 실습 1·2는 `--accelerator mps --cpu-workers 0`만 붙이면 맥 GPU(MPS)로 돕니다.
M2 Max 64 GB에서 A6000과 같은 코드를 재보니 학습은 약 10배, 추론은 약 5배 느리고 점수는 같았습니다. 실측표·설치·주의점은
[`docs/MAC-APPLE-SILICON.md`](docs/MAC-APPLE-SILICON.md).

수동으로 하려면:

1. uv와 Python 설치
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    $HOME/.local/bin/uv python install 3.12
    ```
2. 저장소 clone
    ```bash
    git clone https://github.com/chrisjihee/DeepKNLP-26.09.git
    cd DeepKNLP-26.09
    ```
3. 가상환경 만들고 패키지 설치 (CUDA 12.8 빌드의 PyTorch)
    ```bash
    uv venv .venv --python 3.12
    source .venv/bin/activate
    uv pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128   # torch 먼저 (PyTorch 인덱스만)
    uv pip install -e .                                                                 # 나머지는 PyPI
    python -m ipykernel install --user --name deepknlp --display-name "Python (DeepKNLP)"
    ```
4. 확인
    ```bash
    python -c "import torch, transformers, trl, peft; print(torch.__version__, torch.cuda.is_available(), transformers.__version__, trl.__version__, peft.__version__)"
    ```
5. 실습 모델 미리 받아두기 (강의장 인터넷이 느릴 때)
    ```bash
    hf download skt/A.X-4.0-Light              # 디코더 (실습5 기본)
    hf download paust/pko-t5-base              # 인코더-디코더 (실습4B, 실습5 T5 기준선)
    hf download klue/roberta-base              # 인코더 (실습1·2·4A, 실습5 BERT 기준선)
    ```
* **모델은 Hugging Face 에서 옵니다.** 코드에는 저장소 주소 하나(`skt/A.X-4.0-Light`)만 있고,
  `from_pretrained` 가 내려받아 `~/.cache/huggingface/hub` 에 둡니다 — 한 번 받으면 다시 받지 않습니다.
  저장소 첫 화면이 **모델 카드**이고 라이선스·크기·쓰는 법·성능이 거기 적혀 있습니다
  (실습5 모델: https://huggingface.co/skt/A.X-4.0-Light).
* **이 과정에서 쓰는 모델은 전부 공개 모델이라 로그인도 토큰도 필요하지 않습니다.**
* 일부 저장소는 **gated** 여서 약관 동의와 토큰이 필요합니다(Llama 계열 등). 그럴 때만
  ① 모델 카드에서 약관 동의 → ② Settings → Access Tokens 에서 **읽기 전용** 토큰 발급 →
  ③ `hf auth login` 으로 한 번 넣어 둡니다.
  **토큰을 코드·노트북에 적지 마세요** — `hf auth login` 이나 환경변수 `HF_TOKEN` 을 씁니다.
* GPU 모니터링
    ```bash
    watch -d -n 3 nvidia-smi
    ```
* 주피터 노트북
    ```bash
    jupyter lab            # 브라우저에서 day2/06_LLM파인튜닝.ipynb 열기
    ```

## Target Tasks

### 1일차 — 주피터 노트북 세 개 (`day1/`)

노트북 안에서 학습·평가까지 모두 합니다. 터미널 명령을 외울 필요가 없습니다.

```bash
.venv/bin/jupyter lab       # day1/01_주제분류.ipynb → 02_문장유사도.ipynb → 03_개체명인식.ipynb
```

각 노트북은 **BERT로 한 번, T5로 한 번** 같은 문제를 풉니다. 앞부분에 여러분이 채우는 **미션 셀**
(빈칸 `____` 두 개)이 있고, **그 바로 위**의 객관식 셀에서 답을 고르면 힌트와 정답이 열립니다.
연 정답은 그대로 복사해 미션 셀의 빈칸에 붙여 넣으면 됩니다.

같은 학습을 터미널에서 확인하고 싶을 때 쓰는 명령입니다(선택).

```bash
python task5-llm-ft/bert_baseline.py --task tc --mode budget --limit 300   # 인코더
python task5-llm-ft/t5_baseline.py   --task tc --mode budget --limit 300   # 인코더-디코더
```

### 2일차 — 기계독해와 LLM 파인튜닝

* **기계독해(추출형, 실습4A)** — BERT가 지문에서 답의 시작·끝 위치를 고릅니다: https://ratsgo.github.io/nlpbook/docs/qa
    - `See task4A-qa-ext/README.md`
    - `python task4A-qa-ext/train_qa.py --help`
    - `python task4A-qa-ext/infer_qa.py`
* **기계독해(생성형, 실습4B)** — pko-T5가 같은 KorQuAD를 답을 써내는 방식으로 풉니다
    - `See task4B-qa-gen/README.md`
    - `python task4B-qa-gen/train_qa_seq2seq.py --help`
    - `python task4B-qa-gen/infer_qa_seq2seq.py`
* **LLM 파인튜닝(통합, 실습5)** — 하나의 모델 + LoRA 어댑터 하나로 여섯 태스크: `See task5-llm-ft/README.md`
    - 주피터 노트북(수강생): `day2/06_LLM파인튜닝.ipynb`
    - 주피터 노트북(비교·심화): `task5-llm-ft/bert_vs_llm.ipynb`
    - `python task5-llm-ft/evaluate.py --tasks tc,ner,mrc,sts,sql,math --limit 100 --save output/before.json`
    - `python task5-llm-ft/train.py --data data/llm-ft/train_main.jsonl`
    - `python task5-llm-ft/evaluate.py --adapter output/llm-ft --tasks tc,ner,mrc,sts,sql,math --limit 100 --save output/after.json`
    - `python task5-llm-ft/serve.py --adapter output/llm-ft --port 9005`   (학습한 모델에 직접 물어보기)

### 지난 과정 자료 (자율 학습용)

2026년 3월 과정의 실습 코드는 별도 저장소에 그대로 있습니다 —
**https://github.com/chrisjihee/DeepKNLP-26.03** (감성분류 NSMC · 개체명인식 · 문장 생성 KoGPT2 · 기계독해).
관심 있는 분은 각자 내려받아 참고하시면 됩니다.

이번 저장소는 그 뒤의 흐름을 반영해 다시 짰습니다 — 태스크를 최신 한국어 벤치마크(KLUE·KorQuAD·Spider-ko·GSM8K-ko)로
바꾸고, **같은 데이터·같은 평가셋·같은 채점 함수**로 BERT·T5·GPT 계열 셋을 나란히 비교하며,
LoRA로 하나의 모델이 여섯 태스크를 함께 배우는 데까지 갑니다.

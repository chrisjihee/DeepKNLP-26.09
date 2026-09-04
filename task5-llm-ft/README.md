# 실습5. LLM 파인튜닝 — 같은 태스크를 세 가지 구조로 풀어 보기

1일차 실습에서 이미 두 가지 구조를 써 봤습니다. 여기서 세 번째 구조를 쓰고, **셋을 같은 데이터로 나란히 놓습니다.**

| 구조 | 하는 일 | 이 과정의 실습 |
|---|---|---|
| **BERT** (인코더) | 읽고 **라벨이나 위치를 고른다**. 입력에 없는 말은 지어내지 못한다 | 1일차 실습1·2·3(주제분류·문장유사도·개체명인식) · 2일차 실습4A(추출형 기계독해) |
| **T5** (인코더-디코더) | 읽고 **답을 글자로 써낸다**. 읽기 전용 인코더가 따로 있다 | 1일차 세 노트북의 T5 파트 · 2일차 실습4B(생성형 기계독해, pko-T5) |
| **GPT 계열** (디코더) | 앞말에 이어 **계속 써낸다**. 지시문만 바꾸면 다른 일을 한다 | **실습5 (이 폴더)** |

1일차와 2일차 오전에는 태스크마다 **따로** 학습시켰습니다 — BERT에는 태스크마다 다른 head를 붙였고, T5에는 같은 모델에 태스크 표시만 바꿔 붙였습니다.
여기서는 **하나의 LLM에 LoRA 어댑터 하나**만 붙여 여섯 태스크를 **함께** 학습시킵니다.
태스크를 바꾸는 것은 모델 구조가 아니라 **지시문(prompt)** 입니다.

그리고 같은 평가셋·같은 채점 코드로 BERT 기준선(`bert_baseline.py`)과 T5 기준선(`t5_baseline.py`)을
함께 돌려, 세 방식을 숫자로 비교합니다.

## 태스크 5+1개

| 태스크 | 데이터 | 지표 | 학습 / 평가 | BERT | T5 | GPT 계열 |
|---|---|---|---|:---:|:---:|:---:|
| 주제분류 | KLUE-YNAT | 정확도 (+macro F1) | 800 / 300 | O | O | O |
| 개체명인식 | KLUE-NER | 개체 단위 F1 (텍스트·유형 모두 일치해야 정답) | 1,000 / 300 | O | O | O |
| 기계독해 | KorQuAD v1 | EM(완전일치) / F1(글자 단위) | 800 / 300 | O | O | O |
| 문장유사도 | KLUE-STS | Pearson 상관 (+3점 기준 이진 정확도) | 600 / 300 | O | O | O |
| SQL생성 | Spider-ko | EM (+토큰 F1) | 800 / 300 | **불가** | O | O |
| 수학추론 *(보너스)* | GSM8K-ko | EM(최종 답 일치) | 800 / 300 | **불가** | O | O |

학습/평가 건수는 `build_dataset.py` 기본값입니다. 실제로 만들어진 건수는 `data/llm-ft/stats.json` 에 기록됩니다.

**왜 아래 두 개는 BERT로 못 하는가.** 인코더는 정해진 라벨 중 하나를 고르거나, 지문 안에서 답의 시작·끝 위치를
가리키는 것까지만 합니다. 그런데 `SELECT name FROM singer WHERE age < 30` 이나 "32 − 20 = 12"의 `12` 는
입력 어디에도 그대로 들어 있지 않습니다. 답을 **글자로 써내야** 하므로 생성 모델(T5·LLM)만 할 수 있습니다.
이것이 "구조가 할 수 있는 일의 범위"를 보여주는 자리입니다.

태스크 정의의 정본은 `common.py` 의 `TASKS` · `MAIN_TASKS` · `THREE_WAY_TASKS` · `GEN_ONLY_TASKS` · `BONUS_TASKS` 입니다.
정규 과정은 앞의 다섯이고 **수학추론은 시간이 남을 때 다루는 보너스**입니다 — 파인튜닝으로 오히려 나빠지는
모델이 더 많아서, "파인튜닝이 항상 이기지는 않는다"를 보여 주는 자리로 씁니다(`docs/LECTURE-FACTS.md` §1).

### 왜 감성분류(NSMC)와 자연어추론(KLUE-NLI)을 뺐는가 *(지난 과정과 달라진 점)*

이전 구성의 여섯은 감성분류 · 개체명인식 · 기계독해 · 주제분류 · 자연어추론 · 문장유사도였습니다.
여기에 생성 모델만 할 수 있는 SQL생성 · 수학추론을 넣으면서 감성분류와 자연어추론을 뺐습니다.
근거는 이전 구성으로 돌린 스윕 결과(`results/sweep/`, 소형 LLM 13종, 태스크당 300건)입니다.
**감성분류(NSMC)** 는 학습 전 중앙값이 이미 86.0이고 학습 후가 90.0이라, 파인튜닝으로 무엇이 달라지는지가
잘 드러나지 않습니다. **자연어추론(KLUE-NLI)** 은 학습 후 중앙값 84.3으로 문장유사도(85.4)와 사실상 같은
자리에 있고, 둘 다 "문장 두 개를 놓고 관계를 본다"는 점에서 수업에서 차지하는 위치도 겹칩니다.
하나만 남긴다면 문장유사도 쪽이 낫다고 판단했습니다. 이유는 두 가지입니다.
첫째, "두 문장이 얼마나 비슷한가를 0~5점으로"는 NLP 배경이 없어도 설명 없이 이해됩니다(함의·중립·모순은
개념부터 설명해야 합니다). 둘째, 학습 전→후 변화가 훨씬 극적입니다 — EXAONE-4.0-1.2B 13.7 → 79.9,
kanana-1.5-2.1B-base 12.0 → 84.6, Qwen3.5-0.8B-Base는 **−17.6** → 85.4 입니다.
(위 셋은 **이전 구성 스윕**의 값입니다. 지금 수업 기본 모델인 A.X-4.0-Light 는 학습 전부터 79.3이라
변화폭이 이만큼 크지는 않습니다 — 큰 모델은 학습 전에도 형식을 어느 정도 지킵니다.)
상관이 음수라는 것은 학습 전 모델이 비슷한 문장 쌍에 오히려 낮은 점수를 주고 있었다는 뜻입니다.

두 태스크의 코드·지시문·채점 함수는 지운 것이 아니라 그대로 남아 있습니다(`cls`, `nli`).
옛 구성으로 학습한 어댑터도 `--tasks cls,nli,...` 로 그대로 돌아갑니다.

## 두 개의 노트북

| 노트북 | 대상 | 내용 |
|---|---|---|
| `day2/06_LLM파인튜닝.ipynb` | **수강생 실습** | 데이터 → 토크나이저 → 학습 전 평가 → LoRA → SFTTrainer → 학습 후 평가 → 기준선 비교 → 데모 페이지 |
| `bert_vs_llm.ipynb` | 강사 해설·심화 | LLM 스윕 결과와 BERT·T5 기준선을 표·차트로 비교. 학습 없이 `results/` 만 읽음 |

노트북은 저장소 루트에서 `jupyter lab` 을 띄우고 열면 됩니다 (첫 셀이 작업 폴더를 맞춥니다).
터미널에서 하고 싶으면 아래 명령을 그대로 쓰면 됩니다.

## 실습 흐름 (터미널)

| 단계 | 하는 일 | 파일 |
|---|---|---|
| 1 | 여섯 태스크를 하나의 instruction 형식으로 합치기 | `build_dataset.py` |
| 2 | 학습 전 성능 재기 (기준선) | `evaluate.py` |
| 3 | LoRA로 학습 | `train.py` |
| 4 | 학습 후 성능 재기 (같은 방법으로) | `evaluate.py` |
| 5 | 내 모델에게 직접 물어보기 | `serve.py` |
| 보너스 A | 같은 데이터로 **BERT**(인코더)를 학습시켜 비교 | `bert_baseline.py` |
| 보너스 B | 같은 데이터로 **T5**(인코더-디코더)를 학습시켜 비교 | `t5_baseline.py` |

### 1. 데이터 만들기

```bash
python task5-llm-ft/build_dataset.py
```

- 주 학습셋: 각 데이터셋의 공식 `train`에서만 생성 → `data/llm-ft/train_main.jsonl` (여섯 태스크)
- 평가셋: 공식 `test`/`validation`에서만 생성 → `data/llm-ft/eval_<task>.jsonl` (태스크당 300건)
- **평가셋과 겹치는 문장은 학습셋에서 자동 제외**하고, 제거 후 겹침이 0건인지 다시 확인합니다
- 옛 구성 학습셋(`train.jsonl` 3태스크, `train_all.jsonl` 감성분류·자연어추론 포함)도 함께 만들어 둡니다
  (공개 저장소에는 이번 과정에서 쓰는 `train_main.jsonl` 만 들어 있습니다)

크기를 바꾸려면: `--n-tc 1000 --n-ner 1500 --n-mrc 1000 --n-sts 800 --n-sql 1000 --n-math 1000 ...`

### 2. 학습 전 성능 재기

```bash
python task5-llm-ft/evaluate.py --tasks tc,ner,mrc,sts,sql,math --limit 100 --save output/before.json
```

기본 모델은 `skt/A.X-4.0-Light` 입니다. 출력 예시를 함께 보려면 `--show 2` 를 붙입니다.
`--tasks` 를 생략하면 `data/llm-ft/` 에 평가 파일이 있는 태스크를 모두 돌립니다.

### 3. 학습

```bash
# 먼저 학습에 들어갈 텍스트가 어떻게 만들어지는지 확인 (학습은 하지 않음)
python task5-llm-ft/train.py --data data/llm-ft/train_main.jsonl --inspect

# 학습 (A.X-4.0-Light, 6태스크, 1 epoch → output/llm-ft)
python task5-llm-ft/train.py --data data/llm-ft/train_main.jsonl
```

다른 모델: `--model Qwen/Qwen3.5-2B --out output/llm-ft-qwen35-2b`.
Base 모델(chat template 없음)도 그대로 됩니다 — 단순 대화 형식을 자동으로 붙입니다.
실험용 옵션: `--plain-template`(Instruct 모델에도 단순 형식을 강제 → 답변 부분에만 손실 가능; 평가 때도 같은 옵션),
`--full-loss`(답변 손실을 끄고 전체 시퀀스로 학습). 손실을 어디에 거는지가 점수를 꽤 바꿉니다 —
근거는 `bert_vs_llm.ipynb` §2-1 과 `results/ablation/` (이전 구성 기준).

GPU 메모리가 부족하면:

```bash
# 배치를 줄이고 누적을 늘린다 (유효 배치는 그대로)
python task5-llm-ft/train.py --data data/llm-ft/train_main.jsonl --batch-size 1 --grad-accum 16

# 4비트 양자화(QLoRA)로 메모리를 더 줄인다
python task5-llm-ft/train.py --data data/llm-ft/train_main.jsonl --load-4bit --batch-size 2 --grad-accum 8
```

맥(Apple Silicon)에서는 위 명령이 그대로 MPS로 돕니다. 다만 M2 Max 기준 GPU의 약 1/10 속도라 수업 중에는
`--max-steps 10`(약 3분)으로 흐름만 확인하고, 전체 학습은 따로 걸어두세요 — [`docs/MAC-APPLE-SILICON.md`](../docs/MAC-APPLE-SILICON.md).

### 4. 학습 후 성능 재기

```bash
python task5-llm-ft/evaluate.py --adapter output/llm-ft --tasks tc,ner,mrc,sts,sql,math --limit 100 --save output/after.json
```

### 5. 데모 페이지

```bash
python task5-llm-ft/serve.py --adapter output/llm-ft --port 9005
```

브라우저에서 `http://localhost:9005` 접속.
데모는 **어댑터를 껐다 켜면서 학습 전/후 답변을 나란히** 보여줍니다.
기본으로 여섯 태스크가 모두 나옵니다. 세 방식이 겹치는 네 개만 보려면 `--tasks tc,ner,mrc,sts` 를 붙입니다.

### 보너스 A. BERT(인코더) 기준선

```bash
# budget: LLM과 같은 학습 예제 수 (태스크당 600~1,000건, 3 epoch)
python task5-llm-ft/bert_baseline.py --task tc --mode budget --model klue/roberta-base --limit 300 \
    --save task5-llm-ft/results/bert/roberta-base-tc-budget.json
# full: 공식 학습셋 전체 (BERT 방식의 최고 성능, 1 epoch)
python task5-llm-ft/bert_baseline.py --task tc --mode full --model klue/roberta-base --limit 300 \
    --save task5-llm-ft/results/bert/roberta-base-tc-full.json
```

태스크는 `tc` · `ner` · `mrc` · `sts` 네 개입니다. SQL생성·수학추론은 위에서 설명한 이유로 head를 붙일 수 없습니다.

### 보너스 B. T5(인코더-디코더) 기준선

1일차 T5 파트와 2일차 실습4B에서 쓴 pko-T5를 실습5의 여섯 태스크로 넓힌 것입니다.
T5는 **답을 글자로 생성**하므로 LLM과 **똑같은 채점 함수**를 씁니다. 그래서 SQL생성·수학추론도 점수가 나옵니다.
평가셋도 LLM·BERT와 바이트 단위로 같은 파일입니다.

```bash
# budget: LLM이 실제로 학습한 그 예제들로 학습 (train_main.jsonl 에서 해당 태스크만 꺼냄, 3 epoch)
python task5-llm-ft/t5_baseline.py --task tc --mode budget \
    --save task5-llm-ft/results/t5/pko-t5-base-tc-budget.json

# full: 공식 train 전체로 학습 (1 epoch, 상한 없음 — BERT full 과 같은 조건. --max-train 으로 줄일 수 있다)
python task5-llm-ft/t5_baseline.py --task tc --mode full \
    --save task5-llm-ft/results/t5/pko-t5-base-tc-full.json
```

- `--mode` 의 뜻은 `bert_baseline.py` 와 같습니다.
  **budget** = "같은 데이터를 줬을 때 누가 더 잘 배우나", **full** = "제 실력을 다 냈을 때 얼마나 차이 나나".
- 기본 모델은 `paust/pko-t5-base`(276M)입니다. `--model paust/pko-t5-large` 로 키울 수 있습니다.
- T5는 bf16에서 불안정한 사례가 있어 fp32로 학습합니다. 그만큼 같은 크기의 BERT보다 메모리를 더 씁니다.
- 파일 이름은 `<모델>-<태스크>-<모드>.json` 규칙을 지켜 주세요. `compare.load_t5()` 가 `results/t5/*.json` 을
  통째로 읽어 표로 만듭니다.
- 여섯 태스크 × 두 모드를 한 번에 돌리려면:

  ```bash
  for task in tc ner mrc sts math sql; do
    for mode in budget full; do
      python task5-llm-ft/t5_baseline.py --task $task --mode $mode \
        --save task5-llm-ft/results/t5/pko-t5-base-$task-$mode.json
    done
  done
  ```

## 평가 지표

각 데이터셋의 공식 지표를 씁니다. BERT·T5·GPT 계열 모두 **같은 평가셋·같은 지표·같은 채점 코드**(`common.score_task`)입니다.
채점이 한 곳에 모여 있으므로 "어느 쪽에 유리한 채점"이 끼어들 여지가 없습니다.

| 태스크 | 지표 | 함께 보는 값 |
|---|---|---|
| 주제분류 | 정확도 | macro F1, 형식 깨짐 건수 |
| 개체명인식 | 개체 단위 F1 | 정밀도 / 재현율, 형식 깨짐 건수 |
| 기계독해 | EM(완전일치) | 글자 단위 F1 |
| 문장유사도 | Pearson 상관 | 3점 기준 이진 정확도, 형식 깨짐 건수 |
| 수학추론 | EM(최종 답 일치) | `#### 답` 형식을 지키지 못한 건수 |
| SQL생성 | EM(정규화 후 완전일치) | 토큰 F1, `SELECT` 로 시작하지 않은 건수 |

## 결과

아래 표는 모두 `results/` 안의 JSON에서 계산한 값입니다. 직접 다시 만들려면:

```bash
python -c "import sys; sys.path.insert(0,'task5-llm-ft'); import compare; \
print(compare.pivot_scores(compare.load_bert('task5-llm-ft/results').query(\"mode=='budget'\"), 'score'))"
```

<!-- RESULTS:BEGIN — 생성기가 다시 씀. 손으로 고치지 말 것 -->

아래 표는 모두 `results/` 안의 JSON 에서 자동 계산한 값입니다 (갱신 2026-09-04 02:02). 전체 수치 정본은 `docs/LECTURE-FACTS.md` 입니다.

### BERT(인코더) 기준선 — 5종 × budget / full (태스크당 평가 300건)

**budget** (LLM과 같은 예제 수: 주제분류 800 · 개체명인식 1,000 · 기계독해 800 · 문장유사도 600, 3 epoch)

| 모델 | 파라미터 | 주제분류 | 개체명인식 | 기계독해 | 문장유사도 | 평균(4) | 학습 합(분) |
|---|---:|---:|---:|---:|---:|---:|---:|
| KLUE-RoBERTa-small | 68M | 79.0 | 74.0 | 59.3 | 76.1 | 72.1 | 0.4 |
| KLUE-RoBERTa-base | 111M | 79.7 | 79.1 | 75.7 | 70.2 | 76.2 | 0.6 |
| KLUE-BERT-base | 111M | 81.3 | 79.4 | 69.0 | 77.7 | 76.9 | 0.6 |
| KoELECTRA-base-v3 | 113M | 31.7 | 59.8 | 54.7 | 75.8 | 55.5 | 0.6 |
| KLUE-RoBERTa-large | 337M | 82.3 | 84.2 | 81.7 | 85.9 | 83.5 | 2.0 |

**full** (공식 train 전체: 주제분류 45,678 · 개체명인식 20,999 · 기계독해 60,407 · 문장유사도 11,668, 1 epoch)

| 모델 | 파라미터 | 주제분류 | 개체명인식 | 기계독해 | 문장유사도 | 평균(4) | 학습 합(분) |
|---|---:|---:|---:|---:|---:|---:|---:|
| KLUE-RoBERTa-small | 68M | 86.0 | 85.7 | 85.0 | 86.6 | 85.8 | 7.1 |
| KLUE-RoBERTa-base | 111M | 84.7 | 86.6 | 85.0 | 90.2 | 86.6 | 11.7 |
| KLUE-BERT-base | 111M | 87.0 | 86.5 | 83.7 | 86.4 | 85.9 | 11.6 |
| KoELECTRA-base-v3 | 113M | 84.0 | 85.6 | 83.3 | 89.8 | 85.7 | 11.6 |
| KLUE-RoBERTa-large | 337M | 86.0 | 88.1 | 86.7 | 91.5 | 88.1 | 36.2 |

시간은 A6000 48GB 기준, 네 태스크 학습 시간의 합입니다. 추론은 어느 모델이나 300건에 1초 안팎입니다.

### T5(인코더-디코더) 기준선 — `results/t5/`

| 모델 | 파라미터 | 모드 | 주제분류 | 개체명인식 | 기계독해 | 문장유사도 | SQL생성 | 수학추론 | 학습 합(분) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| pko-T5-base | 276M | budget | 80.7 | 75.0 | 73.0 | 61.5 | 0.3 | 1.3 | 17.7 |
| pko-T5-base | 276M | full | 85.3 | 78.0 | 84.0 | 87.7 | 8.3 | 2.0 | 117.8 |

### GPT 계열(디코더) 스윕 — `results/sweep2/` (각 칸은 학습 전 → 학습 후)

| 모델 | 주제분류 | 개체명인식 | 기계독해 | 문장유사도 | SQL생성 | 수학추론 | 평균(6) | 학습(분) | GPU 최대(GB) | 학습 파라미터 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A.X-4.0-Light (7B) | 59.3 → 82.7 | 44.0 → 72.2 | 61.3 → 87.0 | 79.3 → 87.7 | 17.0 → 35.3 | 51.0 → 53.3 | 52.0 → 69.7 | 20 | 17.1 | 0.55% |
| EXAONE-4.0-1.2B | 40.7 → 71.7 | 12.9 → 60.8 | 51.3 → 75.3 | 13.7 → 80.8 | 6.3 → 22.3 | 44.7 → 32.0 | 28.3 → 57.1 | 10 | 4.3 | 1.18% |
| kanana-1.5-2.1B-base | 27.3 → 82.3 | 0.2 → 81.9 | 66.0 → 85.0 | 12.0 → 86.7 | 6.7 → 39.3 | 1.0 → 36.3 | 18.9 → 68.6 | 15 | 6.1 | 1.09% |
| kanana-1.5-2.1B-instruct | 47.0 → 80.3 | 27.5 → 68.6 | 44.7 → 80.0 | 63.2 → 84.9 | 12.0 → 30.3 | 53.7 → 41.0 | 41.3 → 64.2 | 24 | 6.1 | 1.09% |
| Bllossom-3B (한국어 추가학습) | 45.3 → 76.7 | 10.4 → 60.9 | 60.7 → 75.0 | 7.7 → 67.8 | 22.0 → 25.7 | 37.7 → 35.3 | 30.6 → 56.9 | 32 | 9.3 | 0.75% |
| Qwen3.5-4B | 61.3 → 81.3 | 42.9 → 81.1 | 74.0 → 86.0 | 83.1 → 90.8 | 21.3 → 45.0 | 61.3 → 59.3 | 57.3 → 73.9 | 41 | 12.7 | 0.77% |
| Qwen3.5-2B | 38.7 → 80.7 | 20.1 → 77.2 | 67.3 → 80.7 | 57.6 → 87.4 | 13.3 → 35.0 | 17.7 → 35.0 | 35.8 → 66.0 | 35 | 7.1 | 0.89% |
| Llama-3.2-3B-Instruct | 38.7 → 75.7 | 0.2 → 62.1 | 66.7 → 75.3 | 0.5 → 63.8 | 19.3 → 27.7 | 34.3 → 38.3 | 26.6 → 57.1 | 16 | 9.3 | 0.75% |
| Gemma-4-E2B-it | 58.3 → 78.0 | 47.5 → 75.3 | 57.0 → 78.7 | 82.0 → 86.6 | 11.3 → 39.0 | 66.0 → 46.0 | 53.7 → 67.3 | 32 | 12.9 | 0.74% |
| Ministral-3-3B-Instruct | 44.3 → 76.0 | 24.7 → 74.4 | 45.0 → 82.7 | 61.0 → 80.8 | 10.7 → 35.3 | 55.7 → 42.3 | 40.2 → 65.3 | 22 | 10.5 | 0.64% |
| gpt-oss-20B (MoE) | 46.0 → 75.3 | 37.6 → 80.9 | 51.7 → 81.7 | 81.8 → 88.0 | 22.7 → 46.0 | 70.7 → 62.0 | 51.7 → 72.3 | 48 | 42.9 | 0.14% |

### 세 방식 한 표

| 태스크 | BERT KLUE-RoBERTa-base budget | BERT KLUE-RoBERTa-base full | BERT 최고(full) | T5 budget | T5 full | LLM A.X-4.0-Light (7B) 전→후 | LLM 최고(후) |
|---|---:|---:|---|---:|---:|---:|---|
| 주제분류 (정확도) | 79.7 | 84.7 | 87.0 (KLUE-BERT-base) | 80.7 | 85.3 | 59.3 → 82.7 | 82.7 (A.X-4.0-Light (7B)) |
| 개체명인식 (F1) | 79.1 | 86.6 | 88.1 (KLUE-RoBERTa-large) | 75.0 | 78.0 | 44.0 → 72.2 | 81.9 (kanana-1.5-2.1B-base) |
| 기계독해 (EM) | 75.7 | 85.0 | 86.7 (KLUE-RoBERTa-large) | 73.0 | 84.0 | 61.3 → 87.0 | 87.0 (A.X-4.0-Light (7B)) |
| 문장유사도 (피어슨 상관) | 70.2 | 90.2 | 91.5 (KLUE-RoBERTa-large) | 61.5 | 87.7 | 79.3 → 87.7 | 90.8 (Qwen3.5-4B) |
| SQL생성 (EM) | 불가 | 불가 | 불가 | 0.3 | 8.3 | 17.0 → 35.3 | 46.0 (gpt-oss-20B (MoE)) |
| 수학추론 (EM) | 불가 | 불가 | 불가 | 1.3 | 2.0 | 51.0 → 53.3 | 62.0 (gpt-oss-20B (MoE)) |

### 에폭 · 4bit · 대화 형식 실험 (EXAONE-4.0-1.2B, 학습 후 점수)

| 설정 | 주제분류 | 개체명인식 | 기계독해 | 문장유사도 | SQL생성 | 수학추론 | 평균(6) | 학습(분) | GPU 최대(GB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EXAONE-4.0-1.2B | 71.7 | 60.8 | 75.3 | 80.8 | 22.3 | 32.0 | 57.1 | 10 | 4.3 |
| EXAONE-4.0-1.2B (2에폭) | 76.0 | 69.2 | 76.3 | 83.8 | 27.7 | 37.7 | 61.8 | 19 | 4.3 |
| EXAONE-4.0-1.2B (3에폭) | 77.0 | 69.8 | 74.3 | 85.6 | 27.0 | 34.7 | 61.4 | 29 | 4.3 |
| EXAONE-4.0-1.2B (QLoRA 4bit) | 69.0 | 59.9 | 74.3 | 79.0 | 23.3 | 29.7 | 55.9 | 21 | 2.5 |
| EXAONE-4.0-1.2B (단순 형식) | 72.0 | 73.6 | 75.7 | 84.8 | 28.7 | 31.3 | 61.0 | 13 | 4.3 |

<!-- RESULTS:END -->

## 강사용 스크립트

아래는 강의 준비에 쓰는 도구다. **수강생용 공개 저장소에는 포함하지 않는다**(비공개 정본에만 있다).

| 파일 | 용도 |
|---|---|
| `sweep.sh` | 여러 모델을 같은 데이터·같은 평가로 돌려 결과 폴더에 저장 |
| `tokenizer_stats.py` | 세 방식 토크나이저의 한국어 토큰 수 비교 |
| `pick_demos.py` | 학습으로 실제로 좋아진 예제를 골라 강의 대본까지 생성 |
| `demo_check.py` | 손으로 적은 데모 질문을 학습 전/후 모델에 돌려 확인 |

수강생도 쓰는 것은 따로다 — `bert_baseline.py`(BERT 기준선, 위 "보너스 A")와
`t5_baseline.py`(T5 기준선, "보너스 B"), 결과를 표로 읽는 `compare.py`,
그리고 저장소 루트의 `setup_classroom.sh`(환경 준비)는 공개 저장소에도 들어 있다.

## 왜 이런 선택을 했는가

- **왜 지시문 형식인가**: 여섯 태스크의 입출력을 같은 모양으로 만들면, 모델 구조를 바꾸지 않고
  하나의 모델로 모두 처리할 수 있습니다. 이것이 BERT 시대와 LLM 시대의 가장 큰 차이입니다.
- **왜 세 방식을 다 돌리는가**: "LLM이 좋다/나쁘다"는 결론을 주려는 것이 아닙니다. 같은 데이터·같은
  채점으로 놓으면, 데이터 양·지연 시간·다룰 수 있는 태스크 범위에 따라 답이 달라진다는 것이 보입니다.
- **왜 LoRA인가**: 12억~200억 파라미터를 전부 학습시키려면 수십 GB가 필요하지만,
  LoRA는 전체의 0.5~1.4%(실측 1,100만~5,100만개)만 학습해서 작은 GPU에서도 돌아갑니다.
- **왜 NER에서 offset을 뺐는가**: 작은 모델이 문자 오프셋까지 맞추려면 출력이 쉽게 깨집니다.
  이 실습의 목표는 오프셋 계산이 아니라 "개체명을 찾아낸다"는 것입니다.
- **왜 학습셋이 작은가**: 강의 시간 안에 **최소 1 epoch를 끝내는 것**이 우선입니다.
  더 좋은 성능을 원하면 `--n-*` 옵션으로 늘려서 직접 비교해보세요.
- **왜 종료 토큰을 따로 점검하는가**: 일부 모델은 chat template이 붙이는 턴 종료 토큰이 학습되지
  않았거나 이름이 달라서, 그대로 두면 답을 맞히고도 멈추지 못합니다.
  `common.check_end_token` / `common.stop_token_ids` 가 이를 자동으로 처리합니다
  (자세한 사연은 `day2/06_LLM파인튜닝.ipynb` §10).

## 참고

- TRL SFTTrainer: https://huggingface.co/docs/trl/sft_trainer
- PEFT LoRA: https://huggingface.co/docs/peft/conceptual_guides/lora
- LoRA 논문: Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022
- QLoRA 논문: Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, NeurIPS 2023
- KLUE: Park et al., *KLUE: Korean Language Understanding Evaluation*, NeurIPS 2021 Datasets
- pko-T5: https://github.com/paust-team/pko-t5
- GSM8K: Cobbe et al., *Training Verifiers to Solve Math Word Problems*, 2021
- Spider: Yu et al., *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain
  Semantic Parsing and Text-to-SQL Task*, EMNLP 2018

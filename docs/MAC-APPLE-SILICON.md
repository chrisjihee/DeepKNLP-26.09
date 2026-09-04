# 맥(Apple Silicon)에서 이 실습을 돌릴 수 있나?


> **이 문서의 실습5 수치는 옛 기본 모델 EXAONE-4.0-1.2B 기준입니다.** 수업 기본 모델은
> `skt/A.X-4.0-Light`(7.3B)로 바뀌었고, 7B를 맥에서 학습하는 것은 별개 문제라 다시 재지 않았습니다.
> 새 모델의 GPU 서버·강의장 실측은 `docs/LECTURE-FACTS.md` 를 보세요.

> 2026-09-02 실측. MacBook Pro **M2 Max, 통합 메모리 64 GB**, macOS 26.6.2, torch 2.13.0(MPS) ·
> transformers 5.16.1 · trl 1.12.0 · peft 0.20.0 — 강의장 PC·GPU 서버와 같은 `pyproject.toml`로 설치.
> 비교 대상은 GPU 서버 **RTX A6000 48 GB**와 강의장 PC **RTX 4500 Ada 24 GB**.
> 아래 숫자는 모두 `task5-llm-ft/results/mac/`의 JSON에 있습니다.

## 0. 한 줄 답

**됩니다.** 추론·LoRA 학습·QLoRA(4비트)·BERT 학습 모두 수업 명령 그대로 맥의 GPU(MPS)에서 돌아갑니다 (코드가 장치를 알아서 고릅니다).
같은 어댑터를 맥과 A6000에서 평가하면 **점수는 같게** 나옵니다. 다만 **속도는 A6000의 1/5 ~ 1/10**이라,
실습 5의 전체 학습(300스텝)은 GPU에서 10분, 맥(M2 Max)에서는 약 **95분**이 걸립니다.
수업을 따라가는 용도(작은 스텝 수·작은 평가 건수)라면 충분하고, 전체 학습은 밤에 걸어두는 편이 맞습니다.

## 1. 실측 요약

같은 코드·같은 데이터·같은 설정(EXAONE-4.0-1.2B, LoRA r=16, batch 4 × accum 4, bf16 가중치)입니다.

| 항목 | MacBook Pro M2 Max 64 GB | RTX A6000 48 GB | RTX 4500 Ada 24 GB | 맥 / A6000 |
|---|---|---|---|---|
| LoRA 학습 10스텝 (`--max-steps 10`) | **189.2 s** (18.9 s/스텝) | **19.2 s** (1.9 s/스텝) | — | **9.9×** 느림 |
| 위 10스텝 뒤 손실 | 2.3047 | 2.3085 | — | 같음 |
| 전체 학습 300스텝 (1 epoch, 4,800건) | ≈ 95분 *(10스텝에서 외삽)* | 9.9분 (실측) | 9.7분 (실측) | ≈ 9.6× |
| 학습 중 메모리 | 9.06 GB *(현재 할당, 아래 주석)* | 4.22 GB (최대) | 4.3 GB (최대) | — |
| 학습 후 평가 8태스크 × 30건 (같은 A6000 어댑터) | **225.4 s** | **41.6 s** | — | **5.4×** 느림 |
| BERT 주제분류 budget (roberta-base, 800건 × 3 epoch) | 28.4 s (fp32) | 3.8 s (bf16) | — | 7.5× 느림 |

**점수는 같은가.** A6000에서 학습한 어댑터를 맥에서 그대로 읽어 같은 30건씩 평가했습니다.

| 태스크 (30건) | 맥 M2 Max | RTX A6000 | 비고 |
|---|---|---|---|
| 주제분류 정확도 | 76.67 | 76.67 | 같음 |
| 개체명 F1 | 63.57 | 62.50 | 생성 1~2토큰 차이 |
| 기계독해 EM / F1 | 73.33 / 86.81 | 73.33 / 86.69 | 같음 |
| 문장유사도 피어슨 | 91.27 | 91.44 | 같음 |
| 수학 EM | 16.67 | 26.67 | 긴 생성(수백 토큰)에서 bf16 누적 오차가 다르게 쌓임 — 30건이라 ±3건 |
| SQL EM / 토큰 F1 | 23.33 / 70.19 | 23.33 / 71.57 | 같음 |
| 자연어추론 · 감성분류 (이번 과정에서 뺀 태스크, 참고) | 56.67 · 73.33 | 56.67 · 73.33 | 같음 |

짧은 답(분류·추출·유사도)은 두 기계가 같은 답을 내고, 긴 생성(수학 풀이)은 몇 건이 갈립니다.
이는 장치 간 bf16 연산 순서 차이로 생기는 정상적인 편차이고, 300건 평가로 늘리면 평균은 수렴합니다.
BERT 기준선(roberta-base 주제분류 budget)은 맥 82.67 / A6000 79.67 — 맥은 fp32, GPU는 bf16으로 학습해 시드가 같아도
수치가 조금 다릅니다.

**어느 태스크가 특히 느린가.** 평가는 생성 길이에 비례해 느려집니다 (같은 30건, 맥 / A6000 초):
주제분류 2.5 / 0.3 · 개체명 21.0 / 5.6 · 기계독해 11.7 / 1.2 · 문장유사도 4.3 / 0.4 · **수학 126.3 / 26.2** · SQL 52.0 / 6.9.
수업에서 맥으로 따라갈 때는 `--limit 30` 정도가 적당합니다 (8태스크 4분).

> 메모리 주석: MPS에는 "최대 할당량" API가 없어 `train_meta.json`의 `peak_gpu_gb`는 **학습 직후 현재 할당량**입니다
> (`torch.mps.driver_allocated_memory()`). 통합 메모리라 활동 모니터의 "메모리 압력"으로 보는 편이 정확합니다.
> 64 GB 맥에서는 1.2B 모델 LoRA 학습 중 시스템 전체가 넉넉했습니다.

## 2. 설치

강의장 스크립트(`setup_classroom.sh`)는 CUDA용이라 맥에서는 수동으로 세 줄이면 됩니다.
**PyTorch를 따로 받지 않습니다** — PyPI의 macOS arm64 휠에 MPS가 들어 있습니다.

```bash
git clone https://github.com/chrisjihee/DeepKNLP-26.09.git && cd DeepKNLP-26.09
curl -LsSf https://astral.sh/uv/install.sh | sh                 # uv가 없다면
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -e .                                            # torch 2.13.0 macOS 휠이 함께 설치됨 (약 3분)
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"   # 2.13.0 True
```

모델은 `hf download LGAI-EXAONE/EXAONE-4.0-1.2B` (실습4B·기준선용 `paust/pko-t5-base`, `klue/roberta-base`도 같은 방식)으로 미리 받아두면 수업 중 기다리지 않습니다.

## 3. 실습 5 (LLM 파인튜닝) — 명령 그대로

코드가 장치를 `cuda → mps → cpu` 순서로 스스로 고르므로(`task5-llm-ft/common.py`의 `pick_device()`) 수업 명령을 그대로 칩니다.

```bash
# 학습 전 평가 — 맥에서는 건수를 줄여서 (8태스크 × 30건 ≈ 4분; 300건이면 ≈ 40분)
python task5-llm-ft/evaluate.py --tasks tc,ner,mrc,sts,sql,math --limit 30 --save output/before.json

# 학습 — 수업 중에는 스텝 수를 잘라서 (10스텝 ≈ 3분, 모델 로딩 별도). 전체 300스텝은 ≈ 95분
python task5-llm-ft/train.py --data data/llm-ft/train_main.jsonl --max-steps 10
python task5-llm-ft/train.py --data data/llm-ft/train_main.jsonl               # 전체 1 epoch (밤에)

# 학습 후 평가 · 데모 페이지 — 그대로
python task5-llm-ft/evaluate.py --adapter output/llm-ft --tasks tc,ner,mrc,sts,sql,math --limit 30 --save output/after.json
python task5-llm-ft/serve.py --adapter output/llm-ft --port 9005
```

- `--max-steps 10` 으로 학습한 어댑터는 "돌아간다"를 확인하는 용도입니다. 손실이 거의 안 내려간 상태라
  점수 개선을 보려면 **GPU에서 학습한 어댑터**(`task5-llm-ft/results/sweep2/`의 설정과 같은 `output/sweep/exaone4-1.2b`)를 받아
  `--adapter`로 넘겨 맥에서 평가·서빙만 해도 됩니다 — 위 표가 정확히 그 방식입니다.
- `--load-4bit`(QLoRA)도 **돕니다.** bitsandbytes 0.50부터 MPS 백엔드가 들어 있어 `Linear4bit`(NF4, uint8) 가중치가 실제로 맥 GPU에 올라갑니다
  (실측: 모델 적재 5.8초, 가중치 0.89 GB, 학습 1스텝 17.8초·메모리 4.27 GB — bf16의 9.06 GB보다 절반 이하). 속도 이득은 없으므로
  64 GB 맥에서는 굳이 필요 없고, 16 GB 맥에서 메모리를 아끼는 용도입니다. 1스텝만 확인했고 점수는 재보지 않았습니다.
  bitsandbytes 0.50 릴리스 노트에 따르면 macOS 26 이상에서 `kernels` 패키지를 깔면 Metal 커널을 써서 빨라지고(기본은 느린 대체 구현),
  8비트 옵티마이저(`paged_adamw_8bit`)는 아직 MPS에 없습니다 — 둘 다 이번에 확인하지 않았습니다.
- `bf16` 혼합정밀(AMP)은 CUDA에서만 켭니다. 맥에서는 모델 가중치가 이미 bf16이라 그대로 계산하고, LoRA 가중치는 PEFT가 fp32로
  유지합니다. 이 차이가 위 표의 "손실 같음"으로 확인됐습니다. (transformers는 2025-08 이후 macOS 14+의 MPS에서도 `bf16=True`를
  받아주지만, M1·M2에서는 bf16이 fp32로 흉내내어 계산되어 이득이 없어 끈 채로 둡니다.)
- 어떤 연산이 MPS에 없다는 오류(`NotImplementedError: ... not currently implemented for the MPS device`)가 나면
  `PYTORCH_ENABLE_MPS_FALLBACK=1`을 앞에 붙여 그 연산만 CPU로 보냅니다. 이번 실측에서는 필요하지 않았습니다.
- BERT·T5 기준선(`bert_baseline.py`, `t5_baseline.py`)도 그대로 돕니다. HF Trainer가 MPS를 자동 선택합니다.

## 4. 1일차 노트북과 2일차 기계독해 실습

| 실습 | 맥에서 그대로? | 바꿀 것 |
|---|---|---|
| 1일차 노트북 세 개 `day1/*.ipynb` | ✓ 그대로 (Trainer가 MPS를 스스로 고른다) | 없음. 다만 아래 "재보지 않았다"를 참고 |
| 2일차 실습4A `task4A-qa-ext/train_qa*.sh` | ✓ (`CUDA_VISIBLE_DEVICES`는 무시되고 Trainer가 MPS 선택) | 데이터 축소 권장: `--max_train_samples 2000 --max_eval_samples 500 --max_seq_length 256` |
| 2일차 실습4B `task4B-qa-gen/train_qa_seq2seq-1.sh` | ✓ (pko-t5-base) | 같은 축소 권장; `-2.sh`(pko-t5-large)는 메모리·시간상 비권장 |

```bash
# 1일차 노트북 — 맥에서도 명령은 같다
.venv/bin/jupyter lab        # day1/01_주제분류.ipynb 부터

# 같은 학습을 터미널에서 확인하려면 (태스크는 tc / sts / ner)
python task5-llm-ft/bert_baseline.py --task tc --mode budget --limit 300
python task5-llm-ft/t5_baseline.py   --task tc --mode budget --limit 300
```

> **아직 재보지 않은 것.** 아래 실측표(§2·§3)는 실습5(LLM 파인튜닝)와 2026년 3월 과정의 실습 코드로 잰 값입니다.
> **1일차 노트북 세 개는 맥에서 시간을 재보지 않았습니다.** 모델이 작아(110M·276M) 강의장 GPU에서 노트북당
> 1~2분이니 맥에서도 십여 분 안에 끝날 것으로 보이지만, 확인된 값이 아닙니다.
> 3월 과정 실습 코드(`task1-cls`·`task2-ner`, `--accelerator mps --cpu-workers 0` 필요)는 이 저장소에 없고
> https://github.com/chrisjihee/DeepKNLP-26.03 에 있습니다.

- **`--accelerator cpu`는 쓰지 마세요.** 이 코드에서 `cpu`는 `--cpu-workers`개의 프로세스로 **CPU DDP**를 띄우는 경로라
  노트북에서 메모리·속도 모두 불리합니다. 맥의 정답은 `mps`입니다.
- Lightning 실습은 비-CUDA 장치에서 precision 옵션을 버리고 **fp32**로 돕니다 (`--precision 16-mixed`를 줘도 조용히 무시).
  틀린 것은 아니고 느릴 뿐입니다.
- 실습4A/4B의 `--bf16`은 macOS 14 이상에서 MPS가 받아줍니다. M1·M2에서는 bf16이 fp32로 흉내내어 계산되어 속도 이득이 없고,
  M3 이후는 네이티브입니다. 이상하게 느리거나 손실이 튀면 `--bf16`을 빼고 fp32로 돌리는 것이 유일한 변경입니다.

## 5. 왜 이만큼 느린가 (수강생 설명용)

- **연산 장치 자체의 차이.** 공개 사양으로 A6000은 bf16 텐서코어 약 155 TFLOPS(dense), M2 Max GPU는 fp32 약 13.6 TFLOPS입니다.
  학습(행렬곱이 대부분)에서 10배 차이가 그대로 드러납니다. 추론은 메모리 대역폭에 더 좌우돼(400 GB/s vs 768 GB/s) 차이가 5배로 줄어듭니다.
- **소프트웨어 성숙도.** MPS 백엔드는 CUDA보다 연산 커널 최적화가 덜 되어 있습니다(특히 attention). flash-attention 같은 CUDA 전용
  라이브러리는 쓸 수 없고, bitsandbytes는 0.50부터 MPS를 지원하지만 CUDA만큼 다듬어지진 않았습니다.
- **대신 메모리는 넉넉합니다.** 64 GB 통합 메모리는 24 GB GPU보다 큰 모델을 bf16 그대로 올릴 수 있습니다
  (이번에 1.2B만 실측했고, 더 큰 모델은 시간을 재보지 않았습니다). PyTorch가 MPS에 허용하는 상한은
  `torch.mps.recommended_max_memory()`로 보이며 보통 전체 메모리의 75% 정도(64 GB 맥에서 약 48 GB)입니다.
  단 `device_map="auto"`의 CPU 오프로드는 MPS에서 되지 않으므로 모델이 그 안에 다 들어가야 합니다.
- **비슷한 측정이 논문에도 있습니다.** Feng et al. (2025, arXiv:2501.14925)은 GPT2-large(774M) LoRA fp32 학습에서 M2 Max가
  A6000보다 약 12~14배 느리다고 보고했고, 같은 맥에서 MLX가 PyTorch-MPS보다 20~25% 빨랐습니다. 이번 9.9배와 같은 범위입니다.
- 맥에서 LLM을 **빠르게** 돌리려는 목적이면 PyTorch/MPS보다 Apple의 **MLX**(`mlx-lm`, LoRA·QLoRA 내장)나 llama.cpp(GGUF)가 더 빠릅니다
  (위 논문 기준 20~25%, 공개 블로그들은 4비트 1.5B LoRA에 M1 Max 64 GB에서 약 1,100 tok/s를 보고 — 이번에 재보지는 않았습니다).
  다만 이 실습은 "Transformers·PEFT 코드가 어떻게 생겼는지"를 배우는 것이 목적이라 같은 코드가 그대로 도는 MPS 경로로 안내합니다.

## 6. 한계 · 검증하지 않은 것

- 전체 300스텝 학습(≈95분)은 맥에서 **끝까지 돌리지 않았습니다** — 10스텝의 s/스텝으로 외삽한 값입니다.
- `--load-4bit`(QLoRA)는 1스텝만 돌려 "올라가고 돈다"까지 확인했습니다. 여러 스텝의 속도와 학습 후 점수는 재보지 않았습니다 (§3).
- gpt-oss-20B(수업 밖 참고 모델)는 맥에서 시도하지 않았습니다. A6000에서 bf16으로 42.9 GB를 썼으므로 64 GB 맥에 "올라갈 수는" 있겠지만
  시간은 재보지 않았습니다.
- 8/16 GB 맥은 측정하지 않았습니다. 1.2B 모델 LoRA 학습이 9 GB 이상을 쓰므로 16 GB에서는 `--load-4bit`(4.3 GB) 또는 `--batch-size 1 --grad-accum 16`이 필요할 것으로 보이고,
  8 GB는 권하지 않습니다.
- MPS attention에는 아직 열린 정확도 이슈가 있습니다(transformers #44247 반정밀 양방향 attention, pytorch #163997 등; transformers CI는 MPS에서 돌지 않음).
  이번 결과에서 짧은 답은 모두 일치했지만, 긴 생성이 갈리는 데 이런 요인이 섞였을 가능성을 배제하지 못합니다.
- `evaluate.py` 학습 전 평가의 개체명 F1이 맥에서 0.0으로 나온 것은 MPS 문제가 아니라 학습 전 모델이 빈 리스트 `[]`를 내기 때문입니다
  (A6000에서 같은 8건을 돌려 확인).

## 7. 재현

```bash
# 맥 (이 문서의 숫자)
python task5-llm-ft/evaluate.py --limit 30 --save results/mac/exaone4-1.2b-before-30-m2max.json
python task5-llm-ft/train.py --data data/llm-ft/train_main.jsonl --max-steps 10 --out output/mac/exaone-lora-10steps
python task5-llm-ft/evaluate.py --adapter <A6000에서 학습한 어댑터> --limit 30 --save results/mac/exaone4-1.2b-after-30-m2max.json
python task5-llm-ft/bert_baseline.py --task tc --mode budget --model klue/roberta-base --limit 300 --save results/mac/roberta-base-tc-budget-m2max.json
# A6000 (같은 명령, 같은 어댑터) → results/mac/*-a6000.json
```

| 파일 (`task5-llm-ft/results/mac/`) | 내용 |
|---|---|
| `exaone4-1.2b-before-30-m2max.json` | 학습 전 8태스크 × 30건, 맥 |
| `exaone4-1.2b-after-30-m2max.json` / `-a6000.json` | 같은 A6000 어댑터를 맥 / A6000에서 평가 |
| `exaone4-1.2b-train_meta-10steps-m2max.json` / `-a6000.json` | 10스텝 LoRA 학습 메타 (시간·메모리·손실) |
| `roberta-base-tc-budget-m2max.json` | BERT 주제분류 budget, 맥 (GPU 값은 `results/bert/roberta-base-tc-budget.json`) |

코드 쪽 변경은 `task5-llm-ft/common.py`의 장치 도우미 세 개(`pick_device`·`device_name`·`peak_memory_gb`)와
`train.py --max-steps`, 그리고 하드코딩된 `"cuda"`를 도우미 호출로 바꾼 것이 전부입니다. CUDA 기계의 동작은 바뀌지 않았습니다.

## 8. 출처

- PyTorch 2.6 MPS bf16 autocast: https://github.com/pytorch/pytorch/pull/139390 · transformers MPS bf16 허용(PR #40458): https://github.com/huggingface/transformers/pull/40458
- HF 특수 하드웨어 학습 문서(MPS, `PYTORCH_ENABLE_MPS_FALLBACK`, `device_map="auto"` 제약): https://huggingface.co/docs/transformers/main/en/perf_train_special
- bitsandbytes 0.50.0 "MPS: improved backend": https://github.com/bitsandbytes-foundation/bitsandbytes/releases/tag/0.50.0
- Feng et al., *Profiling Apple Silicon Performance for ML Training* (2025): https://arxiv.org/abs/2501.14925
- mlx-lm LoRA 문서: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md
- 열린 이슈: https://github.com/huggingface/transformers/issues/44247 · https://github.com/pytorch/pytorch/issues/163997

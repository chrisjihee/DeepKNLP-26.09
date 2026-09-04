#!/usr/bin/env bash
# 강의장 PC 준비 스크립트 — AI아카데미 A4021 (2026-09-03/04) 실습5 (task5-llm-ft)
#
#   저장소 루트에서:  bash setup_classroom.sh            # 전체 (환경 → 모델·데이터 내려받기 → 연기 테스트)
#                    bash setup_classroom.sh env        # 파이썬 환경만
#                    bash setup_classroom.sh models-day1 # 1일차 모델만 (klue/roberta-base·pko-t5-base, 1GB 미만)
#                    bash setup_classroom.sh models     # 2일차 수업 모델까지 (약 16GB)
#                    bash setup_classroom.sh models-alt # 대안 모델 2종 (약 8GB, 선택)
#                    bash setup_classroom.sh smoke      # GPU 연기 테스트 (학습 20스텝 + 평가 5건)
#                    bash setup_classroom.sh check      # 설치 상태만 출력
#
# 전제: Ubuntu + NVIDIA 드라이버(nvidia-smi 동작), 인터넷 연결, git clone 완료.
# 소요 시간(대략): env 5~10분(토치 다운로드), models 7~15분(약 16GB), smoke 3~5분.
set -u
cd "$(dirname "$0")"
UV="$HOME/.local/bin/uv"
PY=".venv/bin/python"
STEP=${1:-all}

# 수업에서 실제로 쓰는 모델만 미리 받는다 (강사 스윕용 나머지 모델은 여기 넣지 않는다).
# 세 방식을 한 자리에서 비교하므로 디코더(GPT 계열) 3종 + 인코더-디코더(T5) 1종 + 인코더(BERT) 1종을 받는다.
# 1일차(BERT·T5 실습)에 꼭 필요한 것 — 합쳐서 1GB가 채 안 된다. 수강생 PC는 이것부터 받는다.
MODELS_DAY1=(
  "klue/roberta-base"                           # 인코더 · 1일차 BERT 실습 3개
  "paust/pko-t5-base"                           # 인코더-디코더 · 1일차 T5 파트 (실습4B와 같은 모델)
)
# 2일차(기계독해 + LLM 파인튜닝)에 필요한 것 — 큰 파일이라 1일차 저녁이나 2일차 아침에 받는다.
# 2일차 기계독해 노트북(day2/)은 1일차와 같은 klue/roberta-base·pko-t5-base 를 쓴다 — 새로 받을 것이 없다.
# jinmang2/kpfbert 는 옛 셸 스크립트(task4A-qa-ext/train_qa-1.sh)에서만 쓰므로 목록에 넣지 않는다.
MODELS_DAY2=(
  "skt/A.X-4.0-Light"                           # 디코더 · 실습5 수업 기본 (약 15GB)
)
# 대안 모델 — 수업에 꼭 필요하지는 않다. 강의장 회선이 느릴 때 굳이 받지 않는다.
#   bash setup_classroom.sh models-alt  로 따로 받는다.
MODELS_DAY2_ALT=(
  "kakaocorp/kanana-2-3b-instruct"              # 시간이 모자랄 때 대안 (약 6GB)
  "LGAI-EXAONE/EXAONE-4.0-1.2B"                 # 가장 가벼운 대안 (약 2.4GB)
)
MODELS=("${MODELS_DAY1[@]}" "${MODELS_DAY2[@]}")

# 2026-09 개편 태스크 구성: 세 방식(BERT·T5·GPT 계열) 모두 가능한 4개 + 생성 모델만 가능한 2개
TASKS="tc,ner,mrc,sts,sql,math"

log() { echo; echo "==== [$(date +%H:%M:%S)] $*"; }

check() {
  log "GPU"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "  nvidia-smi 없음 — 드라이버를 확인하세요"
  log "파이썬 환경"
  if [ -x "$PY" ]; then
    $PY - <<'PYEOF'
import importlib.metadata as m, torch
print(f"  python {__import__('sys').version.split()[0]} | torch {torch.__version__} | cuda {torch.cuda.is_available()}")
for p in ["transformers", "trl", "peft", "bitsandbytes", "datasets", "accelerate", "flask", "pandas", "matplotlib", "jupyterlab"]:
    try: print(f"  {p:14s} {m.version(p)}")
    except Exception: print(f"  {p:14s} (없음)")
PYEOF
  else
    echo "  .venv 없음 — 'bash setup_classroom.sh env' 를 먼저 실행"
  fi
  log "데이터 (실습5 학습셋 + 6태스크 평가셋)"
  for f in data/llm-ft/train_main.jsonl \
           data/llm-ft/eval_tc.jsonl data/llm-ft/eval_ner.jsonl data/llm-ft/eval_mrc.jsonl \
           data/llm-ft/eval_sts.jsonl data/llm-ft/eval_math.jsonl data/llm-ft/eval_sql.jsonl; do
    [ -f "$f" ] && echo "  OK  $f" || echo "  없음 $f"
  done
  log "데이터 (1일차 BERT·T5 실습 원본)"
  for f in data/klue-ynat/train.jsonl data/klue-sts/train.jsonl data/klue-ner/train.jsonl; do
    [ -f "$f" ] && echo "  OK  $f" || echo "  없음 $f"
  done
  log "모델 캐시 (~/.cache/huggingface/hub)"
  for m in "${MODELS_DAY1[@]}"; do
    d="$HOME/.cache/huggingface/hub/models--${m//\//--}"
    [ -d "$d" ] && echo "  OK  [1일차] $m" || echo "  없음 [1일차] $m"
  done
  for m in "${MODELS_DAY2[@]}"; do
    d="$HOME/.cache/huggingface/hub/models--${m//\//--}"
    [ -d "$d" ] && echo "  OK  [2일차] $m" || echo "  없음 [2일차] $m"
  done
  for m in "${MODELS_DAY2_ALT[@]}"; do
    d="$HOME/.cache/huggingface/hub/models--${m//\//--}"
    [ -d "$d" ] && echo "  OK  [2일차·대안] $m" || echo "  ―   [2일차·대안] $m (없어도 수업에 지장 없음)"
  done
}

env_setup() {
  log "uv 설치 확인"
  if [ ! -x "$UV" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
  "$UV" --version
  log "Python 3.12 + 가상환경 (.venv)"
  "$UV" python install 3.12
  [ -d .venv ] || "$UV" venv .venv --python 3.12
  log "패키지 설치 1/2 — torch (CUDA 12.8 휠; GPU 서버와 같은 2.11.0)"
  # PyTorch 인덱스에는 requests 등이 옛 버전으로 올라와 있어, 한 번에 섞어 풀면 uv 0.12가 "requests==2.28.1만 있다"며 실패한다.
  # 그래서 torch만 PyTorch 인덱스에서 먼저 받고, 나머지는 PyPI에서 받는다 (이미 설치된 torch는 그대로 둔다).
  "$UV" pip install --python "$PY" "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128 || { echo "  [FAIL] torch 설치"; return 1; }
  log "패키지 설치 2/2 — 나머지 (pyproject.toml, PyPI)"
  "$UV" pip install --python "$PY" -e . || { echo "  [FAIL] 패키지 설치"; return 1; }
  log "주피터 커널 등록 (이름: deepknlp)"
  "$PY" -m ipykernel install --user --name deepknlp --display-name "Python (DeepKNLP)"
  log "설치 확인"
  "$PY" -c "import torch, transformers, trl, peft; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| transformers', transformers.__version__, '| trl', trl.__version__, '| peft', peft.__version__)"
}

models_setup() {
  local -a want
  case "${1:-all}" in
    day1) want=("${MODELS_DAY1[@]}") ;;
    alt)  want=("${MODELS_DAY2_ALT[@]}") ;;
    *)    want=("${MODELS[@]}") ;;
  esac
  log "모델 미리 내려받기 (수업 중 네트워크 대기를 없애기 위해)"
  "$PY" - "${want[@]}" <<'PYEOF'
import sys, time
from pathlib import Path
from huggingface_hub import snapshot_download

PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.py", "*.jinja"]
for m in sys.argv[1:]:
    t0 = time.time()
    p = snapshot_download(m, allow_patterns=PATTERNS)
    # safetensors가 없고 옛 .bin 형식만 올려둔 저장소(예: 일부 T5)는 가중치가 빠진다 → 한 번 더 받는다.
    if not list(Path(p).rglob("*.safetensors")):
        p = snapshot_download(m, allow_patterns=PATTERNS + ["*.bin"])
    print(f"  {m:45s} {time.time()-t0:5.0f}s  {p}")
PYEOF
  log "데이터 확인 (없으면 생성 — KLUE·GSM8K-ko·Spider-ko는 Hugging Face에서 내려받음)"
  if [ ! -f data/llm-ft/train_main.jsonl ] || [ ! -f data/llm-ft/eval_sql.jsonl ]; then
    "$PY" task5-llm-ft/build_dataset.py
  else
    echo "  이미 있음: data/llm-ft/train_main.jsonl"
  fi
}

smoke() {
  mkdir -p output
  log "연기 테스트 1/3 — 렌더링·템플릿 점검 (학습 없음)"
  "$PY" task5-llm-ft/train.py --model skt/A.X-4.0-Light --data data/llm-ft/train_main.jsonl --inspect || { echo "  [FAIL] inspect"; return 1; }
  log "연기 테스트 2/3 — 학습 짧게 (200건, 1 epoch ≈ 12스텝)"
  head -n 200 data/llm-ft/train_main.jsonl > output/smoke-train.jsonl || { echo "  [FAIL] 학습셋 없음 — 'bash setup_classroom.sh models' 를 먼저 실행"; return 1; }
  "$PY" task5-llm-ft/train.py --model skt/A.X-4.0-Light --data output/smoke-train.jsonl --out output/smoke-adapter || { echo "  [FAIL] train"; return 1; }
  log "연기 테스트 3/3 — 평가 5건 × 6태스크($TASKS) (학습 전 / 후)"
  "$PY" task5-llm-ft/evaluate.py --model skt/A.X-4.0-Light --tasks "$TASKS" --limit 5 --batch-size 5 || { echo "  [FAIL] evaluate"; return 1; }
  "$PY" task5-llm-ft/evaluate.py --model skt/A.X-4.0-Light --adapter output/smoke-adapter --tasks "$TASKS" --limit 5 --batch-size 5 --show 1 || { echo "  [FAIL] evaluate(adapter)"; return 1; }
  log "연기 테스트 통과. 실제 수업 학습(train_main.jsonl, 4,800건)은 RTX 4500 Ada에서 9.7분 걸렸습니다 (2026-09-02 강의장 PC 실측, 1.96 s/스텝, GPU 4.3 GB). 평가 6태스크×300건은 약 6분."
  echo "  노트북:  .venv/bin/jupyter lab   →  day2/06_LLM파인튜닝.ipynb"
  echo "  데모:    .venv/bin/python task5-llm-ft/serve.py --model skt/A.X-4.0-Light --adapter output/smoke-adapter --port 9005"
  echo "  기준선:  .venv/bin/python task5-llm-ft/bert_baseline.py --task tc --mode budget --limit 100   (인코더)"
  echo "           .venv/bin/python task5-llm-ft/t5_baseline.py   --task tc --mode budget --limit 100   (인코더-디코더)"
}

case "$STEP" in
  check)  check ;;
  env)    env_setup ;;
  models) models_setup all ;;
  models-day1) models_setup day1 ;;
  models-alt) models_setup alt ;;
  day1)   env_setup && models_setup day1 && check ;;
  smoke)  smoke ;;
  all)    env_setup && models_setup && smoke && check ;;
  *) echo "usage: bash setup_classroom.sh [all|day1|env|models|models-day1|models-alt|smoke|check]"; exit 1 ;;
esac

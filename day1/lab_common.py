"""
[1일차 실습] 노트북 공통 모듈

`day1/01_주제분류.ipynb`·`02_문장유사도.ipynb`·`03_개체명인식.ipynb` 세 노트북이 함께 쓰는 함수를
여기에 모았다. 노트북에는 **학생이 직접 채우는 미션 코드와, 눈으로 볼 값**만 남기고
반복되는 뒷일(학습 루프 설정, 결과 저장, 표 만들기)은 전부 이 파일이 맡는다.

원본은 `task5-llm-ft/bert_baseline.py`·`t5_baseline.py`다. 두 스크립트의 함수를 그대로 옮겨 왔으므로
노트북에서 본 것을 터미널에서도 똑같이 재현할 수 있다:

    python task5-llm-ft/bert_baseline.py --task tc --mode budget
    python task5-llm-ft/t5_baseline.py   --task tc --mode budget

**여기에 없는 것**: 미션 대상 함수(`encode_choice`·`encode_sts`·`encode_ner`, 분류/회귀 head 만들기).
그것들은 학생이 채우는 부분이라 노트북 셀에 남겨 두었다.

평가셋(`data/llm-ft/eval_*.jsonl`, 태스크당 300건)은 절대 학습에 쓰지 않는다.
채점은 BERT·T5·GPT 계열이 **모두 같은 함수**(`common.score_task`)를 쓴다. 그래야 숫자를 나란히 놓을 수 있다.
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          DataCollatorForTokenClassification, DataCollatorWithPadding, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, Trainer, TrainingArguments, default_data_collator)

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
sys.path.insert(0, str(ROOT / "task5-llm-ft"))

from common import (NER_LABELS, TASKS, TC_LABELS, describe, device_name, macro_f1,  # noqa: E402
                    parse_score, peak_memory_gb, pearson, pick_label, read_jsonl,
                    reset_peak_memory, score_mrc, score_ner, score_task)

# --------------------------------------------------------------------------------------
# 설정값 — 세 노트북이 공유한다
# --------------------------------------------------------------------------------------

MODEL_BERT = "klue/roberta-base"          # 인코더. 한국어 RoBERTa (약 110M 파라미터)
MODEL_T5 = "paust/pko-t5-base"            # 인코더-디코더. 한국어 T5 (약 275M 파라미터)
DATA_DIR = Path("data/llm-ft")
OUT_DIR = Path("output/day1")             # 결과 JSON·체크포인트 (git에 올라가지 않는다)
OUT_DIR_DAY2 = Path("output/day2")        # 2일차 노트북(기계독해)의 결과
LLM_RESULT = Path("task5-llm-ft/results/sweep2/ax4-light-7b-after.json")   # 2일차 예고용 참고값
MAX_LEN = 128
MAX_LEN_MRC = 384        # 기계독해는 지문이 들어가 훨씬 길다
# T5 입력 길이 상한. t5_baseline.py 와 같은 1024로 둔다 — 기계독해 지문이 512를 넘는 예제가
# 17.8%(최대 706토큰)나 되어, 512로 자르면 답이 잘려 나간다. 1일차 세 태스크는 최대 222토큰이라
# 이 값이 512든 1024든 결과가 같다.
MAX_SOURCE_T5 = 1024
STRIDE_MRC = 128         # 지문이 384토큰을 넘으면 겹쳐 가며 조각낸다
N_EVAL = 300

# 개체명 BIO 태그 — bert_baseline.py 와 같은 순서여야 한다
BIO_LABELS = ["O"] + [f"{p}-{t}" for t in NER_LABELS for p in ("B", "I")]
BIO2ID = {label: i for i, label in enumerate(BIO_LABELS)}

# 태스크별 대표 지표를 사람 말로
METRIC_NAME = {"accuracy": "정확도(%)", "pearson": "Pearson 상관(×100)", "f1": "개체 F1(%)", "em": "완전일치(%)"}


# --------------------------------------------------------------------------------------
# 0. 환경 — 저장소 루트로 이동하고 장치를 확인한다
# --------------------------------------------------------------------------------------

def go_root() -> Path:
    """노트북이 `day1/`이나 `day1/instructor/`에 있어도 저장소 루트를 작업 폴더로 삼는다."""
    os.chdir(ROOT)
    if str(ROOT / "day1") not in sys.path:
        sys.path.insert(0, str(ROOT / "day1"))
    return ROOT


def set_seed(seed: int = 42) -> None:
    """무작위 초기화(새로 얹는 head의 가중치, 데이터 순서 섞기)를 고정한다.

    이것을 학습 **직전**에 부르면 같은 코드가 같은 점수를 낸다. 옆 사람과 숫자를 비교할 수 있다.
    GPU 종류가 다르면 소수점 아래는 조금 달라질 수 있다.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def env_info() -> None:
    """작업 폴더·라이브러리 버전·GPU를 한눈에 출력한다."""
    import transformers
    # 어느 기계에서 돌려도 계정명·서버 경로가 출력에 남지 않게 폴더 이름만 찍는다
    print("작업 폴더   :", Path.cwd().name)
    print(f"torch {torch.__version__} | transformers {transformers.__version__} | python {sys.version.split()[0]}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"장치        : {torch.cuda.get_device_name(0)}  ({total / 1024**3:.1f} GB, 여유 {free / 1024**3:.1f} GB)")
    else:
        print(f"장치        : {device_name()}  (GPU가 없으면 학습이 매우 느립니다)")


# --------------------------------------------------------------------------------------
# 1. 데이터 — 학습셋(LLM·T5와 같은 것)과 평가셋
# --------------------------------------------------------------------------------------

def load_budget(task: str, data_dir=DATA_DIR) -> list[dict]:
    """LLM이 실제로 학습한 그 예제들 — `train_main.jsonl`에서 해당 태스크만 꺼낸다.

    `bert_baseline.load_budget`·`t5_baseline.load_budget`과 같은 함수이고, 찾는 순서도 같다
    (`train_main.jsonl` → `train_all.jsonl` → `train.jsonl`). 이렇게 해야 BERT·T5·GPT 계열이
    **같은 데이터로 배운 뒤 같은 평가셋에서** 겨루는 공정한 비교가 된다.
    """
    for name in ("train_main.jsonl", "train_all.jsonl", "train.jsonl"):
        src = Path(data_dir) / name
        if src.exists():
            break
    rows = [r for r in read_jsonl(src) if r["task"] == task]
    if not rows:
        raise SystemExit(f"{src}에 {task} 예제가 없습니다. task5-llm-ft/build_dataset.py를 먼저 실행하세요.")
    return rows


def load_full(task: str, limit: int | None = None, raw_dir=Path("data"), data_dir=DATA_DIR) -> list[dict]:
    """공식 train **전체**를 읽는다 — "데이터를 늘리면 얼마나 오르나"를 직접 보는 자리.

    `load_budget` 은 LLM 과 같은 예제 수(800·600·1,000)만 준다. 세 방식을 공정하게 겨루려면
    그래야 하지만, 그만큼 학습이 몇 초 만에 끝나 **학습이 진행되는 것을 볼 수가 없다.**
    이 함수는 같은 데이터셋의 공식 train 전체를 준다(주제분류 45,678 · 개체명 20,999 ·
    기계독해 60,407 · 문장유사도 11,668건).

    **평가셋과 겹치는 예제는 빼고 준다.** 그래야 점수가 부풀지 않는다.
    `bert_baseline.py --mode full` 과 같은 함수라 터미널에서도 같은 값을 재현할 수 있다.

    ⚠️ 이렇게 학습한 모델은 **BERT·T5·GPT 계열 비교 표에 넣지 않는다.** 데이터가 다르면
    비교가 성립하지 않는다. 최종 비교는 언제나 `load_budget` 쪽 값으로 한다.
    """
    sys.path.insert(0, str(ROOT / "task5-llm-ft"))
    from bert_baseline import load_full as _load_full  # noqa: PLC0415
    eval_keys = {r["key"] for r in read_jsonl(Path(data_dir) / f"eval_{task}.jsonl")}
    rows = _load_full(task, Path(raw_dir), eval_keys, limit)
    assert not ({r["key"] for r in rows} & eval_keys), "학습셋과 평가셋이 겹칩니다"
    return rows


def load_eval(task: str, limit: int = N_EVAL, data_dir=DATA_DIR) -> list[dict]:
    """평가셋. **학습에는 절대 쓰지 않는다.**"""
    return read_jsonl(Path(data_dir) / f"eval_{task}.jsonl", limit)


def check_no_leak(train_rows: list[dict], eval_rows: list[dict]) -> None:
    """학습셋과 평가셋이 겹치지 않는지 확인한다(겹치면 점수가 부풀려진다)."""
    overlap = {r["key"] for r in train_rows} & {r["key"] for r in eval_rows}
    assert not overlap, f"학습셋과 평가셋이 {len(overlap)}건 겹칩니다"
    print(f"  학습 {len(train_rows)}건 / 평가 {len(eval_rows)}건 · 겹치는 예제 0건")


def _short(v, n: int = 46) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return s if len(s) <= n else s[: n - 1] + "…"


def preview(rows: list[dict], n: int = 3):
    """예제 몇 건을 표로 보여준다 (입력 필드 + 정답)."""
    import pandas as pd
    recs = []
    for r in rows[:n]:
        rec = {k: _short(v) for k, v in r["input"].items()}
        rec["정답"] = _short(r["gold"], 60)
        recs.append(rec)
    return pd.DataFrame(recs)


def show_tokens(tokenizer, text: str, pair: str | None = None) -> None:
    """토크나이저가 문장을 어떤 서브워드로 쪼개는지 보여준다."""
    enc = tokenizer(text, pair) if pair is not None else tokenizer(text)
    ids = enc["input_ids"]
    toks = tokenizer.convert_ids_to_tokens(ids)
    print(f"원문   : {text}" + (f"\n원문2  : {pair}" if pair else ""))
    print(f"토큰 {len(toks)}개 : {' '.join(toks)}")
    print(f"토큰 id : {ids[:20]}{' …' if len(ids) > 20 else ''}")
    print("\n※ '##'이나 '_'가 붙은 조각은 앞 토큰에 이어지는 **서브워드**입니다.")
    print("   한국어는 조사·어미가 붙어 단어가 길어지므로, 모델은 단어가 아니라 이런 조각 단위로 봅니다.")


# --------------------------------------------------------------------------------------
# 2. BERT (인코더) — 토크나이저·학습·예측
# --------------------------------------------------------------------------------------

def load_bert_tokenizer(model_id: str = MODEL_BERT):
    """klue/roberta-base 처럼 RoBERTa 계열(`type_vocab_size=1`)인데 토크나이저는 BERT식으로
    두 번째 문장에 `token_type_ids=1`을 붙이는 경우가 있다 → 임베딩 인덱스 초과로 CUDA assert.
    모델이 문장 구분 임베딩을 갖지 않으면 `token_type_ids`를 아예 만들지 않는다."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if getattr(AutoConfig.from_pretrained(model_id), "type_vocab_size", 2) < 2:
        tokenizer.model_input_names = [n for n in tokenizer.model_input_names if n != "token_type_ids"]
    return tokenizer


def collator_for(task: str, tokenizer):
    """태스크에 맞는 배치 묶기 함수.

    - 개체명인식: 라벨도 함께 패딩해야 한다(-100으로)
    - 기계독해: 이미 `padding="max_length"`로 길이를 맞춰 두었으므로 그대로 쌓기만 한다
    """
    if task == "ner":
        return DataCollatorForTokenClassification(tokenizer)
    if task == "mrc":
        return default_data_collator
    return DataCollatorWithPadding(tokenizer)


# --------------------------------------------------------------------------------------
# 2-1. 학습 중 검증 — 손실만 보지 말고 "지금 얼마나 하는지"를 같이 본다
# --------------------------------------------------------------------------------------
# 학습 데이터에서 일부(기본 10%)를 떼어 **검증셋**으로 쓴다.
# 평가셋(`data/llm-ft/eval_*.jsonl`)은 마지막 점수를 내는 자리라 여기에 절대 쓰지 않는다.
# 학습 → 검증 → 평가, 세 몫을 나누는 것이 이 실습에서 배울 것 하나다.

VAL_RATIO = 0.1          # 학습 예제의 몇 %를 검증에 떼어 둘 것인가
VAL_MIN = 40             # 그보다 적으면 숫자가 흔들려 볼 값이 못 된다


def split_val(ds, ratio: float = VAL_RATIO, seed: int = 42):
    """학습셋을 (학습, 검증) 으로 나눈다. 검증이 너무 작아지면 나누지 않는다."""
    n_val = int(len(ds) * ratio)
    if ratio <= 0 or n_val < VAL_MIN:
        return ds, None
    sp = ds.train_test_split(test_size=n_val, seed=seed, shuffle=True)
    return sp["train"], sp["test"]


def metrics_for(task: str):
    """태스크마다 검증 화면에 띄울 지표를 돌려준다. 최종 점수와 지표 이름이 같다."""

    def tc_or_cls(pred):
        logits, labels = _logits(pred, task), pred.label_ids
        got = np.asarray(logits).argmax(-1)
        return {"정확도": round(float((got == labels).mean()) * 100, 2)}

    def sts(pred):
        got = np.asarray(_logits(pred, task)).squeeze(-1).tolist()
        return {"피어슨": round(pearson(got, [float(x) for x in pred.label_ids]) * 100, 2)}

    def ner(pred):
        logits, labels = np.asarray(_logits(pred, task)), np.asarray(pred.label_ids)
        got = logits.argmax(-1)
        keep = labels != -100                       # 특수 토큰 자리는 채점에서 뺀다 (미션 m-ner-1)
        p = [BIO_LABELS[i] for i in got[keep]]
        g = [BIO_LABELS[i] for i in labels[keep]]
        ent = [t for t in BIO_LABELS if t != "O"]   # O 를 빼야 "개체를 얼마나 잡나"가 보인다
        return {"토큰 정확도": round(float((got[keep] == labels[keep]).mean()) * 100, 2),
                "개체 F1(토큰)": round(macro_f1(p, g, ent) * 100, 2)}

    def mrc(pred):
        (sl, el), labels = pred.predictions[:2], pred.label_ids
        gs, ge = (labels if isinstance(labels, tuple) else (labels[0], labels[1]))
        ps, pe = np.asarray(sl).argmax(-1), np.asarray(el).argmax(-1)
        both = ((ps == np.asarray(gs)) & (pe == np.asarray(ge))).mean()
        return {"시작·끝 둘 다 맞음": round(float(both) * 100, 2),
                "시작만": round(float((ps == np.asarray(gs)).mean()) * 100, 2)}

    return {"tc": tc_or_cls, "cls": tc_or_cls, "nli": tc_or_cls,
            "sts": sts, "ner": ner, "mrc": mrc}.get(task)


def train_bert(model, tokenizer, train_ds, task: str, *, epochs: float = 3.0, lr: float = 5e-5,
               batch_size: int = 32, seed: int = 42, logging_steps: int = 20, max_len: int = MAX_LEN,
               val_ratio: float = VAL_RATIO):
    """`bert_baseline.py --mode budget`과 같은 설정으로 학습한다.

    - `save_strategy="no"` : 체크포인트를 남기지 않는다(강의장 디스크 절약)
    - `report_to=[]`       : wandb 등 외부 로깅을 끈다
    - `val_ratio`          : 학습 예제에서 이만큼을 떼어 **검증셋**으로 쓴다. 학습 중간중간
                             손실만이 아니라 **그 태스크의 지표**(정확도·F1·피어슨)를 함께 보여 준다.
                             `0` 을 주면 예전처럼 전부 학습에 쓰고 검증을 하지 않는다.
    """
    torch.manual_seed(seed)
    train_ds, val_ds = split_val(train_ds, val_ratio, seed)
    total_steps = math.ceil(len(train_ds) / batch_size) * epochs
    # 검증을 몇 번 보여 줄까 — 학습이 짧아도 4~6번은 찍히게
    eval_every = max(1, int(total_steps // 5))
    kw = {}
    if val_ds is not None:
        kw = dict(eval_strategy="steps", eval_steps=eval_every)
        print(f"  학습 {len(train_ds)}건 · 검증 {len(val_ds)}건 (학습 예제의 {val_ratio:.0%}를 떼어 둠) "
              f"— {eval_every}스텝마다 검증 점수를 함께 봅니다")
    args = TrainingArguments(
        output_dir=str(OUT_DIR / "bert" / task), num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=64,
        warmup_steps=int(total_steps * 0.1), weight_decay=0.01, logging_steps=logging_steps,
        save_strategy="no", report_to=[], seed=seed, bf16=torch.cuda.is_available(), **kw,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      compute_metrics=metrics_for(task) if val_ds is not None else None,
                      data_collator=collator_for(task, tokenizer), processing_class=tokenizer)
    reset_peak_memory()
    t0 = time.time()
    out = trainer.train()
    sec = time.time() - t0
    peak = peak_memory_gb()
    n_params = sum(p.numel() for p in model.parameters())
    meta = {"model": model.name_or_path, "n_train": len(train_ds), "epochs": epochs, "lr": lr,
            "batch_size": batch_size, "max_len": max_len, "device": device_name(),
            "train_seconds": round(sec, 1), "peak_gpu_gb": round(peak, 2),
            "final_loss": round(out.training_loss, 4), "total_params": n_params}
    print(f"\n  학습 {sec / 60:.1f}분({sec:.0f}초) · GPU 최대 {peak:.2f}GB · "
          f"마지막 손실 {out.training_loss:.4f} · 파라미터 {n_params / 1e6:.0f}M (전부 학습)")
    return trainer, meta


def save_model(model, tokenizer, task: str, kind: str, out_dir=OUT_DIR_DAY2) -> Path:
    """학습한 모델을 데모에서 다시 쓸 수 있게 저장한다.

    학습 함수는 `save_strategy="no"` 라 체크포인트를 남기지 않는다(강의장 디스크 절약).
    데모 페이지(`day2/serve_mrc.py`)는 저장된 모델이 있어야 띄울 수 있으므로, 학습이 끝난 뒤
    이 함수로 한 번만 저장한다. 토크나이저도 함께 저장해야 데모가 같은 방식으로 자른다.

        L.save_model(model, tokenizer, "mrc", "bert")   → output/day2/mrc-bert/
    """
    path = Path(out_dir) / f"{task}-{kind}"
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    print(f"  저장: {path}  ({size / 1024 ** 2:.0f}MB)")
    print(f"  데모: python day2/serve_mrc.py --kind {kind} --port 9006")
    return path


def _logits(pred_out, task: str):
    logits = pred_out.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits


def predict_choice(trainer, eval_ds, labels: list[str]) -> list[str]:
    """분류: 가장 점수가 높은 라벨을 **문자열로** 돌려준다 (LLM 출력과 같은 형식)."""
    logits = _logits(trainer.predict(eval_ds), "tc")
    return [labels[int(i)] for i in logits.argmax(-1)]


def predict_sts(trainer, eval_ds) -> list[str]:
    """회귀: 0~5로 자른 뒤 소수점 한 자리 문자열로 (LLM이 내놓는 형식과 같게)."""
    logits = _logits(trainer.predict(eval_ds), "sts")
    return [f"{float(np.clip(v, 0, 5)):.1f}" for v in np.asarray(logits).reshape(-1)]


def decode_ner(text: str, offsets, pred_ids) -> list[dict]:
    """토큰 BIO 예측 → (텍스트, 유형) 개체 목록. `bert_baseline.decode_ner`와 같다."""
    ents, cur = [], None
    for (s, e), pid in zip(offsets, pred_ids):
        if e <= s:
            continue
        tag = BIO_LABELS[pid]
        if tag == "O":
            if cur:
                ents.append(cur)
            cur = None
            continue
        bio, typ = tag.split("-")
        # 새 개체가 시작되는 조건 — 노트북 미션 m-ner-3 이 이 판단을 직접 채운다
        starts_new = (bio == "B") or cur is None or (cur["label"] != typ)
        if starts_new:
            if cur:
                ents.append(cur)
            cur = {"label": typ, "start": s, "end": e}
        else:
            cur["end"] = e
    if cur:
        ents.append(cur)
    return [{"text": text[c["start"]:c["end"]].strip(), "label": c["label"]} for c in ents]


def predict_ner(trainer, eval_ds, tokenizer, eval_rows, max_len: int = MAX_LEN,
                decode=None) -> list[str]:
    """토큰별 BIO 예측 → 개체 목록 JSON 문자열 (LLM이 내놓는 형식과 같게).

    `decode` 를 주면 그 함수로 BIO 를 이어 붙인다 — 노트북 미션 `m-ner-3` 이 그것이다.
    """
    decode = decode or decode_ner
    logits = _logits(trainer.predict(eval_ds), "ner")
    enc = tokenizer([r["input"]["text"] for r in eval_rows], truncation=True,
                    max_length=max_len, return_offsets_mapping=True)
    pred_ids = logits.argmax(-1)
    preds = []
    for i, r in enumerate(eval_rows):
        off = enc["offset_mapping"][i]
        ents = decode(r["input"]["text"], off, pred_ids[i][: len(off)])
        preds.append(json.dumps(ents, ensure_ascii=False))
    return preds


def raw_logits(trainer, eval_ds):
    """모델이 낸 **날것의 점수**. 노트북 미션에서 이것을 문자열로 되돌린다.

    BERT 는 라벨마다(또는 토큰마다) 점수를 낸다. LLM·T5 는 처음부터 글자를 써낸다.
    셋을 **같은 채점 함수**로 재려면 BERT 쪽 숫자를 LLM 이 내놓는 형식의 문자열로 맞춰야 한다.
    그 변환이 노트북의 세 번째 미션이다.
    """
    return _logits(trainer.predict(eval_ds), "")


def predict_bert(task: str, trainer, eval_ds, tokenizer, eval_rows, *,
                 offsets=None, sample_map=None, decode=None, postprocess=None) -> list[str]:
    """태스크에 맞는 예측 함수를 골라 부른다. 기계독해만 `offsets`·`sample_map`이 더 필요하다.

    `decode` 를 주면 개체명인식에서 **노트북이 만든 BIO 이어붙이기 함수**를 대신 쓴다
    (미션 `m-ner-3`). `postprocess` 를 주면 기계독해에서 **노트북이 만든 구간 고르기 함수**를
    대신 쓴다(미션 `m-mrc-3`). 주지 않으면 이 파일의 것을 쓴다.
    """
    if task == "tc":
        return predict_choice(trainer, eval_ds, TC_LABELS)
    if task == "sts":
        return predict_sts(trainer, eval_ds)
    if task == "ner":
        return predict_ner(trainer, eval_ds, tokenizer, eval_rows, decode=decode)
    if task == "mrc":
        assert offsets is not None and sample_map is not None, \
            "기계독해는 encode_mrc_eval 이 돌려준 offsets·sample_map 을 함께 넘겨야 합니다"
        return predict_mrc(trainer, eval_ds, eval_rows, offsets, sample_map, postprocess=postprocess)
    raise ValueError(task)


# --- 기계독해(추출형) — 지문에서 답의 시작·끝 위치를 고른다 ----------------------------
#
# 다른 태스크와 결정적으로 다른 점이 둘 있다.
#   1. 지문이 길어 한 번에 안 들어간다 → `stride`만큼 겹쳐 가며 **여러 조각**으로 나눈다
#      (그래서 예제 300건이 feature 400개쯤이 된다). 조각과 예제를 잇는 것이 `sample_map`이다.
#   2. 답이 라벨이 아니라 **지문 안의 구간**이다 → 라벨은 시작·끝 **토큰 번호**이고,
#      예측을 되돌릴 때 `offset_mapping`으로 다시 글자 위치를 찾아야 한다.
#
# `encode_mrc_train`(학습용 라벨 만들기)은 미션 `m-mrc`이므로 여기 두지 않고 노트북 셀에 남긴다.


def encode_mrc_eval(tokenizer, rows: list[dict], max_len: int = MAX_LEN_MRC, stride: int = STRIDE_MRC):
    """평가용 인코딩. 학습용과 달리 라벨을 만들지 않고, 대신 나중에 답 문자열을 되찾을 수 있도록
    `offset_mapping`과 "이 조각이 몇 번째 예제에서 나왔는지"(`sample_map`)를 함께 돌려준다.

    지문이 아닌 자리(질문·특수토큰)의 offset은 `None`으로 지워 둔다 — 답은 지문에서만 찾아야 한다.
    """
    questions = [r["input"]["question"].strip() for r in rows]
    contexts = [r["input"]["context"] for r in rows]
    enc = tokenizer(questions, contexts, truncation="only_second", max_length=max_len, stride=stride,
                    return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length")
    sample_map = enc.pop("overflow_to_sample_mapping")
    for i in range(len(enc["input_ids"])):
        seq_ids = enc.sequence_ids(i)
        enc["offset_mapping"][i] = [o if seq_ids[k] == 1 else None
                                    for k, o in enumerate(enc["offset_mapping"][i])]
    features = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    if "token_type_ids" in enc:
        features["token_type_ids"] = enc["token_type_ids"]
    return Dataset.from_dict(features), enc["offset_mapping"], sample_map


def postprocess_mrc(rows, offsets, sample_map, start_logits, end_logits,
                    n_best: int = 20, max_answer_len: int = 30) -> list[str]:
    """조각별 로짓 → 예제별 최고 점수 답 문자열. `bert_baseline.postprocess_mrc`와 같다.

    한 예제가 여러 조각으로 나뉘었을 수 있으므로, 조각을 모두 훑어 시작·끝 점수의 합이
    가장 큰 구간을 고른다. 지문 밖(`None`)이거나 끝이 시작보다 앞이거나 너무 긴 구간은 버린다.
    """
    from collections import defaultdict
    by_example = defaultdict(list)
    for i, ex_idx in enumerate(sample_map):
        by_example[ex_idx].append(i)
    preds = []
    for ex_idx, r in enumerate(rows):
        context = r["input"]["context"]
        best, best_score = "", -1e9
        for fi in by_example[ex_idx]:
            off = offsets[fi]
            s_idx = np.argsort(start_logits[fi])[-1: -n_best - 1: -1]
            e_idx = np.argsort(end_logits[fi])[-1: -n_best - 1: -1]
            for s in s_idx:
                for e in e_idx:
                    # 말이 안 되는 짝 — 노트북 미션 m-mrc-3 이 이 판단을 직접 채운다
                    invalid = off[s] is None or off[e] is None or (e < s) or (e - s + 1 > max_answer_len)
                    if invalid:
                        continue
                    score = start_logits[fi][s] + end_logits[fi][e]
                    if score > best_score:
                        best_score = score
                        best = context[off[s][0]: off[e][1]]
        preds.append(best.strip())
    return preds


def predict_mrc(trainer, eval_ds, eval_rows, offsets, sample_map, postprocess=None) -> list[str]:
    """예측 → 답 문자열. 다른 태스크와 달리 로짓이 두 개(시작·끝)로 나온다.

    `postprocess` 를 주면 그 함수로 구간을 고른다 — 노트북 미션 `m-mrc-3` 이 그것이다.
    """
    postprocess = postprocess or postprocess_mrc
    start_logits, end_logits = trainer.predict(eval_ds).predictions
    return postprocess(eval_rows, offsets, sample_map, start_logits, end_logits)


# --------------------------------------------------------------------------------------
# 3. T5 (인코더-디코더) — 같은 데이터를 "글자로 써내는" 방식으로
# --------------------------------------------------------------------------------------

def load_t5_tokenizer(model_id: str = MODEL_T5):
    """pko-t5의 tokenizer.json은 BPE인데 설정에는 T5Tokenizer(sentencepiece)로 적혀 있어
    최신 transformers의 자동 경로가 실패한다. 그럴 때는 tokenizer.json을 직접 읽는다."""
    try:
        return AutoTokenizer.from_pretrained(model_id)
    except Exception as e:  # noqa: BLE001
        print(f"  (AutoTokenizer 실패 → tokenizer.json 직접 로드: {type(e).__name__})")
        from huggingface_hub import hf_hub_download
        from transformers import PreTrainedTokenizerFast
        path = hf_hub_download(model_id, "tokenizer.json")
        return PreTrainedTokenizerFast(tokenizer_file=path, eos_token="</s>",
                                       pad_token="<pad>", unk_token="<unk>")


def to_text(rows: list[dict]) -> tuple[list[str], list[str]]:
    """LLM이 본 것과 **똑같은 지시문·답**을 그대로 쓴다(시스템 프롬프트만 뺀다).
    그래서 T5의 입력은 `messages[1]`(사용자 지시문), 정답은 `messages[2]`(답)이다."""
    return [r["messages"][1]["content"] for r in rows], [r["messages"][2]["content"] for r in rows]


def encode_t5(tokenizer, rows: list[dict], task: str, max_source: int = MAX_SOURCE_T5) -> Dataset:
    src, tgt = to_text(rows)
    max_target = min(TASKS[task]["max_new_tokens"] + 16, 512)
    enc = tokenizer(src, truncation=True, max_length=max_source)
    enc["labels"] = tokenizer(text_target=tgt, truncation=True, max_length=max_target)["input_ids"]
    return Dataset.from_dict(enc)


def _t5_val_metrics(tokenizer, task: str):
    """검증할 때 손실만이 아니라 **그 태스크의 진짜 채점 함수**를 함께 낸다 — mrc 뿐 아니라
    T5로 비교해 보는 tc·sts·ner 도 마찬가지다(01~03 의 "T5로도 해보면" 절).

    `predict_with_generate=True` 라 여기 들어오는 predictions 는 이미 생성된 답 토큰이다
    (BERT 쪽 `metrics_for`는 로짓이 맞았는지 보는 근사치이지만, 여기서는 `evaluate()`가
    최종 채점에 쓰는 것과 같은 함수를 검증 중간에도 그대로 쓴다).
    """
    def compute(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
        gold_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
        if task == "mrc":
            scored = [score_mrc(p, [g]) for p, g in zip(pred_texts, gold_texts)]
            ems, f1s = zip(*scored)
            return {"EM": round(float(np.mean(ems)) * 100, 2), "F1": round(float(np.mean(f1s)) * 100, 2)}
        if task == "tc":
            got = [pick_label(p, TC_LABELS) for p in pred_texts]
            acc = float(np.mean([g == t for g, t in zip(got, gold_texts)])) * 100
            return {"정확도": round(acc, 2)}
        if task == "sts":
            got = [parse_score(p) for p in pred_texts]
            got = [2.5 if v is None else v for v in got]
            gold_v = [parse_score(g) or 0.0 for g in gold_texts]
            return {"피어슨": round(pearson(got, gold_v) * 100, 2)}
        if task == "ner":
            golds = []
            for g in gold_texts:
                try:
                    golds.append(json.loads(g))
                except (json.JSONDecodeError, TypeError):
                    golds.append([])
            triples = [score_ner(p, g) for p, g in zip(pred_texts, golds)]
            tp, npred, ngold = (sum(x) for x in zip(*triples))
            prec = tp / npred if npred else 0.0
            rec = tp / ngold if ngold else 0.0
            f1 = 200 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            return {"개체 F1": round(f1, 2)}
        return {}
    return compute


def train_t5(model, tokenizer, train_ds, task: str, *, epochs: float = 3.0, lr: float = 3e-4,
             batch_size: int = 8, seed: int = 42, val_ratio: float = VAL_RATIO):
    """`t5_baseline.py --mode budget`과 같은 설정. T5는 bf16에서 불안정한 사례가 있어 fp32로 둔다.

    `predict_with_generate=True` 로 검증할 때도 실제로 답을 생성해, 손실만이 아니라
    EM·F1(§6 최종 평가와 같은 채점 함수)까지 함께 본다 — BERT 쪽(`train_bert`)과 같은
    간격으로, 전체 스텝의 1/5마다 본다. 검증셋이 작아(학습 예제의 10%) 생성해도 오래 걸리지 않는다.
    """
    torch.manual_seed(seed)
    train_ds, val_ds = split_val(train_ds, val_ratio, seed)
    total_steps = math.ceil(len(train_ds) / batch_size) * epochs
    eval_every = max(1, int(total_steps // 5))
    kw = {}
    if val_ds is not None:
        kw = dict(eval_strategy="steps", eval_steps=eval_every, predict_with_generate=True,
                  generation_max_length=TASKS[task]["max_new_tokens"], generation_num_beams=1)
        print(f"  학습 {len(train_ds)}건 · 검증 {len(val_ds)}건 (학습 예제의 {val_ratio:.0%}를 떼어 둠) "
              f"— {eval_every}스텝마다 검증 EM·F1 을 함께 봅니다 (생성해서 채점)")
    args = Seq2SeqTrainingArguments(
        output_dir=str(OUT_DIR / "t5" / task), num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size * 2,
        warmup_steps=int(total_steps * 0.1), weight_decay=0.01, logging_steps=20,
        save_strategy="no", report_to=[], seed=seed, bf16=False, **kw,
    )
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                             data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
                             processing_class=tokenizer,
                             compute_metrics=_t5_val_metrics(tokenizer, task) if val_ds is not None else None)
    reset_peak_memory()
    t0 = time.time()
    trainer.train()
    sec = time.time() - t0
    peak = peak_memory_gb()
    n_params = sum(p.numel() for p in model.parameters())
    meta = {"model": model.name_or_path, "n_train": len(train_ds), "epochs": epochs, "lr": lr,
            "batch_size": batch_size, "device": device_name(), "train_seconds": round(sec, 1),
            "peak_gpu_gb": round(peak, 2), "total_params": n_params}
    print(f"\n  학습 {sec / 60:.1f}분({sec:.0f}초) · GPU 최대 {peak:.2f}GB · 파라미터 {n_params / 1e6:.0f}M")
    return trainer, meta


def generate_t5(model, tokenizer, eval_rows: list[dict], task: str, *, batch_size: int = 32,
                max_source: int = MAX_SOURCE_T5) -> list[str]:
    """평가셋 전체에 대해 답을 **써낸다**(greedy). BERT의 argmax와 달리 글자를 하나씩 만든다."""
    model.eval()
    device = model.device
    src, _ = to_text(eval_rows)
    preds = []
    t0 = time.time()
    for i in range(0, len(src), batch_size):
        batch = tokenizer(src[i:i + batch_size], return_tensors="pt", padding=True,
                          truncation=True, max_length=max_source).to(device)
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=TASKS[task]["max_new_tokens"],
                                 num_beams=1, do_sample=False)
        preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    print(f"  생성 {len(preds)}건 · {time.time() - t0:.1f}초")
    return preds


def load_t5(model_id: str = MODEL_T5):
    return load_t5_tokenizer(model_id), AutoModelForSeq2SeqLM.from_pretrained(model_id)


# --------------------------------------------------------------------------------------
# 4. 채점·저장·비교 — BERT·T5·GPT 계열이 모두 같은 함수를 쓴다
# --------------------------------------------------------------------------------------

def evaluate(task: str, preds: list[str], eval_rows: list[dict]) -> dict:
    """`common.score_task`로 채점하고 사람이 읽을 한 줄을 출력한다."""
    summary = score_task(task, preds, eval_rows)
    summary["n"] = len(eval_rows)
    print("  " + describe(task, summary))
    return summary


def save_result(task: str, kind: str, summary: dict, meta: dict, out_dir=OUT_DIR) -> Path:
    """결과를 JSON으로 남긴다. `kind`는 'bert' 또는 't5'."""
    path = Path(out_dir) / f"{task}-{kind}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"kind": kind, "task": task, "mode": "budget",
                                "model": meta.get("model"), "results": {task: summary},
                                "train": meta}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  저장: {path}")
    return path


def _row_score(task: str, pred: str, row: dict) -> float:
    """예제 하나가 얼마나 맞았는지 (0~100). 틀린 예를 고를 때 쓴다."""
    if task == "tc":
        return 100.0 if pick_label(pred, TC_LABELS) == row["gold"] else 0.0
    if task == "sts":
        v = parse_score(pred)
        v = 2.5 if v is None else v
        return max(0.0, 100.0 - abs(v - float(row["gold"])) * 20)     # 오차 5.0 → 0점
    if task == "ner":
        tp, npred, ngold = score_ner(pred, row["gold"])
        prec = tp / npred if npred else 0.0
        rec = tp / ngold if ngold else 0.0
        return 200 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    if task == "mrc":
        return score_mrc(pred, row["gold"])[1] * 100      # (EM, F1) 중 부분 점수 F1
    raise ValueError(task)


def examples(task: str, preds: list[str], eval_rows: list[dict], n: int = 3):
    """**맞힌 예와 틀린 예를 함께** 보여준다 — 맞힌 것 `n`건, 틀린 것 `n`건.

    틀린 것만 보면 "이 모델은 못 쓰겠다"로 읽히기 쉽다. 실제로는 대부분을 맞히고 있고,
    어디서 무너지는지가 따로 있다. 그 둘을 나란히 놓아야 모델을 제대로 판단할 수 있다.

    `채점` 칸은 예제 하나의 점수다 — 주제분류는 맞다/틀리다, 문장유사도는 오차,
    개체명·기계독해는 겹치는 정도(F1)라서 부분 점수가 나온다.
    """
    import pandas as pd

    def ok(sc):
        """이 점수를 '맞았다'로 볼 것인가.

        주제분류·수학추론처럼 맞다/틀리다뿐인 태스크는 만점이어야 맞은 것이다.
        개체명·기계독해·문장유사도·SQL 은 부분 점수가 있으므로 절반을 넘으면 맞은 쪽으로 센다.
        """
        return sc >= (99.9 if task in ("cls", "tc", "nli", "math") else 60.0)

    # key= 를 반드시 준다 — 점수가 같으면 파이썬이 다음 원소를 비교하려다 터진다
    scored = sorted(((_row_score(task, p, r), i) for i, (p, r) in enumerate(zip(preds, eval_rows))),
                    key=lambda x: x[0], reverse=True)
    good = [t for t in scored if ok(t[0])][:n]              # 잘 맞힌 것부터
    bad = [t for t in scored if not ok(t[0])][-n:][::-1]     # 가장 크게 틀린 것부터
    picked = ([("○ 맞음", sc, i) for sc, i in good]
              + [("✗ 틀림", sc, i) for sc, i in bad])

    recs = []
    for mark, sc, i in picked:
        rec = {"채점": f"{mark}  {sc:.0f}점"}
        rec.update({k: _short(v) for k, v in eval_rows[i]["input"].items()})
        rec["정답"] = _short(eval_rows[i]["gold"], 60)
        rec["모델 예측"] = _short(preds[i], 60)
        recs.append(rec)
    # 노트북에서는 본문을 자르지 않는 HTML 카드로 보인다 — 기계독해는 지문 안에 정답·예측 구간을 표시한다
    from common import ExamplesView, examples_html   # noqa: PLC0415
    return ExamplesView(examples_html(task, picked, preds, eval_rows), pd.DataFrame(recs))


def mistakes(task: str, preds: list[str], eval_rows: list[dict], n: int = 3):
    """(옛 이름) 맞힌 예도 함께 보여주도록 바뀌었다 — `examples` 를 쓴다."""
    return examples(task, preds, eval_rows, n)


def load_result(task: str, kind: str, out_dir=OUT_DIR) -> dict | None:
    """앞 노트북이 저장해 둔 결과를 읽는다. 없으면 None (그 노트북을 아직 안 돌린 것)."""
    path = Path(out_dir) / f"{task}-{kind}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["results"][task]


def llm_score(task: str, path=LLM_RESULT) -> float | None:
    """2일차에 쓸 LLM(A.X-4.0-Light, LoRA 학습 후)의 같은 태스크 점수. 없으면 None."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        results = json.loads(path.read_text(encoding="utf-8"))["results"]
        return results[task][TASKS[task]["metric"]]
    except (KeyError, json.JSONDecodeError):
        return None


def compare(task: str, bert_summary: dict, t5_summary: dict):
    """BERT vs T5 (+ 2일차 LLM) 비교표를 만들고, 예고 한 줄을 출력한다."""
    import pandas as pd
    metric = TASKS[task]["metric"]
    rows = [
        {"방식": "BERT (인코더)", "모델": MODEL_BERT, "하는 일": "라벨·태그를 고른다",
         METRIC_NAME.get(metric, metric): bert_summary[metric]},
        {"방식": "T5 (인코더-디코더)", "모델": MODEL_T5, "하는 일": "답을 글자로 써낸다",
         METRIC_NAME.get(metric, metric): t5_summary[metric]},
    ]
    llm = llm_score(task)
    if llm is not None:
        rows.append({"방식": "GPT 계열 (디코더) — 2일차", "모델": "A.X-4.0-Light + LoRA",
                     "하는 일": "지시문을 읽고 이어 써낸다", METRIC_NAME.get(metric, metric): llm})
    df = pd.DataFrame(rows)
    print(f"[{TASKS[task]['name']}] 같은 학습셋 · 같은 평가셋 {N_EVAL}건 · 같은 채점 함수\n")
    if llm is not None:
        print(f"\n▶ 2일차 예고: 같은 태스크를 LLM 하나(+LoRA 어댑터 하나)로 풀면 {llm:.1f}점입니다. "
              f"태스크마다 모델을 따로 만들지 않고 **여섯 태스크를 한 모델로** 배웁니다.")
    return df


def free_gpu() -> None:
    """쓰지 않는 GPU 메모리를 돌려준다. 앞 모델을 `del` 한 뒤에 부른다."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------
# 12. 미션·퀴즈 객관식 (QuizKit) — 노트북 안에서 보기를 눌러 정답을 확인한다
# --------------------------------------------------------------------------------------

def demo_start(kind: str = "bert", port: int = 9006, out_dir=None) -> str:
    """기계독해 데모 서버를 배경에서 켠다 — `bert` · `t5` · `both`. 셀이 멈추지 않는다.

    끌 때는 `L.demo_stop()`. 모델은 `L.save_model` 이 저장한 `output/day2/mrc-*/` 에서 읽는다.
    """
    import sys
    from common import demo_start as _start   # noqa: PLC0415
    cmd = [sys.executable, str(ROOT / "day2" / "serve_mrc.py"), "--kind", kind, "--port", str(port)]
    if out_dir:
        cmd += ["--out-dir", str(out_dir)]
    return _start(cmd, port, name=f"기계독해 데모({kind})", wait=120, cwd=str(ROOT))


def demo_stop(port: int = 9006) -> None:
    """켜 둔 기계독해 데모 서버를 끈다."""
    from common import demo_stop as _stop     # noqa: PLC0415
    _stop(port, name="기계독해 데모")


def demo_status(port: int = 9006) -> bool:
    from common import demo_status as _status  # noqa: PLC0415
    return _status(port, name="기계독해 데모")


def quiz(item_id: str) -> None:
    """객관식 카드를 그린다: L.quiz("m-tc-1"). 문항은 quiz/public.json (`python tools/quizkit.py build`)."""
    tools_dir = str(ROOT / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import quizkit  # noqa: PLC0415
    quizkit.show(item_id)

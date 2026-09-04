"""
[실습5 비교] BERT류 기준선 — 같은 평가셋에서 LLM과 정면 비교

1일차 실습(task1~task4)에서 했던 방식 그대로, 인코더 모델(BERT/RoBERTa/ELECTRA)에
태스크별 head를 붙여 학습한다. 다만 **LLM과 똑같은 평가셋, 똑같은 채점 함수**를 쓴다.
그래야 "BERT vs LLM"을 숫자로 공정하게 비교할 수 있다.

두 가지 학습 모드:
  --mode budget : LLM이 쓴 것과 **같은 소규모 학습셋**(train_all.jsonl의 해당 태스크)으로 학습
                  → "같은 데이터를 줬을 때 누가 더 잘 배우나"
  --mode full   : 데이터셋의 **공식 train 전체**로 학습 (1일차 실습의 방식)
                  → "BERT가 제 실력을 다 냈을 때 LLM과 얼마나 차이 나나"

태스크별 head:
  cls / tc / nli : AutoModelForSequenceClassification  (문장 → 라벨)
  sts            : AutoModelForSequenceClassification  (num_labels=1, 회귀)
  ner            : AutoModelForTokenClassification     (토큰마다 BIO 태그)
  mrc            : AutoModelForQuestionAnswering       (답의 시작/끝 위치)

사용법:
  python task5-llm-ft/bert_baseline.py --task cls --mode budget
  python task5-llm-ft/bert_baseline.py --task ner --mode full --model klue/roberta-base
  python task5-llm-ft/bert_baseline.py --task mrc --mode full --save results/bert/roberta-mrc-full.json
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (AutoConfig, AutoModelForQuestionAnswering, AutoModelForSequenceClassification,
                          AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments, default_data_collator)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (NER_LABELS, NLI_LABELS, TASKS, TC_LABELS, describe, read_jsonl, score_task,  # noqa: E402
                    reset_peak_memory, peak_memory_gb, device_name)

BIO_LABELS = ["O"] + [f"{p}-{t}" for t in NER_LABELS for p in ("B", "I")]
BIO2ID = {l: i for i, l in enumerate(BIO_LABELS)}
CHOICE_LABELS = {"cls": ["부정", "긍정"], "tc": TC_LABELS, "nli": NLI_LABELS}   # 정수 라벨 순서와 같다


# --------------------------------------------------------------------------------------
# 학습 데이터 읽기 — budget(LLM과 동일) / full(공식 train 전체)
# --------------------------------------------------------------------------------------

def load_budget(task: str, data_dir: Path) -> list[dict]:
    for name in ("train_main.jsonl", "train_all.jsonl", "train.jsonl"):
        src = data_dir / name
        if src.exists():
            break
    rows = [r for r in read_jsonl(src) if r["task"] == task]
    if not rows:
        raise SystemExit(f"{src}에 {task} 예제가 없습니다. build_dataset.py를 먼저 실행하세요.")
    return rows


def load_full(task: str, raw_dir: Path, eval_keys: set, max_train: int | None) -> list[dict]:
    """공식 train 파일 전체를 build_dataset의 변환기로 바꿔서 읽는다 (평가셋과 겹치는 것은 제외)."""
    from build_dataset import make_cls, make_mrc, make_ner, make_nli, make_sts, make_tc
    mrc_path = raw_dir / "korquad/train.jsonl"
    if not mrc_path.exists():
        # 공식 train 전체(87MB)는 공개 저장소에 없다 — 절반 세트로 대신한다 (수치가 조금 낮아진다)
        mrc_path = raw_dir / "korquad/train-half.jsonl"
    paths = {"cls": raw_dir / "nsmc/train.jsonl", "ner": raw_dir / "klue-ner/train.jsonl",
             "mrc": mrc_path, "tc": raw_dir / "klue-ynat/train.jsonl",
             "nli": raw_dir / "klue-nli/train.jsonl", "sts": raw_dir / "klue-sts/train.jsonl"}
    conv = {"cls": make_cls, "ner": make_ner, "mrc": lambda r: make_mrc(r, 10**9),
            "tc": make_tc, "nli": make_nli, "sts": make_sts}[task]
    rows, dropped = [], 0
    for r in read_jsonl(paths[task]):
        ex = conv(r)
        if not ex:
            continue
        if ex["key"] in eval_keys:
            dropped += 1
            continue
        rows.append(ex)
        if max_train and len(rows) >= max_train:
            break
    tag = "" if (task != "mrc" or mrc_path.name == "train.jsonl") else " (절반 세트 — 공식 전체는 이 저장소에 없습니다)"
    print(f"  공식 train 전체{tag} {len(rows)}건 (평가셋과 겹쳐 제외 {dropped}건)")
    return rows


# --------------------------------------------------------------------------------------
# 태스크별 전처리 — 원본 입력 → 모델 입력(토큰 id + 라벨)
# --------------------------------------------------------------------------------------

def encode_choice(task, tokenizer, rows, max_len):
    """분류: 문장(또는 문장 쌍) → 정수 라벨"""
    if task == "nli":
        enc = tokenizer([r["input"]["premise"] for r in rows], [r["input"]["hypothesis"] for r in rows],
                        truncation=True, max_length=max_len)
    else:
        enc = tokenizer([r["input"]["text"] for r in rows], truncation=True, max_length=max_len)
    enc["labels"] = [int(r["raw"]["label"]) for r in rows]
    return Dataset.from_dict(dict(enc))


def encode_sts(tokenizer, rows, max_len):
    enc = tokenizer([r["input"]["sentence1"] for r in rows], [r["input"]["sentence2"] for r in rows],
                    truncation=True, max_length=max_len)
    enc["labels"] = [float(r["raw"]["score"]) for r in rows]
    return Dataset.from_dict(dict(enc))


def encode_ner(tokenizer, rows, max_len):
    """개체명: 글자 단위 BIO → 토큰 단위 BIO (토큰의 첫 글자 태그를 따른다)"""
    texts = [r["input"]["text"] for r in rows]
    enc = tokenizer(texts, truncation=True, max_length=max_len, return_offsets_mapping=True)
    all_labels = []
    for i, r in enumerate(rows):
        cl = r["raw"]["char_labels"]
        labels = []
        prev_type = None
        for (s, e) in enc["offset_mapping"][i]:
            if e <= s:                                   # 특수 토큰
                labels.append(-100)
                prev_type = None
                continue
            tag = cl[s] if s < len(cl) else "O"
            if tag == "O":
                labels.append(BIO2ID["O"])
                prev_type = None
            else:
                typ = tag.split("-")[1]
                # 토큰 시작 글자가 B거나, 앞 토큰과 유형이 다르면 B; 아니면 I
                bio = "B" if (tag.startswith("B") or typ != prev_type) else "I"
                labels.append(BIO2ID[f"{bio}-{typ}"])
                prev_type = typ
        all_labels.append(labels)
    enc["labels"] = all_labels
    enc.pop("offset_mapping")
    return Dataset.from_dict(dict(enc))


def decode_ner(tokenizer, text, offsets, pred_ids) -> list[dict]:
    """토큰 BIO 예측 → (텍스트, 유형) 개체 목록"""
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
        if bio == "B" or cur is None or cur["label"] != typ:
            if cur:
                ents.append(cur)
            cur = {"label": typ, "start": s, "end": e}
        else:
            cur["end"] = e
    if cur:
        ents.append(cur)
    return [{"text": text[c["start"]:c["end"]].strip(), "label": c["label"]} for c in ents]


def encode_mrc_train(tokenizer, rows, max_len, stride):
    """기계독해 학습: 긴 지문은 여러 조각으로 나누고(doc stride), 답의 토큰 시작/끝 위치를 라벨로 준다."""
    questions = [r["input"]["question"].strip() for r in rows]
    contexts = [r["input"]["context"] for r in rows]
    enc = tokenizer(questions, contexts, truncation="only_second", max_length=max_len, stride=stride,
                    return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length")
    sample_map = enc.pop("overflow_to_sample_mapping")
    offsets = enc.pop("offset_mapping")
    starts, ends = [], []
    for i, off in enumerate(offsets):
        r = rows[sample_map[i]]
        ans_text = r["raw"]["answer_text"][0]
        ans_start = r["raw"]["answer_start"][0] if r["raw"]["answer_start"] else r["input"]["context"].find(ans_text)
        ans_end = ans_start + len(ans_text)
        seq_ids = enc.sequence_ids(i)
        ctx_s = seq_ids.index(1)
        ctx_e = len(seq_ids) - 1 - seq_ids[::-1].index(1)
        if not (off[ctx_s][0] <= ans_start and off[ctx_e][1] >= ans_end):
            starts.append(0)                            # 이 조각에 답이 없다 → CLS 위치
            ends.append(0)
            continue
        ts = ctx_s
        while ts <= ctx_e and off[ts][0] <= ans_start:
            ts += 1
        te = ctx_e
        while te >= ctx_s and off[te][1] >= ans_end:
            te -= 1
        starts.append(ts - 1)
        ends.append(te + 1)
    enc["start_positions"] = starts
    enc["end_positions"] = ends
    return Dataset.from_dict(dict(enc))


def encode_mrc_eval(tokenizer, rows, max_len, stride):
    questions = [r["input"]["question"].strip() for r in rows]
    contexts = [r["input"]["context"] for r in rows]
    enc = tokenizer(questions, contexts, truncation="only_second", max_length=max_len, stride=stride,
                    return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length")
    sample_map = enc.pop("overflow_to_sample_mapping")
    for i in range(len(enc["input_ids"])):
        seq_ids = enc.sequence_ids(i)
        enc["offset_mapping"][i] = [o if seq_ids[k] == 1 else None for k, o in enumerate(enc["offset_mapping"][i])]
    features = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    if "token_type_ids" in enc:
        features["token_type_ids"] = enc["token_type_ids"]
    return Dataset.from_dict(features), enc["offset_mapping"], sample_map


def postprocess_mrc(rows, offsets, sample_map, start_logits, end_logits, n_best=20, max_answer_len=30):
    """조각별 로짓 → 예제별 최고 점수 답 문자열"""
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
                    if off[s] is None or off[e] is None or e < s or e - s + 1 > max_answer_len:
                        continue
                    score = start_logits[fi][s] + end_logits[fi][e]
                    if score > best_score:
                        best_score = score
                        best = context[off[s][0]: off[e][1]]
        preds.append(best.strip())
    return preds


# --------------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="BERT류 인코더 기준선을 LLM과 같은 평가셋으로 측정합니다.")
    p.add_argument("--task", required=True, choices=list(TASKS))
    p.add_argument("--mode", default="budget", choices=["budget", "full"])
    p.add_argument("--model", default="klue/roberta-base")
    p.add_argument("--data", default="data/llm-ft", help="build_dataset.py 출력 폴더")
    p.add_argument("--raw", default="data", help="원본 데이터 폴더 (full 모드)")
    p.add_argument("--max-train", type=int, default=None, help="full 모드에서 학습 예제 상한")
    p.add_argument("--limit", type=int, default=300, help="평가 예제 수 (LLM 평가와 같게)")
    p.add_argument("--epochs", type=float, default=None, help="기본: budget 3 / full 1")
    p.add_argument("--lr", type=float, default=None, help="기본: budget 5e-5 / full 3e-5")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-len", type=int, default=None, help="기본: mrc 384 / 그 외 128")
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--out", default=None, help="체크포인트 폴더 (기본 output/bert/<task>-<mode>)")
    p.add_argument("--save", default=None, help="결과 JSON 경로")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    task = args.task
    epochs = args.epochs if args.epochs is not None else (3.0 if args.mode == "budget" else 1.0)
    lr = args.lr if args.lr is not None else (5e-5 if args.mode == "budget" else 3e-5)
    max_len = args.max_len or (384 if task == "mrc" else 128)
    out_dir = args.out or f"output/bert/{Path(args.model).name}-{task}-{args.mode}"
    torch.manual_seed(args.seed)

    data_dir = Path(args.data)
    eval_rows = read_jsonl(data_dir / f"eval_{task}.jsonl", args.limit)
    eval_keys = {r["key"] for r in eval_rows}
    print(f"[{TASKS[task]['name']}] 모델 {args.model} / 모드 {args.mode} / 평가 {len(eval_rows)}건")

    if args.mode == "budget":
        train_rows = load_budget(task, data_dir)
        print(f"  LLM과 동일한 학습셋 {len(train_rows)}건")
    else:
        train_rows = load_full(task, Path(args.raw), eval_keys, args.max_train)
    assert not ({r["key"] for r in train_rows} & eval_keys), "학습셋과 평가셋이 겹칩니다"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # klue/roberta-base 처럼 RoBERTa 계열(type_vocab_size=1)인데 토크나이저는 BERT식으로
    # 두 번째 문장에 token_type_ids=1 을 붙이는 경우가 있다 → 임베딩 인덱스 초과로 CUDA assert.
    # 모델이 문장 구분 임베딩을 갖지 않으면 token_type_ids 를 아예 만들지 않는다.
    if getattr(AutoConfig.from_pretrained(args.model), "type_vocab_size", 2) < 2:
        tokenizer.model_input_names = [n for n in tokenizer.model_input_names if n != "token_type_ids"]

    # --- 태스크별 모델과 데이터 ---
    if task in CHOICE_LABELS:
        labels = CHOICE_LABELS[task]
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=len(labels))
        train_ds = encode_choice(task, tokenizer, train_rows, max_len)
        eval_ds = encode_choice(task, tokenizer, eval_rows, max_len)
        collator = DataCollatorWithPadding(tokenizer)
    elif task == "sts":
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1, problem_type="regression")
        train_ds = encode_sts(tokenizer, train_rows, max_len)
        eval_ds = encode_sts(tokenizer, eval_rows, max_len)
        collator = DataCollatorWithPadding(tokenizer)
    elif task == "ner":
        model = AutoModelForTokenClassification.from_pretrained(
            args.model, num_labels=len(BIO_LABELS), id2label=dict(enumerate(BIO_LABELS)), label2id=BIO2ID)
        train_ds = encode_ner(tokenizer, train_rows, max_len)
        eval_ds = encode_ner(tokenizer, eval_rows, max_len)
        collator = DataCollatorForTokenClassification(tokenizer)
    else:  # mrc
        model = AutoModelForQuestionAnswering.from_pretrained(args.model)
        train_ds = encode_mrc_train(tokenizer, train_rows, max_len, args.stride)
        eval_ds, eval_offsets, eval_map = encode_mrc_eval(tokenizer, eval_rows, max_len, args.stride)
        collator = default_data_collator
    print(f"  학습 feature {len(train_ds)}개, 평가 feature {len(eval_ds)}개, max_len {max_len}, "
          f"epochs {epochs}, lr {lr}")

    import math
    total_steps = math.ceil(len(train_ds) / args.batch_size) * epochs
    targs = TrainingArguments(
        output_dir=out_dir, num_train_epochs=epochs, learning_rate=lr,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=64,
        warmup_steps=int(total_steps * 0.1), weight_decay=0.01, logging_steps=20, save_strategy="no",
        report_to=[], seed=args.seed, bf16=torch.cuda.is_available(),
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, data_collator=collator,
                      processing_class=tokenizer)

    reset_peak_memory()
    t0 = time.time()
    train_out = trainer.train()
    train_sec = time.time() - t0
    peak_gb = peak_memory_gb()                # CUDA: 최대 할당량 / MPS: 학습 직후 할당량

    # --- 예측 → LLM과 같은 문자열 형식으로 바꿔 같은 채점 함수에 넣는다 ---
    t1 = time.time()
    pred_out = trainer.predict(eval_ds)
    logits = pred_out.predictions
    if isinstance(logits, tuple) and task != "mrc":
        logits = logits[0]
    if task in CHOICE_LABELS:
        preds = [CHOICE_LABELS[task][int(i)] for i in logits.argmax(-1)]
    elif task == "sts":
        preds = [f"{float(np.clip(v, 0, 5)):.1f}" for v in np.asarray(logits).reshape(-1)]
    elif task == "ner":
        enc = tokenizer([r["input"]["text"] for r in eval_rows], truncation=True, max_length=max_len,
                        return_offsets_mapping=True)
        pred_ids = logits.argmax(-1)
        preds = []
        for i, r in enumerate(eval_rows):
            off = enc["offset_mapping"][i]
            ents = decode_ner(tokenizer, r["input"]["text"], off, pred_ids[i][: len(off)])
            preds.append(json.dumps(ents, ensure_ascii=False))
    else:
        start_logits, end_logits = logits
        preds = postprocess_mrc(eval_rows, eval_offsets, eval_map, start_logits, end_logits)
    eval_sec = time.time() - t1

    summary = score_task(task, preds, eval_rows)
    summary.update({"n": len(eval_rows), "sec": round(eval_sec, 1)})
    print("\n  " + describe(task, summary))

    n_params = sum(p.numel() for p in model.parameters())
    result = {
        "kind": "bert", "model": args.model, "task": task, "mode": args.mode,
        "results": {task: summary},
        "train": {"n_train": len(train_rows), "n_features": len(train_ds), "epochs": epochs, "lr": lr,
                  "batch_size": args.batch_size, "max_len": max_len, "device": device_name(),
                  "train_seconds": round(train_sec, 1),
                  "peak_gpu_gb": round(peak_gb, 2), "final_loss": round(train_out.training_loss, 4),
                  "total_params": n_params, "trainable_params": n_params},
    }
    print(f"  학습 {train_sec/60:.1f}분, GPU 최대 {peak_gb:.2f}GB, 파라미터 {n_params/1e6:.0f}M (전부 학습)")
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  저장: {args.save}")


if __name__ == "__main__":
    main()

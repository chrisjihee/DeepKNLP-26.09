"""
[실습5 비교] T5(인코더-디코더) 기준선 — BERT와 LLM 사이에 있는 세 번째 방식

세 방식의 차이를 한 문장으로:
  BERT (인코더)        : 문장을 읽고 **라벨이나 위치를 고른다**. 없는 말을 지어내지 못한다.
  T5   (인코더-디코더) : 문장을 읽고 **답을 글자로 써낸다**. 읽기 전용 인코더가 따로 있다.
  LLM  (디코더)        : 앞말에 이어 **다음 글자를 계속 써낸다**. 지시문만 바꾸면 다른 일을 한다.

1일차 실습에서 KorQuAD를 추출형(task4A, BERT)과 생성형(task4B, pko-t5)으로 모두 해봤다.
이 스크립트는 그 축을 실습5의 여섯 태스크로 넓힌다. **평가셋과 채점 함수는 LLM·BERT와 완전히 같다.**

두 가지 학습 모드 (bert_baseline.py 와 같은 규칙):
  --mode budget : LLM이 쓴 것과 같은 소규모 학습셋으로 학습
  --mode full   : 공식 train 전체로 학습

사용법:
  python task5-llm-ft/t5_baseline.py --task mrc --mode budget
  python task5-llm-ft/t5_baseline.py --task sql --mode full --model paust/pko-t5-large
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (TASKS, describe, device_name, peak_memory_gb,   # noqa: E402
                    read_jsonl, reset_peak_memory, score_task)

# 태스크별 생성 길이 상한 — LLM과 같은 값을 쓴다(수학추론만 400, SQL은 128).
GEN_TOKENS = {t: v["max_new_tokens"] for t, v in TASKS.items()}


def load_t5_tokenizer(model_id: str):
    """pko-t5의 tokenizer.json은 BPE인데 설정에는 T5Tokenizer(sentencepiece)로 적혀 있어
    최신 transformers의 자동 경로가 실패한다. 그럴 때는 tokenizer.json을 직접 읽는다."""
    try:
        return AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        print(f"  (AutoTokenizer 실패 → tokenizer.json 직접 로드: {type(e).__name__})")
        from huggingface_hub import hf_hub_download
        from transformers import PreTrainedTokenizerFast
        path = hf_hub_download(model_id, "tokenizer.json")
        return PreTrainedTokenizerFast(tokenizer_file=path, eos_token="</s>",
                                       pad_token="<pad>", unk_token="<unk>")


def load_budget(task: str, data_dir: Path) -> list[dict]:
    """LLM이 실제로 학습한 그 예제들 — train_main.jsonl 에서 해당 태스크만 꺼낸다."""
    for name in ("train_main.jsonl", "train_all.jsonl", "train.jsonl"):
        src = data_dir / name
        if src.exists():
            rows = [r for r in read_jsonl(src) if r["task"] == task]
            if rows:
                return rows
    raise SystemExit(f"{data_dir}에 {task} 학습 예제가 없습니다. build_dataset.py를 먼저 실행하세요.")


def load_full(task: str, raw_dir: Path, eval_keys: set, max_train: int | None) -> list[dict]:
    """공식 train 전체를 build_dataset의 변환기로 바꿔 읽는다(평가셋과 겹치는 것은 제외)."""
    import build_dataset as bd
    paths = {"cls": raw_dir / "nsmc/train.jsonl", "ner": raw_dir / "klue-ner/train.jsonl",
             "mrc": raw_dir / "korquad/train.jsonl", "tc": raw_dir / "klue-ynat/train.jsonl",
             "nli": raw_dir / "klue-nli/train.jsonl", "sts": raw_dir / "klue-sts/train.jsonl",
             "math": raw_dir / "gsm8k-ko/train.jsonl", "sql": raw_dir / "spider-ko/train.jsonl"}
    if task == "sql":
        schemas = json.loads((raw_dir / "spider-ko/schema.json").read_text(encoding="utf-8"))
        conv = lambda r: bd.make_sql(r, schemas, 2500)          # noqa: E731
    elif task == "math":
        conv = bd.make_math
    elif task == "mrc":
        conv = lambda r: bd.make_mrc(r, 10**9)                  # noqa: E731
    else:
        conv = {"cls": bd.make_cls, "ner": bd.make_ner, "tc": bd.make_tc,
                "nli": bd.make_nli, "sts": bd.make_sts}[task]
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
    print(f"  공식 train 전체에서 {len(rows)}건 (평가셋과 겹쳐 제외한 것 {dropped}건)")
    return rows


def to_text(rows: list[dict]) -> tuple[list[str], list[str]]:
    """LLM이 본 것과 같은 지시문·답을 그대로 쓴다(시스템 프롬프트만 뺀다)."""
    src = [r["messages"][1]["content"] for r in rows]
    tgt = [r["messages"][2]["content"] for r in rows]
    return src, tgt


def encode(tokenizer, rows, max_source, max_target):
    src, tgt = to_text(rows)
    enc = tokenizer(src, truncation=True, max_length=max_source)
    lab = tokenizer(text_target=tgt, truncation=True, max_length=max_target)
    enc["labels"] = lab["input_ids"]
    return Dataset.from_dict(enc)


def main():
    p = argparse.ArgumentParser(description="T5(인코더-디코더) 기준선을 학습하고 LLM과 같은 지표로 평가합니다.")
    p.add_argument("--task", required=True, choices=list(TASKS))
    p.add_argument("--mode", default="budget", choices=["budget", "full"])
    p.add_argument("--model", default="paust/pko-t5-base")
    p.add_argument("--data", default="data/llm-ft")
    p.add_argument("--raw", default="data")
    p.add_argument("--limit", type=int, default=300, help="평가 예제 수")
    p.add_argument("--max-train", type=int, default=None,
                   help="full 모드에서 읽을 최대 학습 예제 수 (기본: 제한 없음 — BERT full과 조건을 맞춘다)")
    p.add_argument("--epochs", type=float, default=None, help="기본: budget 3, full 1")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=8, help="full 모드에서는 16 정도로 키워도 된다")
    p.add_argument("--max-source", type=int, default=1024)
    p.add_argument("--out", default="output/t5")
    p.add_argument("--save", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    task = args.task
    data_dir, raw_dir = Path(args.data), Path(args.raw)
    eval_rows = read_jsonl(data_dir / f"eval_{task}.jsonl", limit=args.limit)
    eval_keys = {r["key"] for r in eval_rows}
    epochs = args.epochs if args.epochs else (3.0 if args.mode == "budget" else 1.0)

    if args.mode == "budget":
        train_rows = load_budget(task, data_dir)
    else:
        train_rows = load_full(task, raw_dir, eval_keys, args.max_train)

    print(f"[{task}] {TASKS[task]['name']} · {args.mode} · {args.model}")
    print(f"  학습 {len(train_rows)}건 / 평가 {len(eval_rows)}건 · epochs {epochs} · lr {args.lr}")

    tokenizer = load_t5_tokenizer(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    max_target = min(GEN_TOKENS[task] + 16, 512)
    train_ds = encode(tokenizer, train_rows, args.max_source, max_target)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    total_steps = math.ceil(len(train_ds) / args.batch_size) * epochs
    targs = Seq2SeqTrainingArguments(
        output_dir=f"{args.out}/{task}-{args.mode}", num_train_epochs=epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size * 2,
        warmup_steps=int(total_steps * 0.1), weight_decay=0.01, logging_steps=20, save_strategy="no",
        report_to=[], seed=args.seed, bf16=False,          # T5는 bf16에서 불안정한 사례가 있어 fp32로 둔다
    )
    trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=train_ds,
                             data_collator=collator, processing_class=tokenizer)

    reset_peak_memory()
    t0 = time.time()
    trainer.train()
    train_sec = time.time() - t0
    peak_gb = peak_memory_gb()

    # --- 생성 → LLM·BERT와 같은 채점 함수 ---
    model.eval()
    device = model.device
    src, _ = to_text(eval_rows)
    preds = []
    t1 = time.time()
    bs = max(1, args.batch_size)
    for i in range(0, len(src), bs):
        batch = tokenizer(src[i:i + bs], return_tensors="pt", padding=True,
                          truncation=True, max_length=args.max_source).to(device)
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=GEN_TOKENS[task], num_beams=1, do_sample=False)
        preds.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    eval_sec = time.time() - t1

    summary = score_task(task, preds, eval_rows)
    print("\n" + describe(task, summary))
    print(f"  학습 {train_sec/60:.1f}분 / 평가 {eval_sec:.1f}초 / GPU {peak_gb:.2f}GB")

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps({
            "model": args.model, "family": "t5", "task": task, "mode": args.mode,
            "n_train": len(train_rows), "n_eval": len(eval_rows), "epochs": epochs, "lr": args.lr,
            "train_seconds": round(train_sec, 1), "eval_seconds": round(eval_sec, 1),
            "peak_gpu_gb": round(peak_gb, 2), "device": device_name(),
            "total_params": sum(p.numel() for p in model.parameters()),
            "summary": summary,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"저장: {args.save}")


if __name__ == "__main__":
    main()

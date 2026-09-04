"""
[실습5] LLM 파인튜닝 - 통합 instruction 데이터셋 만들기

여러 한국어 NLP 태스크를 하나의 instruction 형식으로 합친다.
BERT 실습에서는 태스크마다 별도의 head(분류기)가 필요했지만,
LLM은 "무엇을 하라"는 지시문만 바꾸면 하나의 모델이 여러 태스크를 모두 처리할 수 있다.
이 스크립트는 그 지시문 데이터를 만든다.

  핵심 3태스크 (1일차 BERT 실습과 같은 데이터):
    cls  감성분류   NSMC        (train / test)
    ner  개체명인식 KLUE-NER    (train / test)
    mrc  기계독해   KorQuAD 1.0 (train / validation)
  확장 3태스크 (KLUE, Hugging Face에서 자동 내려받음):
    tc   주제분류   KLUE-YNAT   (train / validation)
    nli  자연어추론 KLUE-NLI    (train / validation)
    sts  문장유사도 KLUE-STS    (train / validation)

중요 원칙:
  - 학습셋은 각 데이터셋의 공식 train 파일에서만 만든다.
  - 평가셋은 공식 test/validation 파일에서만 만든다.
  - 두 집합이 겹치지 않는지 스크립트가 직접 검증한다. (데이터 리키지 금지)

출력 (기본 data/llm-ft/):
  eval_<task>.jsonl   태스크별 평가셋 (6개 모두)
  train.jsonl         핵심 3태스크 학습셋  — 수업 기본
  train_all.jsonl     6태스크 학습셋      — 확장 실험용
  stats.json

사용법:
  python task5-llm-ft/build_dataset.py                      # 기본 크기로 생성
  python task5-llm-ft/build_dataset.py --n-cls 2000         # 태스크별 개수 조정
  python task5-llm-ft/build_dataset.py --no-ext             # 확장 태스크 없이 (오프라인)
"""

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CORE_TASKS, MAIN_TASKS, NER_LABELS, NLI_LABELS, TASKS,   # noqa: E402
                    TC_LABELS, make_messages)

EXT_TASKS = ["tc", "nli", "sts"]


def norm_key(s: str) -> str:
    """리키지 검사용 정규화 키 — 공백을 없애고 해시한다."""
    return hashlib.sha1("".join(s.split()).encode("utf-8")).hexdigest()


def read_jsonl(path: Path, limit: int | None = None):
    rows = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------------------
# 태스크별 변환기 — 원본 한 줄 -> {"task", "messages", "gold", "input", "raw", "key"}
#   messages : LLM 학습/평가용 대화
#   gold     : 채점용 정답
#   input    : 지시문을 뺀 순수 입력 (BERT 기준선·데모 페이지가 함께 쓴다)
#   raw      : BERT 기준선이 필요로 하는 원본 정답 형식 (정수 라벨, 오프셋, 답 위치 등)
# --------------------------------------------------------------------------------------

def pack(task, payload, answer, gold, raw, key):
    return {"task": task, "messages": make_messages(task, payload, answer),
            "gold": gold, "input": payload, "raw": raw, "key": norm_key(key)}


def make_cls(row: dict) -> dict | None:
    """NSMC: {"id","document","label"} -> 감성분류 지시문"""
    text = (row.get("document") or "").strip()
    if not text:
        return None
    label = int(row["label"])
    answer = "긍정" if label == 1 else "부정"
    return pack("cls", {"text": text}, answer, answer, {"label": label}, text)


def make_ner(row: dict) -> dict | None:
    """KLUE-NER: {"origin","entity_list":[{text,label,offset}]} -> 개체명 JSON 지시문

    LLM 답에서 offset은 일부러 뺀다. 작은 모델이 문자 오프셋까지 맞추려면 출력이 쉽게 깨지고,
    수업의 목표는 '개체명을 찾아낸다'는 것이지 오프셋 계산이 아니기 때문이다.
    (BERT 기준선은 raw.entities의 offset을 써서 토큰 단위로 학습한다)
    """
    origin = row.get("origin") or ""
    text = origin.strip()
    if not text:
        return None
    lead = len(origin) - len(origin.lstrip())                 # 앞 공백만큼 offset이 밀린다
    ents_full = [e for e in row.get("entity_list", []) if e.get("label") in NER_LABELS]
    ents = [{"text": e["text"], "label": e["label"]} for e in ents_full]
    answer = json.dumps(ents, ensure_ascii=False)
    # BERT 기준선용 글자 단위 BIO 태그. 원본의 character_list([글자, 태그])가 있으면 그대로 쓰고,
    # 없으면 offset으로 만든다.
    char_labels = None
    cl = row.get("character_list")
    if cl and len(cl) == len(origin):
        char_labels = [t if t.split("-")[-1] in NER_LABELS or t == "O" else "O" for _, t in cl]
        char_labels = char_labels[lead: lead + len(text)]
    if char_labels is None:
        char_labels = ["O"] * len(text)
        for e in ents_full:
            off = e.get("offset") or []
            if len(off) == 2:
                s, t = off[0] - lead, off[1] - lead
                if text[s:t] != e["text"] and text[s:t + 1] == e["text"]:
                    t += 1                                    # 끝 위치가 포함형(inclusive)인 경우
                for i in range(s, min(t, len(text))):
                    char_labels[i] = ("B-" if i == s else "I-") + e["label"]
    raw = {"entities": [{"text": e["text"], "label": e["label"], "offset": e.get("offset")} for e in ents_full],
           "char_labels": char_labels}
    return pack("ner", {"text": text}, answer, ents, raw, text)


def make_mrc(row: dict, max_context_chars: int) -> dict | None:
    """KorQuAD: {"context","question","answers":{"text":[...],"answer_start":[...]}} -> 기계독해 지시문"""
    context = (row.get("context") or "").strip()
    question = (row.get("question") or "").strip()
    answers = row.get("answers") or {}
    texts = answers.get("text") or []
    if not context or not question or not texts:
        return None
    if len(context) > max_context_chars:
        # 지문이 너무 길면 학습 시간이 급격히 늘어난다. 강의 시간 안에 1 epoch를 끝내는 게 우선.
        return None
    answer = texts[0].strip()
    if not answer:
        return None
    starts = answers.get("answer_start") or []
    raw = {"answer_text": texts, "answer_start": starts}
    return pack("mrc", {"context": context, "question": question}, answer, texts, raw,
                context + "||" + question)


def make_tc(row: dict) -> dict | None:
    """KLUE-YNAT: {"title","label"} -> 뉴스 주제분류 지시문"""
    text = (row.get("title") or "").strip()
    if not text:
        return None
    label = int(row["label"])
    answer = TC_LABELS[label]
    return pack("tc", {"text": text}, answer, answer, {"label": label}, text)


def make_nli(row: dict) -> dict | None:
    """KLUE-NLI: {"premise","hypothesis","label"} -> 자연어추론 지시문"""
    premise = (row.get("premise") or "").strip()
    hypothesis = (row.get("hypothesis") or "").strip()
    if not premise or not hypothesis:
        return None
    label = int(row["label"])
    answer = NLI_LABELS[label]
    return pack("nli", {"premise": premise, "hypothesis": hypothesis}, answer, answer,
                {"label": label}, premise + "||" + hypothesis)


def make_sts(row: dict) -> dict | None:
    """KLUE-STS: {"sentence1","sentence2","labels":{"label":float}} -> 문장유사도 지시문"""
    s1 = (row.get("sentence1") or "").strip()
    s2 = (row.get("sentence2") or "").strip()
    if not s1 or not s2:
        return None
    labels = row.get("labels") or {}
    score = float(labels.get("label", labels.get("real-label", 0.0)))
    answer = f"{score:.1f}"
    return pack("sts", {"sentence1": s1, "sentence2": s2}, answer, round(score, 3),
                {"score": score, "binary": int(labels.get("binary-label", 1 if score >= 3 else 0))},
                s1 + "||" + s2)


def make_math(row: dict) -> dict | None:
    """GSM8K-ko: 한국어 풀이 과정이 붙어 있는 초등 수준 문장제.

    학습 타깃에서 계산기 주석(`<<48/2=24>>`)을 지운다. 그대로 두면 모델이
    평가할 때도 그 기호를 그대로 뱉는다.
    """
    q = (row.get("question") or "").strip()
    a = (row.get("answer") or "").strip()
    if not q or "####" not in a:
        return None
    body = re.sub(r"<<[^>]*>>", "", a)
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", body)
    if not m:
        return None
    gold = m.group(1).replace(",", "")
    answer = body[:m.start()].strip() + f"\n#### {gold}"
    return pack("math", {"question": q}, answer, gold, {"answer": a}, norm_key(q))


def make_sql(row: dict, schemas: dict, max_schema_chars: int) -> dict | None:
    """Spider-ko: 한국어 질문 + DB 스키마 -> SQL 한 줄.

    학습/평가에 쓰는 데이터베이스가 서로 겹치지 않는다(train 140개 / validation 20개).
    즉 리키지가 코드가 아니라 데이터 구조로 막혀 있다.
    """
    q = (row.get("question_ko") or "").strip()
    sql = (row.get("query") or "").strip()
    db = row.get("db_id")
    schema = schemas.get(db)
    if not q or not sql or not schema:
        return None
    if len(schema) > max_schema_chars:
        return None
    return pack("sql", {"schema": schema, "question": q}, sql, sql,
                {"db_id": db, "query": sql}, norm_key(db + "||" + q))


# --------------------------------------------------------------------------------------
# KLUE 확장 태스크 원본 확보 — Hugging Face `klue/klue`에서 받아 jsonl로 저장
# --------------------------------------------------------------------------------------

KLUE_CONFIGS = {"tc": "ynat", "nli": "nli", "sts": "sts"}


def ensure_klue(task: str, data: Path) -> tuple[Path, Path] | None:
    cfg = KLUE_CONFIGS[task]
    d = data / f"klue-{cfg}"
    train_p, val_p = d / "train.jsonl", d / "validation.jsonl"
    if train_p.exists() and val_p.exists():
        return train_p, val_p
    try:
        from datasets import load_dataset
        print(f"  [{task}] Hugging Face에서 klue/klue:{cfg} 내려받는 중...")
        ds = load_dataset("klue/klue", cfg)
    except Exception as e:                                   # 오프라인 등
        print(f"  [{task}] 내려받기 실패 — 건너뜁니다 ({str(e)[:80]})")
        return None
    write_jsonl(train_p, ds["train"])
    write_jsonl(val_p, ds["validation"])
    print(f"  [{task}] 저장: {d} (train {len(ds['train'])} / validation {len(ds['validation'])})")
    return train_p, val_p



def ensure_gsm8k(data: Path) -> tuple[Path, Path] | None:
    """수학추론 원본 확보 — kuotient/gsm8k-ko (GSM8K 공식 분할을 그대로 옮긴 한국어판)."""
    d = data / "gsm8k-ko"
    train_p, test_p = d / "train.jsonl", d / "test.jsonl"
    if train_p.exists() and test_p.exists():
        return train_p, test_p
    try:
        from datasets import load_dataset
        print("  [math] Hugging Face에서 kuotient/gsm8k-ko 내려받는 중...")
        ds = load_dataset("kuotient/gsm8k-ko")
    except Exception as e:
        print(f"  [math] 내려받기 실패 — 건너뜁니다 ({str(e)[:80]})")
        return None
    write_jsonl(train_p, ds["train"])
    write_jsonl(test_p, ds["test"])
    print(f"  [math] 저장: {d} (train {len(ds['train'])} / test {len(ds['test'])})")
    return train_p, test_p


def ensure_spider_ko(data: Path) -> tuple[Path, Path, dict] | None:
    """SQL생성 원본 확보 — huggingface-KREW/spider-ko + richardr1126/spider-schema."""
    d = data / "spider-ko"
    train_p, val_p, sch_p = d / "train.jsonl", d / "validation.jsonl", d / "schema.json"
    if not (train_p.exists() and val_p.exists() and sch_p.exists()):
        try:
            from datasets import load_dataset
            from huggingface_hub import hf_hub_download
            print("  [sql] Hugging Face에서 spider-ko / spider-schema 내려받는 중...")
            ds = load_dataset("huggingface-KREW/spider-ko")
            raw = hf_hub_download("richardr1126/spider-schema",
                                  "spider_schema_rows_v2.json", repo_type="dataset")
        except Exception as e:
            print(f"  [sql] 내려받기 실패 — 건너뜁니다 ({str(e)[:80]})")
            return None
        rows = json.loads(Path(raw).read_text(encoding="utf-8"))
        schemas = {r["db_id"]: r["Schema (values (type))"] for r in rows}
        write_jsonl(train_p, ds["train"])
        write_jsonl(val_p, ds["validation"])
        sch_p.write_text(json.dumps(schemas, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [sql] 저장: {d} (train {len(ds['train'])} / validation {len(ds['validation'])}"
              f" / 스키마 {len(schemas)}개 DB)")
    schemas = json.loads(sch_p.read_text(encoding="utf-8"))
    return train_p, val_p, schemas


# --------------------------------------------------------------------------------------

def build(args):
    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    specs = [
        # (task, 학습 원본, 평가 원본, 변환기, 학습 개수, 평가 개수)
        ("cls", data / "nsmc/train.jsonl", data / "nsmc/test.jsonl", make_cls, args.n_cls, args.n_eval_cls),
        ("ner", data / "klue-ner/train.jsonl", data / "klue-ner/test.jsonl", make_ner, args.n_ner, args.n_eval_ner),
        ("mrc", data / "korquad/train.jsonl", data / "korquad/validation.jsonl",
         lambda r: make_mrc(r, args.max_context_chars), args.n_mrc, args.n_eval_mrc),
    ]
    # 확장 태스크는 핵심 3태스크 **뒤에** 붙인다. 난수 소비 순서가 같아야
    # 확장 태스크를 켜고 꺼도 핵심 3태스크의 평가셋이 바뀌지 않는다.
    if not args.no_ext:
        conv = {"tc": make_tc, "nli": make_nli, "sts": make_sts}
        n_train = {"tc": args.n_tc, "nli": args.n_nli, "sts": args.n_sts}
        n_eval = {"tc": args.n_eval_tc, "nli": args.n_eval_nli, "sts": args.n_eval_sts}
        for task in EXT_TASKS:
            paths = ensure_klue(task, data)
            if paths:
                specs.append((task, paths[0], paths[1], conv[task], n_train[task], n_eval[task]))
    # 수학추론은 맨 뒤에 붙인다 — 앞 태스크들의 난수 소비 순서를 건드리지 않으므로
    # 이 태스크를 켜고 꺼도 기존 평가셋(그리고 이미 측정해 둔 BERT 기준선)이 그대로 유효하다.
    if args.n_math > 0:
        paths = ensure_gsm8k(data)
        if paths:
            specs.append(("math", paths[0], paths[1], make_math, args.n_math, args.n_eval_math))
    if args.n_sql > 0:
        sp = ensure_spider_ko(data)
        if sp:
            train_p, val_p, schemas = sp
            specs.append(("sql", train_p, val_p,
                          lambda r: make_sql(r, schemas, args.max_schema_chars),
                          args.n_sql, args.n_eval_sql))

    train_by_task = {}
    stats = {}

    for task, train_path, eval_path, conv, n_train, n_eval in specs:
        # --- 평가셋을 먼저 만든다: 공식 test/validation 파일에서만 뽑는다 ---
        # 평가셋은 절대 건드리지 않는다. 대신 학습셋에서 겹치는 것을 빼는 방향으로 리키지를 없앤다.
        raw_eval = read_jsonl(eval_path, limit=args.scan_limit)
        rng.shuffle(raw_eval)
        eval_rows = []
        for r in raw_eval:
            ex = conv(r)
            if ex:
                eval_rows.append(ex)
            if len(eval_rows) >= n_eval:
                break
        eval_keys = {e["key"] for e in eval_rows}

        # --- 학습셋: 공식 train 파일에서만 뽑되, 평가셋과 겹치는 입력은 제외한다 ---
        # NSMC처럼 짧은 리뷰("굳 ㅋ" 등)는 train과 test에 같은 문장이 실제로 들어있다.
        # 이런 것을 그대로 학습하면 평가 점수가 부풀려진다.
        raw_train = read_jsonl(train_path, limit=args.scan_limit)
        rng.shuffle(raw_train)
        train_rows = []
        dropped_by_leak = 0
        for r in raw_train:
            ex = conv(r)
            if not ex:
                continue
            if ex["key"] in eval_keys:
                dropped_by_leak += 1
                continue
            train_rows.append(ex)
            if len(train_rows) >= n_train:
                break

        # --- 최종 검증: 겹치면 그대로 중단한다 ---
        train_keys = {e["key"] for e in train_rows}
        overlap = train_keys & eval_keys
        if overlap:
            raise SystemExit(
                f"[리키지 발견] {task}: 제거 후에도 {len(overlap)}건이 겹칩니다. 중단합니다."
            )

        write_jsonl(out / f"eval_{task}.jsonl", eval_rows)
        train_by_task[task] = train_rows
        stats[task] = {
            "name": TASKS[task]["name"], "dataset": TASKS[task]["dataset"],
            "train": len(train_rows), "eval": len(eval_rows),
            "train_src": str(train_path), "eval_src": str(eval_path),
            "dropped_by_leak": dropped_by_leak,
        }
        print(f"  {task:3s} {TASKS[task]['name']:6s}: 학습 {len(train_rows)}건 / 평가 {len(eval_rows)}건"
              f"  (평가셋과 겹쳐 학습에서 제외한 것 {dropped_by_leak}건, 최종 리키지 0건 확인)")

    # 태스크를 섞는다 — 한 태스크만 몰아서 학습하면 뒤쪽 태스크로 치우친다.
    def mixed(tasks):
        rows = [r for t in tasks if t in train_by_task for r in train_by_task[t]]
        random.Random(args.seed + 1).shuffle(rows)
        return rows

    core = mixed(CORE_TASKS)
    write_jsonl(out / "train.jsonl", core)
    ext_tasks = [t for t in EXT_TASKS if t in train_by_task]
    all_rows = mixed(CORE_TASKS + ext_tasks) if ext_tasks else core
    if ext_tasks:
        write_jsonl(out / "train_all.jsonl", all_rows)
    # 2026-09 개편 학습셋 — BERT와 같은 네 태스크 + 수학추론
    main_tasks = [t for t in MAIN_TASKS if t in train_by_task]
    main_rows = mixed(main_tasks)
    write_jsonl(out / "train_main.jsonl", main_rows)

    (out / "stats.json").write_text(
        json.dumps({"total_train": len(core), "total_train_all": len(all_rows),
                    "total_train_main": len(main_rows), "main_tasks": main_tasks,
                    "core_tasks": CORE_TASKS, "ext_tasks": ext_tasks,
                    "per_task": stats, "seed": args.seed}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n핵심 학습셋  : {out}/train.jsonl ({len(core)}건, {'+'.join(CORE_TASKS)})")
    if ext_tasks:
        print(f"확장 학습셋  : {out}/train_all.jsonl ({len(all_rows)}건, {'+'.join(CORE_TASKS + ext_tasks)})")
    print(f"주 학습셋    : {out}/train_main.jsonl ({len(main_rows)}건, {'+'.join(main_tasks)})")
    print(f"평가셋       : {out}/eval_<task>.jsonl  ({', '.join(stats)})")
    print("모든 평가셋은 공식 test/validation 파일에서만 만들었고, 학습셋과 겹치지 않음을 확인했습니다.")


def main():
    p = argparse.ArgumentParser(description="여러 태스크를 하나의 instruction 데이터셋으로 합칩니다.")
    p.add_argument("--data", default="data", help="원본 데이터 폴더")
    p.add_argument("--out", default="data/llm-ft", help="출력 폴더")
    p.add_argument("--n-cls", type=int, default=1200, help="감성분류 학습 예제 수")
    p.add_argument("--n-ner", type=int, default=1000, help="개체명인식 학습 예제 수")
    p.add_argument("--n-mrc", type=int, default=800, help="기계독해 학습 예제 수")
    p.add_argument("--n-tc", type=int, default=800, help="주제분류 학습 예제 수 (확장)")
    p.add_argument("--n-nli", type=int, default=800, help="자연어추론 학습 예제 수 (확장)")
    p.add_argument("--n-sts", type=int, default=600, help="문장유사도 학습 예제 수 (확장)")
    p.add_argument("--n-eval-cls", type=int, default=500, help="감성분류 평가 예제 수")
    p.add_argument("--n-eval-ner", type=int, default=300, help="개체명인식 평가 예제 수")
    p.add_argument("--n-eval-mrc", type=int, default=300, help="기계독해 평가 예제 수")
    p.add_argument("--n-eval-tc", type=int, default=300)
    p.add_argument("--n-eval-nli", type=int, default=300)
    p.add_argument("--n-eval-sts", type=int, default=300)
    p.add_argument("--n-math", type=int, default=800, help="수학추론 학습 예제 수 (0이면 만들지 않음)")
    p.add_argument("--n-eval-math", type=int, default=300, help="수학추론 평가 예제 수")
    p.add_argument("--n-sql", type=int, default=800, help="SQL생성 학습 예제 수 (0이면 만들지 않음)")
    p.add_argument("--n-eval-sql", type=int, default=300, help="SQL생성 평가 예제 수")
    p.add_argument("--max-schema-chars", type=int, default=2500, help="SQL생성 스키마 최대 길이(글자)")
    p.add_argument("--max-context-chars", type=int, default=800, help="기계독해 지문 최대 길이(글자)")
    p.add_argument("--scan-limit", type=int, default=30000, help="원본에서 읽어올 최대 줄 수")
    p.add_argument("--no-ext", action="store_true", help="확장 태스크(KLUE tc/nli/sts)를 만들지 않음")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print("통합 instruction 데이터셋을 만듭니다.")
    build(args)


if __name__ == "__main__":
    main()

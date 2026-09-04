"""
[실습5] 결과 비교 도우미 — results/ 폴더의 JSON을 pandas 표로 모은다.

GPU가 없어도 된다. 강사가 미리 돌려둔 스윕 결과(results/sweep, results/bert)를 읽어서
  - LLM 학습 전/후 점수 (모델 × 태스크)
  - BERT 기준선 점수 (모델 × 태스크 × 학습 예산)
  - 학습 시간·GPU 메모리·학습 파라미터 수
를 한눈에 보이게 만든다. `bert_vs_llm.ipynb` 와 `day2/06_LLM파인튜닝.ipynb` 가 이 모듈을 쓴다.

    from compare import load_sweep, load_bert, MAIN_METRIC
    llm = load_sweep("results")      # 한 줄 = (모델, 태스크)
    bert = load_bert("results")      # 한 줄 = (모델, 태스크, 예산)
"""

import json
import sys
from pathlib import Path

import pandas as pd

# 태스크 이름·데이터·대표 지표. common.TASKS 와 같은 내용이지만, 이 모듈은 GPU·torch 없이도
# 돌아가야 하므로(결과 JSON만 읽는다) 여기에 따로 적어 둔다.
# 2026-09 개편: BERT와 정면으로 겨루는 네 태스크 + 생성 모델만 할 수 있는 두 태스크.
# (NSMC 감성분류는 상향평준화되어, KLUE-STS는 자연어추론과 성격이 겹쳐 뺐다.
#  두 태스크의 옛 결과는 results/sweep/ 에 그대로 남아 있고 아래 이름표도 유지한다.)
THREE_WAY_TASKS = ["tc", "ner", "mrc", "sts"]   # BERT · T5 · GPT 계열 세 방식 모두 가능
GEN_ONLY_TASKS = ["sql", "math"]                # 생성 모델(T5 · LLM)만 가능 — BERT 구조로는 불가
BONUS_TASKS = ["math"]                          # 정규 5태스크 뒤에 시간이 남을 때 다루는 +1
SHARED_TASKS = THREE_WAY_TASKS
LLM_ONLY_TASKS = GEN_ONLY_TASKS                 # (옛 이름 유지)
TASK_ORDER = THREE_WAY_TASKS + GEN_ONLY_TASKS
LEGACY_TASKS = ["cls", "nli"]                   # 옛 스윕 결과를 읽을 때만 쓴다
TASK_NAME = {"cls": "감성분류", "ner": "개체명인식", "mrc": "기계독해",
             "tc": "주제분류", "nli": "자연어추론", "sts": "문장유사도",
             "math": "수학추론", "sql": "SQL생성"}
TASK_DATA = {"cls": "NSMC", "ner": "KLUE-NER", "mrc": "KorQuAD",
             "tc": "KLUE-YNAT", "nli": "KLUE-NLI", "sts": "KLUE-STS",
             "math": "GSM8K-ko", "sql": "Spider-ko"}
MAIN_METRIC = {"cls": "accuracy", "ner": "f1", "mrc": "em",
               "tc": "accuracy", "nli": "accuracy", "sts": "pearson",
               "math": "em", "sql": "em"}                              # 각 데이터셋의 공식 지표
METRIC_KO = {"accuracy": "정확도", "f1": "F1", "em": "EM", "pearson": "피어슨 상관"}
TASK_LABEL = {t: f"{TASK_NAME[t]}\n{TASK_DATA[t]} · {METRIC_KO[MAIN_METRIC[t]]}"
              for t in TASK_ORDER + LEGACY_TASKS}

# 스윕에 쓴 모델들. tag = results/sweep/<tag>-*.json 의 <tag>
#   size_b : 대략의 파라미터 수(십억) — 표시용. 정확한 값은 train_meta.json 의 total_params
#   kind   : Instruct(대화형으로 후학습된 모델) / Base(사전학습만 한 모델)
MODEL_INFO = {
    "qwen35-0.8b":            dict(name="Qwen3.5-0.8B",            org="Alibaba",  kind="Instruct", size_b=0.8),
    "qwen35-0.8b-base":       dict(name="Qwen3.5-0.8B-Base",       org="Alibaba",  kind="Base",     size_b=0.8),
    "exaone4-1.2b":           dict(name="EXAONE-4.0-1.2B",         org="LG",       kind="Instruct", size_b=1.2),
    "qwen35-2b":              dict(name="Qwen3.5-2B",              org="Alibaba",  kind="Instruct", size_b=2.0),
    "qwen35-2b-base":         dict(name="Qwen3.5-2B-Base",         org="Alibaba",  kind="Base",     size_b=2.0),
    "kanana15-2.1b-instruct": dict(name="kanana-1.5-2.1B-instruct", org="Kakao",   kind="Instruct", size_b=2.1),
    "kanana15-2.1b-base":     dict(name="kanana-1.5-2.1B-base",    org="Kakao",    kind="Base",     size_b=2.1),
    "midm2-mini":             dict(name="Mi:dm-2.0-Mini (2.3B)",   org="KT",       kind="Instruct", size_b=2.3),
    "gemma4-e2b-it":          dict(name="Gemma-4-E2B-it",          org="Google",   kind="Instruct", size_b=5.1),
    "qwen35-4b":              dict(name="Qwen3.5-4B",              org="Alibaba",  kind="Instruct", size_b=4.0),
    "gemma4-e4b-it":          dict(name="Gemma-4-E4B-it",          org="Google",   kind="Instruct", size_b=8.0),
    "ax4-light-7b":           dict(name="A.X-4.0-Light (7B)",      org="SKT",      kind="Instruct", size_b=7.2),
    "kanana15-8b-instruct":   dict(name="kanana-1.5-8B-instruct",  org="Kakao",    kind="Instruct", size_b=8.0),
    # ---- 2026-09 개편 라인업 (results/sweep2/) — 기관·설계가 겹치지 않게 고른 8종 + Base 대조 1종 ----
    "llama32-3b":             dict(name="Llama-3.2-3B-Instruct",   org="Meta",     kind="Instruct", size_b=3.2),
    "bllossom-3b":            dict(name="Bllossom-3B (한국어 추가학습)", org="서울과기대", kind="Instruct", size_b=3.2),
    "ministral3-3b":          dict(name="Ministral-3-3B-Instruct", org="Mistral",  kind="Instruct", size_b=3.9),
    "gptoss-20b":             dict(name="gpt-oss-20B (MoE)",       org="OpenAI",   kind="Instruct", size_b=20.9),
    # 에폭 실험 — 같은 모델을 1/2/3에폭으로만 다르게 학습했다
    "exaone4-1.2b-2ep":       dict(name="EXAONE-4.0-1.2B (2에폭)",  org="LG",       kind="Instruct", size_b=1.2),
    "exaone4-1.2b-3ep":       dict(name="EXAONE-4.0-1.2B (3에폭)",  org="LG",       kind="Instruct", size_b=1.2),
    "exaone4-1.2b-4bit":      dict(name="EXAONE-4.0-1.2B (QLoRA 4bit)", org="LG",   kind="Instruct", size_b=1.2),
    # results/ablation/ — kanana 2.1B에서 "대화 형식 × 손실 범위"를 바꿔 본 대조 실험 (bert_vs_llm.ipynb §2-1)
    "kanana15-2.1b-instruct-plain": dict(name="kanana-1.5-2.1B-instruct (단순 형식)", org="Kakao", kind="Instruct", size_b=2.1),
    "kanana15-2.1b-base-fullloss":  dict(name="kanana-1.5-2.1B-base (전체 손실)",    org="Kakao", kind="Base",     size_b=2.1),
    "exaone4-1.2b-plain":           dict(name="EXAONE-4.0-1.2B (단순 형식)",          org="LG",     kind="Instruct", size_b=1.2),
    "gemma4-e2b-it-plain":          dict(name="Gemma-4-E2B-it (단순 형식)",           org="Google", kind="Instruct", size_b=5.1),
}

# ---------------------------------------------------------------------------------------------
# 2026-09 라인업(results/sweep2/) — 강의에서 소개한 대표 모델 계열에서 "서로 구별되는" 것만 골랐다.
#   group : 한국어 특화(국내 기관이 한국어를 주 언어로 만든 것) / 글로벌(다국어 범용)
#   why   : 왜 이 모델이 표에 있는가 — 라인업 표와 강의 수치 문서(docs/LECTURE-FACTS.md)에 그대로 나간다
# 순서 = 표에 나오는 순서(한국어 특화 → 글로벌, 각 묶음 안에서는 크기 순).
LINEUP = [
    "ax4-light-7b", "exaone4-1.2b", "kanana15-2.1b-base", "kanana15-2.1b-instruct", "bllossom-3b",
    "qwen35-4b", "qwen35-2b", "llama32-3b", "gemma4-e2b-it", "ministral3-3b", "gptoss-20b",
]
LINEUP_GROUP = {
    "ax4-light-7b": "한국어 특화", "qwen35-4b": "글로벌",
    "exaone4-1.2b": "한국어 특화", "kanana15-2.1b-base": "한국어 특화", "kanana15-2.1b-instruct": "한국어 특화",
    "bllossom-3b": "한국어 특화",
    "qwen35-2b": "글로벌", "llama32-3b": "글로벌", "gemma4-e2b-it": "글로벌", "ministral3-3b": "글로벌",
    "gptoss-20b": "글로벌",
}
LINEUP_WHY = {
    "ax4-light-7b":           "SKT. **수업에서 직접 학습하는 모델** — 강의장 GPU(24GB)에 들어가는 크기 중 위쪽이고, "
                              "파인튜닝 뒤에도 수학추론이 떨어지지 않는다 (학습 20분, GPU 17GB)",
    "exaone4-1.2b":           "국내 대표 소형 모델. 라인업에서 가장 작다 — 크기 축의 아래쪽 (학습 10분, GPU 4.3GB)",
    "kanana15-2.1b-base":     "kanana-instruct와 **같은 몸, 후학습만 없는 Base** — 지시 따르기를 안 배운 모델이 파인튜닝 뒤 얼마나 따라오나",
    "kanana15-2.1b-instruct": "카카오. 같은 크기대(2B)에서 EXAONE·Qwen과 견주는 두 번째 국내 모델",
    "bllossom-3b":            "Llama-3.2-3B에 **한국어를 추가학습**한 모델 — 바로 아래 원본 Llama와 짝을 이뤄 '한국어 추가학습의 효과'를 본다",
    "qwen35-4b":              "같은 계열의 4B — 라인업에서 파인튜닝 뒤 평균이 가장 높다. 다만 학습이 45분 걸려 수업 기본 모델로는 무겁다",
    "qwen35-2b":              "오픈웨이트 성능 선두 계열(Alibaba). 글로벌 모델이 한국어 태스크에서 어디까지 오나의 기준",
    "llama32-3b":             "Meta의 대표 소형 모델이자 Bllossom의 원본. 한국어 학습 전 상태",
    "gemma4-e2b-it":          "Google. 파라미터 5.1B이지만 임베딩을 빼면 2B급으로 동작(E2B) — 크기 표기를 그대로 믿으면 안 되는 예",
    "ministral3-3b":          "Mistral(유럽). 미국·중국·한국 밖의 대표 계열",
    "gptoss-20b":             "OpenAI의 오픈웨이트 **MoE 20B** — 크기 축의 끝. 21B 중 3.6B만 토큰마다 켜진다. 학습에는 A6000 한 장이 한계",
}
# 짝지어 읽을 대조 — (라벨, tag A, tag B, 무엇이 다른가)
CONTRASTS = [
    ("한국어 추가학습", "llama32-3b", "bllossom-3b", "같은 Llama-3.2-3B 몸에 한국어 데이터를 더 학습시킨 것(Bllossom)이 원본과 어떻게 다른가"),
    ("Base ↔ Instruct", "kanana15-2.1b-base", "kanana15-2.1b-instruct", "같은 kanana 2.1B에서 후학습(지시 따르기)이 있고 없고의 차이 — 파인튜닝 전과 후에 각각"),
    ("1.2B ↔ 20B", "exaone4-1.2b", "gptoss-20b", "가장 작은 모델과 가장 큰 모델 — 크기가 17배면 점수도 그만큼 오르나"),
    ("1.2B ↔ 7.3B", "exaone4-1.2b", "ax4-light-7b", "수업 모델을 6배로 키우면 무엇이 달라지나 — 특히 파인튜닝 뒤 수학추론"),
]
# 거론됐지만 라인업에 넣지 않은 모델 — (이름, 이유). 표 아래 각주로 나간다.
NOT_IN_LINEUP = [
    ("KULLM(구름) 3", "고려대 NLP&AI 연구실. 현행 KULLM3는 SOLAR-10.7B 기반 10.7B로 소형 라인업(1~4B)과 크기대가 다르고, "
                    "24GB 강의장 GPU에서는 QLoRA로도 학습이 빠듯하다. 한국어 추가학습 계열은 Bllossom(3B)이 대표한다"),
    ("HyperCLOVA X SEED 1.5B", "네이버. gated 저장소(약관 동의 필요)라 수강생 환경에서 바로 받을 수 없다"),
    ("Mi:dm 2.0 Mini 2.3B", "KT. 이전 구성(감성분류·자연어추론이 든 6태스크 스윕, `results/sweep/`)에서 돌렸으나 "
                    "kanana·EXAONE과 포지셔닝이 겹쳐 뺐다. 수치는 옛 스윕 표에 남아 있다"),
    ("Gemma-4-E4B-it · Llama-3.1-8B · kanana-1.5-8B 등 8B급", "학습에 20GB 안팎이 들어 24GB 강의장 GPU에 여유가 없다. "
                    "Gemma-4-E4B-it 는 점수가 높았지만 파인튜닝 뒤 수학추론이 71.0 → 60.3 으로 크게 떨어졌다"),
]

T5_INFO = {
    "pko-t5-small": dict(name="pko-T5-small", size_m=95),
    "pko-t5-base": dict(name="pko-T5-base", size_m=276),
    "pko-t5-large": dict(name="pko-T5-large", size_m=821),
    "ke-t5-base": dict(name="KE-T5-base", size_m=276),
}

BERT_INFO = {
    "roberta-small": dict(name="KLUE-RoBERTa-small", size_m=68),
    "bert-base": dict(name="KLUE-BERT-base", size_m=111),
    "roberta-base": dict(name="KLUE-RoBERTa-base", size_m=111),
    "koelectra-base-v3-discriminator": dict(name="KoELECTRA-base-v3", size_m=112),
    "roberta-large": dict(name="KLUE-RoBERTa-large", size_m=337),
}


def _read(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_sweep(res_dir: str = "results", sub: str = "sweep") -> pd.DataFrame:
    """results/sweep/<tag>-{before,after,train_meta}.json → 한 줄 = (모델, 태스크)."""
    d = Path(res_dir) / sub
    rows = []
    # 학습 전 결과만 있는 모델(학습 중·학습 실패)도 한 줄로 보이게 세 파일의 태그를 합친다.
    tags = sorted({p.name.rsplit("-", 1)[0] for suf in ("before", "after", "train_meta")
                   for p in d.glob(f"*-{suf}.json")})
    for tag in tags:
        meta = _read(d / f"{tag}-train_meta.json") or {}
        before = (_read(d / f"{tag}-before.json") or {}).get("results", {})
        after = (_read(d / f"{tag}-after.json") or {}).get("results", {})
        info = MODEL_INFO.get(tag, dict(name=tag, org="?", kind="?", size_b=None))
        for task in TASK_ORDER:
            m = MAIN_METRIC[task]
            b, a = before.get(task), after.get(task)
            rows.append({
                "tag": tag, "model": info["name"], "org": info["org"], "kind": info["kind"],
                "params_b": round(meta.get("total_params", 0) / 1e9, 2) or info["size_b"],
                "task": task, "task_name": TASK_NAME[task], "metric": m,
                "before": b.get(m) if b else None,
                "after": a.get(m) if a else None,
                "before_sec": b.get("sec") if b else None,
                "after_sec": a.get("sec") if a else None,
                "broken_before": b.get("broken_format") if b else None,
                "broken_after": a.get("broken_format") if a else None,
                "n_eval": (a or b or {}).get("n"),
                # 학습 비용 (모델당 한 번 — 6태스크 통합 학습이므로 태스크마다 같은 값)
                "train_min": round(meta.get("train_seconds", 0) / 60, 1) if meta else None,
                "peak_gpu_gb": meta.get("peak_gpu_gb"),
                "trainable_m": round(meta.get("trainable_params", 0) / 1e6, 1) if meta else None,
                "trainable_pct": meta.get("trainable_pct"),
                "n_train": meta.get("n_train"),
                "loss_mode": meta.get("loss_mode"),
                "template_fix": (meta.get("template_fix") or {}).get("action"),
                # 실제로 쓴 대화 형식: 모델 고유(native) / 우리가 붙인 단순 형식(plain)
                "template": "plain" if (meta.get("chat_template") == "plain"
                                        or (meta.get("template_fix") or {}).get("action") == "plain") else "native",
            })
    df = pd.DataFrame(rows)
    if len(df):
        df["delta"] = df["after"] - df["before"]
    return df


def load_bert(res_dir: str = "results", sub: str = "bert") -> pd.DataFrame:
    """results/bert/<model>-<task>-<mode>.json → 한 줄 = (모델, 태스크, 예산)."""
    d = Path(res_dir) / sub
    rows = []
    for p in sorted(d.glob("*.json")):
        r = _read(p)
        if not r or r.get("kind") != "bert":
            continue
        task, mode = r["task"], r["mode"]
        short = r["model"].split("/")[-1]
        info = BERT_INFO.get(short, dict(name=short, size_m=None))
        s = r["results"][task]
        tr = r.get("train", {})
        rows.append({
            "model": info["name"], "short": short, "task": task, "task_name": TASK_NAME[task],
            "mode": mode, "metric": MAIN_METRIC[task], "score": s.get(MAIN_METRIC[task]),
            "eval_sec": s.get("sec"), "n_eval": s.get("n"),
            "n_train": tr.get("n_train"), "epochs": tr.get("epochs"),
            "train_min": round(tr.get("train_seconds", 0) / 60, 2), "peak_gpu_gb": tr.get("peak_gpu_gb"),
            "params_m": round(tr.get("total_params", 0) / 1e6, 1),
        })
    return pd.DataFrame(rows)


def load_t5(res_dir: str = "results", sub: str = "t5") -> pd.DataFrame:
    """results/t5/<model>-<task>-<mode>.json → 한 줄 = (모델, 태스크, 예산).

    T5는 BERT와 같은 두 모드(budget/full)로 학습하지만, 답을 **글자로 생성**하므로
    LLM과 똑같은 채점 함수를 쓴다. 그래서 수학추론·SQL생성도 점수가 나온다.
    """
    d = Path(res_dir) / sub
    rows = []
    for p_ in sorted(d.glob("*.json")):
        r = _read(p_)
        if not r or r.get("family") != "t5":
            continue
        task, mode = r["task"], r["mode"]
        short = r["model"].split("/")[-1]
        s_ = r["summary"]
        rows.append({
            "model": T5_INFO.get(short, {}).get("name", short), "short": short,
            "family": "T5", "task": task, "task_name": TASK_NAME[task], "mode": mode,
            "metric": MAIN_METRIC[task], "score": s_.get(MAIN_METRIC[task]),
            "broken_format": s_.get("broken_format"),
            "eval_sec": r.get("eval_seconds"), "n_eval": r.get("n_eval"),
            "n_train": r.get("n_train"), "epochs": r.get("epochs"),
            "train_min": round(r.get("train_seconds", 0) / 60, 2), "peak_gpu_gb": r.get("peak_gpu_gb"),
            "params_m": round(r.get("total_params", 0) / 1e6, 1),
        })
    return pd.DataFrame(rows)


def load_baselines(res_dir: str = "results") -> pd.DataFrame:
    """BERT(인코더)와 T5(인코더-디코더) 기준선을 한 표로 쌓는다."""
    b = load_bert(res_dir)
    if len(b):
        b = b.assign(family="BERT")
    t = load_t5(res_dir)
    return pd.concat([b, t], ignore_index=True) if len(b) or len(t) else pd.DataFrame()


def pivot_scores(df: pd.DataFrame, value: str = "after", index: str = "model") -> pd.DataFrame:
    """(모델 × 태스크) 표. 열 순서는 TASK_ORDER, 마지막에 평균."""
    t = df.pivot_table(index=index, columns="task", values=value, aggfunc="first")
    t = t.reindex(columns=[c for c in TASK_ORDER if c in t.columns])
    t.columns = [TASK_NAME[c] for c in t.columns]
    t["평균"] = t.mean(axis=1)
    return t.round(1)


def setup_korean_font():
    """matplotlib 에서 한글이 깨지지 않게 설치된 한글 글꼴을 하나 고른다."""
    import matplotlib
    from matplotlib import font_manager
    candidates = ["NanumGothic", "NanumSquare", "Noto Sans CJK KR", "Noto Sans KR", "AppleGothic",
                  "Apple SD Gothic Neo", "Malgun Gothic", "NanumBarunGothic", "Noto Sans CJK TC"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            matplotlib.rcParams["font.family"] = name
            break
    else:
        print("[알림] 한글 글꼴을 찾지 못했습니다. 그래프의 한글이 □로 보일 수 있습니다. (예: sudo apt install fonts-nanum)")
    matplotlib.rcParams["axes.unicode_minus"] = False
    return matplotlib.rcParams["font.family"]


if __name__ == "__main__":
    res = sys.argv[1] if len(sys.argv) > 1 else "results"
    llm = load_sweep(res)
    pd.set_option("display.width", 200)
    print("=== LLM 학습 후 점수 ===")
    print(pivot_scores(llm, "after"))
    print("\n=== LLM 학습 전 점수 ===")
    print(pivot_scores(llm, "before"))
    bert = load_bert(res)
    if len(bert):
        for mode in ["budget", "full"]:
            print(f"\n=== BERT 기준선 ({mode}) ===")
            print(pivot_scores(bert[bert["mode"] == mode], "score"))

"""
[실습4A·4B] 기계독해 — 내가 학습시킨 모델에게 직접 물어보는 데모 페이지

노트북에서 학습을 끝내고 `L.save_model(...)` 로 저장했다면, 그 모델을 그대로 불러
브라우저에서 지문과 질문을 넣어 볼 수 있다. Flask 로 로컬에만 페이지를 띄우므로
외부 서비스도 인터넷도 필요 없다.

    python day2/serve_mrc.py --kind bert            # 실습4A — 추출형(BERT)
    python day2/serve_mrc.py --kind t5              # 실습4B — 생성형(T5)
    python day2/serve_mrc.py --kind both            # 둘을 나란히 (같은 질문에 두 답)
    python day2/serve_mrc.py --kind bert --port 9007

    # 브라우저에서 http://localhost:9006

**둘을 나란히 놓는 것이 이 실습의 요점이다.** 같은 지문·같은 질문에
  추출형은 지문에서 **잘라낸** 답을 내고(지문 밖 답을 원리상 낼 수 없다),
  생성형은 답을 **써낸다**(지문에 없는 말도 낼 수 있고, 그래서 지어내기도 한다).

모델은 노트북이 저장한 곳에서 읽는다 — `output/day2/mrc-bert/` · `output/day2/mrc-t5/`.
없으면 무엇을 먼저 해야 하는지 알려 주고 끝난다.
"""

import argparse
import sys
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
sys.path.insert(0, str(ROOT / "day1"))          # lab_common 은 day1/ 에 있다
sys.path.insert(0, str(ROOT / "task5-llm-ft"))

import lab_common as L  # noqa: E402

KINDS = ("bert", "t5")
KIND_NAME = {"bert": "추출형 (BERT)", "t5": "생성형 (T5)"}


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 모델 하나
# ══════════════════════════════════════════════════════════════════════════════════════════════

class MrcModel:
    """저장된 기계독해 모델 하나. 추출형과 생성형이 답을 만드는 방식이 다르다."""

    def __init__(self, kind: str, path: Path):
        from transformers import AutoTokenizer
        self.kind = kind
        self.path = path
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"  {KIND_NAME[kind]:14s} 불러오는 중 — {path}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        if kind == "bert":
            from transformers import AutoModelForQuestionAnswering
            self.model = AutoModelForQuestionAnswering.from_pretrained(str(path))
        else:
            from transformers import AutoModelForSeq2SeqLM
            self.model = AutoModelForSeq2SeqLM.from_pretrained(str(path))
        self.model.to(self.device).eval()

    # ── 추출형 — 시작·끝 토큰 위치를 찍고 그 사이를 원문에서 잘라낸다 ────────────────────────
    @torch.no_grad()
    def _answer_bert(self, context: str, question: str) -> dict:
        enc = self.tokenizer(question, context, truncation="only_second", max_length=L.MAX_LEN_MRC,
                             return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")[0].tolist()
        seq_ids = enc.sequence_ids(0)
        # klue/roberta-base 는 문장 구분 임베딩이 하나뿐(type_vocab_size=1)인데 토크나이저는 BERT식으로
        # 지문 쪽에 token_type_ids=1 을 붙인다 → 임베딩 인덱스 초과로 CUDA assert. 학습 코드와 같이 뺀다.
        if getattr(self.model.config, "type_vocab_size", 2) < 2:
            enc.pop("token_type_ids", None)
        out = self.model(**{k: v.to(self.device) for k, v in enc.items()})
        start_logits = out.start_logits[0].float().cpu()
        end_logits = out.end_logits[0].float().cpu()

        # 지문(두 번째 시퀀스) 안에서만 고른다 — 질문 쪽을 답으로 내면 안 된다
        mask = torch.tensor([1.0 if s == 1 else 0.0 for s in seq_ids])
        neg = torch.tensor(-1e9)
        s_scores = torch.where(mask > 0, start_logits, neg)
        e_scores = torch.where(mask > 0, end_logits, neg)

        best = (-1e18, 0, 0)
        top_s = torch.topk(s_scores, k=min(20, len(s_scores))).indices.tolist()
        top_e = torch.topk(e_scores, k=min(20, len(e_scores))).indices.tolist()
        for i in top_s:
            for j in top_e:
                if j < i or j - i + 1 > 30:
                    continue
                sc = float(s_scores[i] + e_scores[j])
                if sc > best[0]:
                    best = (sc, i, j)
        _, i, j = best
        cs, ce = offsets[i][0], offsets[j][1]
        prob = float(torch.softmax(s_scores, 0)[i] * torch.softmax(e_scores, 0)[j])
        return {"answer": context[cs:ce], "score": round(prob * 100, 2), "start": cs, "end": ce}

    # ── 생성형 — 답을 글자로 써낸다. 지문 밖 표현도 나올 수 있다 ──────────────────────────────
    @torch.no_grad()
    def _answer_t5(self, context: str, question: str) -> dict:
        # 학습할 때와 **글자까지 같은** 입력을 만든다. T5 의 입력은 사용자 지시문(messages[1])이다
        # — `lab_common.to_text` 가 학습에서 그렇게 쓴다. 여기서 형식이 어긋나면 점수가 통째로 낮아진다.
        from common import make_messages
        prompt = make_messages("mrc", {"context": context, "question": question})[1]["content"]
        enc = self.tokenizer(prompt, truncation=True, max_length=L.MAX_SOURCE_T5, return_tensors="pt")
        out = self.model.generate(**{k: v.to(self.device) for k, v in enc.items()},
                                  max_new_tokens=48, num_beams=4, do_sample=False)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        inside = text in context           # 지문 안에 그대로 있는 답인가
        return {"answer": text, "score": None, "start": None, "end": None, "in_context": inside}

    def answer(self, context: str, question: str) -> dict:
        r = self._answer_bert(context, question) if self.kind == "bert" else self._answer_t5(context, question)
        r["kind"] = self.kind
        r["kind_name"] = KIND_NAME[self.kind]
        return r


# ══════════════════════════════════════════════════════════════════════════════════════════════

DEFAULT_CONTEXT = (
    "한국전자통신연구원(ETRI)은 1976년에 설립된 대한민국의 정부출연연구기관으로, 대전광역시 유성구에 있다. "
    "정보통신기술 분야를 연구하며 CDMA 이동통신 시스템과 와이브로를 세계 최초로 상용화했다. "
    "2023년 기준 연구 인력은 약 2,300명이고, 한 해 예산은 약 6,700억 원이다."
)
DEFAULT_QUESTION = "ETRI는 어디에 있는가?"


def build_app(models: dict) -> Flask:
    app = Flask(__name__, template_folder=str(ROOT / "templates"))

    @app.route("/")
    def index():
        return render_template("serve_mrc.html",
                               kinds=[{"key": k, "name": KIND_NAME[k]} for k in KINDS if k in models],
                               default_context=DEFAULT_CONTEXT, default_question=DEFAULT_QUESTION)

    @app.route("/api", methods=["POST"])
    def api():
        data = request.get_json(force=True) or {}
        context = (data.get("context") or "").strip()
        question = (data.get("question") or "").strip()
        if not context or not question:
            return jsonify({"error": "지문과 질문을 모두 넣어 주세요."}), 400
        results = []
        for k in KINDS:
            if k in models:
                try:
                    results.append(models[k].answer(context, question))
                except Exception as e:  # noqa: BLE001
                    results.append({"kind": k, "kind_name": KIND_NAME[k], "answer": f"(오류: {e})",
                                    "score": None, "start": None, "end": None})
        return jsonify({"context": context, "question": question, "results": results})

    return app


def main() -> int:
    p = argparse.ArgumentParser(description="기계독해 데모 — 내가 학습시킨 모델에게 물어본다")
    p.add_argument("--kind", choices=["bert", "t5", "both"], default="both",
                   help="bert=실습4A 추출형 · t5=실습4B 생성형 · both=둘을 나란히")
    p.add_argument("--out-dir", default=str(L.OUT_DIR_DAY2), help="노트북이 모델을 저장한 폴더")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9006)
    args = p.parse_args()

    want = KINDS if args.kind == "both" else (args.kind,)
    models, missing = {}, []
    for k in want:
        path = Path(args.out_dir) / f"mrc-{k}"
        if path.is_dir() and any(path.iterdir()):
            models[k] = MrcModel(k, path)
        else:
            missing.append((k, path))

    for k, path in missing:
        nb = "day2/04_기계독해_BERT.ipynb" if k == "bert" else "day2/05_기계독해_T5.ipynb"
        print(f"  [없음] {KIND_NAME[k]} — {path} 가 비어 있습니다.")
        print(f"         {nb} 를 끝까지 실행하면 저장됩니다 (L.save_model 셀).")
    if not models:
        print("\n띄울 모델이 없습니다. 위 노트북을 먼저 실행하세요.")
        return 1

    print(f"\n준비 완료 — http://localhost:{args.port}  (멈추려면 Ctrl+C)")
    build_app(models).run(host=args.host, port=args.port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

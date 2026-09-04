"""
[실습5] LLM 파인튜닝 - 데모 웹 페이지

자기가 학습시킨 모델에 직접 질문해보는 단계다.
task1~task4에서 했던 것처럼 Flask로 로컬에 페이지를 띄운다. (외부 서비스 필요 없음)

이 데모의 특징:
  하나의 모델만 메모리에 올리고, LoRA 어댑터를 껐다 켜면서
  "학습 전"과 "학습 후" 답변을 나란히 보여준다.
  같은 질문에 답이 어떻게 달라지는지가 이 실습의 핵심이다.

사용법:
  python task5-llm-ft/serve.py --adapter output/llm-ft
  python task5-llm-ft/serve.py --model Qwen/Qwen3.5-2B --adapter output/llm-ft-qwen35-2b --port 9005
  python task5-llm-ft/serve.py --adapter output/llm-ft --tasks tc,ner,mrc,sts        # BERT·T5와 겨루는 4개만
  # 브라우저에서 http://localhost:9005 접속
"""

import argparse
import sys
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (MAIN_TASKS, TASK_MAX_TOKENS, TASKS, build_prompt, causal_lm_class,   # noqa: E402
                    check_end_token, extra_load_kwargs, generate_one, load_tokenizer, make_messages, pick_device)


class Demo:
    def __init__(self, model_id: str, adapter: str | None):
        # 학습 때 저장한 토크나이저를 우선 쓴다 — Base 모델이면 우리가 붙인 대화 형식이 거기 들어있다.
        self.tokenizer = load_tokenizer(adapter or model_id)

        print(f"베이스 모델 로딩: {model_id}")
        self.model = causal_lm_class(model_id).from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=pick_device(), **extra_load_kwargs(model_id))
        check_end_token(self.model, self.tokenizer, model_id)   # 학습 때와 같은 템플릿 보정

        self.has_adapter = False
        if adapter:
            from peft import PeftModel
            print(f"LoRA 어댑터 붙이기: {adapter}")
            # merge하지 않고 붙여둔다. 그래야 disable_adapter()로 껐다 켤 수 있다.
            self.model = PeftModel.from_pretrained(self.model, adapter)
            self.has_adapter = True
        self.model.eval()
        print("준비 완료")

    def infer(self, task: str, payload: dict) -> dict:
        prompt = build_prompt(self.tokenizer, make_messages(task, payload))
        n = TASK_MAX_TOKENS.get(task, 64)

        after = generate_one(self.model, self.tokenizer, prompt, n)      # 학습 후 (어댑터 켜짐)

        before = None
        if self.has_adapter:
            # 어댑터만 잠깐 끄면 학습 전 모델이 된다 — 메모리를 두 배로 쓰지 않는다.
            with self.model.disable_adapter():
                before = generate_one(self.model, self.tokenizer, prompt, n)

        return {"task": task, "prompt": prompt, "before": before, "after": after}


# 태스크별로 웹 페이지가 받을 입력 칸 이름 (common.TASKS의 지시문 자리표시자와 같다)
TASK_FIELDS = {
    # 세 방식(BERT·T5·GPT 계열)이 모두 풀 수 있는 네 태스크
    "tc": ["text"], "ner": ["text"],
    "mrc": ["context", "question"], "sts": ["sentence1", "sentence2"],
    # 생성 모델(T5·LLM)만 풀 수 있는 두 태스크 — 정답이 라벨도 지문 속 위치도 아니다
    "math": ["question"],                 # 문제 하나
    "sql": ["schema", "question"],        # DB 스키마 + 질문
    # 옛 구성(감성분류·자연어추론)으로 학습한 어댑터를 띄울 때만 쓰인다
    "cls": ["text"], "nli": ["premise", "hypothesis"],
}


def create_app(demo: Demo, tasks: list[str]) -> Flask:
    app = Flask(__name__, template_folder="../templates")
    task_info = [{"id": t, "name": TASKS[t]["name"], "fields": TASK_FIELDS[t]} for t in tasks]

    @app.route("/")
    def index():
        return render_template("serve_llm.html", has_adapter=demo.has_adapter, tasks=task_info)

    @app.route("/api", methods=["POST"])
    def api():
        body = request.get_json(force=True)
        task = body.get("task") or (tasks[0] if tasks else "tc")
        if task not in tasks:
            return jsonify({"error": f"알 수 없는 태스크: {task}"}), 400
        payload = {k: (body.get(k) or "") for k in TASK_FIELDS[task]}
        return jsonify(demo.infer(task, payload))

    return app


def main():
    p = argparse.ArgumentParser(description="학습한 모델에 직접 질문해보는 데모를 띄웁니다.")
    p.add_argument("--model", default="skt/A.X-4.0-Light", help="베이스 모델")
    p.add_argument("--adapter", default="output/llm-ft", help="LoRA 어댑터 경로 (없으면 학습 전 모델만)")
    p.add_argument("--tasks", default=",".join(MAIN_TASKS),
                   help="페이지에 보일 태스크 (쉼표 구분, 기본 6개: tc,ner,mrc,sts,math,sql)")
    p.add_argument("--port", type=int, default=9005)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()

    tasks = [t for t in args.tasks.split(",") if t in TASKS and t in TASK_FIELDS]
    demo = Demo(args.model, args.adapter if args.adapter and args.adapter.lower() != "none" else None)
    app = create_app(demo, tasks)
    print(f"\n브라우저에서 http://localhost:{args.port} 로 접속하세요.\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

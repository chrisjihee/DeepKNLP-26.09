"""
[실습5] LLM 파인튜닝 - 평가

학습 전(어댑터 없음)과 학습 후(어댑터 있음)를 같은 평가셋·같은 지표로 비교한다.
각 데이터셋이 공식적으로 쓰는 지표를 그대로 쓴다:
  감성분류   (NSMC)      정확도
  개체명인식 (KLUE-NER)  개체 단위 F1 (텍스트와 유형이 모두 맞아야 정답)
  기계독해   (KorQuAD)   EM(완전일치) / 글자 단위 F1
  주제분류   (KLUE-YNAT) 정확도 (+ macro F1)
  자연어추론 (KLUE-NLI)  정확도
  문장유사도 (KLUE-STS)  Pearson 상관 (+ 3점 기준 이진 정확도)

사용법:
  # 학습 전 (원본 모델)
  python task5-llm-ft/evaluate.py --save output/before.json

  # 학습 후 (어댑터 적용)
  python task5-llm-ft/evaluate.py --adapter output/llm-ft --save output/after.json

  # 태스크 골라서, 출력 예시도 보면서
  python task5-llm-ft/evaluate.py --tasks cls,ner --show 5
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TASKS, causal_lm_class, check_end_token, describe, evaluate_tasks, extra_load_kwargs, load_tokenizer, pick_device  # noqa: E402


def load_model(model_name: str, adapter: str | None, device: str | None = None, plain_template: bool = False):
    """평가용 모델을 올린다. 어댑터가 있으면 붙이고, 원본 가중치에 합쳐(merge) 추론 속도를 높인다."""
    device = device or pick_device()                       # cuda → mps(Mac) → cpu
    tokenizer = load_tokenizer(adapter or model_name, plain_template)   # 학습 때 저장한 토크나이저(=템플릿)를 우선 쓴다
    model = causal_lm_class(model_name).from_pretrained(model_name, dtype=torch.bfloat16, device_map=device,
                                                        **extra_load_kwargs(model_name))
    if any(p.device.type == "cpu" for p in model.parameters()):   # dequantize 경로가 일부를 CPU에 남기는 경우
        model = model.to(device)
    check_end_token(model, tokenizer, model_name)          # 학습 때와 같은 템플릿 보정 (어댑터 토크나이저면 no-op)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


def main():
    p = argparse.ArgumentParser(description="학습 전/후 모델을 공식 지표로 평가합니다.")
    p.add_argument("--model", default="skt/A.X-4.0-Light")
    p.add_argument("--adapter", default=None, help="LoRA 어댑터 폴더 (없으면 학습 전 평가)")
    p.add_argument("--data", default="data/llm-ft")
    p.add_argument("--tasks", default=None, help="쉼표 구분 (기본: 평가 파일이 있는 모든 태스크)")
    p.add_argument("--limit", type=int, default=300, help="태스크당 평가 예제 수 (None이면 전체)")
    p.add_argument("--batch-size", type=int, default=8,
               help="한 번에 생성할 예제 수. 7B 모델을 24GB 카드에서 돌리는 기준으로 8. "
                    "메모리가 넉넉하면 16으로 올리면 빨라지고, 모자라면 4로 줄인다")
    p.add_argument("--show", type=int, default=0, help="태스크별로 출력 예시를 몇 개 보여줄지")
    p.add_argument("--save", default=None, help="결과 JSON 저장 경로")
    p.add_argument("--keep-preds", action="store_true", help="예측 문자열도 JSON에 저장")
    p.add_argument("--plain-template", action="store_true", help="[실험] 단순 대화 형식 강제 (train.py --plain-template 와 짝)")
    args = p.parse_args()

    tasks = args.tasks.split(",") if args.tasks else None
    label = f"{args.model}" + (f" + {args.adapter}" if args.adapter else " (학습 전)")
    print(f"평가 대상: {label}\n")

    model, tokenizer = load_model(args.model, args.adapter, plain_template=args.plain_template)
    results = evaluate_tasks(model, tokenizer, args.data, tasks, args.limit, args.batch_size,
                             show=args.show, keep_preds=args.keep_preds)

    if args.save:
        out = {"model": args.model, "adapter": args.adapter, "data": args.data, "results": results}
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n저장: {args.save}")


if __name__ == "__main__":
    main()

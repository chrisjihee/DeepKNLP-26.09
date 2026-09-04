"""
[실습5] LLM 파인튜닝 - LoRA 학습

BERT 실습(task1~task4)에서는 태스크마다 별도의 head를 붙이고 따로 학습시켰다.
여기서는 반대로, 하나의 LLM에 LoRA 어댑터 하나만 붙여서
감성분류 / 개체명인식 / 기계독해 (+ 주제분류 / 자연어추론 / 문장유사도)를 한꺼번에 학습시킨다.

핵심 도구:
  - TRL SFTTrainer : 대화 형식 데이터를 그대로 넣으면 학습해준다 (직접 collate 짤 필요가 없다)
  - PEFT LoRA      : 원래 가중치는 얼려두고 아주 작은 행렬만 학습한다 (메모리·시간 절약)

사용법:
  # 렌더링된 학습 텍스트를 눈으로 먼저 확인 (학습 안 함)
  python task5-llm-ft/train.py --inspect

  # 학습 (기본: A.X-4.0-Light, 6태스크 학습셋, 결과는 output/llm-ft)
  python task5-llm-ft/train.py

  # 다른 모델로
  python task5-llm-ft/train.py --model Qwen/Qwen3.5-2B --out output/llm-ft-qwen35-2b

  # Base 모델(chat template 없음)도 그대로 된다 — 단순 대화 형식을 자동으로 붙인다
  python task5-llm-ft/train.py --model Qwen/Qwen3.5-0.8B-Base --out output/llm-ft-qwen35-08b-base

  # 3태스크(감성·개체명·독해)만 있는 핵심 학습셋으로
  python task5-llm-ft/train.py --data data/llm-ft/train.jsonl

  # 강의장 GPU가 작을 때: 4bit 양자화(QLoRA)
  python task5-llm-ft/train.py --load-4bit --batch-size 1 --grad-accum 8
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (LANGUAGE_ONLY_LINEAR, MULTIMODAL_TYPES, causal_lm_class, extra_load_kwargs,  # noqa: E402
                    ensure_chat_template, check_end_token, build_prompt, render_example,
                    pick_device, device_name, peak_memory_gb)


def load_dataset(path: str, tasks: list[str] | None = None) -> Dataset:
    """통합 instruction 데이터를 읽는다. 각 줄은 {"task", "messages", ...} 형식."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                if tasks and d["task"] not in tasks:
                    continue
                rows.append({"messages": d["messages"], "task": d["task"]})
    return Dataset.from_list(rows)


def build_model(args):
    """모델을 준비한다.

    --load-4bit 를 주면 가중치를 4비트로 압축해서 올린다(QLoRA).
    메모리가 작은 GPU에서 큰 모델을 돌릴 때 쓰는 방법이고, 대신 속도는 조금 느려진다.
    """
    kwargs = {"dtype": torch.bfloat16}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    kwargs.update(extra_load_kwargs(args.model))       # gpt-oss 등 계열별 추가 인자
    if args.attn:
        kwargs["attn_implementation"] = args.attn
    model = causal_lm_class(args.model).from_pretrained(args.model, **kwargs)   # 멀티모달 계열은 다른 Auto 클래스
    device = pick_device()                              # cuda → mps(Mac) → cpu
    if not args.load_4bit and device != "cpu":
        model = model.to(device)                        # (bnb 4bit 은 이미 GPU에 있다)
    model.config.use_cache = False        # 학습 중에는 KV 캐시를 끈다
    return model


def supports_assistant_only(tokenizer) -> bool:
    """이 토크나이저의 chat template로 '답변 부분에만 손실 걸기'가 가능한가?

    TRL은 (1) 템플릿에 {% generation %} 마커가 이미 있거나, (2) 알려진 계열(Llama-3, Gemma, Qwen …)이라
    마커가 든 학습용 템플릿으로 바꿔 줄 수 있을 때만 assistant_only_loss를 지원한다.
    """
    try:
        from trl.chat_template_utils import get_training_chat_template, has_generation_markers
    except ImportError:                      # 옛 TRL: 그냥 시도해 보는 수밖에 없다
        return "generation" in (tokenizer.chat_template or "")
    if has_generation_markers(tokenizer.chat_template or ""):
        return True
    try:
        get_training_chat_template(tokenizer)
        return True
    except ValueError:
        return False


def main():
    p = argparse.ArgumentParser(description="하나의 LLM에 LoRA를 붙여 여러 태스크를 함께 학습합니다.")
    p.add_argument("--model", default="skt/A.X-4.0-Light")
    p.add_argument("--data", default="data/llm-ft/train_main.jsonl", help="학습셋 (train_main.jsonl = 이번 과정 6태스크; train_all.jsonl 은 감성분류·자연어추론까지 8태스크, 비공개)")
    p.add_argument("--tasks", default=None, help="쉼표로 구분한 태스크 부분집합 (기본: 파일의 전체)")
    p.add_argument("--out", default="output/llm-ft")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1, help="이 스텝 수만 학습하고 멈춘다 (속도 측정·연기 테스트용; -1이면 epochs 기준)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--load-4bit", action="store_true", help="QLoRA (메모리 작은 GPU용)")
    p.add_argument("--inspect", action="store_true", help="학습하지 않고 렌더링된 예시만 출력")
    p.add_argument("--plain-template", action="store_true",
                   help="[실험] 모델 고유 chat template 대신 [시스템]/[사용자]/[답변] 단순 형식을 강제 (Instruct 모델도 답변 손실 가능)")
    p.add_argument("--full-loss", action="store_true", help="[실험] 답변 부분에만 손실 걸기를 끄고 전체 시퀀스로 학습")
    p.add_argument("--attn", default=None,
                   help="어텐션 구현 지정 (예: flex_attention). gpt-oss처럼 sdpa를 못 쓰는 모델용")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    tasks = args.tasks.split(",") if args.tasks else None
    ds = load_dataset(args.data, tasks)
    counts = {}
    for t in ds["task"]:
        counts[t] = counts.get(t, 0) + 1
    print(f"학습 데이터 {len(ds)}건 — {counts}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    template_kind = ensure_chat_template(tokenizer, force_plain=args.plain_template)
    if template_kind == "plain":
        if args.plain_template:
            print("\n[알림] --plain-template: 모델 고유 형식 대신 단순 대화 형식으로 학습합니다.")
        else:
            print("\n[알림] 이 모델에는 chat template이 없습니다 (Base 모델).")
            print("       [시스템]/[사용자]/[답변] 로 구분하는 단순 대화 형식을 붙여서 학습합니다.")
        print("       평가·데모에서도 같은 형식을 자동으로 씁니다 (어댑터 폴더에 저장되는 토크나이저에 담김).\n")

    # 모델을 먼저 올린다 — 템플릿의 '턴 종료 토큰'을 이 모델이 실제로 낼 수 있는지(출력 임베딩이
    # 학습된 것인지) 확인해야 하기 때문이다. 아니면 템플릿을 고친다 (common.check_end_token 설명 참고).
    model = build_model(args)
    end_info = check_end_token(model, tokenizer, args.model)
    if end_info["action"] == "plain":
        template_kind = "plain"          # Base 모델인데 미학습 템플릿이 들어 있던 경우 (kanana-base)

    # --- 학습 텍스트가 실제로 어떻게 만들어지는지 확인 ---
    # 대화 형식이 모델별 chat template을 거쳐 하나의 문자열이 된다.
    # 여기서 '생각(thinking) 블록'이 끼어들지 않는지 반드시 눈으로 확인해야 한다.
    sample = ds[0]["messages"]
    rendered = render_example(tokenizer, sample)
    print("\n--- 학습에 들어가는 텍스트 ---")
    print(rendered[:1200])

    # 추론할 때 쓰는 프롬프트도 같이 확인한다.
    # 학습 텍스트와 추론 프롬프트의 앞부분이 어긋나면 모델이 엉뚱하게 동작한다.
    infer_prompt = build_prompt(tokenizer, sample)
    print("--- 추론할 때의 프롬프트 (마지막 200자) ---")
    print(infer_prompt[-200:])
    consistent = rendered.startswith(infer_prompt)
    print(f"\n학습 텍스트가 추론 프롬프트로 시작하는가: {'예 (일관됨)' if consistent else '아니오 (확인 필요!)'}")
    if "<think>" in infer_prompt:
        print("참고: 추론 프롬프트에 빈 <think></think> 블록이 이미 들어있으므로,")
        print("      모델이 생성하는 답변에는 그 블록이 나타나지 않는다. (학습/추론 일관)")
    print("--- 끝 ---\n")

    lengths = []
    for m in ds["messages"][:300]:
        lengths.append(len(tokenizer(render_example(tokenizer, m), add_special_tokens=False)["input_ids"]))
    lengths.sort()
    print(f"토큰 길이(300건 표본): 중앙값 {lengths[len(lengths)//2]}, "
          f"95분위 {lengths[int(len(lengths)*0.95)]}, 최대 {lengths[-1]}")
    over = sum(1 for x in lengths if x > args.max_length)
    print(f"--max-length {args.max_length} 초과: {over}/{len(lengths)}건 ({over/len(lengths)*100:.1f}%)\n")

    if args.inspect:
        print("(--inspect 모드이므로 학습하지 않고 종료합니다)")
        return

    lora_kwargs = {}
    if getattr(model.config, "model_type", "") == "gpt_oss":
        # MoE 모델은 파라미터의 91%가 전문가(nn.Parameter 3D 텐서)라
        # all-linear 로는 어텐션(3%)에만 LoRA가 붙는다. 전문가 몇 층을 직접 지정한다.
        n_layers = getattr(model.config, "num_hidden_layers", 24)
        picks = [n_layers // 3, 2 * n_layers // 3, n_layers - 1]
        lora_kwargs["target_parameters"] = [
            f"{i}.mlp.experts.{name}" for i in picks for name in ("gate_up_proj", "down_proj")
        ]
        lora_kwargs["lora_dropout"] = 0.0     # target_parameters(ParamWrapper) 는 dropout 을 지원하지 않는다
        print(f"MoE 모델 감지 — 전문가 층 {picks} 에도 LoRA를 붙인다 (lora_dropout=0)")
    elif getattr(model.config, "model_type", "") in MULTIMODAL_TYPES:
        # 비전 인코더가 함께 든 체크포인트: all-linear 는 비전 층에도 LoRA를 붙이므로 언어 모델 선형층만 고른다.
        lora_kwargs["target_modules"] = LANGUAGE_ONLY_LINEAR
        print("멀티모달 체크포인트 감지 — 비전 인코더는 두고 언어 모델의 선형층에만 LoRA를 붙인다")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
        # target_modules="all-linear": 어느 모델 계열이든 통하도록 자동 선택 (위에서 계열별로 바꿀 수 있다)
        **{"target_modules": "all-linear", "lora_dropout": args.lora_dropout, **lora_kwargs},
    )

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,        # -1이면 epochs 기준 (기본)
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_length,
        logging_steps=10,
        save_strategy="no",              # 마지막에 한 번만 저장한다 (중간 체크포인트 불필요)
        bf16=torch.cuda.is_available(),  # AMP bf16은 CUDA에서만; Mac(MPS)은 가중치가 이미 bf16이라 그대로 계산한다
        report_to=[],
        seed=args.seed,
        gradient_checkpointing=True,
        packing=False,                   # 예제를 이어붙이지 않는다 (태스크 경계를 지킨다)
    )
    # 답변 부분에만 손실을 건다 — 지시문까지 외우게 하면 낭비다.
    # 다만 이 기능은 모델의 chat template이 assistant 구간을 표시해줘야 쓸 수 있다.
    # (템플릿에 {% generation %} 마커가 필요하다. 예: Qwen 계열은 있고, EXAONE-4.0은 없다.
    #  우리가 붙이는 단순 형식에는 마커를 넣어두었다. TRL은 Llama-3·Gemma 등 알려진 계열이면
    #  마커가 든 학습용 템플릿으로 바꿔 주기도 한다.)
    # 마커가 없는 모델이면 전체 시퀀스로 학습한다 — 조금 비효율적일 뿐 학습 자체는 된다.
    # 주의: 이 판정은 SFTTrainer를 만들기 '전에' 끝내야 한다. 트레이너 생성 도중 실패하면 모델에
    #       LoRA가 이미 붙은 채로 남아, 다시 만들 때 LoRA가 두 번 끼워지는 사고가 난다.
    assistant_only = supports_assistant_only(tokenizer)
    if not assistant_only:
        print("\n[알림] 이 모델의 chat template은 assistant 구간 표시가 없어")
        print("       '답변 부분에만 손실 걸기'를 쓸 수 없습니다. 전체 시퀀스로 학습합니다.\n")
    if args.full_loss and assistant_only:
        print("\n[알림] --full-loss: 답변 부분에만 손실 걸기를 끄고 전체 시퀀스로 학습합니다.\n")
        assistant_only = False
    if hasattr(cfg, "assistant_only_loss"):
        cfg.assistant_only_loss = assistant_only
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds.remove_columns("task"),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    loss_mode = ("답변 부분에만 손실" if assistant_only
                 else "전체 시퀀스 (--full-loss)" if args.full_loss else "전체 시퀀스 (템플릿 제약)")
    print(f"손실 계산 방식: {loss_mode}")

    trainer.model.print_trainable_parameters()
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())

    t0 = time.time()
    train_out = trainer.train()
    elapsed = time.time() - t0

    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)      # Base 모델이면 우리가 붙인 chat template도 함께 저장된다

    peak_gb = peak_memory_gb()                # CUDA: 최대 할당량 / MPS: 학습 직후 할당량
    losses = [x["loss"] for x in trainer.state.log_history if "loss" in x]
    meta = {
        "model": args.model,
        "data": args.data,
        "template_fix": end_info,          # 턴 종료 토큰 점검 결과 (ok | eos | plain)
        "n_train": len(ds),
        "per_task": counts,
        "epochs": args.epochs,
        **({"max_steps": args.max_steps} if args.max_steps > 0 else {}),  # 연기 테스트일 때만 기록
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "max_length": args.max_length,
        "lora_r": args.lora_r,
        "load_4bit": args.load_4bit,
        "chat_template": template_kind,
        "loss_mode": loss_mode,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(trainable / total * 100, 3),
        "device": device_name(),
        "train_seconds": round(elapsed, 1),
        "peak_gpu_gb": round(peak_gb, 2),
        "final_loss": round(train_out.training_loss, 4),
        "loss_curve": [round(x, 4) for x in losses],
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "train_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n학습 완료: {elapsed/60:.1f}분, GPU 최대 사용량 {peak_gb:.2f}GB, 최종 손실 {train_out.training_loss:.4f}")
    print(f"어댑터 저장: {args.out}")


if __name__ == "__main__":
    main()

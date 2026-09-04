"""
[실습5] LLM 파인튜닝 - 공통 모듈

build_dataset / train / evaluate / serve / 노트북이 모두 같은 지시문과 같은 채점 함수를 써야 한다.
학습할 때의 지시문과 물어볼 때의 지시문이 한 글자라도 다르면 모델이 제 실력을 못 낸다.
그래서 지시문·태스크 정의·채점 함수를 이 파일 한 곳에 모아 둔다.

이 파일에 있는 것:
  1. 태스크 정의 (TASKS)            — 이름, 지시문, 정답 후보, 최대 생성 길이, 지표
  2. chat template 준비 (ensure_chat_template)  — Base 모델처럼 대화 형식이 없는 모델에 형식을 붙인다
  3. 프롬프트 만들기·생성 (build_prompt, generate)
  4. 채점 함수 (score_*)             — 데이터셋의 공식 지표
  5. 평가 루프 (evaluate_tasks)      — 위 것들을 묶어 평가셋 전체를 채점
"""

import json
import math
import re
from collections import Counter
from pathlib import Path

import torch
from transformers import TrainerCallback
from transformers.utils.notebook import NotebookProgressCallback


def pick_device() -> str:
    """쓸 수 있는 장치를 cuda → mps(Apple Silicon) → cpu 순서로 고른다."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_name() -> str:
    """결과 JSON에 남길 장치 이름 (예: 'NVIDIA RTX A6000', 'Apple M2 Max (MPS)')."""
    dev = pick_device()
    if dev == "cuda":
        return torch.cuda.get_device_name(0)
    if dev == "mps":
        import platform
        import subprocess
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, check=False).stdout.strip()
        except Exception:  # noqa: BLE001
            chip = platform.processor()
        return f"{chip or 'Apple Silicon'} (MPS)"
    return "CPU"


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb() -> float:
    """학습 중 장치 메모리 최대 할당량(GB). MPS는 최대치 API가 없어 현재 할당량(학습 직후)을 돌려준다."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    if torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / 1024**3
    return 0.0


def _relax_llama_config_check():
    """transformers 5.16의 LlamaConfig 검증은 hidden_size가 head 수의 배수여야 한다고 보지만,
    kanana-1.5 처럼 head_dim을 따로 지정한 모델(hidden 1792, 24 heads × head_dim 128)은
    실제로는 문제없이 돈다. head_dim이 명시된 설정에서는 그 검증을 건너뛰게 바꾼다."""
    try:
        from transformers.models.llama.configuration_llama import LlamaConfig
    except Exception:  # noqa: BLE001
        return

    def validate_architecture(self):
        if getattr(self, "head_dim", None) is None and self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(f"The hidden size ({self.hidden_size}) is not a multiple of the number of "
                             f"attention heads ({self.num_attention_heads}).")

    LlamaConfig.validate_architecture = validate_architecture
    vals = getattr(LlamaConfig, "__class_validators__", None)
    if vals is not None:
        LlamaConfig.__class_validators__ = [validate_architecture if v.__name__ == "validate_architecture" else v
                                            for v in vals]


_relax_llama_config_check()

# ======================================================================================
# 1. 태스크 정의
# ======================================================================================
# 모든 태스크가 같은 구조를 쓴다: [무엇을 하라] + [출력 형식] + [입력]
# 출력 형식을 좁게 못박을수록 작은 모델이 안정적으로 따라온다.

SYSTEM_PROMPT = "당신은 한국어 자연어처리를 수행하는 도우미입니다. 지시를 정확히 따르고, 요청된 형식으로만 답하세요."

CLS_LABELS = ["긍정", "부정"]
CLS_INSTRUCTION = """다음 영화 리뷰의 감정을 판단하세요.
'긍정' 또는 '부정' 중 하나만 답하세요.

리뷰: {text}"""

NER_LABELS = {"PS": "사람", "LC": "지역", "OG": "기관", "DT": "날짜", "TI": "시간", "QT": "수량"}
NER_INSTRUCTION = """다음 문장에서 개체명을 모두 찾아 JSON 배열로 출력하세요.
개체명 유형: PS(사람), LC(지역), OG(기관), DT(날짜), TI(시간), QT(수량)
출력 형식: [{{"text": "개체명", "label": "유형"}}]
개체명이 없으면 [] 를 출력하세요.

문장: {text}"""

MRC_INSTRUCTION = """다음 지문을 읽고 질문에 답하세요.
답은 지문에 나온 표현을 그대로 사용하고, 답만 짧게 출력하세요.

지문: {context}

질문: {question}"""

# --- 확장 태스크 (KLUE) --------------------------------------------------------------
# NSMC·KorQuAD처럼 오래되고 유명한 데이터는 최신 LLM이 사전학습 중에 이미 봤을 가능성이 있다.
# 그래서 "모델이 정말 배우는가"를 더 엄격하게 보려면 다른 태스크도 함께 돌려보는 것이 좋다.
TC_LABELS = ["IT과학", "경제", "사회", "생활문화", "세계", "스포츠", "정치"]   # KLUE-YNAT 라벨 0~6 순서
TC_INSTRUCTION = """다음 뉴스 제목의 주제를 분류하세요.
'IT과학', '경제', '사회', '생활문화', '세계', '스포츠', '정치' 중 하나만 답하세요.

제목: {text}"""

NLI_LABELS = ["함의", "중립", "모순"]                       # KLUE-NLI 라벨 0(entailment) 1(neutral) 2(contradiction)
NLI_INSTRUCTION = """다음 전제와 가설의 관계를 판단하세요.
전제가 참일 때 가설이 반드시 참이면 '함의', 반드시 거짓이면 '모순', 알 수 없으면 '중립'입니다.
'함의', '중립', '모순' 중 하나만 답하세요.

전제: {premise}
가설: {hypothesis}"""

STS_INSTRUCTION = """다음 두 문장의 의미가 얼마나 비슷한지 0에서 5 사이의 점수로 매기세요.
0은 전혀 다른 의미, 5는 완전히 같은 의미입니다. 소수점 한 자리 숫자만 답하세요.

문장1: {sentence1}
문장2: {sentence2}"""

MATH_INSTRUCTION = """다음 수학 문제를 푸세요.
풀이 과정을 한국어로 짧게 쓰고, 맨 마지막 줄에는 최종 답만 이렇게 쓰세요: #### 42

문제: {question}"""

SQL_INSTRUCTION = """다음 데이터베이스 스키마를 보고, 질문에 답하는 SQL 문을 작성하세요.
SQL 문 한 줄만 출력하고 설명이나 코드블록 표시는 붙이지 마세요.

스키마: {schema}

질문: {question}"""

TASKS = {
    # 핵심 3태스크 — 1일차 BERT 실습(분류/개체명인식/기계독해)과 짝을 이룬다
    "cls": {"name": "감성분류", "dataset": "NSMC", "metric": "accuracy", "max_new_tokens": 8,
            "labels": CLS_LABELS, "instruction": CLS_INSTRUCTION},
    "ner": {"name": "개체명인식", "dataset": "KLUE-NER", "metric": "f1", "max_new_tokens": 192,
            "instruction": NER_INSTRUCTION},
    "mrc": {"name": "기계독해", "dataset": "KorQuAD", "metric": "em", "max_new_tokens": 48,
            "instruction": MRC_INSTRUCTION},
    # 확장 태스크 — 데이터 오염 가능성이 낮은 다른 태스크로도 확인해본다
    "tc": {"name": "주제분류", "dataset": "KLUE-YNAT", "metric": "accuracy", "max_new_tokens": 8,
           "labels": TC_LABELS, "instruction": TC_INSTRUCTION},
    "nli": {"name": "자연어추론", "dataset": "KLUE-NLI", "metric": "accuracy", "max_new_tokens": 8,
            "labels": NLI_LABELS, "instruction": NLI_INSTRUCTION},
    "sts": {"name": "문장유사도", "dataset": "KLUE-STS", "metric": "pearson", "max_new_tokens": 8,
            "instruction": STS_INSTRUCTION},
    # 추론 태스크 — BERT로는 풀 수 없다(정답이 라벨도 스팬도 아닌 계산 결과다)
    "math": {"name": "수학추론", "dataset": "GSM8K-ko", "metric": "em", "max_new_tokens": 400,
             # 400인 이유: GSM8K-ko 정답 풀이 길이 p99가 384토큰(전량 실측)이다.
             # 320으로 두면 4%가 잘려 마지막 `#### 답` 줄이 사라진다.
             "instruction": MATH_INSTRUCTION},
    "sql": {"name": "SQL생성", "dataset": "Spider-ko", "metric": "em", "max_new_tokens": 128,
            "instruction": SQL_INSTRUCTION},
}
CORE_TASKS = ["cls", "ner", "mrc"]
# 2026-09 개편: BERT와 정확히 같은 네 태스크 + 생성 모델만 되는 SQL생성(정규)·수학추론(보너스)
MAIN_TASKS = ["tc", "ner", "mrc", "sts", "sql", "math"]
# 인코더(BERT)·인코더디코더(T5)·디코더(GPT 계열) 세 방식이 모두 가능한 네 태스크
THREE_WAY_TASKS = ["tc", "ner", "mrc", "sts"]
# 정답이 라벨도 스팬도 아니어서 BERT로는 불가능하고, 생성 모델(T5·LLM)만 되는 두 태스크
# SQL생성이 정규 다섯 번째, 수학추론은 +1 보너스다 (2026-09-02 확정 — 파인튜닝으로 수학이 오히려 나빠지는
# 모델이 많아, "왜 어려운가"를 설명하는 자리로 쓴다).
GEN_ONLY_TASKS = ["sql", "math"]
BONUS_TASKS = ["math"]
SHARED_TASKS = THREE_WAY_TASKS                       # 기준선과 짝이 맞는 태스크
TASK_MAX_TOKENS = {k: v["max_new_tokens"] for k, v in TASKS.items()}


def user_message(task: str, payload: dict) -> str:
    """태스크와 입력으로 사용자 메시지(지시문 + 입력)를 만든다. 학습·평가·데모가 모두 이 함수를 쓴다."""
    return TASKS[task]["instruction"].format(**payload)


def make_messages(task: str, payload: dict, answer: str | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message(task, payload)}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs


# ======================================================================================
# 2. chat template — "대화 형식"을 모델에게 알려주는 규칙
# ======================================================================================
# Instruct 모델은 "여기부터 사용자 말, 여기부터 답변"을 구분하는 형식(chat template)을 이미 갖고 있다.
# 그러나 Base 모델(사전학습만 한 모델)은 그런 형식이 없다. 그냥 다음 단어를 이어 쓸 뿐이다.
# Base 모델을 파인튜닝할 때는 우리가 형식을 정해서 알려줘야 한다. 아래가 그 형식이다.
#
#   [시스템]\n...\n\n[사용자]\n...\n\n[답변]\n<정답><eos>
#
# 새 특수 토큰을 만들지 않고 일반 글자만 쓰기 때문에 어떤 토크나이저에서도 그대로 동작한다.
# {% generation %} 표시는 TRL이 "이 구간(답변)에만 손실을 걸어라"를 알아보는 데 쓴다.
PLAIN_CHAT_TEMPLATE = (
    "{% if bos_token %}{{ bos_token }}{% endif %}"
    "{% for m in messages %}"
    "{% if m['role'] == 'system' %}[시스템]\n{{ m['content'] }}\n\n"
    "{% elif m['role'] == 'user' %}[사용자]\n{{ m['content'] }}\n\n"
    "{% elif m['role'] == 'assistant' %}[답변]\n{% generation %}{{ m['content'] }}{{ eos_token }}{% endgeneration %}\n\n"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}[답변]\n{% endif %}"
)


def ensure_chat_template(tokenizer, force_plain: bool = False) -> str:
    """토크나이저에 chat template이 없으면(Base 모델) 위의 단순 형식을 붙인다.

    force_plain=True 이면 모델 고유 형식이 있어도 단순 형식으로 바꾼다 — Instruct 모델을
    '답변 부분에만 손실' 조건으로 학습해 보고 싶을 때(강사용 실험) 쓴다.
    돌려주는 값: "native"(모델 고유 형식 사용) 또는 "plain"(우리가 붙인 형식 사용)
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "chat_template", None) and not force_plain:
        return "native"
    tokenizer.chat_template = PLAIN_CHAT_TEMPLATE
    return "plain"


def load_tokenizer(path: str, force_plain: bool = False):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    ensure_chat_template(tok, force_plain)
    return tok


def end_of_turn_token_id(tokenizer) -> int | None:
    """chat template이 답변 뒤에 붙이는 '턴 종료 토큰'의 id를 찾는다.

    더미 대화 한 턴을 렌더링해 뒤에서부터 첫 특수 토큰을 고른다.
    (Qwen은 <|im_end|>, Llama-3 계열은 <|eot_id|>, EXAONE은 [|endofturn|],
     A.X-4.0-Light 은 Qwen2.5 기반이라 <|im_end|>, 우리 단순 형식은 eos_token)
    """
    msgs = [{"role": "user", "content": "질문"}, {"role": "assistant", "content": "답"}]
    text = render_example(tokenizer, msgs)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    special = set(tokenizer.all_special_ids) | set(tokenizer.get_added_vocab().values())
    for tid in reversed(ids):
        if tid in special:
            return tid
    return None


def check_end_token(model, tokenizer, model_id: str = "", verbose: bool = True) -> dict:
    """턴 종료 토큰을 모델이 '실제로 낼 수 있는지' 확인하고, 아니면 템플릿을 고친다.

    왜 필요한가: 출력층(lm_head)의 어떤 행이 다른 특수 토큰들과 완전히 같은 값이면, 그 토큰은
    사전학습에서 한 번도 쓰이지 않아 초기값 그대로인 것이다. softmax에서 같은 행들은 항상 같은
    확률을 받으므로 그 토큰만 골라 내는 법을 파인튜닝으로도 배울 수 없다 — 모델은 답을 맞히고도
    멈추지 못하고 쓰레기 토큰(예: 바이트 0x00)을 뱉는다.
      · Midm-2.0-Mini-Instruct: 템플릿은 <|eot_id|>로 턴을 끝내지만 그 행이 <|start_header_id|> 등과 동일.
        모델은 실제로 <|end_of_text|>(eos)로 턴을 끝낸다 → 템플릿의 <|eot_id|>를 eos_token으로 바꾼다.
      · kanana-1.5-2.1b-base: Base 모델인데 토크나이저에 Llama-3 대화 템플릿이 들어 있다(특수 토큰 미학습)
        → 우리의 단순 형식(PLAIN_CHAT_TEMPLATE)으로 바꾼다.
    돌려주는 값: {"end_token", "duplicates", "action"}  (action: "ok" | "eos" | "plain")
    """
    eot = end_of_turn_token_id(tokenizer)
    info = {"end_token": tokenizer.convert_ids_to_tokens(eot) if eot is not None else None, "duplicates": 0, "action": "ok"}
    if eot is None:
        return info
    out = model.get_output_embeddings()
    weight = getattr(out, "weight", None)
    if weight is None or weight.dim() != 2 or weight.shape[0] <= eot:
        return info
    special = sorted(set(tokenizer.all_special_ids) | set(tokenizer.get_added_vocab().values()))
    special = [i for i in special if i < weight.shape[0]]
    with torch.no_grad():
        rows = weight[special].float()
        sims = torch.nn.functional.cosine_similarity(rows, weight[eot].float()[None], dim=-1)
        norms = rows.norm(dim=-1)
        same = (sims > 0.9999) & ((norms - weight[eot].float().norm()).abs() < 1e-3)
    dup = int(same.sum().item()) - 1
    info["duplicates"] = dup
    if dup <= 0:
        return info
    is_base = re.search(r"[-_]base\b", model_id, re.IGNORECASE) is not None
    if is_base or tokenizer.eos_token is None:
        tokenizer.chat_template = PLAIN_CHAT_TEMPLATE
        info["action"] = "plain"
    else:
        tokenizer.chat_template = tokenizer.chat_template.replace(info["end_token"], tokenizer.eos_token)
        info["action"] = "eos"
    if verbose:
        print(f"\n[알림] 턴 종료 토큰 {info['end_token']} 의 출력 임베딩이 다른 특수 토큰 {dup}개와 완전히 같습니다"
              f" (사전학습에서 쓰이지 않은 토큰 → 파인튜닝으로도 배울 수 없음).")
        if info["action"] == "plain":
            print("       Base 모델이므로 토크나이저에 든 대화 템플릿 대신 [시스템]/[사용자]/[답변] 단순 형식을 씁니다.\n")
        else:
            print(f"       템플릿의 {info['end_token']} 를 eos_token({tokenizer.eos_token}) 으로 바꿔서 학습·추론합니다.\n")
    return info



def extra_load_kwargs(model_name: str) -> dict:
    """모델 계열에 따라 from_pretrained 에 더 넣어야 하는 인자.

    gpt-oss-20b 는 전문가(MoE) 가중치가 MXFP4 로 압축되어 있는데, 그 전용 커널은 Hopper 이상 GPU에서만
    돌아간다. A6000(Ampere)에서는 bf16 으로 풀어서(dequantize) 올려야 하고(약 39GB), 어텐션은
    sdpa 를 지원하지 않아 eager 로 둔다.
    """
    from transformers import AutoConfig
    try:
        cfg = AutoConfig.from_pretrained(model_name)
    except Exception:
        return {}
    if getattr(cfg, "model_type", "") == "gpt_oss":
        from transformers import Mxfp4Config
        return {"quantization_config": Mxfp4Config(dequantize=True), "attn_implementation": "eager"}
    return {}


def causal_lm_class(model_name: str):
    """이 체크포인트를 '텍스트 생성 모델'로 올릴 때 쓸 Auto 클래스.

    대부분은 AutoModelForCausalLM 이면 되지만, Ministral-3 처럼 비전 인코더가 함께 든(멀티모달)
    체크포인트는 설정 클래스가 달라서(Mistral3Config) 거절된다. 그런 계열은 이미지-텍스트용 Auto 클래스로
    올린다 — 텍스트만 넣으면 안의 언어 모델이 그대로 동작하고, generate()·labels 손실도 똑같이 쓸 수 있다.
    """
    from transformers import AutoConfig, AutoModelForCausalLM
    try:
        cfg = AutoConfig.from_pretrained(model_name)
    except Exception:
        return AutoModelForCausalLM
    if getattr(cfg, "model_type", "") in MULTIMODAL_TYPES:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText
    return AutoModelForCausalLM


MULTIMODAL_TYPES = {"mistral3"}   # 텍스트 전용 Auto 클래스가 거절하는 (비전+텍스트) 설정 계열

# 멀티모달 체크포인트에 LoRA를 붙일 때: 비전 인코더·투영층은 건너뛰고 언어 모델의 선형층에만 붙인다.
LANGUAGE_ONLY_LINEAR = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def build_prompt(tokenizer, messages: list[dict]) -> str:
    """대화 형식을 모델이 읽는 문자열로 바꾼다 (답변 자리까지만).

    EXAONE-4.0, Qwen3/3.5는 '생각(thinking) 모드'를 가진 하이브리드 모델이다
    (수업 기본 모델 A.X-4.0-Light 에는 그 모드가 없어 이 처리가 그냥 지나간다).
    실습에서는 생각 과정 없이 답만 바로 내도록 고정한다 — 그래야 출력이 짧고 예측 가능하다.
    """
    msgs = [m for m in messages if m["role"] != "assistant"]
    try:
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    except TypeError:                       # enable_thinking 인자를 모르는 토크나이저
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # gpt-oss(harmony 형식)는 답 앞에 'analysis' 채널(생각)을 먼저 쓴다.
    # 다른 모델의 enable_thinking=False 와 같은 뜻으로, 'final' 채널을 미리 열어 답만 바로 받는다.
    if text.endswith("<|start|>assistant") and "<|channel|>" in tokenizer.get_vocab():
        text += "<|channel|>final<|message|>"
    return text


def render_example(tokenizer, messages: list[dict]) -> str:
    """학습에 들어가는 전체 텍스트(답변 포함)를 렌더링한다. 눈으로 확인할 때 쓴다."""
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False)


def stop_token_ids(tokenizer, model=None) -> list[int]:
    """생성을 멈춰야 하는 토큰들.

    모델마다 '대화 끝' 토큰 이름이 다르다 — Qwen <|im_end|>, EXAONE [|endofturn|], Llama-3 <|eot_id|>,
    Gemma-3 <end_of_turn>, Gemma-4 <turn|> … 이름을 나열해 두는 방식은 새 모델이 나올 때마다 깨진다
    (실제로 Gemma-4에서 <turn|> 를 빠뜨려 NER 출력이 멈추지 않고 반복된 적이 있다).
    그래서 세 곳에서 모은다: (1) tokenizer.eos_token, (2) chat template이 답변 뒤에 실제로 붙이는 토큰,
    (3) 모델의 generation_config.eos_token_id (제작사가 지정한 종료 토큰 목록).
    """
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    eot = end_of_turn_token_id(tokenizer)
    if eot is not None:
        ids.add(eot)
    gen_cfg = getattr(model, "generation_config", None) if model is not None else None
    cfg_eos = getattr(gen_cfg, "eos_token_id", None)
    for tid in (cfg_eos if isinstance(cfg_eos, list) else [cfg_eos]):
        if isinstance(tid, int) and tid >= 0:
            ids.add(tid)
    # 잘 알려진 이름들도 있으면 넣는다 (템플릿이 없는 Base 모델에 대비한 안전장치)
    for tok in ["<|im_end|>", "<|endoftext|>", "<|end_of_text|>", "<|eot_id|>", "[|endofturn|]", "<end_of_turn>", "<turn|>",
                "<|return|>", "<|end|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0 and tid != getattr(tokenizer, "unk_token_id", -1):
            ids.add(tid)
    return sorted(ids)


# ======================================================================================
# 3. 생성
# ======================================================================================

@torch.no_grad()
def generate(model, tokenizer, prompts: list[str], max_new_tokens: int, batch_size: int = 16,
             desc: str | None = None) -> list[str]:
    """프롬프트 목록을 배치로 묶어 생성한다. 채점 재현성을 위해 greedy(do_sample=False)로 고정.

    `desc` 를 주면 진행 막대를 띄운다 — 수학추론처럼 몇 분씩 걸리는 태스크에서
    화면이 멈춘 것처럼 보이지 않게 하려는 것이다.
    """
    stop_ids = stop_token_ids(tokenizer, model)
    outs = []
    steps = range(0, len(prompts), batch_size)
    if desc and len(prompts) > batch_size:
        try:
            from tqdm.auto import tqdm  # noqa: PLC0415
            steps = tqdm(steps, desc=desc, unit="배치", leave=False)
        except Exception:  # noqa: BLE001
            pass
    for i in steps:
        chunk = prompts[i: i + batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, padding_side="left").to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id, eos_token_id=stop_ids)
        for j in range(len(chunk)):
            new_tokens = gen[j][enc["input_ids"].shape[1]:]
            outs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return outs


def generate_one(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    return generate(model, tokenizer, [prompt], max_new_tokens, batch_size=1)[0]


# ======================================================================================
# 4. 채점 함수 — 각 데이터셋의 공식 지표
# ======================================================================================

def pick_label(pred: str, labels: list[str]) -> str | None:
    """출력에서 정답 후보 중 하나를 찾는다. 정확히 하나만 등장해야 인정한다 (둘 다 쓰면 오답)."""
    first_line = pred.strip().split("\n")[0] if pred.strip() else ""
    found = [lab for lab in labels if lab in first_line]
    if len(found) == 1:
        return found[0]
    found = [lab for lab in labels if lab in pred]
    return found[0] if len(found) == 1 else None


def score_choice(pred: str, gold: str, labels: list[str]) -> float:
    """분류형 태스크(감성분류/주제분류/자연어추론): 맞으면 1, 틀리거나 형식이 깨지면 0."""
    return 1.0 if pick_label(pred, labels) == gold else 0.0


def score_cls(pred: str, gold: str) -> float:
    return score_choice(pred, gold, CLS_LABELS)


def parse_ner(pred: str) -> list[tuple[str, str]]:
    """개체명 출력에서 JSON 배열을 뽑아 (텍스트, 유형) 목록으로 만든다.

    작은 모델은 JSON 앞뒤에 설명을 붙이거나 배열을 덜 닫는 일이 흔하다.
    그래서 가장 바깥 대괄호만 잘라내서 파싱을 시도한다.
    """
    m = re.search(r"\[.*\]", pred, re.DOTALL)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for it in items:
        if isinstance(it, dict) and "text" in it and "label" in it:
            out.append((str(it["text"]).strip(), str(it["label"]).strip()))
    return out


def score_ner(pred: str, gold: list) -> tuple[int, int, int]:
    """개체 단위 F1을 위한 (맞은 개수, 예측 개수, 정답 개수). 텍스트와 유형이 모두 맞아야 정답."""
    pset = Counter(parse_ner(pred))
    gset = Counter((g["text"].strip(), g["label"].strip()) for g in gold)
    tp = sum((pset & gset).values())
    return tp, sum(pset.values()), sum(gset.values())


def ner_f1_one(pred: str, gold: list) -> float:
    tp, np_, ng = score_ner(pred, gold)
    if not np_ or not ng:
        return 1.0 if (not np_ and not ng) else 0.0
    prec, rec = tp / np_, tp / ng
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def _norm_answer(s: str) -> str:
    """KorQuAD 채점 전처리 — 공백과 구두점을 정리한다."""
    s = re.sub(r"[\"'`·,.!?;:()\[\]{}<>《》〈〉「」『』]", "", s)
    return "".join(s.split())


def score_mrc(pred: str, golds: list[str]) -> tuple[float, float]:
    """기계독해: EM(완전일치)과 글자 단위 F1. 정답 후보가 여러 개면 최대값."""
    p = _norm_answer(pred.strip().split("\n")[0])
    best_em, best_f1 = 0.0, 0.0
    for g in golds:
        g = _norm_answer(g)
        best_em = max(best_em, 1.0 if p == g else 0.0)
        common = Counter(p) & Counter(g)
        n = sum(common.values())
        if n == 0 or not p or not g:
            f1 = 0.0
        else:
            prec, rec = n / len(p), n / len(g)
            f1 = 2 * prec * rec / (prec + rec)
        best_f1 = max(best_f1, f1)
    return best_em, best_f1


# --- 수학추론: 최종 정답 숫자 뽑기 -----------------------------------------------------
# 모델은 지시한 `#### 18` 말고 `#### 정답: 18`, `**#### 18**`, `#### $18` 처럼도 쓴다.
# 이걸 못 잡으면 실력이 아니라 정규식 때문에 점수가 무너진다(실측: 95% -> 7.5%).
MATH_PRED = re.compile(r"####\s*\**\s*(?:정답|답|answer)?\s*[::]?\s*\**\s*\$?\s*(-?[\d,]+(?:\.\d+)?)", re.I)
MATH_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def norm_number(s: str) -> str:
    """'1,200' '$1200' '1200.0' 을 모두 '1200' 으로 맞춘다.

    학습 전 모델은 자릿수가 수백 개인 숫자를 뱉기도 한다. 그러면 float 이 inf 가 되어
    round() 가 터진다(2026-09-03 EAGLE-3B 학습 전 평가에서 실제로 겪음). 채점이 멈추면
    안 되므로, 수로 다룰 수 없는 것은 문자열 그대로 돌려주고 오답으로 처리되게 둔다.
    """
    s = str(s).replace(",", "").replace("$", "").strip()
    try:
        f = float(s)
        if not math.isfinite(f):
            return s
        return str(int(round(f))) if abs(f - round(f)) < 1e-6 else str(f)
    except (ValueError, OverflowError):
        return s


def parse_math(pred: str) -> tuple[str | None, bool]:
    """(정답 문자열, 형식을 지켰는가). 형식이 깨지면 마지막 숫자라도 주워 본다."""
    m = MATH_PRED.findall(pred)
    if m:
        return norm_number(m[-1]), True
    nums = MATH_NUM.findall(pred)
    return (norm_number(nums[-1]) if nums else None), False


def sql_token_f1(pred: str, gold: str) -> float:
    """SQL 토큰 다중집합 F1 — 뜻은 같은데 표면이 다른 경우를 EM보다 관대하게 본다."""
    import collections
    pt = collections.Counter(re.findall(r"[a-z_][a-z_0-9]*|\d+|[^\s]", pred))
    gt = collections.Counter(re.findall(r"[a-z_][a-z_0-9]*|\d+|[^\s]", gold))
    common = sum((pt & gt).values())
    if not common:
        return 0.0
    prec, rec = common / sum(pt.values()), common / sum(gt.values())
    return 2 * prec * rec / (prec + rec)


def norm_sql(s: str) -> str:
    """공백·대소문자·괄호 주변 공백만 맞춘 뒤 완전일치로 본다(실행 비교가 아니다)."""
    s = re.sub(r"```(?:sql)?|```", " ", s)
    s = s.split(";")[0]
    s = re.sub(r"\s+", " ", s).strip().lower()
    return re.sub(r"\s*([(),])\s*", r"\1", s)


def parse_score(pred: str) -> float | None:
    """문장유사도: 출력에서 첫 숫자를 읽는다. 0~5 밖이면 잘라낸다."""
    m = re.search(r"-?\d+(?:\.\d+)?", pred)
    if not m:
        return None
    return min(5.0, max(0.0, float(m.group(0))))


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (vx * vy) if vx and vy else 0.0


def macro_f1(preds: list[str | None], golds: list[str], labels: list[str]) -> float:
    f1s = []
    for lab in labels:
        tp = sum(1 for p, g in zip(preds, golds) if p == lab and g == lab)
        fp = sum(1 for p, g in zip(preds, golds) if p == lab and g != lab)
        fn = sum(1 for p, g in zip(preds, golds) if p != lab and g == lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return sum(f1s) / len(f1s)


def row_score(task: str, pred: str, row: dict) -> float:
    """예제 **하나**가 얼마나 맞았는지 0~100. 맞힌 예·틀린 예를 고를 때 쓴다.

    `score_task` 는 태스크 전체를 공식 지표로 채점한다. 이 함수는 그 지표를 예제 하나에
    적용한 것이라 값이 정확히 같지는 않다 — 피어슨 상관처럼 여러 건이 있어야 정의되는
    지표는 예제 하나에서 계산할 수 없어 오차로 대신한다. 고르는 용도로만 쓴다.
    """
    gold = row["gold"]
    if task in ("cls", "tc", "nli"):
        return 100.0 if pick_label(pred, TASKS[task]["labels"]) == gold else 0.0
    if task == "ner":
        tp, npred, ngold = score_ner(pred, gold)
        prec = tp / npred if npred else 0.0
        rec = tp / ngold if ngold else 0.0
        return 200 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    if task == "mrc":
        return score_mrc(pred, gold)[1] * 100                 # (EM, F1) 중 부분 점수 F1
    if task == "sts":
        v = parse_score(pred)
        v = 2.5 if v is None else v
        return max(0.0, 100.0 - abs(v - float(gold)) * 20)    # 오차 5.0 이면 0점
    if task == "math":
        v, _ = parse_math(pred)
        return 100.0 if v is not None and v == norm_number(gold) else 0.0
    if task == "sql":
        return sql_token_f1(norm_sql(pred), norm_sql(gold)) * 100
    raise ValueError(task)


# ── 데모 서버를 노트북에서 켜고 끄기 ──────────────────────────────────────────────────────
#
# `!python serve.py` 로 띄우면 그 셀이 서버를 붙들고 있어 노트북을 더 진행할 수 없다.
# 대신 배경 프로세스로 띄우고, 다른 셀에서 끈다. 커널을 재시작해도 포트로 찾아서 끌 수 있다.

_DEMO: dict = {}          # port → {"proc", "log", "url"}


def _port_open(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sk:
        sk.settimeout(0.3)
        return sk.connect_ex(("127.0.0.1", port)) == 0


def demo_start(cmd: list[str], port: int, name: str = "데모 서버", wait: int = 180, cwd=None) -> str:
    """서버를 배경에서 켜고, 응답할 때까지 기다린 뒤 주소를 돌려준다.

    이미 그 포트가 열려 있으면 새로 띄우지 않고 주소만 알려 준다.
    모델을 읽는 데 시간이 걸리므로(BERT 몇 초, 7B LLM 1~2분) 진행을 몇 초마다 찍는다.
    """
    import subprocess, sys, time, tempfile, os
    url = f"http://localhost:{port}"
    if _port_open(port):
        print(f"{name}가 이미 켜져 있습니다 → {url}   (끄려면 demo_stop({port}))")
        _show_link(url)
        return url
    log_path = os.path.join(tempfile.gettempdir(), f"deepknlp-demo-{port}.log")
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=cwd,
                            start_new_session=True)          # 커널 인터럽트에 같이 죽지 않게
    _DEMO[port] = {"proc": proc, "log": log_path, "url": url}
    print(f"{name} 켜는 중 — 모델을 읽습니다 (로그: {log_path})")
    t0 = time.time()
    while time.time() - t0 < wait:
        if proc.poll() is not None:                        # 먼저 죽었다 — 모델이 없거나 오류
            print(f"\n{name}가 시작하지 못했습니다. 로그 마지막 줄들:")
            print(_tail(log_path, 12))
            return ""
        if _port_open(port):
            print(f"\n켜졌습니다 ({time.time() - t0:.0f}초) → {url}   끌 때는 demo_stop({port})")
            _show_link(url)
            return url
        if int(time.time() - t0) % 5 == 0:
            print(".", end="", flush=True)
        time.sleep(1)
    print(f"\n{wait}초 안에 응답이 없습니다. 로그를 보세요: {log_path}")
    print(_tail(log_path, 8))
    return ""


def demo_stop(port: int, name: str = "데모 서버") -> None:
    """켜 둔 서버를 끈다. 이 커널이 띄운 것이 아니어도(재시작 뒤, 터미널에서 띄운 것) 포트로 찾아 끈다."""
    import os, signal, subprocess, time
    info = _DEMO.pop(port, None)
    if info and info["proc"].poll() is None:
        os.killpg(os.getpgid(info["proc"].pid), signal.SIGTERM)
        for _ in range(20):
            if info["proc"].poll() is not None:
                break
            time.sleep(0.25)
        if info["proc"].poll() is None:
            os.killpg(os.getpgid(info["proc"].pid), signal.SIGKILL)
    if _port_open(port):                                   # 이 커널이 모르는 프로세스 — 포트로 찾는다
        try:
            out = subprocess.run(["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"], capture_output=True, text=True).stdout
            for pid in out.split():
                os.kill(int(pid), signal.SIGTERM)
            time.sleep(1)
        except Exception:                                  # noqa: BLE001
            pass
    if _port_open(port):
        print(f"{name}가 아직 {port} 포트를 잡고 있습니다. 터미널에서:  lsof -iTCP:{port}")
    else:
        print(f"{name}를 껐습니다 (포트 {port}). 다시 켜려면 켜기 셀을 실행하세요.")


def demo_status(port: int, name: str = "데모 서버") -> bool:
    on = _port_open(port)
    print(f"{name}: {'켜져 있음 → http://localhost:' + str(port) if on else '꺼져 있음'}")
    return on


def _tail(path: str, n: int) -> str:
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
        return "\n".join("   " + ln for ln in lines[-n:])
    except OSError:
        return ""


def _show_link(url: str) -> None:
    try:
        from IPython.display import HTML, display    # noqa: PLC0415
        display(HTML(f'<div style="margin:6px 0;font-size:15px">🌐 <a href="{url}" target="_blank" '
                     f'style="font-weight:600">{url}</a> — 새 탭에서 열립니다. 다 보고 나면 <b>끄기 셀</b>을 실행하세요.</div>'))
    except Exception:                                      # noqa: BLE001
        pass


# ── 맞힌 예·틀린 예를 HTML 로 — 표는 긴 지문을 자른다. 잘린 자리에 정답이 있기도 하다 ──────────
#
# 노트북에서 `L.examples(...)` / `examples_table(...)` 의 결과가 그대로 화면에 그려진다.
# 본문은 자르지 않고 전부 보여 주고, 태스크마다 보기 좋은 모양을 따로 둔다:
#   기계독해  지문 안에서 정답 구간(초록 배경)과 모델이 고른 구간(주황 밑줄)을 표시한다
#   개체명    정답·예측 개체를 칩으로 — 둘 다 있으면 초록, 정답에만(놓침) 회색, 예측에만(잘못 잡음) 빨강
#   나머지    문장 전체 + 정답/예측 나란히. 수학추론·SQL 은 긴 답이라 코드 상자로.

_EX_CSS = """
<style>
.exv{font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Noto Sans KR","Malgun Gothic",sans-serif;
     font-size:13.5px;line-height:1.65;color:#23201d;max-width:980px}
.exv .card{border:1px solid #e3ded6;border-left:5px solid #9aa;border-radius:10px;padding:10px 14px;margin:8px 0;background:#fff}
.exv .card.ok{border-left-color:#2f7d4a}.exv .card.bad{border-left-color:#c0392b}
.exv .head{display:flex;gap:10px;align-items:baseline;margin-bottom:6px}
.exv .mark{font-weight:700;padding:1px 9px;border-radius:999px;font-size:12.5px}
.exv .ok .mark{background:#e6f4ea;color:#2f7d4a}.exv .bad .mark{background:#fdecea;color:#c0392b}
.exv .score{color:#6f6a63;font-size:12.5px}
.exv .lab{display:inline-block;min-width:48px;color:#6f6a63;font-size:12px;font-weight:600;margin-right:6px;vertical-align:top}
.exv .fld{margin:3px 0;word-break:break-all;white-space:pre-wrap}
.exv .ga{display:flex;gap:18px;flex-wrap:wrap;margin-top:6px;padding-top:6px;border-top:1px dashed #e3ded6}
.exv .ga b{font-weight:600;color:#6f6a63;margin-right:6px;font-size:12px}
.exv .val{padding:1px 7px;border-radius:5px;background:#f4f1ec}
.exv .val.g{background:#e6f4ea}.exv .val.p{background:#fff4d6}
.exv mark.g{background:#c7ecd2;color:inherit;border-radius:3px;padding:0 1px}
.exv mark.p{background:transparent;color:inherit;border-bottom:3px solid #e07b39;padding:0 1px}
.exv mark.gp{background:#c7ecd2;color:inherit;border-radius:3px;border-bottom:3px solid #e07b39;padding:0 1px}
.exv .chip{display:inline-block;margin:2px 4px 2px 0;padding:1px 8px;border-radius:999px;font-size:12.5px;border:1px solid transparent}
.exv .chip.both{background:#e6f4ea;color:#1f5c36;border-color:#9fd3b0}
.exv .chip.miss{background:#f1efeb;color:#6f6a63;border-color:#d9d3c9;text-decoration:line-through}
.exv .chip.extra{background:#fdecea;color:#8b1e12;border-color:#f2b8b0}
.exv .chip small{opacity:.7;margin-left:4px}
.exv pre{margin:4px 0;padding:8px 10px;background:#f4f1ec;border-radius:7px;white-space:pre-wrap;word-break:break-all;font-size:12.5px}
.exv .legend{color:#6f6a63;font-size:12px;margin:2px 0 6px}
@media (prefers-color-scheme: dark){
 .exv{color:#e9e5df}.exv .card{background:#1f1e23;border-color:#3a3740}
 .exv .ok .mark{background:#1d3326;color:#6cc48a}.exv .bad .mark{background:#3a1d1a;color:#f19a8e}
 .exv .val{background:#26242b}.exv .val.g{background:#1d3326}.exv .val.p{background:#3a2f10}
 .exv mark.g,.exv mark.gp{background:#2a5a3a}.exv .chip.both{background:#1d3326;color:#9fd3b0;border-color:#2a5a3a}
 .exv .chip.miss{background:#2b2a2f;color:#9c968d;border-color:#3a3740}.exv .chip.extra{background:#3a1d1a;color:#f19a8e;border-color:#6b2a22}
 .exv pre{background:#26242b}.exv .ga{border-top-color:#3a3740}
}
</style>
"""

# ── 원본 데이터 → 변환된 학습 예제, 대화(messages) — 눈으로 보기 좋게 ──────────────────────
#
# "원본 한 줄이 어떻게 지시문·정답으로 바뀌는가"와 "여러 턴이 하나의 학습 텍스트가 될 때
# 어디가 system·user·assistant인지"를 표에 몰아넣지 않고 역할별 말풍선으로 색을 나눠 보여준다.

_CHAT_CSS = """
<style>
.chv{font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Noto Sans KR","Malgun Gothic",sans-serif;
     font-size:13.5px;line-height:1.6;color:#23201d;max-width:980px}
.chv .msg{border-radius:10px;padding:8px 12px;margin:6px 0;white-space:pre-wrap;word-break:break-word;border:1px solid transparent}
.chv .msg .role{display:block;font-size:11px;font-weight:700;letter-spacing:.04em;margin-bottom:4px;text-transform:uppercase}
.chv .msg.system{background:#f1efeb;border-color:#d9d3c9;color:#5c574f}
.chv .msg.system .role{color:#8a8478}
.chv .msg.user{background:#eaf1fb;border-color:#bcd4f2;margin-right:12%}
.chv .msg.user .role{color:#2f6fb0}
.chv .msg.assistant{background:#e6f4ea;border-color:#a9d9b8;margin-left:12%}
.chv .msg.assistant .role{color:#2f7d4a}
.chv .box{border:1px solid #e3ded6;border-radius:10px;padding:10px 14px;margin:8px 0;background:#fff}
.chv .box .lab{font-size:11px;font-weight:700;letter-spacing:.05em;color:#8a8478;margin-bottom:6px}
.chv .box.raw{border-left:5px solid #8a8478}
.chv .box.conv{border-left:5px solid #7a5cff}
.chv pre{margin:0;background:#f4f1ec;border-radius:7px;padding:8px 10px;white-space:pre-wrap;word-break:break-word;font-size:12.5px;
         font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.chv .kv{margin:2px 0}
.chv .kv b{color:#6f6a63;font-weight:600;font-size:12px;margin-right:6px}
.chv .arrow{text-align:center;color:#8a8478;font-size:20px;margin:2px 0}
@media (prefers-color-scheme: dark){
 .chv{color:#e9e5df}
 .chv .msg.system{background:#26242b;border-color:#3a3740;color:#9c968d}
 .chv .msg.user{background:#1c2a3a;border-color:#2c4a70}.chv .msg.user .role{color:#7fb0e8}
 .chv .msg.assistant{background:#1d3326;border-color:#2a5a3a}.chv .msg.assistant .role{color:#6cc48a}
 .chv .box{background:#1f1e23;border-color:#3a3740}
 .chv pre{background:#26242b}
}
</style>
"""


def chat_html(messages: list[dict], title: str = "") -> str:
    """대화(messages) 하나를 역할별 말풍선으로. system=회색·user=파랑·assistant=초록, 줄바꿈 그대로 유지."""
    role_ko = {"system": "system · 역할 설명", "user": "user · 지시문+입력", "assistant": "assistant · 정답"}
    body = "".join(f'<div class="msg {m["role"]}"><span class="role">{_esc(role_ko.get(m["role"], m["role"]))}'
                   f'</span>{_esc(m["content"])}</div>' for m in messages)
    head = f'<div class="lab">{_esc(title)}</div>' if title else ""
    return f'{_CHAT_CSS}<div class="chv">{head}{body}</div>'


def raw_sample(task: str, n: int = 1, data_dir: str = "data/llm-ft", raw_root: str = "data") -> list[dict]:
    """`stats.json` 이 가리키는 원본 파일에서 그 태스크의 원본 한 줄(가공 전)을 그대로 읽어 온다."""
    stats = json.loads((Path(data_dir) / "stats.json").read_text(encoding="utf-8"))
    src = Path(stats["per_task"][task]["train_src"])
    if not src.exists():
        # 공식 train 전체(예: KorQuAD 87MB)는 공개 저장소에 없다 — 절반 세트로 대신한다
        half = src.with_name(src.stem + "-half" + src.suffix)
        if half.exists():
            src = half
    return read_jsonl(src, n)


def raw_to_train_html(task: str, raw_row: dict, converted: dict) -> str:
    """원본 한 줄 → 변환된 학습 예제(messages 포함) 를 나란히 보여준다. 표로 자르지 않는다."""
    raw_json = json.dumps(raw_row, ensure_ascii=False, indent=2)
    kv = "".join(f'<div class="kv"><b>{_esc(k)}</b>{_esc(v)}</div>'
                 for k, v in converted.items() if k != "messages")
    conv_box = (f'<div class="box conv"><div class="lab">변환됨 — {_esc(TASKS[task]["name"])} '
                f'({_esc(TASKS[task]["dataset"])})</div>{kv}</div>')
    chat = chat_html(converted["messages"], title="messages (학습에 들어가는 대화)")
    return (f'{_CHAT_CSS}<div class="chv">'
            f'<div class="box raw"><div class="lab">원본 한 줄 (가공 전)</div><pre>{_esc(raw_json)}</pre></div>'
            f'<div class="arrow">↓ build_dataset.py 의 make_{task} 변환</div>'
            f'{conv_box}{chat}</div>')


_FIELD_KO = {"text": "문장", "sentence1": "문장 1", "sentence2": "문장 2", "context": "지문",
             "question": "질문", "premise": "전제", "hypothesis": "가설", "schema": "스키마", "db_id": "DB"}


def _esc(v) -> str:
    import html as _h
    return _h.escape(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))


def _mrc_context_html(context: str, golds, pred: str) -> str:
    """지문에 정답 구간(초록)과 예측 구간(주황 밑줄)을 겹쳐 표시한다."""
    spans = []                                     # (start, end, kind)
    for g in (golds if isinstance(golds, list) else [golds]):
        g = str(g).strip()
        if g:
            k = context.find(g)
            if k >= 0:
                spans.append((k, k + len(g), "g"))
                break                                 # 정답 후보 여럿이면 첫 번째만
    p = (pred or "").strip().split("\n")[0]
    if p:
        k = context.find(p)
        if k >= 0:
            spans.append((k, k + len(p), "p"))
    if not spans:
        return _esc(context)
    # 글자마다 어떤 표시가 겹치는지 계산해 구간별로 나눈다
    n = len(context)
    kinds = [set() for _ in range(n)]
    for a, b, kd in spans:
        for i in range(max(0, a), min(n, b)):
            kinds[i].add(kd)
    out, i = [], 0
    while i < n:
        j = i
        while j < n and kinds[j] == kinds[i]:
            j += 1
        seg = _esc(context[i:j])
        if kinds[i]:
            cls = "gp" if kinds[i] == {"g", "p"} else next(iter(kinds[i]))
            out.append(f'<mark class="{cls}">{seg}</mark>')
        else:
            out.append(seg)
        i = j
    return "".join(out)


def _ner_chips_html(gold, pred: str) -> str:
    """정답·예측 개체를 칩으로. 둘 다=초록, 정답에만=회색(놓침), 예측에만=빨강(잘못 잡음)."""
    def norm(ents):
        out = []
        for e in ents or []:
            if isinstance(e, dict) and "text" in e and "label" in e:
                out.append((str(e["text"]).strip(), str(e["label"]).strip()))
        return out
    g = norm(gold if isinstance(gold, list) else [])
    try:
        pj = json.loads(pred) if isinstance(pred, str) else pred
        m = re.search(r"\[.*\]", pred, re.DOTALL) if isinstance(pred, str) and not isinstance(pj, list) else None
        if m:
            pj = json.loads(m.group(0))
    except Exception:
        pj = None
    p = norm(pj if isinstance(pj, list) else [])
    gs, ps = set(g), set(p)
    chips = []
    for t, l in g:
        chips.append(f'<span class="chip {"both" if (t, l) in ps else "miss"}">{_esc(t)}<small>{_esc(l)}</small></span>')
    for t, l in p:
        if (t, l) not in gs:
            chips.append(f'<span class="chip extra">{_esc(t)}<small>{_esc(l)}</small></span>')
    if pj is None and isinstance(pred, str) and pred.strip():
        chips.append(f'<span class="chip extra">형식 깨짐: {_esc(pred[:80])}</span>')
    return "".join(chips) or '<span class="chip miss">(개체 없음)</span>'


def examples_html(task: str, picked, preds: list[str], rows: list[dict]) -> str:
    """`picked` = [(표시, 점수, 인덱스)] → 카드 HTML. 본문은 자르지 않는다."""
    cards = []
    for mark, sc, i in picked:
        r, pred = rows[i], preds[i]
        ok = mark.startswith("○")
        head = (f'<div class="head"><span class="mark">{_esc(mark)}</span>'
                f'<span class="score">{sc:.0f}점</span></div>')
        body = []
        inp = r.get("input", {})
        if task == "mrc":
            body.append(f'<div class="fld"><span class="lab">질문</span>{_esc(inp.get("question", ""))}</div>')
            body.append(f'<div class="fld"><span class="lab">지문</span>'
                        f'{_mrc_context_html(inp.get("context", ""), r.get("gold"), pred)}</div>')
        else:
            for k, v in inp.items():
                body.append(f'<div class="fld"><span class="lab">{_esc(_FIELD_KO.get(k, k))}</span>{_esc(v)}</div>')
        if task == "ner":
            body.append(f'<div class="ga"><div><b>개체</b>{_ner_chips_html(r.get("gold"), pred)}</div></div>')
        elif task in ("math", "sql"):
            body.append('<div class="ga" style="display:block">'
                        f'<div><b>정답</b></div><pre>{_esc(r.get("gold"))}</pre>'
                        f'<div><b>모델 답</b></div><pre>{_esc(pred)}</pre></div>')
        else:
            gold = r.get("gold")
            gold_s = ", ".join(map(str, gold)) if isinstance(gold, list) else str(gold)
            pred_s = pred if str(pred).strip() else "(빈 답)"
            body.append(f'<div class="ga"><span><b>정답</b><span class="val g">{_esc(gold_s)}</span></span>'
                        f'<span><b>모델 답</b><span class="val p">{_esc(pred_s)}</span></span></div>')
        cards.append(f'<div class="card {"ok" if ok else "bad"}">{head}{"".join(body)}</div>')
    legend = {"mrc": "지문에서 <mark class=\"g\">초록 배경</mark>은 정답 구간, <mark class=\"p\">주황 밑줄</mark>은 모델이 고른 구간입니다. 겹치면 둘 다 보입니다.",
              "ner": "개체 칩 — <span class=\"chip both\">둘 다</span> <span class=\"chip miss\">정답에만(놓침)</span> <span class=\"chip extra\">예측에만(잘못 잡음)</span>"}.get(task, "")
    return _EX_CSS + '<div class="exv">' + (f'<div class="legend">{legend}</div>' if legend else "") + "".join(cards) + "</div>"


class ExamplesView:
    """노트북에서는 HTML 카드로, 그 밖에서는 표로 보인다."""

    def __init__(self, html: str, frame):
        self.html, self.frame = html, frame

    def _repr_html_(self):
        return self.html

    def __repr__(self):
        return self.frame.to_string(index=False)

    def to_frame(self):
        return self.frame


def examples_table(task: str, preds: list[str], rows: list[dict], n: int = 2):
    """**맞힌 예 `n`건과 틀린 예 `n`건을 함께** 표로 돌려준다.

    틀린 것만 보면 "이 모델은 못 쓰겠다"로 읽히기 쉽다. 실제로는 대부분을 맞히고 있고
    어디서 무너지는지가 따로 있다. 그 둘을 나란히 놓아야 모델을 제대로 판단할 수 있다.
    """
    import pandas as pd

    def short(v, k=70):
        t = " ".join(str(v).split())
        return t if len(t) <= k else t[:k] + "…"

    def ok(sc):
        """이 점수를 '맞았다'로 볼 것인가.

        주제분류·수학추론처럼 맞다/틀리다뿐인 태스크는 만점이어야 맞은 것이다.
        개체명·기계독해·문장유사도·SQL 은 부분 점수가 있으므로 절반을 넘으면 맞은 쪽으로 센다.
        """
        return sc >= (99.9 if task in ("cls", "tc", "nli", "math") else 60.0)

    # key= 를 반드시 준다 — 점수가 같으면 파이썬이 다음 원소를 비교하려다 터진다
    scored = sorted(((row_score(task, p, r), i) for i, (p, r) in enumerate(zip(preds, rows))),
                    key=lambda x: x[0], reverse=True)
    good = [t for t in scored if ok(t[0])][:n]              # 잘 맞힌 것부터
    bad = [t for t in scored if not ok(t[0])][-n:][::-1]     # 가장 크게 틀린 것부터
    picked = ([("○ 맞음", sc, i) for sc, i in good]
              + [("✗ 틀림", sc, i) for sc, i in bad])

    recs = []
    for mark, sc, i in picked:
        rec = {"채점": f"{mark}  {sc:.0f}점"}
        rec.update({k: short(v) for k, v in rows[i]["input"].items()})
        rec["정답"] = short(rows[i]["gold"], 50)
        rec["모델 예측"] = short(preds[i], 50)
        recs.append(rec)
    # 노트북에서는 본문을 자르지 않는 HTML 카드로 보인다 (표는 긴 지문을 잘라 정답이 가려지기도 했다)
    return ExamplesView(examples_html(task, picked, preds, rows), pd.DataFrame(recs))


def score_task(task: str, preds: list[str], rows: list[dict]) -> dict:
    """태스크 하나의 예측 목록을 공식 지표로 채점해 요약 dict를 돌려준다."""
    n = len(rows)
    if task in ("cls", "tc", "nli"):
        labels = TASKS[task]["labels"]
        picked = [pick_label(p, labels) for p in preds]
        acc = sum(1 for p, r in zip(picked, rows) if p == r["gold"]) / n
        out = {"accuracy": round(acc * 100, 2), "broken_format": sum(1 for p in picked if p is None)}
        if task == "tc":
            out["macro_f1"] = round(macro_f1(picked, [r["gold"] for r in rows], labels) * 100, 2)
        return out
    if task == "ner":
        tp = np_ = ng = 0
        for p, r in zip(preds, rows):
            a, b, c = score_ner(p, r["gold"])
            tp, np_, ng = tp + a, np_ + b, ng + c
        prec = tp / np_ if np_ else 0.0
        rec = tp / ng if ng else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        broken = sum(1 for p in preds if not re.search(r"\[.*\]", p, re.DOTALL))
        return {"f1": round(f1 * 100, 2), "precision": round(prec * 100, 2),
                "recall": round(rec * 100, 2), "broken_format": broken}
    if task == "mrc":
        ems = f1s = 0.0
        for p, r in zip(preds, rows):
            e, f = score_mrc(p, r["gold"])
            ems, f1s = ems + e, f1s + f
        return {"em": round(ems / n * 100, 2), "f1": round(f1s / n * 100, 2)}
    if task == "sts":
        xs, ys, broken = [], [], 0
        for p, r in zip(preds, rows):
            v = parse_score(p)
            if v is None:
                broken += 1
                v = 2.5                      # 형식이 깨지면 중간값으로 채운다 (점수 손해를 본다)
            xs.append(v)
            ys.append(float(r["gold"]))
        acc_bin = sum(1 for x, y in zip(xs, ys) if (x >= 3.0) == (y >= 3.0)) / n
        return {"pearson": round(pearson(xs, ys) * 100, 2), "binary_accuracy": round(acc_bin * 100, 2),
                "broken_format": broken}
    if task == "math":
        ok = broken = 0
        for p, r in zip(preds, rows):
            v, formatted = parse_math(p)
            if not formatted:
                broken += 1
            if v is not None and v == norm_number(r["gold"]):
                ok += 1
        return {"em": round(ok / n * 100, 2), "broken_format": broken}
    if task == "sql":
        ok = broken = 0
        f1s = 0.0
        for p, r in zip(preds, rows):
            q, g = norm_sql(p), norm_sql(r["gold"])
            if not q.startswith("select"):
                broken += 1
            if q == g:
                ok += 1
            f1s += sql_token_f1(q, g)
        # em은 표면이 글자까지 같아야 인정한다(엄격). 같은 뜻이라도 컬럼 순서나
        # 별칭이 다르면 0점이므로, 토큰 F1을 함께 본다.
        return {"em": round(ok / n * 100, 2), "token_f1": round(f1s / n * 100, 2),
                "broken_format": broken}
    raise ValueError(task)


class MidTrainEvalCallback(TrainerCallback):
    """학습 중간중간 아주 작은 검증셋으로 **실제 채점 지표**(경향)를 보여준다.

    SFTTrainer 는 태스크가 여섯 개고 태스크마다 채점 함수가 달라(EM·F1·피어슨·macro-F1 …)
    Trainer 표준 eval_dataset 하나로는 담을 수 없다. 그래서 `evaluate_tasks`를 학습 훅에서
    직접, 아주 적은 건수로 불러 "손실이 아니라 진짜 성능이 오르는지"를 보여준다.
    """

    def __init__(self, tokenizer, tasks: list[str] = None, data_dir: str = "data/llm-ft",
                limit: int = 20, every_frac: float = 0.1):
        self.tokenizer = tokenizer
        self.tasks = tasks or MAIN_TASKS
        self.data_dir = data_dir
        self.limit = limit
        self.every_frac = every_frac
        self._next_frac = every_frac
        self._rows = []
        self._display_id = None
        self.latest = None            # 가장 최근 점검 결과 {태스크 이름: 점수} — RichProgressCallback 이 표에 병합한다
        self.render_table = True      # Rich 표가 붙어 있으면 False 로 꺼서 표가 두 번 나오지 않게 한다

    def _show(self):
        """누적된 중간 점검을 표 하나로 — 매번 같은 자리에서 갱신되어(update_display)
        Training Loss 표 바로 아래에 검증 성능 표가 함께 자란다."""
        import pandas as pd  # noqa: PLC0415
        from IPython.display import display, update_display  # noqa: PLC0415
        df = pd.DataFrame(self._rows).set_index("진행률")
        if self._display_id is None:
            self._display_id = "midtrain-eval-" + str(id(self))
            print("검증 성능 (태스크당 {}건 — 경향만 봅니다)".format(self.limit))
            display(df, display_id=self._display_id)
        else:
            update_display(df, display_id=self._display_id)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        import torch  # noqa: PLC0415
        if model is None or not state.max_steps:
            return control
        progress = state.global_step / state.max_steps
        if progress + 1e-9 < self._next_frac:
            return control
        self._next_frac += self.every_frac
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                results = evaluate_tasks(model, self.tokenizer, data_dir=self.data_dir, tasks=self.tasks,
                                         limit=self.limit, batch_size=max(self.limit, 4), quiet=True)
            scores = {TASKS[t]["name"]: round(main_metric(t, r), 1) for t, r in results.items()}
            self.latest = dict(scores)
            row = {"진행률": f"{progress * 100:.0f}%"}
            row.update(scores)
            self._rows.append(row)
            if self.render_table:
                self._show()
        finally:
            if was_training:
                model.train()
        return control


class RichProgressCallback(NotebookProgressCallback):
    """기본 진행률 표(Step · Training Loss 두 열)를 대신한다.

    한 표에: 진행률 · Training Loss · 토큰 정확도(다음 토큰을 얼마나 맞히나) · 학습률 · GPU 메모리 ·
    경과/남은 시간, 그리고 `MidTrainEvalCallback` 이 잰 태스크별 검증 점수(그 스텝에 점검이 있었으면).
    기본 `NotebookProgressCallback.on_log` 는 Step·Training Loss 만 쓰도록 굳어 있어 상속해 바꾼다.
    """

    def __init__(self, mid_eval: "MidTrainEvalCallback | None" = None):
        super().__init__()
        self.mid_eval = mid_eval
        self._t0 = None
        if mid_eval is not None:
            mid_eval.render_table = False          # 검증 점수는 이 표 안에 들어가므로 별도 표는 끈다

    def on_train_begin(self, args, state, control, **kwargs):
        import time  # noqa: PLC0415
        super().on_train_begin(args, state, control, **kwargs)
        self._t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        import time  # noqa: PLC0415
        if not logs or "loss" not in logs or self.training_tracker is None:
            return
        step, mx = state.global_step, state.max_steps or 0
        if getattr(self, "_first_line", True):
            self.training_tracker.inner_table = None   # 부모가 미리 잡아 둔 머리(Step·Training Loss)를 버리고 내 순서로
            self._first_line = False
        values = {"Step": step,
                  "진행률": f"{step / mx * 100:.0f}%" if mx else "",
                  "Training Loss": f"{float(logs['loss']):.4f}"}
        acc = logs.get("mean_token_accuracy")
        values["토큰 정확도"] = f"{float(acc) * 100:.1f}%" if acc is not None else ""
        lr = logs.get("learning_rate")
        values["학습률"] = f"{float(lr):.1e}" if lr is not None else ""
        values["GPU 메모리"] = (f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB"
                             if torch.cuda.is_available() else "")
        if self._t0:
            elapsed = time.time() - self._t0
            values["경과"] = f"{elapsed / 60:.1f}분"
            values["남은 예상"] = f"{elapsed / step * (mx - step) / 60:.1f}분" if (mx and step) else ""
        else:
            values["경과"] = values["남은 예상"] = ""
        if self.mid_eval is not None:
            latest = self.mid_eval.latest
            for t in self.mid_eval.tasks:
                name = TASKS[t]["name"]
                v = latest.get(name) if latest else None
                values[name] = f"{v:.1f}" if isinstance(v, (int, float)) else ""
            self.mid_eval.latest = None
        self.training_tracker.write_line(values)


def attach_rich_progress(trainer, mid_eval: "MidTrainEvalCallback | None" = None):
    """Trainer 의 기본 진행률 표 콜백을 떼고 RichProgressCallback 을 붙인다. 노트북 §6 에서 부른다."""
    from transformers import ProgressCallback  # noqa: PLC0415
    trainer.remove_callback(NotebookProgressCallback)
    trainer.remove_callback(ProgressCallback)
    rich = RichProgressCallback(mid_eval)
    trainer.add_callback(rich)
    return rich


def main_metric(task: str, summary: dict) -> float:
    """비교표에 쓸 대표 지표 하나."""
    return summary[TASKS[task]["metric"]]


def describe(task: str, s: dict) -> str:
    t = TASKS[task]
    head = f"{t['name']}({t['dataset']})".ljust(16)
    if task in ("cls", "nli"):
        return f"{head} 정확도 {s['accuracy']:5.2f}%   (형식깨짐 {s['broken_format']}건)"
    if task == "tc":
        return f"{head} 정확도 {s['accuracy']:5.2f}% / macro-F1 {s['macro_f1']:5.2f}   (형식깨짐 {s['broken_format']}건)"
    if task == "ner":
        return (f"{head} F1 {s['f1']:5.2f}%  (정밀도 {s['precision']:.2f} / 재현율 {s['recall']:.2f}, "
                f"형식깨짐 {s['broken_format']}건)")
    if task == "mrc":
        return f"{head} EM {s['em']:5.2f}% / F1 {s['f1']:5.2f}%"
    if task == "sql":
        return (f"{head} 완전일치(EM) {s['em']:5.2f}% / 토큰F1 {s['token_f1']:5.2f}"
                f"   (SELECT로 시작 안 함 {s['broken_format']}건)")
    if task == "math":
        return f"{head} 정답률(EM) {s['em']:5.2f}%   (형식깨짐 {s['broken_format']}건)"
    if task == "sts":
        return f"{head} Pearson {s['pearson']:5.2f} / 이진정확도 {s['binary_accuracy']:5.2f}%   (형식깨짐 {s['broken_format']}건)"
    return f"{head} {s}"


# ======================================================================================
# 5. 평가 루프
# ======================================================================================

def read_jsonl(path, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate_tasks(model, tokenizer, data_dir: str = "data/llm-ft", tasks: list[str] | None = None,
                   limit: int | None = 300, batch_size: int = 16, show: int = 0,
                   keep_preds: bool = False, quiet: bool = False) -> dict:
    """평가셋 전체를 생성 → 채점한다. 학습 전/후 모델에 똑같이 적용하면 공정한 비교가 된다."""
    import time
    data_dir = Path(data_dir)
    tasks = tasks or [t for t in TASKS if (data_dir / f"eval_{t}.jsonl").exists()]
    results = {}
    for task in tasks:
        path = data_dir / f"eval_{task}.jsonl"
        if not path.exists():
            if not quiet:
                print(f"  {task}: 평가 파일 없음 ({path}) — 건너뜁니다")
            continue
        rows = read_jsonl(path, limit)
        prompts = [build_prompt(tokenizer, r["messages"]) for r in rows]
        t0 = time.time()
        preds = generate(model, tokenizer, prompts, TASK_MAX_TOKENS[task], batch_size,
                         desc=f"{TASKS[task]['name']} {len(rows)}건 생성 중")
        summary = score_task(task, preds, rows)
        summary.update({"n": len(rows), "sec": round(time.time() - t0, 1)})
        if keep_preds:
            summary["preds"] = preds
        results[task] = summary
        if not quiet:
            print("  " + describe(task, summary) + f"   [{len(rows)}건, {summary['sec']}초]")
        if show:
            print(f"\n  --- {task} 출력 예시 ---")
            for p, r in list(zip(preds, rows))[:show]:
                user_msg = [m for m in r["messages"] if m["role"] == "user"][0]["content"]
                print(f"  [입력] {user_msg.split(chr(10))[-1][:160]}")
                print(f"  [정답] {json.dumps(r['gold'], ensure_ascii=False)[:120]}")
                print(f"  [예측] {p[:200]!r}")
                print()
    return results


def results_table(runs: dict[str, dict], tasks: list[str] | None = None) -> str:
    """{이름: results} 여러 개를 마크다운 표로 만든다. 노트북과 보고서에서 쓴다."""
    tasks = tasks or [t for t in TASKS if any(t in r for r in runs.values())]
    cols = [f"{TASKS[t]['name']} {TASKS[t]['metric']}" for t in tasks]
    lines = ["| 모델 | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, res in runs.items():
        cells = []
        for t in tasks:
            cells.append(f"{main_metric(t, res[t]):.2f}" if t in res else "-")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)

"""transformers 5 호환 보조 함수.

paust/pko-t5-base 저장소는 설정에 `T5Tokenizer`(sentencepiece)라고 적혀 있지만 실제 파일은
BPE `tokenizer.json`이라서, transformers 5의 `AutoTokenizer`가 로드 중 실패한다
(`TypeError: argument 'vocab': 'dict' object is not an instance of 'Sequence'`).
그런 경우 `tokenizer.json`을 그대로 읽는 `PreTrainedTokenizerFast`로 대체한다 —
결과 토크나이저는 같은 어휘·같은 특수 토큰(</s>, <pad>)을 쓰므로 학습·추론 결과는 동일하다.
"""
import logging

from transformers import AutoTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


def load_tokenizer(name_or_path: str, **kwargs):
    """`AutoTokenizer.from_pretrained`를 먼저 시도하고, 실패하면 `PreTrainedTokenizerFast`로 읽는다."""
    try:
        return AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    except Exception as e:  # noqa: BLE001 — 저장소별 설정 불일치는 예외 종류가 다양하다
        logger.warning(f"AutoTokenizer failed for {name_or_path} ({type(e).__name__}); "
                       f"falling back to PreTrainedTokenizerFast(tokenizer.json)")
        kwargs.pop("use_fast", None)
        return PreTrainedTokenizerFast.from_pretrained(name_or_path, **kwargs)

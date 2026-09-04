# ETRI 언어지능연구실 공개 모델을 이 실습에 넣어 보면 (2026-09-03 실측)

https://huggingface.co/etri-lirs — 모델 16종. 이 문서는 그중 둘을 **이 수업의 실습 코드·데이터·채점 함수
그대로** 돌려 본 기록이다. 라인업(강의 본 표)에는 넣지 않았다. 강의에서 "국내 연구소 모델은 어떤가"라는
질문이 나올 때 쓰는 **보너스 자료**다. 수치는 아래 명령을 그대로 실행해 나온 값이다.

## 1. EAGLE-3B — 파인튜닝이 무엇을 가르치는지 가장 선명하게 보여 준다

`etri-lirs/eagle-3b-preview` (LlamaForCausalLM, 3.13B). 실습5와 **똑같이** LoRA로 학습했다
(`train_main.jsonl` 4,800건 · 1 epoch · A6000 48GB · 11.4분 · GPU 최대 8.03GB · 학습 파라미터 22.9M = 0.73%).

| 태스크 | 학습 전 | 학습 후 | 형식깨짐(전 → 후) |
|---|---:|---:|---|
| 주제분류 (정확도) | 1.0 | **73.0** | 297 → 2 |
| 개체명인식 (개체 F1) | 0.0 | **52.7** | 126 → 2 |
| 기계독해 (EM) | 0.0 | **56.3** | — |
| 문장유사도 (Pearson) | 3.0 | **32.0** | 162 → 0 |
| SQL생성 (EM) | 0.0 | 4.7 | 300 → 0 |
| 수학추론 (EM) | 1.0 | 2.7 | 232 → 79 |
| **평균(6)** | **0.8** | **36.9** | |

읽는 법. **학습 전 평균이 0.8이다.** 지시문을 줘도 형식을 거의 지키지 못한다 —
주제분류 300건 중 297건이 "일곱 라벨 중 하나"라는 형식을 깨뜨렸다. 파인튜닝 4,800건 뒤에는 형식깨짐이
2건으로 줄고 평균이 36.9까지 오른다. 이 저장소의 라인업에서 **학습 전→후 상승폭이 가장 크다**
(수업 기본 모델 A.X-4.0-Light (7B) 는 52.0 → 69.7, EXAONE-4.0-1.2B 는 28.3 → 57.1).

동시에, 절대 점수는 라인업 맨 아래쪽이다. 3B인데도 1.2B EXAONE(57.1)보다 낮고,
수업 기본 모델(69.7)과는 격차가 더 크다. 두 사실을 같이 놓으면
이 수업의 요지가 그대로 나온다 — **파인튜닝은 "형식"과 "이 일을 하는 법"을 아주 빠르게 가르치지만,
모델이 원래 갖고 있지 않은 능력을 만들어 주지는 않는다.** 수학추론이 2.7에 머무는 것이 같은 이야기다.

직접 재보려면 실습5와 같은 두 명령이면 된다(모델 이름만 바꾼 것이다).

```bash
python task5-llm-ft/evaluate.py --model etri-lirs/eagle-3b-preview --limit 300 --save output/eagle-before.json
python task5-llm-ft/train.py    --model etri-lirs/eagle-3b-preview --data data/llm-ft/train_main.jsonl --out output/eagle
python task5-llm-ft/evaluate.py --model etri-lirs/eagle-3b-preview --adapter output/eagle --limit 300 --save output/eagle-after.json
```

## 2. KEByT5 — 토큰 프리(바이트 단위) 인코더-디코더

`etri-lirs/kebyt5-base-preview` · `kebyt5-small-preview`. 낱말이나 서브워드가 아니라 **바이트를 직접**
다루는 T5다. 1일차 T5 파트와 **똑같은 조건**(주제분류 800건 · 3에폭 · 평가 300건)으로 학습했다.

| 모델 | 주제분류 정확도 | macro-F1 | 형식깨짐 | 학습 시간 | GPU 최대 |
|---|---:|---:|---:|---:|---:|
| pko-T5-base (수업 기본) | **80.7** | — | — | 0.6분 | 4.8GB |
| KEByT5-base | 59.7 | 49.9 | 69 | 2.1분 | 13.6GB |
| KEByT5-small | 52.3 | 40.9 | 56 | 1.1분 | 7.4GB |

읽는 법. 이 조건에서는 pko-T5보다 낮다. 바이트 단위 모델은 같은 문장을 훨씬 긴 열로 다루므로
(한글 한 글자가 3바이트) 800건·3에폭 같은 소규모 학습에서 불리하고, 메모리도 더 쓴다.
설계 의도는 성능표의 이 칸이 아니라 다른 데 있다 — **어떤 글자든 토크나이저 사전에 없어서 깨지는 일이 없다.**
오타·신조어·이모지·여러 언어가 섞인 입력에 강하다. 성능표 한 칸으로 우열을 말할 수 없는 예다.

```bash
python task5-llm-ft/t5_baseline.py --task tc --mode budget --model etri-lirs/kebyt5-base-preview --limit 300
```

## 3. 그 밖에 공개된 것

EAGLE 계열은 5.4B(MLA 구조)·6.7B도 있고, KEByT5는 small/base/large와 GBST 변형이 있다.
`etri-lirs/SFE`·`KoTSQA-v.2.0` 데이터셋도 공개돼 있다. 위 두 명령의 `--model` 만 바꾸면
같은 자리에서 그대로 재 볼 수 있다.

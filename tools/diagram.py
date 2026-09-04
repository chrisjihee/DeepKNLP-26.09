"""
노트북에 넣을 그림을 SVG 로 만든다.

왜 SVG 인가 —
  ASCII 로 그린 그림은 **한글이 등폭 글꼴에서도 폭이 제각각**이라 줄이 어긋난다.
  PNG 는 확대하면 뭉개지고 파일이 따로 생긴다. matplotlib 은 한글 글꼴을 따로 깔아야 한다.
  SVG 는 파일 하나 없이 노트북 안에 그대로 들어가고, 확대해도 선명하고,
  글자를 브라우저가 그리므로 한글 폭 문제가 없다. 외부 라이브러리도 필요 없다.

주피터의 밝게/어둡게를 따라간다 — `currentColor` 와 CSS 변수를 쓰지 않고,
`@media (prefers-color-scheme: dark)` 를 SVG 안에 넣어 두 벌의 색을 준비한다.

    from diagram import flow, mrc_chunks
    display(HTML(flow([...])))
"""

from __future__ import annotations

import html

FONT = ('-apple-system, BlinkMacSystemFont, "Apple SD Gothic Neo", "Pretendard", '
        '"Noto Sans KR", "Malgun Gothic", sans-serif')
MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace'

# 밝게 / 어둡게 두 벌. SVG 안에 <style> 로 넣어 뷰어의 테마를 따른다.
_STYLE = """
  .bg   { fill:#fbfaf8; }
  .box  { fill:#ffffff; stroke:#d9d3c9; stroke-width:1.2; }
  .box.accent { fill:#efeaff; stroke:#7a5cff; }
  .box.eval   { fill:#e6f4ea; stroke:#2f7d4a; }
  .box.out    { fill:#fff4d6; stroke:#8a5a00; }
  .t    { fill:#23201d; font-size:13px; }
  .t.sm { font-size:11.5px; fill:#6f6a63; }
  .t.b  { font-weight:700; }
  .t.mono { font-size:12px; }
  .arrow { stroke:#6f6a63; stroke-width:1.4; fill:none; }
  .ah    { fill:#6f6a63; }
  .tag   { fill:#7a5cff; font-size:11px; font-weight:700; }
  .side  { stroke:#c9c2b6; stroke-width:1.1; fill:none; stroke-dasharray:3 3; }
@media (prefers-color-scheme: dark) {
  .bg   { fill:#17161a; }
  .box  { fill:#1f1e23; stroke:#3a3740; }
  .box.accent { fill:#2a2340; stroke:#a48cff; }
  .box.eval   { fill:#1d3326; stroke:#6cc48a; }
  .box.out    { fill:#3a2f10; stroke:#f0c060; }
  .t    { fill:#e9e5df; }
  .t.sm { fill:#9c968d; }
  .arrow, .side { stroke:#9c968d; }
  .ah    { fill:#9c968d; }
  .tag   { fill:#a48cff; }
}
"""


def _esc(s: str) -> str:
    return html.escape(str(s), quote=True)


def _wrap(svg_body: str, w: int, h: int, title: str = "") -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" '
            f'style="max-width:{w}px;height:auto;display:block;margin:6px 0;font-family:{FONT}" '
            f'role="img" aria-label="{_esc(title)}">'
            f'<style>{_STYLE}</style>'
            f'<rect class="bg" x="0" y="0" width="{w}" height="{h}" rx="10"/>'
            f'{svg_body}</svg>')


def _arrow_down(x: int, y1: int, y2: int) -> str:
    return (f'<path class="arrow" d="M{x} {y1} L{x} {y2 - 7}"/>'
            f'<path class="ah" d="M{x - 4} {y2 - 7} L{x + 4} {y2 - 7} L{x} {y2} Z"/>')


def flow(steps: list[dict], width: int = 760, title: str = "실습 흐름") -> str:
    """세로 흐름도. 각 단계는 다음 키를 갖는다.

        text  : 상자 안 큰 글자 (필수)
        sub   : 상자 안 작은 글자
        kind  : ""|"accent"|"eval"|"out"  — 색
        mono  : True 면 상자 글자를 등폭으로 (파일 이름 등)
        edge  : **다음 상자로 가는 화살표 옆**에 적을 설명
        tag   : 그 단계가 노트북 몇 절인지 (오른쪽에 §5 처럼)
    """
    pad_x, box_h, gap = 24, 52, 46
    box_w = width - pad_x * 2 - 90            # 오른쪽 90px 은 절 표시 자리
    cx = pad_x + box_w // 2
    body, y = [], 20

    for i, st in enumerate(steps):
        kind = (" " + st["kind"]) if st.get("kind") else ""
        h = box_h + (14 if st.get("sub") else 0)
        body.append(f'<rect class="box{kind}" x="{pad_x}" y="{y}" width="{box_w}" height="{h}" rx="9"/>')
        cls = "t b mono" if st.get("mono") else "t b"
        ty = y + (24 if st.get("sub") else 31)
        body.append(f'<text class="{cls}" x="{cx}" y="{ty}" text-anchor="middle">{_esc(st["text"])}</text>')
        if st.get("sub"):
            body.append(f'<text class="t sm" x="{cx}" y="{ty + 19}" text-anchor="middle">{_esc(st["sub"])}</text>')
        if st.get("tag"):
            body.append(f'<text class="tag" x="{pad_x + box_w + 14}" y="{y + h // 2 + 4}">{_esc(st["tag"])}</text>')

        if i < len(steps) - 1:
            y2 = y + h + gap
            body.append(_arrow_down(cx, y + h, y2))
            if st.get("edge"):
                body.append(f'<text class="t sm" x="{cx + 12}" y="{y + h + gap // 2 + 4}">{_esc(st["edge"])}</text>')
            y = y2
        else:
            y = y + h

    return _wrap("".join(body), width, y + 20, title)


def mrc_chunks(width: int = 760) -> str:
    """긴 지문을 겹쳐 가며 조각내는 그림 (실습4A §4)."""
    pad, bar_h, gap = 24, 22, 34
    label_w = 62
    note_w = 230                       # 오른쪽 설명이 잘리지 않게 자리를 비워 둔다
    full_w = width - pad * 2 - label_w - note_w
    body = []
    rows = [("지문", 0, full_w, "", "box"),
            ("조각 1", 0, int(full_w * 0.42), "", "box accent"),
            ("조각 2", int(full_w * 0.30), int(full_w * 0.42), "앞 조각과 128토큰 겹침", "box accent"),
            ("조각 3", int(full_w * 0.60), int(full_w * 0.40), "답이 없다 → [CLS] 자리(0, 0)", "box out")]
    y = 22
    for name, off, w, note, cls in rows:
        body.append(f'<text class="t sm" x="{pad}" y="{y + 15}">{_esc(name)}</text>')
        body.append(f'<rect class="{cls}" x="{pad + label_w + off}" y="{y}" width="{w}" height="{bar_h}" rx="5"/>')
        if note:
            body.append(f'<text class="t sm" x="{pad + label_w + off + w + 10}" y="{y + 15}">{_esc(note)}</text>')
        y += gap
    body.append(f'<text class="t sm" x="{pad}" y="{y + 12}">'
                f'조각마다 답이 어디 있는지(또는 없는지)를 따로 표시한다 — 그것이 미션 m-mrc-1·2 다.</text>')
    return _wrap("".join(body), width, y + 30, "지문을 겹쳐 가며 조각내기")


def lora(width: int = 620) -> str:
    """LoRA — 원래 가중치는 얼리고 작은 행렬 A·B 만 학습하는 그림 (실습5 · 강의 사이트 공용)."""
    cx_in, cx_out = 70, width - 70
    y_top, y_main, y_low = 40, 130, 220
    body = []
    # 원래 가중치 W — 위쪽, 얼려진 것(회색)
    w_x, w_w = width // 2 - 110, 220
    body.append(f'<rect class="box" x="{w_x}" y="{y_top}" width="{w_w}" height="46" rx="9"/>')
    body.append(f'<text class="t b" x="{width // 2}" y="{y_top + 20}" text-anchor="middle">원래 가중치 W</text>')
    body.append(f'<text class="t sm" x="{width // 2}" y="{y_top + 37}" text-anchor="middle">(얼림 — 학습하지 않는다)</text>')
    # 입력 -> W 지나 -> 출력 (주선)
    body.append(f'<text class="t b" x="{cx_in - 8}" y="{y_main + 5}" text-anchor="end">입력</text>')
    body.append(f'<text class="t b" x="{cx_out + 8}" y="{y_main + 5}" text-anchor="start">출력</text>')
    body.append(f'<path class="arrow" d="M{cx_in} {y_main} L{cx_out - 7} {y_main}"/>')
    body.append(f'<path class="ah" d="M{cx_out - 7} {y_main - 4} L{cx_out - 7} {y_main + 4} L{cx_out} {y_main} Z"/>')
    body.append(f'<path class="side" d="M{width // 2} {y_top + 46} L{width // 2} {y_main}"/>')  # W 와 주선 연결
    # 입력에서 아래로 갈라져 A -> B -> 출력으로 다시 합류 (LoRA 경로, accent 색)
    a_x, b_x, ab_y, ab_w, ab_h = cx_in + 70, cx_in + 210, y_low, 74, 40
    body.append(f'<path class="side" d="M{cx_in} {y_main} L{cx_in} {ab_y + ab_h // 2}"/>')
    body.append(f'<path class="arrow" d="M{cx_in} {ab_y + ab_h // 2} L{a_x - 7} {ab_y + ab_h // 2}"/>')
    body.append(f'<path class="ah" d="M{a_x - 7} {ab_y + ab_h // 2 - 4} L{a_x - 7} {ab_y + ab_h // 2 + 4} '
                f'L{a_x} {ab_y + ab_h // 2} Z"/>')
    body.append(f'<rect class="box accent" x="{a_x}" y="{ab_y}" width="{ab_w}" height="{ab_h}" rx="8"/>')
    body.append(f'<text class="t b" x="{a_x + ab_w // 2}" y="{ab_y + ab_h // 2 + 5}" text-anchor="middle">A</text>')
    body.append(f'<path class="arrow" d="M{a_x + ab_w} {ab_y + ab_h // 2} L{b_x - 7} {ab_y + ab_h // 2}"/>')
    body.append(f'<path class="ah" d="M{b_x - 7} {ab_y + ab_h // 2 - 4} L{b_x - 7} {ab_y + ab_h // 2 + 4} '
                f'L{b_x} {ab_y + ab_h // 2} Z"/>')
    body.append(f'<rect class="box accent" x="{b_x}" y="{ab_y}" width="{ab_w}" height="{ab_h}" rx="8"/>')
    body.append(f'<text class="t b" x="{b_x + ab_w // 2}" y="{ab_y + ab_h // 2 + 5}" text-anchor="middle">B</text>')
    body.append(f'<path class="side" d="M{b_x + ab_w} {ab_y + ab_h // 2} L{cx_out} {ab_y + ab_h // 2} '
                f'L{cx_out} {y_main}"/>')
    body.append(f'<path class="ah" d="M{cx_out - 4} {y_main - 7} L{cx_out + 4} {y_main - 7} '
                f'L{cx_out} {y_main} Z"/>')
    body.append(f'<text class="tag" x="{(a_x + b_x + ab_w) // 2}" y="{ab_y + ab_h + 20}" text-anchor="middle">'
                f'(작은 행렬 두 개만 학습한다)</text>')
    return _wrap("".join(body), width, ab_y + ab_h + 34, "LoRA — 원래 가중치는 얼리고 작은 행렬만 학습")


def lora_dims(width: int = 860) -> str:
    """LoRA — 원래 방식과 LoRA 방식을 차원(d×k · r×k · d×r)과 함께 나란히 비교 (실습5 §5)."""
    y_head, y_main = 30, 130
    left_cx, right_cx = 190, width - 190
    box_w, box_h = 70, 44

    def flow(cx, label_left, label_mid, label_right, w_kind, extra=None):
        out = []
        x0 = cx - 150
        out.append(f'<text class="t b" x="{x0}" y="{y_main + 5}">{label_left}</text>')
        wx = cx - box_w // 2
        out.append(f'<path class="arrow" d="M{x0 + 26} {y_main} L{wx - 7} {y_main}"/>')
        out.append(f'<path class="ah" d="M{wx - 7} {y_main - 4} L{wx - 7} {y_main + 4} L{wx} {y_main} Z"/>')
        out.append(f'<rect class="box{" accent" if w_kind == "accent" else ""}" x="{wx}" y="{y_main - box_h // 2}" '
                    f'width="{box_w}" height="{box_h}" rx="8"/>')
        out.append(f'<text class="t b" x="{cx}" y="{y_main + 5}" text-anchor="middle">{label_mid}</text>')
        x1 = cx + 150
        out.append(f'<path class="arrow" d="M{wx + box_w} {y_main} L{x1 - 26} {y_main}"/>')
        out.append(f'<path class="ah" d="M{x1 - 26} {y_main - 4} L{x1 - 26} {y_main + 4} L{x1 - 19} {y_main} Z"/>')
        out.append(f'<text class="t b" x="{x1}" y="{y_main + 5}">{label_right}</text>')
        if extra:
            out.extend(extra)
        return "".join(out)

    body = []
    body.append(f'<text class="t b" x="{left_cx}" y="{y_head}" text-anchor="middle">원래 가중치 W</text>')
    body.append(f'<text class="t sm" x="{left_cx}" y="{y_head + 17}" text-anchor="middle">(d×k, 얼림)</text>')
    body.append(flow(left_cx, "x", "W", "h", "plain"))

    body.append(f'<text class="t b" x="{width // 2}" y="{y_main + 8}" text-anchor="middle" '
                f'style="font-size:26px">⇒</text>')

    body.append(f'<text class="t b" x="{right_cx}" y="{y_head}" text-anchor="middle">LoRA: W + B·A</text>')
    body.append(f'<text class="t sm" x="{right_cx}" y="{y_head + 17}" text-anchor="middle">'
                f'A: r×k (작음) · B: d×r (작음)</text>')
    # 오른쪽: x -> W(회색, 얼림) 그대로 두고, 그 아래로 A->B(보라, 학습) 우회해 h 에 더해진다
    rx0, rw = right_cx - 150, right_cx - 60 - (right_cx - 150)
    wx = right_cx - box_w // 2 - 60
    body.append(f'<text class="t b" x="{rx0}" y="{y_main + 5}">x</text>')
    body.append(f'<path class="arrow" d="M{rx0 + 14} {y_main} L{wx - 7} {y_main}"/>')
    body.append(f'<path class="ah" d="M{wx - 7} {y_main - 4} L{wx - 7} {y_main + 4} L{wx} {y_main} Z"/>')
    body.append(f'<rect class="box" x="{wx}" y="{y_main - box_h // 2}" width="{box_w - 15}" height="{box_h}" rx="8"/>')
    body.append(f'<text class="t b" x="{wx + (box_w - 15) // 2}" y="{y_main + 5}" text-anchor="middle">W</text>')
    ab_y = y_main + 55
    ax = wx
    body.append(f'<path class="side" d="M{rx0 + 14} {y_main} L{rx0 + 14} {ab_y} L{ax} {ab_y}"/>')
    body.append(f'<path class="arrow" d="M{ax} {ab_y} L{ax + 40 - 7} {ab_y}"/>')
    body.append(f'<path class="ah" d="M{ax + 40 - 7} {ab_y - 4} L{ax + 40 - 7} {ab_y + 4} L{ax + 40} {ab_y} Z"/>')
    body.append(f'<rect class="box accent" x="{ax + 40}" y="{ab_y - 16}" width="34" height="32" rx="6"/>')
    body.append(f'<text class="t b" x="{ax + 57}" y="{ab_y + 5}" text-anchor="middle">A</text>')
    body.append(f'<path class="arrow" d="M{ax + 74} {ab_y} L{ax + 108} {ab_y}"/>')
    body.append(f'<path class="ah" d="M{ax + 101} {ab_y - 4} L{ax + 101} {ab_y + 4} L{ax + 108} {ab_y} Z"/>')
    body.append(f'<rect class="box accent" x="{ax + 108}" y="{ab_y - 16}" width="34" height="32" rx="6"/>')
    body.append(f'<text class="t b" x="{ax + 125}" y="{ab_y + 5}" text-anchor="middle">B</text>')
    hx = wx + (box_w - 15) + 66
    body.append(f'<path class="side" d="M{ax + 142} {ab_y} L{hx} {ab_y} L{hx} {y_main}"/>')
    body.append(f'<path class="ah" d="M{hx - 4} {y_main - 7} L{hx + 4} {y_main - 7} L{hx} {y_main} Z"/>')
    body.append(f'<path class="arrow" d="M{hx} {y_main} L{right_cx + 150 - 26} {y_main}"/>')
    body.append(f'<path class="ah" d="M{right_cx + 150 - 26} {y_main - 4} L{right_cx + 150 - 26} {y_main + 4} '
                f'L{right_cx + 150 - 19} {y_main} Z"/>')
    body.append(f'<text class="t b" x="{right_cx + 150}" y="{y_main + 5}">h</text>')

    return _wrap("".join(body), width, ab_y + 30, "LoRA — 원래 방식과 나란히 비교")


def show(svg: str) -> None:
    """주피터에서 그린다."""
    from IPython.display import HTML, display  # noqa: PLC0415
    display(HTML(svg))


if __name__ == "__main__":   # 눈으로 확인할 때: python tools/diagram.py > /tmp/d.html
    print("<!doctype html><meta charset=utf-8><body style='margin:20px;max-width:820px'>")
    print(flow([
        {"text": "원본 데이터", "sub": "KLUE · KorQuAD · GSM8K-ko · Spider-ko", "edge": "build_dataset.py", "tag": "§2"},
        {"text": "train_main.jsonl", "sub": "6태스크 · messages 형식", "mono": True,
         "edge": "tokenizer + chat template", "tag": "§3"},
        {"text": "학습 전 평가", "sub": "기준선 — 아직 아무것도 배우지 않은 상태", "kind": "eval",
         "edge": "LoRA 어댑터 부착 → SFTTrainer 1 epoch", "tag": "§4"},
        {"text": "학습", "sub": "PEFT + TRL · 전체의 0.55%만 학습", "kind": "accent",
         "edge": "같은 평가셋 · 같은 채점", "tag": "§5·§6"},
        {"text": "학습 후 평가", "sub": "전후 비교 → BERT·T5와 비교", "kind": "eval", "tag": "§7·§8"},
        {"text": "데모 페이지", "sub": "내 모델에게 직접 물어보기", "kind": "out", "tag": "§9"},
    ]))
    print(mrc_chunks())
    print("</body>")

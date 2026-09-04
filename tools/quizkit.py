"""
QuizKit — 미션·퀴즈를 객관식 GUI로 (주피터 노트북과 강의 페이지 공용)

    python tools/quizkit.py check              # quiz/bank/*.json 검증 (필드·정답 유일·URL 형식·난독화 왕복)
    python tools/quizkit.py build              # quiz/bank/*.json → quiz/public.json (난독화된 공개본)
    python tools/quizkit.py demo               # 전 문항을 한 HTML 로 → output/quizkit-demo.html (브라우저 점검용)
    python tools/quizkit.py render m-tc-1      # 문항 하나의 HTML 을 표준출력으로

노트북에서는:

    import sys; sys.path.insert(0, "tools"); import quizkit
    quizkit.show("m-tc-1")          # 1일차 노트북은 L.quiz("m-tc-1") 이 이 함수를 부른다

설계 문서: docs/QUIZKIT-SPEC.md(비공개 정본에만 있음). 문항 원본(`quiz/bank/`)은 강사용이며 공개본에서 제외되고,
`quiz/public.json` 만 공개된다. 정답·오답 이유는 보기 텍스트를 열쇠로 XOR 난독화되어 있어
소스를 열어도 평문이 바로 보이지는 않는다(암호가 아니라 난독화다 — 보기 4개 중 하나가 열쇠이므로).
외부 패키지 없음(stdlib 만).
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BANK_DIR = ROOT / "quiz" / "bank"
PUBLIC = ROOT / "quiz" / "public.json"
SITE = ROOT / "lecture-site"
JS_PATH = SITE / "quiz.js"
CSS_PATH = SITE / "quiz.css"

REQUIRED = ("kind", "notebook", "title", "question", "options", "correct", "why_wrong", "hints", "answer")
KINDS = ("mission", "quiz")
ID_RE = re.compile(r"^[a-z][a-z0-9-]*$")
URL_RE = re.compile(r"https?://[^\s)\]>\"']+")


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 1. 문항 은행 읽기
# ══════════════════════════════════════════════════════════════════════════════════════════════

def load_bank(bank_dir: Path = BANK_DIR) -> dict[str, dict]:
    """quiz/bank/*.json 을 모두 읽어 {id: item} 으로 합친다. 파일은 {id: item} 또는 {"items": {...}}."""
    items: dict[str, dict] = {}
    for path in sorted(bank_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data = data.get("items", data) if isinstance(data, dict) else data
        if isinstance(data, list):
            data = {d["id"]: d for d in data}
        for iid, item in data.items():
            if iid in items:
                raise SystemExit(f"[quizkit] 문항 id 중복: {iid} ({path.name})")
            item = dict(item)
            item.setdefault("id", iid)
            item["_file"] = path.name
            items[iid] = item
    return items


def load_public(path: Path = PUBLIC) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다. 먼저 `python tools/quizkit.py build` 를 실행하세요.")
    return json.loads(path.read_text(encoding="utf-8"))["items"]


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 2. 소형 마크다운 → HTML (코드 펜스 · 인라인 코드 · 굵게 · 링크 · 목록 · 단락만 지원)
# ══════════════════════════════════════════════════════════════════════════════════════════════

_FENCE_RE = re.compile(r"```([\w+-]*)\n(.*?)\n```", re.S)


def _inline(s: str) -> str:
    """이스케이프 뒤 인라인 마크다운. `code`, **bold**, [text](url), 줄 끝 두 칸 줄바꿈.

    인라인 코드를 먼저 자리표시자로 빼 두고 나머지를 변환한다 — 그래야 `**앞 `코드` 뒤**` 처럼
    굵게 표시 안에 코드가 섞여도 `**` 가 그대로 보이지 않는다.
    """
    holds: list[str] = []

    def hold(m: re.Match) -> str:
        holds.append(f"<code>{html.escape(m.group(1))}</code>")
        return f"\x00{len(holds) - 1}\x00"

    t = re.sub(r"`([^`\n]+)`", hold, s)
    t = html.escape(t, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
               lambda m: f'<a href="{html.escape(m.group(2), quote=True)}" target="_blank" rel="noopener">{m.group(1)}</a>', t)
    t = re.sub(r"\x00(\d+)\x00", lambda m: holds[int(m.group(1))], t)
    return t.replace("  \n", "<br>\n")


def md_to_html(md: str) -> str:
    """블록 단위 변환. ```펜스는 <pre><code>, '- ' 줄은 <ul>, 빈 줄로 나뉜 덩어리는 <p>."""
    if not md:
        return ""
    md = md.strip("\n")
    pieces: list[str] = []
    pos = 0
    for m in _FENCE_RE.finditer(md):
        pieces.append(_blocks(md[pos:m.start()]))
        lang = m.group(1) or ""
        cls = f' class="language-{html.escape(lang)}"' if lang else ""
        pieces.append(f"<pre><code{cls}>{html.escape(m.group(2))}</code></pre>")
        pos = m.end()
    pieces.append(_blocks(md[pos:]))
    return "".join(p for p in pieces if p)


def _blocks(md: str) -> str:
    out: list[str] = []
    for chunk in re.split(r"\n\s*\n", md.strip("\n")):
        lines = [ln for ln in chunk.split("\n") if ln.strip()]
        if not lines:
            continue
        if all(re.match(r"^\s*[-*] ", ln) for ln in lines):
            out.append("<ul>" + "".join(f"<li>{_inline(re.sub(r'^\s*[-*] ', '', ln))}</li>" for ln in lines) + "</ul>")
        elif len(lines) >= 2 and all(ln.strip().startswith("|") for ln in lines) \
                and re.match(r"^\s*\|?\s*:?-{3,}", lines[1]):
            # 마크다운 표 — 첫 줄 머리, 둘째 줄 구분선, 나머지 본문. 구분선의 ---: 로 오른쪽 정렬을 읽는다.
            def cells(ln):
                return [c.strip() for c in ln.strip().strip("|").split("|")]
            aligns = ["right" if c.rstrip().endswith(":") and not c.lstrip().startswith(":") else "left"
                      for c in cells(lines[1])]
            def row(ln, tag):
                cs = cells(ln)
                return "<tr>" + "".join(
                    f'<{tag}{" style=\"text-align:right\"" if i < len(aligns) and aligns[i] == "right" else ""}>'
                    f"{_inline(c)}</{tag}>" for i, c in enumerate(cs)) + "</tr>"
            out.append("<table>" + row(lines[0], "th") + "".join(row(ln, "td") for ln in lines[2:]) + "</table>")
        else:
            out.append(f"<p>{_inline(chr(10).join(lines))}</p>")
    return "".join(out)


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 3. 난독화 (docs/QUIZKIT-SPEC.md(비공개 정본에만 있음) §4) — stdlib 만, JS 쪽 quiz.js 와 같은 계산
# ══════════════════════════════════════════════════════════════════════════════════════════════

def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _keystream(key: str, n: int) -> bytes:
    out = bytearray()
    i = 0
    while len(out) < n:
        out += hashlib.sha256(f"{key}:{i}".encode("utf-8")).digest()
        i += 1
    return bytes(out[:n])


def obfuscate(plain: str, key: str) -> str:
    """base64( xor( utf8("OK:" + plain), keystream(NFC(key)) ) )."""
    data = ("OK:" + plain).encode("utf-8")
    ks = _keystream(_nfc(key), len(data))
    return base64.b64encode(bytes(a ^ b for a, b in zip(data, ks))).decode("ascii")


def deobfuscate(payload: str, key: str) -> str | None:
    data = base64.b64decode(payload)
    ks = _keystream(_nfc(key), len(data))
    try:
        s = bytes(a ^ b for a, b in zip(data, ks)).decode("utf-8")
    except UnicodeDecodeError:
        return None
    return s[3:] if s.startswith("OK:") else None


def _mask_index(iid: str, correct: int) -> str:
    ks = hashlib.sha256(iid.encode("utf-8")).digest()
    raw = str(correct).encode("ascii")
    return base64.b64encode(bytes(a ^ b for a, b in zip(raw, ks))).decode("ascii")


def _unmask_index(iid: str, k: str) -> int:
    ks = hashlib.sha256(iid.encode("utf-8")).digest()
    raw = base64.b64decode(k)
    return int(bytes(a ^ b for a, b in zip(raw, ks)).decode("ascii"))


def build_item(iid: str, item: dict) -> dict:
    """공개본 항목 하나. 정답 해설·오답 이유는 각 보기 텍스트를 열쇠로 난독화한다."""
    opts = [str(o) for o in item["options"]]
    keys = [_nfc(o) for o in opts]
    correct = int(item["correct"])
    why = list(item.get("why_wrong") or [""] * len(opts))
    answer_html = md_to_html(item["answer"])
    payloads = []
    for i, key in enumerate(keys):
        if i == correct:
            payloads.append(obfuscate("A:" + answer_html, key))
        else:
            payloads.append(obfuscate("W:" + _inline(why[i] if i < len(why) else ""), key))
    return {
        "id": iid,
        "kind": item["kind"],
        "notebook": item["notebook"],
        "title": item["title"],
        "q": md_to_html(item["question"]),
        "opts": [_inline(o) for o in opts],
        "keys": keys,
        "p": payloads,
        "k": _mask_index(iid, correct),
        "hints": [md_to_html(h) for h in (item.get("hints") or [])],
    }


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 4. check / build
# ══════════════════════════════════════════════════════════════════════════════════════════════

def check(items: dict[str, dict], verbose: bool = True) -> list[str]:
    errors: list[str] = []
    for iid, it in items.items():
        where = f"{it.get('_file', '?')}:{iid}"
        if not ID_RE.match(iid):
            errors.append(f"{where}: id 형식 (소문자·숫자·하이픈)")
        for f in REQUIRED:
            if f not in it:
                errors.append(f"{where}: 필드 없음 `{f}`")
        if errors and errors[-1].startswith(where):
            continue
        if it["kind"] not in KINDS:
            errors.append(f"{where}: kind 는 mission|quiz")
        opts = it["options"]
        if not (3 <= len(opts) <= 6):
            errors.append(f"{where}: 보기는 3~6개 (지금 {len(opts)})")
        if len({_nfc(str(o)) for o in opts}) != len(opts):
            errors.append(f"{where}: 보기 텍스트가 서로 같으면 열쇠가 겹칩니다")
        c = it["correct"]
        if not isinstance(c, int) or not (0 <= c < len(opts)):
            errors.append(f"{where}: correct 는 0..{len(opts) - 1} 의 정수")
        why = it["why_wrong"]
        if len(why) != len(opts):
            errors.append(f"{where}: why_wrong 개수({len(why)}) ≠ 보기 개수({len(opts)})")
        else:
            for i, w in enumerate(why):
                if i != c and not str(w).strip():
                    errors.append(f"{where}: why_wrong[{i}] 이 비어 있음 (오답 이유는 모두 채움)")
        hints = it["hints"]
        if len(hints) != 3:
            errors.append(f"{where}: hints 는 3단계 (지금 {len(hints)})")
        else:
            if not URL_RE.search(hints[1]):
                errors.append(f"{where}: hints[1] 에 근거자료 링크(URL)가 없음")
            if "AI" not in hints[2] and "물어" not in hints[2]:
                errors.append(f"{where}: hints[2] 는 'AI에게 이렇게 물어보세요' 안내여야 함")
        for url in URL_RE.findall(" ".join(hints) + " " + it["answer"] + " " + it["question"]):
            if not re.match(r"^https?://[\w.-]+(:\d+)?(/|$)", url):
                errors.append(f"{where}: URL 형식 이상 `{url[:60]}`")
        if it["kind"] == "mission":
            if "____" not in it["question"]:
                errors.append(f"{where}: 미션 question 에 빈칸 `____` 이 없음")
            if "answer_code" not in it or not str(it["answer_code"]).strip():
                errors.append(f"{where}: 미션은 answer_code 필요")
            elif "____" in it["answer_code"]:
                errors.append(f"{where}: answer_code 에 `____` 이 남아 있음")
        # 난독화 왕복
        try:
            pub = build_item(iid, it)
            got = deobfuscate(pub["p"][c], pub["keys"][c])
            if not got or not got.startswith("A:"):
                errors.append(f"{where}: 정답 payload 왕복 실패")
            if _unmask_index(iid, pub["k"]) != c:
                errors.append(f"{where}: k 왕복 실패")
            for i in range(len(opts)):
                if i != c:
                    got = deobfuscate(pub["p"][i], pub["keys"][i])
                    if got is None or not got.startswith("W:"):
                        errors.append(f"{where}: 오답 payload[{i}] 왕복 실패")
                    if deobfuscate(pub["p"][c], pub["keys"][i]) is not None:
                        errors.append(f"{where}: 오답 열쇠로 정답이 열림 (보기 {i})")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{where}: build 실패 — {e}")
    if verbose:
        by_file: dict[str, int] = {}
        for it in items.values():
            by_file[it.get("_file", "?")] = by_file.get(it.get("_file", "?"), 0) + 1
        print(f"[quizkit] 문항 {len(items)}개: " + ", ".join(f"{k} {v}" for k, v in sorted(by_file.items())))
        for e in errors:
            print("  ✗", e)
        if not errors:
            print("  ✓ 검증 통과")
    return errors


def build(out: Path = PUBLIC) -> dict:
    items = load_bank()
    errors = check(items, verbose=True)
    if errors:
        raise SystemExit(f"[quizkit] 오류 {len(errors)}건 — public.json 을 만들지 않습니다")
    public = {"version": 1, "items": {iid: build_item(iid, it) for iid, it in items.items()}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(public, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[quizkit] 생성: {out.relative_to(ROOT)} ({out.stat().st_size // 1024}KB, {len(items)}문항)")
    return public


# ══════════════════════════════════════════════════════════════════════════════════════════════
# 5. 렌더링 — 노트북(display(HTML)) · 강의 페이지(make_site.py) 공용
# ══════════════════════════════════════════════════════════════════════════════════════════════

def assets_html() -> str:
    """quiz.css + quiz.js 를 인라인으로. 노트북 출력마다 넣어도 JS 쪽이 한 번만 초기화한다."""
    css = CSS_PATH.read_text(encoding="utf-8")
    js = JS_PATH.read_text(encoding="utf-8")
    return f"<style>{css}</style>\n<script>{js}</script>\n"


_counter = 0


def mount_html(iid: str, public: dict[str, dict] | None = None, mount_script: bool = True) -> str:
    """문항 하나의 카드 자리(<div class="qk" data-qk=…>). mount_script=True 면 바로 그리는 스크립트도 붙인다."""
    global _counter
    public = public if public is not None else load_public()
    if iid not in public:
        raise KeyError(f"QuizKit 에 없는 문항 id: {iid!r} (있는 것: {', '.join(sorted(public))})")
    _counter += 1
    dom_id = f"qk-{iid}-{_counter}"
    data = html.escape(json.dumps(public[iid], ensure_ascii=False, separators=(",", ":")), quote=True)
    out = f'<div class="qk" id="{dom_id}" data-qk="{data}"></div>\n'
    if mount_script:
        out += ("<script>(function(){var s=document.currentScript;var el=(s&&s.previousElementSibling&&s.previousElementSibling.classList"
                f"&&s.previousElementSibling.classList.contains('qk'))?s.previousElementSibling:document.getElementById('{dom_id}');"
                "function go(n){if(window.QuizKit&&el){window.QuizKit.mount(el);}else if(n>0){setTimeout(function(){go(n-1);},60);}}"
                "go(50);})();</script>\n")
    return out


_FIT_JS = """
<script>
(function () {
  // 주피터랩의 밝게/어둡게를 그대로 따른다 (같은 출처라 부모 문서를 읽을 수 있다)
  function theme() {
    try {
      var b = window.parent && window.parent.document && window.parent.document.body;
      var v = b && b.getAttribute('data-jp-theme-light');
      if (v === 'true') { document.documentElement.setAttribute('data-theme', 'light'); }
      else if (v === 'false') { document.documentElement.setAttribute('data-theme', 'dark'); }
    } catch (e) {}
  }
  theme(); setInterval(theme, 2000);
  // 카드의 실제 높이를 재서 iframe 을 거기에 맞춘다.
  //
  // ⚠️ documentElement.scrollHeight 를 쓰면 안 된다. 그 값은 최소한 뷰포트(=iframe) 높이라서
  //    "높이를 키운다 → scrollHeight 가 그만큼 커진다 → 또 키운다" 로 끝없이 자란다.
  //    카드가 계속 아래로 늘어나 화면이 저절로 밀려 올라가는 것이 그 증상이다.
  //    그래서 내용 요소(.qk)의 높이만 재고, 값이 실제로 달라졌을 때만 쓴다.
  var last = -1;
  function measure() {
    var el = document.querySelector('.qk');
    if (!el) { return 0; }
    return Math.ceil(el.getBoundingClientRect().height) + 4;   // 4px 는 테두리 여유
  }
  function fit() {
    try {
      var fe = window.frameElement;
      if (!fe) { return; }
      var h = measure();
      if (!h || Math.abs(h - last) <= 1) { return; }   // 1px 이내면 건드리지 않는다 (되먹임 차단)
      last = h;
      fe.style.height = h + 'px';
    } catch (e) {}
  }
  function watch() {
    var el = document.querySelector('.qk');
    if (!el) { return setTimeout(watch, 60); }
    if (window.ResizeObserver) { new ResizeObserver(fit).observe(el); }   // documentElement 가 아니라 카드
    new MutationObserver(fit).observe(el, { subtree: true, childList: true, attributes: true });
    fit();
  }
  watch();
  window.addEventListener('load', fit);
  document.addEventListener('click', function () { setTimeout(fit, 30); });
  setInterval(fit, 1500);   // 안전망. 값이 같으면 아무것도 쓰지 않는다
  fit();
})();
</script>
"""


def render(iid: str, assets: bool = True) -> str:
    """노트북 셀 하나에 넣을 HTML.

    JupyterLab 은 `display(HTML(...))` 안의 <script> 를 실행하지 않는 판이 있어, 카드를 **iframe(srcdoc)**
    안에 넣는다. srcdoc 은 부모와 같은 출처라 안에서 `window.frameElement` 로 높이를 맞출 수 있고,
    localStorage 도 노트북 페이지와 같은 저장소를 쓴다.
    """
    inner = ("<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
             "<style>html,body{margin:0;padding:0;background:transparent}</style>"
             + (assets_html() if assets else "")
             + "</head><body>" + mount_html(iid) + _FIT_JS + "</body></html>")
    srcdoc = html.escape(inner, quote=True)
    # <div> 로 감싼다 — IPython 은 <iframe 으로 시작하는 HTML 에 "IFrame 을 쓰라"는 경고를 띄운다.
    return ('<div class="qk-host" style="margin:2px 0">'
            f'<iframe title="QuizKit {html.escape(iid)}" srcdoc="{srcdoc}" allow="clipboard-write" '
            'style="width:100%;min-width:0;height:340px;border:0;display:block;overflow:hidden" '
            'scrolling="no" loading="eager"></iframe></div>')


def show(iid: str) -> None:
    """주피터에서 문항을 그린다: quizkit.show("m-tc-1")."""
    from IPython.display import HTML, display  # noqa: PLC0415
    display(HTML(render(iid)))


def demo_html(public: dict[str, dict] | None = None) -> str:
    public = public if public is not None else load_public()
    cards = "".join(f"<h3 style='font-family:sans-serif;margin:28px 0 6px'>{html.escape(iid)} · {html.escape(it['notebook'])}</h3>"
                    + mount_html(iid, public, mount_script=False) for iid, it in public.items())
    return ("<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>QuizKit demo</title>"
            + assets_html() + "</head><body style='max-width:900px;margin:20px auto;padding:0 16px'>"
            "<h1 style='font-family:sans-serif'>QuizKit — 전 문항 점검</h1>" + cards +
            "<script>QuizKit.mountAll();</script></body></html>")


# ══════════════════════════════════════════════════════════════════════════════════════════════

def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "check"
    if cmd == "check":
        return 1 if check(load_bank()) else 0
    if cmd == "build":
        build()
        return 0
    if cmd == "demo":
        out = ROOT / "output" / "quizkit-demo.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(demo_html(), encoding="utf-8")
        print(f"[quizkit] 데모: {out}")
        return 0
    if cmd == "render":
        print(render(argv[2]))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))

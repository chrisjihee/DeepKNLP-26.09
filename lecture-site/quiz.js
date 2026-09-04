/* QuizKit — 미션·퀴즈 객관식 GUI. 주피터 노트북 출력과 강의 페이지(index.html)가 같은 파일을 쓴다.
 *
 * 문항 데이터는 tools/quizkit.py build 가 만든 quiz/public.json 의 항목 하나(JSON)이며, 카드 요소의
 * data-qk 속성에 들어 있다. 정답·오답 이유는 보기 텍스트를 열쇠로 한 XOR 난독화(§4 of docs/QUIZKIT-SPEC.md)
 * 상태이고, 보기를 누르면 그 보기 텍스트로 복호를 시도해 "OK:A:"(정답) / "OK:W:"(오답 이유)를 가른다.
 *
 * 외부 스크립트·폰트 없음. crypto.subtle 을 쓰지 않는다(http 로 여는 주피터에서는 없다) — SHA-256 은 순수 JS.
 */
(function (global) {
  if (global.QuizKit && global.QuizKit.version >= 1) { return; }

  // ── SHA-256 (순수 JS, 문자열 → Uint8Array 32바이트) ─────────────────────────────────────────
  var K = [];
  (function () {
    var primes = 0, n = 2;
    while (primes < 64) {
      var isPrime = true;
      for (var f = 2; f * f <= n; f++) { if (n % f === 0) { isPrime = false; break; } }
      if (isPrime) { K[primes++] = (Math.pow(n, 1 / 3) % 1 * 4294967296) >>> 0; }
      n++;
    }
  })();
  var H0 = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
  function rotr(x, n) { return (x >>> n) | (x << (32 - n)); }
  function sha256(bytes) {
    var l = bytes.length, bitLen = l * 8;
    var padLen = ((l + 9 + 63) >> 6) << 6;
    var m = new Uint8Array(padLen);
    m.set(bytes); m[l] = 0x80;
    var dv = new DataView(m.buffer);
    dv.setUint32(padLen - 8, Math.floor(bitLen / 4294967296)); dv.setUint32(padLen - 4, bitLen >>> 0);
    var H = H0.slice(), w = new Int32Array(64);
    for (var off = 0; off < padLen; off += 64) {
      var i;
      for (i = 0; i < 16; i++) { w[i] = dv.getInt32(off + i * 4); }
      for (i = 16; i < 64; i++) {
        var s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >>> 3);
        var s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >>> 10);
        w[i] = (w[i - 16] + s0 + w[i - 7] + s1) | 0;
      }
      var a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
      for (i = 0; i < 64; i++) {
        var S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        var ch = (e & f) ^ (~e & g);
        var t1 = (h + S1 + ch + K[i] + w[i]) | 0;
        var S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        var maj = (a & b) ^ (a & c) ^ (b & c);
        var t2 = (S0 + maj) | 0;
        h = g; g = f; f = e; e = (d + t1) | 0; d = c; c = b; b = a; a = (t1 + t2) | 0;
      }
      H[0] = (H[0] + a) | 0; H[1] = (H[1] + b) | 0; H[2] = (H[2] + c) | 0; H[3] = (H[3] + d) | 0;
      H[4] = (H[4] + e) | 0; H[5] = (H[5] + f) | 0; H[6] = (H[6] + g) | 0; H[7] = (H[7] + h) | 0;
    }
    var out = new Uint8Array(32), ov = new DataView(out.buffer);
    for (var j = 0; j < 8; j++) { ov.setInt32(j * 4, H[j]); }
    return out;
  }

  var enc = new TextEncoder();
  var decFatal = new TextDecoder('utf-8', { fatal: true });
  function utf8(s) { return enc.encode(s); }
  function b64bytes(s) { var bin = atob(s), u = new Uint8Array(bin.length); for (var i = 0; i < bin.length; i++) { u[i] = bin.charCodeAt(i); } return u; }
  function keystream(key, n) {
    var out = new Uint8Array(n), pos = 0, i = 0;
    while (pos < n) {
      var block = sha256(utf8(key + ':' + i));
      var take = Math.min(32, n - pos);
      out.set(block.subarray(0, take), pos); pos += take; i++;
    }
    return out;
  }
  // 복호. 성공하면 'OK:' 뒤의 문자열, 실패하면 null
  function decode(b64, key) {
    try {
      var ct = b64bytes(b64), ks = keystream(key.normalize('NFC'), ct.length);
      var pt = new Uint8Array(ct.length);
      for (var i = 0; i < ct.length; i++) { pt[i] = ct[i] ^ ks[i]; }
      var s = decFatal.decode(pt);
      return s.indexOf('OK:') === 0 ? s.slice(3) : null;
    } catch (e) { return null; }
  }

  // ── 상태 저장 (브라우저에만) ────────────────────────────────────────────────────────────────
  function loadState(id) { try { return JSON.parse(localStorage.getItem('qk:' + id) || 'null') || {}; } catch (e) { return {}; } }
  function saveState(id, st) { try { localStorage.setItem('qk:' + id, JSON.stringify(st)); } catch (e) {} }
  function clearState(id) { try { localStorage.removeItem('qk:' + id); } catch (e) {} }

  var MARKS = ['①', '②', '③', '④', '⑤', '⑥'];
  var COUNTDOWN = 10;    // 정답 바로 보기 — 확인 2회 뒤 기다리는 시간(초)
  var WRONG_WAIT = 5;    // 오답을 고른 뒤 채점까지 기다리는 시간(초) — 정답은 곧바로 보여준다
  var HINT_WAIT = 5;     // 힌트 보기 버튼을 누른 뒤 열리기까지 기다리는 시간(초)

  function el(tag, cls, html) { var e = document.createElement(tag); if (cls) { e.className = cls; } if (html != null) { e.innerHTML = html; } return e; }
  function text(tag, cls, t) { var e = document.createElement(tag); if (cls) { e.className = cls; } e.textContent = t; return e; }

  function copyText(t, btn) {
    function done(ok) { var old = btn.textContent; btn.textContent = ok ? '복사됨 ✓' : '복사 실패'; setTimeout(function () { btn.textContent = old; }, 1400); }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(t).then(function () { done(true); }, function () { done(fallback()); });
    } else { done(fallback()); }
    function fallback() {
      try {
        var ta = document.createElement('textarea'); ta.value = t; ta.style.position = 'fixed'; ta.style.opacity = '0';
        document.body.appendChild(ta); ta.select(); var ok = document.execCommand('copy'); document.body.removeChild(ta); return ok;
      } catch (e) { return false; }
    }
  }
  function addCopyButtons(root) {
    var pres = root.querySelectorAll('pre');
    for (var i = 0; i < pres.length; i++) {
      (function (pre) {
        // 버튼을 pre 의 자식이 아니라 형제로 둔다 — 코드를 드래그해 선택·복사할 때
        // 버튼 글자("복사")가 선택 범위에 섞여 붙여넣기가 깨지는 것을 막는다.
        var wrap = document.createElement('div'); wrap.className = 'qk-pre-wrap';
        pre.parentNode.insertBefore(wrap, pre);
        wrap.appendChild(pre);
        var b = text('button', 'qk-copy', '복사'); b.type = 'button';
        b.addEventListener('click', function () { copyText(pre.textContent.replace(/^\n+|\n+$/g, ''), b); });
        wrap.appendChild(b);
      })(pres[i]);
    }
  }

  // ── 카드 하나 ───────────────────────────────────────────────────────────────────────────────
  function mount(host) {
    if (!host || host.getAttribute('data-qk-mounted')) { return; }
    var item;
    try { item = JSON.parse(host.getAttribute('data-qk')); } catch (e) { host.textContent = 'QuizKit: 문항 데이터를 읽을 수 없습니다.'; return; }
    host.setAttribute('data-qk-mounted', '1');
    host.classList.add('qk');
    host.innerHTML = '';

    var st = loadState(item.id);
    var kind = item.kind === 'mission' ? '미션' : '퀴즈';
    var card = el('div', 'qk-card'); card.setAttribute('data-kind', item.kind);

    var head = el('div', 'qk-head');
    head.appendChild(text('span', 'qk-badge', kind));
    head.appendChild(text('span', 'qk-id', item.id));
    head.appendChild(text('span', 'qk-title', item.title || ''));
    card.appendChild(head);
    card.appendChild(el('div', 'qk-q', item.q));

    var opts = el('div', 'qk-opts' + (allShort(item.opts) ? ' qk-two' : ''));
    var optBtns = [];
    item.opts.forEach(function (o, i) {
      var b = el('button', 'qk-opt'); b.type = 'button'; b.setAttribute('data-i', i);
      b.appendChild(text('span', 'qk-mark', MARKS[i] || (i + 1)));
      b.appendChild(el('span', 'qk-otext', o));
      b.addEventListener('click', function () { choose(i); });
      opts.appendChild(b); optBtns.push(b);
    });
    card.appendChild(opts);

    var bar = el('div', 'qk-bar');
    var hintBtn = text('button', 'qk-btn', ''); hintBtn.type = 'button';
    var revealBtn = text('button', 'qk-btn qk-danger', '정답 바로 보기'); revealBtn.type = 'button';
    var status = el('span', 'qk-status');
    bar.appendChild(hintBtn); bar.appendChild(revealBtn); bar.appendChild(status);
    card.appendChild(bar);

    var msg = el('div', 'qk-msg');
    var hints = el('div', 'qk-hints');
    var ans = el('div', 'qk-ans'); ans.hidden = true;
    card.appendChild(msg); card.appendChild(hints); card.appendChild(ans);
    host.appendChild(card);

    var nHints = (item.hints || []).length;
    var stage = 0, timer = null, gTimer = null, hTimer = null;   // 공개 · 채점 · 힌트
    st.hints = st.hints || 0; st.wrong = st.wrong || [];

    function allShort(arr) { return arr.every(function (o) { return o.replace(/<[^>]+>/g, '').length <= 28; }); }
    function updateHintBtn() {
      if (st.hints >= nHints) { hintBtn.textContent = nHints ? '힌트 모두 열림 (' + nHints + '/' + nHints + ')' : '힌트 없음'; hintBtn.disabled = true; }
      else { hintBtn.textContent = '힌트 보기 (' + st.hints + '/' + nHints + ')'; hintBtn.disabled = false; }
    }
    var HINT_LABEL = ['힌트 1 · 개념', '힌트 2 · 근거자료 · 검색', '힌트 3 · AI에게 이렇게 물어보세요'];
    function showHint(k) {   // 0-based
      if (k >= nHints || hints.children.length > k) { return; }
      var h = el('div', 'qk-hint'); h.appendChild(text('span', 'qk-hl', HINT_LABEL[k] || ('힌트 ' + (k + 1))));
      h.appendChild(el('div', '', item.hints[k]));
      hints.appendChild(h);
    }
    function openHints(n) { for (var k = 0; k < n; k++) { showHint(k); } }
    function nextHint() {   // 실제로 여는 것. 오답 뒤 자동으로 열릴 때도 이 길로 온다.
      if (st.hints < nHints) { st.hints++; showHint(st.hints - 1); saveState(item.id, st); }
      updateHintBtn();
    }
    // 버튼으로 여는 힌트는 곧바로 열리지 않는다 — 연달아 눌러 힌트를 다 열어 버리는 것을 막는다.
    function requestHint() {
      if (hTimer || gTimer || timer || st.hints >= nHints) { return; }
      var left = HINT_WAIT;
      hintBtn.disabled = true;
      msg.innerHTML = '';
      var n = el('div', 'qk-note qk-n-ask');
      var line = el('div', ''); n.appendChild(line);
      var row = el('div', 'qk-row');
      var cancel = text('button', 'qk-btn qk-primary', '취소 — 조금 더 생각해 볼게요'); cancel.type = 'button';
      cancel.addEventListener('click', function () {
        clearInterval(hTimer); hTimer = null; msg.innerHTML = ''; updateHintBtn();
      });
      row.appendChild(cancel); n.appendChild(row); msg.appendChild(n);
      function tick() {
        line.innerHTML = '<span class="qk-count">' + left + '초</span> 뒤에 힌트가 열립니다. '
          + '그동안 문제를 다시 읽어 보세요 — 힌트 없이 풀어야 남는 것이 있습니다.';
        if (left <= 0) {
          clearInterval(hTimer); hTimer = null;
          msg.innerHTML = ''; nextHint();
          return;
        }
        left--;
      }
      tick(); hTimer = setInterval(tick, 1000);
    }
    hintBtn.addEventListener('click', requestHint);

    function lockAll() { optBtns.forEach(function (b) { b.disabled = true; }); }
    function showAnswer(idx, body, how) {   // how: 'solved' | 'opened'
      lockAll(); stopTimer(); msg.innerHTML = '';
      optBtns.forEach(function (b, i) { if (i === idx) { b.classList.add('qk-right'); b.querySelector('.qk-mark').textContent = '✓'; } else if (!b.classList.contains('qk-wrong')) { b.classList.add('qk-dim'); } });
      ans.innerHTML = '';
      ans.appendChild(text('div', 'qk-al', how === 'solved' ? '정답 · 해설' : '정답 · 해설 (열어 봄)'));
      ans.appendChild(el('div', '', body));
      addCopyButtons(ans);
      ans.hidden = false;
      revealBtn.disabled = true; hintBtn.disabled = true;
      card.classList.add(how === 'solved' ? 'qk-done' : 'qk-opened');
      status.className = 'qk-status ' + (how === 'solved' ? 'qk-good' : 'qk-warnc');
      status.textContent = how === 'solved' ? (st.wrong.length ? '정답입니다 (오답 ' + st.wrong.length + '회)' : '정답입니다') : '정답을 열어 보았습니다';
      var reset = text('button', 'qk-reset', '다시 풀기'); reset.type = 'button';
      reset.addEventListener('click', function () { clearState(item.id); host.removeAttribute('data-qk-mounted'); mount(host); });
      status.appendChild(reset);
    }
    function markWrong(i, why) {
      var b = optBtns[i]; b.disabled = true; b.classList.add('qk-wrong'); b.querySelector('.qk-mark').textContent = '✗';
      if (why != null) { msg.innerHTML = ''; var n = el('div', 'qk-note qk-n-bad'); n.innerHTML = '<b>' + (MARKS[i] || (i + 1)) + ' 는 아닙니다.</b> ' + (why || ''); msg.appendChild(n); }
    }
    // 보기를 누르면 곧바로 맞았는지 알려 주지 않는다. 기다리는 동안 스스로 근거를 대 보게 한다.
    // 곧바로 알려 주면 보기를 차례로 눌러 정답을 찾아내는 것이 가장 빠른 길이 되어 버린다.
    function choose(i) {
      if (gTimer || timer) { return; }
      if (hTimer) { clearInterval(hTimer); hTimer = null; }   // 힌트를 기다리던 중이면 그만둔다

      var wasOpen = optBtns.filter(function (b) { return !b.disabled; });
      optBtns.forEach(function (b) { b.disabled = true; });
      hintBtn.disabled = true; revealBtn.disabled = true;

      var plain = decode(item.p[i], item.keys[i]);
      if (plain == null) {
        wasOpen.forEach(function (b) { b.disabled = false; }); updateHintBtn();
        msg.innerHTML = ''; msg.textContent = '확인할 수 없습니다 — 문항 데이터가 손상됐습니다. 강사에게 알려 주세요.';
        return;
      }
      var isCorrect = plain.indexOf('A:') === 0;

      msg.innerHTML = '';
      var n = el('div', 'qk-note qk-n-ask');
      var line = el('div', ''); n.appendChild(line);
      msg.appendChild(n);

      function finish() {
        clearInterval(gTimer); gTimer = null;
        revealBtn.disabled = false;
        if (isCorrect) {
          st.done = true; saveState(item.id, st);
          showAnswer(i, plain.slice(2), 'solved');
          return;
        }
        // 오답 — 고른 것만 잠그고 나머지는 다시 열어 준다
        wasOpen.forEach(function (b) { if (b !== optBtns[i]) { b.disabled = false; } });
        if (st.wrong.indexOf(i) < 0) { st.wrong.push(i); }
        markWrong(i, plain.slice(2));
        nextHint();   // 오답마다 힌트 하나가 자동으로 열린다 (이미 기다린 뒤이므로 곧바로)
        saveState(item.id, st);
        updateHintBtn();
        var remain = optBtns.filter(function (b) { return !b.disabled; }).length;
        if (remain === 1) { msg.appendChild(el('div', 'qk-note qk-n-ask', '남은 보기가 하나입니다 — 왜 그것이 답인지 한 줄로 설명해 보고 누르세요.')); }
      }

      if (isCorrect) { finish(); return; }   // 정답은 곧바로 보여준다

      var left = WRONG_WAIT;
      function tick() {
        line.innerHTML = '<b>' + (MARKS[i] || (i + 1)) + '</b> 를 고르셨습니다. '
          + '<span class="qk-count">' + left + '초</span> 뒤에 채점합니다 — '
          + '그동안 <b>왜 그 보기라고 생각했는지</b> 한 줄로 말해 보세요.';
        if (left <= 0) { finish(); return; }
        left--;
      }
      tick(); gTimer = setInterval(tick, 1000);
    }

    // ── 정답 바로 보기: 확인 2회 → 10초 → 공개 ─────────────────────────────────────────────
    function correctIndex() {
      try {
        var kb = b64bytes(item.k), ks = sha256(utf8(item.id)), s = '';
        for (var i = 0; i < kb.length; i++) { s += String.fromCharCode(kb[i] ^ ks[i]); }
        var idx = parseInt(s, 10); return isNaN(idx) ? -1 : idx;
      } catch (e) { return -1; }
    }
    function stopTimer() { if (timer) { clearInterval(timer); timer = null; } }
    function askBox(html, yesLabel) {
      msg.innerHTML = '';
      var n = el('div', 'qk-note qk-n-ask'); n.appendChild(el('div', '', html));
      var row = el('div', 'qk-row');
      var no = text('button', 'qk-btn qk-primary', '더 생각해 볼게요'); no.type = 'button';
      var yes = text('button', 'qk-btn', yesLabel); yes.type = 'button';
      no.addEventListener('click', function () { stage = 0; stopTimer(); msg.innerHTML = ''; msg.appendChild(el('div', 'qk-note qk-n-ask', '좋습니다. 막히면 <b>힌트 보기</b>를 먼저 눌러 보세요 — 개념 → 근거자료 → AI에게 물어볼 질문 순서로 열립니다.')); });
      yes.addEventListener('click', advance);
      row.appendChild(no); row.appendChild(yes); n.appendChild(row); msg.appendChild(n);
    }
    function advance() {
      stage++;
      if (stage === 1) { askBox('정말 정답을 보시겠습니까? 힌트를 먼저 보는 쪽을 권합니다.', '보겠습니다'); }
      else if (stage === 2) { askBox('한 번만 더 확인합니다 — 보기 중 <b>무엇이 아닌지</b> 하나라도 지울 수 있나요? 더 고민하지 않고 답을 보시겠습니까?', '그래도 보겠습니다'); }
      else {
        var left = COUNTDOWN;
        msg.innerHTML = '';
        var n = el('div', 'qk-note qk-n-ask');
        var line = el('div', ''); n.appendChild(line);
        var row = el('div', 'qk-row'); var cancel = text('button', 'qk-btn qk-primary', '취소 — 더 생각해 볼게요'); cancel.type = 'button';
        cancel.addEventListener('click', function () { stopTimer(); stage = 0; msg.innerHTML = ''; });
        row.appendChild(cancel); n.appendChild(row); msg.appendChild(n);
        function tick() {
          line.innerHTML = '<span class="qk-count">' + left + '초</span> 뒤에 열립니다. 그동안 보기를 다시 읽어 보세요.';
          if (left <= 0) {
            stopTimer(); var idx = correctIndex();
            var body = idx >= 0 ? decode(item.p[idx], item.keys[idx]) : null;
            if (body == null || body.indexOf('A:') !== 0) { msg.textContent = '정답을 열 수 없습니다 — 문항 데이터가 손상됐습니다. 강사에게 알려 주세요.'; return; }
            st.opened = true; saveState(item.id, st);
            showAnswer(idx, body.slice(2), 'opened');
          }
          left--;
        }
        tick(); timer = setInterval(tick, 1000);
      }
    }
    revealBtn.addEventListener('click', function () { if (stage === 0 && !timer && !gTimer && !hTimer) { advance(); } });

    // ── 저장된 상태 복원 ──────────────────────────────────────────────────────────────────
    st.wrong.forEach(function (i) { if (optBtns[i]) { markWrong(i, null); } });
    openHints(st.hints);
    updateHintBtn();
    if (st.done || st.opened) {
      var idx = correctIndex(), body = idx >= 0 ? decode(item.p[idx], item.keys[idx]) : null;
      if (body != null && body.indexOf('A:') === 0) { showAnswer(idx, body.slice(2), st.done ? 'solved' : 'opened'); }
    } else if (st.wrong.length) {
      status.textContent = '오답 ' + st.wrong.length + '회 — 이어서 풀어 보세요';
    }
  }

  function mountAll(root) {
    var nodes = (root || document).querySelectorAll('.qk[data-qk]:not([data-qk-mounted])');
    for (var i = 0; i < nodes.length; i++) { mount(nodes[i]); }
  }

  global.QuizKit = { version: 1, mount: mount, mountAll: mountAll, decode: decode, sha256: sha256 };
  if (document.readyState === 'loading') { document.addEventListener('DOMContentLoaded', function () { mountAll(); }); }
  else { mountAll(); }
})(typeof window !== 'undefined' ? window : this);

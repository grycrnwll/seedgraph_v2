// The text fold shared by quote highlight, selection anchoring and the highlight
// measurement (v3 stage 3 design §3.2). A port of src/seedgraph/library/textfold.py
// (and of the mojibake fold in library/mojibake.py); both run the shared vectors in
// tests/fixtures/textfold_vectors.json, so they cannot drift.
//
// fold(text) -> {text, starts, ends, rawSpan(start, end)}: the folded string and, per
// folded character, the raw [start, end) it came from. Offsets are JavaScript string
// (UTF-16) indices; they equal Python's code-point offsets for text without astral
// characters.
//
// findPassage(haystack, needle) -> [[start, end], ...] over folded strings: every full
// hit, else (needles of 30+ characters) three 30-character windows in order within
// 1.5x the needle's length.

export const WINDOW = 30;
export const SPREAD = 1.5;

// Step 1: what the PDF does not print (page delimiters, math, link targets, tags,
// line-end hyphenation, soft hyphen and zero-width characters).
const REMOVE = new RegExp(
  "^[ \\t]*\\{[0-9]+\\}-{3,}[ \\t\\r]*$" +
    "|\\$\\$[\\s\\S]*?\\$\\$" +
    "|\\$[^$\\n]+\\$" +
    '|\\]\\([^)\\s]*(?:\\s+"[^"]*")?\\)' +
    "|<\\/?[A-Za-z][A-Za-z0-9]*(?:\\s[^<>]*)?\\/?>" +
    "|-[ \\t]*\\r?\\n[ \\t]*(?=\\p{L})" +
    "|[\\u00ad\\u200b-\\u200d\\u2060\\ufeff]",
  "gmu",
);

// Step 2: lost-ligature mojibake (families A and B), in context.
const MOJIBAKE = new RegExp(
  "(?<ffi>(?<=[a-z])\\u00a2 ?(?=[a-z])|(?<![A-Za-z])\\u00a2(?=[a-z]{2}))" +
    "|(?<fi>(?<=[a-z])[\\u00d6\\u2026](?=[a-z])|(?<![A-Za-z])[\\u00d6\\u2026](?=[a-z]{2}))" +
    "|(?<ff>(?<=[a-z])[\\u00a7\\u00a4](?=[a-z])|(?<![A-Za-z])[\\u00a7\\u00a4](?=[a-z]{2}))" +
    "|(?<fl>(?<=[a-z])\\u00e1(?=[a-z]))" +
    "|(?<apos>(?<=[a-z])\\u00ed(?=s(?![\\p{L}\\p{N}\\p{M}_])))",
  "gu",
);
const MOJIBAKE_REPL = { ffi: "ffi", fi: "fi", ff: "ff", fl: "fl", apos: "'" };

const MARK = /^\p{M}$/u;
const LETTER = /^\p{L}$/u;

function keep(ch) {
  return (ch >= "0" && ch <= "9") || LETTER.test(ch);
}

function mojibakeFold(text) {
  const out = [];
  const spans = [];
  let cursor = 0;
  for (const m of text.matchAll(MOJIBAKE)) {
    for (let i = cursor; i < m.index; i++) {
      out.push(text[i]);
      spans.push([i, i + 1]);
    }
    const group = Object.keys(m.groups).find((k) => m.groups[k] !== undefined);
    for (const ch of MOJIBAKE_REPL[group]) {
      out.push(ch);
      spans.push([m.index, m.index + m[0].length]);
    }
    cursor = m.index + m[0].length;
  }
  for (let i = cursor; i < text.length; i++) {
    out.push(text[i]);
    spans.push([i, i + 1]);
  }
  return { chars: out, spans };
}

export function fold(text) {
  // 1. drop unprinted markup, keeping each surviving unit's raw index
  const kept = [];
  const kmap = [];
  let cursor = 0;
  for (const m of text.matchAll(REMOVE)) {
    for (let i = cursor; i < m.index; i++) {
      kept.push(text[i]);
      kmap.push(i);
    }
    cursor = m.index + m[0].length;
  }
  for (let i = cursor; i < text.length; i++) {
    kept.push(text[i]);
    kmap.push(i);
  }
  // 2. mojibake, composing its spans with kmap
  const mj = mojibakeFold(kept.join(""));
  const raw = mj.spans.map(([s, e]) => [kmap[s], kmap[e - 1] + 1]);
  // Regroup UTF-16 units into code points (a surrogate pair is one character).
  const cps = [];
  for (let i = 0; i < mj.chars.length; i++) {
    const c = mj.chars[i];
    const code = c.charCodeAt(0);
    if (code >= 0xd800 && code <= 0xdbff && i + 1 < mj.chars.length) {
      cps.push({ ch: c + mj.chars[i + 1], start: raw[i][0], end: raw[i + 1][1] });
      i++;
    } else {
      cps.push({ ch: c, start: raw[i][0], end: raw[i][1] });
    }
  }
  // 3 + 4. per cluster: NFKC, lower case, filter
  let outText = "";
  const starts = [];
  const ends = [];
  let i = 0;
  while (i < cps.length) {
    let j = i + 1;
    while (j < cps.length && MARK.test(cps[j].ch)) j++;
    const start = cps[i].start;
    const end = cps[j - 1].end;
    const cluster = cps.slice(i, j).map((c) => c.ch).join("").normalize("NFKC");
    for (const ch of cluster) {
      for (let lc of ch.toLowerCase()) {
        if (lc === "ς") lc = "σ"; // final sigma
        if (keep(lc)) {
          outText += lc;
          for (let u = 0; u < lc.length; u++) {
            starts.push(start);
            ends.push(end);
          }
        }
      }
    }
    i = j;
  }
  return {
    text: outText,
    starts,
    ends,
    rawSpan(s, e) {
      if (!(0 <= s && s < e && e <= outText.length)) {
        throw new RangeError(`folded range [${s}, ${e}) is empty or out of bounds`);
      }
      return [starts[s], ends[e - 1]];
    },
  };
}

function allHits(haystack, needle) {
  const hits = [];
  let pos = haystack.indexOf(needle);
  while (pos !== -1) {
    hits.push([pos, pos + needle.length]);
    pos = haystack.indexOf(needle, pos + 1);
  }
  return hits;
}

export function findPassage(haystack, needle) {
  if (!needle) return [];
  const hits = allHits(haystack, needle);
  const n = needle.length;
  if (hits.length || n < WINDOW) return hits;
  const mid = Math.floor(n / 2);
  const half = Math.floor(WINDOW / 2);
  const wStart = needle.slice(0, WINDOW);
  const wMid = needle.slice(mid - half, mid - half + WINDOW);
  const wEnd = needle.slice(n - WINDOW);
  const out = [];
  let a = haystack.indexOf(wStart);
  while (a !== -1) {
    if (!out.length || a >= out[out.length - 1][1]) {
      const b = haystack.indexOf(wMid, a);
      const c = b !== -1 ? haystack.indexOf(wEnd, b) : -1;
      if (c !== -1 && c + WINDOW - a <= SPREAD * n) out.push([a, c + WINDOW]);
    }
    a = haystack.indexOf(wStart, a + 1);
  }
  return out;
}

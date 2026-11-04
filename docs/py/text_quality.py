"""A hand-written text-quality network: ~100k parameters, no training, no vocabulary.

Every weight in this file was set by reasoning about what separates usable text from
junk, in the spirit of "design the weights instead of growing them". The network is a
real neural network (embedding -> conv -> ReLU-RNN -> pooling -> 4-layer MLP) and can be
exported to PyTorch, but almost all of its ~100k parameters are zero: the logic lives in
a few thousand non-zero weights, and `explain()` prints which units fired.

It is language-agnostic by construction. The input is not words but per-character
Unicode attributes (letter / case / digit / punctuation class / script group / ...), so
the same weights score Swedish, Georgian, Japanese, Arabic or Hindi. Script-specific
expectations (no spaces in Chinese, no case in Arabic, no sentence punctuation in Thai)
are handled by gating units that switch a detector off when the text's script makes it
meaningless.

What it measures is *document quality* in the pretraining-filter sense (Gopher / C4 /
FineWeb-style rules): is this clean, well-formed running text, or boilerplate, tables,
markup, mojibake, repetition, shouting, fragments? It does not model grammar; a
character-level network with hand-set weights cannot know that "hunden springer" is
right and "hunden springa" is wrong. For that you still need a trained model.

Output: a regression score in [0, 100]. Roughly: >= 80 clean prose, 60-80 usable but
flawed, 40-60 doubtful, < 40 junk.

    >>> from handnet.text_quality import HandNet
    >>> net = HandNet()
    >>> net.score("Hunden springer i parken. Den är glad.")
    >>> net.explain(text)        # which detectors fired, in points
    >>> net.n_params, net.n_nonzero
"""
from __future__ import annotations

import math
import unicodedata
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------------- shapes
A = 40       # character attributes (input dim)
D = 40       # embedding dim
K = 5        # conv width
C = 48       # conv channels
H = 16       # RNN counters
F = C + 2 * H + 2          # pooled features: mean(conv), max(rnn), mean(rnn), loglen, bias
H1, H2, H3, H4 = 512, 86, 16, 2   # MLP widths
MAX_CHARS = 20000          # longer documents are scored on their first MAX_CHARS chars

# ----------------------------------------------------------------------------- attributes
ATTR = [
    "letter", "upper", "lower", "digit", "space", "newline", "otherws",
    "terminal", "clause", "quote", "open", "close", "dash", "symbol", "markup",
    "control", "garbage", "combining", "emoji", "ellipsis",
    "latin", "cyrillic", "greek", "abjad", "indic", "han", "kana", "hangul", "seasian",
    "caucasian", "otherletter",
    "same_prev", "moji_lead", "moji_tail", "bias", "slash", "caseless", "period_comma",
    "_r38", "_r39",
]
AI = {name: i for i, name in enumerate(ATTR)}

_TERMINAL = set(".!?。！？؟।॥։።۔܀")
_CLAUSE = set(",;:、，；：؛،")
_QUOTE = set("\"'«»“”‘’„‚‹›「」『』")
_OPEN, _CLOSE = set("([{（［｛"), set(")]}）］｝")
_DASH = set("-–—−‐‑")
_MARKUP = set("<>{}|\\^~=_#*@`")
_MOJI_LEAD = set("ÃÂâÐÑÎÏ")
_MOJI_TAIL = set("€™‚ƒ„…†‡ˆ‰Š‹ŒŽ‘’“”•–—˜š›œžŸ¡¢£¤¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿")


def _script(cp: int) -> str:
    if cp < 0x0250 or 0x1E00 <= cp < 0x1F00 or 0xFF21 <= cp < 0xFF5B:
        return "latin"
    if 0x0370 <= cp < 0x0400 or 0x1F00 <= cp < 0x2000:
        return "greek"
    if 0x0400 <= cp < 0x0530:
        return "cyrillic"
    if 0x0530 <= cp < 0x0590 or 0x10A0 <= cp < 0x1100 or 0x2D00 <= cp < 0x2D30:
        return "caucasian"        # Armenian, Georgian
    if 0x0590 <= cp < 0x0900:
        return "abjad"            # Hebrew, Arabic, Syriac, Thaana
    if 0x0900 <= cp < 0x0E00:
        return "indic"            # Devanagari .. Sinhala
    if 0x0E00 <= cp < 0x0F00 or 0x1000 <= cp < 0x10A0 or 0x1780 <= cp < 0x1800:
        return "seasian"          # Thai, Lao, Myanmar, Khmer (no spaces between words)
    if 0x3040 <= cp < 0x3100 or 0x31F0 <= cp < 0x3200 or 0xFF66 <= cp < 0xFFA0:
        return "kana"
    if 0x1100 <= cp < 0x1200 or 0x3130 <= cp < 0x3190 or 0xAC00 <= cp < 0xD7B0:
        return "hangul"
    if 0x2E80 <= cp < 0x3040 or 0x3100 <= cp < 0x31F0 or 0x3400 <= cp < 0xA000 \
            or 0xF900 <= cp < 0xFB00 or 0x20000 <= cp < 0x30000:
        return "han"
    return "otherletter"


_SPACELESS = ("han", "kana", "seasian")


def _attrs(ch: str, prev: str) -> List[int]:
    """Indices of the attributes that are on for character `ch`."""
    cp = ord(ch)
    on = [AI["bias"]]
    if ch == prev:
        on.append(AI["same_prev"])
    cat = unicodedata.category(ch)
    if ch == "\n" or ch == "\r":
        on.append(AI["newline"])
    elif ch == " " or cp == 0x00A0 or cp == 0x3000:      # space, NBSP, ideographic space
        on.append(AI["space"])
    elif cat == "Zs" or ch == "\t":
        on.append(AI["otherws"])
    elif cat[0] == "L":
        on += [AI["letter"], AI[_script(cp)]]
        if 0x10D0 <= cp < 0x1100 or 0x2D00 <= cp < 0x2D30:
            on.append(AI["caseless"])   # Georgian Mkhedruli is "Ll" in Unicode but written caseless
        elif cat == "Lu" or cat == "Lt":
            on.append(AI["upper"])
        elif cat == "Ll":
            on.append(AI["lower"])
        else:
            on.append(AI["caseless"])
    elif cat[0] == "N":
        on.append(AI["digit"])
    elif cat[0] == "M":
        on.append(AI["combining"])
    elif ch in _TERMINAL:
        on.append(AI["terminal"])
    elif ch == "…":
        on.append(AI["ellipsis"])
    elif ch in _CLAUSE:
        on.append(AI["clause"])
    elif ch in _QUOTE:
        on.append(AI["quote"])
    elif ch in _OPEN:
        on.append(AI["open"])
    elif ch in _CLOSE:
        on.append(AI["close"])
    elif ch in _DASH:
        on.append(AI["dash"])
    elif ch == "/":
        on.append(AI["slash"])
    elif ch in _MARKUP:
        on.append(AI["markup"])
    elif cat in ("Co", "Cn", "Cs") or cp == 0xFFFD:
        on.append(AI["garbage"])
    elif cat[0] == "S":
        if 0x1F000 <= cp < 0x1FB00 or 0x2600 <= cp < 0x27C0:
            on.append(AI["emoji"])
        else:
            on.append(AI["symbol"])
    elif cat in ("Cc", "Cf"):
        on.append(AI["control"])
    if ch in _MOJI_LEAD:
        on.append(AI["moji_lead"])
    if ch in _MOJI_TAIL:
        on.append(AI["moji_tail"])
    if ch == "." or ch == ",":
        on.append(AI["period_comma"])
    return on


def encode(text: str) -> np.ndarray:
    """(T, A) binary attribute matrix. This is the network's tokenizer."""
    text = text[:MAX_CHARS]
    x = np.zeros((max(len(text), 1), A), dtype=np.float32)
    prev = ""
    for t, ch in enumerate(text):
        x[t, _attrs(ch, prev)] = 1.0
        prev = ch
    return x


# ----------------------------------------------------------------------------- conv channels
CONV = [
    # pass-through rates
    "letter", "upper", "lower", "digit", "space", "newline", "terminal", "clause",
    "quote", "bracket", "dash", "symbol", "markup", "control", "garbage", "combining",
    "emoji", "ellipsis",
    # pattern detectors
    "sent_start",        # ". A"  terminal, space, Upper
    "sent_end_caseless", # "。字" terminal followed by a caseless letter
    "word_start",        # space/newline then letter
    "repeat4",           # four identical non-space chars in a row
    "mojibake",          # Ã¶ â€™ ...
    "url",               # "://"
    "digit_run5",
    "upper_run5",        # SHOUTING
    "punct_run3",        # "!!!" "---" "***"
    "space_punct",       # " ." " ,"
    "double_space",
    "lower_after_term",  # ". a"
    "markup_run2",       # "<<" "{{" "||" "=="
    "emoji_run2",
    "bullet_line",       # newline then "-", "*", digit, symbol
    "line_end_letter",   # line breaks mid-sentence
    "line_end_term",     # line ends with sentence punctuation
    # script rates
    "latin", "cyrillic", "greek", "abjad", "indic", "han", "kana", "hangul", "seasian",
    "caucasian", "otherletter", "caseless",
    "alnum_mix",         # "abc123"
]
CI = {name: i for i, name in enumerate(CONV)}
assert len(CONV) == C

# ----------------------------------------------------------------------------- rnn counters
RNN = ["since_space", "since_term", "since_newline", "words_since_term", "since_letter",
       "position"] + ["_r%d" % i for i in range(6, H)]
RI = {name: i for i, name in enumerate(RNN)}

# ----------------------------------------------------------------------------- pooled features
FEAT = (["m:" + c for c in CONV] + ["max:" + r for r in RNN] + ["mean:" + r for r in RNN]
        + ["loglen", "bias"])
FI = {name: i for i, name in enumerate(FEAT)}
assert len(FEAT) == F


def relu(x):
    return np.maximum(x, 0.0)


class HandNet:
    """The network. All weights are set in `_build_*`; nothing is learned."""

    def __init__(self, short_points: float = 18.0):
        """`short_points`: how many points a very short text can lose (18 for document
        filtering; use ~0 when scoring single sentences)."""
        self.short_points = short_points
        self.E = np.zeros((A, D), dtype=np.float32)
        self.Wc = np.zeros((K, D, C), dtype=np.float32)   # tap k=0 is t-(K-1)//2 ... k=K-1 is t+2
        self.bc = np.zeros(C, dtype=np.float32)
        self.Wh = np.zeros((H, H), dtype=np.float32)
        self.Wx = np.zeros((C, H), dtype=np.float32)
        self.bh = np.zeros(H, dtype=np.float32)
        self.W1 = np.zeros((F, H1), dtype=np.float32)
        self.b1 = np.zeros(H1, dtype=np.float32)
        self.W2 = np.zeros((H1, H2), dtype=np.float32)
        self.b2 = np.zeros(H2, dtype=np.float32)
        self.W3 = np.zeros((H2, H3), dtype=np.float32)
        self.b3 = np.zeros(H3, dtype=np.float32)
        self.W4 = np.zeros((H3, H4), dtype=np.float32)
        self.b4 = np.zeros(H4, dtype=np.float32)
        self.Wo = np.zeros((H4, 1), dtype=np.float32)
        self.bo = np.zeros(1, dtype=np.float32)
        self.units1: List[str] = []   # names of layer-1 units
        self.units2: List[str] = []
        self.units3: List[str] = []
        self.terms: List[Tuple[str, str, float, int, int]] = []  # (kind, name, points, a, b)
        self._build_embedding()
        self._build_conv()
        self._build_rnn()
        self._build_mlp()

    # ------------------------------------------------------------------ layer builders
    def _build_embedding(self):
        # Identity: the attributes are already the representation we want.
        for i in range(A):
            self.E[i, i] = 1.0

    def _tap(self, ch: str, offset: int, attr: str, w: float = 1.0):
        """Conv weight: channel `ch` reads attribute `attr` at position t+offset."""
        self.Wc[offset + (K - 1) // 2, AI[attr], CI[ch]] += w

    def _build_conv(self):
        tap, b = self._tap, self.bc
        for name in CONV[:18]:
            if name == "bracket":
                tap("bracket", 0, "open"); tap("bracket", 0, "close")
            else:
                tap(name, 0, name)
        # "-----", "=====", "*****" separator lines: count the run once, not per character
        for name in ("markup", "symbol", "dash"):
            tap(name, 0, "same_prev", -1.0)
        # ". A": terminal at t-2, space or newline at t-1, Upper at t
        tap("sent_start", -2, "terminal"); tap("sent_start", -1, "space")
        tap("sent_start", -1, "newline"); tap("sent_start", 0, "upper"); b[CI["sent_start"]] = -2
        # "。字" (no space) or "다. 다" (space): terminal(t-1)*2, or terminal(t-2)+space(t-1)
        tap("sent_end_caseless", -1, "terminal", 2.0); tap("sent_end_caseless", -2, "terminal")
        tap("sent_end_caseless", -1, "space"); tap("sent_end_caseless", 0, "caseless")
        b[CI["sent_end_caseless"]] = -2
        tap("word_start", -1, "space"); tap("word_start", -1, "newline"); tap("word_start", 0, "letter")
        b[CI["word_start"]] = -1
        for k in (0, -1, -2):
            tap("repeat4", k, "same_prev")
        for a in ("space", "newline", "dash", "markup", "symbol"):   # separators are not repetition
            tap("repeat4", 0, a, -3)
        b[CI["repeat4"]] = -2
        tap("mojibake", -1, "moji_lead"); tap("mojibake", 0, "moji_tail"); b[CI["mojibake"]] = -1
        tap("url", -2, "clause"); tap("url", -1, "slash"); tap("url", 0, "slash"); b[CI["url"]] = -2
        for k in range(-4, 1):
            tap("digit_run5", k, "digit"); tap("upper_run5", k, "upper")
        b[CI["digit_run5"]] = -4; b[CI["upper_run5"]] = -4
        for k in (0, -1, -2):
            for a in ("terminal", "clause", "quote", "ellipsis"):     # "!!!" ",,," but not "---"
                tap("punct_run3", k, a)
        b[CI["punct_run3"]] = -2
        tap("space_punct", -1, "space"); tap("space_punct", 0, "period_comma"); b[CI["space_punct"]] = -1
        tap("double_space", -1, "space"); tap("double_space", 0, "space"); b[CI["double_space"]] = -1
        tap("lower_after_term", -2, "terminal"); tap("lower_after_term", -1, "space")
        tap("lower_after_term", 0, "lower"); b[CI["lower_after_term"]] = -2
        tap("markup_run2", -1, "markup"); tap("markup_run2", 0, "markup"); b[CI["markup_run2"]] = -1
        tap("emoji_run2", -1, "emoji"); tap("emoji_run2", 0, "emoji"); b[CI["emoji_run2"]] = -1
        tap("bullet_line", -1, "newline")
        for a in ("dash", "symbol", "markup", "digit", "open"):
            tap("bullet_line", 0, a)
        b[CI["bullet_line"]] = -1
        tap("line_end_letter", -1, "letter"); tap("line_end_letter", 0, "newline"); b[CI["line_end_letter"]] = -1
        tap("line_end_term", -1, "terminal"); tap("line_end_term", 0, "newline"); b[CI["line_end_term"]] = -1
        for s in ("latin", "cyrillic", "greek", "abjad", "indic", "han", "kana", "hangul",
                  "seasian", "caucasian", "otherletter", "caseless"):
            tap(s, 0, s)
        tap("alnum_mix", -1, "letter"); tap("alnum_mix", 0, "digit"); b[CI["alnum_mix"]] = -1

    def _build_rnn(self):
        """h_t = relu(Wh h_{t-1} + Wx c_t + bh). Every counter keeps itself (Wh=1), adds its
        increment, and is reset by a large negative input (relu clamps it to 0)."""
        BIG = -1e4
        def counter(name, inc: Dict[str, float], reset: Iterable[str], bias=0.0):
            i = RI[name]
            self.Wh[i, i] = 1.0
            self.bh[i] = bias
            for ch, w in inc.items():
                self.Wx[CI[ch], i] += w
            for ch in reset:
                self.Wx[CI[ch], i] += BIG
        counter("since_space", {"letter": 1.0, "digit": 1.0, "combining": 1.0}, ["space", "newline"])
        counter("since_term", {}, ["terminal", "newline"], bias=1.0)
        counter("since_newline", {}, ["newline"], bias=1.0)
        counter("words_since_term", {"word_start": 1.0}, ["terminal", "newline"])
        counter("since_letter", {}, ["letter"], bias=1.0)
        counter("position", {}, [], bias=1.0)

    # ------------------------------------------------------------------ MLP builders
    def _unit1(self, name: str, lin: Dict[str, float], bias: float) -> int:
        j = len(self.units1)
        assert j < H1, "layer 1 full"
        self.units1.append(name)
        for f, w in lin.items():
            self.W1[FI[f], j] = w
        self.b1[j] = bias
        return j

    def _capped(self, kind: str, name: str, lin: Dict[str, float], bias: float, points: float):
        """A 'range' unit: min(relu(lin.f + bias), 1) * points, built from two ReLUs.
        `lin` and `bias` are scaled so that the unit reads 1.0 at full severity."""
        a = self._unit1(kind + ":" + name, lin, bias)
        b = self._unit1(kind + ":" + name + ":over", lin, bias - 1.0)
        self.terms.append((kind, name, points, a, b))

    def _build_mlp(self):
        P, R = "penalty", "reward"
        spaceless = {"m:han": -100.0, "m:kana": -100.0, "m:seasian": -100.0}   # gate off for CJK/Thai
        caseless = dict(spaceless, **{"m:abjad": -100.0, "m:indic": -100.0,
                                       "m:caucasian": -100.0, "m:hangul": -100.0, "m:otherletter": -100.0})

        # ---- noise: things that should essentially never occur in clean text
        self._capped(P, "garbage_or_control", {"m:garbage": 60, "m:control": 60}, -0.03, 25)
        self._capped(P, "mojibake", {"m:mojibake": 150}, -0.05, 35)
        self._capped(P, "markup", {"m:markup": 12, "m:markup_run2": 30}, -0.15, 20)
        self._capped(P, "symbols", {"m:symbol": 12}, -0.25, 10)
        self._capped(P, "urls", {"m:url": 300}, -0.1, 12)
        self._capped(P, "emoji", {"m:emoji": 40, "m:emoji_run2": 60}, -0.1, 10)
        self._capped(P, "zalgo", dict({"m:combining": 15}, **{"m:abjad": -100, "m:indic": -100,
                                                                 "m:seasian": -100}), -0.3, 15)
        # ---- repetition
        self._capped(P, "char_repeat", {"m:repeat4": 60}, -0.1, 20)
        self._capped(P, "punct_runs", {"m:punct_run3": 60}, -0.15, 12)
        self._capped(P, "digit_runs", {"m:digit_run5": 25}, -0.1, 8)
        # ---- composition
        # letters + combining marks (vowel signs in Indic/Arabic/Thai are part of the word)
        self._capped(P, "few_letters", {"m:letter": -3.0, "m:combining": -3.0}, 1.95, 25)  # <65% -> rises, 32% -> full
        self._capped(P, "comma_heavy", dict({"m:clause": 50}, **spaceless), -1.5, 10)   # >3% commas: keyword lists
        self._capped(P, "digit_heavy", {"m:digit": 6.0}, -0.6, 15)           # >10% digits
        self._capped(P, "shouting", {"m:upper_run5": 8, "m:upper": 4.0}, -0.5, 12)   # ~15% of text in caps runs
        self._capped(P, "alnum_mix", {"m:alnum_mix": 50}, -0.25, 8)
        # ---- formatting
        self._capped(P, "space_before_punct", {"m:space_punct": 150}, -0.1, 8)
        self._capped(P, "double_spaces", {"m:double_space": 30}, -0.2, 5)
        self._capped(P, "missing_capitals", dict({"m:lower_after_term": 60}, **caseless), -0.2, 6)
        self._capped(P, "bullet_lines", {"m:bullet_line": 60}, -0.2, 12)
        self._capped(P, "broken_lines", {"m:line_end_letter": 60, "m:line_end_term": -40}, -0.15, 12)
        self._capped(P, "ellipsis", {"m:ellipsis": 150}, -0.2, 6)
        self._capped(P, "symbol_streak", {"max:since_letter": 1 / 60}, -1.0, 6)      # 60+ chars with no letter
        # ---- structure (gated for scripts where the counter is meaningless)
        self._capped(P, "long_words", dict({"mean:since_space": 1 / 6}, **spaceless), -1.5, 15)  # mean run >9
        self._capped(P, "giant_token", dict({"max:since_space": 1 / 40}, **spaceless), -1.0, 12)  # 40+ chars
        self._capped(P, "no_sentence_end", dict({"max:since_term": 1 / 500, "m:seasian": -100}), -1.0, 15)
        self._capped(P, "few_sentence_ends", {"m:terminal": -300, "m:seasian": -100}, 1.5, 12)   # <0.5%
        self._capped(P, "fragments", dict({"mean:words_since_term": -0.5}, **spaceless), 1.5, 12)  # <3 words/sent
        self._capped(P, "very_short_sentences", {"m:sent_start": 40, "m:sent_end_caseless": 40}, -1.6, 10)  # >4%
        # <~250 chars, full at ~30; a CJK/Korean/Thai character carries ~2-3 Latin characters' worth
        self._capped(P, "too_short", {"loglen": -5.0, "m:han": -0.6, "m:kana": -0.6, "m:hangul": -0.4,
                                      "m:seasian": -0.4}, 3.6, self.short_points)
        # ---- script mixing: counts scripts present above a threshold (layer-2 logic below)
        self.script_units = []
        # Japanese mixes han and kana by design, so they count as one script ("cjk").
        for s, lin, thr in (("latin", {"m:latin": 100.0}, 0.25), ("cyrillic", {"m:cyrillic": 100.0}, 0.05),
                            ("greek", {"m:greek": 100.0}, 0.05), ("abjad", {"m:abjad": 100.0}, 0.05),
                            ("indic", {"m:indic": 100.0}, 0.05),
                            ("cjk", {"m:han": 100.0, "m:kana": 100.0}, 0.05),
                            ("hangul", {"m:hangul": 100.0}, 0.05), ("seasian", {"m:seasian": 100.0}, 0.05),
                            ("caucasian", {"m:caucasian": 100.0}, 0.05),
                            ("otherletter", {"m:otherletter": 100.0}, 0.10)):
            a = self._unit1("script:" + s, lin, -100.0 * thr)
            b = self._unit1("script:" + s + ":over", lin, -100.0 * thr - 1.0)
            self.script_units.append((a, b))
        # ---- rewards for the shape of good running text
        self._capped(R, "sentences", {"m:sent_start": 300, "m:sent_end_caseless": 300}, -0.6, 8)   # >=~0.5%
        self._capped(R, "clauses", {"m:clause": 200}, -0.4, 4)                                    # >=0.7%
        self._capped(R, "letters", {"m:letter": 10, "m:combining": 10}, -6.6, 6)                   # >=76%
        self._capped(R, "length", {"loglen": 4.0, "m:han": 0.45, "m:kana": 0.45, "m:hangul": 0.3,
                                   "m:seasian": 0.3}, -2.6, 6)                                     # >=~600 chars
        self._capped(R, "word_shape", dict({"mean:since_space": 0.5}, **spaceless), -1.0, 4)        # mean run >=4
        # CJK/Thai cannot earn word_shape; give it back through the sentence-end channel
        self._capped(R, "spaceless_bonus", {"m:han": 20, "m:kana": 20, "m:seasian": 20}, -0.5, 4)

        # ---- length gates. Rates measured on a few dozen characters are noise, so rate-based
        # groups are limited to min(gate, penalty). Character composition (letters, digits) is
        # reliable from ~30 chars; sentence structure only from ~100.
        gates = {
            "composition": (self._unit1("gate:len30", {"loglen": 6.8}, -2.92),      # 0 at 30, 1 at 100
                            self._unit1("gate:len30:over", {"loglen": 6.8}, -3.92)),
            "structure": (self._unit1("gate:len100", {"loglen": 8.0}, -4.64),      # 0 at 100, 1 at 300
                          self._unit1("gate:len100:over", {"loglen": 8.0}, -5.64)),
        }

        # ---- layer 2: named sub-scores (points) + the gate scaled to its budget
        groups = {
            "noise": (["garbage_or_control", "mojibake", "markup", "symbols", "urls", "emoji", "zalgo"], False),
            "repetition": (["char_repeat", "punct_runs", "digit_runs"], False),
            "composition": (["few_letters", "digit_heavy", "shouting", "alnum_mix", "comma_heavy"], True),
            "formatting": (["space_before_punct", "double_spaces", "missing_capitals", "bullet_lines",
                            "broken_lines", "ellipsis", "symbol_streak"], False),
            "tokens": (["long_words", "giant_token", "no_sentence_end"], False),
            "structure": (["few_sentence_ends", "fragments", "very_short_sentences"], True),
            "length": (["too_short"], False),
        }
        self.groups = groups
        by_name = {(k, n): (p, a, b) for k, n, p, a, b in self.terms}
        gate_unit = {}
        for g, (names, gated) in groups.items():
            j = self._unit2("penalty:" + g)
            budget = 0.0
            for n in names:
                p, a, b = by_name[(P, n)]
                self.W2[a, j] += p; self.W2[b, j] -= p
                budget += p
            if gated:
                ga, gb = gates[g]
                jg = gate_unit[g] = self._unit2("gate:" + g)     # the gate scaled to the group's budget
                self.W2[ga, jg] = budget; self.W2[gb, jg] = -budget
        j = self._unit2("reward")
        for (k, n), (p, a, b) in by_name.items():
            if k == R:
                self.W2[a, j] += p; self.W2[b, j] -= p
        # extra scripts = relu(#scripts - 1); capped at 2 extra
        j1 = self._unit2("penalty:mixed_scripts"); j2 = self._unit2("penalty:mixed_scripts:over")
        for a, b in self.script_units:
            self.W2[a, j1] += 1; self.W2[b, j1] -= 1
            self.W2[a, j2] += 1; self.W2[b, j2] -= 1
        self.b2[j1] = -1.0; self.b2[j2] = -3.0

        # ---- layer 3: pass every sub-score through, and for gated groups add
        # over = relu(penalty - gate), so that penalty - over = min(penalty, gate)
        for j, name in enumerate(self.units2):
            k = self._unit3(name)
            self.W3[j, k] = 1.0
        for g, (names, gated) in groups.items():
            if gated:
                k = self._unit3("over:" + g)
                self.W3[self.units2.index("penalty:" + g), k] = 1.0
                self.W3[gate_unit[g], k] = -1.0

        # ---- layer 4: z = 72 + reward - penalties; out = clip(z, 0, 100) = relu(z) - relu(z-100)
        self.BASE = 72.0
        for k, name in enumerate(self.units3):
            if name.startswith("gate:"):
                continue
            w = -1.0
            if name == "reward" or name.startswith("over:"):
                w = 1.0
            if name == "penalty:mixed_scripts":
                w = -12.0
            if name == "penalty:mixed_scripts:over":
                w = +12.0
            self.W4[k, 0] = w; self.W4[k, 1] = w
        self.b4[0] = self.BASE; self.b4[1] = self.BASE - 100.0
        self.Wo[0, 0] = 1.0; self.Wo[1, 0] = -1.0

    def _unit2(self, name: str) -> int:
        j = len(self.units2)
        assert j < H2, "layer 2 full"
        self.units2.append(name)
        return j

    def _unit3(self, name: str) -> int:
        k = len(self.units3)
        assert k < H3, "layer 3 full"
        self.units3.append(name)
        return k

    # ------------------------------------------------------------------ forward
    def features(self, text: str) -> np.ndarray:
        x = encode(text)                                   # (T, A)
        e = x @ self.E                                     # (T, D)
        T = e.shape[0]
        pad = np.zeros(((K - 1) // 2, D), dtype=np.float32)
        ep = np.concatenate([pad, e, pad])                 # (T+K-1, D)
        c = np.zeros((T, C), dtype=np.float32) + self.bc
        for k in range(K):
            c += ep[k:k + T] @ self.Wc[k]
        c = relu(c)                                        # (T, C)
        # ReLU-RNN with Wh = I: h_t = relu(h_{t-1} + u_t) has the closed form
        # h_t = S_t - min(0, min_{k<=t} S_k) with S = cumsum(u)  (Lindley recursion).
        assert np.array_equal(self.Wh, np.diag(np.diag(self.Wh))) and set(np.diag(self.Wh)) <= {0.0, 1.0}
        u = c @ self.Wx + self.bh                          # (T, H)
        S = np.cumsum(u, axis=0)
        h = S - np.minimum(np.minimum.accumulate(S, axis=0), 0.0)
        h *= np.diag(self.Wh)[None, :]
        f = np.concatenate([c.mean(0), h.max(0), h.mean(0),
                            [math.log1p(T) / 8.0, 1.0]]).astype(np.float32)
        return f

    def forward(self, text: str) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        f = self.features(text)
        h1 = relu(f @ self.W1 + self.b1)
        h2 = relu(h1 @ self.W2 + self.b2)
        h3 = relu(h2 @ self.W3 + self.b3)
        h4 = relu(h3 @ self.W4 + self.b4)
        y = float((h4 @ self.Wo + self.bo)[0])
        return y, f, h1, h2

    def score(self, text: str) -> float:
        return self.forward(text)[0]

    def score_many(self, texts: Sequence[str]) -> List[float]:
        return [self.score(t) for t in texts]

    def explain(self, text: str) -> Dict[str, float]:
        """Points each named term contributed (negative = penalty)."""
        y, f, h1, h2 = self.forward(text)
        out = {"score": round(y, 1), "base": self.BASE}
        scale = {}   # gated groups: every term is shrunk by the same factor min(1, gate/penalty)
        for g, (names, gated) in self.groups.items():
            tot = h2[self.units2.index("penalty:" + g)]
            gate = h2[self.units2.index("gate:" + g)] if gated else 0.0
            for n in names:
                scale[n] = min(1.0, gate / tot) if (gated and tot > 1e-6) else 1.0
        for k, n, p, a, b in self.terms:
            v = (h1[a] - h1[b]) * p * scale.get(n, 1.0)
            if v > 1e-6:
                out[("+" if k == "reward" else "-") + n] = round(float(v), 1)
        j1 = self.units2.index("penalty:mixed_scripts")
        j2 = self.units2.index("penalty:mixed_scripts:over")
        v = 12.0 * (h2[j1] - h2[j2])
        if v > 1e-6:
            out["-mixed_scripts"] = round(float(v), 1)
        return out

    # ------------------------------------------------------------------ bookkeeping
    @property
    def tensors(self) -> Dict[str, np.ndarray]:
        return {"E": self.E, "Wc": self.Wc, "bc": self.bc, "Wh": self.Wh, "Wx": self.Wx, "bh": self.bh,
                "W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2, "W3": self.W3,
                "b3": self.b3, "W4": self.W4, "b4": self.b4, "Wo": self.Wo, "bo": self.bo}

    @property
    def n_params(self) -> int:
        return sum(int(t.size) for t in self.tensors.values())

    @property
    def n_nonzero(self) -> int:
        return sum(int(np.count_nonzero(t)) for t in self.tensors.values())

    def save(self, path: str):
        np.savez(path, **self.tensors)

    def to_torch(self):
        """The same network as an nn.Module (for GPU batching or fine-tuning)."""
        import torch
        import torch.nn as nn

        net = self

        class HandNetTorch(nn.Module):
            def __init__(self):
                super().__init__()
                self.E = nn.Parameter(torch.tensor(net.E))
                self.conv = nn.Conv1d(D, C, K, padding=(K - 1) // 2)
                self.conv.weight.data = torch.tensor(net.Wc).permute(2, 1, 0).contiguous()
                self.conv.bias.data = torch.tensor(net.bc)
                self.Wh = nn.Parameter(torch.tensor(net.Wh)); self.Wx = nn.Parameter(torch.tensor(net.Wx))
                self.bh = nn.Parameter(torch.tensor(net.bh))
                self.l1 = nn.Linear(F, H1); self.l2 = nn.Linear(H1, H2); self.l3 = nn.Linear(H2, H3)
                self.l4 = nn.Linear(H3, H4); self.out = nn.Linear(H4, 1)
                for lin, W, b in ((self.l1, net.W1, net.b1), (self.l2, net.W2, net.b2),
                                  (self.l3, net.W3, net.b3), (self.l4, net.W4, net.b4),
                                  (self.out, net.Wo, net.bo)):
                    lin.weight.data = torch.tensor(W.T.copy()); lin.bias.data = torch.tensor(b)

            def forward(self, x):                      # x: (B, T, A) attribute tensor
                e = x @ self.E
                c = torch.relu(self.conv(e.transpose(1, 2))).transpose(1, 2)   # (B, T, C)
                h = torch.zeros(x.shape[0], H, device=x.device)
                hs = []
                for t in range(c.shape[1]):
                    h = torch.relu(h @ self.Wh + c[:, t] @ self.Wx + self.bh)
                    hs.append(h)
                hs = torch.stack(hs, 1)
                T = torch.tensor(float(c.shape[1]))
                f = torch.cat([c.mean(1), hs.max(1).values, hs.mean(1),
                               (torch.log1p(T) / 8.0).expand(x.shape[0], 1), torch.ones(x.shape[0], 1)], 1)
                z = torch.relu(self.l4(torch.relu(self.l3(torch.relu(self.l2(torch.relu(self.l1(f))))))))
                return self.out(z).squeeze(-1)

        return HandNetTorch()


def main(argv=None):
    import argparse, json, sys
    ap = argparse.ArgumentParser(description="hand-written text-quality score (0-100)")
    ap.add_argument("--input", help="JSONL of {text}; default: read texts from stdin, one per line")
    ap.add_argument("--output", help="JSONL out with handnet score added")
    ap.add_argument("--explain", action="store_true")
    args = ap.parse_args(argv)
    net = HandNet()
    print("params=%d nonzero=%d" % (net.n_params, net.n_nonzero), file=sys.stderr)
    if args.input:
        with open(args.input, encoding="utf-8") as fh:
            docs = [json.loads(l) for l in fh]
        out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
        for d in docs:
            d["handnet"] = net.explain(d["text"]) if args.explain else round(net.score(d["text"]), 1)
            out.write(json.dumps(d, ensure_ascii=False) + "\n")
    else:
        for line in sys.stdin:
            print(json.dumps(net.explain(line.rstrip("\n")), ensure_ascii=False))


if __name__ == "__main__":
    main()

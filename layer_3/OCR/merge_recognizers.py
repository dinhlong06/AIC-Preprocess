"""
merge_recognizers.py -- CPU-only merge of the two independent recognizer
outputs from ocr/paddle_engine.py (output_vietocr.json + output_paddle_origin.json)
into one final per-box text choice.

Kept OUT of paddle_engine.py::run() on purpose: choosing which recognizer's
answer to keep is a decision that needs tuning/re-measuring over time (see
cases below), and this repo's dataset is expensive to re-OCR on GPU (hours,
on a shared cluster). Both recognizers already ran on every box and both
outputs are saved -- so re-deciding "which one wins" is a few-seconds CPU
pass over the saved JSON, not a GPU rerun.

Per box, matched by (frame_id, box) between the two files:
- present in only one recognizer's output (the other returned empty text for
  that box) -- take that one, nothing to compare.
- present in both -- VietOCR wins by default (it has a real Vietnamese
  vocabulary and doesn't drop 70 accented characters like Paddle's own
  recognizer does, see paddle_engine.py docstring), EXCEPT for the measured
  override cases below.

Digit/time/percent *words* (clock overlays, page counters, "3%" style
figures, a year inside a longer ticker sentence) get a word-level override
even when VietOCR was confident overall: VietOCR was measured to hallucinate
or drop digits on these tokens ("9 TRIEU" -> "99 TRIEU", a frame-edge
-truncated "1" -> "la", verified against the source frames, not just
confidence scores). Paddle's recognizer reads clock/digit fonts more
reliably, so for each word position where Paddle's own word is a pure
digit/punctuation token and differs from VietOCR's, Paddle's word wins.

Generalized from digit-only to any ascii-fold-invariant VietOCR word (i.e.
`asciifold(word) == word` -- digits, English words, codes, anything with no
Vietnamese accent mark to restore): VietOCR's edge over Paddle is its
Vietnamese vocabulary and accent restoration, neither of which is doing any
work on a word that was already plain ASCII, so Paddle's reading of that
word wins instead. Unlike the digit case above, this checks VietOCR's own
word (not Paddle's) -- not yet re-measured against source frames the way the
digit-only version was, so re-verify if it starts misfiring.

An earlier version also swapped short (<=2 letter) all-caps words mid
-sentence the same way, on the theory that codes/abbreviations ("MI", "CV")
give VietOCR's decoder too little context. Measured live in production: a
Paddle word being <=2 letters mid-sentence is usually its own vowel-drop bug
("Vu" -> "V", "Bo" -> "B", "My" -> "M") rather than a genuine short code --
that signal only holds when the *whole* line is short, not one word inside a
longer one, so it was removed from the per-word merge and kept only in the
whole-line override below.

The digit override only looks at Paddle's own word at each position (never
VietOCR's) -- the question being asked is "is Paddle's answer here a token
type it's known to read reliably", not anything about what VietOCR said.
Surrounding words are left untouched, and it's skipped entirely when the two
sides don't split into the same number of words -- misaligned counts (a
dropped or merged word on either side) make position-by-position swapping
unsafe, so the line is left as VietOCR produced it.

A second override takes a whole line, not just one word: a <=2-word,
fully-uppercase line (word counts matching between the two recognizers)
takes Paddle's answer outright. Measured against source frames (not
confidence scores) on 28 real disagreements in this bucket: Paddle correct
17/28 (61%), VietOCR correct 6/28 (21%), the rest a tie -- mostly English
brand/logo/signage text ("KIA", "BIA SAIGON", "GROW AGAIN", "60K", "OCTOBER
1ST-3RD") that VietOCR (trained on Vietnamese printed prose) misreads more
often than Paddle does. An earlier "any <=2-word phrase" idea (no all-caps
requirement) was rejected for false-positiving heavily on ordinary
Vietnamese headlines ("Chu truong" -> "Chu truon", "Nguyen" -> "Nguyn" --
Paddle's letter-drop bug from above, just on shorter lines). All-caps alone
doesn't fully protect against this either: "TIN CHINH"/"TIEP THEO" are
themselves all-caps, but they're saved by the word-count-match requirement
above -- Paddle collapses them to one word ("TINCHINH"/"TIEPTHEO", no space
detected) while VietOCR keeps two, so the mismatched count already skips
them without needing a separate check.

A third override, added 2026-08-27 investigating a title banner
("Few / A Few / Little / A Little") that VietOCR read as one glued blob
("ew/AFew/Little/Alittle", 0.7536 conf) while Paddle read correctly
("Few / A Few / Little / A Little", 0.9983 conf): VietOCR collapsed a
multi-word box into one long space-less blob while Paddle split the same box
into multiple words. The two word-count-matching overrides above can't catch
this -- the merge itself is what breaks the word count they require.
Deliberately NOT a confidence comparison: Paddle scores systematically
higher than VietOCR, so raw score margin isn't a trustworthy signal here
(measured on batch1: even at Paddle confidence 0.81-0.83, the merged-word
signal alone was still right in every one of 1,363 sampled cases). A wider
version of this rule ("whichever side splits into more words wins", applied
whenever VietOCR isn't fully merged) was measured and REJECTED: on 23,936
cases where Paddle had more words but VietOCR wasn't a fully-merged blob,
Paddle was usually WRONG -- it drops vowels/consonants on long Vietnamese
sentences (e.g. VietOCR "mot thanh nien duoi nuoc tu vong" correct vs Paddle
"i, mt thanh nien dui nuc t vong" garbled), so "more words" there is a
symptom of Paddle's letter-drop bug, not of reading more correctly. Keep the
override narrow: only when VietOCR is a single space-less blob.
"""

from __future__ import annotations

import json
import sys
import unicodedata


def _asciifold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _merge_ascii_words(text: str, p_text: str) -> str:
    """Word-level override, generalized from the old digit-only check: a
    VietOCR word that's already its own ascii-fold (digits, English words,
    codes -- nothing for accent-restoration to do) needs none of VietOCR's
    Vietnamese-vocabulary strength, so Paddle's reading wins instead.
    Mismatched word counts (VietOCR gluing multiple Paddle-separated words
    into one garbled token, e.g. "HTV 7HD" -> "Millip") can't be swapped
    word-by-word, so fall back to Paddle's whole line when VietOCR's is
    ascii-fold-invariant -- same signal, just line-level instead of word."""
    words, p_words = text.split(), p_text.split()
    if len(words) != len(p_words):
        return p_text if text and _asciifold(text) == text else text
    return " ".join(
        p if _asciifold(w) == w and p != w else w
        for w, p in zip(words, p_words)
    )


def _is_allcaps(s: str) -> bool:
    letters = [c for c in s if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def _prefer_paddle_short_line(text: str, p_text: str) -> str | None:
    if text == p_text:
        return None
    words, p_words = text.split(), p_text.split()
    if not (1 <= len(words) <= 2) or len(words) != len(p_words):
        return None
    if not _is_allcaps(text):
        return None
    return p_text


def looks_merged(text: str, p_text: str) -> bool:
    return len(text) >= 12 and " " not in text and " " in p_text


def choose(text: str, p_text: str) -> str:
    """Given both recognizers' text for the SAME box, return the winner."""
    if looks_merged(text, p_text):
        return p_text
    return _prefer_paddle_short_line(text, p_text) or _merge_ascii_words(text, p_text)


def merge_frame(vietocr_texts: list[dict], paddle_texts: list[dict]) -> list[dict]:
    """Merge one frame's two independent box lists into the final choice per box."""
    paddle_by_box = {tuple(t["box"]): t for t in paddle_texts}
    seen_boxes = set()
    merged = []

    for t in vietocr_texts:
        box = tuple(t["box"])
        seen_boxes.add(box)
        pt = paddle_by_box.get(box)
        if pt is None:
            merged.append(t)
        else:
            final_text = choose(t["text"], pt["text"])
            confidence = pt["confidence"] if final_text == pt["text"] else t["confidence"]
            merged.append({"text": final_text, "confidence": confidence, "box": t["box"]})

    for t in paddle_texts:
        if tuple(t["box"]) not in seen_boxes:
            merged.append(t)

    return merged


def main() -> None:
    vietocr_path, paddle_path, output_path = sys.argv[1:4]

    vietocr = {r["frame_id"]: r for r in json.loads(open(vietocr_path, encoding="utf-8").read())}
    paddle = {r["frame_id"]: r for r in json.loads(open(paddle_path, encoding="utf-8").read())}

    changed = 0
    records = []
    for frame_id, rv in vietocr.items():
        rp = paddle.get(frame_id)
        paddle_texts = rp["texts"] if rp else []
        merged_texts = merge_frame(rv["texts"], paddle_texts)
        before = [t["text"] for t in rv["texts"]]
        after = [t["text"] for t in merged_texts]
        if before != after:
            changed += 1
        records.append({"frame_id": frame_id, "texts": merged_texts})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[->] {len(records)} frame, {changed} frame co it nhat 1 box bi doi -> {output_path}")


if __name__ == "__main__":
    main()

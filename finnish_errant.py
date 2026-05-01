"""
Finnish ERRANT: Linguistically-aware GEC evaluation for Finnish.

Mirrors the ERRANT pipeline (English) but adapted for Finnish morphology:

    src, pred, ref
         ↓
    1. Parse    — stanza: tokens + POS + lemma + morphological features
         ↓
    2. Align    — Levenshtein with linguistically-informed substitution costs
         ↓
    3. Classify — Finnish-specific error taxonomy (NOUN:CASE, VERB:FORM, ...)
         ↓
    4. Score    — tp/fp/fn via set operations → F0.5 (+ per-type breakdown)

Setup:
    pip install stanza
    python -c "import stanza; stanza.download('fi')"

Usage:
    from revita_library.gec.evaluation.finnish_errant import FinnishERRANT

    errant = FinnishERRANT()

    # Score a full dataset
    results = errant.score(sources, predictions, references)
    print(results['f05'])           # overall F0.5
    print(results['by_type'])       # F0.5 per error type

    # Inspect individual edits
    edits = errant.annotate("Minä asua Helsingissä", "Minä asun Helsingissä")
    for e in edits:
        print(e)  # Edit(SUB 'asua'→'asun' [VERB:FORM] src=(1, 2))
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):  # fallback: no-op
        return it

try:
    import Levenshtein as _Levenshtein  # type: ignore
except ImportError:
    _Levenshtein = None

_HARMONY_PAIRS = frozenset({
    ("a", "ä"), ("ä", "a"),
    ("o", "ö"), ("ö", "o"),
    ("u", "y"), ("y", "u"),
})


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Token:
    text: str
    lemma: str
    upos: str           # Universal POS (NOUN, VERB, ADJ, ...)
    feats: frozenset    # Frozenset of "Feature=Value" strings
    idx: int            # Position in sentence

    def feat(self, key: str) -> Optional[str]:
        """Get a specific morphological feature value, e.g. feat('Case') → 'Gen'."""
        for f in self.feats:
            k, _, v = f.partition("=")
            if k == key:
                return v
        return None


@dataclass(frozen=True)
class Edit:
    """A single edit operation between source and hypothesis."""
    op: str                  # SUB, INS, DEL
    src_tokens: tuple        # Source tokens (empty for INS)
    hyp_tokens: tuple        # Hypothesis tokens (empty for DEL)
    src_interval: tuple      # (start, end) span in source token list
    hyp_interval: tuple      # (start, end) span in hypothesis token list
    error_types: tuple       # Tuple of error types, e.g. ("VOWEL:HARMONY", "CASE")

    @property
    def error_type(self) -> str:
        """Primary (first) error type — for backwards compatibility."""
        return self.error_types[0] if self.error_types else ErrorType.OTHER

    def __repr__(self):
        src = " ".join(t.text for t in self.src_tokens) or "∅"
        hyp = " ".join(t.text for t in self.hyp_tokens) or "∅"
        types = "+".join(self.error_types)
        return f"Edit({self.op} '{src}'→'{hyp}' [{types}] src={self.src_interval})"


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------

class ErrorType:
    # --- Verb errors ---
    VERB_FORM   = "VERB:FORM"       # same lemma, catch-all for unclassified form change
    VERB_TENSE  = "VERB:TENSE"      # menee → meni (Tense feature differs)
    VERB_VOICE  = "VERB:VOICE"      # active/passive (Voice feature differs)
    VERB_MOOD   = "VERB:MOOD"       # tulee → tulisi (Mood feature differs: Ind/Cnd/Imp/Pot)
    VERB_PART   = "VERB:PART"       # juokseva → juossut (VerbForm changes to/from Part)

    # --- Unified morphological errors (cross-POS, same lemma) ---
    PERSON      = "PERSON"          # Person feature differs (verbs, pronouns)
    DEGREE      = "DEGREE"          # Degree feature differs (adj + comparative nouns)
    CASE        = "CASE"            # Case feature differs (nouns, adj, pronouns, verbs)
    NUMBER      = "NUMBER"          # Number feature differs
    NOUN_POSS   = "NOUN:POSS"       # possessive suffix added/changed
    CLITIC      = "CLITIC"          # clitic suffix differs (-kin, -kaan, -han, etc.)

    # Backwards-compat aliases
    VERB_PERS   = PERSON
    NOUN_CASE   = CASE
    ADJ_CASE    = CASE
    NOUN_NUM    = NUMBER
    ADJ_FORM    = DEGREE

    # --- Cross-token / structural errors ---
    AGR         = "AGR"             # agreement mismatch across tokens (detected externally)
    WO          = "WO"              # word order error (detected externally)

    # --- Lexical errors ---
    WORD_CHOICE = "WORD:CHOICE"     # same POS, different lemma
    CONJ        = "CONJ"            # wrong conjunction/connective (ja → mutta)
    PART        = "PART"            # wrong/missing particle (-kin, -kaan, -han, etc.)

    # --- Finnish-specific surface errors ---
    VOWEL_HARM  = "VOWEL:HARMONY"   # talossä → talossa (back/front vowel mismatch: a↔ä, o↔ö, u↔y)
    COMPOUND    = "COMPOUND"        # autotalli → auto talli (compound split or join)

    # --- General surface errors ---
    SPELL       = "SPELL"           # spelling error (no POS/lemma match, edit dist ≤ 2)
    PUNCT       = "PUNCT"           # punctuation error
    INSERT      = "INSERT"          # missing word
    DELETE      = "DELETE"          # extra word
    OTHER       = "OTHER"           # catch-all


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FinnishERRANT:
    """
    Finnish ERRANT: linguistically-aware GEC evaluation using stanza.

    Args:
        use_gpu: Use GPU for stanza parsing (faster on large datasets)
        lazy_load: Don't load stanza until first call (useful for import)
    """

    def __init__(self, use_gpu: bool = False, lazy_load: bool = True):
        self._nlp = None
        self._use_gpu = use_gpu
        self._parse_cache: dict[str, list[Token]] = {}
        if not lazy_load:
            self._load_stanza()

    def _build_pipeline(self):
        import stanza
        return stanza.Pipeline(
            lang="fi",
            processors="tokenize,pos,lemma",
            use_gpu=self._use_gpu,
            tokenize_batch_size=128,
            pos_batch_size=3000,
            lemma_batch_size=200,
            verbose=False,
        )

    def _load_stanza(self):
        try:
            import stanza
        except ImportError:
            print("Error: stanza not installed.\nRun: pip install stanza")
            sys.exit(1)

        try:
            self._nlp = self._build_pipeline()
        except Exception:
            print("Finnish stanza model not found. Downloading...")
            stanza.download("fi", verbose=False)
            self._nlp = self._build_pipeline()

        try:
            import torch
            cuda_ok = torch.cuda.is_available()
        except Exception:
            cuda_ok = False
        print(f"✓ stanza Finnish pipeline loaded (use_gpu={self._use_gpu}, cuda_available={cuda_ok})")

    # ------------------------------------------------------------------
    # 1. Parse
    # ------------------------------------------------------------------

    @staticmethod
    def _doc_to_tokens(doc) -> list[Token]:
        tokens: list[Token] = []
        idx = 0
        for sentence in doc.sentences:
            for word in sentence.words:
                feats = frozenset(word.feats.split("|")) if word.feats else frozenset()
                tokens.append(Token(
                    text=word.text,
                    lemma=word.lemma or word.text.lower(),
                    upos=word.upos or "X",
                    feats=feats,
                    idx=idx,
                ))
                idx += 1
        return tokens

    def _parse_many(self, texts, batch_size: int = 0) -> None:
        """
        Batch-parse a collection of texts and populate the parse cache. Texts
        already cached are skipped. Empty strings are cached as empty token
        lists without invoking Stanza.

        Args:
            batch_size: If > 0, chunk Stanza calls into groups of this size
                        (useful to limit peak memory). If 0, send everything
                        in one call (fastest, but uses more memory).
        """
        unique: list[str] = []
        seen: set[str] = set()
        for t in texts:
            if t in self._parse_cache or t in seen:
                continue
            if not t:
                self._parse_cache[t] = []
                continue
            seen.add(t)
            unique.append(t)

        if not unique:
            return

        if self._nlp is None:
            self._load_stanza()

        from stanza import Document
        chunk = batch_size if batch_size and batch_size > 0 else len(unique)
        for start in range(0, len(unique), chunk):
            batch_texts = unique[start:start + chunk]
            in_docs = [Document([], text=t) for t in batch_texts]
            out_docs = self._nlp(in_docs)
            for text, doc in zip(batch_texts, out_docs):
                self._parse_cache[text] = self._doc_to_tokens(doc)

    def parse(self, text: str) -> list[Token]:
        """Parse Finnish text into a list of Tokens with POS, lemma, and feats."""
        cached = self._parse_cache.get(text)
        if cached is not None:
            return cached
        self._parse_many([text])
        return self._parse_cache.get(text, [])

    # ------------------------------------------------------------------
    # 2. Align
    # ------------------------------------------------------------------

    def _substitution_cost(self, t1: Token, t2: Token) -> float:
        """
        Linguistically-informed substitution cost between two tokens.

          0.0 — identical text
          0.1 — same lemma (morphological variant)
          0.4 — same POS (different word, same category)
          1.0 — completely different
        """
        if t1.text.lower() == t2.text.lower():
            return 0.0
        if t1.lemma == t2.lemma:
            return 0.1
        if t1.upos == t2.upos:
            return 0.4
        return 1.0

    def align(self, src: list[Token], hyp: list[Token]) -> list[Edit]:
        """
        Align src and hyp using Levenshtein DP with linguistically-informed costs.
        Returns only non-KEEP edit operations.
        """
        n, m = len(src), len(hyp)

        dp = [[0.0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            dp[i][0] = i
        for j in range(1, m + 1):
            dp[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                sub_cost = self._substitution_cost(src[i - 1], hyp[j - 1])
                dp[i][j] = min(
                    dp[i - 1][j - 1] + sub_cost,
                    dp[i - 1][j] + 1.0,
                    dp[i][j - 1] + 1.0,
                )

        edits = []
        i, j = n, m
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                sub_cost = self._substitution_cost(src[i - 1], hyp[j - 1])
                if dp[i][j] == dp[i - 1][j - 1] + sub_cost:
                    if sub_cost > 0:
                        edits.append(Edit(
                            op="SUB",
                            src_tokens=(src[i - 1],),
                            hyp_tokens=(hyp[j - 1],),
                            src_interval=(i - 1, i),
                            hyp_interval=(j - 1, j),
                            error_types=(),
                        ))
                    i -= 1
                    j -= 1
                    continue

            if i > 0 and dp[i][j] == dp[i - 1][j] + 1.0:
                edits.append(Edit(
                    op="DEL",
                    src_tokens=(src[i - 1],),
                    hyp_tokens=(),
                    src_interval=(i - 1, i),
                    hyp_interval=(j, j),
                    error_types=(),
                ))
                i -= 1
            else:
                edits.append(Edit(
                    op="INS",
                    src_tokens=(),
                    hyp_tokens=(hyp[j - 1],),
                    src_interval=(i, i),
                    hyp_interval=(j - 1, j),
                    error_types=(),
                ))
                j -= 1

        return list(reversed(edits))

    # ------------------------------------------------------------------
    # 3. Classify
    # ------------------------------------------------------------------

    def classify(self, edit: Edit) -> Edit:
        """Assign Finnish-specific error types to an edit (may return multiple for SUB edits)."""
        return Edit(
            op=edit.op,
            src_tokens=edit.src_tokens,
            hyp_tokens=edit.hyp_tokens,
            src_interval=edit.src_interval,
            hyp_interval=edit.hyp_interval,
            error_types=tuple(self._classify(edit)),
        )

    def _classify(self, edit: Edit) -> list[str]:
        # INS / DEL / PUNCT always go alone — no co-occurring types possible
        if edit.op == "INS":
            return [ErrorType.INSERT]
        if edit.op == "DEL":
            return [ErrorType.DELETE]

        src_tok = edit.src_tokens[0]
        hyp_tok = edit.hyp_tokens[0]

        if src_tok.upos == "PUNCT" or hyp_tok.upos == "PUNCT":
            return [ErrorType.PUNCT]

        # SUB edits: collect all matching types (a single word change can be
        # e.g. both VOWEL:HARMONY and CASE at the same time)
        types = []

        # Surface check: vowel harmony is independent of morphological features
        if _is_vowel_harmony_error(src_tok.text, hyp_tok.text):
            types.append(ErrorType.VOWEL_HARM)

        if src_tok.lemma == hyp_tok.lemma:
            # Morphological feature checks — collect all that differ
            if _fd(src_tok, hyp_tok, "VerbForm"):
                types.append(ErrorType.VERB_PART)
            if _fd(src_tok, hyp_tok, "PartForm"):
                types.append(ErrorType.VERB_TENSE)
            if _fd(src_tok, hyp_tok, "Voice"):
                types.append(ErrorType.VERB_VOICE)
            if _fd(src_tok, hyp_tok, "Mood"):
                types.append(ErrorType.VERB_MOOD)
            if _fd(src_tok, hyp_tok, "Tense"):
                types.append(ErrorType.VERB_TENSE)
            if _fd(src_tok, hyp_tok, "Person"):
                types.append(ErrorType.PERSON)
            if _fd(src_tok, hyp_tok, "Degree"):
                types.append(ErrorType.DEGREE)
            if _fd(src_tok, hyp_tok, "Case"):
                types.append(ErrorType.CASE)
            if _fd(src_tok, hyp_tok, "Number"):
                types.append(ErrorType.NUMBER)
            if _fd(src_tok, hyp_tok, "PossNumber") or _fd(src_tok, hyp_tok, "PersPoss"):
                types.append(ErrorType.NOUN_POSS)
            clitic_s, clitic_h = src_tok.feat("Clitic"), hyp_tok.feat("Clitic")
            if clitic_s != clitic_h and (clitic_s is not None or clitic_h is not None):
                types.append(ErrorType.CLITIC)
            if not types:
                types.append(ErrorType.VERB_FORM)

        elif src_tok.upos == hyp_tok.upos:
            if _is_compound_variant(src_tok.text, hyp_tok.text):
                types.append(ErrorType.COMPOUND)
            elif src_tok.upos in ("CCONJ", "SCONJ"):
                types.append(ErrorType.CONJ)
            elif src_tok.upos == "PART":
                types.append(ErrorType.PART)
            else:
                types.append(ErrorType.WORD_CHOICE)

        else:
            if _is_compound_variant(src_tok.text, hyp_tok.text):
                types.append(ErrorType.COMPOUND)
            elif _is_spelling_variant(src_tok.text, hyp_tok.text):
                types.append(ErrorType.SPELL)
            else:
                types.append(ErrorType.OTHER)

        return types or [ErrorType.OTHER]

    # ------------------------------------------------------------------
    # 4. Full pipeline
    # ------------------------------------------------------------------

    def annotate(self, src_text: str, hyp_text: str) -> list[Edit]:
        """Full pipeline: parse → align → classify."""
        src_tokens = self.parse(src_text)
        hyp_tokens = self.parse(hyp_text)
        return [self.classify(e) for e in self.align(src_tokens, hyp_tokens)]

    # ------------------------------------------------------------------
    # 5. Score
    # ------------------------------------------------------------------

    def error_samples(
        self,
        sources: list[str],
        predictions: list[str],
        references: list[str],
        n: int = 10,
        batch: bool = True,
        batch_size: int = 0,
    ) -> dict:
        """
        Collect example FP, FN, and OTHER-type samples for error analysis.

        Args:
            batch:      If True, prewarm the parse cache with one batched
                        Stanza call over all unique inputs (fast).
                        If False, parse on demand one sentence at a time
                        (slow; only useful for debugging or very memory-
                        constrained environments).
            batch_size: When batch=True, chunk Stanza calls into groups of
                        this size. 0 = single call for all inputs.

        Returns:
            {
                'fp':    [ {'corrupted', 'prediction', 'reference', 'fp_edits'}, ... ],
                'fn':    [ {'corrupted', 'prediction', 'reference', 'fn_edits'}, ... ],
                'other': [ {'corrupted', 'prediction', 'reference', 'other_edits'}, ... ],
            }
        """
        if batch:
            # Batch-parse every unique input string upfront so subsequent
            # annotate() calls become cache lookups + alignment.
            self._parse_many(list(sources) + list(predictions) + list(references),
                             batch_size=batch_size)

        fp_samples, fn_samples, other_samples = [], [], []

        for src, pred, ref in tqdm(zip(sources, predictions, references),
                                    total=len(sources), desc="Collecting FP/FN/OTHER", unit="ex"):
            sys_edits  = self.annotate(src, pred)
            gold_edits = self.annotate(src, ref)

            sys_keys  = {_edit_key(e): e for e in sys_edits}
            gold_keys = {_edit_key(e): e for e in gold_edits}

            fp_edits    = [sys_keys[k]  for k in sys_keys  if k not in gold_keys]
            fn_edits    = [gold_keys[k] for k in gold_keys if k not in sys_keys]
            other_edits = [e for e in fp_edits if ErrorType.OTHER in e.error_types]

            if fp_edits and len(fp_samples) < n:
                fp_samples.append({
                    'corrupted':  src,
                    'prediction': pred,
                    'reference':  ref,
                    'fp_edits':   [repr(e) for e in fp_edits],
                })
            if fn_edits and len(fn_samples) < n:
                fn_samples.append({
                    'corrupted':  src,
                    'prediction': pred,
                    'reference':  ref,
                    'fn_edits':   [repr(e) for e in fn_edits],
                })
            if other_edits and len(other_samples) < n:
                other_samples.append({
                    'corrupted':   src,
                    'prediction':  pred,
                    'reference':   ref,
                    'other_edits': [repr(e) for e in other_edits],
                })
            if len(fp_samples) >= n and len(fn_samples) >= n and len(other_samples) >= n:
                break

        return {'fp': fp_samples, 'fn': fn_samples, 'other': other_samples}

    def score(
        self,
        sources: list[str],
        predictions: list[str],
        references: list[str],
        verbose: bool = False,
        return_per_sample: bool = False,
        collect_samples: int = 0,
        batch: bool = True,
        batch_size: int = 0,
    ) -> dict:
        """
        Compute F0.5, precision, recall — overall and per error type.

        Args:
            sources:     Original corrupted sentences
            predictions: Model predictions
            references:  Gold corrections
            collect_samples: If > 0, also gather up to N FP / FN / OTHER example
                             dicts during the same pass (replaces a second
                             traversal via error_samples()).
            batch:       If True, prewarm the parse cache with one batched
                         Stanza call over all unique inputs (fast — recommended).
                         If False, parse on demand one sentence at a time
                         (slow; useful for debugging or memory-constrained
                         environments).
            batch_size:  When batch=True, chunk Stanza calls into groups of
                         this size. 0 = single call for all inputs.

        Returns:
            {
                'precision': float, 'recall': float, 'f05': float,
                'tp': int, 'fp': int, 'fn': int,
                'by_type': {error_type: {'precision', 'recall', 'f05', 'tp', 'fp', 'fn'}},
                # optional:
                'per_sample': [...],
                'samples': {'fp': [...], 'fn': [...], 'other': [...]},
            }
        """
        if batch:
            # Batch-parse every unique input string in ONE Stanza call. After
            # this, annotate()/parse() become pure cache lookups + alignment +
            # classification in Python, which are negligible compared to the
            # Stanza forward passes.
            print(f"[Info] Batch-parsing up to {len(sources)*3} texts with Stanza"
                  f"{f' (chunks of {batch_size})' if batch_size else ''}...")
            self._parse_many(list(sources) + list(predictions) + list(references),
                             batch_size=batch_size)
        else:
            print("[Info] Batch parsing disabled — falling back to per-sentence Stanza calls.")

        total_tp = total_fp = total_fn = 0
        type_counts: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        per_sample = []

        fp_samples: list[dict] = []
        fn_samples: list[dict] = []
        other_samples: list[dict] = []
        want_samples = collect_samples > 0

        for src, pred, ref in tqdm(zip(sources, predictions, references),
                                    total=len(sources), desc="Scoring", unit="ex"):
            sys_edits_list  = self.annotate(src, pred)
            gold_edits_list = self.annotate(src, ref)

            sys_keys_to_edit  = {_edit_key(e): e for e in sys_edits_list}
            gold_keys_to_edit = {_edit_key(e): e for e in gold_edits_list}
            sys_edits  = set(sys_keys_to_edit.keys())
            gold_edits = set(gold_keys_to_edit.keys())

            tp_keys = sys_edits & gold_edits
            fp_keys = sys_edits - gold_edits
            fn_keys = gold_edits - sys_edits

            total_tp += len(tp_keys)
            total_fp += len(fp_keys)
            total_fn += len(fn_keys)

            for key in tp_keys:
                for t in gold_keys_to_edit[key].error_types or (ErrorType.OTHER,):
                    type_counts[t]["tp"] += 1
            for key in fp_keys:
                for t in sys_keys_to_edit[key].error_types or (ErrorType.OTHER,):
                    type_counts[t]["fp"] += 1
            for key in fn_keys:
                for t in gold_keys_to_edit[key].error_types or (ErrorType.OTHER,):
                    type_counts[t]["fn"] += 1

            if verbose:
                print(f"\nsrc:  {src}\npred: {pred}\nref:  {ref}")
                print(f"  tp={len(tp_keys)} fp={len(fp_keys)} fn={len(fn_keys)}")

            if return_per_sample:
                fp_by_type: dict[str, int] = defaultdict(int)
                for key in fp_keys:
                    for t in sys_keys_to_edit[key].error_types or (ErrorType.OTHER,):
                        fp_by_type[t] += 1
                per_sample.append({
                    "tp": len(tp_keys),
                    "fp": len(fp_keys),
                    "fn": len(fn_keys),
                    "fp_by_type": dict(fp_by_type),
                })

            if want_samples:
                need_fp    = len(fp_samples)    < collect_samples
                need_fn    = len(fn_samples)    < collect_samples
                need_other = len(other_samples) < collect_samples
                if need_fp and fp_keys:
                    fp_edits = [sys_keys_to_edit[k] for k in fp_keys]
                    fp_samples.append({
                        'corrupted':  src,
                        'prediction': pred,
                        'reference':  ref,
                        'fp_edits':   [repr(e) for e in fp_edits],
                    })
                    if need_other:
                        other_edits = [e for e in fp_edits if ErrorType.OTHER in e.error_types]
                        if other_edits:
                            other_samples.append({
                                'corrupted':   src,
                                'prediction':  pred,
                                'reference':   ref,
                                'other_edits': [repr(e) for e in other_edits],
                            })
                if need_fn and fn_keys:
                    fn_edits = [gold_keys_to_edit[k] for k in fn_keys]
                    fn_samples.append({
                        'corrupted':  src,
                        'prediction': pred,
                        'reference':  ref,
                        'fn_edits':   [repr(e) for e in fn_edits],
                    })

        overall = _compute_f05(total_tp, total_fp, total_fn)
        by_type = {
            etype: _compute_f05(c["tp"], c["fp"], c["fn"])
            for etype, c in sorted(type_counts.items())
        }
        result = {**overall, "by_type": by_type}
        if return_per_sample:
            result["per_sample"] = per_sample
        if want_samples:
            result["samples"] = {
                "fp": fp_samples,
                "fn": fn_samples,
                "other": other_samples,
            }
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fd(t1: Token, t2: Token, key: str) -> bool:
    """Feature differs: True only when both tokens have the feature and values differ."""
    v1, v2 = t1.feat(key), t2.feat(key)
    return v1 is not None and v2 is not None and v1 != v2


def _edit_key(edit: Edit) -> tuple:
    """Unique key for an edit: source span + hypothesis tokens text."""
    return (edit.src_interval, tuple(t.text for t in edit.hyp_tokens))


def _is_vowel_harmony_error(a: str, b: str) -> bool:
    """True if the only differences between a and b are vowel harmony pairs (a↔ä, o↔ö, u↔y)."""
    if len(a) != len(b):
        return False
    diffs = [(x.lower(), y.lower()) for x, y in zip(a, b) if x.lower() != y.lower()]
    return len(diffs) > 0 and all(pair in _HARMONY_PAIRS for pair in diffs)


def _is_compound_variant(a: str, b: str) -> bool:
    """True if one string is a prefix/suffix of the other (≥4 chars difference), suggesting compound join/split."""
    x, y = a.lower(), b.lower()
    longer, shorter = (x, y) if len(x) >= len(y) else (y, x)
    return (
        len(longer) - len(shorter) >= 4
        and (longer.startswith(shorter) or longer.endswith(shorter))
    )


def _is_spelling_variant(a: str, b: str) -> bool:
    """True if Levenshtein edit distance between a and b is ≤ 2."""
    if abs(len(a) - len(b)) > 2:
        return False
    if _Levenshtein is not None:
        return _Levenshtein.distance(a, b) <= 2
    return sum(c1 != c2 for c1, c2 in zip(a, b)) + abs(len(a) - len(b)) <= 2


def _compute_f05(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    beta = 0.5
    f05 = (
        (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    return {
        "precision": round(precision * 100, 4),
        "recall":    round(recall * 100, 4),
        "f05":       round(f05 * 100, 4),
        "tp": tp, "fp": fp, "fn": fn,
    }

# FinnishERRANT

Evaluation tool for Finnish Grammatical Error Correction (GEC) systems.

Based on [ERRANT](https://github.com/chrisjbryant/errant) (Bryant et al., 2017), adapted for Finnish morphology using the [Stanza](https://stanfordnlp.github.io/stanza/) Finnish NLP pipeline.

**Output:** Precision, Recall, F0.5 — broken down by error type (CASE, VERB:FORM, VOWEL:HARMONY, etc.)

---

## Setup

```bash
pip install -r requirements.txt
```

The Finnish Stanza model downloads automatically on first run.

---

## Input format

Your predictions file must be a JSONL file — one JSON object per line, with three fields:

```json
{"corrupted": "Minä asua Helsingissä", "prediction": "Minä asun Helsingissä", "reference": "Minä asun Helsingissä"}
```

| Field | Description |
|---|---|
| `corrupted` | The original sentence with errors (model input) |
| `prediction` | Your model's correction |
| `reference` | The gold-standard correction |

---

## Usage

**Basic evaluation (prints results to terminal):**
```bash
python evaluate.py --predictions my_model.jsonl
```

**Save full report to a directory:**
```bash
python evaluate.py --predictions my_model.jsonl --report-dir reports/my_model
```

This saves:
- `report.json` — all metrics + breakdown by error type
- `report.md` — human-readable version
- `10_fp.json` — 10 example false corrections (over-corrections)
- `10_fn.json` — 10 example missed errors
- `10_other.json` — 10 example OTHER-type edits

**Evaluate multiple files at once:**
```bash
python evaluate.py --predictions model_a.jsonl model_b.jsonl --report-dir reports/
```

**Other options:**
```bash
--samples 20       # save 20 FP/FN examples instead of 10
--max-examples 500 # evaluate only first 500 examples (quick check)
--gpu              # use GPU for faster parsing
```

---

## Output example

```
============================================================
ERRANT EVALUATION RESULTS
============================================================
  F0.5:      50.80%
  Precision: 53.70%
  Recall:    41.90%
  TP: 1234  FP: 567  FN: 890

By error type:
  Type                   F0.5        P        R     TP     FP     FN
  ------------------------------------------------------------------
  CASE                  61.20    63.50    54.10    456    123    210
  VERB:FORM             48.30    51.20    39.80    234     98    178
  VOWEL:HARMONY         44.10    47.80    35.20    123     67    145
  ...
============================================================
```

---

## Error types

| Type | Description | Example |
|---|---|---|
| CASE | Wrong grammatical case | *talossa* → *talosta* |
| VERB:FORM | Wrong verb form | *asua* → *asun* |
| NUMBER | Singular vs plural | *talo* → *talot* |
| PERSON | Wrong person | *asuu* → *asun* |
| VERB:TENSE | Wrong tense | *menee* → *meni* |
| VERB:MOOD | Wrong mood | *tulee* → *tulisi* |
| VOWEL:HARMONY | Back/front vowel mismatch | *talossä* → *talossa* |
| SPELL | Spelling error (edit distance ≤ 2) | |
| COMPOUND | Compound split or joined | *auto talli* → *autotalli* |
| INSERT / DELETE | Missing or extra word | |
| WORD:CHOICE | Wrong word, same POS | |
| OTHER | Unclassified | |

---

## Why F0.5?

In GEC, **over-correcting correct text is worse than missing an error**. F0.5 weights Precision twice as heavily as Recall, penalising false corrections more than missed corrections.
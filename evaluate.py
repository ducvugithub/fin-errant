#!/usr/bin/env python3
"""
Evaluate GEC predictions using FinnishERRANT.

Input JSONL — each line must have three fields:
    {"corrupted": "...", "prediction": "...", "reference": "..."}

Usage:
    python evaluate.py --predictions my_model.jsonl
    python evaluate.py --predictions my_model.jsonl --report-dir reports/my_model
    python evaluate.py --predictions my_model.jsonl --report-dir reports/my_model --samples 20
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from finnish_errant import FinnishERRANT


def load_predictions(path: Path):
    with open(path, encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    sources     = [d['corrupted']   for d in data]
    predictions = [d['prediction']  for d in data]
    references  = [d['reference']   for d in data]
    return sources, predictions, references


def format_report(results: dict, predictions_path: Path) -> str:
    lines = []
    lines.append("# ERRANT Evaluation Report")
    lines.append("")
    lines.append(f"**Predictions:** `{predictions_path}`  ")
    lines.append(f"**Examples:** {results.get('num_examples', 'N/A')}")
    lines.append("")
    lines.append("```")
    lines.append("=" * 60)
    lines.append("ERRANT EVALUATION RESULTS")
    lines.append("=" * 60)
    lines.append(f"  F0.5:      {results['f05']:.2f}%")
    lines.append(f"  Precision: {results['precision']:.2f}%")
    lines.append(f"  Recall:    {results['recall']:.2f}%")
    lines.append(f"  TP: {results['tp']}  FP: {results['fp']}  FN: {results['fn']}")

    if results.get('by_type'):
        lines.append("")
        lines.append("By error type:")
        lines.append(f"  {'Type':<20} {'F0.5':>7}  {'P':>7}  {'R':>7}  {'TP':>5}  {'FP':>5}  {'FN':>5}")
        lines.append(f"  {'-'*58}")
        for etype, m in sorted(results['by_type'].items(), key=lambda x: -x[1]['f05']):
            if m['tp'] + m['fp'] + m['fn'] == 0:
                continue
            lines.append(
                f"  {etype:<20} {m['f05']:>7.2f}  {m['precision']:>7.2f}  {m['recall']:>7.2f}"
                f"  {m['tp']:>5}  {m['fp']:>5}  {m['fn']:>5}"
            )
    if results.get('zero_error'):
        z = results['zero_error']
        overcorrected = round(z['overcorrection_rate'] / 100 * z['count'])
        lines.append("")
        lines.append("0-Error Sample Analysis (over-correction):")
        lines.append(f"  {'Number of samples':<35} {z['count']}")
        lines.append(f"  {'Number of samples over-corrected':<35} {overcorrected}/{z['count']} ({z['overcorrection_rate']:.1f}%)")
        lines.append(f"  {'Number of total FP edits':<35} {z['fp']}")
        if z.get('fp_by_type'):
            lines.append("")
            lines.append("  FP edits by type:")
            lines.append(f"  {'Type':<20} {'FP':>5}")
            lines.append(f"  {'-'*28}")
            for etype, count in sorted(z['fp_by_type'].items(), key=lambda x: -x[1]):
                lines.append(f"  {etype:<20} {count:>5}")
    lines.append("=" * 60)
    lines.append("```")
    return "\n".join(lines)


def print_results(results: dict):
    print("\n" + "=" * 60)
    print("ERRANT EVALUATION RESULTS")
    print("=" * 60)
    print(f"  F0.5:      {results['f05']:.2f}%")
    print(f"  Precision: {results['precision']:.2f}%")
    print(f"  Recall:    {results['recall']:.2f}%")
    print(f"  TP: {results['tp']}  FP: {results['fp']}  FN: {results['fn']}")

    if results.get('by_type'):
        print("\nBy error type:")
        print(f"  {'Type':<20} {'F0.5':>7}  {'P':>7}  {'R':>7}  {'TP':>5}  {'FP':>5}  {'FN':>5}")
        print(f"  {'-'*58}")
        for etype, m in sorted(results['by_type'].items(), key=lambda x: -x[1]['f05']):
            if m['tp'] + m['fp'] + m['fn'] == 0:
                continue
            print(f"  {etype:<20} {m['f05']:>7.2f}  {m['precision']:>7.2f}  {m['recall']:>7.2f}"
                  f"  {m['tp']:>5}  {m['fp']:>5}  {m['fn']:>5}")
    if results.get('zero_error'):
        z = results['zero_error']
        overcorrected = round(z['overcorrection_rate'] / 100 * z['count'])
        print(f"\n0-Error Sample Analysis (over-correction):")
        print(f"  {'Number of samples':<35} {z['count']}")
        print(f"  {'Number of samples over-corrected':<35} {overcorrected}/{z['count']} ({z['overcorrection_rate']:.1f}%)")
        print(f"  {'Number of total FP edits':<35} {z['fp']}")
        if z.get('fp_by_type'):
            print(f"\n  FP edits by type:")
            print(f"  {'Type':<20} {'FP':>5}")
            print(f"  {'-'*28}")
            for etype, count in sorted(z['fp_by_type'].items(), key=lambda x: -x[1]):
                print(f"  {etype:<20} {count:>5}")
    print("=" * 60 + "\n")


def evaluate_one(errant, predictions_path: Path, report_dir: Path, max_examples, samples_n, verbose):
    print(f"\n[Info] Loading predictions from {predictions_path}")
    sources, predictions, references = load_predictions(predictions_path)
    if max_examples:
        sources, predictions, references = sources[:max_examples], predictions[:max_examples], references[:max_examples]
    print(f"[Info] Loaded {len(predictions):,} examples")

    results = errant.score(sources, predictions, references, verbose=verbose, return_per_sample=True)
    results['num_examples'] = len(predictions)

    zero_idx = [i for i, (s, r) in enumerate(zip(sources, references)) if s == r]
    if zero_idx:
        zero_fp = sum(results['per_sample'][i]['fp'] for i in zero_idx)
        zero_fp_by_type: dict = defaultdict(int)
        for i in zero_idx:
            for etype, count in results['per_sample'][i]['fp_by_type'].items():
                zero_fp_by_type[etype] += count
        overcorrected = sum(1 for i in zero_idx if sources[i] != predictions[i])
        results['zero_error'] = {
            'count': len(zero_idx),
            'fp': zero_fp,
            'tp': 0,
            'fn': 0,
            'overcorrection_rate': overcorrected / len(zero_idx) * 100,
            'fp_by_type': dict(zero_fp_by_type),
        }

    results.pop('per_sample', None)
    print_results(results)

    if report_dir:
        report_dir.mkdir(parents=True, exist_ok=True)

        with open(report_dir / "report.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[Info] Report saved to {report_dir / 'report.json'}")

        with open(report_dir / "report.md", 'w', encoding='utf-8') as f:
            f.write(format_report(results, predictions_path))
        print(f"[Info] Report saved to {report_dir / 'report.md'}")

        print(f"[Info] Collecting {samples_n} FP/FN samples...")
        samples = errant.error_samples(sources, predictions, references, n=samples_n)
        for key, path in [('fp', f"{samples_n}_fp.json"), ('fn', f"{samples_n}_fn.json"), ('other', f"{samples_n}_other.json")]:
            with open(report_dir / path, 'w', encoding='utf-8') as f:
                json.dump(samples[key], f, indent=2, ensure_ascii=False)
        print(f"[Info] FP/FN/OTHER samples saved to {report_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate GEC predictions using FinnishERRANT'
    )
    parser.add_argument('--predictions', type=Path, required=True, nargs='+',
                        help='One or more prediction JSONL files')
    parser.add_argument('--report-dir', type=Path, default=None,
                        help='Directory to save report files (optional)')
    parser.add_argument('--max-examples', type=int, default=None,
                        help='Limit number of examples to evaluate')
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of FP/FN example samples to save (default: 10)')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for stanza parsing (faster)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    for p in args.predictions:
        if not p.exists():
            print(f"[Error] File not found: {p}")
            sys.exit(1)

    print(f"[Info] Loading FinnishERRANT (stanza Finnish model)...")
    errant = FinnishERRANT(use_gpu=args.gpu)

    for pred_path in args.predictions:
        if len(args.predictions) == 1:
            report_dir = args.report_dir
        else:
            report_dir = (args.report_dir / pred_path.stem) if args.report_dir else None
        evaluate_one(errant, pred_path, report_dir, args.max_examples, args.samples, args.verbose)


if __name__ == '__main__':
    main()
"""
Leave-One-Year-Out Cross-Validation
====================================
For each year in corpus.json:
  1. Self-trained: run growth on that year's text alone
  2. Transfer: train on (corpus minus that year), then evaluate on that year

Reports F1 vs jieba for both settings, showing where transfer helps.

Usage:
    python train.py [corpus_json_path]

    Default: corpus/corpus.json

Requires:
    - jieba (baseline)
    - tabulate (auto-installed via core._make_table)
"""
from __future__ import annotations
import sys, json, re, os
from typing import Dict, List
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.join(_THIS_DIR, '..'))

import jieba
from scripts.core import AtomicCrystalGrowth, Config, _make_table


# ─── Utilities ────────────────────────────────────────────────────────────────

def load_corpus(path: str) -> List[Dict[str, str]]:
    return json.load(open(path, 'r', encoding='utf-8'))

def get_sents(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'[。？！.?!<>\n]+', text) if s.strip()]

def tokenize_jieba(text: str) -> List[List[str]]:
    return [list(jieba.cut(s)) for s in get_sents(text)]

def boundaries(tokens: List[str]) -> set:
    p, b = 0, []
    for t in tokens:
        p += len(t); b.append(p)
    return set(b[:-1])

def compute_f1(crystal: List[List[str]], jieba_t: List[List[str]]) -> float:
    tp, rf = 0, 0
    for c, j in zip(crystal, jieba_t):
        cb, jb = boundaries(c), boundaries(j)
        tp += len(cb & jb); rf += len(cb) + len(jb)
    return 2 * tp / max(rf, 1)


# ─── Evaluate one year ────────────────────────────────────────────────────────

def evaluate_year(year: str, year_text: str,
                  corpus_text: str, jieba_s: List[List[str]]) -> Dict:
    n = len(jieba_s)

    g1 = AtomicCrystalGrowth(year_text)
    r1 = g1.run(quiet=True)
    f1_self = compute_f1(r1[:n], jieba_s[:n])

    gc = AtomicCrystalGrowth(corpus_text)
    gc.run(quiet=True)
    test_init = AtomicCrystalGrowth._preprocess(year_text)
    combined = gc.sentences + test_init
    n_c = len(gc.sentences)
    g2 = AtomicCrystalGrowth('', config=Config())
    g2.sentences = combined
    g2.iteration = gc.iteration
    for i in range(gc.iteration, gc.iteration + 20):
        mode = 'all' if i % 2 == 0 else 'atomic'
        changed = g2.grow(mode, i)
        if not changed and mode == 'all' and i >= gc.iteration + 2:
            break
    r2 = g2.sentences[n_c:]
    f1_transfer = compute_f1(r2[:n], jieba_s[:n])

    return {
        'year': year,
        'len': len(year_text),
        'f1_self': f1_self,
        'f1_transfer': f1_transfer,
        'delta': f1_transfer - f1_self,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(corpus_path: str) -> None:
    data = load_corpus(corpus_path)
    all_texts = {a['year']: a['text'] for a in data}
    years = sorted(all_texts.keys(), key=lambda y: int(y))

    print("=" * 70)
    print("  Leave-One-Year-Out Cross-Validation")
    print("=" * 70)
    print(f"Corpus: {corpus_path} ({len(years)} years)")
    print(f"Total chars: {sum(len(t) for t in all_texts.values())}\n")

    results: List[Dict] = []
    for year in years:
        year_text = all_texts[year]
        corpus_text = '\n'.join(t for y, t in all_texts.items() if y != year)
        jieba_s = tokenize_jieba(year_text)

        r = evaluate_year(year, year_text, corpus_text, jieba_s)
        results.append(r)

        arrow = '↑' if r['delta'] > 0 else ('↓' if r['delta'] < 0 else '→')
        print(f"  {year} ({r['len']:>4d} chars): "
              f"self={r['f1_self']:.4f}  "
              f"transfer={r['f1_transfer']:.4f}  "
              f"Δ={r['delta']:+.4f} {arrow}")

    print("\n" + "-" * 70)
    avg_self = sum(r['f1_self'] for r in results) / len(results)
    avg_trans = sum(r['f1_transfer'] for r in results) / len(results)
    wins = sum(1 for r in results if r['delta'] > 0)
    losses = sum(1 for r in results if r['delta'] < 0)

    print(f"\nAverage F1 (self):     {avg_self:.4f}")
    print(f"Average F1 (transfer): {avg_trans:.4f}")
    print(f"Average Δ:             {avg_trans - avg_self:+.4f}")
    print(f"Transfer wins:  {wins}/{len(results)} ({wins/max(len(results),1)*100:.0f}%)")
    print(f"Transfer losses: {losses}/{len(results)} ({losses/max(len(results),1)*100:.0f}%)")

    results_by_delta = sorted(results, key=lambda r: -r['delta'])
    print("\nBest transfer improvements (top 5):")
    for r in results_by_delta[:5]:
        if r['delta'] > 0:
            print(f"  {r['year']}: +{r['delta']:.4f}  "
                  f"(self={r['f1_self']:.4f} → transfer={r['f1_transfer']:.4f})")

    print("\nWorst transfer regressions (bottom 5):")
    for r in results_by_delta[-5:]:
        if r['delta'] < 0:
            print(f"  {r['year']}: {r['delta']:.4f}  "
                  f"(self={r['f1_self']:.4f} → transfer={r['f1_transfer']:.4f})")

    print("\nAll results:")
    tbl = [[r['year'], f"{r['len']:>4d}", f"{r['f1_self']:.4f}",
            f"{r['f1_transfer']:.4f}", f"{r['delta']:+.4f}",
            '↑' if r['delta'] > 0 else ('↓' if r['delta'] < 0 else '→')]
           for r in sorted(results, key=lambda x: -int(x['year']))]
    print(_make_table(tbl, headers=['Year', 'Chars', 'F1(self)', 'F1(transfer)', 'Δ', ''],
                      colalign=('center', 'center', 'center', 'center', 'center', 'center')))

    return results


if __name__ == '__main__':
    default = os.path.join(os.path.dirname(_THIS_DIR), 'corpus', 'corpus.json')
    path = sys.argv[1] if len(sys.argv) > 1 else default
    main(path)

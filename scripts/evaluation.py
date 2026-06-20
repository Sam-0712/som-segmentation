"""
Evaluation — Crystal Growth vs Jieba Baseline
==============================================
Compare AtomicCrystalGrowth output against jieba standard segmentation
line by line. Reports boundary F1, granularity, and n-gram overlap.

Usage:
    python evaluation.py [text_path]

    Default text_path: corpus/example.txt

Requires:
    - jieba (baseline)
    - tabulate (auto-installed via core._make_table)
"""
from __future__ import annotations
import sys, re, os
from typing import Dict, List
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.join(_THIS_DIR, '..'))

import jieba
from scripts.core import AtomicCrystalGrowth, _make_table


# ─── Boundary utilities ───────────────────────────────────────────────────────

def sent_to_bounds(tokens: List[str]) -> List[int]:
    pos, bounds = 0, []
    for t in tokens:
        pos += len(t); bounds.append(pos)
    return bounds[:-1]


def compute_boundary_metrics(crystal: List[List[str]], jieba_t: List[List[str]]) -> Dict:
    tp_pre, tp_rec, den_pre, den_rec = 0, 0, 0, 0
    sym_diff = 0.0
    n = min(len(crystal), len(jieba_t))
    for c, j in zip(crystal, jieba_t):
        cb = set(sent_to_bounds(c)); jb = set(sent_to_bounds(j))
        inter = cb & jb
        tp_pre += len(inter); den_pre += len(jb)
        tp_rec += len(inter); den_rec += len(cb)
        ul = len(cb) + len(jb)
        if ul > 0:
            sym_diff += 2 * len(cb ^ jb) / ul
    prec = tp_pre / max(den_pre, 1)
    rec = tp_rec / max(den_rec, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {'precision': round(prec, 4), 'recall': round(rec, 4),
            'f1': round(f1, 4),
            'avg_symmetric_diff': round(sym_diff / max(n, 1), 4)}


def compute_granularity(crystal: List[List[str]], jieba_t: List[List[str]]) -> Dict:
    lens_c = [len(s) for s in crystal]
    lens_j = [len(s) for s in jieba_t]
    avg_c = sum(lens_c) / max(len(lens_c), 1)
    avg_j = sum(lens_j) / max(len(lens_j), 1)
    all_lens_c = [len(t) for s in crystal for t in s]
    all_lens_j = [len(t) for s in jieba_t for t in s]
    return {
        'avg_particles_crystal': round(avg_c, 2),
        'avg_particles_jieba': round(avg_j, 2),
        'avg_token_len_crystal': round(sum(all_lens_c) / max(len(all_lens_c), 1), 3),
        'avg_token_len_jieba': round(sum(all_lens_j) / max(len(all_lens_j), 1), 3),
        'ratio': round(avg_c / max(avg_j, 0.01), 3),
    }


def compute_ngram_overlap(crystal: List[List[str]], jieba_t: List[List[str]]) -> Dict:
    s1c, s1j, s2c, s2j = set(), set(), set(), set()
    for c, j in zip(crystal, jieba_t):
        s1c.update(c); s1j.update(j)
        for a, b in zip(c[:-1], c[1:]): s2c.add((a, b))
        for a, b in zip(j[:-1], j[1:]): s2j.add((a, b))
    def jacc(a, b):
        u = a | b
        return len(a & b) / max(len(u), 1)
    return {'unigram_jaccard': round(jacc(s1c, s1j), 4),
            'bigram_jaccard': round(jacc(s2c, s2j), 4)}


# ─── Main comparison ──────────────────────────────────────────────────────────

def compare(text_path: str, sample_count: int = 20) -> None:
    raw_text = open(text_path, 'r', encoding='utf-8').read()

    g = AtomicCrystalGrowth(raw_text)
    crystal_result = g.run(quiet=True)

    from scripts.core import split_sentences
    raw_sents = split_sentences(raw_text)
    jieba_result = [list(jieba.cut(s)) for s in raw_sents]

    n = min(len(crystal_result), len(jieba_result))
    crystal_result = crystal_result[:n]
    jieba_result = jieba_result[:n]

    bm = compute_boundary_metrics(crystal_result, jieba_result)
    gm = compute_granularity(crystal_result, jieba_result)
    nm = compute_ngram_overlap(crystal_result, jieba_result)

    print("=" * 70)
    print("  Crystal Growth vs Jieba — Baseline Report")
    print("=" * 70)
    s = g.stats
    print(f"\nText: {text_path}")
    print(f"Sentences: {n}")
    print(f"Crystal: {s['rounds']} rounds | {s['initial']}→{s['final']} particles ({s['reduction_pct']:.1f}%)")

    print("\n[1] Boundary Metrics:")
    print(_make_table([['Crystal vs Jieba', str(bm['precision']), str(bm['recall']),
                        str(bm['f1']), str(bm['avg_symmetric_diff'])]],
                      headers=['', 'Precision', 'Recall', 'F1', 'Sym Diff'],
                      colalign=('left', 'center', 'center', 'center', 'center')))

    print("\n[2] Granularity:")
    print(_make_table([
        ['Avg particles/sent', str(gm['avg_particles_crystal']), str(gm['avg_particles_jieba'])],
        ['Avg token length', str(gm['avg_token_len_crystal']), str(gm['avg_token_len_jieba'])],
        ['Ratio (crystal/jieba)', str(gm['ratio']), '-']],
        headers=['Metric', 'Crystal', 'Jieba'],
        colalign=('left', 'center', 'center')))

    print("\n[3] N-gram Overlap:")
    print(_make_table([
        ['Unigram Jaccard', str(nm['unigram_jaccard'])],
        ['Bigram Jaccard', str(nm['bigram_jaccard'])]],
        headers=['Metric', 'Score'],
        colalign=('left', 'center')))

    print(f"\n[4] Sample comparison (first {min(sample_count, n)} sentences):")
    sample = []
    for i in range(min(sample_count, n)):
        cs = ' / '.join(crystal_result[i])
        js = ' / '.join(jieba_result[i])
        diff = len(set(sent_to_bounds(crystal_result[i])) ^ set(sent_to_bounds(jieba_result[i])))
        if len(cs) > 50: cs = cs[:50] + ' …'
        if len(js) > 50: js = js[:50] + ' …'
        sample.append([str(i + 1), cs, js, str(diff)])
    print(_make_table(sample, headers=['#', 'Crystal', 'Jieba', 'Diff'],
                      colalign=('center', 'left', 'left', 'center')))

    print("\n" + "=" * 70)
    print(f"  F1: {bm['f1']:.3f}  |  Jaccard: {nm['unigram_jaccard']:.3f}  |  Ratio: {gm['ratio']:.2f}x")
    if bm['f1'] > 0.6:
        print("  → High boundary agreement with jieba")
    elif bm['f1'] > 0.4:
        print("  → Moderate boundary agreement")
    else:
        print("  → Low boundary agreement")
    print("=" * 70)

    return {'boundary': bm, 'granularity': gm, 'ngram': nm, 'sentences_compared': n}


if __name__ == '__main__':
    default = os.path.join(os.path.dirname(_THIS_DIR), 'corpus', 'example.txt')
    path = sys.argv[1] if len(sys.argv) > 1 else default
    compare(path)

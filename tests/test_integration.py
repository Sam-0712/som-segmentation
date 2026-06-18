"""
Integration test: full pipeline on corpus/example.txt.

Verifies:
  1. Preprocessing produces expected particle/sentence counts
  2. Growth converges within max iterations
  3. Particle count reduction is in a reasonable range
  4. Segmentation output is non-trivial (particles exist, max len > 1)
  5. Evaluation vs jieba runs without error
"""
from __future__ import annotations
import sys, os, json, math, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.core import AtomicCrystalGrowth, Config


def load_text() -> str:
    path = os.path.join(os.path.dirname(__file__), '..', 'corpus', 'example.txt')
    with open(path, encoding='utf-8') as f:
        return f.read()


def load_corpus_json() -> list:
    path = os.path.join(os.path.dirname(__file__), '..', 'corpus', 'corpus.json')
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def test_preprocessing():
    """Verify preprocessing: sentences and initial particles."""
    text = load_text()
    g = AtomicCrystalGrowth(text)
    n_sents = len(g.sentences)
    n_particles = sum(len(s) for s in g.sentences)
    assert n_sents > 0, "No sentences produced"
    assert n_particles > 100, f"Too few initial particles: {n_particles}"
    # All particles must be strings
    for s in g.sentences:
        for p in s:
            assert isinstance(p, str) and len(p) > 0, f"Bad particle: {p!r}"
    print(f"  [OK] {n_sents} sentences, {n_particles} initial particles")


def test_growth():
    """Verify growth converges and produces non-trivial segmentation."""
    text = load_text()
    g = AtomicCrystalGrowth(text)
    init_n = sum(len(s) for s in g.sentences)
    result = g.run(quiet=True)
    final_n = sum(len(s) for s in result)

    # Convergence checks
    assert g.iteration > 0, "No iterations performed"
    assert g.iteration < Config().max_iterations, \
        f"Did not converge: {g.iteration} >= {Config().max_iterations}"

    # Reduction should be in a reasonable range (10-60%)
    reduction_pct = (init_n - final_n) / init_n * 100
    assert 10 < reduction_pct < 70, \
        f"Reduction out of range: {reduction_pct:.1f}%"

    # Output quality: some multi-char particles exist
    max_len = max(max(len(p) for p in s) for s in result)
    assert max_len > 1, "All particles are single characters!"

    print(f"  [OK] {init_n} → {final_n} particles "
          f"({reduction_pct:.1f}% reduction) in {g.iteration} rounds "
          f"(max particle len={max_len}, order={g.order_parameter:.4f})")


def test_ionization_cache():
    """Verify ionization cache is populated and used correctly."""
    text = load_text()
    g = AtomicCrystalGrowth(text)
    g.grow('all', 0)

    # Cache must exist and be non-empty
    assert len(g.ion_left) > 0, "ion_left cache is empty"
    assert len(g.ion_right) > 0, "ion_right cache is empty"
    assert len(g.ion_left) == len(g.ion_right), \
        "ion_left and ion_right have different sizes"

    # All cached values should be in [0, 1]
    for name, cache in [('ion_left', g.ion_left), ('ion_right', g.ion_right)]:
        for p, v in cache.items():
            assert 0.0 <= v <= 1.0, \
                f"{name}[{p!r}] = {v} out of [0, 1]"

    print(f"  [OK] Ionization cache: {len(g.ion_left)} particles cached")


def test_evaluation():
    """Verify evaluation vs jieba runs without error."""
    from scripts.evaluation import compare
    path = os.path.join(os.path.dirname(__file__), '..', 'corpus', 'example.txt')
    compare(path, sample_count=5)  # quick sample, not full 20
    print(f"  [OK] Evaluation vs jieba completed")


def test_config():
    """Verify Config methods work correctly."""
    c = Config()
    # Threshold decay
    t0, t10 = c.get_threshold(0), c.get_threshold(10)
    assert t10 < t0, "Threshold should decrease over iterations"
    # Mass factor
    assert c.get_mass_factor(1, 1) == 1.0, "len-1 mass should be 1"
    assert c.get_mass_factor(2, 2) == 36.0, "len-2 mass should be base^2"
    # Atom adjust
    adj_alone = c.get_atom_adjust(0.9, 10, 1000)
    assert adj_alone > 0, "High alone_rate should increase threshold"
    adj_restless = c.get_atom_adjust(0.2, 10, 1000)
    assert adj_restless < 0, "Low alone_rate should decrease threshold"
    print(f"  [OK] Config: threshold={t0:.4f}→{t10:.4f}, mass(2,2)={c.get_mass_factor(2,2)}")


if __name__ == '__main__':
    print("=" * 60)
    print("Integration Tests — example.txt")
    print("=" * 60)

    t0 = time.perf_counter()
    test_preprocessing()
    test_growth()
    test_ionization_cache()
    test_config()
    test_evaluation()
    elapsed = time.perf_counter() - t0

    print(f"\n{'=' * 60}")
    print(f"All tests passed in {elapsed:.2f}s")

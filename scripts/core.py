"""
Atomic Crystal Growth — Unsupervised Chinese Word Segmentation
================================================================
A physics-inspired algorithm that treats text as a particle system.
Particles (characters/substrings) spontaneously bond based on
statistical forces: NPMI (affinity) drives merging, contextual entropy
(ionization) resists it. Greedy growth + periodic dissolution.

  E_net = NPMI - ionization * mass_factor
  merge if E_net > threshold & is local maximum
"""
from __future__ import annotations
import re, sys, time, collections, math, subprocess
from typing import Dict, List, Tuple, Optional, DefaultDict

# ── tabulate 自动安装 ──────────────────────────────────────────────────────────

def _make_table(rows, headers, **kwargs):
    """Format a table using tabulate (auto-installed if missing)."""
    try:
        from tabulate import tabulate
    except ImportError:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', 'tabulate', '-q'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        from tabulate import tabulate
    return tabulate(rows, headers=headers, tablefmt='grid', **kwargs)

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

class Config:
    """Hyper-parameters for the crystal growth algorithm.

    Key parameters:
        mass_base: controls merge resistance for longer particles (6^(len1+len2-2))
        atom_*: personality-based threshold adjustment for single characters
        dissolution_*: anti-entropy mechanism that splits weak bonds
    """
    def __init__(self) -> None:
        # ─── Threshold ───
        self.base_threshold: float = 0.0
        self.threshold_decay: float = 0.01          # γ per round

        # ─── Mass factor: base ** (len1+len2-2) ───
        self.mass_base: float = 6.0

        # ─── Atom personality ───
        self.atom_independent_threshold: float = 0.70   # P(alone) > 0.7 → protect
        self.atom_independent_penalty: float = 0.50
        self.atom_restless_threshold: float = 0.35      # P(alone) < 0.35 → encourage merge
        self.atom_restless_bonus: float = 0.40
        self.freq_floor: float = 0.02                   # frequency floor → extra protection

        # ─── Entropy smoothing ───
        self.entropy_alpha: float = 1e-6

        # ─── Iteration ───
        self.max_iterations: int = 50

        # ─── Polarity ───
        self.polarity_weight: float = 0.15

        # ─── Convergence detection ───
        self.convergence_window: int = 5
        self.convergence_tol: float = 1e-3
        self.particle_plateau_tol: int = 3

        # ─── Dissolution (anti-entropy) ───
        self.dissolution_enabled: bool = True
        self.dissolution_interval: int = 5
        self.dissolution_min_length: int = 3
        self.dissolution_start_round: int = 3
        self.dissolution_independence_ratio: float = 0.15

    def get_threshold(self, it: int) -> float:
        """Dynamic threshold: linear decay + cosine oscillation."""
        m = 0.85 + 0.15 * math.cos(0.3 * it)
        return self.base_threshold - self.threshold_decay * it * m

    def get_mass_factor(self, l1: int, l2: int) -> float:
        """Binding saturation: longer particles need stronger evidence to merge."""
        return self.mass_base ** (l1 + l2 - 2.0)

    def get_atom_adjust(self, alone_rate: float, freq: int, total_p: int) -> float:
        """Adjust threshold based on atom personality and frequency."""
        adj = 0.0
        if alone_rate > self.atom_independent_threshold:
            adj += (alone_rate - self.atom_independent_threshold) * self.atom_independent_penalty
        elif alone_rate < self.atom_restless_threshold:
            adj -= (self.atom_restless_threshold - alone_rate) * self.atom_restless_bonus
        ratio = freq / max(total_p, 1)
        if ratio > self.freq_floor:
            adj += (ratio - self.freq_floor) * 1.5
        return adj

# ═══════════════════════════════════════════════════════════════════════════
# Core algorithm
# ═══════════════════════════════════════════════════════════════════════════

class AtomicCrystalGrowth:
    """Unsupervised word segmentation via crystal growth simulation."""

    def __init__(self, text: str, config: Optional[Config] = None) -> None:
        self.cfg = config or Config()
        self.sentences: List[List[str]] = self._preprocess(text)
        self.iteration: int = 0

        # Runtime stats (recalculated each grow() call)
        self.total_particles: int = 0
        self.total_pairs: int = 0
        self.vocab_size: int = 0
        self.freq_1: Dict[str, int] = {}            # unigram frequencies
        self.freq_2: Dict[Tuple[str, str], int] = {} # bigram frequencies
        self.left_n: Dict[str, Dict[str, int]] = {}    # left neighbor distributions
        self.right_n: Dict[str, Dict[str, int]] = {}   # right neighbor distributions
        self.atom_per: Dict[str, float] = {}           # P(alone) for single chars
        self.order_parameter: float = 0.0              # structural order metric

    # ────────── Preprocessing ──────────

    @staticmethod
    def _preprocess(raw: str) -> List[List[str]]:
        """Split raw text into sentences → atomic particles.

        Segmentation boundaries:  。？！.?!<>\n
        Each sentence is split into particles:
          - Bracket-enclosed expressions: (1), [2.3]
          - Consecutive ASCII/digits: AI, 2026
          - Single Chinese characters or punctuation
        """
        sents = [s.strip() for s in re.split(r'[。？！.?!<>\n]+', raw) if s.strip()]
        result: List[List[str]] = []
        for s in sents:
            parts = re.findall(
                r'(?:[（(\[【{][0-9\-\.]+?[）)\]】}])'
                r'|[a-zA-Z0-9]+'
                r'|[^\s\w\(\)\[\]\{\}（）「」【】]+'
                r'|[^\s]', s)
            cleaned: List[str] = []
            for p in parts:
                if len(p) > 1 and any(c in p for c in "（([【{") \
                        and not re.match(r'^[（\(\[【{].+[）\)\]】]$', p):
                    cleaned.extend(list(p))
                else:
                    cleaned.append(p)
            if cleaned:
                result.append(cleaned)
        return result

    # ────────── Statistics ──────────

    def calc_stats(self) -> None:
        """Count unigram/bigram frequencies and neighbor distributions.

        Also computes the order parameter (mean normalized ionization),
        which measures structural ordering — higher = more ordered.
        """
        f1: DefaultDict[str, int] = collections.defaultdict(int)
        f2: DefaultDict[Tuple[str, str], int] = collections.defaultdict(int)
        ln: DefaultDict[str, DefaultDict[str, int]] = \
            collections.defaultdict(lambda: collections.defaultdict(int))
        rn: DefaultDict[str, DefaultDict[str, int]] = \
            collections.defaultdict(lambda: collections.defaultdict(int))
        ct: DefaultDict[str, int] = collections.defaultdict(int)  # char total
        ca: DefaultDict[str, int] = collections.defaultdict(int)  # char alone
        tp, tpairs = 0, 0

        for ps in self.sentences:
            n = len(ps)
            tp += n
            for p in ps:
                f1[p] += 1
                if len(p) == 1:
                    ct[p] += 1; ca[p] += 1
                else:
                    for ch in p: ct[ch] += 1
            for i in range(n - 1):
                a, b = ps[i], ps[i + 1]
                f2[(a, b)] += 1; rn[a][b] += 1; ln[b][a] += 1
                tpairs += 1

        self.total_particles = tp
        self.total_pairs = tpairs
        self.freq_1 = dict(f1)
        self.freq_2 = dict(f2)
        self.left_n = {k: dict(v) for k, v in ln.items()}
        self.right_n = {k: dict(v) for k, v in rn.items()}
        self.vocab_size = len(f1)
        self.atom_per = {ch: ca[ch] / max(ct[ch], 1) for ch in ct}

        # Order parameter: mean normalized ionization across all particles
        o_sum = o_cnt = 0.0
        for p, cnt in f1.items():
            if cnt >= 2:
                hl = self._entropy(self.left_n.get(p, {}), cnt)
                hr = self._entropy(self.right_n.get(p, {}), cnt)
                d = math.log2(cnt + 2)
                ion = ((hl / d if d > 0 else 0) + (hr / d if d > 0 else 0)) / 2
                o_sum += ion; o_cnt += 1
        self.order_parameter = o_sum / max(o_cnt, 1)

    def _entropy(self, dist: Dict[str, int], total: int) -> float:
        """Shannon entropy with Laplace smoothing."""
        a = self.cfg.entropy_alpha
        V = self.vocab_size
        den = total + a * V
        e = 0.0
        for c in dist.values():
            p = (c + a) / den
            e -= p * math.log2(p)
        u = V - len(dist)
        if u > 0:
            p0 = a / den
            e -= u * p0 * math.log2(p0)
        return e

    def _npmi(self, a: str, b: str) -> float:
        """Normalized Pointwise Mutual Information ∈ [-1, 1]."""
        c = self.freq_2.get((a, b), 0)
        if c == 0: return -1.0
        pp = c / self.total_pairs
        p_a = self.freq_1[a] / self.total_particles
        p_b = self.freq_1[b] / self.total_particles
        pmi = math.log2(pp / (p_a * p_b))
        return pmi / (-math.log2(pp)) if pp > 0 else 0.0

    def _polarity(self, p: str) -> float:
        """Directional binding preference: |H_L - H_R| / (H_L + H_R)."""
        cnt = self.freq_1.get(p, 0)
        if cnt <= 1: return 0.0
        hl = self._entropy(self.left_n.get(p, {}), cnt)
        hr = self._entropy(self.right_n.get(p, {}), cnt)
        t = hl + hr
        return 0.0 if t < 1e-12 else abs(hl - hr) / t

    @staticmethod
    def _is_punct(p: str) -> bool:
        return bool(re.match(r'^[^\s\w]+$', p))

    # ────────── Growth round ──────────

    def grow(self, mode: str = 'all', iteration: int = 0) -> bool:
        """Execute one round of crystal growth.

        Args:
            mode: 'all' — all particles can merge; 'atomic' — only single chars
            iteration: current round number for threshold calculation

        Returns:
            True if any merge happened
        """
        self.calc_stats()
        if self.vocab_size <= 1: return False
        changed = False
        new_sents: List[List[str]] = []

        for ps in self.sentences:
            n = len(ps)
            if n < 2: new_sents.append(ps); continue

            energies: List[float] = []
            thresholds: List[float] = []

            for i in range(n - 1):
                a, b = ps[i], ps[i + 1]
                t = self.cfg.get_threshold(iteration)

                # Skip: atomic mode with multi-char particles, or punctuation
                if (mode == 'atomic' and (len(a) > 1 or len(b) > 1)) \
                        or self._is_punct(a) or self._is_punct(b):
                    energies.append(-float('inf')); thresholds.append(t); continue

                # Atom personality adjustment
                for p in (a, b):
                    if len(p) == 1:
                        t += self.cfg.get_atom_adjust(
                            self.atom_per.get(p, 1.0),
                            self.freq_1.get(p, 0), self.total_particles)

                # Polarity: lower threshold if merge direction matches preference
                pa1, pa2 = self._polarity(a), self._polarity(b)
                if pa1 > 0.3 or pa2 > 0.3:
                    rd = self.right_n.get(a, {})
                    ld = self.left_n.get(b, {})
                    ds = max(rd.get(b, 0) / max(self.freq_1.get(a, 1), 1),
                             ld.get(a, 0) / max(self.freq_1.get(b, 1), 1))
                    t -= self.cfg.polarity_weight * ds * (pa1 + pa2)

                thresholds.append(t)

                # Net energy
                npmi = self._npmi(a, b)
                c1, c2 = max(self.freq_1.get(a, 1), 1), max(self.freq_1.get(b, 1), 1)
                hr = self._entropy(self.right_n.get(a, {}), c1)
                hl = self._entropy(self.left_n.get(b, {}), c2)
                d1, d2 = math.log2(c1 + 2), math.log2(c2 + 2)
                ion = ((hr / d1 if d1 > 0 else 0) + (hl / d2 if d2 > 0 else 0)) / 2
                net = npmi - ion * self.cfg.get_mass_factor(len(a), len(b))
                energies.append(net)

            # Local maximum detection: only merge pairs that are locally strongest
            merge: List[int] = []
            for i in range(len(energies)):
                e = energies[i]; t = thresholds[i]
                ep = energies[i - 1] if i > 0 else -float('inf')
                en = energies[i + 1] if i < len(energies) - 1 else -float('inf')
                if e > t and e >= ep and e >= en: merge.append(i)

            if merge:
                changed = True
                np_: List[str] = []; idx = 0; ms = set(merge)
                while idx < n:
                    if idx in ms and idx < n - 1:
                        np_.append(ps[idx] + ps[idx + 1]); idx += 2
                    else:
                        np_.append(ps[idx]); idx += 1
                new_sents.append(np_)
            else:
                new_sents.append(ps)

        self.sentences = new_sents
        self.iteration += 1
        return changed

    # ────────── Dissolution (anti-entropy) ──────────

    def dissolve(self) -> bool:
        """Reverse entropy: split particles whose sub-parts appear independently.

        For each particle of length >= 3, find the internal split point
        where the two halves have the highest independent frequency.
        If the independence score exceeds threshold, split.
        """
        cfg = self.cfg
        if not cfg.dissolution_enabled or self.iteration < cfg.dissolution_start_round:
            return False
        changed = False
        new_sents: List[List[str]] = []
        for ps in self.sentences:
            np_: List[str] = []
            for p in ps:
                if len(p) < cfg.dissolution_min_length:
                    np_.append(p); continue
                best_k, best_sc = -1, 0.0
                for k in range(1, len(p)):
                    a, b = p[:k], p[k:]
                    fa, fb = self.freq_1.get(a, 0), self.freq_1.get(b, 0)
                    sc = max(fa / max(self.freq_1.get(p, 1), 1),
                             fb / max(self.freq_1.get(p, 1), 1))
                    if fa > 0 and fb > 0 and sc > best_sc:
                        best_sc = sc; best_k = k
                if best_k > 0 and best_sc > cfg.dissolution_independence_ratio:
                    changed = True
                    np_.append(p[:best_k]); np_.append(p[best_k:])
                else:
                    np_.append(p)
            new_sents.append(np_)
        self.sentences = new_sents
        return changed

    # ────────── Main loop ──────────

    def run(self, max_iters: Optional[int] = None,
            quiet: bool = False) -> List[List[str]]:
        """Run the full growth process until convergence.

        Alternates between 'all' and 'atomic' modes,
        runs dissolution every cfg.dissolution_interval rounds,
        stops when both particle count and order parameter stabilize.

        Args:
            max_iters: maximum iterations (default: cfg.max_iterations)
            quiet: suppress progress output

        Returns:
            segmented sentences
        """
        max_iters = max_iters or self.cfg.max_iterations
        init_n = sum(len(s) for s in self.sentences)
        if not quiet:
            print(f"Crystal growth | {len(self.sentences)} sentences | "
                  f"{init_n} initial particles")
        ps_ok = os_ok = 0
        prev_p = init_n; prev_o = 0.0
        round_data = []

        for i in range(max_iters):
            mode = 'all' if i % 2 == 0 else 'atomic'
            changed = self.grow(mode, i)
            total_p = sum(len(s) for s in self.sentences)
            curr_o = self.order_parameter

            dissolved = False
            if self.cfg.dissolution_enabled and i > 0 \
                    and i % self.cfg.dissolution_interval == 0:
                dissolved = self.dissolve()
                if dissolved: total_p = sum(len(s) for s in self.sentences)

            ps_ok = ps_ok + 1 if (prev_p - total_p == 0
                                  and not changed and not dissolved) else 0
            os_ok = os_ok + 1 if abs(curr_o - prev_o) < self.cfg.convergence_tol else 0

            if not quiet:
                arrow = '↓' if changed or dissolved else '→'
                label = 'global' if mode == 'all' else 'atomic'
                round_data.append([i+1, label, total_p, f"{curr_o:.4f}", arrow])

            prev_p = total_p; prev_o = curr_o

            if ps_ok >= self.cfg.particle_plateau_tol \
                    and os_ok >= self.cfg.convergence_window:
                if not quiet:
                    round_data.append([i+1, 'CONVERGED', '-', '-', '✓'])
                break

        final_n = sum(len(s) for s in self.sentences)
        self.stats = {'initial': init_n, 'final': final_n,
                      'reduction': init_n - final_n,
                      'reduction_pct': (init_n - final_n) / max(init_n, 1) * 100,
                      'rounds': self.iteration,
                      'order': round(self.order_parameter, 4)}
        if not quiet:
            print(_make_table(round_data,
                              headers=['R', 'Mode', 'Particles', 'Order', 'Δ'],
                              colalign=('center', 'center', 'center', 'center', 'center')))
            stats_table = [[init_n, final_n, self.stats['reduction'],
                            f"{self.stats['reduction_pct']:.1f}%",
                            self.iteration, round(self.order_parameter, 4)]]
            print(_make_table(stats_table,
                              headers=['Initial', 'Final', 'Reduction', 'Pct', 'Rounds', 'Order'],
                              colalign=('center', 'center', 'center', 'center', 'center', 'center')))
        return self.sentences


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
        try: sys.stdout.reconfigure(encoding='utf-8')
        except: pass
    text = open('corpus/example.txt', encoding='utf-8').read()
    g = AtomicCrystalGrowth(text)
    r = g.run()
    print(f"\nSample segmentation (first 15 sentences):")
    tbl = [[str(i+1),
            ' / '.join(s[:15]) + (' …' if len(s) > 15 else '')]
           for i, s in enumerate(r[:15])]
    print(_make_table(tbl, headers=['#', 'Segmentation'],
                      colalign=('center', 'left')))

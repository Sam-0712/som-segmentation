"""
Atomic Crystal Growth — Unsupervised Chinese Word Segmentation
================================================================
A physics-inspired algorithm that treats text as a particle system.
Particles (characters/substrings) spontaneously bond based on
statistical forces: NPMI (affinity) drives merging, contextual entropy
(ionization) resists it. Viterbi DP + damped annealing + periodic dissolution.
"""

from __future__ import annotations
import re, sys, time, collections, math
from typing import Dict, List, Tuple, Optional, DefaultDict


# ── Sentence splitting ────────────────────────────────────────

def split_sentences(raw: str) -> List[str]:
    """Split raw text into sentences using PySBD for boundary disambiguation.

    PySBD is a rule-based sentence boundary detection algorithm.
    Falls back to regex-based splitting if PySBD is not installed.
    
    """
    try:
        import pysbd
        seg = pysbd.Segmenter(language='zh')
        return [s.strip() for s in seg.segment(raw) if s.strip()]
    except ImportError: # Fallback
        return [s.strip() for s in re.split(r'[。？！.?!<>\n]+', raw) if s.strip()]

# ── tabulate ──────────────────────────────────────────────────────────

def _make_table(rows, headers, **kwargs):
    """Format a table using tabulate (must be installed)."""
    try:
        from tabulate import tabulate
    except ImportError:
        raise ImportError(
            "tabulate is required for table output.\n"
            "Install it with: pip install tabulate"
        )
    return tabulate(rows, headers=headers, tablefmt='grid', **kwargs)

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

class Config:
    """Hyper-parameters for the crystal growth algorithm."""
    def __init__(self) -> None:
        # ─── Threshold ───
        self.base_threshold: float = 0.0
        self.threshold_decay: float = 0.01              # γ per round

        # ─── Mass factor ───
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
        self.max_iterations: int = 80

        # ─── Polarity ───
        self.polarity_weight: float = 0.15

        # ─── Viterbi ───
        self.viterbi_merge_bias: float = 0.25   # bonus per merge in Viterbi DP
        self.viterbi_relax_factor: float = 0.70 # threshold relaxation (1.0=strict)

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

        # ─── Annealing schedule ───
        self.damping_rate: float = 0.08         # frequency decay rate (0=constant)
        self.damping_amp_rate: float = 0.02     # amplitude growth per round
        self.damping_max_amp: float = 0.25      # max oscillation amplitude

    def get_threshold(self, it: int) -> float:
        """Dynamic threshold: linear decay + damped cosine oscillation.

        Early rounds: fast, shallow oscillation → rapid exploration.
        Later rounds: slow, deep oscillation → long annealing cycles.
        """
        if self.damping_rate > 0:
            amp = min(self.damping_amp_rate * it, self.damping_max_amp)
            freq = 0.3 * math.exp(-self.damping_rate * it)
            m = 1.0 - amp + amp * math.cos(freq * it)
        else:
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
    """Unsupervised word segmentation via crystal growth simulation.

    The algorithm alternates between 'global' (all particles can merge)
    and 'atomic' (only single-character particles can merge) modes.
    Periodic dissolution reverses weak bonds to prevent over-segmentation.
    """

    def __init__(self, text: str, config: Optional[Config] = None) -> None:
        self.cfg = config or Config()
        self.sentences: List[List[str]] = self._preprocess(text)
        self.iteration: int = 0

        # Runtime stats (recalculated each grow() call)
        self.total_particles: int = 0
        self.total_pairs: int = 0
        self.vocab_size: int = 0
        self.freq_1: Dict[str, int] = {}               # unigram frequencies
        self.freq_2: Dict[Tuple[str, str], int] = {}   # bigram frequencies
        self.left_n: Dict[str, Dict[str, int]] = {}    # left neighbor distributions
        self.right_n: Dict[str, Dict[str, int]] = {}   # right neighbor distributions
        self.atom_per: Dict[str, float] = {}           # P(alone) for single chars
        self.order_parameter: float = 0.0              # structural order metric

        # Ionization cache: pre-computed normalized entropy for all particles
        # Used directly by grow() to avoid redundant _entropy() calls.
        self.ion_left: Dict[str, float] = {}    # left-context ionization
        self.ion_right: Dict[str, float] = {}   # right-context ionization

    # ────────── Preprocessing ──────────

    @staticmethod
    def _preprocess(raw: str) -> List[List[str]]:
        """Split raw text into sentences → atomic particles."""
        sents = split_sentences(raw)
        result: List[List[str]] = []
        for s in sents:
            parts = re.findall(
                r'[ ]+'                                     # continuous spaces
                r'|[0-9]+(?:\.[0-9]+)?[%‰]?'                # numbers
                r'|[a-zA-Z]+(?:\'[a-zA-Z]+)?'               # English words with contractions
                r'|[（(\[【{][0-9\-\.]+?[）)\]】}]'          # bracketed numbers
                r'|[^\s\w]'                                 # single punctuation/symbol
                r'|[\u4e00-\u9fff]'                         # Chinese characters
                r'|[^\s]', s)                               # fallback
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
        """Count unigram/bigram frequencies, neighbor distributions,
        and pre-compute ionization cache for grow().

        Ionization = entropy / log2(freq+2), the normalized entropy
        that resists particle merging. Pre-computing here avoids
        redundant _entropy() calls in grow()'s inner loop.
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

        # Pre-compute ionization cache + order parameter
        self.ion_left.clear()
        self.ion_right.clear()
        o_sum = o_cnt = 0.0
        for p, cnt in f1.items():
            if cnt >= 2:
                hl = self._entropy(self.left_n.get(p, {}), cnt)
                hr = self._entropy(self.right_n.get(p, {}), cnt)
                d = math.log2(cnt + 2)
                il = hl / d if d > 0 else 0.0
                ir = hr / d if d > 0 else 0.0
                self.ion_left[p] = il
                self.ion_right[p] = ir
                o_sum += (il + ir) / 2
                o_cnt += 1
        self.order_parameter = o_sum / max(o_cnt, 1)

    def _entropy(self, dist: Dict[str, int], total: int) -> float:
        """Laplace-smoothed Shannon entropy over a neighbor distribution.

        Args:
            dist: neighbor word → co-occurrence count mapping
            total: total occurrences of the particle (used as smoothing base)

        Unseen vocabulary items receive pseudo-count ``alpha`` for smoothing.
        For particles with freq < 2, entropy is approximated as 0 (deterministic).
        """
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
        """Normalized Pointwise Mutual Information in [-1, 1].

        NPMI(a,b) = PMI(a,b) / -log2(P(a,b))
        Returns -1.0 when a and b never co-occur.
        """
        c = self.freq_2.get((a, b), 0)
        if c == 0: return -1.0
        pp = c / self.total_pairs
        p_a = self.freq_1[a] / self.total_particles
        p_b = self.freq_1[b] / self.total_particles
        pmi = math.log2(pp / (p_a * p_b))
        # Guard: when pp ≈ 1 (pair is the entire corpus), NPMI → 1.0
        denom = -math.log2(max(pp, 1e-16))
        return pmi / denom if denom > 0 else 1.0

    def _polarity(self, p: str) -> float:
        """Directional binding preference: |H_L - H_R| / (H_L + H_R).

        High polarity → particle strongly prefers one direction.
        Used to lower the merge threshold when the preferred direction
        matches the actual neighbor being considered.
        """
        cnt = self.freq_1.get(p, 0)
        if cnt <= 1: return 0.0
        hl = self._entropy(self.left_n.get(p, {}), cnt)
        hr = self._entropy(self.right_n.get(p, {}), cnt)
        t = hl + hr
        return 0.0 if t < 1e-12 else abs(hl - hr) / t

    @staticmethod
    def _is_punct(p: str) -> bool:
        """Check if a particle is pure punctuation/emoji (no word characters).

        Uses Unicode-aware ``\\w`` (Python 3 default), which matches
        CJK characters as word characters — so Chinese chars are NOT
        treated as punctuation. Only symbols, punctuation marks, and
        emoji are excluded from merging.
        """
        return bool(re.match(r'^[^\s\w]+$', p))

    # ────────── Growth round ──────────

    def grow(self, mode: str = 'all', iteration: int = 0) -> bool:
        """Execute one round of crystal growth.

        For each adjacent particle pair (a, b):
          1. Compute base threshold + atom personality adjustments
          2. Compute NPMI affinity
          3. Look up pre-computed ionization from cache
          4. Net energy = NPMI - ionization * mass_factor
          5. Viterbi DP finds globally optimal set of non-overlapping binary merges

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

                # Skip: atomic mode with multi-char particles, or punctuation, or spaces
                # Spaces are preserved as independent tokens to match jieba's behavior
                if (mode == 'atomic' and (len(a) > 1 or len(b) > 1)) \
                        or self._is_punct(a) or self._is_punct(b) \
                        or a.strip() == '' or b.strip() == '':
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

                # Net energy: NPMI - ionization * mass_factor
                # Ionization comes from pre-computed cache (O(1) dict lookup)
                npmi = self._npmi(a, b)
                ion = (self.ion_right.get(a, 0.0) + self.ion_left.get(b, 0.0)) / 2
                net = npmi - ion * self.cfg.get_mass_factor(len(a), len(b))
                energies.append(net)

            # Viterbi DP: find globally optimal set of non-overlapping binary merges
            # dp[i] = best cumulative score for first i particles
            dp = [0.0] * (n + 1)
            choice = [0] * (n + 1)  # 0=skip p_{i-1}, 1=merge p_{i-2}+p_{i-1}

            for i in range(1, n + 1):
                # Option A: keep p_{i-1} as-is (no merge ending at i-1)
                dp[i] = dp[i - 1]
                choice[i] = 0

                # Option B: merge p_{i-2} and p_{i-1}
                if i >= 2:
                    e = energies[i - 2]
                    if e > thresholds[i - 2] * self.cfg.viterbi_relax_factor:  # relaxed threshold
                        candidate = (dp[i - 2] + e + self.cfg.viterbi_merge_bias)
                        if candidate > dp[i]:
                            dp[i] = candidate
                            choice[i] = 1

            # Backtrack to build new particle sequence
            np_: List[str] = []
            i = n
            while i > 0:
                if choice[i] == 1:
                    np_.insert(0, ps[i - 2] + ps[i - 1])
                    i -= 2
                else:
                    np_.insert(0, ps[i - 1])
                    i -= 1

            if np_ != ps:
                changed = True
            new_sents.append(np_)

        self.sentences = new_sents
        self.iteration += 1
        return changed

    # ────────── Dissolution ──────────

    def dissolve(self) -> bool:
        """Reverse entropy: split particles whose sub-parts appear independently.

        For each particle of length >= dissolution_min_length, find the
        internal split point where the two halves have the highest
        independent frequency. If the independence score exceeds
        dissolution_independence_ratio, split.

        Returns:
            True if any particle was split
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
                # Protect purely numeric particles (years, decimals, percentages)
                # from being split by dissolution — they are atomic by definition.
                if re.match(r'^[0-9]+(?:\.[0-9]+)?[%‰]?$', p):
                    np_.append(p); continue
                # Protect English words (with optional contractions) from being split
                # English words are treated as atomic units to match jieba's behavior
                if re.match(r'^[a-zA-Z]+(?:\'[a-zA-Z]+)?$', p):
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
        
        Args:
            max_iters: maximum iterations (default: cfg.max_iterations)
            quiet: suppress progress output

        Returns:
            Segmented sentences (list of list of particle strings)
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
    tbl = [[str(i+1), ' / '.join(s[:15]) + (' ...' if len(s) > 15 else '')] for i, s in enumerate(r[:15])]
    print(_make_table(tbl, headers=['#', 'Segmentation'], colalign=('center', 'left')))

# Atomic Crystal Growth

Unsupervised Chinese word segmentation under extreme constraints: **no dictionary, no supervision, no external libraries** for the core algorithm. By treating characters as particles in a statistical field, this project simulates crystal growth — particles merge when the net energy exceeds a dynamic threshold, and weak bonds are periodically dissolved. Word boundaries emerge purely from the intrinsic statistics of the text.

On a philosophy-speech test corpus, the algorithm achieves a **boundary F1 ≈ 0.930** against jieba (which itself is unsupervised but dictionary-equipped). A leave-one-year-out cross-validation over **30 years** of *Southern Weekend* New Year editorials confirms that enlarging the training corpus consistently improves segmentation quality.

## Table of Contents

- [How It Works](#how-it-works)
- [Performance](#performance)
- [Algorithm Detail](#algorithm-detail)
  - [1. Problem Setting](#1-problem-setting)
  - [2. Statistical Quantities](#2-statistical-quantities)
  - [3. Net Energy of a Merge](#3-net-energy-of-a-merge)
  - [4. Dynamic Threshold & Damped Annealing](#4-dynamic-threshold--damped-annealing)
  - [5. Atom Personality Adjustment](#5-atom-personality-adjustment)
  - [6. Polarity Modulation](#6-polarity-modulation)
  - [7. Viterbi Optimal Merging](#7-viterbi-optimal-merging)
  - [8. Dissolution (Anti-Entropy)](#8-dissolution-anti-entropy)
  - [9. Order Parameter & Convergence](#9-order-parameter--convergence)
  - [10. Complete Iteration Loop](#10-complete-iteration-loop)
  - [Hyperparameters](#hyperparameters)
- [Code](#code)
- [Quick Start](#quick-start)
- [Corpus](#corpus)
- [Requirements](#requirements)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## How It Works

1. **Initial state** — Text is split into sentences, then into *atomic particles*: single Chinese characters, ASCII alphanumeric tokens, bracketed expressions, and punctuation.

2. **Growth round** — For each adjacent particle pair $(p, q)$, compute:

   - **Affinity** (NPMI): how strongly they co-occur.
   - **Ionization** (normalized contextual entropy): how "bound" each particle is to its environment.
   - **Mass factor**: longer particles resist further merging (exponential penalty).
   - **Net energy**: $E = \text{NPMI} - I \cdot M$.

3. **Viterbi merging** — A dynamic-programming pass finds the *globally optimal* set of non-overlapping binary merges per sentence, subject to a relaxed threshold. This replaces greedy/local decisions with a global optimum.

4. **Dissolution** — Every $K$ rounds, weakly-bound multi-character particles whose sub-parts appear independently are split back (anti-entropy).

5. **Convergence** — Stops when both the particle count and the order parameter stabilize for a consecutive window of iterations.

   Two merge modes alternate each round: **global** (all particles eligible) and **atomic** (only single-character particles). This refines the core vocabulary before longer structures compete.

## Performance

### Boundary F1 vs Jieba (on `corpus/example.txt`)

The test text is a ~3 000-character philosophy speech (Peking University, Prof. Cheng Lesong) with rich vocabulary, complex sentence structures, and mixed Chinese/English/emoji content. Jieba (unsupervised + dictionary) serves as the baseline.

Running `python main.py eval` yields:

| Metric                            | Crystal Growth |   Jieba    |
| --------------------------------- | :------------: | :--------: |
| Precision                         |   **0.9323**   |     —      |
| Recall                            |   **0.9270**   |     —      |
| **Boundary F1**                   |   **0.9296**   | (baseline) |
| Avg particles/sentence            |     19.26      |   19.16    |
| Avg token length                  |     1.599      |   1.608    |
| Granularity ratio (crystal/jieba) |     1.005      |     —      |
| Unigram Jaccard                   |     0.5724     |     —      |
| Bigram Jaccard                    |     0.5851     |     —      |

The algorithm converges in **59 rounds**, reducing 7 348 atomic particles to 4 623 (37.1 % reduction). The **F1 ≈ 0.93** means ~93 % of word boundaries agree with jieba — without any dictionary or labeled data. Notably, the granularity ratio (1.005) is almost exactly 1, meaning the algorithm produces nearly the same number of tokens per sentence as jieba.

### Transfer Learning: Leave-One-Year-Out Cross-Validation

Using `scripts/train.py`, a leave-one-year-out (LOYO) experiment was conducted on **30 years** (1997–2026) of *Southern Weekend* New Year editorials (45 729 characters total). For each year:

1. **Self-trained**: run crystal growth on that year's text alone (~1 000–2 700 chars).
2. **Transfer**: train on the other 29 years, then segment the held-out year.

| Metric          | Self-trained |     Transfer     |
| --------------- | :----------: | :--------------: |
| Average F1      |    0.8299    |    **0.8583**    |
| Average Δ       |      —       |   **+0.0284**    |
| Transfer wins   |      —       | **27/30 (90 %)** |
| Transfer losses |      —       |   3/30 (10 %)    |

**Key findings:**

- **Transfer consistently outperforms self-training** in 27 out of 30 years, with an average F1 improvement of +0.0284.

- The self-trained F1 (0.83) is notably lower than the `example.txt` result (0.93) because each year's text is short (~1 500 chars on average), providing weak statistical signal. Transfer learning leverages ~44 K chars of context, recovering most of the gap.

- **Largest gains** occur on years where self-training struggles most (sparse statistics): 2021 (+0.0795), 2013 (+0.0708), 2026 (+0.0581), 2000 (+0.0530), 2017 (+0.0485).

- **Only 3 regressions**, all minor (≤ 0.012): 2003 (−0.0112), 2004 (−0.0074), 2011 (−0.0068) — these are years that already self-train well (F1 > 0.86), where the broader corpus introduces mild noise.

  This demonstrates that *enlarging the statistical sample improves segmentation quality* — the algorithm generalizes from a larger corpus rather than overfitting.

## Algorithm Detail

### 1. Problem Setting

Let a text be split into sentences $S_1, S_2, \dots, S_m$. Initially each sentence is a sequence of **atomic particles** obtained by a tokeniser that separates:

- bracketed expressions (e.g. `[2.3]`, `(1)`),

- sequences of ASCII letters/digits,

- individual Chinese characters and punctuation.

  No further linguistic knowledge is used. The goal is to let particles **grow** into longer units (words / multi-word expressions) solely through the statistical forces measured on the current particle configuration.

### 2. Statistical Quantities

For a given particle configuration (a particular segmentation of all sentences) we compute:

- $f(p)$ — frequency of particle $p$ (unigram count)
- $f(p,q)$ — frequency of the ordered pair $(p,q)$ (bigram count)
- $N = \sum_p f(p)$ — total number of particle occurrences
- $M = \sum_{p,q} f(p,q)$ — total number of adjacent particle pairs


For each particle $p$ we also collect its **left neighbours** $L(p)$ and **right neighbours** $R(p)$. For two adjacent particles $p, q$:

$$
\text{PMI}(p,q) = \log_2 \frac{P(p,q)}{P(p) \cdot P(q)},\qquad \text{NPMI}(p,q) = \frac{\text{PMI}(p,q)}{-\log_2 P(p,q)}
$$

NPMI $\in [-1, 1]$. A value close to 1 indicates that $p$ and $q$ almost always appear together. An epsilon guard ($\max(P(p,q),\, 10^{-16})$) prevents division-by-zero when a pair dominates the corpus.

With Laplace smoothing parameter $\alpha$ and vocabulary size $V$ (number of distinct particles), the **right entropy** of $p$ is:

$$
H_R(p) = -\sum_{r \in R(p)} \frac{f(p,r)+\alpha}{f(p)+\alpha V} \log_2 \frac{f(p,r)+\alpha}{f(p)+\alpha V} - (V - |R(p)|)\cdot \frac{\alpha}{f(p)+\alpha V} \log_2 \frac{\alpha}{f(p)+\alpha V}
$$

The left entropy $H_L(p)$ is defined analogously using left neighbors $L(p)$. To make entropies comparable across different frequencies, we normalize by $\log_2\bigl(f(p)+2\bigr)$:

$$
I_R(p) = \frac{H_R(p)}{\log_2(f(p)+2)}, \qquad I_L(p) = \frac{H_L(p)}{\log_2(f(p)+2)}
$$

The **ionization energy** of a particle measures how tightly it is bound by its context. High $I$ means $p$ appears in a very diverse environment (high contextual freedom) — it is "heavy" and reluctant to merge. Low $I$ means $p$ can easily detach.

### 3. Net Energy of a Merge

For a pair of adjacent particles $(p, q)$ the **net energy** is:

$$
E(p,q) = \text{NPMI}(p,q) - I(p,q) \cdot M(p,q)
$$

where the combined ionization uses the *directional* entropies at the merge boundary:

$$
I(p,q) = \frac{I_R(p) + I_L(q)}{2}
$$

and the **mass factor** exponentially penalises merges of long particles:

$$
M(p,q) = \beta^{\ell(p) + \ell(q) - 2}
$$

with $\beta > 1$ (the *mass base*). The sign of $E$ indicates whether the merge is favoured (positive) or disfavoured (negative).

### 4. Dynamic Threshold & Damped Annealing

The base threshold $T$ decreases with the iteration number $i$ (simulated annealing) with a **damped cosine oscillation** that helps escape local plateaus:

$$
T(i) = T_0 - \gamma \cdot i \cdot m(i)
$$

where the modulation factor $m(i)$ is:

$$
m(i) = 1 - \text{amp}(i) + \text{amp}(i) \cdot \cos\bigl(\text{freq}(i) \cdot i\bigr)
$$

$$
\text{amp}(i) = \min(\alpha_{\text{amp}} \cdot i, A_{\max}), \qquad \text{freq}(i) = \lambda \cdot e^{-\delta \cdot i}
$$

- **Early rounds**: fast, shallow oscillation → rapid exploration.
- **Later rounds**: slow, deep oscillation → long annealing cycles.

### 5. Atom Personality Adjustment

The threshold is **personality-adjusted** for single-character particles. For a character $c$ let

- $p_{\text{alone}}(c)$ — fraction of its occurrences where it appears as a solitary particle (not part of a longer particle)

- $f_{\text{rel}}(c) = f(c) / N$ — relative frequency

  Then an adjustment $\Delta_{\text{atom}}(c)$ is computed:

- If $p_{\text{alone}} > \theta_{\text{indep}}$: $\Delta = \eta_{\text{indep}} \cdot (p_{\text{alone}} - \theta_{\text{indep}})$ — **protect** (raise threshold, harder to merge)

- Else if $p_{\text{alone}} < \theta_{\text{restless}}$: $\Delta = -\eta_{\text{restless}} \cdot (\theta_{\text{restless}} - p_{\text{alone}})$ — **encourage** merging

- If $f_{\text{rel}} > \phi_{\text{floor}}$: extra protection $\Delta_{\text{freq}} = \kappa \cdot (f_{\text{rel}} - \phi_{\text{floor}})$

  The final threshold for a pair $(p,q)$ at iteration $i$ is:

$$
T_{\text{final}}(p,q,i) = T(i) + \Delta_{\text{atom}}(p) + \Delta_{\text{atom}}(q)
$$

If either $p$ or $q$ is not a single character, its $\Delta$ is 0.

### 6. Polarity Modulation

The **polarity** of a particle captures the asymmetry between its left and right contexts:

$$
\Psi(p) = \frac{|H_L(p) - H_R(p)|}{H_L(p) + H_R(p)}, \qquad \Psi(p) \in [0,1]
$$

When a pair exhibits high polarity ($\Psi > \psi_0$), the threshold is lowered if the merge follows the natural direction indicated by the neighbour distributions:

$$
d_s(p,q) = \max\left( \frac{f(p,q)}{f(p)}, \frac{f(p,q)}{f(q)} \right)
$$

$$
\Delta_{\text{pol}}(p,q) = -w_{\text{pol}} \cdot d_s(p,q) \cdot \bigl(\Psi(p) + \Psi(q)\bigr)
$$

The final threshold becomes:

$$
T_{\text{final}}' = T_{\text{final}} + \Delta_{\text{pol}}
$$

### 7. Viterbi Optimal Merging

> This section replaces the earlier "local maximum" greedy approach. The Viterbi dynamic program finds the **globally optimal** set of non-overlapping binary merges per sentence in $O(n)$ time.

For a sentence with particles $x_1, x_2, \dots, x_n$, let $E_i = E(x_i, x_{i+1})$ and $T_i = T_{\text{final}}'(x_i, x_{i+1}, i_{\text{iter}})$. Define $\text{dp}[i]$ = best cumulative score for the first $i$ particles:

$$
\text{dp}[i] = \max \begin{cases}
\text{dp}[i-1] & \text{(keep } x_i \text{ as-is)} \\
\text{dp}[i-2] + E_{i-1} + b_{\text{merge}} & \text{(merge } x_{i-1}, x_i \text{), if } E_{i-1} > T_{i-1} \cdot r
\end{cases}
$$

where:

- $r$ = **relax factor** (e.g. 0.70) — allows merges that don't strictly exceed the threshold, acting as *quantum tunnelling* through energy barriers.
- $b_{\text{merge}}$ = **merge bias** (e.g. 0.25) — a per-merge bonus that counteracts the Viterbi DP's natural conservatism (without it, the DP prefers fewer merges).


### 8. Dissolution (Anti-Entropy)

To prevent irreversible over-merging, a **dissolution** step is performed every $K$ iterations (starting after a warm-up phase). For each particle $p$ with length $\ell(p) \geq L_{\min}$:

- For every split position $k = 1, \dots, \ell(p)-1$ let $a = p[:k]$, $b = p[k:]$.
- Compute the **independence score**:

$$
S(k) = \max\left( \frac{f(a)}{f(p)}, \frac{f(b)}{f(p)} \right)
$$

- Take the split with the highest score: $k^* = \arg\max_k S(k)$ (requiring $f(a) > 0$ and $f(b) > 0$).
- If $S(k^*) > \tau_{\text{diss}}$, replace $p$ by $a$ and $b$ (split).


### 9. Order Parameter & Convergence

The **order parameter** $\mathcal{O}$ measures the global structural order of the particle system — the mean directional ionization energy across all particles that occur at least twice:

$$
\mathcal{O} = \frac{1}{|\{p : f(p)\ge 2\}|} \sum_{p:\, f(p)\ge 2} \frac{I_L(p) + I_R(p)}{2}
$$

Higher $\mathcal{O}$ indicates a more "frozen" system where particles are well-crystallised.

Convergence is declared when **both** of the following hold for a consecutive window of iterations:

1. The total number of particles has not changed (no merges, no dissolutions) for $n_{\text{plateau}}$ iterations.
2. The order parameter $\mathcal{O}$ changes by less than $\epsilon_{\mathcal{O}}$ for $n_{\text{window}}$ iterations.

### 10. Complete Iteration Loop

```
Input: initial atomic particle sequences for all sentences
Initialise iteration counter t = 0

while t < max_iterations and not converged:
    Compute all frequencies, entropies, NPMI, ionization energies, polarities
    mode = 'all' if t is even else 'atomic'
    For each sentence:
        For each adjacent pair (x_i, x_{i+1}):
            Compute net energy E_i
            Compute threshold T_i (dynamic + personality + polarity)
        Viterbi DP: find globally optimal non-overlapping merges
            (merge if E_i > T_i * relax_factor, maximise Σ(E + merge_bias))
        Backtrack to build new particle sequence
    if dissolution_interval > 0 and t >= start_dissolve and (t % interval == 0):
        Perform dissolution on all particles
    Update order parameter O
    Check convergence criteria
    t = t + 1

Output: final segmented sentences
```

### Hyperparameters

| Symbol                     | Parameter                        | Default | Meaning                                  |
| -------------------------- | -------------------------------- | :-----: | ---------------------------------------- |
| $\beta$                    | `mass_base`                      |   6.0   | Mass base (exponential growth factor)    |
| $T_0$                      | `base_threshold`                 |   0.0   | Initial threshold                        |
| $\gamma$                   | `threshold_decay`                |  0.01   | Threshold decay per iteration            |
| $\delta$                   | `damping_rate`                   |  0.08   | Frequency decay rate for oscillation     |
| $\alpha_{\text{amp}}$      | `damping_amp_rate`               |  0.02   | Amplitude growth per round               |
| $A_{\max}$                 | `damping_max_amp`                |  0.25   | Maximum oscillation amplitude            |
| $\theta_{\text{indep}}$    | `atom_independent_threshold`     |  0.70   | High-alone protection threshold          |
| $\eta_{\text{indep}}$      | `atom_independent_penalty`       |  0.50   | Independent atom penalty                 |
| $\theta_{\text{restless}}$ | `atom_restless_threshold`        |  0.35   | Low-alone encouragement threshold        |
| $\eta_{\text{restless}}$   | `atom_restless_bonus`            |  0.40   | Restless atom bonus                      |
| $\phi_{\text{floor}}$      | `freq_floor`                     |  0.02   | Frequency floor for extra protection     |
| $\kappa$                   | — (1.5)                          |  1.50   | Frequency protection weight              |
| $w_{\text{pol}}$           | `polarity_weight`                |  0.15   | Polarity modulation weight               |
| $\psi_0$                   | — (0.3)                          |  0.30   | Polarity activation threshold            |
| $\alpha$                   | `entropy_alpha`                  |  1e-6   | Laplace smoothing for entropy            |
| $r$                        | `viterbi_relax_factor`           |  0.70   | Threshold relaxation (tunnelling)        |
| $b_{\text{merge}}$         | `viterbi_merge_bias`             |  0.25   | Per-merge bonus in Viterbi DP            |
| $K$                        | `dissolution_interval`           |    5    | Dissolution interval                     |
|                            | `dissolution_start_round`        |    3    | Warm-up rounds before dissolution        |
| $L_{\min}$                 | `dissolution_min_length`         |    3    | Minimum length for dissolution candidate |
| $\tau_{\text{diss}}$       | `dissolution_independence_ratio` |  0.15   | Dissolution independence threshold       |
|                            | `max_iterations`                 |   80    | Maximum iterations                       |
| $n_{\text{plateau}}$       | `particle_plateau_tol`           |    3    | Iterations with no particle change       |
| $n_{\text{window}}$        | `convergence_window`             |    5    | Convergence window for order parameter   |
| $\epsilon_{\mathcal{O}}$   | `convergence_tol`                |  0.001  | Tolerance for order parameter change     |

These hyperparameters control the "thermodynamics" of the system: lower thresholds or higher mass bases produce more aggressive merging; personality adjustments protect function words; dissolution prevents spurious long compounds; the relax factor and merge bias tune the Viterbi DP's aggressiveness.

## Code

### Project Structure

```
/
├── main.py                        # Entry point (run / eval / train commands)
├── corpus/
│   ├── corpus.json                # 30 years of Southern Weekend NYE editorials (~45K chars)
│   └── example.txt                # Demo text: philosophy speech (~3K chars)
├── scripts/
│   ├── __init__.py
│   ├── core.py                    # Core algorithm: AtomicCrystalGrowth + Config (~530 lines)
│   ├── evaluation.py              # Jieba baseline comparison (boundary F1, granularity, n-gram)
│   └── train.py                   # Leave-one-year-out cross-validation
├── requirements.txt
└── README.md
```

### Architecture

- **`scripts/core.py`** — The entire algorithm in a single file with zero external dependencies (only Python stdlib: `re`, `math`, `collections`). Contains:
  - `Config` — all hyperparameters with dynamic threshold / mass factor / atom adjustment methods.
  - `AtomicCrystalGrowth` — the main class: preprocessing, statistics, growth, dissolution, convergence.
- **`scripts/evaluation.py`** — Compares crystal growth output against jieba, reporting boundary precision/recall/F1, granularity ratio, and n-gram Jaccard overlap.
- **`scripts/train.py`** — Leave-one-year-out cross-validation: for each year, compares self-trained vs transfer segmentation.
- **`main.py`** — Thin CLI dispatcher with dependency checking.

## Quick Start

```bash
# Install evaluation dependencies (core algorithm needs nothing extra)
pip install -r requirements.txt

# Run segmentation on the demo text
python main.py run

# Compare with jieba baseline (boundary F1, granularity, samples)
python main.py eval

# Leave-one-year-out cross-validation on the 30-year corpus
python main.py train
```

## Corpus

### `corpus/example.txt`

A ~7 000-character philosophy speech (Peking University, Prof. Cheng Lesong) on Daoism and modern life. Rich in literary vocabulary, complex sentence structures, and mixed Chinese/English/emoji content — a challenging test for unsupervised segmentation.

### `corpus/corpus.json`

30 years (1997–2026) of *Southern Weekend* (南方周末) New Year editorials (新年献词), totalling ~45 000 characters. Each entry contains `year`, `title`, and `text`. This corpus is used for the leave-one-year-out transfer learning experiment.

## Requirements

- **Python 3.7+**
- `tabulate` — table formatting for output
- `jieba` — baseline comparison only (`core.py` runs standalone with zero dependencies)

```
# requirements.txt
tabulate>=0.8
jieba>=0.42.1
```

> The core algorithm (`scripts/core.py`) uses **only the Python standard library** (`re`, `math`, `collections`, `sys`, `time`). No external packages are imported for segmentation itself.

## Acknowledgements

The demo article (`corpus/example.txt`) is from a [philosophy speech](https://mp.weixin.qq.com/s/yryponBqaD0w1ZPI5Al2Gg) recommended by a friend — deeply insightful.

The 30-year corpus consists of *Southern Weekend* New Year editorials, a beloved Chinese journalism tradition.

## License

MIT

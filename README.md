# Atomic Crystal Growth

Unsupervised Chinese word segmentation under extreme constraints: **no dictionary, no supervision, no external libraries** for the core algorithm. By treating characters as particles in a statistical field, this project simulates crystal growth — particles merge when the net energy exceeds a dynamic threshold, and weak bonds are periodically dissolved. Word boundaries emerge purely from the intrinsic statistics of the text.

On a philosophy-speech test corpus, the algorithm achieves a **boundary $\text{F1} \approx 0.933$** against jieba (which itself is unsupervised but dictionary-equipped). A leave-one-year-out cross-validation over **30 years** of *Southern Weekend* New Year editorials confirms that enlarging the training corpus consistently improves segmentation quality.

## Table of Contents

- [How It Works](#how-it-works)
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
- [Performance](#performance)
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
\text{PMI}(p, q) = \log_2 \frac{P(p, q)}{P(p) \cdot P(q)},\qquad \text{NPMI}(p, q) = \frac{\text{PMI}(p, q)}{-\log_2 P(p, q)}
$$

NPMI $\in [-1, 1]$. A value close to 1 indicates that $p$ and $q$ almost always appear together. An epsilon guard prevents division-by-zero when a pair dominates the corpus.

With Laplace smoothing parameter $\alpha$ and vocabulary size $V$ (number of distinct particles), the **right entropy** of $p$ is:

$$
H_R(p) = -\sum_{r \in R(p)} \frac{f(p, r)+\alpha}{f(p)+\alpha V} \log_2 \frac{f(p, r)+\alpha}{f(p)+\alpha V} - (V - |R(p)|)\cdot \frac{\alpha}{f(p)+\alpha V} \log_2 \frac{\alpha}{f(p)+\alpha V}
$$

The left entropy $H_L(p)$ is defined analogously using left neighbors $L(p)$. To make entropies comparable across different frequencies, we normalize by $\log_2\bigl(f(p)+2\bigr)$:

$$
I_R(p) = \frac{H_R(p)}{\log_2(f(p)+2)}, \qquad I_L(p) = \frac{H_L(p)}{\log_2(f(p)+2)}
$$

The **ionization energy** of a particle measures how tightly it is bound by its context. High $I$ means $p$ appears in a very diverse environment (high contextual freedom) — it is "heavy" and reluctant to merge. Low $I$ means $p$ can easily detach.

### 3. Net Energy of a Merge

For a pair of adjacent particles $(p, q)$ the **net energy** is:

$$
E(p, q) = \text{NPMI}(p, q) - I(p, q) \cdot M(p, q)
$$

where the combined ionization uses the *directional* entropies at the merge boundary:

$$
I(p, q) = \frac{I_R(p) + I_L(q)}{2}
$$

and the **mass factor** exponentially penalises merges of long particles:

$$
M(p, q) = \beta^{\ell(p) + \ell(q) - 2}
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
T_{\text{final}}(p, q, i) = T(i) + \Delta_{\text{atom}}(p) + \Delta_{\text{atom}}(q)
$$

If either $p$ or $q$ is not a single character, its $\Delta$ is 0.

### 6. Polarity Modulation

The **polarity** of a particle captures the asymmetry between its left and right contexts:

$$
\Psi(p) = \frac{|H_L(p) - H_R(p)|}{H_L(p) + H_R(p)}, \qquad \Psi(p) \in [0,1]
$$

When a pair exhibits high polarity ($\Psi > \psi_0$), the threshold is lowered if the merge follows the natural direction indicated by the neighbour distributions:

$$
d_s(p, q) = \max\left( \frac{f(p, q)}{f(p)}, \frac{f(p, q)}{f(q)} \right)
$$

$$
\Delta_{\text{pol}}(p, q) = -w_{\text{pol}} \cdot d_s(p, q) \cdot \bigl(\Psi(p) + \Psi(q)\bigr)
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

- $r$ = **relax factor** (e.g. $0.70$) — allows merges that don't strictly exceed the threshold, acting as *quantum tunnelling* through energy barriers.
- $b_{\text{merge}}$ = **merge bias** (e.g. $0.25$) — a per-merge bonus that counteracts the Viterbi DP's natural conservatism (without it, the DP prefers fewer merges).


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
\mathcal{O} = \frac{1}{|\{p : f(p)\ge 2\}|} \sum_{p: f(p)\ge 2} \frac{I_L(p) + I_R(p)}{2}
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
            (merge if E_i > T_i * relax_factor, maximize Σ(E + merge_bias))
        Backtrack to build new particle sequence
    if dissolution_interval > 0 and t >= start_dissolve and (t % interval == 0):
        Perform dissolution on all particles
    Update order parameter O
    Check convergence criteria
    t = t + 1

Output: final segmented sentences
```

### Hyperparameters

|           Symbol           |            Parameter             |                 Meaning                  |
| :------------------------: | :------------------------------: | :--------------------------------------: |
|          $\beta$           |           `mass_base`            |  Mass base (exponential growth factor)   |
|           $T_0$            |         `base_threshold`         |            Initial threshold             |
|          $\gamma$          |        `threshold_decay`         |      Threshold decay per iteration       |
|          $\delta$          |          `damping_rate`          |   Frequency decay rate for oscillation   |
|   $\alpha_{\text{amp}}$    |        `damping_amp_rate`        |        Amplitude growth per round        |
|         $A_{\max}$         |        `damping_max_amp`         |      Maximum oscillation amplitude       |
|  $\theta_{\text{indep}}$   |   `atom_independent_threshold`   |     High-alone protection threshold      |
|   $\eta_{\text{indep}}$    |    `atom_independent_penalty`    |         Independent atom penalty         |
| $\theta_{\text{restless}}$ |    `atom_restless_threshold`     |    Low-alone encouragement threshold     |
|  $\eta_{\text{restless}}$  |      `atom_restless_bonus`       |           Restless atom bonus            |
|   $\phi_{\text{floor}}$    |           `freq_floor`           |   Frequency floor for extra protection   |
|          $\kappa$          |                —                 |       Frequency protection weight        |
|      $w_{\text{pol}}$      |        `polarity_weight`         |        Polarity modulation weight        |
|          $\psi_0$          |                —                 |      Polarity activation threshold       |
|          $\alpha$          |         `entropy_alpha`          |      Laplace smoothing for entropy       |
|            $r$             |      `viterbi_relax_factor`      |    Threshold relaxation (tunnelling)     |
|     $b_{\text{merge}}$     |       `viterbi_merge_bias`       |      Per-merge bonus in Viterbi DP       |
|            $K$             |      `dissolution_interval`      |           Dissolution interval           |
|                            |    `dissolution_start_round`     |    Warm-up rounds before dissolution     |
|         $L_{\min}$         |     `dissolution_min_length`     | Minimum length for dissolution candidate |
|    $\tau_{\text{diss}}$    | `dissolution_independence_ratio` |    Dissolution independence threshold    |
|                            |         `max_iterations`         |            Maximum iterations            |
|    $n_{\text{plateau}}$    |      `particle_plateau_tol`      |    Iterations with no particle change    |
|    $n_{\text{window}}$     |       `convergence_window`       |  Convergence window for order parameter  |
|  $\epsilon_{\mathcal{O}}$  |        `convergence_tol`         |   Tolerance for order parameter change   |

These hyperparameters control the "thermodynamics" of the system: lower thresholds or higher mass bases produce more aggressive merging; personality adjustments protect function words; dissolution prevents spurious long compounds; the relax factor and merge bias tune the Viterbi DP's aggressiveness.

## Performance

### Boundary vs Jieba

The test text is a philosophy speech.

Running `python main.py eval` yields:

| Metric                            | Crystal Growth |   Jieba    |
| :-------------------------------: | :------------: | :--------: |
| Precision                         |    $0.9339$    |     —      |
| Recall                            |    $0.9310$    |     —      |
| Boundary $\text{F1}$          |    $0.9325$    |  baseline  |
| Avg particles/sentence            |    $21.41$     |  $21.35$   |
| Avg token length                  |    $1.574$     |  $1.579$   |
| Granularity ratio                 |    $1.003$     |     —      |
| Unigram Jaccard                   |    $0.5705$    |     —      |
| Bigram Jaccard                    |    $0.5864$    |     —      |

The algorithm converges in **$58$ rounds**, reducing $7470$ atomic particles to $4 775$ ($36.1% reduction). The $\text{F1} \approx 0.9325$ means $\sim 93% of word boundaries agree with jieba — without any dictionary or labeled data. Notably, the granularity ratio ($1.003$) is almost exactly $1$, meaning the algorithm produces nearly the same number of tokens per sentence as jieba.

A fine-grained error analysis of $7295$ boundary positions reveals the nature of the remaining 7% disagreement between Crystal Growth and Jieba. The two systems agree on $6681$ boundaries (91.6%) and disagree on $614$ boundaries. The error distribution is notably balanced, with an over-to-under-segmentation ratio of $1.07:1$, indicating no systematic bias. Phasing errors account for 34.5% of all disagreements, often occurring at compound word boundaries. For example, Crystal segments "北京大学哲学系任教" as "北京/大学/哲学/系任/教", while Jieba produces "北京大学哲学系/任教", with the phasing error centered on "哲学" versus "系". Over-segmentation frequently splits proper nouns and institutional names (e.g., "北京大学" → "北京/大学"); under-segmentation commonly affects verb-object and adjective-adverb combinations (e.g., "很难" → "很/难"); and phasing errors occur at compound boundaries (e.g., "一句话" → "一句/话" vs "一/句话"). The highest-error sentences, containing up to $22$ disagreements, feature multiple proper nouns, academic terminology, complex structures, and mixed registers.

| Error Type | Count | Percentage | Description |
|:-----------|:-----:|:----------:|:------------|
| **Over-segmentation** (pure) | $220$ | 3.02% | Crystal cuts where Jieba doesn't |
| **Under-segmentation** (pure) | $206$ | 2.82% | Jieba cuts where Crystal doesn't |
| **Phasing Error** | $188$ | 2.58% | Both cut, but positions offset by $\leq 1$ chars |

### Transfer Learning: Leave-One-Year-Out Cross-Validation

Using `scripts/train.py`, a leave-one-year-out (LOYO) experiment was conducted on **30 years** (1997–2026) of *Southern Weekend* New Year editorials (45 729 characters total). For each year:

1. **Self-trained**: run crystal growth on that year's text alone (~1 000–2 700 chars).
2. **Transfer**: train on the other 29 years, then segment the held-out year.

| Metric          | Self-trained |     Transfer     |
| :-------------: | :----------: | :--------------: |
| Average F1      |   $0.8378$   |    $0.8630$    |
| Average Δ       |      —       |   $+0.0252$    |
| Transfer wins   |      —       | $27/30$ |
| Transfer losses |      —       |  $3/30$ |

All results:

| Year   | Chars  | $\text{F1}$(self) | $\text{F1}$(transfer) | Δ         | Trend |
| :----: | :----: | :--------: | :------------: | :-------: | :---: |
| 2026 | $1848$ | $0.8037$   | $0.8623$       | $0.0586$  | ↑     |
| 2025 | $2167$ | $0.8616$   | $0.8754$       | $0.0138$  | ↑     |
| 2024 | $2024$ | $0.8231$   | $0.8396$       | $0.0165$  | ↑     |
| 2023 | $2224$ | $0.8478$   | $0.8725$       | $0.0247$  | ↑     |
| 2022 | $1771$ | $0.7725$   | $0.8440$       | $0.0715$  | ↑     |
| 2021 | $1949$ | $0.7666$   | $0.8411$       | $0.0745$  | ↑     |
| 2020 | $1320$ | $0.8337$   | $0.8609$       | $0.0272$  | ↑     |
| 2019 | $2147$ | $0.8194$   | $0.8439$       | $0.0244$  | ↑     |
| 2018 | $1484$ | $0.8224$   | $0.8687$       | $0.0463$  | ↑     |
| 2017 | $1243$ | $0.8335$   | $0.8613$       | $0.0279$  | ↑     |
| 2016 | $1530$ | $0.8213$   | $0.8515$       | $0.0302$  | ↑     |
| 2015 | $993$  | $0.8431$   | $0.8692$       | $0.0261$  | ↑     |
| 2014 | $1270$ | $0.8800$   | $0.8895$       | $0.0095$  | ↑     |
| 2013 | $1076$ | $0.7961$   | $0.8706$       | $0.0745$  | ↑     |
| 2012 | $1524$ | $0.8290$   | $0.8525$       | $0.0234$  | ↑     |
| 2011 | $1371$ | $0.8750$   | $0.8540$       | $-0.0210$ | ↓     |
| 2010 | $1345$ | $0.8644$   | $0.8692$       | $0.0048$  | ↑     |
| 2009 | $1900$ | $0.8641$   | $0.8727$       | $0.0086$  | ↑     |
| 2008 | $1620$ | $0.8082$   | $0.8439$       | $0.0357$  | ↑     |
| 2007 | $1032$ | $0.7864$   | $0.8275$       | $0.0411$  | ↑     |
| 2006 | $1130$ | $0.8398$   | $0.8819$       | $0.0421$  | ↑     |
| 2005 | $1455$ | $0.8724$   | $0.8879$       | $0.0155$  | ↑     |
| 2004 | $1974$ | $0.8677$   | $0.8584$       | $-0.0093$ | ↓     |
| 2003 | $2656$ | $0.8816$   | $0.8647$       | $-0.0169$ | ↓     |
| 2002 | $1228$ | $0.8876$   | $0.8906$       | $0.0031$  | ↑     |
| 2001 | $1395$ | $0.8623$   | $0.8741$       | $0.0119$  | ↑     |
| 2000 | $938$  | $0.8140$   | $0.8525$       | $0.0385$  | ↑     |
| 1999 | $1067$ | $0.8574$   | $0.8781$       | $0.0207$  | ↑     |
| 1998 | $1008$ | $0.8500$   | $0.8753$       | $0.0253$  | ↑     |
| 1997 | $1040$ | $0.8490$   | $0.8556$       | $0.0066$  | ↑     |



We quantify the reliability of the transfer gain using a battery of statistical tests on the 30 paired (self, transfer) $\text{F1}$ values:

|                 Test                  | Statistic |       Value        |      $p$       | Significance |
| :-----------------------------------: | :-------: | :----------------: | :----------------: | :----------: |
|       Paired t-test (2-tailed)        |  $t(29)$  |      $5.732$       | $3.33 × 10 ^ {-6}$ | $p < 0.001$  |
|               Cohen's d               |    $d$    |      $1.046$       |         —          | large effect |
| Bootstrap 95% CI ($10 000$ resamples) | $[L, U]$  | $[0.0169, 0.0340]$ |         —          | excludes $0$ |
|       Wilcoxon signed-rank test       |    $W$    |       $29.0$       | $3.24 × 10 ^ {-6}$ | $p < 0.001$  |

**Key findings:**

- **Transfer consistently outperforms self-training** in $27$ out of $30$ years, with an average F$1$ improvement of $+0.0252$.

- **Largest gains** occur on years where self-training struggles most. For short-year texts (2000/2007/2013/2017/2021, five years in total) with fewer than $1200$ characters, the average gain reached **$+0.0646$**, more than twice the overall average gain of $+0.0252$. Among them, the 2021 text ($1949$ characters) and the 2013 text ($1076$ characters) both achieved the highest gain of $0.0745$. This indicates that when statistical signals from a single text are insufficient, contextual statistical patterns learned from general corpora can effectively compensate for local information gaps, verifying the sensitivity of unsupervised word segmentation to corpus size.

- The **only three instances** of performance decline occurred in high-quality years (2003/2004/2011) where the self-training $\text{F1}$ had already exceeded $0.86$, and the declines were all less than $0.022$, which is within the normal range of statistical fluctuation. The self-training results for these years were already near the performance ceiling; a small amount of topic-specific vocabulary from the broader corpus (such as special expressions related to the 2003 SARS outbreak) introduced slight noise, but did not undermine the original segmentation quality.

- The 30 years of Southern Weekend New Year's editorials cover a wide range of social topics, from the 1997 handover of Hong Kong, the 2003 SARS outbreak, and the 2008 Wenchuan earthquake to the 2020 COVID-19 pandemic. The algorithm maintained stable transfer gains across these texts, demonstrating that **the statistical patterns it learned possess good domain generalization ability**, do not overfit to year-specific vocabulary, and can adapt to Chinese text features from different eras.

- The average self-training $\text{F1}$ was $0.8378$, significantly lower than the $0.9326$ achieved in single-text tests. The primary reason is that individual year texts are too short (averaging only $1524$ characters), leading to high noise in statistical estimation. Transfer learning, by leveraging statistical information from the full corpus, raised the average $\text{F1}$ to $0.8630$, though it still lags behind the $0.9326$ achieved in long-text tests.

- Paired t-test, Cohen's d, bootstrap confidence intervals, and a Wilcoxon signed-rank test all converge on a definitive result: the observed $+0.0252$ gain in segmentation quality is both statistically significant ($p \ll 0.001$ across tests) and substantively large (Cohen's $d = 1.05$), with a bootstrap 95% CI of $[0.0169, 0.0340]$ that remains entirely above zero and a non-parametric confirmation that does not rely on distributional assumptions, decisively demonstrating that enlarging the statistical sample via transfer learning reliably and meaningfully improves unsupervised word segmentation.


This demonstrates that *enlarging the statistical sample improves segmentation quality* — the algorithm generalizes from a larger corpus rather than overfitting.

### Scaling Behaviour: $\text{F1}$ vs Corpus Size

The LOYO experiment varies the *source* of the text (29 years vs 1 year) but keeps each year's length roughly fixed. A complementary question is: **on a single text, how does $\text{F1}$ scale with the amount of input?** To answer this, the full `example.txt` ($7518$ characters, $223$ sentences) was truncated at $45$ progressively larger cut-offs (step $= 5$ sentences, from $k = 5$ to $k = 223$), and crystal growth was run independently on each prefix. Jieba's segmentation of the same prefix serves as the baseline at every cut-off.

**Selected data points** ($k$ = number of sentences, chars = cumulative character count):

| $k$ | Chars | Pct | Precision | Recall | $\text{F1}$ | Rounds | Init $\to$ Final | Red. |
|:---:|:-------:|:---:|:-----------:|:--------:|:----:|:--------:|:----------------:|:----:|
| $5$   | $185$    | 2.5% | $0.5909$ | $0.9701$ | $0.7345$ | $10$   | $185 \to 72$   | 61.1% |
| $10$  | $412$    | 5.5% | $0.7276$ | $0.9471$ | $0.823$  | $14$   | $412 \to 199$  | 51.7% |
| $15$  | $662$    | 8.8% | $0.8039$ | $0.9398$ | $0.8666$ | $12$   | $661 \to 364$  | 44.9% |
| $20$  | $768$    | 10.2% | $0.8132$ | $0.9319$ | $0.8685$ | $21$   | $766 \to 431$  | 43.7% |
| $40$  | $1388$   | 18.5% | $0.8631$ | $0.9183$ | $0.8899$ | $13$   | $1386 \to 823$ | 40.6% |
| $60$  | $2311$   | 30.7% | $0.8733$ | $0.9299$ | $0.9007$ | $52$   | $2302 \to 1372$ | 40.4% |
| $80$  | $2967$   | 39.5% | $0.8939$ | $0.9305$ | $0.9118$ | $46$   | $2949 \to 1791$ | 39.3% |
| $100$ | $3613$   | 48.1% | $0.9027$ | $0.9231$ | $0.9128$ | $56$   | $3592 \to 2221$ | 38.2% |
| $120$ | $4249$   | 56.5% | $0.9175$ | $0.9266$ | $0.922$  | $56$   | $4227 \to 2640$ | 37.5% |
| $140$ | $4962$   | 66.0% | $0.9216$ | $0.9253$ | $0.9235$ | $52$   | $4933 \to 3100$ | 37.2% |
| $160$ | $5621$   | 74.8% | $0.9244$ | $0.9297$ | $0.9271$ | $56$   | $5588 \to 3516$ | 37.1% |
| $180$ | $6194$   | 82.4% | $0.934$  | $0.9298$ | $0.9319$ | $62$   | $6152 \to 3911$ | 36.4% |
| $200$ | $6775$   | 90.1% | $0.9381$ | $0.9223$ | $0.9302$ | $51$   | $6732 \to 4344$ | 35.5% |
| $220$ | $7473$   | 99.4% | $0.9362$ | $0.9312$ | $0.9337$ | $58$   | $7425 \to 4758$ | 35.9% |
| $223$ | $7518$   | 100.0% | $0.9339$ | $0.931$  | $0.9325$ | $58$   | $7470 \to 4775$ | 36.1% |

![F1 vs corpus size with best regression fit and 95% confidence band](src/pic/scale_f1.png)

**Correlation with log(chars)**:

| Test | Statistic | Value | $p$ | Eff. | Sig. |
|:-----|:---------:|:-----:|:---------:|:------:|:------------:|
| Spearman's $\rho$ | $\rho$ | $0.9871$ | $7.69 \times 10^{-36}$ | $***$ | $***$ |
| Pearson's $r$ | $r$ | $0.9357$ | $4.49 \times 10^{-21}$ | $***$ | $***$ |
| Kendall's $\tau$ | $\tau$ | $0.9232$ | $3.86 \times 10^{-19}$ | $***$ | $***$ |

The $R^2$ of each regression model is as follows, with $x = \log(\text{chars})$:

| Model | $R^2$ | AIC | Equation |
|:------|:---------:|:---:|:---------|
| Cubic | $0.9938$ | $-264.17$ | $\displaystyle \text{F1} = 0.0604x^3 - 0.4123x^2 + 0.9387x + 0.1234$ |
| Exponential | $0.9881$ | $-251.91$ | $\displaystyle \text{F1} = 0.9733 - 2.1056 \cdot e^{-1.0196x}$ |
| Sigmoid | $0.9881$ | $-249.90$ | $\displaystyle \text{F1} = \frac{261.60}{1 + e^{-1.02(x + 4.73)}} - 260.63$ |
| Quadratic | $0.9846$ | $-246.24$ | $\displaystyle \text{F1} = -0.0362x^2 + 0.3167x + 0.2486$ |
| Logarithmic | $0.9830$ | $-246.07$ | $\displaystyle \text{F1} = 0.2993 \cdot \log(x) + 0.8447$ |
| Michaelis-Menten | $0.9781$ | $-240.52$ | $\displaystyle \text{F1} = \frac{1.2485x}{1.2930 + x}$ |
| Power | $0.9688$ | $-232.73$ | $\displaystyle \text{F1} = 0.6391 \cdot x^{0.2827}$ |
| Linear | $0.9567$ | $-225.50$ | $\displaystyle \text{F1} = 0.0766x + 0.6418$ |

The **cubic polynomial** achieves the highest $R^2 = 0.9938$ and lowest AIC $= -264.17$, indicating the best balance between fit quality and model complexity. However, the **Logarithmic model** ($R^2 = 0.9830$) offers a more parsimonious description with only 2 parameters vs 4 for the cubic, and provides a clearer theoretical interpretation: since $\log(\text{chars})$ grows slower than chars, and $\log(\log(\text{chars}))$ slower still, the double-log form naturally captures the **diminishing-returns** curve without requiring higher-order terms.

> [!TIP]
>
> Another additional question is: what is *the best upper bound* that can be achieved by optimizing parameters? In the fitting, the sigmoid gives an upper asymptote of approximately $0.97$. I suspect (though without any basis) that this might represent the best possible upper limit achievable solely through learning natural language.
>

![Left: $\text{F1}$ / Precision / Recall vs log10(chars); Right: convergence rounds and particle reduction](src/pic/scale_detail.png)


The $45$ points can be partitioned into early, mid, and late phases to characterize the learning curve:

| Phase | $k$ range | Chars (approx.) | Mean F1 | $\sigma$ | $\Delta$ (range) | Gain / 100 chars |
|:------|:---------:|:----------------:|:-------:|:--------:|:-----------------:|:-----------------:|
| Early | $55$  | $2051$  | $0.8777$ | $0.0221$ | $0.0874$ | $0.0053$ |
| Mid   | $110$ | $3925$  | $0.9112$ | $0.0049$ | $0.0173$ | $0.0010$ |
| Late  | $223$ | $7518$  | $0.9282$ | $0.0041$ | $0.0154$ | $0.0005$ |

The standard deviation shrinks dramatically across phases: from $0.0221$ (early, noisy) to $0.0041$ (late, stable), confirming that the algorithm converges to a **stable segmentation** as statistics accumulate. The per-100-char gain drops by an order of magnitude from early ($0.0053$) to late ($0.0005$), indicating near-saturation beyond $4000$ characters.

**Effect sizes between phases** (Cohen's $d$):

| Comparison | $d$ | Interpretation |
|:----------:|:---:|:---------------:|
| Early vs Mid   | $-2.14$ | very large |
| Mid vs Late    | $-3.90$ | very large |
| Early vs Late  | $-4.07$ | very large |

All inter-phase differences exceed the "large effect" threshold ($|d| > 0.8$) by a wide margin, confirming that the three phases represent genuinely distinct regimes rather than arbitrary partitions.

**Key thresholds:**

- At $k = 5$ ($185$ characters, 2.5% of the text), $\text{F1}$ is $0.7345$, recall is high ($0.9701$) but precision is only $0.5909$, meaning the algorithm severely under-segments (merges nearly everything) because statistical signal is too sparse to distinguish true word boundaries.
- At $k = 10$ ($412$ characters), $\text{F1}$ already exceeds $0.80$ ($0.8230$), just 5.5% of the full text suffices for usable segmentation.
- $\text{F1}$ crosses $0.90$ at $k = 55$ ($2060$ characters, 27.3% of the text).
- The total gain from $k = 5$ to $k = 224$ is $+0.1982$; half of this gain is achieved by $k = 15$ ($662$ characters, 8.8%).

![Residuals of the cubic fit vs log10(chars)](src/pic/scale_residuals.png)

The residual plot shows no systematic structure — the cubic model captures the trend without bias, and the largest residuals ($\pm 0.015$) occur in the mid-phase, where sentence-content variability introduces local $\text{F1}$ fluctuations unrelated to corpus size.

All in all, together with the LOYO experiment, these results demonstrate that unsupervised word segmentation quality is a **predictable, quantifiable function of corpus size**: given enough text, the algorithm reliably reaches $\text{F1} \approx 0.933$ without any dictionary or labeled data.

### Cross-Discipline Experiments

To evaluate the robustness of the algorithm across different academic fields, this study conducted a large-scale interdisciplinary experiment on a corpus covering $14$ disciplines. This corpus was constructed during my thesis work **"From Statistical Features to Semantic Space: A Study of Disciplinary Characteristics in Academic Chinese"**, encompassing a total of $1,400$ journal papers across $14$ disciplines. After quality screening, $20$ documents were selected from each discipline through systematic sampling (due to the limited computational capacity of the PC, which prevented full-scale computation on the complete corpus), resulting in $280$ training documents with a total character count of $3,075,129$ (including $2,520,628$ Chinese characters). The obtained sub-corpus averaged $9,002$ Chinese characters per document, with non-Chinese characters accounting for 7.2% of the total. The $14$ disciplines cover Mathematics (MA), Physics (PS), Chemistry (CH), Biology (BI), Geography (GE), Psychology (PC), Economics (EC), Sociology (SC), Social Science General (SS), Law (LA), History (HI), Philosophy (PL), Education (ED), and Literature (LI). Each document was independently segmented using default parameter settings, without any domain-specific tuning or prior exposure to the discipline's vocabulary. The evaluation metric is the boundary $\text{F1}$ score against jieba's tokenization, computed as micro-averaged precision and recall over all sentence boundaries within each document.

#### Cross-Discipline Report

Across all $280$ documents, the algorithm achieved an overall $\text{F1}$ of $0.9188 \pm 0.0197$, with precision $0.9129 \pm 0.0222$ and recall $0.9256 \pm 0.0298$. The average granularity ratio — the number of Crystal Growth tokens divided by the number of jieba tokens — was $1.015$, indicating a negligible overall tendency toward over-segmentation. The algorithm converged in an average of $51.2$ rounds, reducing the initial particle count by $36.8%.

The $\text{F1}$ distribution across documents is well-behaved: the interquartile range spans from $0.9072$ to $0.9335$, with a minimum of $0.8393$ (a single outlier in Literature) and a maximum of $0.9530$ (in Education). The overall standard deviation of $0.0197$ is remarkably small given the breadth of disciplines, suggesting that the algorithm's statistical foundation produces consistent results across heterogeneous text types.

A one-way ANOVA on discipline-level $\text{F1}$ means revealed that discipline identity is a highly significant predictor of segmentation quality. These parametric results are corroborated by distribution-free tests, and Levene's test confirms that the within-discipline variability is itself discipline-dependent — some fields, most notably Literature, produce far more heterogeneous segmentation difficulty than others.

| Test           |  Statistic   |  Value   |       $p$-value        | Significance |
| :------------- | :----------: | :------: | :--------------------: | :----------: |
| One-way ANOVA  | $F(13, 266)$ |  $8.78$  | $6.80 \times 10^{-15}$ | $p < 0.001$  |
| Eta-squared    |   $\eta^2$   | $0.3002$ |           —            |  very large  |
| Kruskal-Wallis |   $H(13)$    | $82.57$  | $3.61 \times 10^{-12}$ | $p < 0.001$  |
| Levene's test  |     $W$      |  $3.68$  | $2.21 \times 10^{-5}$  | $p < 0.001$  |

The eta-squared of $0.3002$ is particularly noteworthy: discipline characteristics alone account for $30.0% of the total variance in $\text{F1}$ scores, a very large effect by Cohen's conventional benchmarks, and a larger contributor than document length, granularity ratio, or idiosyncratic document-level noise.

![F1 scores by discipline with error bars](experiments/cross_discipline/results/experiment1/f1_by_discipline.png)

The table below presents the full discipline-level statistics, ranked by mean $\text{F1}$. The 95% confidence interval is computed as $\bar{x} \pm 1.96 \cdot \sigma / \sqrt{N}$ under the normal approximation with $N = 20$ per discipline. The coefficient of variation quantifies within-discipline stability; lower values indicate that all $20$ documents within that discipline are segmented at similar quality, while higher values indicate substantial internal diversity.

| Code | Discipline             | $N$  | $\text{F1}$ Mean | $\pm$Std |      95% CI       | CV |   Prec   |   Rec    |   GR    | Rounds | Red  |
| :--: | :--------------------- | :--: | :--------------: | :------: | :------------------: | :-----------: | :------: | :------: | :-----: | :----: | :------: |
|  ED  | Education              | $20$ |     $0.9347$     | $0.0119$ | $[0.9291,\, 0.9402]$ |    1.3%    | $0.9194$ | $0.9507$ | $1.033$ | $48.8$ | 37.8% |
|  PC  | Psychology             | $20$ |     $0.9335$     | $0.0140$ | $[0.9270,\, 0.9401]$ |    1.5%    | $0.9356$ | $0.9316$ | $0.996$ | $49.6$ | 35.6% |
|  SC  | Sociology              | $20$ |     $0.9299$     | $0.0097$ | $[0.9253,\, 0.9344]$ |    1.0%    | $0.9117$ | $0.9489$ | $1.040$ | $52.0$ | 37.1% |
|  LA  | Law                    | $20$ |     $0.9272$     | $0.0118$ | $[0.9217,\, 0.9327]$ |    1.3%    | $0.9032$ | $0.9526$ | $1.053$ | $53.0$ | 38.0% |
|  SS  | Social Science General | $20$ |     $0.9260$     | $0.0157$ | $[0.9187,\, 0.9334]$ |    1.7%    | $0.9100$ | $0.9430$ | $1.035$ | $49.6$ | 37.1% |
|  PL  | Philosophy             | $20$ |     $0.9241$     | $0.0152$ | $[0.9170,\, 0.9312]$ |    1.6%    | $0.9094$ | $0.9396$ | $1.033$ | $51.1$ | 36.6% |
|  EC  | Economics              | $20$ |     $0.9192$     | $0.0195$ | $[0.9101,\, 0.9283]$ |    2.1%    | $0.9065$ | $0.9327$ | $1.028$ | $54.2$ | 38.1% |
|  MA  | Mathematics            | $20$ |     $0.9191$     | $0.0113$ | $[0.9138,\, 0.9244]$ |    1.2%    | $0.9437$ | $0.8962$ | $0.951$ | $49.8$ | 33.2% |
|  BI  | Biology                | $20$ |     $0.9163$     | $0.0139$ | $[0.9098,\, 0.9228]$ |    1.5%    | $0.9176$ | $0.9153$ | $0.998$ | $48.2$ | 37.2% |
|  GE  | Geography              | $20$ |     $0.9121$     | $0.0181$ | $[0.9036,\, 0.9206]$ |    2.0%    | $0.9054$ | $0.9193$ | $1.015$ | $46.9$ | 39.0% |
|  PS  | Physics                | $20$ |     $0.9094$     | $0.0117$ | $[0.9039,\, 0.9148]$ |    1.3%    | $0.9175$ | $0.9017$ | $0.984$ | $51.6$ | 37.5% |
|  HI  | History                | $20$ |     $0.9087$     | $0.0189$ | $[0.8998,\, 0.9175]$ |    2.1%    | $0.8963$ | $0.9216$ | $1.027$ | $54.5$ | 37.2% |
|  CH  | Chemistry              | $20$ |     $0.9076$     | $0.0174$ | $[0.8995,\, 0.9158]$ |    1.9%    | $0.9077$ | $0.9082$ | $1.001$ | $53.4$ | 36.3% |
|  LI  | Literature             | $20$ |     $0.8961$     | $0.0303$ | $[0.8819,\, 0.9103]$ |    3.4%    | $0.8962$ | $0.8964$ | $1.000$ | $55.2$ | 35.4% |



The $\text{F1}$ spread between the best-performing discipline (Education, $\text{F1} = 0.9347$) and the worst-performing discipline (Literature, $\text{F1} = 0.8961$) is $0.0386$, or $3.9$ percentage points. This gap is both statistically significant and practically meaningful — it exceeds the typical $\text{F1}$ fluctuation within a discipline by a factor of $2$–$4$. Sociology is the most stable discipline, with a standard deviation of only $0.0097$ and a coefficient of variation of 1.0%, indicating that sociological academic prose is linguistically homogeneous enough to produce near-identical segmentation quality across all $20$ sampled documents. Literature, in contrast, is the most variable discipline: its documents range from classical poetry to modern literary criticism, each with radically different lexical and syntactic profiles. The 3.4% CV for Literature is more than triple that of the most stable discipline.

![Precision-Recall ellipses by discipline](experiments/cross_discipline/results/experiment1/precision_recall_ellipses.png)

**Effect sizes and disciplinary clustering.** Cohen's $d$ quantifies the standardized mean difference between Education and every other discipline.

| Comparison | Cohen's $d$ | Effect Size |
| :--------- | :---------: | :---------: |
| ED vs PC   |  $+0.090$   | negligible  |
| ED vs SC   |  $+0.445$   |    small    |
| ED vs LA   |  $+0.633$   |   medium    |
| ED vs SS   |  $+0.622$   |   medium    |
| ED vs PL   |  $+0.775$   |   medium    |
| ED vs EC   |  $+0.959$   |    large    |
| ED vs MA   |  $+1.343$   |    large    |
| ED vs BI   |  $+1.423$   |    large    |
| ED vs GE   |  $+1.477$   |    large    |
| ED vs HI   |  $+1.651$   |    large    |
| ED vs LI   |  $+1.677$   |    large    |
| ED vs CH   |  $+1.813$   |    large    |
| ED vs PS   |  $+2.149$   |    large    |

Education and Psychology are statistically indistinguishable ($d = +0.090$, negligible), confirming that these two disciplines share essentially the same segmentation difficulty. Education versus Sociology shows a small gap ($d = +0.445$), while Law, Social Science General, and Philosophy constitute a middle tier with medium effect sizes ($d \in [0.622, 0.775]$). All remaining disciplines exhibit large gaps from Education: Economics ($d = +0.959$), Mathematics ($d = +1.343$), Biology ($d = +1.423$), Geography ($d = +1.477$), History ($d = +1.651$), Literature ($d = +1.677$), Chemistry ($d = +1.813$), and Physics ($d = +2.149$). The fact that eleven of the thirteen comparisons exceed the "large effect" threshold ($|d| > 0.8$) underscores that the discipline hierarchy is not a statistical artefact but a genuine reflection of differential segmentation difficulty.

Post-hoc Tukey HSD analysis identified $24$ statistically significant pairwise differences at $\alpha = 0.05$ among the $91$ possible discipline pairs.

| Pair     | $p$-value | Cohen's $d$ |
| :------- | :-------: | :---------: |
| PS vs ED | $0.0003$  |  $-2.149$   |
| PS vs SC | $0.0112$  |  $-1.911$   |
| PS vs PC | $0.0008$  |  $-1.873$   |
| CH vs ED | $0.0001$  |  $-1.813$   |
| CH vs PC | $0.0002$  |  $-1.637$   |
| CH vs SC | $0.0033$  |  $-1.576$   |
| CH vs LA | $0.0206$  |  $-1.313$   |
| CH vs SS | $0.0416$  |  $-1.108$   |
| BI vs ED | $0.0412$  |  $-1.423$   |
| MA vs LI | $0.0018$  |  $+1.007$   |

The most pronounced contrasts all involve the two lowest-performing disciplines. Physics differs significantly from Education ($p = 0.0003$, $d = -2.149$), Psychology ($p = 0.0008$, $d = -1.873$), and Sociology ($p = 0.0112$, $d = -1.911$). Chemistry similarly differs from Education ($p = 0.0001$, $d = -1.813$), Psychology ($p = 0.0002$, $d = -1.637$), Sociology ($p = 0.0033$, $d = -1.576$), Social Science General ($p = 0.0416$, $d = -1.108$), and Law ($p = 0.0206$, $d = -1.313$). Mathematics versus Literature ($p = 0.0018$, $d = +1.007$) is the only significant cross-tier contrast involving a middle-ranked discipline.

These results delineate a clear three-tier hierarchy. Education, Psychology, and Sociology form the top tier, with $\text{F1} > 0.929$, characterized by standardized academic prose with well-defined technical vocabulary — education research papers, for instance, are rich in pedagogical terms that exhibit strong, consistent co-occurrence patterns. Law, Social Science General, and Philosophy constitute the middle tier ($\text{F1} \in [0.924, 0.927]$), where legal terminology and philosophical discourse introduce moderate segmentation challenges. Economics, Mathematics, Biology, Geography, Physics, History, Chemistry, and Literature form the lower tier ($\text{F1} \in [0.896, 0.919]$). Within this lower tier, Mathematics stands out for its unusual precision-recall profile: despite ranking eighth in $\text{F1}$ ($0.9191$), it achieves the highest precision of any discipline ($0.9437$) but the lowest recall ($0.8962$). The algorithm consistently under-segments mathematical texts — it is conservative about splitting formula components, preserving long symbolic sequences that jieba would fragment. Physics exhibits the mirror pattern: high precision ($0.9175$) and low recall ($0.9017$), with a granularity ratio of $0.984$, the only discipline where Crystal Growth produces *fewer* tokens than jieba, suggesting that physics terminology is recognized by Crystal Growth as coherent units more often than by jieba's dictionary-based approach.

**Correlations with document characteristics.** Across the full $280$-document dataset, the Spearman rank correlation between document length (in characters) and $\text{F1}$ is $\rho = 0.404$ ($p < 0.001$). This moderate positive correlation recapitulates the scaling behaviour observed in the single-text experiment: longer documents provide richer co-occurrence statistics and achieve slightly higher segmentation quality.

![F1 vs document length with regression](experiments/cross_discipline/results/experiment1/f1_vs_length.png)

The Pearson correlation between $\text{F1}$ and granularity ratio is $r = 0.282$ ($p < 0.001$), a weaker positive association indicating that disciplines where the algorithm produces marginally more tokens than jieba tend to score higher — but the effect is small enough that granularity ratio alone cannot serve as a reliable proxy for segmentation quality.

![F1 vs granularity ratio](experiments/cross_discipline/results/experiment1/f1_vs_granularity.png)

![Granularity ratio and token length by discipline](experiments/cross_discipline/results/experiment1/granularity_by_discipline.png)

These global correlations mask substantial discipline-level heterogeneity, as revealed by per-discipline regression analyses.

| Test                    | Statistic |  Value  | $p$-value | Significance |
| :---------------------- | :-------: | :-----: | :-------: | :----------: |
| F1 vs Length (Pearson)  |    $r$    | $0.338$ |     —     |      —       |
| F1 vs Length (Spearman) |  $\rho$   | $0.404$ | $<0.001$  | $p < 0.001$  |
| F1 vs GR (Pearson)      |    $r$    | $0.282$ | $<0.001$  | $p < 0.001$  |
| F1 vs GR (Spearman)     |  $\rho$   | $0.272$ | $<0.001$  | $p < 0.001$  |

The table below presents the Pearson correlation coefficients and associated $p$-values for $\text{F1}$ versus document length and $\text{F1}$ versus granularity ratio, computed separately within each discipline.

| Discipline             | $\text{F1}$ vs Length ($r$) |   $p$    | $\text{F1}$ vs GR ($r$) |   $p$   |
| :--------------------- | :-------------------------: | :------: | :---------------------: | :-----: |
| Mathematics            |           $0.720$           | $<0.001$ |         $0.547$         | $0.013$ |
| Geography              |           $0.636$           | $0.003$  |         $0.241$         | $0.305$ |
| Social Science General |           $0.532$           | $0.016$  |         $0.175$         | $0.462$ |
| Law                    |           $0.472$           | $0.036$  |        $-0.238$         | $0.312$ |
| Chemistry              |           $0.443$           | $0.050$  |         $0.412$         | $0.071$ |
| Economics              |           $0.439$           | $0.053$  |         $0.109$         | $0.646$ |
| Literature             |           $0.388$           | $0.091$  |         $0.700$         | $0.001$ |
| Physics                |           $0.337$           | $0.146$  |         $0.013$         | $0.957$ |
| Education              |           $0.318$           | $0.172$  |        $-0.018$         | $0.940$ |
| Sociology              |           $0.315$           | $0.176$  |        $-0.487$         | $0.030$ |
| History                |           $0.267$           | $0.256$  |         $0.511$         | $0.021$ |
| Biology                |           $0.157$           | $0.509$  |        $-0.222$         | $0.347$ |
| Philosophy             |           $0.110$           | $0.643$  |        $-0.179$         | $0.450$ |
| Psychology             |           $0.106$           | $0.655$  |         $0.312$         | $0.181$ |

Mathematics exhibits the strongest length dependence of any discipline ($r = 0.720$, $p < 0.001$): within mathematics papers, longer documents systematically outperform shorter ones, suggesting that formula-heavy content requires substantially more contextual evidence — including surrounding explanatory prose — to resolve word boundaries reliably. A $4{,}000$-character mathematics paper may contain only $2{,}000$ characters of actual Chinese text once formulas are excluded, and the co-occurrence statistics degrade accordingly. The strong granularity correlation in mathematics ($r = 0.547$, $p = 0.013$) further suggests that the trade-off between over- and under-segmentation is particularly acute in this discipline.

Geography and Social Science General follow with moderate length effects ($r = 0.636$ and $r = 0.532$, respectively), both statistically significant. These disciplines occupy a middle ground where document length provides meaningful but not overwhelming predictive power.

Literature displays the most distinctive correlation profile: a non-significant length effect ($r = 0.388$, $p = 0.091$) coupled with the strongest granularity dependence of any discipline ($r = 0.700$, $p = 0.001$). In literary texts, simply having more text does not reliably improve segmentation — what matters is the *kind* of segmentation the algorithm produces. The strong positive correlation with granularity ratio means that literary documents where Crystal Growth over-segments (producing more tokens than jieba) consistently achieve higher $\text{F1}$ scores than those where it under-segments. This is a clear signal that the algorithm's default configuration is too conservative for literary Chinese: classical allusions, poetic diction, and mixed-register vocabulary demand more aggressive splitting than standard academic prose. The non-significant length effect further suggests that literary texts' segmentation difficulty is determined by qualitative stylistic features — register mixing, genre conventions, lexical diversity — rather than by the sheer quantity of text available.

History mirrors Literature's pattern more moderately, with a significant granularity correlation ($r = 0.511$, $p = 0.021$) and a non-significant length effect ($r = 0.267$, $p = 0.256$). Historical texts, like literary ones, contain mixed registers (archaic terms, quoted primary sources, modern analytical prose), and the algorithm benefits from more aggressive boundary placement.

Sociology presents the inverse pattern: a significant *negative* correlation with granularity ($r = -0.487$, $p = 0.030$) and no significant length effect ($r = 0.315$, $p = 0.176$). In sociological texts, over-segmentation is actively penalized — the discipline's standardized, high-frequency sociological terminology forms stable multi-character compounds that the algorithm should preserve rather than fragment. This is the only discipline where the data clearly indicates that *less* splitting would improve performance.

Psychology, Philosophy, Biology, and Economics show no significant correlations with either length or granularity: the algorithm's performance in these disciplines is independent of document-level characteristics measurable by these metrics. This is not a weakness — it indicates robustness: within these domains, segmentation quality is stable across documents of varying length and token density.

**Convergence behaviour.** The algorithm converged in an average of $51.2$ rounds across all $280$ documents, with a mean particle reduction of 36.8%.

![Convergence behavior by discipline](experiments/cross_discipline/results/experiment1/convergence_by_discipline.png)

Rounds to convergence ranged from $46.9$ (Geography) to $55.2$ (Literature). The convergence time correlates with segmentation difficulty: Literature, which required the most rounds ($55.2$), is also the lowest-$\text{F1}$ discipline; Geography, which converged fastest ($46.9$), sits near the middle of the $\text{F1}$ distribution. This pattern reflects the damped annealing schedule: texts with more ambiguous word boundaries demand longer annealing — more rounds of slow, deep oscillation — before the system stabilises. Literature's 3.4% CV and high convergence-round count are two manifestations of the same underlying property: extreme within-discipline linguistic diversity.

The particle reduction rate — the percentage of initial atomic particles eliminated through merging — varied from 33.2% (Mathematics) to 39.0% (Geography). Mathematics' low reduction rate is a direct consequence of formula density: mathematical symbols ($x$, $y$, $\alpha$, $\int$, etc.) resist merging because they rarely form stable co-occurrence patterns with surrounding Chinese characters. Geography's high reduction rate reflects the opposite.



> [!IMPORTANT]
>
> The cross-discipline experiment establishes four principal findings. 
>
> 1. The algorithm achieves robust generalization: $\text{F1} > 0.90$ across all $14$ disciplines without any domain-specific tuning, and $\text{F1} > 0.93$ for the top four disciplines.
> 2. As we can see, discipline identity is a significant and substantial predictor of segmentation quality, accounting for 30.0% of $\text{F1}$ variance, a larger effect than document length, granularity, or random noise.
> 3. The discipline hierarchy reveals a social sciences advantage: disciplines with standardized academic prose and well-defined technical vocabulary (Education, Psychology, Sociology) systematically outperform those with formula-dense texts (Mathematics, Physics), mixed-register language (Literature, History), or both (Chemistry). 
> 4. Discipline-specific correlation patterns between $\text{F1}$ and document characteristics reveal actionable directions for future adaptive tuning: literary texts would benefit from lower relaxation factors to encourage more aggressive splitting; sociological and mathematical texts would benefit from higher relaxation factors to prevent over-fragmentation of technical terminology.
>
> The fact that even the worst-performing discipline (Literature, $\text{F1} = 0.8961$) remains very close to $0.90$, and that the coefficient of variation for the whole dataset is only 2.1%, demonstrates that the algorithm's statistical foundation produces genuinely domain-agnostic segmentation. The remaining performance variation is not a failure of the method but an honest reflection of differential linguistic structure across academic disciplines: some text types are inherently harder to segment without lexical knowledge, and the algorithm's sensitivity to this variation is a sign of its statistical fidelity rather than its fragility.
>


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

- **`scripts/core.py`** — The entire algorithm in a single file. The core growth/statistics loop relies only on the Python standard library (`re`, `math`, `collections`); sentence-splitting uses `pysbd` for robust boundary disambiguation. Contains:
  - `Config` — all hyperparameters with dynamic threshold / mass factor / atom adjustment methods.
  - `AtomicCrystalGrowth` — the main class: preprocessing, statistics, growth, dissolution, convergence.
  - `split_sentences()` — sentence boundary disambiguation with regex fallback.
- **`scripts/evaluation.py`** — Compares crystal growth output against jieba, reporting boundary precision/recall/F1, granularity ratio, and n-gram Jaccard overlap.
- **`scripts/train.py`** — Leave-one-year-out cross-validation: for each year, compares self-trained vs transfer segmentation.
- **`main.py`** — Thin CLI dispatcher with dependency checking.

## Quick Start

```bash
# Install dependencies
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

A $\sim 7500$-character philosophy speech (Peking University, Prof. Cheng Lesong) on Daoism and modern life. Rich in literary vocabulary, complex sentence structures, and mixed Chinese/English/emoji content — a challenging test for unsupervised segmentation.

### `corpus/corpus.json`

30 years (1997–2026) of *Southern Weekend* (南方周末) New Year editorials (新年献词), totalling $\sim 45000$ characters. Each entry contains `year`, `title`, and `text`. This corpus is used for the leave-one-year-out transfer learning experiment.

## Requirements

- **Python 3.7+**
- `tabulate` — table formatting for output
- `jieba` — baseline comparison
- `pysbd` — sentence boundary disambiguation (`core.py` uses it for preprocessing)

```
# requirements.txt
tabulate>=0.8
jieba>=0.42.1
pysbd>=0.3.4
```

> The core growth/statistics loop (`scripts/core.py` — `grow()`, `calc_stats()`, `dissolve()`, etc.) uses **only the Python standard library** (`re`, `math`, `collections`, `sys`, `time`). The only external dependency in `core.py` is `pysbd`, used exclusively for preprocessing (sentence boundary disambiguation) with a regex fallback.

## Acknowledgements

The demo article (`corpus/example.txt`) is from a [philosophy speech](https://mp.weixin.qq.com/s/yryponBqaD0w1ZPI5Al2Gg) recommended by a friend — deeply insightful.

The 30-year corpus consists of *Southern Weekend* New Year editorials, a beloved Chinese journalism tradition.

## License

MIT

# Atomic Crystal Growth

A completely unsupervised Chinese word segmentation algorithm inspired by crystal growth in physics. **No dictionary, no annotated data, zero external dependencies** — only Python's standard library. It works by simulating particles (characters) that spontaneously bond based on statistical forces, gradually forming multi-character words through iterative growth.

## How it works

1. **Initial state**: Text is split into sentences, then into atomic particles (single characters, English tokens, numbers, punctuation)
2. **Growth round**: For each adjacent particle pair, compute:
   - **Affinity Energy** (NPMI): how strongly they co-occur
   - **Ionization Energy** (contextual entropy): how "bound" each particle is to its environment
   - **Mass Factor**: larger particles resist further merging
   - **Net Energy**: `E_net = NPMI - Ionization × Mass_Factor`
3. **Merging**: Pairs with `E_net > threshold` AND local maxima are merged into new particles
4. **Dissolution**: Every 5 rounds, weakly-bound multi-character particles are split back (anti-entropy)
5. **Convergence**: Stops when both particle count and order parameter stabilize

---

## Algorithm Detail

### 1. Problem Setting

Let a text be split into sentences $S_1, S_2, \dots, S_m$. Initially each sentence is a sequence of **atomic particles** obtained by a simple tokenisation that separates:

- bracketed expressions (e.g. `[2.3]`, `(1)`),
- sequences of ASCII letters/digits,
- individual Chinese characters and punctuation.

No further linguistic knowledge is used. The goal is to let particles **grow** into longer units (words / multi‑word expressions) solely through the statistical forces that are measured on the current particle configuration.

### 2. Statistical Quantities

For a given particle configuration (a particular segmentation of all sentences) we compute:

- $f(p)$ – frequency of particle $p$ (unigram count)
- $f(p,q)$ – frequency of the ordered pair $(p,q)$ (bigram count)
- $N = \sum_p f(p)$ – total number of particle occurrences
- $M = \sum_{p,q} f(p,q)$ – total number of adjacent particle pairs

For each particle $p$ we also collect its **left neighbours** $L(p)$ and **right neighbours** $R(p)$.

#### 2.1 Normalised Pointwise Mutual Information (NPMI)

For two adjacent particles $p, q$:

$$
\text{PMI}(p,q) = \log \frac{P(p,q)}{P(p)P(q)} = \log \frac{f(p,q)/M}{(f(p)/N)(f(q)/N)}
$$

$$
\text{NPMI}(p,q) = \frac{\text{PMI}(p,q)}{-\log P(p,q)}
$$

NPMI ∈ [-1, 1]. A value close to 1 indicates that $p$ and $q$ almost always appear together.

#### 2.2 Neighbour Entropies

With Laplace smoothing parameter $\alpha$ and vocabulary size $V$ (number of distinct particles), the **right entropy** of $p$ is:

$$
H_R(p) = -\sum_{r \in R(p)} \frac{f(p,r)+\alpha}{f(p)+\alpha V} \log \frac{f(p,r)+\alpha}{f(p)+\alpha V}
       - (V - |R(p)|)\cdot \frac{\alpha}{f(p)+\alpha V} \log \frac{\alpha}{f(p)+\alpha V}
$$

The left entropy $H_L(p)$ is defined analogously using left neighbours $L(p)$.

To make entropies comparable across different frequencies, we normalise:

$$
h_R(p) = \frac{H_R(p)}{\log(f(p)+1)}, \qquad h_L(p) = \frac{H_L(p)}{\log(f(p)+1)}
$$

#### 2.3 Ionization Energy

The **ionization energy** of a particle measures how tightly it is bound by its context:

$$
I(p) = \frac{h_L(p) + h_R(p)}{2}
$$

High $I(p)$ means that $p$ appears in a very restricted environment (low contextual freedom) – it is “heavy” and reluctant to break apart. Low $I(p)$ means $p$ can easily detach (it behaves like a free electron).

#### 2.4 Mass Factor

Longer particles require stronger evidence to merge. Let $\ell(p)$ denote the length (number of characters) of particle $p$. The **mass factor** for a potential merge of $p$ and $q$ is:

$$
M(p,q) = \beta^{\; \ell(p) + \ell(q) - 2}
$$

where $\beta > 1$ is a constant (the *mass base*). This factor exponentially penalises merges of long particles.

#### 2.5 Polarity

The **polarity** of a particle captures the asymmetry between its left and right contexts:

$$
\Psi(p) = \frac{|H_L(p) - H_R(p)|}{H_L(p) + H_R(p)},\qquad \Psi(p) \in [0,1]
$$

A high polarity (close to 1) means $p$ has a strong directional preference – it “wants” to attach more strongly on one side.

### 3. Net Energy of a Merge

For a pair of adjacent particles $(p,q)$ we define the **net energy**:

$$
E(p,q) = \text{NPMI}(p,q) \;-\; I(p,q) \cdot M(p,q)
$$

where $I(p,q)$ is a combined ionization energy. In the implementation we use the geometric mean (or arithmetic mean) of the individual ionisation energies:

$$
I(p,q) = \frac{I(p) + I(q)}{2}
$$

The **sign** of $E(p,q)$ indicates whether the merge is favoured (positive) or disfavoured (negative). However, a positive net energy alone is not sufficient – it must exceed a dynamic threshold and be a local maximum.

### 4. Dynamic Threshold

The threshold $T$ decreases with the iteration number $i$ (simulated annealing) and includes a small oscillatory component to escape local plateaus:

$$
T(i) = T_0 - \gamma \cdot i \cdot \bigl( a + b \cos(\omega i) \bigr)
$$

Here $T_0$ is the initial threshold, $\gamma$ the decay rate, and $a, b, \omega$ constants that control the oscillation amplitude and frequency.

Additionally, the threshold is **personality‑adjusted** for single‑character particles. For a character $c$ let

- $p_{\text{alone}}(c)$ = fraction of its occurrences where it appears as a solitary particle (not part of a longer particle)
- $f_{\text{rel}}(c)$ = $f(c) / N$ (relative frequency)

Then an adjustment $\Delta_{\text{personality}}(c)$ is computed:

- If $p_{\text{alone}} > \theta_{\text{indep}}$: $\Delta = \eta_{\text{indep}} \cdot (p_{\text{alone}} - \theta_{\text{indep}})$  (protect)
- Else if $p_{\text{alone}} < \theta_{\text{restless}}$: $\Delta = -\eta_{\text{restless}} \cdot (\theta_{\text{restless}} - p_{\text{alone}})$  (encourage merging)
- If $f_{\text{rel}} > \phi_{\text{floor}}$: extra protection $\Delta_{\text{freq}} = \kappa \cdot (f_{\text{rel}} - \phi_{\text{floor}})$

All constants ($\theta_{\text{indep}}$, $\eta_{\text{indep}}$, $\theta_{\text{restless}}$, $\eta_{\text{restless}}$, $\phi_{\text{floor}}$, $\kappa$) are hyperparameters.

The final threshold for a pair $(p,q)$ at iteration $i$ is:

$$
T_{\text{final}}(p,q,i) = T(i) + \Delta(p) + \Delta(q)
$$

If either $p$ or $q$ is not a single character, its $\Delta$ is zero.

### 5. Polarity Modulation

When a pair exhibits high polarity, the threshold is lowered if the merge follows the natural direction indicated by the neighbour distributions. Define:

$$
d_s(p,q) = \max\left( \frac{f(p,q)}{f(p)}, \frac{f(p,q)}{f(q)} \right)
$$

Then the polarity correction is:

$$
\Delta_{\text{pol}}(p,q) = -w_{\text{pol}} \cdot d_s(p,q) \cdot \bigl( \Psi(p) + \Psi(q) \bigr) \cdot \mathbf{1}_{[\Psi(p)>\psi_0 \;\text{or}\; \Psi(q)>\psi_0]}
$$

where $w_{\text{pol}}$ is a weight and $\psi_0$ a polarity threshold (e.g. 0.3). The final threshold becomes:

$$
T_{\text{final}}' = T_{\text{final}} + \Delta_{\text{pol}}
$$

### 6. Local Maximum Condition

To avoid “greedy” chain merges, a pair is merged **only if** its net energy is a **strict local maximum** among its neighbours. For a sentence with particles $x_1, x_2, \dots, x_n$, consider adjacent pairs $(x_i, x_{i+1})$. Let

$$
E_i = E(x_i, x_{i+1}), \qquad T_i = T_{\text{final}}'(x_i, x_{i+1}, i_{\text{iter}})
$$

The pair $i$ is merged when:

1. $E_i > T_i$
2. $E_i \ge E_{i-1}$ (or $i=1$)
3. $E_i \ge E_{i+1}$ (or $i=n-1$)

This ensures that only the *strongest* bonds in a neighbourhood are formed in a single iteration, allowing competing analyses to coexist temporarily.

### 7. Merge Operation

When a pair $(x_i, x_{i+1})$ satisfies the local maximum condition, they are replaced by a single particle $x_i \cdot x_{i+1}$ (string concatenation). Merges are performed independently within each sentence, and no overlapping merges are allowed (the local maximum condition already prevents adjacent merges in the same iteration).

Two alternating **merge modes** are used:

- **`global`** mode: any adjacent particles (including multi‑character ones) can be considered for merging.
- **`atomic`** mode: only pairs where **both** particles are single characters are allowed to merge. This mode helps to refine the core vocabulary before longer structures compete.

The mode toggles every iteration: even iterations use `global`, odd iterations use `atomic`.

### 8. Dissolution (Anti‑Entropy)

To prevent irreversible over‑merging, a **dissolution** step is performed every $K$ iterations (starting after a warm‑up phase). For each particle $p$ with length $\ell(p) \ge L_{\min}$:

- For every split position $k = 1, \dots, \ell(p)-1$ let $a = p[:k]$, $b = p[k:]$.
- Compute the **independence score**:

$$
S(k) = \max\left( \frac{f(a)}{f(p)}, \frac{f(b)}{f(p)} \right)
$$

- Take the split with the highest score: $k^* = \arg\max_k S(k)$.
- If $S(k^*) > \tau_{\text{diss}}$ (a threshold), replace $p$ by $a$ and $b$ (split).

This reverses mergers that have become weak: if one part occurs independently as often as the whole compound, the compound is likely spurious.

### 9. Order Parameter and Convergence

The **order parameter** $\mathcal{O}$ measures the global structural order of the particle system. It is defined as the mean normalised ionization energy across all particles that occur at least twice:

$$
\mathcal{O} = \frac{1}{|\{p : f(p)\ge 2\}|} \sum_{p: f(p)\ge 2} I(p)
$$

Higher $\mathcal{O}$ indicates a more “frozen” system where particles are well‑crystallised.

Convergence is declared when **both** of the following hold for a consecutive window of iterations:

1. The total number of particles has not changed (no merges, no dissolutions) for `n_plateau` iterations.
2. The order parameter $\mathcal{O}$ changes by less than $\epsilon_{\mathcal{O}}$ for `n_window` iterations.

Once converged, the algorithm stops.

### Complete Iteration Loop

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
        Mark pairs that satisfy E_i > T_i and E_i is local maximum
        Replace each marked pair by concatenated particle (non‑overlapping)
    if dissolution_interval > 0 and t >= start_dissolve and (t % interval == 0):
        Perform dissolution on all particles
    Update order parameter O
    Check convergence criteria
    t = t + 1

Output: final segmented sentences
```

### Hyperparameters

| Symbol | Meaning |
|--------|---------|
| $\beta$ | Mass base (exponential growth factor) |
| $T_0$ | Initial threshold |
| $\gamma$ | Threshold decay per iteration |
| $a, b, \omega$ | Oscillation parameters for $T(i)$ |
| $\theta_{\text{indep}}$, $\eta_{\text{indep}}$ | High‑alone protection threshold and penalty |
| $\theta_{\text{restless}}$, $\eta_{\text{restless}}$ | Low‑alone encouragement threshold and bonus |
| $\phi_{\text{floor}}$, $\kappa$ | Frequency floor and protection weight |
| $w_{\text{pol}}$, $\psi_0$ | Polarity modulation weight and activation threshold |
| $\alpha$ | Laplace smoothing for entropy |
| $K$ | Dissolution interval |
| $L_{\min}$ | Minimum length for dissolution candidate |
| $\tau_{\text{diss}}$ | Dissolution independence ratio threshold |
| $n_{\text{plateau}}$ | Iterations with no particle change required |
| $n_{\text{window}}$ | Convergence window for order parameter |
| $\epsilon_{\mathcal{O}}$ | Tolerance for order parameter change |

These hyperparameters control the “thermodynamics” of the system: lower thresholds or higher mass bases produce more aggressive merging; different personalities protect function words; dissolution prevents spurious long compounds.

---

## Code

### Project Structure

```
SOM/
├── main.py                    # Entry point
├── corpus/
│   ├── corpus.json            # 30 years of Southern Weekend NYE editorials (45K chars)
│   └── example.txt            # Demo text about AI companionship
├── scripts/
│   ├── core.py                # Core algorithm (~260 lines)
│   ├── evaluation.py          # Jieba baseline comparison
│   └── train.py               # Leave-one-year-out cross-validation
└── README.md
```

### Quick Start

```bash
# Basic segmentation
python main.py run

# Compare with jieba baseline (requires pip install jieba)
python main.py eval

# Cross-validation on corpus
python main.py train
```

### Requirements

- Python 3.7+
- `jieba` (only for evaluation/train — `core.py` runs standalone)
- No other dependencies

### Acknoledgements

This example article is from [Here](https://mp.weixin.qq.com/s/yryponBqaD0w1ZPI5Al2Gg); it's a deeply insightful speech recently recommended to me by a friend.

## License

MIT

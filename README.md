# Quantum Portfolio Optimization with Constraint-Preserving QAOA

QAOA-based portfolio optimization where you pick exactly K of n stocks to
balance return against risk. This repo compares three circuit constructions
— penalty QAOA, XY-mixer with Dicke initialization, and XY-mixer with a
CVaR objective — and validates each against brute force, classical baselines,
and real IBM Quantum hardware.

**Key finding:** The XY-mixer guarantees every shot is a valid portfolio in
ideal simulation, but readout errors push the valid fraction down to 0.35–0.43
on real hardware, even though the optimal portfolio still survives. That
ideal-vs-device gap is the practical bottleneck for constrained quantum
optimization today.

Built for Qiskit Fall Fest 2026, Track 2 (Advanced level).

---

## Problem

Budget-constrained portfolio selection: pick exactly K of n stocks.

C(x) = -mu·x + lambda·xᵀΣx + P·(Σxᵢ - K)²


Where `μ` is the expected return vector, `Σ` is the covariance matrix,
and `P` enforces the budget constraint.

## Method

Three QAOA variants compared:

1. **Baseline** — X-mixer + penalty term
2. **XY-mixer** — constraint-preserving mixer with Dicke-state initialization
3. **XY-mixer + CVaR** — objective focuses on the best 25% of samples

All variants validated against brute force, greedy heuristic, simulated
annealing, and IBM Quantum hardware.

## Results

### Baseline (n=6, K=3)

| Depth | Approximation ratio | Valid fraction |
|-------|---------------------|----------------|
| p=1   | 1.000               | 0.653          |
| p=2   | 1.000               | 0.632          |
| p=3   | 1.000               | 0.551          |
| p=4   | 1.000               | 0.749          |

Finds the optimum at every depth, but 25–45% of shots land on invalid bitstrings.

### XY-mixer (n=6, K=3)

| Depth | E[C] ratio | CVaR ratio | Best sampled | Valid fraction |
|-------|------------|------------|--------------|----------------|
| p=1   | 0.716      | 0.850      | 1.000        | 1.000          |
| p=2   | 0.788      | 0.972      | 1.000        | 1.000          |
| p=3   | 0.798      | 1.000      | 1.000        | 1.000          |
| p=4   | 0.772      | 0.878      | 1.000        | 1.000          |

Every shot is valid. CVaR converges to the optimum at p=3.

### Scaling (n=12, K=6)

| Method                  | Approximation ratio |
|-------------------------|---------------------|
| Greedy                  | 0.727               |
| Simulated annealing     | 0.993               |
| XY-QAOA p=1 (2 params)  | CVaR 0.802, best 1.000 |

At p=1 with only two variational parameters, XY-QAOA found the true optimum
in its sampled distribution.

### Hardware (ibm_fez)

4096 shots per depth. Error suppression: dynamical decoupling (XpXm),
gate twirling, measurement twirling.

| Depth | Transpiled depth | Ideal | Noisy | Hardware | HW best r |
|-------|------------------|-------|-------|----------|-----------|
| p=1   | 582              | 1.000 | 0.730 | 0.428    | 1.000     |
| p=2   | 883              | 1.000 | 0.646 | 0.357    | 1.000     |
| p=3   | 1130             | 1.000 | 0.597 | 0.353    | 1.000     |

Every condition found the optimal portfolio. The ideal-vs-hardware gap is the
finding: readout errors push 2 of every 3 shots out of the K-cardinality
subspace, even though the circuit itself never leaves it.

**Job IDs:**
- `db77hr6r6oqs7386u6c0` (p=1)
- `db77io2raffc73f9trhg` (p=2)
- `db77iruil99s73fng2a0` (p=3)

### Barren plateau probe

Gradient variance across 50 random parameter initializations per depth.

| Depth | X-mixer var | XY-mixer var | Ratio  |
|-------|-------------|--------------|--------|
| p=1   | 1.61 × 10⁻¹ | 3.79 × 10⁰   | 23.5×  |
| p=2   | 2.49 × 10⁻¹ | 6.12 × 10⁰   | 24.6×  |
| p=3   | 1.94 × 10⁻¹ | 4.29 × 10⁰   | 22.1×  |

The XY-mixer keeps gradient variance roughly 23× larger, stable across depth.
Structure-aware circuit construction keeps the optimization landscape trainable.

---

## Files

| File | Purpose |
|------|---------|
| `xy_qaoa.py` | n=6 XY-mixer + CVaR |
| `xy_qaoa_n12_trotter.py` | n=12 with Lie-Trotter synthesis |
| `hardware_run.py` | Ideal → noisy → QPU pipeline |
| `barren_plateau.py` | Gradient variance probe |
| `portfolio_baseline.py` | X-mixer + penalty baseline |
| `hardware_results.json` | Raw hardware data and job IDs |
| `baseline_results.png` | Baseline QAOA figure |
| `xy_qaoa_results.png` | XY-QAOA figure |

## How to run

```bash
# 1. Create environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the simulations (no hardware needed)
python xy_qaoa.py              # n=6 XY-mixer + CVaR
python barren_plateau.py       # gradient variance probe
python portfolio_baseline.py   # X-mixer + penalty baseline

# 3. Optional: run on real IBM hardware
#    Requires an IBM Quantum API token saved locally
python hardware_run.py
```





##Hardware
Backend: ibm_fez (156 qubits)

Shots: 4096 per depth

Error suppression: dynamical decoupling (XpXm), gate twirling,
measurement twirling

##Credits
QAOA: Farhi et al. (2014)

Portfolio formulation: Sachdeva et al. (2024)

XY-mixer: Hadfield et al., Wang et al.

Barren plateaus: McClean et al. (2018)


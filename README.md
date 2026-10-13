# Quantum Portfolio Optimization with Constraint-Preserving QAOA

# Quantum Portfolio Optimization with Constraint-Preserving QAOA

QAOA-based portfolio optimization where you pick exactly K of n stocks to balance return against risk. This repo compares three circuit constructions — penalty QAOA, XY-mixer with Dicke initialization, and XY-mixer with a CVaR objective — and validates each against brute force, classical baselines (greedy, simulated annealing), and real IBM Quantum hardware.

**Key finding:** The XY-mixer guarantees every shot is a valid portfolio in ideal simulation, but readout errors push the valid fraction down to 0.35–0.43 on real hardware even though the optimal portfolio still survives. That ideal-vs-device gap is the practical bottleneck for constraint-constrained quantum optimization today.

Built for Qiskit Fall Fest 2026, Track 2. Advanced level.

## Problem
Budget-constrained portfolio selection: pick exactly K of n stocks.
Cost: C(x) = -mu.x + lambda.x^T.Sigma.x + P.(sum(x_i)-K)^2

## Method
Three QAOA variants compared:
1. X-mixer + penalty (baseline)
2. XY-mixer + Dicke state
3. XY-mixer + CVaR objective

Validated against brute force, greedy, simulated annealing, IBM Quantum hardware.

## Results

Baseline n=6: valid fraction 0.55-0.75 across p=1..4.
XY-mixer n=6: valid fraction 1.000, best r=1.000 at all depths.
CVaR converges to optimum at p=3.
n=12 K=6: greedy 0.727, SA 0.993, XY-QAOA p=1 (2 params) best 1.000.
Hardware ibm_fez: valid 0.35-0.43, best r=1.000 at p=1,2,3.
Barren plateau: XY-mixer gradient variance 23x larger than X-mixer.

## Files
xy_qaoa.py - n=6 XY-mixer + CVaR
xy_qaoa_n12_trotter.py - n=12 with Lie-Trotter synthesis
hardware_run.py - ideal/noisy/QPU pipeline
barren_plateau.py - gradient variance probe
portfolio_baseline.py - X-mixer + penalty baseline
hardware_results.json - raw hardware data
baseline_results.png, xy_qaoa_results.png - figures


## Hardware job IDs
db77hr6r6oqs7386u6c0
db77io2raffc73f9trhg
db77iruil99s73fng2a0

## Credits
QAOA: Farhi et al. 2014. Portfolio: Sachdeva et al. 2024.
XY-mixer: Hadfield et al., Wang et al.
Barren plateaus: McClean et al. 2018.

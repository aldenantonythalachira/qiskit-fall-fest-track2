"""
v2: XY-mixer + Dicke state + CVaR for portfolio optimization (n=6, K=3).
Compares expected-value vs CVaR objective on constraint-preserving QAOA.
"""
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import to_ising

np.random.seed(42)

# -----------------------------
# Problem
# -----------------------------
n, K = 6, 3
lambda_risk = 0.5

mu = np.random.uniform(0.05, 0.25, size=n)
A = np.random.randn(n, n)
Sigma = A @ A.T / n + 1e-3 * np.eye(n)

all_x = np.array([[(i >> k) & 1 for k in range(n)] for i in range(2**n)])
def cost(x): return float(-mu @ x + lambda_risk * x @ Sigma @ x)
all_costs = np.array([cost(x) for x in all_x])
valid_mask = all_x.sum(axis=1) == K
feasible_costs = all_costs[valid_mask]
Cmin_f = feasible_costs.min()
Cmax_f = feasible_costs.max()
best_x = all_x[valid_mask][np.argmin(feasible_costs)]
print(f"Feasible states: {valid_mask.sum()}")
print(f"Cmin={Cmin_f:.4f}  Cmax={Cmax_f:.4f}")
print(f"Optimal x = {best_x}, cost = {cost(best_x):.4f}")

# -----------------------------
# Ising operator for pure portfolio objective (NO penalty)
# -----------------------------
qp = QuadraticProgram()
for i in range(n):
    qp.binary_var(f"x{i}")
qp.minimize(
    linear={f"x{i}": -float(mu[i]) + float(lambda_risk * Sigma[i, i]) for i in range(n)},
    quadratic={(f"x{i}", f"x{j}"): float(2 * lambda_risk * Sigma[i, j])
               for i in range(n) for j in range(i + 1, n)},
    constant=0.0,
)
cost_op, offset = to_ising(qp)
print(f"Cost Ising terms: {len(cost_op)}")

# -----------------------------
# XY mixer: H_M = sum_{i<j} (X_i X_j + Y_i Y_j)
# -----------------------------
def xy_mixer(nq):
    terms = []
    for i in range(nq):
        for j in range(i + 1, nq):
            sx = ["I"] * nq; sx[nq - 1 - i] = "X"; sx[nq - 1 - j] = "X"
            terms.append(("".join(sx), 1.0))
            sy = ["I"] * nq; sy[nq - 1 - i] = "Y"; sy[nq - 1 - j] = "Y"
            terms.append(("".join(sy), 1.0))
    return SparsePauliOp.from_list(terms)

mixer_op = xy_mixer(n)
print(f"Mixer terms: {len(mixer_op)}")

# -----------------------------
# Dicke state |D^n_K>: uniform superposition of K-cardinality states
# -----------------------------
dicke_vec = np.zeros(2**n)
for i in range(2**n):
    if all_x[i].sum() == K:
        dicke_vec[i] = 1.0
dicke_vec /= np.linalg.norm(dicke_vec)
print(f"Dicke state norm: {np.linalg.norm(dicke_vec):.4f}")

# -----------------------------
# Circuit builders
# -----------------------------
def build_circuit(c_op, m_op, reps, init_vec):
    nq = c_op.num_qubits
    gammas = ParameterVector("g", reps)
    betas = ParameterVector("b", reps)
    qc = QuantumCircuit(nq)
    qc.initialize(init_vec, range(nq))
    for L in range(reps):
        qc.append(PauliEvolutionGate(c_op, time=gammas[L]), range(nq))
        qc.append(PauliEvolutionGate(m_op, time=betas[L]), range(nq))
    return qc

def get_probs(params, c_op, m_op, reps, init_vec):
    qc = build_circuit(c_op, m_op, reps, init_vec)
    bound = qc.assign_parameters(params)
    return Statevector(bound).probabilities()

def expected_cost(probs, costs):
    return float(np.sum(probs * costs))

def cvar_cost(probs, costs, alpha=0.25):
    idx = np.argsort(costs)
    sp = probs[idx]; sc = costs[idx]
    remaining = alpha; val = 0.0
    for p, c in zip(sp, sc):
        if remaining <= 0:
            break
        take = min(remaining, p)
        val += take * c
        remaining -= take
    return val / alpha

# -----------------------------
# Run XY-QAOA for p = 1..4
# -----------------------------
results = {}
for p in [1, 2, 3, 4]:
    t0 = time.time()
    qc = build_circuit(cost_op, mixer_op, p, dicke_vec)
    x0 = np.random.uniform(0, np.pi, qc.num_parameters)

    def obj(params):
        probs = get_probs(params, cost_op, mixer_op, p, dicke_vec)
        return expected_cost(probs, all_costs)

    res = minimize(obj, x0, method="COBYLA",
                   options={"maxiter": 300, "rhobeg": 0.5})
    probs = get_probs(res.x, cost_op, mixer_op, p, dicke_vec)

    exp_c = expected_cost(probs, all_costs)
    cv_c = cvar_cost(probs, all_costs, alpha=0.25)

    best_sampled_cost, best_sampled_x, valid_prob = np.inf, None, 0.0
    for i in range(2**n):
        if probs[i] < 1e-8:
            continue
        valid_prob += probs[i]
        if all_costs[i] < best_sampled_cost:
            best_sampled_cost = all_costs[i]
            best_sampled_x = all_x[i]

    r_exp = (Cmax_f - exp_c) / (Cmax_f - Cmin_f)
    r_cvar = (Cmax_f - cv_c) / (Cmax_f - Cmin_f)
    r_best = (Cmax_f - best_sampled_cost) / (Cmax_f - Cmin_f)

    dt = time.time() - t0
    results[p] = dict(exp=exp_c, cv=cv_c, best=best_sampled_cost,
                      r_exp=r_exp, r_cvar=r_cvar, r_best=r_best,
                      valid=valid_prob, best_x=best_sampled_x, time=dt)

    print(f"p={p}  [{dt:.1f}s]  valid_frac={valid_prob:.4f}  "
          f"E[C]={exp_c:.4f} (r={r_exp:.4f})  "
          f"CVaR={cv_c:.4f} (r={r_cvar:.4f})  "
          f"best={best_sampled_cost:.4f} (r={r_best:.4f})  x={best_sampled_x}")

# -----------------------------
# Plot
# -----------------------------
ps = sorted(results.keys())
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(ps, [results[p]["r_exp"] for p in ps], "o-", label="E[C]")
ax[0].plot(ps, [results[p]["r_cvar"] for p in ps], "s-", label="CVaR(0.25)")
ax[0].plot(ps, [results[p]["r_best"] for p in ps], "^-", label="best sampled")
ax[0].axhline(1.0, color="r", ls="--", label="optimal")
ax[0].set_xlabel("QAOA depth p")
ax[0].set_ylabel("Feasible approximation ratio")
ax[0].set_title("XY-QAOA on portfolio (n=6, K=3)")
ax[0].legend(); ax[0].grid(alpha=0.3)

ax[1].bar([str(p) for p in ps], [results[p]["valid"] for p in ps], color="steelblue")
ax[1].axhline(1.0, color="r", ls="--", label="all valid")
ax[1].set_xlabel("QAOA depth p")
ax[1].set_ylabel("Valid fraction")
ax[1].set_title("Constraint satisfaction (XY-mixer)")
ax[1].set_ylim(0, 1.1); ax[1].legend()
plt.tight_layout()
plt.savefig("xy_qaoa_results.png", dpi=120)
print("Saved xy_qaoa_results.png")

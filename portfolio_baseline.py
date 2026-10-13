
"""
Baseline: n=6, K=3 portfolio selection with QAOA (X-mixer + penalty).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Statevector
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import to_ising

np.random.seed(42)

# -----------------------------
# Problem setup
# -----------------------------
n, K = 6, 3
lambda_risk = 0.5
P = 5.0

mu = np.random.uniform(0.05, 0.25, size=n)
A = np.random.randn(n, n)
Sigma = A @ A.T / n + 1e-3 * np.eye(n)

def portfolio_cost(x):
    x = np.asarray(x, dtype=float)
    return float(-mu @ x + lambda_risk * x @ Sigma @ x + P * (x.sum() - K) ** 2)

# -----------------------------
# QUBO -> Ising
# -----------------------------
lin = -mu + lambda_risk * np.diag(Sigma) + P * (1 - 2 * K)
quad = {}
for i in range(n):
    for j in range(i + 1, n):
        quad[(i, j)] = 2 * lambda_risk * Sigma[i, j] + 2 * P
const = P * K ** 2

qp = QuadraticProgram()
for i in range(n):
    qp.binary_var(f"x{i}")
qp.minimize(
    linear={f"x{i}": float(lin[i]) for i in range(n)},
    quadratic={(f"x{i}", f"x{j}"): float(quad[(i, j)]) for (i, j) in quad},
    constant=float(const),
)
conv = QuadraticProgramToQubo()
qubo = conv.convert(qp)
ising_op, offset = to_ising(qubo)
print(f"Ising terms: {len(ising_op)}")

# -----------------------------
# Brute force reference
# -----------------------------
all_x = np.array([[(i >> k) & 1 for k in range(n)] for i in range(2 ** n)])
all_costs = np.array([portfolio_cost(x) for x in all_x])
valid_mask = all_x.sum(axis=1) == K
valid_costs = all_costs[valid_mask]
best_valid_idx = np.where(valid_mask)[0][np.argmin(valid_costs)]
C_min = valid_costs.min()
C_max = all_costs.max()
print(f"Brute force best x = {all_x[best_valid_idx]}, cost = {C_min:.4f}")
print(f"C_max = {C_max:.4f}, C_min(valid) = {C_min:.4f}")

# -----------------------------
# QAOA circuit
# -----------------------------
def build_qaoa(cost_op, reps):
    nq = cost_op.num_qubits
    gammas = ParameterVector("g", reps)
    betas = ParameterVector("b", reps)
    qc = QuantumCircuit(nq)
    qc.h(range(nq))
    for L in range(reps):
        qc.append(PauliEvolutionGate(cost_op, time=gammas[L]), range(nq))
        for q in range(nq):
            qc.rx(2 * betas[L], q)
    return qc

def expected_cost(params, cost_op, reps):
    qc = build_qaoa(cost_op, reps)
    bound = qc.assign_parameters(params)
    sv = Statevector(bound)
    probs = sv.probabilities()
    return float(sum(probs[i] * portfolio_cost(all_x[i]) for i in range(2 ** n)))

# -----------------------------
# Run QAOA for p = 1..4
# -----------------------------
results = {}
for p in [1, 2, 3, 4]:
    qc = build_qaoa(ising_op, p)
    x0 = np.random.uniform(0, np.pi, qc.num_parameters)
    res = minimize(
        lambda v: expected_cost(v, ising_op, p),
        x0, method="COBYLA",
        options={"maxiter": 300, "rhobeg": 0.5},
    )
    bound = qc.assign_parameters(res.x)
    probs = Statevector(bound).probabilities()

    best_cost, best_x, valid_prob = np.inf, None, 0.0
    for i in range(2 ** n):
        if probs[i] < 1e-8:
            continue
        c = portfolio_cost(all_x[i])
        if c < best_cost:
            best_cost, best_x = c, all_x[i]
        if all_x[i].sum() == K:
            valid_prob += probs[i]

    r = (C_max - best_cost) / (C_max - C_min)
    results[p] = dict(r=r, cost=best_cost, x=best_x, valid=valid_prob)
    print(f"p={p}: r={r:.4f}  cost={best_cost:.4f}  valid_frac={valid_prob:.4f}  x={best_x}")

# -----------------------------
# Plot
# -----------------------------
ps = sorted(results.keys())
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(ps, [results[p]["r"] for p in ps], "o-")
ax[0].axhline(1.0, color="r", ls="--", label="perfect")
ax[0].set_xlabel("QAOA depth p"); ax[0].set_ylabel("Approximation ratio")
ax[0].set_title("Approximation ratio vs depth"); ax[0].legend()
ax[1].plot(ps, [results[p]["valid"] for p in ps], "s-", color="orange")
ax[1].set_xlabel("QAOA depth p"); ax[1].set_ylabel("Valid fraction")
ax[1].set_title("Budget constraint satisfaction")
plt.tight_layout()
plt.savefig("baseline_results.png", dpi=120)
print("Saved baseline_results.png")


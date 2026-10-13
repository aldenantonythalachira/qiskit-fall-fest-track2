"""
XY-QAOA on n=12, K=6 portfolio + classical baselines.
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

np.random.seed(7)

n, K = 12, 6
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
optimal_x = all_x[valid_mask][np.argmin(feasible_costs)]
print(f"Total states: {2**n}   Feasible: {valid_mask.sum()}")
print(f"Cmin={Cmin_f:.4f}  Cmax={Cmax_f:.4f}")
print(f"Optimal x = {optimal_x}")

# ---- classical baselines ----
def greedy(mu, Sigma, K):
    order = np.argsort(-mu / np.sqrt(np.diag(Sigma)))
    x = np.zeros(len(mu), dtype=int)
    x[order[:K]] = 1
    return x

def sa(mu, Sigma, K, iters=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(len(mu), dtype=int)
    x[rng.choice(len(mu), K, replace=False)] = 1
    def c(x): return -mu @ x + lambda_risk * x @ Sigma @ x
    cur = c(x)
    for _ in range(iters):
        i_on = np.where(x == 1)[0]
        i_off = np.where(x == 0)[0]
        if len(i_on) == 0 or len(i_off) == 0: break
        i = rng.choice(i_on); j = rng.choice(i_off)
        x[i], x[j] = 0, 1
        new = c(x)
        if new < cur:
            cur = new
        else:
            x[i], x[j] = 1, 0
    return x

g_x = greedy(mu, Sigma, K)
s_x = sa(mu, Sigma, K)
g_cost = cost(g_x); s_cost = cost(s_x)
print(f"Greedy  cost={g_cost:.4f}  r={(Cmax_f-g_cost)/(Cmax_f-Cmin_f):.4f}  x={g_x}")
print(f"SA      cost={s_cost:.4f}  r={(Cmax_f-s_cost)/(Cmax_f-Cmin_f):.4f}  x={s_x}")

# ---- Ising ----
qp = QuadraticProgram()
for i in range(n): qp.binary_var(f"x{i}")
qp.minimize(
    linear={f"x{i}": -float(mu[i]) + float(lambda_risk * Sigma[i, i]) for i in range(n)},
    quadratic={(f"x{i}", f"x{j}"): float(2 * lambda_risk * Sigma[i, j])
               for i in range(n) for j in range(i + 1, n)},
    constant=0.0,
)
cost_op, _ = to_ising(qp)
print(f"Cost Ising terms: {len(cost_op)}")

# ---- XY mixer ----
def xy_mixer(nq):
    terms = []
    for i in range(nq):
        for j in range(i + 1, nq):
            sx = ["I"] * nq; sx[nq-1-i] = "X"; sx[nq-1-j] = "X"
            terms.append(("".join(sx), 1.0))
            sy = ["I"] * nq; sy[nq-1-i] = "Y"; sy[nq-1-j] = "Y"
            terms.append(("".join(sy), 1.0))
    return SparsePauliOp.from_list(terms)

mixer_op = xy_mixer(n)
print(f"Mixer terms: {len(mixer_op)}")

# ---- Dicke |D^12_6> ----
dicke = np.zeros(2**n)
for i in range(2**n):
    if all_x[i].sum() == K:
        dicke[i] = 1.0
dicke /= np.linalg.norm(dicke)
print(f"Dicke norm: {np.linalg.norm(dicke):.4f}")

def build(c_op, m_op, reps, init):
    nq = c_op.num_qubits
    g = ParameterVector("g", reps); b = ParameterVector("b", reps)
    qc = QuantumCircuit(nq)
    qc.initialize(init, range(nq))
    for L in range(reps):
        qc.append(PauliEvolutionGate(c_op, time=g[L]), range(nq))
        qc.append(PauliEvolutionGate(m_op, time=b[L]), range(nq))
    return qc

def probs_of(params, c_op, m_op, reps, init):
    qc = build(c_op, m_op, reps, init)
    return Statevector(qc.assign_parameters(params)).probabilities()

def cvar(probs, costs, alpha=0.25):
    idx = np.argsort(costs); sp = probs[idx]; sc = costs[idx]
    rem = alpha; val = 0.0
    for p, c in zip(sp, sc):
        if rem <= 0: break
        t = min(rem, p); val += t * c; rem -= t
    return val / alpha

results = {}
for p in [1, 2, 3]:
    t0 = time.time()
    qc = build(cost_op, mixer_op, p, dicke)
    x0 = np.random.uniform(0, np.pi, qc.num_parameters)
    def obj(v):
        pb = probs_of(v, cost_op, mixer_op, p, dicke)
        return float(np.sum(pb * all_costs))
    res = minimize(obj, x0, method="COBYLA",
                   options={"maxiter": 100, "rhobeg": 0.4, "tol": 1e-3})
    pb = probs_of(res.x, cost_op, mixer_op, p, dicke)
    exp_c = float(np.sum(pb * all_costs))
    cv_c  = cvar(pb, all_costs, 0.25)
    best_c = float(min(all_costs[i] for i in range(2**n) if pb[i] > 1e-8))
    valid_p = float(sum(pb[i] for i in range(2**n) if all_x[i].sum() == K))
    r_exp = (Cmax_f - exp_c) / (Cmax_f - Cmin_f)
    r_cv  = (Cmax_f - cv_c)  / (Cmax_f - Cmin_f)
    r_bst = (Cmax_f - best_c)/ (Cmax_f - Cmin_f)
    dt = time.time() - t0
    results[p] = dict(r_exp=r_exp, r_cv=r_cv, r_bst=r_bst,
                      exp=exp_c, cv=cv_c, bst=best_c, valid=valid_p, t=dt)
    print(f"p={p} [{dt:.0f}s] valid={valid_p:.4f} "
          f"E[C]r={r_exp:.4f} CVaRr={r_cv:.4f} bestr={r_bst:.4f}")

# ---- plot ----
ps = sorted(results)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].axhline(1.0, color="r", ls="--", label="optimal")
ax[0].axhline((Cmax_f-g_cost)/(Cmax_f-Cmin_f), color="g", ls=":", label=f"greedy r={(Cmax_f-g_cost)/(Cmax_f-Cmin_f):.3f}")
ax[0].axhline((Cmax_f-s_cost)/(Cmax_f-Cmin_f), color="b", ls=":", label=f"SA r={(Cmax_f-s_cost)/(Cmax_f-Cmin_f):.3f}")
ax[0].plot(ps, [results[p]["r_exp"] for p in ps], "o-", label="E[C]")
ax[0].plot(ps, [results[p]["r_cv"]  for p in ps], "s-", label="CVaR(0.25)")
ax[0].plot(ps, [results[p]["r_bst"] for p in ps], "^-", label="best sampled")
ax[0].set_xlabel("QAOA depth p"); ax[0].set_ylabel("Feasible ratio")
ax[0].set_title("XY-QAOA n=12, K=6 vs classical"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[1].bar([str(p) for p in ps], [results[p]["valid"] for p in ps], color="steelblue")
ax[1].set_ylim(0, 1.1); ax[1].set_xlabel("p"); ax[1].set_ylabel("Valid fraction")
ax[1].set_title("Constraint satisfaction"); ax[1].axhline(1.0, color="r", ls="--")
plt.tight_layout(); plt.savefig("xy_qaoa_n12.png", dpi=120)
print("Saved xy_qaoa_n12.png")

"""
Fast n=12 XY-QAOA. Pre-built circuit, Statevector, reduced maxiter.
"""
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import to_ising

np.random.seed(7)
n, K, lam = 12, 6, 0.5

mu = np.random.uniform(0.05, 0.25, size=n)
A = np.random.randn(n, n)
Sigma = A @ A.T / n + 1e-3 * np.eye(n)

def cost_of(x):
    x = np.atleast_1d(np.asarray(x, dtype=float))
    return float(-mu @ x + lam * x @ Sigma @ x)

all_x = np.array([[(i >> k) & 1 for k in range(n)] for i in range(2**n)], dtype=int)
all_costs = np.array([cost_of(x) for x in all_x])
valid = all_x.sum(axis=1) == K
Cmin = float(all_costs[valid].min())
Cmax = float(all_costs[valid].max())
opt_x = all_x[valid][int(np.argmin(all_costs[valid]))]
print(f"Feasible: {int(valid.sum())}  Cmin={Cmin:.4f}  Cmax={Cmax:.4f}")
print(f"Optimal x = {opt_x}  cost = {cost_of(opt_x):.4f}")

# ---------------- classical baselines ----------------
def greedy():
    o = np.argsort(-mu / np.sqrt(np.diag(Sigma)))
    x = np.zeros(n, dtype=int)
    x[o[:K]] = 1
    return x

def sa(iters=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n, dtype=int)
    for idx in rng.choice(n, K, replace=False):
        x[int(idx)] = 1
    cur = cost_of(x)
    for _ in range(iters):
        on  = np.where(x == 1)[0]
        off = np.where(x == 0)[0]
        if len(on) == 0 or len(off) == 0:
            break
        i = int(rng.choice(on))
        j = int(rng.choice(off))
        x[i], x[j] = 0, 1
        new = cost_of(x)
        if new < cur:
            cur = new
        else:
            x[i], x[j] = 1, 0
    return x

g_x = greedy()
s_x = sa()
g_cost = cost_of(g_x)
s_cost = cost_of(s_x)
r_g = (Cmax - g_cost) / (Cmax - Cmin)
r_s = (Cmax - s_cost) / (Cmax - Cmin)
print(f"Greedy r={r_g:.4f}  cost={g_cost:.4f}")
print(f"SA     r={r_s:.4f}  cost={s_cost:.4f}")

# ---------------- Ising ----------------
qp = QuadraticProgram()
for i in range(n):
    qp.binary_var(f"x{i}")
qp.minimize(
    linear={f"x{i}": -float(mu[i]) + float(lam * Sigma[i, i]) for i in range(n)},
    quadratic={(f"x{i}", f"x{j}"): float(2 * lam * Sigma[i, j])
               for i in range(n) for j in range(i + 1, n)},
    constant=0.0,
)
cost_op, _ = to_ising(qp)
print(f"Cost Ising terms: {len(cost_op)}")

# ---------------- XY mixer ----------------
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

# ---------------- Dicke |D^12_6> ----------------
dicke = np.zeros(2**n)
for i in range(2**n):
    if all_x[i].sum() == K:
        dicke[i] = 1.0
dicke /= np.linalg.norm(dicke)
print(f"Dicke norm: {np.linalg.norm(dicke):.4f}")

# ---------------- pre-build circuit ----------------
def build_param_circuit(reps):
    g = ParameterVector("g", reps)
    b = ParameterVector("b", reps)
    qc = QuantumCircuit(n)
    qc.initialize(dicke, range(n))
    for L in range(reps):
        qc.append(PauliEvolutionGate(cost_op, time=g[L]), range(n))
        qc.append(PauliEvolutionGate(mixer_op, time=b[L]), range(n))
    return qc

def cvar(pb, costs, alpha=0.25):
    idx = np.argsort(costs)
    sp, sc = pb[idx], costs[idx]
    rem, val = alpha, 0.0
    for p, c in zip(sp, sc):
        if rem <= 0:
            break
        t = min(rem, p)
        val += t * c
        rem -= t
    return val / alpha

results = {}
for p in [1, 2]:
    print(f"\n=== p={p} ===", flush=True)
    t0 = time.time()
    qc = build_param_circuit(p)
    nparams = 2 * p
    print(f"  built circuit ({time.time()-t0:.1f}s), {nparams} params", flush=True)

    def obj(v):
        bound = qc.assign_parameters(v)
        sv = Statevector(bound)
        return float(np.sum(np.abs(sv.data) ** 2 * all_costs))

    x0 = np.random.uniform(0, np.pi, nparams)
    res = minimize(obj, x0, method="COBYLA",
                   options={"maxiter": 60, "rhobeg": 0.5, "tol": 1e-2})
    print(f"  COBYLA done ({time.time()-t0:.1f}s, nfev={res.nfev})", flush=True)

    sv = Statevector(qc.assign_parameters(res.x))
    pb = np.abs(sv.data) ** 2
    exp_c = float(np.sum(pb * all_costs))
    cv_c = cvar(pb, all_costs, 0.25)
    best_c = float(min(all_costs[i] for i in range(2**n) if pb[i] > 1e-8))
    valid_p = float(sum(pb[i] for i in range(2**n) if valid[i]))

    r_e = (Cmax - exp_c) / (Cmax - Cmin)
    r_c = (Cmax - cv_c) / (Cmax - Cmin)
    r_b = (Cmax - best_c) / (Cmax - Cmin)
    results[p] = dict(r_e=r_e, r_c=r_c, r_b=r_b, v=valid_p, t=time.time() - t0)
    print(f"  p={p}: valid={valid_p:.4f}  E[C]r={r_e:.4f}  CVaRr={r_c:.4f}  bestr={r_b:.4f}", flush=True)

print("\n=== SUMMARY ===")
print(f"Greedy r = {r_g:.4f}")
print(f"SA     r = {r_s:.4f}")
for p in sorted(results):
    r = results[p]
    print(f"p={p}: E[C]={r['r_e']:.4f}  CVaR={r['r_c']:.4f}  best={r['r_b']:.4f}  valid={r['v']:.4f}  time={r['t']:.0f}s")

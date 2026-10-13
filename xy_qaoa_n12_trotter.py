"""n=12 XY-QAOA with LieTrotter synthesis — ~50x faster."""
import warnings; warnings.filterwarnings("ignore")
import time, numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import to_ising

np.random.seed(7)
n, K, lam = 12, 6, 0.5
mu = np.random.uniform(0.05, 0.25, size=n)
A = np.random.randn(n, n); Sigma = A @ A.T / n + 1e-3*np.eye(n)

all_x = np.array([[(i>>k)&1 for k in range(n)] for i in range(2**n)], dtype=int)
all_costs = np.array([-mu@x + lam*x@Sigma@x for x in all_x])
valid = all_x.sum(axis=1) == K
Cmin, Cmax = float(all_costs[valid].min()), float(all_costs[valid].max())

qp = QuadraticProgram()
for i in range(n): qp.binary_var(f"x{i}")
qp.minimize(linear={f"x{i}": -float(mu[i])+float(lam*Sigma[i,i]) for i in range(n)},
            quadratic={(f"x{i}",f"x{j}"): float(2*lam*Sigma[i,j])
                       for i in range(n) for j in range(i+1,n)}, constant=0.0)
cost_op, _ = to_ising(qp)

def xy_mixer(nq):
    terms = []
    for i in range(nq):
        for j in range(i+1, nq):
            sx = ["I"]*nq; sx[nq-1-i]="X"; sx[nq-1-j]="X"; terms.append(("".join(sx),1.0))
            sy = ["I"]*nq; sy[nq-1-i]="Y"; sy[nq-1-j]="Y"; terms.append(("".join(sy),1.0))
    return SparsePauliOp.from_list(terms)
mixer_op = xy_mixer(n)

dicke = np.zeros(2**n)
for i in range(2**n):
    if all_x[i].sum() == K: dicke[i] = 1.0
dicke /= np.linalg.norm(dicke)

# KEY CHANGE: LieTrotter synthesis forces gate-level decomposition at build time
lt = LieTrotter(reps=1)
def build(reps):
    g = ParameterVector("g", reps); b = ParameterVector("b", reps)
    qc = QuantumCircuit(n)
    qc.initialize(dicke, range(n))
    for L in range(reps):
        qc.append(PauliEvolutionGate(cost_op, time=g[L], synthesis=lt), range(n))
        qc.append(PauliEvolutionGate(mixer_op, time=b[L], synthesis=lt), range(n))
    return qc

def cvar(pb, costs, alpha=0.25):
    idx = np.argsort(costs); sp, sc = pb[idx], costs[idx]
    rem, val = alpha, 0.0
    for p, c in zip(sp, sc):
        if rem <= 0: break
        t = min(rem, p); val += t*c; rem -= t
    return val/alpha

for p in [1, 2]:
    print(f"\n=== p={p} ===", flush=True)
    t0 = time.time()
    qc = build(p)
    # Assign parameters symbolically once -> the bound circuit has only numeric gates
    def obj(v):
        bound = qc.assign_parameters(v)
        sv = Statevector(bound)
        return float(np.sum(np.abs(sv.data)**2 * all_costs))

    x0 = np.random.uniform(0, np.pi, 2*p)
    res = minimize(obj, x0, method="COBYLA",
                   options={"maxiter": 60, "rhobeg": 0.5, "tol": 1e-2})
    print(f"  done ({time.time()-t0:.1f}s, nfev={res.nfev})", flush=True)

    sv = Statevector(qc.assign_parameters(res.x))
    pb = np.abs(sv.data)**2
    exp_c = float(np.sum(pb * all_costs))
    cv_c  = cvar(pb, all_costs, 0.25)
    best_c = float(min(all_costs[i] for i in range(2**n) if pb[i] > 1e-8))
    valid_p = float(sum(pb[i] for i in range(2**n) if valid[i]))
    r_e = (Cmax-exp_c)/(Cmax-Cmin); r_c = (Cmax-cv_c)/(Cmax-Cmin); r_b = (Cmax-best_c)/(Cmax-Cmin)
    print(f"  p={p}: valid={valid_p:.4f}  E[C]r={r_e:.4f}  CVaRr={r_c:.4f}  bestr={r_b:.4f}", flush=True)

"""
Hardware validation: XY-QAOA on IBM Quantum.
Runs ideal sim -> noisy sim -> real QPU with error mitigation.
Target: n=6, K=3, depths p=1,2,3.
"""
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import to_ising

# ============================================================
# CONFIG
# ============================================================
BACKEND_NAME = "ibm_fez"        # change to ibm_kingston / ibm_marrakesh if busy
SHOTS = 4096
DEPTHS = [1, 2, 3]
n, K, lam = 6, 3, 0.5
np.random.seed(42)

# ============================================================
# PROBLEM
# ============================================================
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
print(f"Feasible: {int(valid.sum())}  Cmin={Cmin:.4f}  Cmax={Cmax:.4f}")

# Ising
qp = QuadraticProgram()
for i in range(n): qp.binary_var(f"x{i}")
qp.minimize(
    linear={f"x{i}": -float(mu[i]) + float(lam*Sigma[i,i]) for i in range(n)},
    quadratic={(f"x{i}", f"x{j}"): float(2*lam*Sigma[i,j])
               for i in range(n) for j in range(i+1,n)},
    constant=0.0)
cost_op, _ = to_ising(qp)

# XY mixer
def xy_mixer(nq):
    terms = []
    for i in range(nq):
        for j in range(i+1, nq):
            sx = ["I"]*nq; sx[nq-1-i]="X"; sx[nq-1-j]="X"; terms.append(("".join(sx),1.0))
            sy = ["I"]*nq; sy[nq-1-i]="Y"; sy[nq-1-j]="Y"; terms.append(("".join(sy),1.0))
    return SparsePauliOp.from_list(terms)
mixer_op = xy_mixer(n)

# Dicke
dicke = np.zeros(2**n)
for i in range(2**n):
    if all_x[i].sum() == K: dicke[i] = 1.0
dicke /= np.linalg.norm(dicke)

# ============================================================
# BUILD + OPTIMIZE (ideal)
# ============================================================
def build_param_circuit(reps, measure=False):
    g = ParameterVector("g", reps); b = ParameterVector("b", reps)
    qc = QuantumCircuit(n)
    qc.initialize(dicke, range(n))
    for L in range(reps):
        qc.append(PauliEvolutionGate(cost_op, time=g[L]), range(n))
        qc.append(PauliEvolutionGate(mixer_op, time=b[L]), range(n))
    if measure: qc.measure_all()
    return qc

def cvar(pb, costs, alpha=0.25):
    idx = np.argsort(costs); sp, sc = pb[idx], costs[idx]
    rem, val = alpha, 0.0
    for p, c in zip(sp, sc):
        if rem <= 0: break
        t = min(rem, p); val += t*c; rem -= t
    return val/alpha

print("\n=== IDEAL SIMULATION ===")
ideal_results = {}
for p in DEPTHS:
    t0 = time.time()
    qc = build_param_circuit(p)
    def obj(v):
        sv = Statevector(qc.assign_parameters(v))
        return float(np.sum(np.abs(sv.data)**2 * all_costs))
    x0 = np.random.uniform(0, np.pi, 2*p)
    res = minimize(obj, x0, method="COBYLA",
                   options={"maxiter": 150, "rhobeg": 0.5})
    sv = Statevector(qc.assign_parameters(res.x))
    pb = np.abs(sv.data)**2
    valid_p = float(sum(pb[i] for i in range(2**n) if valid[i]))
    r_cv = (Cmax - cvar(pb, all_costs, 0.25)) / (Cmax - Cmin)
    ideal_results[p] = dict(params=res.x, r_cv=r_cv, valid=valid_p)
    print(f"  p={p}: valid={valid_p:.4f}  CVaR r={r_cv:.4f}  ({time.time()-t0:.0f}s)")

# ============================================================
# NOISY SIMULATION (fake backend model)
# ============================================================
print("\n=== NOISY SIMULATION ===")
service = QiskitRuntimeService()
backend = service.backend(BACKEND_NAME)
print(f"Backend: {backend.name}  qubits={backend.num_qubits}")

noise_model = NoiseModel.from_backend(backend)
noisy_sim = AerSimulator(noise_model=noise_model)

noisy_results = {}
for p in DEPTHS:
    qc = build_param_circuit(p, measure=True)
    qc_bound = qc.assign_parameters(ideal_results[p]["params"])
    pm = transpile(qc_bound, noisy_sim, optimization_level=3)
    job = noisy_sim.run(pm, shots=SHOTS)
    counts = job.result().get_counts()

    total = sum(counts.values())
    valid_shots = sum(c for bs, c in counts.items()
                      if bs.count("1") == K)
    # best sampled cost
    best_c = min(cost_of([int(b) for b in bs[::-1]]) for bs in counts
                 if bs.count("1") == K) if valid_shots > 0 else np.inf
    r_b = (Cmax - best_c) / (Cmax - Cmin) if np.isfinite(best_c) else 0
    noisy_results[p] = dict(valid=valid_shots/total, r_best=r_b)
    print(f"  p={p}: valid_frac={valid_shots/total:.4f}  best r={r_b:.4f}")

# ============================================================
# HARDWARE
# ============================================================
print(f"\n=== HARDWARE: {BACKEND_NAME} ===")
sampler = Sampler(mode=backend)
sampler.options.dynamical_decoupling.enable = True
sampler.options.dynamical_decoupling.sequence_type = "XpXm"
sampler.options.twirling.enable_gates = True
sampler.options.twirling.enable_measure = True
print("DD: XpXm ON, gate twirling ON, measurement twirling ON")

hw_results = {}
job_ids = []
for p in DEPTHS:
    qc = build_param_circuit(p, measure=True)
    qc_bound = qc.assign_parameters(ideal_results[p]["params"])
    pm = transpile(qc_bound, backend, optimization_level=3)
    depth_t = pm.depth()
    ops = pm.count_ops()
    print(f"\n  p={p}: transpiled depth={depth_t}  ops={dict(ops)}")

    job = sampler.run([pm], shots=SHOTS)
    jid = job.job_id()
    job_ids.append(jid)
    print(f"  submitted job {jid} — waiting...")
    result = job.result()
    counts = result[0].data.meas.get_counts()

    total = sum(counts.values())
    valid_shots = sum(c for bs, c in counts.items() if bs.count("1") == K)
    best_c = min(cost_of([int(b) for b in bs[::-1]]) for bs in counts
                 if bs.count("1") == K) if valid_shots > 0 else np.inf
    r_b = (Cmax - best_c) / (Cmax - Cmin) if np.isfinite(best_c) else 0

    hw_results[p] = dict(valid=valid_shots/total, r_best=r_b,
                         depth=depth_t, job_id=jid)
    print(f"  p={p}: valid_frac={valid_shots/total:.4f}  best r={r_b:.4f}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print(f"{'p':<4} {'Ideal valid':<14} {'Noisy valid':<14} {'HW valid':<12} "
      f"{'Ideal r':<10} {'Noisy r':<10} {'HW r':<10}")
print("-"*70)
for p in DEPTHS:
    print(f"{p:<4} {ideal_results[p]['valid']:<14.4f} "
          f"{noisy_results[p]['valid']:<14.4f} {hw_results[p]['valid']:<12.4f} "
          f"{ideal_results[p]['r_cv']:<10.4f} {noisy_results[p]['r_best']:<10.4f} "
          f"{hw_results[p]['r_best']:<10.4f}")

print(f"\nJob IDs: {job_ids}")
print("Check QPU time at https://quantum.cloud.ibm.com/workloads")

# Save results
import json
out = {"ideal": {str(k): {kk: float(vv) if isinstance(vv, (int,float,np.floating))
                          else str(vv) for kk, vv in v.items()}
                 for k, v in ideal_results.items()},
       "noisy": {str(k): v for k, v in noisy_results.items()},
       "hardware": {str(k): v for k, v in hw_results.items()},
       "job_ids": job_ids,
       "backend": BACKEND_NAME}
with open("hardware_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("Saved hardware_results.json")

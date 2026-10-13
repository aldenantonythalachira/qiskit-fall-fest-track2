"""Gradient variance vs depth: X-mixer vs XY-mixer."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import to_ising

np.random.seed(42)
n, K, lam = 6, 3, 0.5
mu = np.random.uniform(0.05, 0.25, size=n)
A = np.random.randn(n, n); Sigma = A @ A.T / n + 1e-3*np.eye(n)
all_x = np.array([[(i>>k)&1 for k in range(n)] for i in range(2**n)])
all_costs = np.array([-mu@x + lam*x@Sigma@x for x in all_x])

qp = QuadraticProgram()
for i in range(n): qp.binary_var(f"x{i}")
qp.minimize(linear={f"x{i}": -float(mu[i])+float(lam*Sigma[i,i]) for i in range(n)},
            quadratic={(f"x{i}",f"x{j}"): float(2*lam*Sigma[i,j])
                       for i in range(n) for j in range(i+1,n)}, constant=0.0)
cost_op, _ = to_ising(qp)

# XY mixer
def xy_mixer(nq):
    terms=[]
    for i in range(nq):
        for j in range(i+1,nq):
            sx=["I"]*nq; sx[nq-1-i]="X"; sx[nq-1-j]="X"; terms.append(("".join(sx),1.0))
            sy=["I"]*nq; sy[nq-1-i]="Y"; sy[nq-1-j]="Y"; terms.append(("".join(sy),1.0))
    return SparsePauliOp.from_list(terms)
xy_op = xy_mixer(n)
x_mixer = SparsePauliOp.from_list([("I"*(n-1-i)+"X"+"I"*i, 1.0) for i in range(n)])

# Dicke
dicke = np.zeros(2**n)
for i in range(2**n):
    if all_x[i].sum() == K: dicke[i]=1.0
dicke /= np.linalg.norm(dicke)

def build(c_op, m_op, reps, init):
    g=ParameterVector("g",reps); b=ParameterVector("b",reps)
    qc=QuantumCircuit(n); qc.initialize(init, range(n))
    for L in range(reps):
        qc.append(PauliEvolutionGate(c_op, time=g[L]), range(n))
        qc.append(PauliEvolutionGate(m_op, time=b[L]), range(n))
    return qc

def grad_norm(qc, params):
    eps = 1e-3
    g = np.zeros_like(params)
    for i in range(len(params)):
        p1 = params.copy(); p1[i] += eps
        p2 = params.copy(); p2[i] -= eps
        def f(p): return float(np.sum(np.abs(Statevector(qc.assign_parameters(p)).data)**2*all_costs))
        g[i] = (f(p1)-f(p2))/(2*eps)
    return np.linalg.norm(g)

print("Sampling gradient variance...")
for label, mixer, init in [("X-mixer + |+>", x_mixer, np.ones(2**n)/np.sqrt(2**n)),
                            ("XY-mixer + Dicke", xy_op, dicke)]:
    print(f"\n{label}:")
    for p in [1,2,3]:
        qc = build(cost_op, mixer, p, init)
        grads = []
        for _ in range(50):
            v = np.random.uniform(0, np.pi, 2*p)
            grads.append(grad_norm(qc, v))
        print(f"  p={p}: grad_var={np.var(grads):.6e}")

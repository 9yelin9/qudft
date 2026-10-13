#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import pennylane as qp
from pennylane import numpy as pnp

from itertools import product
from pennylane.pauli import PauliSentence, PauliWord

parser = argparse.ArgumentParser()
parser.add_argument('M', type=int)
args = parser.parse_args()

M = args.M
N = 2**M
wires = list(range(M))
bases = np.array([list(np.binary_repr(i, width=M)) for i in range(N)], dtype='int')

dev = qp.device('default.qubit')

def gen_gamma():
    """
    em = {(0, 0): np.array([[1, 0], [0, 0]], dtype='complex'), # elementary matrices
          (0, 1): np.array([[0, 1], [0, 0]], dtype='complex'),
          (1, 0): np.array([[0, 0], [1, 0]], dtype='complex'),
          (1, 1): np.array([[0, 0], [0, 1]], dtype='complex')}

    gamma = []
    for i, j in [(i, j) for i in bases for j in bases]:
        gamma_ij = np.eye(1, dtype='complex')
        gamma_ji = np.eye(1, dtype='complex')
        for m in wires:
            gamma_ij = np.kron(gamma_ij, em[(i[m], j[m])])
            gamma_ji = np.kron(gamma_ji, em[(j[m], i[m])])
        gamma.append(qp.pauli_decompose((gamma_ij + gamma_ji)/2, pauli=True))
    return gamma
    """

    rules = {
        (0, 0): [('I', 0.5), ('Z',  0.5)],
        (1, 1): [('I', 0.5), ('Z', -0.5)],
        (0, 1): [('X', 0.5)],
        (1, 0): [('X', 0.5)],
    }

    gamma = []
    for i, j in product(bases, bases):
        sentence = PauliSentence({PauliWord({}): 1.0})

        for m in wires:
            next_sentence = PauliSentence({})
            for pw, c in sentence.items():
                for pauli_char, coeff in rules[(i[m], j[m])]:
                    new_pw = PauliWord({**pw, m: pauli_char}) if pauli_char != 'I' else pw
                    new_sentence = PauliSentence({new_pw: c * coeff})
                    next_sentence += new_sentence
            sentence = next_sentence

        sentence.simplify()
        gamma.append(sentence)
    return gamma

def is_z(G):
    for r in range(G.shape[0]):
        if any(G[r, :M]): return False
        if sum(G[r, M:]) > 1: return False
    return True

def apply_H(G, m):
    G[:, [m, M+m]] = G[:, [M+m, m]]
    return G

def apply_S(G, m):
    G[:, M+m] = (G[:, M+m] + G[:, m]) % 2
    return G

def apply_CNOT(G, target, control):
    G[:, control] = (G[:, control] + G[:, target]) % 2
    G[:, M+target] = (G[:, M+target] + G[:, M+control]) % 2
    return G

@qp.qnode(dev)
def circuit(x):
    qp.RY(x[0], wires=0)
    qp.RX(x[1], wires=1)
    return qp.expval(obs)
#print(qp.draw(circuit)([np.pi/4, np.pi/4]))

gamma = gen_gamma()
#print(gamma)

obs_total = sorted({obs for gamma_ij in gamma for obs in gamma_ij.keys()}, key=str)
#print(obs_total, len(obs_total))

obs_group = qp.pauli.group_observables(obs_total, grouping_type='commuting', method='rlf')
#print(obs_group, len(obs_group))
print(f'M={M}, len(obs_total)={len(obs_total)}, len(obs_group)={len(obs_group)}')
#sys.exit(1)

for g in obs_group:
    print(g)

    # 1. encode binary symplectic matrix
    G = qp.pauli.observables_to_binary_matrix(g, n_qubits=M)
    T = np.eye(G.shape[0], dtype=int)
    T_inv = np.eye(G.shape[0], dtype=int)
    print('#1 G=', G, sep='\n')

    # 2. binary row reduction
    pivot = 0
    for c in range(G.shape[1]):  
        pivot_cur = -1
        for r in range(pivot, G.shape[0]): 
            if G[r, c]:
                pivot_cur = r
                break

        if pivot_cur < 0: continue

        G[[pivot, pivot_cur]] = G[[pivot_cur, pivot]]
        T[[pivot, pivot_cur]] = T[[pivot_cur, pivot]]
        T_inv[:, [pivot, pivot_cur]] = T_inv[:, [pivot_cur, pivot]]
        for r in range(pivot+1, G.shape[0]):
            if G[r, c]:
                G[r] = (G[r] + G[pivot]) % 2
                T[r] = (T[r] + T[pivot]) % 2
                T_inv[:, pivot] = (T_inv[:, pivot] + T_inv[:, r]) % 2
        pivot += 1
    G = G[:pivot]
    T = T[:pivot]
    T_inv = T_inv[:, :pivot]
    print('#2 G=', G, 'T=', T, 'T_inv=', T_inv, '(T @ T_inv) % 2=', (T @ T_inv) % 2, sep='\n')

    # 3. elimination
    U = []
    used = []
    for r in range(pivot):
        for m in wires:
            if G[r, m] and G[r, M+m]: # Y -> X
                G = apply_S(G, m)
                U.append(('S', m))
            if G[r, m] and not G[r, M+m]: # X -> Z
                G = apply_H(G, m)
                U.append(('H', m))

        support = [m for m in wires if G[r, M+m]]
        control = [m for m in support if m not in used][0]
        used.append(control)
        for target in support:
            if target != control:
                G = apply_CNOT(G, target, control)
                U.append(('C', target, control))
    print('#3 U=', U, 'G=', G, '\n', sep='\n')



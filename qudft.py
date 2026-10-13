#!/usr/bin/env python3
import os
import re
import sys
import time
import scipy
import argparse
import numpy as np

from datetime import datetime
from itertools import product

import pennylane as qp
from pennylane import numpy as pnp

import jax
import jax.numpy as jnp
import optax

np.random.seed(42)
np.set_printoptions(precision=3, linewidth=120, suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument('mat', type=str)
parser.add_argument('-k', nargs=3, type=int, default=[1, 1, 1])
parser.add_argument('-sc', nargs=3, type=int, default=[1, 1, 1])
parser.add_argument('-qpu', action='store_true')
parser.add_argument('-pyscf', action='store_true')
args = parser.parse_args()

class VQE:
    def __init__(self, mat, mf, fn_head, is_qpu=False, shots=1000):
        self.mat = mat
        self.mf = mf
        self.fn_head = fn_head
        self.is_qpu = is_qpu

        self.N = self.mat.nao
        self.Ne = self.mat.nelectron
        print(f'N={self.N}\nNe={self.Ne}')

        self.ncell = self.mat.ncell
        print(f'ncell={self.ncell}')

        self.nkpts = len(self.mf.kpts)
        print(f'nkpts={self.nkpts}\nkpts=\n{self.mf.kpts}')

        self.M = int(np.ceil(np.log2(self.N)))
        #self.L = 2 + self.M + np.log2(self.nkpts).astype(int)
        self.L = (self.M - 1) * (np.log2(self.ncell + 1).astype(int) + np.log2(self.nkpts + 1).astype(int))
        self.bases = jnp.asarray([list(np.binary_repr(i, width=self.M)) for i in range(self.N)], dtype='int')
        self.wires = list(range(self.M))
        #self.weight = [(self.N - k) / (self.N * (self.N + 1) / 2) for k in range(self.N)]
        self.pairs = [(i, j) for i in self.wires for j in self.wires[i + 1:]]
        self.npairs = len(self.pairs)
        print(f'M={self.M}\nL={self.L}\nnpairs={self.npairs}')

        self.dev = qp.device('default.qubit', wires=self.M)
        #self.dev = qp.device('ionq.simulator', wires=self.M, noise_model='forte-enterprise-1')
        #self.dev = qp.device('ionq.qpu', wires=self.M, backend='forte-enterprise-1')

        #self.qnode_eks = qp.QNode(self.circuit_eks, self.dev)
        if self.is_qpu:
            self.shots = shots
            self.gamma = self.gen_gamma()
            self.qnode = qp.QNode(self.circuit_rho_qpu, self.dev, interface='jax', diff_method='parameter-shift')
            self.qnode = qp.set_shots(self.qnode, shots=self.shots)
            self.get_rho = self.get_rho_qpu
            fn_suffix = '_qpu' + str(self.shots)
        else:
            self.shots = None
            self.gamma = None
            self.qnode = qp.QNode(self.circuit_rho_cpu, self.dev, interface='jax', diff_method='backprop')
            self.get_rho = self.get_rho_cpu
            fn_suffix = ''
        print(f'is_qpu={self.is_qpu}\nshots={self.shots}')

        self.fn_save = f'data/{self.fn_head}_M{self.M}_L{self.L}{fn_suffix}.out'
        script = (
        f'fn_save {self.fn_save}\n'
        f'is_qpu  {int(self.is_qpu)}\n'
        f'shots   {self.shots}\n'
        f'\n'
        f'N  {self.N}\n'
        f'Ne {self.Ne}\n'
        f'\n'
        f'nkpts {self.nkpts}\n'
        f'kpts\n{self.mf.kpts}\n'
        f'\n'
        f'M {self.M}\n'
        f'L {self.L}\n'
        f'\n')
        with open(self.fn_save, 'w') as f: f.write(script)
        print(f'fn_save={self.fn_save}\n')

        self.h1e = jnp.asarray(self.mf.get_hcore(self.mat).reshape(self.nkpts, self.N, self.N))
        self.s1e = jnp.asarray(self.mf.get_ovlp(self.mat).reshape(self.nkpts, self.N, self.N))

        self.shm, self.shm_dagger = [], []
        for k in range(self.nkpts):
            self.shm += [scipy.linalg.fractional_matrix_power(self.s1e[k], -0.5)]
            self.shm_dagger += [self.shm[k].conj().T]
        self.shm, self.shm_dagger = jnp.asarray(self.shm), jnp.asarray(self.shm_dagger)

    def gen_gamma(self):
        em = {(0, 0): np.array([[1, 0], [0, 0]], dtype='complex'), # elementary matrices
              (0, 1): np.array([[0, 1], [0, 0]], dtype='complex'),
              (1, 0): np.array([[0, 0], [1, 0]], dtype='complex'),
              (1, 1): np.array([[0, 0], [0, 1]], dtype='complex')}

        gamma = []
        for i, j in [(i, j) for i in self.bases for j in self.bases]:
            gamma_ij = np.eye(1, dtype='complex')
            gamma_ji = np.eye(1, dtype='complex')
            for m in self.wires:
                gamma_ij = np.kron(gamma_ij, em[(int(i[m]), int(j[m]))])
                gamma_ji = np.kron(gamma_ji, em[(int(j[m]), int(i[m]))])
            gamma.append(qp.pauli_decompose(gamma_ij + gamma_ji)/2)
        return gamma

    """ QDFT
    def prep_circuit(self, theta, basis):
        qp.BasisState(basis, wires=self.wires)
        for m in self.wires: qp.RY(theta[m], wires=m)
        for n in range(self.L):
            for m in self.wires[:-1]: qp.CNOT(wires=[m, m+1])
            for m in self.wires:      qp.RY(theta[self.M * (n+1) + m], wires=m)
    """

    """ quDFTv2
    def prep_circuit(self, theta, basis):
        qp.BasisState(basis, wires=self.wires)
        for n in range(self.L):
            for m in self.wires:
                qp.RX(theta[n, m, 0], wires=m)
                qp.CNOT(wires=[m, (m + 1) % self.M])
                qp.RY(theta[n, m, 1], wires=m)
                qp.CNOT(wires=[m, (m + 1) % self.M])
                qp.RZ(theta[n, m, 2], wires=m)
    """

    def prep_circuit(self, theta, basis):
        qp.BasisState(basis, wires=self.wires)
        qp.ArbitraryUnitary(theta, wires=self.wires)
        """
        for l in range(self.L):
            offset = l % (self.M - 1) + 1
            for i in self.wires:
                j = (i + offset) % self.M
                qp.RY(theta[l, i, 0], wires=i)
                qp.CNOT(wires=[i, j])
        """

    """ QDFT
    def circuit_eks(self, theta, basis, hamiltonian):
        self.prep_circuit(theta, basis)
        return qp.expval(hamiltonian)
    """

    def circuit_rho_cpu(self, theta, basis): # rho_ij = <i|psi><psi|j>
        self.prep_circuit(theta, basis)
        return qp.density_matrix(self.wires)

    def circuit_rho_qpu(self, theta, basis): # rho_ij = <psi|gamma_ij|psi>
        self.prep_circuit(theta, basis)
        return [qp.expval(gamma_ij) for gamma_ij in self.gamma]

    """ quDFTv2
    def get_rho_cpu(self, theta, fock):
        theta = theta.reshape(self.nkpts, self.L, self.M)
        def get_rho_raw_k(theta_k, shm_k, shm_dagger_k):
            def get_rho_raw_kb(basis):
                rho_raw_kb = self.qnode(theta_k, basis)[:self.N, :self.N]
                return shm_dagger_k @ rho_raw_kb @ shm_k
            rho_raw_k = jax.vmap(get_rho_raw_kb)(self.bases)
            return rho_raw_k
        rho_raw = jax.vmap(get_rho_raw_k)(theta, self.shm, self.shm_dagger)
        return rho_raw
    """

    def get_rho_cpu(self, theta):
        #theta = theta.reshape(self.nkpts, self.L, self.M, -1)
        theta = theta.reshape(self.nkpts, 4**self.M - 1)
        def get_rho_raw_k(theta_k, shm_k, shm_dagger_k):
            def get_rho_raw_kb(basis):
                rho_raw_kb = self.qnode(theta_k, basis)[:self.N, :self.N]
                return shm_k @ rho_raw_kb @ shm_dagger_k # orth -> AO
            rho_raw_k = jax.vmap(get_rho_raw_kb)(self.bases)
            return rho_raw_k
        rho_raw = jax.vmap(get_rho_raw_k)(theta, self.shm, self.shm_dagger)
        return rho_raw

    def get_rho_qpu(self, theta):
        #theta = theta.reshape(self.nkpts, self.L, self.M, -1)
        rho_raw = []
        for theta_k, shm_k, shm_dagger_k in zip(theta, self.shm, self.shm_dagger):
            rho_raw_k = []
            for basis in self.bases:
                rho_raw_kb = jnp.asarray(self.qnode(theta_k, basis)).reshape(self.N, self.N)
                rho_raw_kb = shm_k @ rho_raw_kb @ shm_dagger_k # orth -> AO
                rho_raw_k.append(rho_raw_kb)
            rho_raw.append(jnp.stack(rho_raw_k))
        rho_raw = jnp.stack(rho_raw)
        return rho_raw

    """ quDFTv1
    def cost_rho(self, theta, qnode, gamma, rho_target):
        rho = self.get_rho(theta, qnode, gamma)
        return jnp.linalg.norm(rho - rho_target)**2
    """

    """ QDFT
    def cost_qdft(self, theta, gamma, fock, hamiltonian, alpha):
        eks = []
        for basis, weight in zip(self.bases[:self.Nocc], self.weight[:self.Nocc]):
            eks.append(self.qnode_eks(theta, basis, hamiltonian) * weight)
        eks = np.sum(eks)

        rho = self.get_rho_qpu(theta, gamma)
        R = fock @ rho @ self.s1e - self.s1e @ rho @ fock # FDS - SDF = 0
        return eks + alpha * np.linalg.norm(R, 'fro')**2
    """

    def cost_qudft(self, theta, fock_old):
        rho_raw = self.get_rho(theta)
        eks = jnp.einsum('kij,kbji->kb', fock_old, rho_raw).real
        occ = jnp.asarray(self.mf.get_occ(eks))
        rho = jnp.einsum('kb,kbij->kij', occ, rho_raw)

        vhf = self.mf.get_veff(self.mat, rho)
        #etot = self.mf.energy_tot(rho[0], self.h1e[0], vhf)
        etot = self.mf.energy_tot(rho, self.h1e, vhf)
        fock = self.mf.get_fock(self.h1e, self.s1e, vhf, rho)

        w = 1.0 / len(self.h1e)
        nelec = w * jnp.einsum('kij,kji->', self.s1e, rho).real
        e1 = w * jnp.einsum('kij,kji->', self.h1e, rho).real
        eh = vhf.ecoul.real
        exc = vhf.exc.real
        enuc = self.mf.energy_nuc()

        jax.debug.print('e1={e1:f} eh={eh:f} exc={exc:f} etot={etot:f}',
                        e1=e1, eh=eh, exc=exc, etot=etot)

        return etot, fock

    def grad_qudft(self, theta):
        theta = np.asarray(theta)
        return np.asarray(self.cost_grad(theta), dtype=float)

    """ quDFTv1
    def init_rho(self, theta, qnode, gamma, max_itr=100, tol=1e-2):
        rho_target = self.mf.get_init_guess(self.mat, self.mf.init_guess, s1e=self.s1e)

        t0 = time.time()
        res = scipy.optimize.minimize(self.cost_rho, theta,
                                      args=(qnode, gamma, rho_target),
                                      method='COBYLA',
                                      tol=tol,
                                      options={'disp': True})
        self.save_result(self.init_rho.__name__, t0)
        return pnp.array(res.x, dtype=float, requires_grad=True), rho_target
    """

    def pyscf(self, max_itr=1000, e_tol=1e-3, pat_max=3, mix=0.3):
        rho = self.mf.get_init_guess(self.mat, self.mf.init_guess, s1e=self.s1e)
        vhf = self.mf.get_veff(self.mat, rho)
        etot = self.mf.energy_tot(rho, self.h1e, vhf)

        pat = 0
        t0 = time.time()
        for itr in range(max_itr):
            rho_old = rho
            etot_old = etot

            fock = self.mf.get_fock(self.h1e, self.s1e, vhf, rho) # fock = h1e + vhf
            energy, coeff = self.mf.eig(fock, self.s1e)
            occ = self.mf.get_occ(energy, coeff)
            rho = self.mf.make_rdm1(coeff, occ)
            rho = mix * rho + (1 - mix) * rho_old # Mixing
            #vhf = self.mf.get_veff(self.mat, rho, rho_old, vhf) # DIIS
            vhf = self.mf.get_veff(self.mat, rho, rho_old, vhf)
            etot = self.mf.energy_tot(rho, self.h1e, vhf)
            print(f'itr= {itr+1:3d} etot= {etot:9f} delta= {etot - etot_old:9f}')

            if abs(etot - etot_old) < e_tol:
                if pat == pat_max: break
                else: pat += 1
        self.save_result(self.pyscf.__name__, t0, etot)

    """ QDFT
    def qdft(self, theta, shots=None, max_itr=1000, tol=1e-6, mix=0.3, alpha=10):
        gamma = self.gen_gamma()
        qnode = self.qnode_rho_qpu
        qp.set_shots(qnode, shots=shots)
        theta, rho = self.init_rho(theta, qnode, gamma)

        vhf = self.mf.get_veff(self.mat, rho)
        etot = self.mf.energy_tot(rho, self.h1e, vhf)

        t0 = time.time()
        for itr in range(max_itr):
            rho_old = rho
            etot_old = etot

            fock = self.mf.get_fock(self.h1e, self.s1e, vhf, rho)
            hamiltonian = qp.Hamiltonian(fock.ravel(), gamma)
            res = scipy.optimize.minimize(self.cost_qdft, theta,
                                          args=(gamma, fock, hamiltonian, alpha),
                                          method='L-BFGS-B',
                                          options={'disp': False})
            theta = res.x
            rho = mix * self.get_rho_qpu(theta) + (1 - mix) * rho_old
            vhf = self.mf.get_veff(self.mat, rho)
            etot = self.mf.energy_tot(rho, self.h1e, vhf)
            print(f'itr= {itr+1:3d} etot= {etot:9f} delta= {etot - etot_old:9f}')

            if abs(etot - etot_old) < tol: break
        self.save_result(self.qdft.__name__, t0, etot)
    """

    """ quDFTv1
    def qudftv1(self, theta, shots=None, tol=1e-3):
        gamma = self.gen_gamma()
        qnode = qp.set_shots(self.qnode_rho_qpu, shots=shots)
        theta, _ = self.init_rho(theta, qnode, gamma)

        t0 = time.time()
        res = scipy.optimize.minimize(self.cost_qudft, theta,
                                      args=(qnode, gamma),
                                      method='L-BFGS-B',
                                      tol=tol,
                                      options={'disp': True})
        theta, etot = res.x, res.fun
        self.save_result(self.qudftv1.__name__, t0, etot)
    """

    def qudftv2(self, theta,
                opt0='lion',  lr0=0.1,  tol0=5,    max_itr0=100,
                opt1='nadam', lr1=0.01, tol1=1e-4, max_itr1=1000):
        script = (
        f'opt0      {opt0}\n'
        f'lr0       {lr0}\n'
        f'tol0      {tol0}\n'
        f'max_itr0  {max_itr0}\n'
        f'\n'
        f'opt1      {opt1}\n'
        f'lr1       {lr1}\n'
        f'tol1      {tol1}\n'
        f'max_itr1  {max_itr1}\n'
        f'\n')
        with open(self.fn_save, 'a') as f: f.write(script)

        rho = jnp.asarray(self.mf.get_init_guess(self.mat, self.mf.init_guess, s1e=self.s1e))
        vhf = self.mf.get_veff(self.mat, rho)
        etot = self.mf.energy_tot(rho, self.h1e, vhf)
        fock = self.mf.get_fock(self.h1e, self.s1e, vhf, rho)

        cost = lambda theta, fock: self.cost_qudft(theta, fock)
        vng = jax.value_and_grad(cost, argnums=0, has_aux=True)

        t0 = time.time()

        """ lion
        print(f'\nwarm-start ({opt0} lr={lr0})')
        solver = getattr(optax, opt0)(learning_rate=lr0)
        state = solver.init(theta)
        etot_best = np.inf
        itr_best = 0

        print(f'{"itr":>5s} {"etot":>10s} {"etot-etot_best":>15s}')
        for i in range(max_itr0):
            fock_old = fock
            (etot, fock), grad = vng(theta, fock_old)
            print(f'{i:5d} {etot:10f} {etot-etot_best:15f}')

            if etot < etot_best:
                etot_best = etot
                itr_best = i
                theta_best = theta
                fock_best = fock_old
            if i - itr_best > tol0: break

            update, state = solver.update(grad, state, theta)
            theta = optax.apply_updates(theta, update)
        theta = theta_best
        fock = fock_best
        """

        print(f'\noptimization ({opt1} lr={lr1})')
        solver = getattr(optax, opt1)(learning_rate=lr1)
        state = solver.init(theta)

        print(f'{"itr":>5s} {"etot":>10s} {"grad":>10s}')
        for i in range(max_itr1):
            etot_old, fock_old = etot, fock
            (etot, fock), grad = vng(theta, fock_old)
            grad_norm = jnp.linalg.norm(grad) / jnp.sqrt(grad.size)
            print(f'{i:5d} {etot:10f} {grad_norm:10f}')

            if grad_norm < tol1: break

            update, state = solver.update(grad, state, theta)
            theta = optax.apply_updates(theta, update)

        self.save_result(self.qudftv2.__name__, t0, etot)
        return theta

    def save_result(self, method, t0, etot):
        t1 = time.time()
        tm, ts = divmod(int(t1 - t0), 60)
        print(f'{method} time elapsed: {tm}m {ts}s\n')

        script = (
        f'etot {etot:f}\n'
        f'\n'
        f'start_time   {datetime.fromtimestamp(t0):%Y-%m-%d %H:%M:%S}\n'
        f'end_time     {datetime.fromtimestamp(t1):%Y-%m-%d %H:%M:%S}\n'
        f'elapsed_time {tm}m {ts}s\n')
        with open(self.fn_save, 'a') as f: f.write(script)
        print(f'Saved at {self.fn_save}\n')

def supercell(mat, sc):
    avec = np.asarray(mat.lattice_vectors()) # Bohr
    coords = np.asarray(mat.atom_coords()) # Bohr
    symbol = np.asarray([mat.atom_symbol(i) for i in range(mat.natm)])

    atom = []
    for i in range(sc[0]):
        for j in range(sc[1]):
            for k in range(sc[2]):
                r = i * avec[0] + j * avec[1] + k * avec[2]
                for s, c in zip(symbol, coords):
                    atom.append((s, c + r))

    mat_sc = gto.Cell() 
    mat_sc.unit = 'Bohr'
    mat_sc.atom = atom
    mat_sc.a = [(n * a).tolist() for n, a in zip(sc, avec)]
    mat_sc.basis = mat.basis
    mat_sc.pseudo = mat.pseudo
    mat_sc.build()
    return mat_sc

"""
if re.match('H', args.mat):
    from pyscfad import gto, dft # Gaussian-Type Orbital (GTO)

    N = 4
    a = 1.0
    xc =  'LDA'
    mat = gto.Mole()
    mat.atom = '; '.join([f'H {a*i:.1f} 0 0' for i in range(N)])
    mat.basis = 'sto-3g'
    mat.pseudo = 'gth-lda'
"""
    
if re.match('Si', args.mat):
    from pyscfad.pbc import gto, dft

    a = 5.46
    xc = 'PBE'
    mat = gto.Cell()
    mat.atom = f'Si 0 0 0; Si {a/4} {a/4} {a/4}'
    mat.a = [[0., a/2, a/2], [a/2, 0., a/2], [a/2, a/2, 0.]]
    mat.basis = 'gth-szv'
    mat.pseudo = 'gth-pbe'

elif re.match('Al', args.mat):
    from pyscfad.pbc import gto, dft

    a = 4.04
    xc = 'PBE'
    mat = gto.Cell()
    mat.atom = f'Al 0 0 0; Al 0 {a/2} {a/2}'
    mat.a = [[0., a, a], [a/2, 0., a/2], [a/2, a/2, 0.]]
    mat.spin = 0
    mat.basis = 'gth-szv-molopt-sr'
    mat.pseudo = 'gth-pbe'

elif re.match('Fe', args.mat):
    from pyscfad.pbc import gto, dft

    a = 2.86
    xc = 'PBE'
    mat = gto.Cell()
    mat.atom = f'Fe 0 0 0'
    mat.a = [[-a/2, a/2, a/2], [a/2, -a/2, a/2], [a/2, a/2, -a/2]]
    mat.spin = 4
    mat.basis = 'gth-szv-molopt-sr'
    mat.pseudo = 'gth-pbe'

mat.build()
mat = supercell(mat, args.sc)
mat.ncell = np.prod(args.sc).astype(int)

kpts = mat.make_kpts(args.k)

mf = dft.KRKS(mat, xc=xc, kpts=kpts) # Restricted KS-DFT (RKS): the spin-orbitals are either spin-up or spin-down
#mf = dft.UKS(mat, xc=xc, kpts=kpts) # Unrestricted KS-DFT (UKS): the orbitals can have either alpha or beta spin

mf.sigma = 0.001
mf.smearing_method = 'fermi'

import inspect
if args.pyscf:
    t0 = time.time()

    mf.kernel() # pyscfad/pbc/scf/khf.py modified: use fractional occupation 

    rho = mf.make_rdm1()
    h1e = mf.get_hcore(mf.cell, mf.kpts)
    s1e = mf.get_ovlp(mf.cell, mf.kpts)
    vhf = mf.get_veff(mf.cell, rho)

    w = 1.0 / len(mf.kpts)
    nelec = w * np.einsum('kij,kji->', s1e, rho).real
    e1 = w * np.einsum('kij,kji->', h1e, rho).real
    eh = vhf.ecoul.real
    exc = vhf.exc.real
    enuc = mf.energy_nuc()
    etot = mf.energy_tot(rho, h1e, vhf)

    print(f'e1={e1:f} eh={eh:f} exc={exc:f} etot={etot:f}')

    t1 = int(time.time() - t0)
    tm, ts = divmod(t1, 60)
    print(f'time elapsed: {tm}m {ts}s\n')
    sys.exit(0)

vqe = VQE(mat, mf, f'{args.mat}_sc{'x'.join(map(str, args.sc))}_k{'x'.join(map(str, args.k))}', is_qpu=args.qpu)
#theta = jnp.asarray(0.01 * np.random.uniform(-1, 1, vqe.nkpts * vqe.L * vqe.M * 1))
theta = jnp.asarray(0.01 * np.random.uniform(-1, 1, vqe.nkpts * (4**vqe.M - 1)))
vqe.qudftv2(theta)

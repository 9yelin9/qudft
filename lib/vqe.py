"""Core variational quantum eigensolver used by quDFT."""

import os
import time

import h5py
import scipy
import numpy as np

import pennylane as qp
from pennylane import numpy as pnp

import jax
import jax.numpy as jnp
from jax.scipy.special import xlogy

import optax

from lib.ao_encoding import structured_ao_encoding
from lib.ansatz import Ansatz, select_adaptive_paulis


def reserve_result_paths(fn_head, directory='data'):
    """Atomically reserve input/output paths, suffixing only on collision."""
    number = 0
    while True:
        suffix = '' if number == 0 else f'_rep{number:02d}'
        stem = os.path.join(directory, f'{fn_head}{suffix}')
        fn_in = f'{stem}.in'
        fn_out = f'{stem}.out'
        if os.path.exists(fn_out):
            number += 1
            continue
        try:
            fd = os.open(fn_in, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            number += 1
            continue
        return fn_in, fn_out, fd


class VQE:
    def __init__(self, mat, mf, fn_head, is_qpu=False, shots=1000,
                 device_name='default.qubit', ansatz='au-oddy', topology='linear',
                 nparam=28, ncnot=12, circuit=None, ground_state=False,
                 min_coverage=False, fixed_occupation=False,
                 fock_occupation=False, fixed_paulis=None, resume_from=None,
                 dry_run=False):
        self.mat = mat
        self.mf = mf
        self.fn_head = fn_head
        self.is_qpu = is_qpu
        self.ground_state = ground_state
        self.fixed_occupation = fixed_occupation
        self.fock_occupation = fock_occupation
        self.resume_from = resume_from
        self.dry_run = bool(dry_run)
        if self.fixed_occupation and self.fock_occupation:
            raise ValueError(
                    'fixed_occupation and fock_occupation are mutually exclusive')

        self.N = self.mat.nao
        self.nelec = self.mat.nelectron
        self.ncell = self.mat.ncell
        self.nkpts = len(self.mf.kpts)

        (bases, ao_to_basis, self.natom_bits,
         self.nlocal_bits) = structured_ao_encoding(self.mat)
        self.M = bases.shape[1]
        #self.L = 2 + self.M + np.log2(self.nkpts).astype(int)
        self.bases = jnp.asarray(bases, dtype='int')
        # AO-ordered list of computational-basis indices.  Selecting matrix
        # rows/columns in this order converts circuit output back to PySCF's
        # AO ordering without changing h1e, s1e, or the DFT routines.
        self.ao_to_basis = jnp.asarray(ao_to_basis, dtype=int)
        self.wires = list(range(self.M))
        self.atom_wires = self.wires[:self.natom_bits]
        self.local_wires = self.wires[self.natom_bits:]
        self.ao_counts = (
                np.asarray(self.mat.aoslice_by_atom(), dtype=int)[:, 3]
                - np.asarray(self.mat.aoslice_by_atom(), dtype=int)[:, 2])
        #self.weight = [(self.N - k) / (self.N * (self.N + 1) / 2) for k in range(self.N)]

        self.ansatz_circuit = Ansatz(
                ansatz,
                wires=self.wires,
                atom_wires=self.atom_wires,
                local_wires=self.local_wires,
                ao_counts=self.ao_counts,
                kpts=self.mf.kpts,
                topology=topology,
                nparam=nparam,
                ncnot=ncnot,
                circuit=circuit,
                basis_indices=ao_to_basis,
                min_coverage=min_coverage,
                fixed_paulis=fixed_paulis,
                )
        self.ansatz = self.ansatz_circuit.name
        self.topology = self.ansatz_circuit.topology
        self.nparam = self.ansatz_circuit.nparam
        self.requested_nparam = self.ansatz_circuit.requested_nparam
        self.ncnot = self.ansatz_circuit.ncnot
        self.npauli = self.ansatz_circuit.npauli

        self.ntheta = self.nkpts * self.nparam
        self.neta = 0 if self.fixed_occupation else self.nkpts * self.N
        npaired = self.nelec // 2
        fixed_occ = [2.0] * npaired
        if self.nelec % 2:
            fixed_occ.append(1.0)
        fixed_occ += [0.0] * (self.N - len(fixed_occ))
        self.fixed_occ = jnp.asarray(
            [fixed_occ] * self.nkpts, dtype=float)

        self.device_name = device_name
        if self.device_name == 'default.qubit':
            self.dev = qp.device('default.qubit', wires=self.M)
        elif self.device_name == 'ionq.simulator':
            self.dev = qp.device(
                'ionq.simulator', wires=self.M, noise_model='forte-1',
                dry_run=self.dry_run)
        elif self.device_name == 'ionq.qpu':
            self.dev = qp.device(
                'ionq.qpu', wires=self.M, backend='forte-1',
                dry_run=self.dry_run)
        else:
            raise ValueError(f'Unknown quantum device: {self.device_name}')

        #self.qnode_eks = qp.QNode(self.circuit_eks, self.dev)
        if self.is_qpu:
            self.shots = shots
            self.gamma = self.gen_gamma()
            self.qnode = qp.QNode(self.circuit_rho_qpu, self.dev, interface='jax', diff_method='parameter-shift')
            self.qnode = qp.set_shots(self.qnode, shots=self.shots)
            self.get_rho_raw = self.get_rho_raw_qpu
            fn_suffix = (
                f'_{self.device_name.replace(".", "-")}_shots{self.shots}')
            if self.dry_run:
                fn_suffix += '_dry-run'
        else:
            self.shots = None
            self.gamma = None
            self.qnode = qp.QNode(self.circuit_rho_cpu, self.dev, interface='jax', diff_method='backprop')
            self.get_rho_raw = self.get_rho_raw_cpu
            fn_suffix = ''
        print(f'device={self.device_name}\nis_qpu={self.is_qpu}\n'
              f'shots={self.shots}\ndry_run={self.dry_run}')

        if self.resume_from:
            self.fn_out = os.path.abspath(self.resume_from)
            if not os.path.isfile(self.fn_out):
                raise FileNotFoundError(
                    f'Resume checkpoint does not exist: {self.fn_out}')
            if not self.fn_out.endswith('.out'):
                raise ValueError('--resume checkpoint must end in .out')
            self.fn_in = self.fn_out[:-4] + '.in'
            fn_in_fd = None
        else:
            self.fn_in, self.fn_out, fn_in_fd = reserve_result_paths(
                self.fn_head + fn_suffix)
        config = (
        f'fn_in  {self.fn_in}\n'
        f'fn_out {self.fn_out}\n'
        f'\n'
        f'device {self.device_name}\n'
        f'is_qpu {int(self.is_qpu)}\n'
        f'shots  {self.shots}\n'
        f'dry_run {int(self.dry_run)}\n'
        f'\n'
        f'N     {self.N}\n'
        f'nelec {self.nelec}\n'
        f'ncell {self.ncell}\n'
        f'nkpts {self.nkpts}\n'
        f'kpts\n{self.mf.kpts}\n'
        f'\n'
        f'M      {self.M}\n'
        f'natom_bits {self.natom_bits}\n'
        f'nlocal_bits {self.nlocal_bits}\n'
        f'atom_wires {self.atom_wires}\n'
        f'local_wires {self.local_wires}\n'
        f'ao_to_basis {np.asarray(self.ao_to_basis)}\n'
        f'npauli {self.npauli}\n'
        f'\n'
        f'ansatz   {self.ansatz}\n'
        f'topology {self.topology if self.ansatz == "rycn" else "n/a"}\n'
        f'nparam   {self.nparam}\n'
        f'ncnot    {self.ncnot}\n'
        f'nintra   {self.ansatz_circuit.nintra}\n'
        f'ninter   {self.ansatz_circuit.ninter}\n'
        f'paulis   {self.ansatz_circuit.pauli}\n'
        f'ground_state {int(self.ground_state)}\n'
        f'fixed_occupation {int(self.fixed_occupation)}\n'
        f'fock_occupation {int(self.fock_occupation)}\n'
        f'sigma       {self.mf.sigma}\n'
        f'\n'
        f'ntheta {self.ntheta}\n'
        f'neta   {self.neta}\n'
        f'\n'
        )
        print(config)
        if fn_in_fd is None:
            with open(self.fn_in, 'a') as f:
                f.write(f'\nresume_from {self.fn_out}\n{config}')
        else:
            with os.fdopen(fn_in_fd, 'w') as f:
                f.write(config)

        self.h1e = jnp.asarray(self.mf.get_hcore(self.mat).reshape(self.nkpts, self.N, self.N))
        self.s1e = jnp.asarray(self.mf.get_ovlp(self.mat).reshape(self.nkpts, self.N, self.N))

        #self.shp, self.shp_dagger = [], []
        self.shm, self.shm_dagger = [], []
        for k in range(self.nkpts):
            #self.shp += [scipy.linalg.fractional_matrix_power(self.s1e[k],  0.5)]
            self.shm += [scipy.linalg.fractional_matrix_power(self.s1e[k], -0.5)]
            #self.shp_dagger += [self.shp[k].conj().T]
            self.shm_dagger += [self.shm[k].conj().T]
        #self.shp, self.shp_dagger = jnp.asarray(self.shp), jnp.asarray(self.shp_dagger)
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
                qp.RY(theta[n, m, 0], wires=m)
                qp.CNOT(wires=[m, (m + 1) % self.M])
    """

    def prep_circuit(self, theta, basis):
        qp.BasisState(basis, wires=self.wires)
        self.ansatz_circuit.apply(theta)

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
    def get_rho_raw_cpu(self, theta, fock):
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

    def get_rho_raw_cpu(self, theta):
        def get_rho_raw_k(theta_k, shm_k, shm_dagger_k):
            def get_rho_raw_kb(basis):
                rho_basis = self.qnode(theta_k, basis)
                return rho_basis[
                    self.ao_to_basis[:, None], self.ao_to_basis[None, :]]
            return jax.vmap(get_rho_raw_kb)(self.bases)
        rho_raw = jax.vmap(get_rho_raw_k)(theta, self.shm, self.shm_dagger)
        return rho_raw

    def get_rho_raw_qpu(self, theta):
        rho_raw = []
        for theta_k, shm_k, shm_dagger_k in zip(theta, self.shm, self.shm_dagger):
            rho_raw_k = []
            for basis in self.bases:
                rho_raw_kb = jnp.asarray(self.qnode(theta_k, basis)).reshape(self.N, self.N)
                rho_raw_k.append(rho_raw_kb)
            rho_raw.append(jnp.stack(rho_raw_k))
        rho_raw = jnp.stack(rho_raw)
        return rho_raw

    def get_occ(self, eta, max_itr=20):
        eta = eta.reshape(self.nkpts, self.N)
        nelec_tot = self.nkpts * self.nelec

        def newton_raphson(_, x):
            p  = jax.nn.sigmoid(eta + x) # 1 / (1 + exp[-(eta + x)])
            f  = 2.0 * jnp.sum(p) - nelec_tot
            df = 2.0 * jnp.sum(p * (1.0 - p))
            return x - f / jnp.maximum(df, 1e-12)

        mu = jax.lax.fori_loop(0, max_itr, newton_raphson, jnp.asarray(0.0))
        return 2.0 * jax.nn.sigmoid(eta + mu)

    def optimization_occ(self, eta):
        """Return fixed Aufbau occupations or variational occupations."""
        if self.fixed_occupation:
            return self.fixed_occ
        return self.get_occ(eta)

    def eta_gradient_norm(self, grad_eta):
        """Return zero when eta is absent from fixed-occupation optimization."""
        if self.fixed_occupation:
            if grad_eta.size:
                raise ValueError('fixed occupation must not contain eta')
            return jnp.asarray(0.0)
        return jnp.linalg.norm(grad_eta) / jnp.sqrt(grad_eta.size)

    def convergence_tolerances(self, tol):
        """Return gradient and free-energy tolerances for this backend."""
        if self.device_name == 'ionq.qpu':
            return 1e-2, 1e-3
        return tol, tol

    """ quDFTv2
    def get_rho(self, theta, fock):
        theta = theta.reshape(self.nkpts, self.nparam)
        rho_raw = self.shm[:, None, :, :] @ self.get_rho_raw(theta) @ self.shm_dagger[:, None, :, :] # orth -> AO
        eks = jnp.einsum('kij,kbji->kb', fock, rho_raw).real
        occ = jnp.asarray(self.mf.get_occ(eks))
        rho = jnp.einsum('kb,kbij->kij', occ, rho_raw)
        return rho, occ
    """

    def get_rho(self, theta, occ):
        theta = theta.reshape(self.nkpts, self.nparam)
        rho_raw = self.shm[:, None, :, :] @ self.get_rho_raw(theta) @ self.shm_dagger[:, None, :, :] # orth -> AO
        rho = jnp.einsum('kb,kbij->kij', occ, rho_raw)
        return rho

    def get_orbital_projectors(self, theta):
        """Return AO-basis projectors for every circuit orbital."""
        theta = theta.reshape(self.nkpts, self.nparam)
        return (self.shm[:, None, :, :] @ self.get_rho_raw(theta)
                @ self.shm_dagger[:, None, :, :])

    def get_fock_occupation(self, theta, fock):
        """Update occupations from Fock expectation values of circuit orbitals.

        The Fock matrix is treated as the fixed matrix from the preceding SCF
        iteration.  Occupations are deliberately evaluated outside automatic
        differentiation: they are an SCF update, not variational parameters.
        """
        projectors = self.get_orbital_projectors(theta)
        eks = jnp.einsum('kij,kbji->kb', fock, projectors).real
        occ = self.mf.get_occ(np.asarray(jax.device_get(eks)))
        return jnp.asarray(occ), eks

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

    """ quDFTv2
    def cost_qudft(self, theta, fock_old):
        rho, occ = self.get_rho(theta, fock_old)
        vhf = self.mf.get_veff(self.mat, rho)
        etot = self.mf.energy_tot(rho, self.h1e, vhf)
        fock = self.mf.get_fock(self.h1e, self.s1e, vhf, rho)

        w = 1.0 / self.nkpts
        nelec = w * jnp.einsum('kij,kji->', self.s1e, rho).real
        e1 = w * jnp.einsum('kij,kji->', self.h1e, rho).real
        eh = vhf.ecoul.real
        exc = vhf.exc.real
        enuc = self.mf.energy_nuc()
        return etot, (fock, e1, eh, exc)
    """

    def cost_qudft(self, param):
        theta, eta = param[:self.ntheta], param[self.ntheta:]
        occ = self.optimization_occ(eta)
        rho = self.get_rho(theta, occ)
        vhf = self.mf.get_veff(self.mat, rho)
        etot = self.mf.energy_tot(rho, self.h1e, vhf)

        w = 1 / self.nkpts
        p = occ / 2.0
        entropy = -(2.0 * w) * jnp.sum(
                xlogy(p, p) + xlogy(1.0 - p, 1.0 - p))
        efree = etot if self.ground_state else etot - self.mf.sigma * entropy

        e1 = w * jnp.einsum('kij,kji->', self.h1e, rho).real
        eh = vhf.ecoul.real
        exc = vhf.exc.real
        #enuc = self.mf.energy_nuc()
        return efree, (etot, entropy, e1, eh, exc)

    def cost_qudft_occ(self, theta, occ):
        """Free energy for externally supplied, non-variational occupations."""
        rho = self.get_rho(theta, occ)
        vhf = self.mf.get_veff(self.mat, rho)
        etot = self.mf.energy_tot(rho, self.h1e, vhf)

        w = 1 / self.nkpts
        p = occ / 2.0
        entropy = -(2.0 * w) * jnp.sum(
                xlogy(p, p) + xlogy(1.0 - p, 1.0 - p))
        efree = etot if self.ground_state else etot - self.mf.sigma * entropy
        e1 = w * jnp.einsum('kij,kji->', self.h1e, rho).real
        eh = vhf.ecoul.real
        exc = vhf.exc.real
        fock = self.mf.get_fock(self.h1e, self.s1e, vhf, rho)
        return efree, (etot, entropy, e1, eh, exc, rho, fock)

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

    def qudftv2(self, theta, eta, opt='nadam', lr_theta=0.02, lr_eta=0.1,
                tol=1e-5, max_itr=1000, record_gradients=False,
                resume=False):
        if self.fock_occupation:
            return self.qudftv2_fock_occ(
                theta, opt=opt, lr_theta=lr_theta, tol=tol,
                max_itr=max_itr, record_gradients=record_gradients)
        if self.ansatz in ('au-swap', 'au-grow'):
            return self.qudftv2_adaptive(
                theta, eta, opt=opt, lr_theta=lr_theta, lr_eta=lr_eta,
                tol=tol, max_itr=max_itr)
        has_eta = not self.fixed_occupation
        grad_tol, efree_tol = self.convergence_tolerances(tol)
        config = (
        f'opt      {opt}\n'
        f'lr_theta {lr_theta}\n'
        + (f'lr_eta   {lr_eta}\n' if has_eta else '') +
        f'grad_tol {grad_tol}\n'
        f'efree_tol {efree_tol}\n'
        f'max_itr  {max_itr}\n'
        f'\n'
        )
        print(config)
        with open(self.fn_in, 'a') as f: f.write(config)

        param = jnp.concatenate((theta, eta))
        vng = jax.value_and_grad(self.cost_qudft, has_aux=True)

        solver = getattr(optax, opt)(learning_rate=lr_theta)
        state = solver.init(param)
        start_iteration = 0
        efree_previous = None
        resume_mode = None
        if resume:
            (param, state, start_iteration,
             efree_previous, resume_mode) = self.load_optimizer_checkpoint(
                 solver, param)
            print(f'Resuming {self.fn_out} at itr {start_iteration} '
                  f'({resume_mode})')
        eta_label = f' lr_eta={lr_eta}' if has_eta else ''
        print(f'\noptimization ({opt} lr_theta={lr_theta}{eta_label})')

        t0 = time.time()
        self.init_iteration_history(t0, resume=resume)
        if not resume:
            self.save_optimizer_checkpoint(
                param, state, next_iteration=0, efree_previous=None)
        elif resume_mode.startswith('legacy'):
            self.save_optimizer_checkpoint(
                param, state, next_iteration=start_iteration,
                efree_previous=efree_previous)
        header = (f'{"itr":>5s} {"efree":>12s} {"etot":>12s} '
                  f'{"entropy":>12s} {"e1":>12s} {"eh":>12s} '
                  f'{"exc":>12s} {"grad_theta":>12s}')
        if has_eta:
            header += f' {"grad_eta":>12s}'
        print(header + f' {"delta_efree":>12s}')
        converged = False
        grad_theta_history = []
        efree_history = []
        completed_iterations = start_iteration
        grad_theta_l2 = jnp.asarray(np.nan)
        grad_eta_l2 = jnp.asarray(0.0 if not has_eta else np.nan)
        delta_efree = jnp.asarray(np.nan)
        if resume and start_iteration:
            with h5py.File(self.fn_out, 'r') as f:
                history = f['history']
                if len(history.get('iteration', ())):
                    grad_theta_l2 = jnp.asarray(history['grad_theta'][-1])
                    delta_efree = jnp.asarray(history['delta_efree'][-1])
                    if has_eta and 'grad_eta' in history:
                        grad_eta_l2 = jnp.asarray(history['grad_eta'][-1])
        for i in range(start_iteration, max_itr):
            (efree, (etot, entropy, e1, eh, exc)), grad = vng(param)
            grad_theta = grad[:self.ntheta]
            grad_theta_l2 = (jnp.linalg.norm(grad_theta) /
                             jnp.sqrt(grad_theta.size))
            grad_eta = grad[self.ntheta:]
            grad_eta_l2 = self.eta_gradient_norm(grad_eta)
            delta_efree = (jnp.inf if efree_previous is None
                           else jnp.abs(efree - efree_previous))
            if record_gradients:
                grad_theta_history.append(np.asarray(grad_theta))
                efree_history.append(float(efree))
            row = (f'{i:5d} {efree:12f} {etot:12f} {entropy:12f} '
                   f'{e1:12f} {eh:12f} {exc:12f} '
                   f'{grad_theta_l2:12f}')
            if has_eta:
                row += f' {grad_eta_l2:12f}'
            print(row + f' {delta_efree:12f}')

            checkpoint = {
                'iteration': i,
                'efree': efree,
                'etot': etot,
                'entropy': entropy,
                'e1': e1,
                'eh': eh,
                'exc': exc,
                'grad_theta': grad_theta_l2,
                'grad_theta_vector': grad_theta,
                'delta_efree': delta_efree,
                'theta': param[:self.ntheta],
                'occ': self.optimization_occ(param[self.ntheta:]),
            }
            if has_eta:
                checkpoint.update(
                    eta=param[self.ntheta:], grad_eta=grad_eta_l2)
            self.append_iteration_history(**checkpoint)
            completed_iterations = i + 1

            if (grad_theta_l2 < grad_tol and
                    grad_eta_l2 < grad_tol and
                    delta_efree < efree_tol):
                converged = True
                break
            efree_previous = efree

            update, state = solver.update(grad, state, param)
            if has_eta:
                update = update.at[self.ntheta:].multiply(lr_eta / lr_theta)
            param = optax.apply_updates(param, update)
            self.save_optimizer_checkpoint(
                param, state, next_iteration=i + 1,
                efree_previous=efree_previous)
        t1 = time.time()

        efree, (etot, entropy, e1, eh, exc) = self.cost_qudft(param)
        theta, eta = param[:self.ntheta], param[self.ntheta:]
        occ = self.optimization_occ(eta)
        rho = self.get_rho(theta, occ)

        gradient_result = {}
        if record_gradients:
            gradient_result = {
                'gradient_paulis': np.asarray(
                    self.ansatz_circuit.pauli, dtype='S'),
                'grad_theta_history': np.asarray(grad_theta_history),
                'efree_history': np.asarray(efree_history),
            }
        occupation_result = {} if self.fixed_occupation else {
            'eta': eta,
            'grad_eta': grad_eta_l2,
        }

        self.save_result(
                self.qudftv2.__name__,
                t0=t0,
                t1=t1,
                theta=theta,
                occ=occ,
                rho=rho,
                efree=efree,
                etot=etot,
                entropy=entropy,
                e1=e1,
                eh=eh,
                exc=exc,
                niter=completed_iterations,
                converged=converged,
                grad_theta=grad_theta_l2,
                delta_efree=delta_efree,
                grad_tol=grad_tol,
                efree_tol=efree_tol,
                **gradient_result,
                **occupation_result,
                )
        return param

    def qudftv2_fock_occ(self, theta, opt='nadam', lr_theta=0.02,
                         tol=1e-5, max_itr=1000,
                         record_gradients=False):
        """Optimize circuit angles while occupations follow an SCF update."""
        if self.ansatz in ('au-swap', 'au-grow'):
            raise ValueError(
                    '--fock-occ is not yet supported with adaptive ansatzes')

        grad_tol, efree_tol = self.convergence_tolerances(tol)
        config = (
            f'opt      {opt}\n'
            f'lr_theta {lr_theta}\n'
            f'grad_tol {grad_tol}\n'
            f'efree_tol {efree_tol}\n'
            f'max_itr  {max_itr}\n'
            f'occupation fock\n\n'
        )
        print(config)
        with open(self.fn_in, 'a') as f:
            f.write(config)

        rho_initial = jnp.asarray(self.mf.get_init_guess(
            self.mat, self.mf.init_guess, s1e=self.s1e))
        vhf_initial = self.mf.get_veff(self.mat, rho_initial)
        fock = self.mf.get_fock(
            self.h1e, self.s1e, vhf_initial, rho_initial)

        vng = jax.value_and_grad(
            self.cost_qudft_occ, argnums=0, has_aux=True)
        solver = getattr(optax, opt)(learning_rate=lr_theta)
        state = solver.init(theta)

        print(f'\nFock-occupation optimization ({opt} lr_theta={lr_theta})')
        print(f'{"itr":>5s} {"efree":>12s} {"etot":>12s} '
              f'{"entropy":>12s} {"grad_theta":>12s} '
              f'{"delta_occ":>12s} {"delta_efree":>12s}')

        t0 = time.time()
        self.init_iteration_history(t0)
        efree_previous = None
        occ_previous = None
        converged = False
        grad_theta_history = []
        efree_history = []
        occ_history = []
        eks_history = []

        for i in range(max_itr):
            occ, eks = self.get_fock_occupation(theta, fock)
            ((efree, (etot, entropy, e1, eh, exc, rho, fock_new)),
             grad_theta) = vng(theta, occ)
            grad_theta_l2 = (jnp.linalg.norm(grad_theta) /
                             jnp.sqrt(grad_theta.size))
            delta_occ = (jnp.inf if occ_previous is None else
                         jnp.max(jnp.abs(occ - occ_previous)))
            delta_efree = (jnp.inf if efree_previous is None else
                           jnp.abs(efree - efree_previous))

            occ_history.append(np.asarray(occ))
            eks_history.append(np.asarray(eks))
            if record_gradients:
                grad_theta_history.append(np.asarray(grad_theta))
                efree_history.append(float(efree))

            print(f'{i:5d} {efree:12f} {etot:12f} {entropy:12f} '
                  f'{grad_theta_l2:12f} {delta_occ:12f} '
                  f'{delta_efree:12f}')
            self.append_iteration_history(
                iteration=i,
                efree=efree,
                etot=etot,
                entropy=entropy,
                e1=e1,
                eh=eh,
                exc=exc,
                grad_theta=grad_theta_l2,
                grad_theta_vector=grad_theta,
                delta_occ=delta_occ,
                delta_efree=delta_efree,
                theta=theta,
                occ=occ,
                eks=eks,
                rho=rho,
                fock=fock_new,
            )

            if (grad_theta_l2 < grad_tol and delta_occ < efree_tol and
                    delta_efree < efree_tol):
                converged = True
                fock = fock_new
                break

            efree_previous = efree
            occ_previous = occ
            fock = fock_new
            update, state = solver.update(grad_theta, state, theta)
            theta = optax.apply_updates(theta, update)

        t1 = time.time()
        occ, eks = self.get_fock_occupation(theta, fock)
        efree, (etot, entropy, e1, eh, exc, rho, fock) = (
            self.cost_qudft_occ(theta, occ))

        gradient_result = {}
        if record_gradients:
            gradient_result = {
                'gradient_paulis': np.asarray(
                    self.ansatz_circuit.pauli, dtype='S'),
                'grad_theta_history': np.asarray(grad_theta_history),
                'efree_history': np.asarray(efree_history),
            }

        self.save_result(
            self.qudftv2_fock_occ.__name__,
            t0=t0, t1=t1, theta=theta,
            eta=np.asarray([]), occ=occ, rho=rho, eks=eks, fock=fock,
            efree=efree, etot=etot, entropy=entropy,
            e1=e1, eh=eh, exc=exc,
            niter=i + 1, converged=converged,
            grad_theta=grad_theta_l2, grad_eta=0.0,
            delta_occ=delta_occ, delta_efree=delta_efree,
            grad_tol=grad_tol, efree_tol=efree_tol,
            occ_history=np.asarray(occ_history),
            eks_history=np.asarray(eks_history),
            **gradient_result,
        )
        return theta

    def qudftv2_swap(self, theta, eta, opt='nadam', lr_theta=0.02,
                     lr_eta=0.1, tol=1e-5, max_itr=1000):
        """Optimize a fixed-size Pauli circuit with approved 1-to-1 swaps."""
        if self.is_qpu:
            raise ValueError('au-swap currently requires CPU simulation')

        active_tol = 1e-4
        excluded_tol = 5e-4
        patience_required = 10
        trial_steps = 20
        min_improvement = 1e-6
        active_count = self.requested_nparam
        pool_words = list(self.ansatz_circuit.pauli_pool)
        pool_index = {word: index for index, word in enumerate(pool_words)}

        config = (
            f'opt      {opt}\n'
            f'lr_theta {lr_theta}\n'
            f'lr_eta   {lr_eta}\n'
            f'tol      {tol}\n'
            f'max_itr  {max_itr}\n'
            f'adaptive_mode swap\n'
            f'adaptive_pool {len(pool_words)}\n'
            f'active_gates {active_count}\n'
            f'active_tol {active_tol}\n'
            f'excluded_tol {excluded_tol}\n'
            f'patience {patience_required}\n'
            f'trial_steps {trial_steps}\n'
            f'min_improvement {min_improvement}\n\n'
        )
        print(config)
        with open(self.fn_in, 'a') as f:
            f.write(config)

        def set_circuit(words):
            """Install exactly the supplied Pauli rotations in the QNode."""
            self.ansatz_circuit.pauli = list(words)
            self.ansatz_circuit.nparam = len(words)
            self.nparam = len(words)
            self.ntheta = self.nkpts * self.nparam
            self.qnode = qp.QNode(
                self.circuit_rho_cpu, self.dev,
                interface='jax', diff_method='backprop')

        vng = jax.value_and_grad(self.cost_qudft, has_aux=True)

        # Rank the full pool without ever installing the full pool as one
        # circuit. Each insertion direction is evaluated with one PauliRot.
        initial_score = []
        for word in pool_words:
            set_circuit([word])
            probe = jnp.concatenate((jnp.zeros(self.ntheta), eta))
            (_, _), probe_grad = vng(probe)
            score = jnp.linalg.norm(
                probe_grad[:self.ntheta].reshape(self.nkpts, 1), axis=0)[0]
            initial_score.append(float(score))

        initial_order = np.argsort(-np.asarray(initial_score), kind='stable')
        active_words = [pool_words[int(index)]
                        for index in initial_order[:active_count]]
        excluded_words = [word for word in pool_words
                          if word not in active_words]
        set_circuit(active_words)

        param = jnp.concatenate((jnp.zeros(self.ntheta), eta))
        solver = getattr(optax, opt)(learning_rate=lr_theta)
        state = solver.init(param)
        pool_gradient_history = []
        active_index_history = []
        efree_history = []
        grad_eta_history = []
        occ_history = []
        swap_iteration = []
        swap_energy_change = []

        print(f'\nau-swap optimization ({opt} lr_theta={lr_theta} '
              f'lr_eta={lr_eta}, gates={active_count})')
        print(f'{"itr":>5s} {"efree":>12s} {"etot":>12s} '
              f'{"entropy":>12s} {"excluded":>12s} '
              f'{"active_grad":>12s} {"grad_eta":>12s} '
              f'{"delta_efree":>12s}')

        t0 = time.time()
        efree_previous = None
        converged = False
        patience = 0
        excluded_max = 0.0 if not excluded_words else jnp.inf
        print(f'active[0]={active_words}')

        for i in range(max_itr):
            structure_changed = False
            (efree, (etot, entropy, e1, eh, exc)), grad = vng(param)
            grad_theta = grad[:self.ntheta]
            grad_eta = grad[self.ntheta:]
            active_grad_l2 = jnp.linalg.norm(grad_theta)
            grad_eta_l2 = self.eta_gradient_norm(grad_eta)
            delta_efree = (jnp.inf if efree_previous is None else
                           jnp.abs(efree - efree_previous))

            grad_theta_k = np.asarray(grad_theta).reshape(
                self.nkpts, active_count)
            pool_snapshot = np.full(len(pool_words), np.nan)
            for position, word in enumerate(active_words):
                pool_snapshot[pool_index[word]] = np.linalg.norm(
                    grad_theta_k[:, position])
            active_indices = np.full(len(pool_words), -1, dtype=int)
            active_indices[:active_count] = [
                pool_index[word] for word in active_words]
            active_index_history.append(active_indices)
            efree_history.append(float(efree))
            grad_eta_history.append(float(grad_eta_l2))
            occ_history.append(np.asarray(
                self.optimization_occ(param[self.ntheta:])))

            patience = patience + 1 if active_grad_l2 < active_tol else 0
            if patience >= patience_required and excluded_words:
                active_theta = np.asarray(
                    param[:self.ntheta]).reshape(
                        self.nkpts, active_count)
                eta_current = param[self.ntheta:]
                excluded_grad = []

                # Candidate circuits contain P+1 gates only while evaluating
                # the zero-parameter insertion derivative. The persistent
                # optimization circuit is restored to exactly P gates below.
                for candidate in excluded_words:
                    set_circuit(active_words + [candidate])
                    probe_theta = np.concatenate((
                        active_theta,
                        np.zeros((self.nkpts, 1))), axis=1)
                    probe = jnp.concatenate((
                        jnp.asarray(probe_theta).reshape(-1), eta_current))
                    (_, _), probe_grad = vng(probe)
                    probe_grad_k = np.asarray(
                        probe_grad[:self.ntheta]).reshape(
                            self.nkpts, active_count + 1)
                    score = np.linalg.norm(probe_grad_k[:, -1])
                    excluded_grad.append(score)
                    pool_snapshot[pool_index[candidate]] = score

                set_circuit(active_words)
                param = jnp.concatenate((
                    jnp.asarray(active_theta).reshape(-1), eta_current))
                excluded_grad = np.asarray(excluded_grad)
                candidate_offset = int(np.argmax(np.abs(excluded_grad)))
                excluded_max = float(abs(excluded_grad[candidate_offset]))
                candidate = excluded_words[candidate_offset]

                if excluded_max > excluded_tol:
                    current_energy = float(efree)
                    best = None
                    for remove in range(active_count):
                        trial_words = (
                            active_words[:remove]
                            + active_words[remove + 1:]
                            + [candidate])
                        set_circuit(trial_words)
                        kept_theta = np.delete(active_theta, remove, axis=1)
                        trial_theta = np.concatenate((
                            kept_theta,
                            np.zeros((self.nkpts, 1))), axis=1)
                        trial_param = jnp.concatenate((
                            jnp.asarray(trial_theta).reshape(-1),
                            eta_current))
                        trial_solver = getattr(
                            optax, opt)(learning_rate=lr_theta)
                        trial_state = trial_solver.init(trial_param)
                        for _ in range(trial_steps):
                            (_, _), trial_grad = vng(trial_param)
                            update, trial_state = trial_solver.update(
                                trial_grad, trial_state, trial_param)
                            update = update.at[self.ntheta:].multiply(
                                lr_eta / lr_theta)
                            trial_param = optax.apply_updates(
                                trial_param, update)
                        trial_energy = float(
                            self.cost_qudft(trial_param)[0])
                        if best is None or trial_energy < best[0]:
                            best = (trial_energy, trial_words, trial_param)

                    if best[0] < current_energy - min_improvement:
                        energy_change = best[0] - current_energy
                        active_words = best[1]
                        excluded_words = [
                            word for word in pool_words
                            if word not in active_words]
                        set_circuit(active_words)
                        param = best[2]
                        state = solver.init(param)
                        swap_iteration.append(i)
                        swap_energy_change.append(energy_change)
                        structure_changed = True
                        print(f'swap[{i}] dE={energy_change:+.6e} '
                              f'active={active_words}')
                    else:
                        set_circuit(active_words)
                        param = jnp.concatenate((
                            jnp.asarray(active_theta).reshape(-1),
                            eta_current))
                    patience = 0

            pool_gradient_history.append(pool_snapshot)
            print(f'{i:5d} {efree:12f} {etot:12f} {entropy:12f} '
                  f'{excluded_max:12f} {active_grad_l2:12f} '
                  f'{grad_eta_l2:12f} {delta_efree:12f}')

            if (active_grad_l2 < tol and excluded_max < excluded_tol and
                    grad_eta_l2 < tol and delta_efree < tol):
                converged = True
                break
            if structure_changed:
                efree_previous = float(efree)
                continue
            efree_previous = efree
            update, state = solver.update(grad, state, param)
            update = update.at[self.ntheta:].multiply(lr_eta / lr_theta)
            param = optax.apply_updates(param, update)

        t1 = time.time()
        # The saved/final QNode is guaranteed to contain exactly P gates.
        set_circuit(active_words)
        efree, (etot, entropy, e1, eh, exc) = self.cost_qudft(param)
        theta, eta = param[:self.ntheta], param[self.ntheta:]
        occ = self.optimization_occ(eta)
        rho = self.get_rho(theta, occ)
        self.save_result(
            self.qudftv2_swap.__name__, t0=t0, t1=t1,
            theta=theta, eta=eta, occ=occ, rho=rho,
            efree=efree, etot=etot, entropy=entropy,
            e1=e1, eh=eh, exc=exc, niter=i + 1,
            converged=converged, grad_theta=active_grad_l2,
            grad_eta=grad_eta_l2, delta_efree=delta_efree,
            gradient_paulis=np.asarray(pool_words, dtype='S'),
            active_paulis=np.asarray(active_words, dtype='S'),
            circuit_gate_count=active_count,
            grad_theta_history=np.asarray(pool_gradient_history),
            active_index_history=np.asarray(active_index_history),
            efree_history=np.asarray(efree_history),
            grad_eta_history=np.asarray(grad_eta_history),
            occ_history=np.asarray(occ_history),
            adaptive_mode=np.bytes_('swap'),
            active_count=active_count,
            swap_iteration=np.asarray(swap_iteration, dtype=int),
            swap_energy_change=np.asarray(swap_energy_change))
        return param

    def qudftv2_adaptive(self, theta, eta, opt='nadam', lr_theta=0.02,
                         lr_eta=0.1, tol=1e-5, max_itr=1000):
        """Optimize either a fixed-size swap or incrementally growing circuit."""
        if self.is_qpu:
            raise ValueError('adaptive ansatzes currently require CPU simulation')
        adaptive_mode = 'swap' if self.ansatz == 'au-swap' else 'grow'
        has_eta = not self.fixed_occupation
        active_tol = 1e-4
        excluded_tol = 5e-4
        patience_required = 10
        trial_steps = 20
        min_improvement = 1e-6
        max_active = min(self.requested_nparam, self.nparam)
        config = (
            f'opt      {opt}\n'
            f'lr_theta {lr_theta}\n'
            + (f'lr_eta   {lr_eta}\n' if has_eta else '') +
            f'tol      {tol}\n'
            f'max_itr  {max_itr}\n'
            f'adaptive_mode {adaptive_mode}\n'
            f'adaptive_pool {self.nparam}\n'
            f'max_active {max_active}\n'
            f'active_tol {active_tol}\n'
            f'excluded_tol {excluded_tol}\n'
            f'patience {patience_required}\n'
            f'trial_steps {trial_steps}\n'
            f'min_improvement {min_improvement}\n\n'
        )
        print(config)
        with open(self.fn_in, 'a') as f:
            f.write(config)

        pool_words = list(self.ansatz_circuit.pauli)
        pool_index = {word: index for index, word in enumerate(pool_words)}

        def set_order(words):
            self.ansatz_circuit.pauli = list(words)
            self.qnode = qp.QNode(
                self.circuit_rho_cpu, self.dev,
                interface='jax', diff_method='backprop')

        def make_active_mask(count):
            mask_k = jnp.zeros(self.nparam).at[:count].set(1.0)
            return jnp.tile(mask_k, self.nkpts)

        vng = jax.value_and_grad(self.cost_qudft, has_aux=True)
        zero_param = jnp.concatenate((jnp.zeros_like(theta), eta))
        (_, _), initial_grad = vng(zero_param)
        initial_grad_k = np.asarray(
            initial_grad[:self.ntheta]).reshape(self.nkpts, self.nparam)
        initial_score = np.linalg.norm(initial_grad_k, axis=0)
        initial_order = np.argsort(-initial_score, kind='stable')
        active_count = max_active if adaptive_mode == 'swap' else 1
        initial_indices = list(map(int, initial_order[:active_count]))
        active_words = [pool_words[index] for index in initial_indices]
        excluded_words = [word for word in pool_words if word not in active_words]
        set_order(active_words + excluded_words)
        param = jnp.concatenate((jnp.zeros_like(theta), eta))
        active_count = len(active_words)
        active_mask = make_active_mask(active_count)
        solver = getattr(optax, opt)(learning_rate=lr_theta)
        state = solver.init(param)
        pool_gradient_history = []
        active_index_history = []
        efree_history = []
        grad_eta_history = []
        occ_history = []
        swap_iteration = []
        swap_energy_change = []
        grow_iteration = []
        grow_pauli = []

        eta_label = f' lr_eta={lr_eta}' if has_eta else ''
        print(f'\n{self.ansatz} optimization '
              f'({opt} lr_theta={lr_theta}{eta_label})')
        header = (f'{"itr":>5s} {"efree":>12s} {"etot":>12s} '
                  f'{"entropy":>12s} {"excluded":>12s} '
                  f'{"active_grad":>12s}')
        if has_eta:
            header += f' {"grad_eta":>12s}'
        print(header + f' {"delta_efree":>12s}')
        t0 = time.time()
        self.init_iteration_history(t0)
        efree_previous = None
        converged = False
        patience = 0
        excluded_max = jnp.inf
        print(f'active[0]={active_words}')

        for i in range(max_itr):
            structure_changed = False
            (efree, (etot, entropy, e1, eh, exc)), full_grad = vng(param)
            theta_checkpoint = np.asarray(param[:self.ntheta])
            eta_checkpoint = np.asarray(param[self.ntheta:])
            occ_checkpoint = np.asarray(
                self.optimization_occ(param[self.ntheta:]))
            active_grad_theta = full_grad[:self.ntheta] * active_mask
            grad = full_grad.at[:self.ntheta].set(active_grad_theta)
            grad_eta = grad[self.ntheta:]
            active_grad_l2 = jnp.linalg.norm(active_grad_theta)
            grad_eta_l2 = self.eta_gradient_norm(grad_eta)
            delta_efree = (jnp.inf if efree_previous is None else
                           jnp.abs(efree - efree_previous))

            grad_theta_k = np.asarray(
                full_grad[:self.ntheta]).reshape(self.nkpts, self.nparam)
            pool_snapshot = np.full(self.nparam, np.nan)
            for position, word in enumerate(active_words):
                pool_snapshot[pool_index[word]] = np.linalg.norm(
                    grad_theta_k[:, position])
            active_indices = np.full(self.nparam, -1, dtype=int)
            active_indices[:active_count] = [
                pool_index[word] for word in active_words]
            active_index_history.append(active_indices)
            efree_history.append(float(efree))
            grad_eta_history.append(float(grad_eta_l2))
            occ_history.append(np.asarray(
                self.optimization_occ(param[self.ntheta:])))

            patience = patience + 1 if active_grad_l2 < active_tol else 0
            if patience >= patience_required and excluded_words:
                # In the current ordering, every excluded generator is
                # appended after the active circuit and evaluated at zero.
                excluded_grad_k = grad_theta_k[:, active_count:self.nparam]
                excluded_grad = np.linalg.norm(excluded_grad_k, axis=0)
                for offset, value in enumerate(excluded_grad, active_count):
                    pool_snapshot[pool_index[
                        self.ansatz_circuit.pauli[offset]]] = value
                candidate_offset = int(np.argmax(np.abs(excluded_grad)))
                excluded_max = float(abs(excluded_grad[candidate_offset]))
                candidate = excluded_words[candidate_offset]

                if (adaptive_mode == 'grow' and
                        excluded_max > excluded_tol and
                        active_count < max_active):
                    active_theta = np.asarray(
                        param[:self.ntheta]).reshape(
                            self.nkpts, self.nparam)[:, :active_count]
                    active_words.append(candidate)
                    excluded_words.remove(candidate)
                    set_order(active_words + excluded_words)
                    active_count += 1
                    theta_k = np.concatenate((
                        active_theta,
                        np.zeros((self.nkpts,
                                  self.nparam - active_count + 1))), axis=1)
                    param = jnp.concatenate((
                        jnp.asarray(theta_k).reshape(-1),
                        param[self.ntheta:]))
                    active_mask = make_active_mask(active_count)
                    state = solver.init(param)
                    grow_iteration.append(i)
                    grow_pauli.append(candidate)
                    structure_changed = True
                    print(f'grow[{i}] add={candidate} '
                          f'active_count={active_count}')
                    patience = 0

                elif adaptive_mode == 'swap' and excluded_max > excluded_tol:
                    current_energy = float(efree)
                    best = None
                    active_theta = np.asarray(
                        param[:self.ntheta]).reshape(
                            self.nkpts, self.nparam)[:, :active_count]
                    trial_mask = make_active_mask(active_count)
                    for remove in range(active_count):
                        trial_words = (
                            active_words[:remove]
                            + active_words[remove + 1:]
                            + [candidate])
                        trial_excluded = [
                            word for word in pool_words
                            if word not in trial_words]
                        set_order(trial_words + trial_excluded)
                        kept_theta = np.delete(active_theta, remove, axis=1)
                        trial_theta_k = np.concatenate((
                            kept_theta,
                            np.zeros((self.nkpts, 1)),
                            np.zeros((self.nkpts,
                                      self.nparam - active_count))), axis=1)
                        trial_param = jnp.concatenate((
                            jnp.asarray(trial_theta_k).reshape(-1),
                            param[self.ntheta:]))
                        trial_solver = getattr(
                            optax, opt)(learning_rate=lr_theta)
                        trial_state = trial_solver.init(trial_param)
                        for _ in range(trial_steps):
                            (_, _), trial_grad = vng(trial_param)
                            trial_grad = trial_grad.at[
                                :self.ntheta].multiply(trial_mask)
                            update, trial_state = trial_solver.update(
                                trial_grad, trial_state, trial_param)
                            update = update.at[self.ntheta:].multiply(
                                lr_eta / lr_theta)
                            trial_param = optax.apply_updates(
                                trial_param, update)
                            trial_param = trial_param.at[
                                :self.ntheta].multiply(trial_mask)
                        trial_energy = float(self.cost_qudft(trial_param)[0])
                        if best is None or trial_energy < best[0]:
                            best = (trial_energy, trial_words,
                                    trial_excluded, trial_param)

                    if best[0] < current_energy - min_improvement:
                        energy_change = best[0] - current_energy
                        active_words = best[1]
                        excluded_words = best[2]
                        set_order(active_words + excluded_words)
                        param = best[3]
                        state = solver.init(param)
                        swap_iteration.append(i)
                        swap_energy_change.append(energy_change)
                        structure_changed = True
                        print(f'swap[{i}] dE={energy_change:+.6e} '
                              f'active={active_words}')
                    else:
                        set_order(active_words + excluded_words)
                    patience = 0

            pool_gradient_history.append(pool_snapshot)
            row = (f'{i:5d} {efree:12f} {etot:12f} {entropy:12f} '
                   f'{excluded_max:12f} {active_grad_l2:12f}')
            if has_eta:
                row += f' {grad_eta_l2:12f}'
            print(row + f' {delta_efree:12f}')
            checkpoint = {
                'iteration': i,
                'efree': efree,
                'etot': etot,
                'entropy': entropy,
                'e1': e1,
                'eh': eh,
                'exc': exc,
                'excluded_grad': excluded_max,
                'active_grad': active_grad_l2,
                'delta_efree': delta_efree,
                'theta': theta_checkpoint,
                'occ': occ_checkpoint,
                'pool_gradient': pool_snapshot,
                'active_indices': active_indices,
                'active_count': active_count,
            }
            if has_eta:
                checkpoint.update(
                    eta=eta_checkpoint, grad_eta=grad_eta_l2)
            self.append_iteration_history(**checkpoint)

            pool_exhausted = not excluded_words
            growth_complete = (adaptive_mode == 'grow' and
                               active_count >= max_active)
            no_insert_direction = excluded_max < excluded_tol
            if (active_grad_l2 < tol and
                    (no_insert_direction or growth_complete or pool_exhausted) and
                    grad_eta_l2 < tol and
                    delta_efree < tol):
                converged = True
                break
            if structure_changed:
                efree_previous = float(efree)
                continue
            efree_previous = efree
            update, state = solver.update(grad, state, param)
            if has_eta:
                update = update.at[self.ntheta:].multiply(lr_eta / lr_theta)
            param = optax.apply_updates(param, update)
            param = param.at[:self.ntheta].multiply(active_mask)

        t1 = time.time()
        efree, (etot, entropy, e1, eh, exc) = self.cost_qudft(param)
        theta, eta = param[:self.ntheta], param[self.ntheta:]
        occ = self.optimization_occ(eta)
        rho = self.get_rho(theta, occ)
        occupation_result = {} if self.fixed_occupation else {
            'eta': eta,
            'grad_eta': grad_eta_l2,
            'grad_eta_history': np.asarray(grad_eta_history),
        }
        self.save_result(
            self.qudftv2_adaptive.__name__, t0=t0, t1=t1,
            theta=theta, occ=occ, rho=rho,
            efree=efree, etot=etot, entropy=entropy,
            e1=e1, eh=eh, exc=exc, niter=i + 1,
            converged=converged, grad_theta=active_grad_l2,
            delta_efree=delta_efree,
            gradient_paulis=np.asarray(
                pool_words, dtype='S'),
            grad_theta_history=np.asarray(pool_gradient_history),
            active_index_history=np.asarray(active_index_history),
            efree_history=np.asarray(efree_history),
            occ_history=np.asarray(occ_history),
            adaptive_mode=np.bytes_(adaptive_mode),
            active_count=active_count,
            swap_iteration=np.asarray(swap_iteration, dtype=int),
            swap_energy_change=np.asarray(swap_energy_change),
            grow_iteration=np.asarray(grow_iteration, dtype=int),
            grow_pauli=np.asarray(grow_pauli, dtype='S'),
            **occupation_result)
        return param

    def ansatz_circuit_coverage(self, word):
        """Return valid AO pairs coupled by one Pauli word."""
        nqubits = len(word)
        flip_mask = sum(
            1 << (nqubits - 1 - wire)
            for wire, operator in enumerate(word) if operator in 'XY')
        valid = set(map(int, np.asarray(self.ao_to_basis)))
        return {
            (state, state ^ flip_mask) for state in valid
            if state < (state ^ flip_mask) and (state ^ flip_mask) in valid
        }

    def save_optimizer_checkpoint(self, param, state, next_iteration,
                                  efree_previous):
        """Save the next parameters and complete Optax state atomically enough."""
        leaves, _ = jax.tree_util.tree_flatten(state)
        with h5py.File(self.fn_out, 'a') as f:
            if 'checkpoint' in f:
                del f['checkpoint']
            checkpoint = f.create_group('checkpoint')
            checkpoint.attrs['next_iteration'] = int(next_iteration)
            checkpoint.attrs['efree_previous'] = (
                np.nan if efree_previous is None else float(efree_previous))
            checkpoint.create_dataset('param', data=np.asarray(param))
            opt_state = checkpoint.create_group('opt_state')
            opt_state.attrs['leaf_count'] = len(leaves)
            for index, leaf in enumerate(leaves):
                opt_state.create_dataset(str(index), data=np.asarray(leaf))
            f.flush()

    def load_optimizer_checkpoint(self, solver, param_template):
        """Restore an exact checkpoint, or warm-start an older history file."""
        with h5py.File(self.fn_out, 'r+') as f:
            if 'checkpoint' in f:
                checkpoint = f['checkpoint']
                param = jnp.asarray(checkpoint['param'][...])
                if param.shape != param_template.shape:
                    raise ValueError(
                        f'Resume parameter shape {param.shape} does not match '
                        f'current shape {param_template.shape}')
                template_state = solver.init(param)
                template_leaves, tree = jax.tree_util.tree_flatten(
                    template_state)
                opt_state = checkpoint['opt_state']
                count = int(opt_state.attrs['leaf_count'])
                if count != len(template_leaves):
                    raise ValueError(
                        'Resume optimizer state is incompatible with the '
                        'current optimizer')
                leaves = []
                for index, template in enumerate(template_leaves):
                    value = jnp.asarray(opt_state[str(index)][...])
                    if value.shape != np.shape(template):
                        raise ValueError(
                            'Resume optimizer-state shape is incompatible')
                    leaves.append(value)
                state = jax.tree_util.tree_unflatten(tree, leaves)
                next_iteration = int(
                    checkpoint.attrs['next_iteration'])
                previous = float(checkpoint.attrs['efree_previous'])
                efree_previous = None if np.isnan(previous) else previous

                history = f.get('history')
                if history is not None and 'iteration' in history:
                    iterations = np.asarray(history['iteration'])
                    keep = int(np.count_nonzero(
                        iterations < next_iteration))
                    for dataset in history.values():
                        if dataset.shape and dataset.shape[0] > keep:
                            dataset.resize(keep, axis=0)
                    f.attrs['completed_iterations'] = keep
                f.flush()
                return (param, state, next_iteration,
                        efree_previous, 'exact optimizer checkpoint')

            history = f.get('history')
            if (history is None or 'iteration' not in history or
                    len(history['iteration']) == 0):
                raise ValueError(
                    'Resume file has neither an optimizer checkpoint nor a '
                    'completed iteration')
            theta = jnp.asarray(history['theta'][-1])
            if self.fixed_occupation:
                eta = jnp.asarray([])
            elif 'eta' in history:
                eta = jnp.asarray(history['eta'][-1])
            else:
                raise ValueError('Legacy resume file does not contain eta')
            param = jnp.concatenate((theta, eta))
            if param.shape != param_template.shape:
                raise ValueError(
                    f'Resume parameter shape {param.shape} does not match '
                    f'current shape {param_template.shape}')
            state = solver.init(param)
            if (self.fixed_occupation and
                    'grad_theta_vector' in history):
                grad = jnp.asarray(history['grad_theta_vector'][-1])
                update, state = solver.update(grad, state, param)
                param = optax.apply_updates(param, update)
            next_iteration = int(history['iteration'][-1]) + 1
            efree_previous = float(history['efree'][-1])
            return (param, state, next_iteration, efree_previous,
                    'legacy warm start; optimizer momentum reset')

    def init_iteration_history(self, t0, resume=False):
        """Create a new history or mark an existing checkpoint as resumed."""
        if resume:
            with h5py.File(self.fn_out, 'a') as f:
                if 'history' not in f:
                    raise ValueError('Resume file does not contain history')
                f.attrs['status'] = 'running'
                f.attrs['resume_count'] = int(
                    f.attrs.get('resume_count', 0)) + 1
                f.attrs['last_resume_t0'] = float(t0)
                f.flush()
            return
        with h5py.File(self.fn_out, 'w') as f:
            f.attrs['status'] = 'running'
            f.attrs['t0'] = float(t0)
            f.attrs['theta_seed'] = int(getattr(self, 'theta_seed', 42))
            f.create_group('history')

    def append_iteration_history(self, **values):
        """Append one completed iteration and flush it immediately to disk."""
        with h5py.File(self.fn_out, 'a') as f:
            history = f.require_group('history')
            lengths = []
            for key, value in values.items():
                array = np.asarray(value)
                if key not in history:
                    shape = (0,) + array.shape
                    maxshape = (None,) + array.shape
                    history.create_dataset(
                        key, shape=shape, maxshape=maxshape,
                        dtype=array.dtype, chunks=True)
                dataset = history[key]
                if dataset.shape[1:] != array.shape:
                    raise ValueError(
                        f'History field {key} changed shape from '
                        f'{dataset.shape[1:]} to {array.shape}')
                index = dataset.shape[0]
                dataset.resize(index + 1, axis=0)
                dataset[index] = array
                lengths.append(index + 1)
            if lengths and len(set(lengths)) != 1:
                raise ValueError('Iteration history fields have unequal lengths')
            if lengths:
                f.attrs['completed_iterations'] = lengths[0]
            f.flush()

    def save_result(self, method, **kwargs):
        tm, ts = divmod(int(kwargs['t1'] - kwargs['t0']), 60)
        print(f'\n{method} time elapsed: {tm}m {ts}s\n')

        with h5py.File(self.fn_out, 'a') as f:
            if 'result' in f:
                del f['result']
            result = f.create_group('result')
            for key, val in kwargs.items():
                result.create_dataset(key, data=np.asarray(val))
            f.attrs['status'] = 'complete'
            f.attrs['completed_iterations'] = int(kwargs.get('niter', 0))
            f.flush()

        print(f'Saved at {self.fn_out}\n')

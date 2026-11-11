"""General-commuting Pauli grouping support for VQE."""

import hashlib
import os

import jax.numpy as jnp
import numpy as np
import pennylane as qp


class GroupedVQEMixin:
    """VQE variant that measures the Pauli terms of rho in Pauli groups."""

    def __init__(self, *args, **kwargs):
        self.grouping_type = kwargs.pop('grouping_type', 'gc')
        if self.grouping_type not in ('gc', 'qwc'):
            raise ValueError(f'Unknown grouping type: {self.grouping_type}')
        super().__init__(*args, **kwargs)
        if not self.is_qpu:
            raise ValueError('Pauli grouping requires the --qpu option')

        self._setup_rho_pauli_groups()
        self.get_rho_raw = self.get_rho_raw_qpu_grouped

    def _setup_rho_pauli_groups(self):
        """Deduplicate gamma terms and make RLF general-commuting groups."""
        # gamma depends on the AO-to-computational-basis assignment as well as
        # M and N.  Keep structured encodings from reusing an incompatible
        # cache produced by the former contiguous AO indexing.
        basis_bytes = np.asarray(self.ao_to_basis, dtype=np.int64).tobytes()
        basis_tag = hashlib.sha1(basis_bytes).hexdigest()[:10]
        self.fn_gc = (f'data/{self.grouping_type}_M{self.M}_N{self.N}'
                      f'_B{basis_tag}.npz')
        if os.path.exists(self.fn_gc):
            self._load_rho_pauli_groups()
            return

        obs_by_key = {}
        gamma_terms = []

        for gamma_ij in self.gamma:
            coeffs, observables = gamma_ij.terms()
            terms_ij = []
            constant = 0.0
            for coeff, obs in zip(coeffs, observables):
                key = qp.pauli.pauli_word_to_string(
                        obs, wire_map={wire: wire for wire in self.wires})
                if set(key) <= {'I'}:
                    constant += float(np.real(coeff))
                else:
                    obs_by_key.setdefault(key, obs)
                    terms_ij.append((key, float(np.real(coeff))))
            gamma_terms.append((constant, terms_ij))

        obs_total = list(obs_by_key.values())
        self.rho_obs_keys = list(obs_by_key)
        self.rho_obs_index = {key: i for i, key in enumerate(self.rho_obs_keys)}
        grouping = ('commuting' if self.grouping_type == 'gc' else 'qwc')
        self.rho_obs_groups = qp.pauli.group_observables(
                obs_total, grouping_type=grouping, method='rlf')

        ngamma = self.N * self.N
        self.rho_gamma_constant = jnp.asarray(
                [constant for constant, _ in gamma_terms])
        coeff = np.zeros((ngamma, len(obs_total)))
        for ij, (_, terms_ij) in enumerate(gamma_terms):
            for key, value in terms_ij:
                coeff[ij, self.rho_obs_index[key]] += value
        self.rho_gamma_coeff = jnp.asarray(coeff)

        self.rho_group_indices = []
        self.rho_group_eigenvalues = []
        self.rho_group_unitaries = []
        self.rho_group_qnodes = []
        for group in self.rho_obs_groups:
            indices = [self.rho_obs_index[
                qp.pauli.pauli_word_to_string(
                    obs, wire_map={wire: wire for wire in self.wires})]
                for obs in group]
            if self.grouping_type == 'gc':
                rotations, eigenvalues = self._gc_diagonalization(group)
            else:
                rotations, eigenvalues = self._qwc_diagonalization(group)
            if len(rotations) != 1 or rotations[0][0] != 'U':
                raise ValueError('GC cache requires one unitary per group')
            self.rho_group_indices.append(indices)
            self.rho_group_eigenvalues.append(jnp.asarray(eigenvalues))
            self.rho_group_unitaries.append(np.asarray(rotations[0][1]))
            self.rho_group_qnodes.append(
                    self._make_rho_group_qnode(rotations))

        print(f'rho Pauli grouping: {len(obs_total)} terms -> '
              f'{len(self.rho_obs_groups)} {self.grouping_type.upper()} groups')
        self._save_rho_pauli_groups()

    def _save_rho_pauli_groups(self):
        """Cache all M/N-dependent GC measurement data in one NPZ file."""
        group_sizes = np.asarray([len(indices)
                                  for indices in self.rho_group_indices])
        group_offsets = np.concatenate(([0], np.cumsum(group_sizes)))
        group_indices = np.concatenate(self.rho_group_indices)
        group_eigenvalues = np.concatenate(
                [np.asarray(values) for values in self.rho_group_eigenvalues],
                axis=0)

        os.makedirs(os.path.dirname(self.fn_gc), exist_ok=True)
        np.savez(
                self.fn_gc,
                version=np.asarray(1),
                M=np.asarray(self.M),
                N=np.asarray(self.N),
                gamma_constant=np.asarray(self.rho_gamma_constant),
                gamma_coeff=np.asarray(self.rho_gamma_coeff),
                group_offsets=group_offsets,
                group_indices=group_indices,
                group_eigenvalues=group_eigenvalues,
                group_unitaries=np.stack(self.rho_group_unitaries),
                )
        print(f'{self.grouping_type.upper()} measurement cache saved at '
              f'{self.fn_gc}')

    def _load_rho_pauli_groups(self):
        """Load cached GC data and recreate only the measurement QNodes."""
        with np.load(self.fn_gc) as cache:
            if (int(cache['version']) != 1 or int(cache['M']) != self.M or
                    int(cache['N']) != self.N):
                raise ValueError(f'Incompatible GC cache: {self.fn_gc}')

            self.rho_gamma_constant = jnp.asarray(cache['gamma_constant'])
            self.rho_gamma_coeff = jnp.asarray(cache['gamma_coeff'])
            offsets = np.asarray(cache['group_offsets'])
            indices = np.asarray(cache['group_indices'])
            eigenvalues = np.asarray(cache['group_eigenvalues'])
            unitaries = np.asarray(cache['group_unitaries'])

        self.rho_obs_keys = list(range(self.rho_gamma_coeff.shape[1]))
        self.rho_obs_index = {}
        self.rho_obs_groups = [None] * (len(offsets) - 1)
        self.rho_group_indices = []
        self.rho_group_eigenvalues = []
        self.rho_group_unitaries = list(unitaries)
        self.rho_group_qnodes = []
        for group, unitary in enumerate(unitaries):
            start, stop = offsets[group:group + 2]
            self.rho_group_indices.append(indices[start:stop].tolist())
            self.rho_group_eigenvalues.append(
                    jnp.asarray(eigenvalues[start:stop]))
            self.rho_group_qnodes.append(
                    self._make_rho_group_qnode([('U', unitary)]))

        print(f'{self.grouping_type.upper()} measurement cache loaded from '
              f'{self.fn_gc}: '
              f'{self.rho_gamma_coeff.shape[1]} terms -> '
              f'{len(self.rho_group_qnodes)} '
              f'{self.grouping_type.upper()} groups')

    def _qwc_diagonalization(self, group):
        """Build a tensor-product basis rotation for one QWC group."""
        axes = ['I'] * self.M
        for obs in group:
            word = qp.pauli.pauli_word_to_string(
                obs, wire_map={wire: wire for wire in self.wires})
            for wire, axis in enumerate(word):
                if axis == 'I':
                    continue
                if axes[wire] not in ('I', axis):
                    raise ValueError('Non-QWC observables found in QWC group')
                axes[wire] = axis

        identity = np.eye(2, dtype=complex)
        hadamard = qp.matrix(qp.Hadamard(0))
        s_dagger = qp.matrix(qp.adjoint(qp.S)(0))
        local = {
            'I': identity,
            'Z': identity,
            'X': hadamard,
            'Y': hadamard @ s_dagger,
        }
        rotation = np.eye(1, dtype=complex)
        for axis in axes:
            rotation = np.kron(rotation, local[axis])

        eigenvalues = []
        for obs in group:
            matrix = qp.matrix(obs, wire_order=self.wires)
            diagonal = rotation @ matrix @ rotation.conj().T
            off_diagonal = diagonal - np.diag(np.diag(diagonal))
            if not np.allclose(off_diagonal, 0.0, atol=1e-8):
                raise ValueError('QWC diagonalization failed')
            eigenvalues.append(np.real(np.diag(diagonal)))
        return [('U', rotation)], np.asarray(eigenvalues)

    def _gc_diagonalization(self, group):
        """Find the gcg.py Clifford rotation and Z-basis eigenvalue table."""
        G = qp.pauli.observables_to_binary_matrix(group, n_qubits=self.M)

        # Binary row reduction; only independent generators are needed to
        # construct the simultaneous diagonalizing Clifford circuit.
        pivot = 0
        for column in range(G.shape[1]):
            pivot_cur = next((row for row in range(pivot, G.shape[0])
                              if G[row, column]), -1)
            if pivot_cur < 0:
                continue
            G[[pivot, pivot_cur]] = G[[pivot_cur, pivot]]
            for row in range(pivot + 1, G.shape[0]):
                if G[row, column]:
                    G[row] = (G[row] + G[pivot]) % 2
            pivot += 1
        G = G[:pivot]

        rotations = []
        used = []
        for row in range(pivot):
            for wire in self.wires:
                if G[row, wire] and G[row, self.M + wire]:
                    G[:, self.M + wire] = (
                            G[:, self.M + wire] + G[:, wire]) % 2
                    rotations.append(('S', wire))
                if G[row, wire] and not G[row, self.M + wire]:
                    G[:, [wire, self.M + wire]] = \
                            G[:, [self.M + wire, wire]]
                    rotations.append(('H', wire))

            support = [wire for wire in self.wires
                       if G[row, self.M + wire]]
            control = next(wire for wire in support if wire not in used)
            used.append(control)
            for target in support:
                if target != control:
                    G[:, control] = (G[:, control] + G[:, target]) % 2
                    G[:, self.M + target] = (
                            G[:, self.M + target] +
                            G[:, self.M + control]) % 2
                    rotations.append(('C', target, control))

        # Build the Clifford matrix once to retain the signs discarded by the
        # binary symplectic representation.  Each row contains the eigenvalues
        # of one original observable for every computational-basis outcome.
        unitary = np.eye(2**self.M, dtype=complex)
        for rotation in rotations:
            if rotation[0] == 'H':
                gate = qp.Hadamard(wires=rotation[1])
            elif rotation[0] == 'S':
                gate = qp.S(wires=rotation[1])
            else:
                gate = qp.CNOT(wires=[rotation[1], rotation[2]])
            unitary = unitary @ qp.matrix(gate, wire_order=self.wires)

        eigenvalues = []
        diagonalization_failed = False
        for obs in group:
            matrix = qp.matrix(obs, wire_order=self.wires)
            diagonal = unitary.conj().T @ matrix @ unitary
            off_diagonal = diagonal - np.diag(np.diag(diagonal))
            if not np.allclose(off_diagonal, 0.0, atol=1e-8):
                diagonalization_failed = True
                break
            eigenvalues.append(np.real(np.diag(diagonal)))

        if diagonalization_failed:
            # Resolve any convention mismatch between gcg.py's symplectic
            # CNOT labels and the backend by constructing the same common
            # eigenbasis directly.  Binary-weighted coefficients distinguish
            # every joint +/-1 eigenvalue signature.
            matrices = [qp.matrix(obs, wire_order=self.wires) for obs in group]
            combined = sum((2.0**i) * matrix
                           for i, matrix in enumerate(matrices))
            _, eigenvectors = np.linalg.eigh(combined)
            rotation = eigenvectors.conj().T
            rotations = [('U', rotation)]
            eigenvalues = []
            for matrix in matrices:
                diagonal = rotation @ matrix @ rotation.conj().T
                off_diagonal = diagonal - np.diag(np.diag(diagonal))
                if not np.allclose(off_diagonal, 0.0, atol=1e-8):
                    raise ValueError('GC simultaneous diagonalization failed')
                eigenvalues.append(np.real(np.diag(diagonal)))
        else:
            # Apply the verified gcg.py Clifford as one fixed unitary.  This
            # avoids backend-dependent H/S/CNOT ordering conventions.
            rotations = [('U', unitary.conj().T)]
        return rotations, np.asarray(eigenvalues)

    def _make_rho_group_qnode(self, rotations):
        """Create one QNode that measures one diagonalized GC group."""

        def circuit(theta, basis):
            self.prep_circuit(theta, basis)
            for rotation in reversed(rotations):
                if rotation[0] == 'H':
                    qp.Hadamard(wires=rotation[1])
                elif rotation[0] == 'S':
                    qp.adjoint(qp.S)(wires=rotation[1])
                elif rotation[0] == 'C':
                    qp.CNOT(wires=[rotation[1], rotation[2]])
                else:
                    qp.QubitUnitary(rotation[1], wires=self.wires)
            return qp.probs(wires=self.wires)

        qnode = qp.QNode(circuit, self.dev, interface='jax',
                         diff_method='parameter-shift')
        return qp.set_shots(qnode, shots=self.shots)

    def measure_rho_pauli_groups(self, theta, basis):
        """Measure grouped Pauli words and reconstruct one raw density matrix."""
        values = [None] * len(self.rho_obs_keys)
        for indices, eigenvalues, qnode in zip(
                self.rho_group_indices, self.rho_group_eigenvalues,
                self.rho_group_qnodes):
            probabilities = jnp.asarray(qnode(theta, basis))
            measured = eigenvalues @ probabilities
            for index, value in zip(indices, measured):
                values[index] = value

        obs_values = jnp.stack(values)
        rho_flat = self.rho_gamma_constant + self.rho_gamma_coeff @ obs_values
        return rho_flat.reshape(self.N, self.N)

    def get_rho_raw_qpu_grouped(self, theta):
        """Grouped counterpart of get_rho_raw_qpu; existing method is unchanged."""
        rho_raw = []
        for theta_k in theta:
            rho_raw_k = [self.measure_rho_pauli_groups(theta_k, basis)
                         for basis in self.bases]
            rho_raw.append(jnp.stack(rho_raw_k))
        return jnp.stack(rho_raw)


def grouped_vqe_class(base_class):
    """Create the grouped variant without importing qudft.py as a module."""
    return type('GroupedVQE', (GroupedVQEMixin, base_class), {})

"""Orbital-rotation ansatz definitions and circuit helpers."""

import json
from itertools import product

import numpy as np
import pennylane as qml

from lib.ao_encoding import encoding_index


def _pauli_coverage(word, basis_indices):
    """Return valid unordered basis-state pairs coupled by a Pauli word."""
    nqubits = len(word)
    flip_mask = sum(
            1 << (nqubits - 1 - wire)
            for wire, operator in enumerate(word)
            if operator in 'XY')
    valid = set(map(int, basis_indices))
    return {
        (state, state ^ flip_mask)
        for state in valid
        if state < (state ^ flip_mask) and (state ^ flip_mask) in valid
    }


def _pauli_direction_coverage(word, basis_indices):
    """Return coupled orbital pairs labelled by real/imaginary direction.

    An odd number of Y factors makes ``-1j * P`` real and therefore gives
    the real orbital-rotation direction.  An even number gives the
    complementary imaginary direction.
    """
    direction = 'R' if word.count('Y') % 2 == 1 else 'I'
    return {
        (left, right, direction)
        for left, right in _pauli_coverage(word, basis_indices)
    }


def _pauli_noncommutes(left, right):
    """Return whether two equal-length Pauli words anticommute."""
    if len(left) != len(right):
        raise ValueError('Pauli words must have equal length')
    mismatches = sum(
        left_op != 'I' and right_op != 'I' and left_op != right_op
        for left_op, right_op in zip(left, right)
    )
    return mismatches % 2 == 1


def pauli_priority(nqubits, atom_wires=(0,), basis_indices=None):
    """Rank odd-Y Pauli words using structured-AO coverage.

    The greedy ordering maximizes newly covered real/imaginary pair
    directions, then the number of anticommuting previously selected words.
    It next minimizes Pauli weight, prefers intra-atom mixing, and finally
    uses lexical order as a deterministic tie-break.
    """
    if nqubits < 1:
        raise ValueError('nqubits must be positive')
    atom_wires = tuple(atom_wires)
    if any(wire < 0 or wire >= nqubits for wire in atom_wires):
        raise ValueError('Invalid atom wire')
    if len(set(atom_wires)) != len(atom_wires):
        raise ValueError('Duplicate atom wires')

    if basis_indices is None:
        basis_indices = range(2**nqubits)
    basis_indices = tuple(map(int, basis_indices))
    if len(set(basis_indices)) != len(basis_indices):
        raise ValueError('Duplicate basis index')
    if any(index < 0 or index >= 2**nqubits
           for index in basis_indices):
        raise ValueError('Basis index lies outside the Hilbert space')

    remaining = [
        ''.join(operators)
        for operators in product('IXYZ', repeat=nqubits)
        if operators.count('Y') % 2 == 1
    ]
    coverage = {
        word: _pauli_direction_coverage(word, basis_indices)
        for word in remaining
    }
    covered = set()
    priority = []

    while remaining:
        def key(word):
            weight = sum(operator != 'I' for operator in word)
            inter = any(word[wire] in 'XY' for wire in atom_wires)
            noncommuting = sum(
                _pauli_noncommutes(word, selected) for selected in priority)
            return (
                -len(coverage[word] - covered),
                -noncommuting,
                weight,
                inter,
                word,
            )

        word = min(remaining, key=key)
        remaining.remove(word)
        priority.append(word)
        covered.update(coverage[word])

    return priority


def select_paulis(priority, nparam, min_coverage=False,
                  basis_indices=None):
    """Select a prefix and optionally require complete pair-direction coverage."""
    priority = list(priority)
    if not priority:
        raise ValueError('priority must not be empty')
    if not 1 <= nparam <= len(priority):
        raise ValueError(
                f'nparam must be between 1 and {len(priority)}, got {nparam}')

    nqubits = len(priority[0])
    if any(len(word) != nqubits for word in priority):
        raise ValueError('All Pauli words must have equal length')
    if basis_indices is None:
        basis_indices = range(2**nqubits)
    basis_indices = tuple(map(int, basis_indices))

    selected = priority[:nparam]
    if min_coverage:
        directions = {
            'R' if word.count('Y') % 2 == 1 else 'I'
            for word in priority
        }
        target = {
            (left, right, direction)
            for offset, left in enumerate(basis_indices)
            for right in basis_indices[offset + 1:]
            for direction in directions
        }
        covered = set().union(*(
            _pauli_direction_coverage(word, basis_indices)
            for word in selected))
        if covered != target:
            cumulative = set()
            minimum = None
            for count, word in enumerate(priority, start=1):
                cumulative.update(
                    _pauli_direction_coverage(word, basis_indices))
                if cumulative == target:
                    minimum = count
                    break
            raise ValueError(
                    f'nparam={nparam} does not cover every valid AO '
                    f'pair direction; '
                    f'at least {minimum} prioritized strings are required')
    return selected


def select_adaptive_paulis(paulis, gradients, basis_indices):
    """Select high-gradient words while covering every valid AO pair."""
    paulis = list(paulis)
    gradients = np.asarray(gradients)
    if gradients.shape != (len(paulis),):
        raise ValueError('gradients must contain one value per Pauli word')
    basis_indices = tuple(map(int, basis_indices))
    target = {
        (left, right)
        for offset, left in enumerate(basis_indices)
        for right in basis_indices[offset + 1:]
    }
    order = sorted(
        range(len(paulis)), key=lambda index: (-abs(gradients[index]), index))
    selected = []
    covered = set()
    for index in order:
        new_pairs = _pauli_coverage(paulis[index], basis_indices) - covered
        if not new_pairs:
            continue
        selected.append(index)
        covered.update(new_pairs)
        if covered == target:
            return selected
    raise ValueError('Pauli pool cannot cover every valid AO pair')


def _mixing_wires(atom_wires, local_wires):
    """Validate and return wires in PennyLane/MSB encoding order."""
    atom_wires = list(atom_wires)
    local_wires = list(local_wires)
    wires = atom_wires + local_wires
    if not wires:
        raise ValueError('At least one encoding wire is required')
    if len(set(wires)) != len(wires):
        raise ValueError('Atom and local wire lists must be disjoint')
    return atom_wires, local_wires, wires


def _two_level_ry(theta, state_a, state_b, wires):
    """Mix exactly two computational basis states with an RY block."""
    if state_a == state_b:
        raise ValueError('The two states to mix must be different')

    dimension = 2**len(wires)
    if not (0 <= state_a < dimension and 0 <= state_b < dimension):
        raise ValueError('Mixing state lies outside the encoded Hilbert space')

    identity = qml.math.eye(dimension, like=theta)
    ket_a = identity[:, state_a]
    ket_b = identity[:, state_b]
    cosine = qml.math.cos(theta / 2)
    sine = qml.math.sin(theta / 2)
    unitary = (
        identity
        + (cosine - 1) * (
            qml.math.outer(ket_a, ket_a)
            + qml.math.outer(ket_b, ket_b))
        + sine * (
            qml.math.outer(ket_b, ket_a)
            - qml.math.outer(ket_a, ket_b))
    )
    qml.QubitUnitary(unitary, wires=wires)


def intra_mixing(theta, atom, local_a, local_b,
                  atom_wires, local_wires):
    """Mix two local orbitals within one atom."""
    atom_wires, local_wires, wires = _mixing_wires(
            atom_wires, local_wires)
    state_a = encoding_index(
            atom, local_a, len(atom_wires), len(local_wires))
    state_b = encoding_index(
            atom, local_b, len(atom_wires), len(local_wires))
    _two_level_ry(theta, state_a, state_b, wires)


def inter_mixing(theta, atom_a, local_a, atom_b, local_b,
                  atom_wires, local_wires):
    """Mix arbitrary orbitals belonging to two different atoms."""
    atom_wires, local_wires, wires = _mixing_wires(
            atom_wires, local_wires)
    if atom_a == atom_b:
        raise ValueError('inter_mixing requires two different atoms')
    state_a = encoding_index(
            atom_a, local_a, len(atom_wires), len(local_wires))
    state_b = encoding_index(
            atom_b, local_b, len(atom_wires), len(local_wires))
    _two_level_ry(theta, state_a, state_b, wires)


def load_circuit(path):
    """Load and validate a JSON RY-CNOT circuit description."""
    with open(path) as handle:
        data = json.load(handle)

    ry_wires = [int(wire) for wire in data['ry_wires']]
    cnot_after = [[tuple(map(int, pair)) for pair in pairs]
                  for pairs in data['cnot_after']]
    if len(ry_wires) != len(cnot_after):
        raise ValueError('ry_wires and cnot_after must have equal length')

    return {
        'name': data.get('name', 'custom'),
        'ry_wires': ry_wires,
        'cnot_after': cnot_after,
        'nparam': len(ry_wires),
        'ncnot': sum(map(len, cnot_after)),
    }


def topology_schedule(nqubits, nparam, ncnot, topology):
    """Construct a deterministic RY wire sequence and CNOT schedule."""
    wires = list(range(nqubits))
    ry_wires = [index % nqubits for index in range(nparam)]
    linear_pairs = [(wire, wire + 1) for wire in range(nqubits - 1)]
    pairs = {
        'linear': linear_pairs,
        'reverse': [(target, control)
                    for control, target in reversed(linear_pairs)],
        'ring': linear_pairs + [(nqubits - 1, 0)],
        'all': [(control, target)
                for first in wires for second in wires[first + 1:]
                for control, target in ((first, second), (second, first))],
    }[topology]

    cnot_after = [[] for _ in range(nparam)]
    if ncnot and not pairs:
        raise ValueError('CNOT gates require at least two qubits')
    for gate_index in range(ncnot):
        slot = ((gate_index + 1) * nparam - 1) // ncnot
        cnot_after[slot].append(pairs[gate_index % len(pairs)])
    return ry_wires, cnot_after


def apply_circuit(theta, ry_wires, cnot_after):
    """Apply an RY-CNOT schedule to the active PennyLane circuit."""
    for angle, wire, pairs_after in zip(theta, ry_wires, cnot_after):
        qml.RY(angle, wires=wire)
        for control, target in pairs_after:
            qml.CNOT(wires=[control, target])


class Ansatz:
    """Configure and apply one of the supported orbital ansatzes."""

    def __init__(self, name, wires, atom_wires, local_wires, ao_counts,
                 kpts, topology='linear', nparam=28, ncnot=12,
                 circuit=None, basis_indices=None, min_coverage=False,
                 fixed_paulis=None):
        self.name = name
        self.requested_nparam = nparam
        if nparam < 1:
            raise ValueError('nparam must be positive')
        self.wires = list(wires)
        self.atom_wires = list(atom_wires)
        self.local_wires = list(local_wires)
        self.ao_counts = np.asarray(ao_counts, dtype=int)
        self.topology = topology
        self.circuit = circuit
        self.ry_wires = None
        self.cnot_after = None
        self.pauli = []
        self.pauli_pool = []
        self.priority = []
        self.intra_pairs = []
        self.inter_pairs = []

        if circuit is not None:
            circuit_data = load_circuit(circuit)
            self.name = 'rycn'
            self.topology = circuit_data['name']
            self.ry_wires = circuit_data['ry_wires']
            self.cnot_after = circuit_data['cnot_after']
            nparam = circuit_data['nparam']
            ncnot = circuit_data['ncnot']

        if self.name == 'au-oddy':
            self.pauli = [
                qml.pauli.pauli_word_to_string(
                    p, wire_map={wire: wire for wire in self.wires})
                for p in qml.pauli.pauli_group(len(self.wires))
                if not isinstance(p, qml.Identity)
            ]
            self.pauli = [p for p in self.pauli
                          if p.count('Y') % 2 == 1]
            self.nparam = len(self.pauli)
            self.ncnot = 0
        elif self.name in ('au-all', 'au-swap', 'au-grow'):
            self.pauli_pool = [
                ''.join(operators)
                for operators in product('IXYZ', repeat=len(self.wires))
                if any(operator != 'I' for operator in operators)
            ]
            self.pauli = list(self.pauli_pool)
            self.nparam = len(self.pauli)
            self.ncnot = 0
        elif self.name == 'au-fixed':
            if fixed_paulis is None:
                raise ValueError('au-fixed requires fixed_paulis')
            self.pauli = list(fixed_paulis)
            if len(self.pauli) != nparam:
                raise ValueError('fixed_paulis count must equal nparam')
            expected_width = len(self.wires)
            if any(len(word) != expected_width for word in self.pauli):
                raise ValueError(
                        f'every Pauli string must have width {expected_width}')
            if any(set(word) - set('IXYZ') for word in self.pauli):
                raise ValueError('Pauli strings may contain only I, X, Y, Z')
            if any(set(word) <= {'I'} for word in self.pauli):
                raise ValueError('Identity is not a valid au-fixed generator')
            if len(set(self.pauli)) != len(self.pauli):
                raise ValueError('au-fixed Pauli strings must be unique')
            self.nparam = len(self.pauli)
            self.ncnot = 0
        elif self.name == 'au-prio':
            self.priority = pauli_priority(
                    len(self.wires),
                    atom_wires=self.atom_wires,
                    basis_indices=basis_indices)
            self.pauli = select_paulis(
                    self.priority,
                    nparam=nparam,
                    min_coverage=min_coverage,
                    basis_indices=basis_indices)
            self.nparam = len(self.pauli)
            self.ncnot = 0
        elif self.name == 'ao-full':
            if len(kpts) != 1 or not np.allclose(kpts, 0.0):
                raise ValueError(
                        'ao-full ansatz currently requires Gamma point')
            self._setup_atom_orbital_pairs()
            self.nparam = len(self.intra_pairs) + len(self.inter_pairs)
            self.ncnot = 0
        elif self.name == 'rycn':
            self.nparam = nparam
            self.ncnot = ncnot
            if self.ry_wires is None:
                self.ry_wires, self.cnot_after = topology_schedule(
                        len(self.wires), self.nparam, self.ncnot,
                        self.topology)
        else:
            raise ValueError(f'Unknown ansatz: {self.name}')

        if self.nparam < 1:
            raise ValueError('nparam must be positive')
        if self.ncnot < 0:
            raise ValueError('ncnot must be non-negative')

    @property
    def npauli(self):
        return len(self.pauli)

    @property
    def nintra(self):
        return len(self.intra_pairs)

    @property
    def ninter(self):
        return len(self.inter_pairs)

    def _setup_atom_orbital_pairs(self):
        """List all intra pairs first, followed by all inter pairs."""
        for atom, count in enumerate(self.ao_counts):
            self.intra_pairs.extend(
                    (atom, local_a, local_b)
                    for local_a in range(count)
                    for local_b in range(local_a + 1, count))
        self.inter_pairs.extend(
                (atom_a, local_a, atom_b, local_b)
                for atom_a, count_a in enumerate(self.ao_counts)
                for atom_b in range(atom_a + 1, len(self.ao_counts))
                for local_a in range(count_a)
                for local_b in range(self.ao_counts[atom_b]))

    def apply(self, theta):
        """Apply the configured parameterized circuit."""
        if self.name in ('au-oddy', 'au-all', 'au-swap', 'au-grow', 'au-fixed', 'au-prio'):
            for angle, pauli in zip(theta, self.pauli):
                qml.PauliRot(angle, pauli, wires=self.wires)
            return

        if self.name == 'ao-full':
            for angle, (atom, local_a, local_b) in zip(
                    theta, self.intra_pairs):
                intra_mixing(
                        angle, atom, local_a, local_b,
                        self.atom_wires, self.local_wires)
            for angle, (atom_a, local_a, atom_b, local_b) in zip(
                    theta[self.nintra:], self.inter_pairs):
                inter_mixing(
                        angle, atom_a, local_a, atom_b, local_b,
                        self.atom_wires, self.local_wires)
            return

        apply_circuit(theta, self.ry_wires, self.cnot_after)

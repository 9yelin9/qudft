"""Structured atom/local-orbital binary encoding helpers."""

import numpy as np


def encoding_index(atom, local, natom_bits, nlocal_bits):
    """Return the ``atom bits | local-AO bits`` basis-state index."""
    if not 0 <= atom < 2**natom_bits:
        raise ValueError(
                f'atom index {atom} does not fit in {natom_bits} bits')
    if not 0 <= local < 2**nlocal_bits:
        raise ValueError(
                f'local orbital index {local} does not fit in '
                f'{nlocal_bits} bits')
    return (atom << nlocal_bits) | local


def structured_ao_encoding(mat):
    """Map PySCF AOs to ``atom bits | local-AO bits`` basis states.

    PennyLane interprets the first entry passed to ``BasisState`` as wire 0
    and as the most-significant bit of the computational-basis index. Atom
    bits are therefore placed on the lowest-numbered wires, followed by the
    local-orbital bits.
    """
    ao_slices = np.asarray(mat.aoslice_by_atom(), dtype=int)
    natom = len(ao_slices)
    if natom < 1:
        raise ValueError('The cell must contain at least one atom')

    ao_counts = ao_slices[:, 3] - ao_slices[:, 2]
    if np.any(ao_counts < 1):
        raise ValueError('Every atom must have at least one AO')

    natom_bits = (natom - 1).bit_length()
    nlocal_bits = (int(ao_counts.max()) - 1).bit_length()
    nqubits = natom_bits + nlocal_bits
    if nqubits < 1:
        # PennyLane devices require a wire even for a one-dimensional AO
        # space. Its only physical basis state remains |0>.
        nqubits = 1
        nlocal_bits = 1

    ao_to_basis = np.empty(mat.nao, dtype=int)
    assigned = np.zeros(mat.nao, dtype=bool)
    for atom, (_, _, ao_start, ao_stop) in enumerate(ao_slices):
        for local, ao in enumerate(range(ao_start, ao_stop)):
            ao_to_basis[ao] = encoding_index(
                    atom, local, natom_bits, nlocal_bits)
            assigned[ao] = True

    if not np.all(assigned):
        raise ValueError('aoslice_by_atom() does not cover every AO')
    if len(np.unique(ao_to_basis)) != mat.nao:
        raise ValueError('Structured AO encoding produced duplicate states')

    bases = np.asarray([
        list(np.binary_repr(index, width=nqubits))
        for index in ao_to_basis
    ], dtype=int)
    return bases, ao_to_basis, natom_bits, nlocal_bits

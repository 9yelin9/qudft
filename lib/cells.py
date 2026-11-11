"""Periodic-cell construction helpers."""

import numpy as np


def supercell(mat, sc):
    """Build a periodic supercell by replicating a primitive PySCF cell."""
    from pyscfad.pbc import gto

    avec = np.asarray(mat.lattice_vectors())  # Bohr
    coords = np.asarray(mat.atom_coords())  # Bohr
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
    mat_sc.spin = int(mat.spin * np.prod(sc))
    mat_sc.build()
    return mat_sc

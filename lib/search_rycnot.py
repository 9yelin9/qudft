#!/usr/bin/env python3
"""Search N-parameter RY-CNOT circuits against a reference density matrix."""

import argparse
import csv
import json
import os

import h5py
import numpy as np
import scipy.linalg
import scipy.optimize


def ry_matrix(angle):
    c, s = np.cos(angle / 2.0), np.sin(angle / 2.0)
    return np.array([[c, -s], [s, c]])


def one_qubit_gate(gate, wire, nqubits):
    result = np.ones((1, 1))
    for current in range(nqubits):
        result = np.kron(result, gate if current == wire else np.eye(2))
    return result


def cnot_gate(control, target, nqubits):
    size = 2**nqubits
    gate = np.zeros((size, size))
    for column in range(size):
        bits = list(map(int, np.binary_repr(column, width=nqubits)))
        if bits[control]:
            bits[target] ^= 1
        row = int(''.join(map(str, bits)), 2)
        gate[row, column] = 1.0
    return gate


def project_occupations(values, nelec):
    """Euclidean projection onto 0 <= occ <= 2 and sum(occ) = nelec."""
    lo, hi = -2.0 - values.max(), 2.0 - values.min()
    for _ in range(60):
        shift = (lo + hi) / 2.0
        occupations = np.clip(values + shift, 0.0, 2.0)
        if occupations.sum() < nelec:
            lo = shift
        else:
            hi = shift
    return np.clip(values + (lo + hi) / 2.0, 0.0, 2.0)


def circuit_unitary(theta, ry_wires, cnot_after, cnot_matrices, nqubits):
    unitary = np.eye(2**nqubits)
    for angle, wire, pairs in zip(theta, ry_wires, cnot_after):
        unitary = one_qubit_gate(ry_matrix(angle), wire, nqubits) @ unitary
        for pair in pairs:
            unitary = cnot_matrices[tuple(pair)] @ unitary
    return unitary


def fit_circuit(candidate, targets, nelec, nqubits, restarts, rng):
    pairs = [(control, target_wire) for control in range(nqubits)
             for target_wire in range(nqubits) if control != target_wire]
    cnot_matrices = {pair: cnot_gate(*pair, nqubits) for pair in pairs}
    errors, theta_kpts, occupations_kpts = [], [], []
    for target in targets:
        target_norm = np.linalg.norm(target)

        def objective(theta):
            unitary = circuit_unitary(
                    theta, candidate['ry_wires'], candidate['cnot_after'],
                    cnot_matrices, nqubits)
            diagonal = np.diag(unitary.T @ target @ unitary)
            occupations = project_occupations(diagonal, nelec)
            residual = unitary @ np.diag(occupations) @ unitary.T - target
            return (np.linalg.norm(residual) / target_norm)**2

        best = None
        starts = [np.zeros(len(candidate['ry_wires']))]
        starts += [rng.uniform(-np.pi, np.pi, len(candidate['ry_wires']))
                   for _ in range(restarts - 1)]
        for initial in starts:
            result = scipy.optimize.minimize(
                    objective, initial, method='L-BFGS-B',
                    bounds=[(-np.pi, np.pi)] * len(initial),
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-9})
            if best is None or result.fun < best.fun:
                best = result

        unitary = circuit_unitary(
                best.x, candidate['ry_wires'], candidate['cnot_after'],
                cnot_matrices, nqubits)
        occupations = project_occupations(
                np.diag(unitary.T @ target @ unitary), nelec)
        errors.append(float(np.sqrt(best.fun)))
        theta_kpts.append(best.x.tolist())
        occupations_kpts.append(occupations.tolist())

    return max(errors), errors, theta_kpts, occupations_kpts


def random_candidate(nparam, nqubits, rng, ncnot=None):
    while True:
        ry_wires = rng.integers(0, nqubits, nparam).tolist()
        if len(set(ry_wires)) == min(nparam, nqubits):
            break
    if ncnot is None:
        ncnot = int(rng.integers(max(2, nqubits), 2 * nparam + 1))
    cnot_after = [[] for _ in range(nparam)]
    for _ in range(ncnot):
        control = int(rng.integers(0, nqubits))
        target = int(rng.integers(0, nqubits - 1))
        if target >= control:
            target += 1
        slot = int(rng.integers(0, nparam))
        cnot_after[slot].append([control, target])
    return {'ry_wires': ry_wires, 'cnot_after': cnot_after}


def si_overlap(kmesh):
    from pyscfad.pbc import dft, gto

    lattice = 5.46
    cell = gto.Cell()
    cell.atom = f'Si 0 0 0; Si {lattice/4} {lattice/4} {lattice/4}'
    cell.a = [[0., lattice/2, lattice/2],
              [lattice/2, 0., lattice/2],
              [lattice/2, lattice/2, 0.]]
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pbe'
    cell.build()
    mean_field = dft.KRKS(cell, xc='PBE', kpts=cell.make_kpts(kmesh))
    return np.asarray(mean_field.get_ovlp(cell, mean_field.kpts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', default='data/Si_sc1x1x1_k1x1x1_pyscf.out')
    parser.add_argument('--k', nargs=3, type=int, default=[1, 1, 1])
    parser.add_argument('--candidates', type=int, default=250)
    parser.add_argument('--restarts', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261102)
    parser.add_argument('--ncnot', type=int,
                        help='use exactly this many CNOT gates per candidate')
    parser.add_argument('--ry-wires', nargs='+', type=int,
                        help='fixed RY wire sequence')
    parser.add_argument('--all-single-cnot', action='store_true',
                        help='enumerate every one-CNOT direction and position')
    parser.add_argument('--output', default='data/Si_N8_rycnot_search')
    args = parser.parse_args()

    with h5py.File(args.reference) as handle:
        rho_ao = np.asarray(handle['result/rho'])
        occ_reference = np.asarray(handle['result/occ'])
        nelec = int(round(occ_reference[0].sum()))
    overlap = si_overlap(args.k)
    if len(rho_ao) != len(overlap):
        raise ValueError('Reference k-point count does not match --k')
    targets = []
    for overlap_k, rho_k in zip(overlap, rho_ao):
        overlap_sqrt = scipy.linalg.fractional_matrix_power(overlap_k, 0.5)
        target = overlap_sqrt @ rho_k @ overlap_sqrt
        target = np.real_if_close((target + target.T.conj()) / 2.0)
        if np.max(np.abs(np.imag(target))) > 1e-8:
            raise ValueError('Complex k-point density requires a complex ansatz')
        targets.append(np.real(target))
    norb = targets[0].shape[0]
    nqubits = int(round(np.log2(norb)))
    if 2**nqubits != norb:
        raise ValueError('The search currently requires a power-of-two orbital count')

    rng = np.random.default_rng(args.seed)
    if args.all_single_cnot:
        ry_wires = args.ry_wires or [wire % nqubits for wire in range(norb)]
        candidate_specs = []
        for slot in range(len(ry_wires)):
            for control in range(nqubits):
                for target_wire in range(nqubits):
                    if control == target_wire:
                        continue
                    cnot_after = [[] for _ in ry_wires]
                    cnot_after[slot] = [[control, target_wire]]
                    candidate_specs.append({
                        'ry_wires': list(ry_wires),
                        'cnot_after': cnot_after,
                    })
    else:
        candidate_specs = [random_candidate(
                len(args.ry_wires) if args.ry_wires else norb,
                nqubits, rng, args.ncnot) for _ in range(args.candidates)]
        if args.ry_wires:
            for candidate in candidate_specs:
                candidate['ry_wires'] = list(args.ry_wires)

    candidates = []
    total_candidates = len(candidate_specs)
    for index, candidate in enumerate(candidate_specs):
        error, errors, theta, occupations = fit_circuit(
                candidate, targets, nelec, nqubits, args.restarts, rng)
        candidate.update(index=index, rho_error=error,
                         rho_errors=errors, theta=theta,
                         occupations=occupations,
                         ncnot=sum(map(len, candidate['cnot_after'])))
        candidates.append(candidate)
        if (index + 1) % 25 == 0:
            best = min(item['rho_error'] for item in candidates)
            print(f'{index + 1:4d}/{total_candidates}: best rho error={best:.6e}')

    candidates.sort(key=lambda item: item['rho_error'])
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output + '.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(
                handle, fieldnames=['rank', 'index', 'rho_error', 'ncnot',
                                    'ry_wires', 'cnot_after'])
        writer.writeheader()
        for rank, candidate in enumerate(candidates, 1):
            writer.writerow({
                'rank': rank,
                'index': candidate['index'],
                'rho_error': candidate['rho_error'],
                'ncnot': candidate['ncnot'],
                'ry_wires': json.dumps(candidate['ry_wires']),
                'cnot_after': json.dumps(candidate['cnot_after']),
            })
    for rank, candidate in enumerate(candidates[:10], 1):
        circuit = {
            'name': f'N8-search-rank{rank}',
            'ry_wires': candidate['ry_wires'],
            'cnot_after': candidate['cnot_after'],
            'search_rho_error': candidate['rho_error'],
            'search_rho_errors': candidate['rho_errors'],
            'search_theta': candidate['theta'],
            'search_occupations': candidate['occupations'],
        }
        with open(f'{args.output}_rank{rank}.json', 'w') as handle:
            json.dump(circuit, handle, indent=2)
    print(f'best rho error={candidates[0]["rho_error"]:.8e}')
    print(f'Saved {args.output}.csv and top-10 JSON circuits')


if __name__ == '__main__':
    main()

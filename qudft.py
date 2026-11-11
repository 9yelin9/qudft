#!/usr/bin/env python3
import os
import re
import sys
import time
import h5py
import hashlib
import scipy
import argparse
import numpy as np

import jax.numpy as jnp

from lib.bands import plot_band_structure
from lib.cells import supercell
from lib.grouped_vqe import grouped_vqe_class
from lib.occupations import install_fractional_occupations
from lib.vqe import VQE

np.random.seed(42)
np.set_printoptions(precision=3, linewidth=120, suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument('mat', type=str)

# Value options
parser.add_argument('--k', nargs=3, type=int, default=[1, 1, 1])
parser.add_argument('--sc', nargs=3, type=int, default=[1, 1, 1])
parser.add_argument('--tol', type=float, default=1e-4)
parser.add_argument('--max-itr', type=int, default=500)
parser.add_argument('--shots', type=int, default=1000,
                    help='shots per circuit for --qpu (default: 1000)')
parser.add_argument('--sigma', type=float, default=0.01,
                    help='Fermi smearing in Hartree (default: 0.01)')
parser.add_argument('--dev', choices=['default.qubit', 'ionq.simulator', 'ionq.qpu'],
                    default='default.qubit',
                    help='quantum device (default: default.qubit)')
parser.add_argument('--ansatz', choices=['au-oddy', 'au-all', 'au-swap', 'au-grow', 'au-fixed', 'au-prio', 'rycn', 'ao-full'],
                    default='au-oddy',
                    help=('orbital-rotation circuit; ao-full applies all '
                          'intra-atom then all inter-atom AO mixings at Gamma '
                          '(default: au-oddy)'))
parser.add_argument('--topology', choices=['linear', 'reverse', 'ring', 'all'],
                    default='linear', help='CNOT pattern for --ansatz rycn')
parser.add_argument('--nparam', type=int, default=28,
                    help=('number of parameters for --ansatz au-prio or rycn; '
                          'fixed active count for au-swap or maximum active '
                          'count for au-grow; explicit gate count for au-fixed'))
parser.add_argument('--ncnot', type=int, default=12,
                    help='number of CNOT gates for --ansatz rycn')
parser.add_argument('--circuit', type=str,
                    help='JSON RYCN circuit; overrides topology/nparam/ncnot')
parser.add_argument('--paulis', nargs='+', type=str.upper,
                    help=('explicit Pauli strings for --ansatz au-fixed; '
                          'the count must equal --nparam'))
parser.add_argument('--band', nargs='?', const='', metavar='FILE',
                    help='plot bands; FILE is optional with --pyscf')
parser.add_argument('--resume', metavar='FILE',
                    help=('resume an interrupted .out checkpoint; max-itr is '
                          'the final absolute iteration limit'))

# Boolean flags
parser.add_argument('--ground-state', action='store_true',
                    help=('optimize zero-temperature etot without changing '
                          'the selected occupation mode'))
parser.add_argument('--min-coverage', action='store_true',
                    help='require au-prio to cover every valid AO pair')
parser.add_argument('--gradient-history', action='store_true',
                    help='save every iteration\'s theta-gradient vector')
occupation_group = parser.add_mutually_exclusive_group()
occupation_group.add_argument(
        '--fixed-occ', dest='occupation_mode', action='store_const',
        const='fixed',
        help='fix Aufbau occupations (default)')
occupation_group.add_argument(
        '--fock-occ', dest='occupation_mode', action='store_const',
        const='fock',
        help=('update fractional occupations from circuit-orbital '
              'Fock expectation values'))
occupation_group.add_argument(
        '--var-occ', dest='occupation_mode',
        action='store_const', const='variational',
        help='optimize fractional occupations through eta')
parser.set_defaults(occupation_mode='fixed')
parser.add_argument('--group', action='store_true',
                    help='measure rho with RLF Pauli grouping (requires --qpu)')
parser.add_argument('--grouping', choices=['gc', 'qwc'], default='gc',
                    help='Pauli grouping type used with --group (default: gc)')
parser.add_argument('--qpu', action='store_true')
parser.add_argument('--dry-run', action='store_true',
                    help='submit IonQ jobs in dry-run mode (requires --qpu and an IonQ device)')
parser.add_argument('--pyscf', action='store_true')
args = parser.parse_args()
args.fixed_occ = args.occupation_mode == 'fixed'
args.fock_occ = args.occupation_mode == 'fock'
if args.sigma < 0:
    parser.error('--sigma must be non-negative')
if args.shots < 1:
    parser.error('--shots must be positive')
if args.dev != 'default.qubit' and not args.qpu:
    parser.error(f'--dev {args.dev} requires --qpu')
if args.group and not args.qpu:
    parser.error('--group requires --qpu')
if not args.group and args.grouping != 'gc':
    parser.error('--grouping requires --group')
if args.dry_run and (not args.qpu or not args.dev.startswith('ionq.')):
    parser.error('--dry-run requires --qpu with an IonQ device')
if args.resume and (args.fock_occ or args.ansatz in ('au-swap', 'au-grow')):
    parser.error('--resume currently supports non-adaptive qudftv2 optimization')
if args.ansatz == 'au-fixed':
    if args.circuit:
        parser.error('--circuit cannot be used with --ansatz au-fixed')
    if not args.paulis:
        parser.error('--ansatz au-fixed requires --paulis')
    if len(args.paulis) != args.nparam:
        parser.error('--paulis count must equal --nparam')
elif args.paulis:
    parser.error('--paulis can only be used with --ansatz au-fixed')


GroupedVQE = grouped_vqe_class(VQE)


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
    mat.atom = f'Al 0 0 0'
    mat.a = [[0., a/2, a/2], [a/2, 0., a/2], [a/2, a/2, 0.]]
    mat.spin = 1
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

mf.sigma = 0.0 if args.ground_state else args.sigma
mf.smearing_method = 'fermi'
fn_head = f'{args.mat}_sc{'x'.join(map(str, args.sc))}_k{'x'.join(map(str, args.k))}'
pyscf_output_suffix = '_pyscf_gs' if args.ground_state else '_pyscf'

if args.band is not None:
    plot_band_structure(mf, args.band or None, compare_pyscf=args.pyscf,
                        stem=fn_head,
                        fn_pyscf=f'data/{fn_head}{pyscf_output_suffix}.out')
    sys.exit(0)

if args.pyscf:
    t0 = time.time()

    install_fractional_occupations(mf)
    mf.kernel()

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
    occ = np.asarray(mf.mo_occ)
    energy = np.asarray(mf.mo_energy)
    probability = occ / 2.0
    entropy = -(2.0 * w) * np.sum(
            scipy.special.xlogy(probability, probability) +
            scipy.special.xlogy(1.0 - probability, 1.0 - probability))
    efree = etot - mf.sigma * entropy

    print(f'e1={e1:f} eh={eh:f} exc={exc:f} etot={etot:f} '
          f'entropy={entropy:f} efree={efree:f}')

    t1 = time.time()
    tm, ts = divmod(int(t1 - t0), 60)
    print(f'time elapsed: {tm}m {ts}s\n')

    fn_out = f'data/{fn_head}{pyscf_output_suffix}.out'
    with h5py.File(fn_out, 'w') as f:
        result = f.create_group('result')
        for key, val in {
                't0': t0,
                't1': t1,
                'rho': rho,
                'occ': occ,
                'energy': energy,
                'e1': e1,
                'eh': eh,
                'exc': exc,
                'etot': etot,
                'entropy': entropy,
                'efree': efree,
                'sigma': mf.sigma,
                }.items():
            result.create_dataset(key, data=np.asarray(val))
    print(f'Saved at {fn_out}\n')
    sys.exit(0)

ansatz_suffix = ''
if args.circuit:
    circuit_stem = os.path.splitext(os.path.basename(args.circuit))[0]
    ansatz_suffix = f'_rycn-{circuit_stem}'
elif args.ansatz == 'rycn':
    ansatz_suffix = f'_rycn-{args.topology}_P{args.nparam}_C{args.ncnot}'
elif args.ansatz == 'au-prio':
    ansatz_suffix = f'_au-prio_P{args.nparam}'
elif args.ansatz == 'au-all':
    ansatz_suffix = '_au-all'
elif args.ansatz == 'au-swap':
    ansatz_suffix = f'_au-swap_P{args.nparam}'
elif args.ansatz == 'au-grow':
    ansatz_suffix = f'_au-grow_Pmax{args.nparam}'
elif args.ansatz == 'au-fixed':
    pauli_tag = hashlib.sha1(','.join(args.paulis).encode()).hexdigest()[:8]
    ansatz_suffix = f'_au-fixed_P{args.nparam}_{pauli_tag}'
elif args.ansatz == 'ao-full':
    ansatz_suffix = '_ao-full'
if args.ground_state:
    ansatz_suffix += '_gs'
if args.fixed_occ:
    ansatz_suffix += '_fixed-occ'
elif args.fock_occ:
    ansatz_suffix += '_fock-occ'
if args.group:
    ansatz_suffix += f'_{args.grouping}'

if args.fock_occ:
    install_fractional_occupations(mf)

vqe_cls = GroupedVQE if args.group else VQE
grouping_options = ({'grouping_type': args.grouping} if args.group else {})
theta_seed = 42
rng = np.random.default_rng(theta_seed)
vqe = vqe_cls(
        mat, mf,
        f'{fn_head}_qudft{ansatz_suffix}',
        is_qpu=args.qpu,
        shots=args.shots,
        device_name=args.dev,
        ansatz=args.ansatz,
        topology=args.topology,
        nparam=args.nparam,
        ncnot=args.ncnot,
        circuit=args.circuit,
        ground_state=args.ground_state,
        min_coverage=args.min_coverage,
        fixed_occupation=args.fixed_occ,
        fock_occupation=args.fock_occ,
        fixed_paulis=args.paulis,
        resume_from=args.resume,
        dry_run=args.dry_run,
        **grouping_options,
        )
vqe.theta_seed = theta_seed
with open(vqe.fn_in, 'a') as f:
    f.write(f'theta_seed {theta_seed}\n\n')

theta = jnp.asarray(0.01 * rng.uniform(-1, 1, vqe.ntheta))
if args.fixed_occ:
    eta = jnp.asarray([])
else:
    npaired = vqe.nelec // 2
    eta_init = [4.0] * npaired
    if vqe.nelec % 2:
        eta_init += [0.0]
    eta_init += [-4.0] * (vqe.N - len(eta_init))
    eta = jnp.array(vqe.nkpts * eta_init)
vqe.qudftv2(
        theta, eta, tol=args.tol, max_itr=args.max_itr,
        record_gradients=args.gradient_history, resume=bool(args.resume))

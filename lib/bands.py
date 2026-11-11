"""Band-structure calculation and plotting helpers."""

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy
from ase.cell import Cell

HA2EV = 27.211386245988


def plot_band_structure(mf, fn_out=None, npoints=200, path=None, ylim=None,
                        compare_pyscf=False, stem=None, fn_pyscf=None):
    rho = None
    occ = None
    if fn_out:
        with h5py.File(fn_out, 'r') as f:
            rho = np.asarray(f['result/rho'])
            if 'result/occ' in f:
                occ = np.asarray(f['result/occ'])
    elif not compare_pyscf:
        raise ValueError('A quDFT .out file or --pyscf is required for --band')

    bandpath = Cell(np.asarray(mf.cell.lattice_vectors())).bandpath(path=path, npoints=npoints)
    kpts_band = mf.cell.get_abs_kpts(bandpath.kpts)

    def get_band_energy(rho_kpts, occ_kpts=None, align_occ=False,
                        efermi=None):
        # PySCFAD의 VXC 객체를 처리하기 위해 mf.get_bands 대신 Fock 행렬을
        # 명시적으로 만든다.
        h1e = mf.get_hcore(mf.cell, kpts_band)
        s1e = mf.get_ovlp(mf.cell, kpts_band)
        vhf = mf.get_veff(mf.cell, rho_kpts, kpts=mf.kpts,
                          kpts_band=kpts_band)
        fock = mf.get_fock(h1e, s1e, vhf, rho_kpts)
        energy, _ = mf.eig(fock, s1e)
        energy = np.asarray(energy)

        # SCF mesh에서 Fermi energy를 구해 band energy의 기준으로 사용한다.
        h1e_mesh = mf.get_hcore(mf.cell, mf.kpts)
        s1e_mesh = mf.get_ovlp(mf.cell, mf.kpts)
        vhf_mesh = mf.get_veff(mf.cell, rho_kpts, kpts=mf.kpts)
        fock_mesh = mf.get_fock(h1e_mesh, s1e_mesh, vhf_mesh, rho_kpts)
        energy_mesh, coeff_mesh = mf.eig(fock_mesh, s1e_mesh)
        energy_mesh = np.asarray(energy_mesh)
        coeff_mesh = np.asarray(coeff_mesh)
        if occ_kpts is None:
            raise ValueError('Occupation data is required to determine E_F')
        occ_kpts = np.asarray(occ_kpts)
        if occ_kpts.shape != energy_mesh.shape:
            raise ValueError(f'occ has shape {occ_kpts.shape}; '
                             f'expected {energy_mesh.shape}')

        if align_occ:
            # quDFT occupation은 variational-orbital 순서로 저장된다. 저장된
            # rho를 energy-sorted KS basis에 투영해 orbital permutation을
            # 찾고, 그 순열만 occ에 적용한다.
            occ_aligned = np.empty_like(occ_kpts)
            for k in range(len(mf.kpts)):
                coeff_k = coeff_mesh[k]
                overlap_k = np.asarray(s1e_mesh[k])
                rho_k = np.asarray(rho_kpts[k])
                rho_mo = (coeff_k.conj().T @ overlap_k @ rho_k @
                          overlap_k @ coeff_k)
                occ_projected = np.real(np.diag(rho_mo))
                row, column = scipy.optimize.linear_sum_assignment(
                        np.abs(occ_kpts[k, :, None] -
                               occ_projected[None, :]))
                occ_aligned[k, column] = occ_kpts[k, row]
            occ_kpts = occ_aligned

        if efermi is None:
            # 저장된 occupation의 총 전자 수를 사용한다. 개별 quDFT
            # occupation은 최적화 중 완전한 Fermi-Dirac 분포가 아닐 수 있어
            # 각 값에서 chemical potential을 직접 역산하지 않는다.
            energy_flat = energy_mesh.ravel()
            nelec_total = float(occ_kpts.sum())
            partial = (occ_kpts > 0.05) & (occ_kpts < 1.95)
            if np.any(partial):
                def electron_error(mu):
                    occupation = 2.0 * scipy.special.expit(
                            (mu - energy_flat) / mf.sigma)
                    return occupation.sum() - nelec_total

                margin = max(1.0, 50.0 * mf.sigma)
                efermi = scipy.optimize.brentq(
                        electron_error, energy_flat.min() - margin,
                        energy_flat.max() + margin)
            else:
                nocc_total = int(round(nelec_total / 2.0))
                energy_sorted = np.sort(energy_flat)
                if nocc_total == 0 or nocc_total == energy_sorted.size:
                    raise ValueError('Cannot locate occupied/unoccupied band edge')
                efermi = 0.5 * (energy_sorted[nocc_total - 1] +
                                energy_sorted[nocc_total])
        return (energy - efermi) * HA2EV, efermi

    x, special_x, labels = bandpath.get_linear_kpoint_axis()

    fig, ax = plt.subplots(constrained_layout=True, dpi=150)

    ev_pyscf = None
    efermi_ref = None
    if compare_pyscf:
        if fn_pyscf is None or not os.path.exists(fn_pyscf):
            raise FileNotFoundError(
                    f'PySCF result not found: {fn_pyscf}. '
                    'Run with --pyscf first.')
        with h5py.File(fn_pyscf, 'r') as f:
            rho_pyscf = np.asarray(f['result/rho'])
            occ_pyscf = np.asarray(f['result/occ'])
        ev_pyscf, efermi_ref = get_band_energy(
                rho_pyscf, occ_pyscf)
        for band in range(ev_pyscf.shape[1]):
            ax.plot(x, ev_pyscf[:, band], color='black', linewidth=1,
                    linestyle='-', label='PySCF' if band == 0 else None)

    ev_qudft = None
    if rho is not None:
        ev_qudft, _ = get_band_energy(
                rho, occ, align_occ=True, efermi=efermi_ref)

    if ev_qudft is not None:
        for band in range(ev_qudft.shape[1]):
            ax.plot(x, ev_qudft[:, band], color='red', linestyle='None',
                    marker='.', markersize=2.5,
                    label='quDFT' if band == 0 else None)

    ax.legend()

    ax.axhline(0, color='0.35', linestyle='--', linewidth=0.8)
    for xpos in special_x:
        ax.axvline(xpos, color='0.8', linewidth=0.7)
    ax.set(xlim=(x[0], x[-1]), xticks=special_x, xticklabels=labels,
           ylabel=r'$E - E_\mathrm{F}$ (eV)')
    if ylim is not None:
        ax.set_ylim(*ylim)

    os.makedirs('fig', exist_ok=True)
    if stem is None:
        stem = os.path.splitext(os.path.basename(fn_out))[0]
    if ev_qudft is not None and compare_pyscf:
        suffix = '_qudft_pyscf.png'
    elif compare_pyscf:
        suffix = '_pyscf.png'
    else:
        suffix = '_qudft.png'
    fn_fig = os.path.join('fig', stem + suffix)
    fig.savefig(fn_fig, dpi=300)
    plt.close(fig)
    print(f'Band structure saved at {fn_fig}')
    return x, ev_qudft, ev_pyscf

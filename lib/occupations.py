"""Occupation helpers used by the PySCF reference calculation."""

import numpy as np
import scipy
import jax.numpy as jnp
from types import MethodType


def get_occ_pyscf_fractional(mf, mo_energy_kpts=None, mo_coeff_kpts=None):
    """Fermi occupations with the exact (possibly odd) electron count."""
    if mo_energy_kpts is None:
        mo_energy_kpts = mf.mo_energy

    energy = np.asarray(mo_energy_kpts, dtype=float)
    shape = energy.shape
    energy_flat = energy.reshape(-1)
    nkpts = len(mf.kpts)
    nelectron = float(mf.cell.tot_electrons(nkpts))

    sigma = mf.sigma
    if sigma is None or sigma <= 0:
        order = np.argsort(energy_flat)
        occ_flat = np.zeros_like(energy_flat)
        npair, odd = divmod(int(round(nelectron)), 2)
        occ_flat[order[:npair]] = 2.0
        if odd:
            occ_flat[order[npair]] = 1.0
        return jnp.asarray(occ_flat.reshape(shape))

    sigma = float(sigma)

    def occupation(mu):
        return 2.0 * scipy.special.expit((mu - energy_flat) / sigma)

    margin = max(1.0, 50.0 * sigma)
    mu_lo = float(energy_flat.min() - margin)
    mu_hi = float(energy_flat.max() + margin)
    for _ in range(100):
        mu = 0.5 * (mu_lo + mu_hi)
        if occupation(mu).sum() < nelectron:
            mu_lo = mu
        else:
            mu_hi = mu

    occ = occupation(0.5 * (mu_lo + mu_hi)).reshape(shape)
    return jnp.asarray(occ)


def install_fractional_occupations(mf):
    """Install the exact-electron-count occupation routine on a PySCF object."""
    mf.get_occ = MethodType(get_occ_pyscf_fractional, mf)

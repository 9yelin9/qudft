#!/usr/bin/env python3
"""Estimate one quDFT optimizer iteration with IonQ QPU dry-run jobs."""

import argparse
import json
import os
import time

import jax.numpy as jnp
import numpy as np
import pennylane as qml
import requests

from pennylane_ionq.api_client import APIClient, Job
from pyscfad.pbc import dft, gto

from lib.cells import supercell
from lib.grouped_vqe import grouped_vqe_class
from lib.vqe import VQE


def silicon_gamma_cell():
    """Return the same primitive Si Gamma-point system as qudft.py."""
    a = 5.46
    cell = gto.Cell()
    cell.atom = f'Si 0 0 0; Si {a/4} {a/4} {a/4}'
    cell.a = [[0.0, a / 2, a / 2],
              [a / 2, 0.0, a / 2],
              [a / 2, a / 2, 0.0]]
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pbe'
    cell.build()
    cell = supercell(cell, [1, 1, 1])
    cell.ncell = 1
    mf = dft.KRKS(cell, xc='PBE', kpts=cell.make_kpts([1, 1, 1]))
    mf.sigma = 0.01
    mf.smearing_method = 'fermi'
    return cell, mf


def shifted_parameters(theta):
    """Return the base and two-term parameter-shift points for four angles."""
    points = [theta]
    for index in range(theta.size):
        plus = theta.at[index].add(np.pi / 2)
        minus = theta.at[index].add(-np.pi / 2)
        points.extend((plus, minus))
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shots', type=int, required=True)
    parser.add_argument('--grouping', choices=['gc', 'qwc'], required=True)
    parser.add_argument('--limit-jobs', type=int)
    parser.add_argument('--output')
    args = parser.parse_args()

    tracked_jobs = []
    original_init = Job.__init__

    def tracked_init(self, *job_args, **job_kwargs):
        original_init(self, *job_args, **job_kwargs)
        tracked_jobs.append(self)

    Job.__init__ = tracked_init
    try:
        cell, mf = silicon_gamma_cell()
        grouped_vqe = grouped_vqe_class(VQE)
        vqe = grouped_vqe(
            cell, mf,
            f'cost-estimate_{args.grouping}_shots{args.shots}',
            is_qpu=True,
            shots=args.shots,
            device_name='ionq.qpu',
            ansatz='au-fixed',
            nparam=4,
            fixed_occupation=True,
            fixed_paulis=['YZZ', 'YIZ', 'YZI', 'YII'],
            dry_run=True,
            grouping_type=args.grouping,
        )
        rng = np.random.default_rng(42)
        theta = jnp.asarray(0.01 * rng.uniform(-1, 1, vqe.nparam))
        points = shifted_parameters(theta)

        job_ids = []
        stop = False
        for basis in vqe.bases:
            for qnode in vqe.rho_group_qnodes:
                tapes = [qnode.construct((point, basis), {})
                         for point in points]
                before = len(tracked_jobs)
                qml.execute(tapes, vqe.dev, diff_method=None)
                created = tracked_jobs[before:]
                if len(created) != 1 or created[0].id.value is None:
                    raise RuntimeError('Could not identify submitted IonQ job')
                job_ids.append(created[0].id.value)
                print(f'submitted {len(job_ids)}: {job_ids[-1]}', flush=True)
                if args.limit_jobs and len(job_ids) >= args.limit_jobs:
                    stop = True
                    break
            if stop:
                break
    finally:
        Job.__init__ = original_init

    client = APIClient()
    costs = []
    job_details = []
    for job_id in job_ids:
        payload = {}
        for _ in range(15):
            response = client.get(f'jobs/{job_id}/cost')
            payload = response.json()
            if payload.get('estimated_cost'):
                break
            time.sleep(1)
        estimate = payload.get('estimated_cost') or {}
        value = float(estimate.get('value', 0.0))
        costs.append(value)
        job_details.append({
            'id': job_id,
            'estimated_cost': value,
            'unit': estimate.get('unit'),
        })

    result = {
        'shots': args.shots,
        'grouping': args.grouping,
        'groups': len(vqe.rho_group_qnodes),
        'basis_states': len(vqe.bases),
        'circuits_per_job': len(points),
        'jobs_submitted': len(job_ids),
        'circuits_submitted': len(job_ids) * len(points),
        'total_shots_submitted': len(job_ids) * len(points) * args.shots,
        'estimated_cost': float(np.sum(costs)),
        'currency': (job_details[0]['unit'] if job_details else None),
        'jobs': job_details,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as output:
            json.dump(result, output, indent=2)


if __name__ == '__main__':
    main()

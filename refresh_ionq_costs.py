#!/usr/bin/env python3
"""Refresh missing IonQ dry-run cost estimates in result JSON files."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from pennylane_ionq.api_client import APIClient


def get_cost(job_id):
    payload = APIClient().get(f'jobs/{job_id}/cost').json()
    estimate = payload.get('estimated_cost') or {}
    if 'value' not in estimate:
        raise RuntimeError(f'Cost is still unavailable for {job_id}')
    return job_id, float(estimate['value']), estimate.get('unit')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()

    for path in args.files:
        with open(path) as source:
            result = json.load(source)
        missing = [job['id'] for job in result['jobs']
                   if not job.get('estimated_cost')]
        with ThreadPoolExecutor(max_workers=12) as executor:
            refreshed = dict(
                (job_id, (value, unit))
                for job_id, value, unit in executor.map(get_cost, missing))
        for job in result['jobs']:
            if job['id'] in refreshed:
                value, unit = refreshed[job['id']]
                job['estimated_cost'] = value
                job['unit'] = unit
        result['estimated_cost'] = sum(
            job['estimated_cost'] for job in result['jobs'])
        units = {job['unit'] for job in result['jobs'] if job['unit']}
        result['currency'] = units.pop() if len(units) == 1 else None
        with open(path, 'w') as output:
            json.dump(result, output, indent=2)
        print(path, result['estimated_cost'])


if __name__ == '__main__':
    main()

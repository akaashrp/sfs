"""Cloud-only Qwen load points; historical Bridges manifests remain unchanged."""
import argparse
import copy
from pathlib import Path

from scripts.cloud.common import digest, read, write, validate_bundle

QWEN_QPS = (7., 8., 8.6, 8.75)


def update_rates(manifest):
    result = copy.deepcopy(manifest)
    for cell in result['cells']:
        if cell['family'] != 'qwen':
            continue
        old = cell['qps']
        cell['qps'] = {8.3: 8., 8.9: 8.75}.get(old, old)
        if cell['qps'] not in QWEN_QPS:
            raise ValueError(f'Unexpected Qwen load point: {old}')
        cell['id'] = cell['id'].rsplit('-', 1)[0] + f'-{cell["qps"]:g}'
    if len({c['id'] for c in result['cells']}) != len(result['cells']):
        raise ValueError('Rate update produced duplicate cells')
    if 'families' in result:
        result['families']['qwen']['qps'] = list(QWEN_QPS)
    return result


def update_bundle(bundle, audit):
    bundle = Path(bundle).resolve()
    before = validate_bundle(bundle)
    after = update_rates(before)
    previous_hash = digest(bundle/'bundle.json')
    if before != after:
        backup = bundle/'bundle.before-qps-20260916.json'
        if backup.exists() and read(backup) != before:
            raise ValueError('Conflicting bundle backup')
        if not backup.exists():
            write(backup, before)
        write(bundle/'bundle.json', after)
    write(audit, {'status': 'PASS_QWEN_RATE_UPDATE', 'qwen_qps': list(QWEN_QPS),
          'previous_bundle_sha256': previous_hash, 'bundle_sha256': digest(bundle/'bundle.json'),
          'artifact_file_hashes_unchanged': before['files'] == after['files'],
          'ministral_cells_unchanged': [c for c in before['cells'] if c['family'] == 'ministral'] ==
                                      [c for c in after['cells'] if c['family'] == 'ministral']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--audit', required=True)
    options = parser.parse_args()
    update_bundle(options.bundle, options.audit)

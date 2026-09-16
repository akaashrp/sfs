"""Prefetch pinned checkpoints into the shared cloud cache without serving dependencies."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scripts.cloud.common import digest, read, write


def download_models(models, cache, output, *, workers=2, family='all'):
    from huggingface_hub import snapshot_download

    selected = {key: value for key, value in models.items()
                if family == 'all' or value['family'] == family}
    if not selected or workers < 1:
        raise ValueError('Select at least one model and a positive worker count')
    cache = Path(cache).resolve()
    output = Path(output)
    paths, audits = {}, {}

    def fetch(key, model):
        print(f'Downloading {key} at {model["revision"]}', flush=True)
        patterns = ['*.json', '*.jinja', '*.model', '*.tiktoken', 'tekken.json']
        patterns += (['consolidated.safetensors'] if model['family'] == 'ministral'
                     else ['*.safetensors'])
        path = Path(snapshot_download(repo_id=model['repo'], revision=model['revision'],
                                     cache_dir=cache, allow_patterns=patterns, max_workers=4))
        weights = list(path.glob('*.safetensors'))
        if not weights:
            raise ValueError(f'Missing weights for {key}')
        index = path / 'model.safetensors.index.json'
        if model['family'] == 'qwen' and index.exists():
            for shard in set(read(index)['weight_map'].values()):
                if not (path / shard).is_file():
                    raise ValueError(f'Missing shard for {key}: {shard}')
        audit = {str(p): digest(p) for p in path.iterdir() if p.is_file()}
        print(f'Complete and hashed: {key}', flush=True)
        return key, str(path), audit

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, key, model) for key, model in selected.items()]
        for future in as_completed(futures):
            key, path, audit = future.result()
            paths[key], audits[key] = path, audit
            write(str(output) + '.progress.json', {
                'completed': sorted(paths), 'remaining': sorted(set(selected) - set(paths)),
                'paths': paths, 'revisions': selected})
    # Publish the serving map only after every requested checkpoint is complete.
    if output.exists():
        previous = read(output)
        for key in set(previous) & set(paths):
            if previous[key] != paths[key]:
                raise ValueError(f'Existing model path disagrees: {key}')
        paths = {**previous, **paths}
    write(output, paths)
    write(str(output) + '.audit.json', {
        'revisions': selected,
        'file_sha256': {name: value for audit in audits.values() for name, value in audit.items()}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--family', choices=['all', 'qwen', 'ministral'], default='all')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    download_models(read(args.manifest)['models'], args.cache, args.output,
                    workers=args.workers, family=args.family)

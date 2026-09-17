from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + f".{os.getpid()}.pending")
    with pending.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def set_option(argv, flag, value):
    """Replace every old scalar occurrence so effective CLI values are unambiguous."""
    out, i = [], 0
    while i < len(argv):
        if argv[i] == flag:
            i += 2
        else:
            out.append(argv[i]); i += 1
    return [*out, flag, str(value)]


def expand(value, bundle, models):
    replacements = {"@BUNDLE@": str(Path(bundle).resolve()), "@ROOT@": str(ROOT)}
    replacements.update({f"@MODEL_{key}@": str(path) for key, path in models.items()})
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [expand(v, bundle, models) for v in value]
    if isinstance(value, dict):
        return {k: expand(v, bundle, models) for k, v in value.items()}
    return value


def source_hashes():
    paths = [*ROOT.joinpath("src").rglob("*.py"), *ROOT.joinpath("src/assets").rglob("*"),
             *ROOT.joinpath("vllm/vllm").rglob("*.py")]
    paths += [p for p in (ROOT/'scripts/cloud').glob('*') if p.suffix in ('.py','.sh','.txt')]
    paths += [ROOT/'scripts/cloud/campaign.json',ROOT/'scripts/cloud/protected-source.json']
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths)) if p.is_file()}


def source_digest(hashes):
    """Stable digest of a source pin, used to name accepted prior source lineages."""
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_bundle(bundle):
    bundle = Path(bundle).resolve()
    manifest = read(bundle / "bundle.json")
    if manifest.get("schema_version") != 1 or len(manifest["cells"]) != 68:
        raise ValueError("Expected the frozen 68-cell bundle")
    for name, expected in manifest["files"].items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle) or digest(path) != expected:
            raise ValueError(f"Missing or changed bundle artifact: {name}")
    return manifest


def portable_cache(bundle, family):
    """Relocate only cache location metadata; frozen prompt bytes remain untouched."""
    bundle = Path(bundle).resolve()
    source = bundle/family/'holdout'
    base = Path(os.environ.get('SFS_INPUTS', str(ROOT/'.scratch/inputs'))).resolve()
    target = base/digest(bundle/'bundle.json')[:12]/family/'holdout'
    manifest = read(source/'manifest.json')
    manifest.update(source_bucket_dir=str(target), tokenizer_id=str(bundle/'tokenizers'/read(bundle/'bundle.json')['families'][family]['models'][1]))
    manifest['cloud_relocation'] = {'original_manifest_sha256':digest(source/'manifest.json'),
        'changed_fields':['source_bucket_dir','tokenizer_id'], 'prompt_bytes_changed':False}
    target.mkdir(parents=True,exist_ok=True)
    for path in source.glob('*.jsonl'):
        dest = target/path.name
        if dest.exists():
            if not dest.is_symlink() or dest.resolve() != path.resolve(): raise ValueError('Conflicting relocated cache')
        else:dest.symlink_to(path)
    dest = target/'manifest.json'
    if dest.exists():
        if read(dest) != manifest: raise ValueError('Relocated cache metadata changed')
    else:write(dest,manifest)
    return target


def portable_routebalance(bundle, family):
    bundle = Path(bundle).resolve()
    source = bundle/family/'routebalance'
    base = Path(os.environ.get('SFS_INPUTS', str(ROOT/'.scratch/inputs'))).resolve()
    target = base/digest(bundle/'bundle.json')[:12]/family/'routebalance'
    cache = Path(os.environ.get('HF_HUB_CACHE', str(ROOT/'.scratch/hf/hub'))).resolve()
    metadata = read(source/'metadata.json')
    encoder = metadata['encoder']
    snapshot = cache/('models--'+encoder['model_id'].replace('/', '--'))/'snapshots'/encoder['revision']
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    if not snapshot.exists(): snapshot.symlink_to(bundle/'encoder',target_is_directory=True)
    metadata['encoder_cache_dir'] = str(cache)
    metadata['cloud_relocation'] = {'original_metadata_sha256':digest(source/'metadata.json'),
        'changed_fields':['encoder_cache_dir'],'predictor_weights_changed':False}
    target.mkdir(parents=True,exist_ok=True)
    for path in source.iterdir():
        if path.name == 'metadata.json':continue
        dest = target/path.name
        if not dest.exists():dest.symlink_to(path)
        elif not dest.is_symlink() or dest.resolve() != path.resolve():raise ValueError('Conflicting native predictor')
    if (target/'metadata.json').exists():
        if read(target/'metadata.json') != metadata:raise ValueError('Relocated predictor metadata changed')
    else:write(target/'metadata.json',metadata)
    return target


@contextmanager
def locks(directory, keys):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    streams = []
    try:
        for key in sorted(set(keys)):
            stream = (directory / (key + ".lock")).open("a+")
            streams.append(stream)
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"Already owned by another process: {key}") from None
        yield
    finally:
        for stream in streams:
            stream.close()

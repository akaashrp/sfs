"""Freeze required launch settings into a batch script, independent of export policy."""
from pathlib import Path
import hashlib
import re
import shlex


def read_settings(path):
    settings = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"export ([A-Z][A-Z0-9_]*)=(.*)", line)
        if not match:
            raise ValueError("Launch settings must be literal export assignments")
        key, expression = match.groups()
        values = shlex.split(expression)
        if len(values) != 1 or key in settings:
            raise ValueError("Invalid or duplicate launch setting")
        settings[key] = values[0]
    if not settings.get("SFS_ROOT"):
        raise ValueError("SFS_ROOT must be frozen into the batch script")
    return settings


def render(script, settings):
    script = Path(script).resolve()
    data = script.read_bytes()
    lines = data.decode().splitlines(keepends=True)
    if not lines or not lines[0].startswith("#!"):
        raise ValueError("Expected executable shell script")
    position = max([0, *[i for i, line in enumerate(lines) if line.startswith("#SBATCH")]]) + 1
    if any(not re.fullmatch(r"[A-Z][A-Z0-9_]*", key) for key in settings):
        raise ValueError("Invalid environment variable name")
    checksum = hashlib.sha256(data).hexdigest() + "  " + str(script)
    block = ["\nset -euo pipefail\n",
             "printf '%s\\n' " + shlex.quote(checksum) + " | sha256sum --status -c\n"]
    block += ["export " + key + "=" + shlex.quote(str(value)) + "\n" for key, value in settings.items()]
    block += ["export SLURM_EXPORT_ENV=ALL\n",
              'if [[ "${SFS_PREFLIGHT_ONLY:-0}" == 1 ]]; then export VALIDATE_ONLY=1; else unset VALIDATE_ONLY; fi\n']
    return "".join(lines[:position] + block + lines[position:])

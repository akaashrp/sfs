"""Copy a wheel's compiled extensions into the source tree that shadows it on PYTHONPATH."""
import pathlib
import sys
import zipfile

wheel, dest = sys.argv[1], pathlib.Path(sys.argv[2])
with zipfile.ZipFile(wheel) as archive:
    names = [n for n in archive.namelist() if n.endswith('.so')]
    for name in names:
        archive.extract(name, dest)
print(f'placed {len(names)} extensions into {dest}')
for name in names:
    print(f'    {name}  {(dest / name).stat().st_size} bytes')
if not names:
    raise SystemExit('wheel carried no extensions')

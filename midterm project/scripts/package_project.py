"""Export portable source and optionally data/models without credentials/caches."""
import argparse
import hashlib
from pathlib import Path
import zipfile
from ml.common import ROOT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-models', action='store_true')
    parser.add_argument('--include-data', action='store_true')
    args = parser.parse_args()
    target = ROOT / 'release'
    target.mkdir(exist_ok=True)
    name = 'mailguard-complete.zip' if args.include_models else 'mailguard-source.zip'
    dest = target / name
    omitted = {'.venv', '.git', '__pycache__', '.pytest_cache', 'node_modules', 'release', 'logs', '.env', '.ssh'}
    with zipfile.ZipFile(dest, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
        for path in sorted(ROOT.rglob('*')):
            rel = path.relative_to(ROOT)
            if not path.is_file() or any(p in omitted for p in rel.parts):
                continue
            if rel.parts[0]=='artifacts' and not args.include_models:
                continue
            if rel.parts[:2] in [('data', 'raw'), ('data', 'processed')] and not args.include_data:
                continue
            if path.suffix in ['.pyc', '.tsbuildinfo']:
                continue
            archive.write(path, str(Path('midterm project') / rel))
    sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    dest.with_suffix('.zip.sha256').write_text(f'{sha}  {name}\n')
    print(dest)
    print('bytes:', dest.stat().st_size, 'sha256:', sha)

if __name__ == '__main__':
    main()

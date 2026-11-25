"""Expand a detached copy of the verified public outputs; no private data."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    archive = Path(__file__).resolve().parents[1]
    output = args.out.resolve()
    if output == archive or output.is_relative_to(archive):
        parser.error('Choose an output directory outside the published archive.')
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory must be empty.')
    mapping = json.loads((archive / 'source_files.json').read_text())
    for published, metadata in mapping.items():
        if not metadata['source'].startswith('results/'):
            continue
        content = (archive / published).read_bytes()
        if metadata['encoding'] == 'gzip':
            content = gzip.decompress(content)
        assert hashlib.sha256(content).hexdigest() == metadata['original_sha256']
        destination = (output / metadata['source'].removeprefix('results/')).resolve()
        assert destination.is_relative_to(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    print('Expanded unchanged public outputs and certificates into', output)


if __name__ == '__main__':
    main()

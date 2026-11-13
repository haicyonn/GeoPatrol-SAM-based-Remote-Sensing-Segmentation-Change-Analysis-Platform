"""WSL entry point: validate immutable inference artifacts and store in MySQL."""
import argparse
import json
from pathlib import Path

from .contract import local_path
from .repository import connect, ingest, migrate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--database', default='xunbian_backend', choices=['xunbian_backend', 'xunbian_backend_test'])
    args = parser.parse_args()
    path = local_path(args.manifest)
    if path.stat().st_size > 30_000_000:
        raise ValueError('Manifest exceeds the import size limit')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    with connect(args.database) as db:
        migrate(db)
        result = ingest(db, manifest)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()

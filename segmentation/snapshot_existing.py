"""Read-only fingerprints proving this new subsystem preserves existing facts."""
import argparse
import hashlib
import json
from pathlib import Path

from .repository import connect


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    args = parser.parse_args(); result = {'tables': {}, 'source_files': {}}
    with connect() as db, db.cursor() as cur:
        cur.execute('START TRANSACTION READ ONLY')
        cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema=DATABASE() AND table_name NOT LIKE 'segmentation\\_%' ORDER BY table_name")
        tables = [next(iter(row.values())) for row in cur.fetchall()]
        for table in tables:
            if not table.replace('_', '').isalnum():
                raise ValueError('Unexpected table identity')
            cur.execute(f'SELECT * FROM `{table}`')
            rows = sorted(json.dumps(row, sort_keys=True, ensure_ascii=False, default=str, separators=(',', ':')) for row in cur.fetchall())
            digest = hashlib.sha256('\n'.join(rows).encode()).hexdigest()
            result['tables'][table] = dict(rows=len(rows), sha256=digest)
        db.rollback()
    root = Path(__file__).resolve().parents[1]
    files = list((root / 'backend-java/src').rglob('*.java')) + [root / 'backend-java/src/main/resources/schema.sql', root / 'tools/java_backend_bridge.py']
    for path in files:
        result['source_files'][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'tables': len(result['tables']), 'source_files': len(files)}))


if __name__ == '__main__':
    main()

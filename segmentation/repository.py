"""Dedicated MySQL segmentation tables; never writes change-detection tables."""
import json
import shlex
import time
from pathlib import Path

import pymysql

from .contract import canonical_json, digest_json, validate_manifest


DDL = (
    '''CREATE TABLE IF NOT EXISTS segmentation_models (
       model_id CHAR(64) PRIMARY KEY, name VARCHAR(200) NOT NULL,
       identity_json JSON NOT NULL, training_json JSON NOT NULL, created_ms BIGINT NOT NULL
       ) ENGINE=InnoDB''',
    '''CREATE TABLE IF NOT EXISTS segmentation_batches (
       batch_id CHAR(36) PRIMARY KEY, model_id CHAR(64) NOT NULL,
       label VARCHAR(200) NOT NULL, task_type VARCHAR(32) NOT NULL,
       sample_count INT NOT NULL, manifest_digest CHAR(64) NOT NULL,
       created_ms BIGINT NOT NULL, imported_ms BIGINT NOT NULL,
       FOREIGN KEY(model_id) REFERENCES segmentation_models(model_id),
       KEY segmentation_created(created_ms)
       ) ENGINE=InnoDB''',
    '''CREATE TABLE IF NOT EXISTS segmentation_samples (
       sample_id CHAR(36) PRIMARY KEY, batch_id CHAR(36) NOT NULL,
       name VARCHAR(200) NOT NULL, region_id VARCHAR(64) NOT NULL,
       width INT NOT NULL, height INT NOT NULL,
       FOREIGN KEY(batch_id) REFERENCES segmentation_batches(batch_id),
       UNIQUE KEY segmentation_batch_name(batch_id,name)
       ) ENGINE=InnoDB''',
    '''CREATE TABLE IF NOT EXISTS segmentation_predictions (
       sample_id CHAR(36) PRIMARY KEY, foreground_pixels BIGINT NOT NULL,
       foreground_ratio DOUBLE NOT NULL, component_count INT NOT NULL,
       inference_ms DOUBLE NOT NULL,
       FOREIGN KEY(sample_id) REFERENCES segmentation_samples(sample_id)
       ) ENGINE=InnoDB''',
    '''CREATE TABLE IF NOT EXISTS segmentation_assets (
       asset_id CHAR(36) PRIMARY KEY, sample_id CHAR(36) NOT NULL,
       role VARCHAR(24) NOT NULL, path TEXT NOT NULL, sha256 CHAR(64) NOT NULL,
       byte_size BIGINT NOT NULL,
       FOREIGN KEY(sample_id) REFERENCES segmentation_samples(sample_id),
       UNIQUE KEY segmentation_sample_role(sample_id,role), KEY segmentation_hash(sha256)
       ) ENGINE=InnoDB''',
    '''CREATE TABLE IF NOT EXISTS segmentation_evaluations (
       batch_id CHAR(36) PRIMARY KEY, scope VARCHAR(32) NOT NULL,
       image_count INT NOT NULL, metrics_json JSON NOT NULL,
       counts_json JSON NOT NULL, provenance_json JSON NOT NULL,
       FOREIGN KEY(batch_id) REFERENCES segmentation_batches(batch_id)
       ) ENGINE=InnoDB''',
)


def connect(database='xunbian_backend'):
    if database not in ('xunbian_backend', 'xunbian_backend_test'):
        raise ValueError('Unknown application database')
    config = Path.home() / '.config/xunbian-backend/backend.env'
    values = {}
    for line in config.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, value = line.removeprefix('export ').split('=', 1)
        values[key] = shlex.split(value)[0]
    return pymysql.connect(unix_socket='/var/run/mysqld/mysqld.sock',
                           user=values['XB_DB_USER'], password=values['XB_DB_PASSWORD'],
                           database=database, charset='utf8mb4',
                           cursorclass=pymysql.cursors.DictCursor,
                           connect_timeout=5, read_timeout=30, write_timeout=30)


def migrate(db):
    checksum = digest_json(DDL)
    with db.cursor() as cur:
        cur.execute("SELECT GET_LOCK('xunbian-segmentation-schema-v1',10) AS acquired")
        if cur.fetchone()['acquired'] != 1:
            raise RuntimeError('Segmentation schema migration is already running')
        try:
            cur.execute('''CREATE TABLE IF NOT EXISTS segmentation_schema_migrations (
                version INT PRIMARY KEY, checksum CHAR(64) NOT NULL, applied_ms BIGINT NOT NULL
                ) ENGINE=InnoDB''')
            cur.execute('SELECT checksum FROM segmentation_schema_migrations WHERE version=1')
            row = cur.fetchone()
            if row:
                if row['checksum'] != checksum:
                    raise RuntimeError('Applied segmentation migration checksum differs')
            else:
                for statement in DDL:
                    cur.execute(statement)
                cur.execute('INSERT INTO segmentation_schema_migrations(version,checksum,applied_ms) VALUES(1,%s,%s)',
                            (checksum, int(time.time() * 1000)))
            db.commit()
        finally:
            cur.execute("SELECT RELEASE_LOCK('xunbian-segmentation-schema-v1')")


def ingest(db, manifest):
    digest = validate_manifest(manifest)
    batch = manifest['batch_id']; model = manifest['model']; now = int(time.time() * 1000)
    db.begin()
    try:
        with db.cursor() as cur:
            cur.execute('SELECT manifest_digest FROM segmentation_batches WHERE batch_id=%s FOR UPDATE', (batch,))
            prior = cur.fetchone()
            if prior:
                if prior['manifest_digest'] != digest:
                    raise ValueError('Existing segmentation batch cannot be overwritten')
                db.commit()
                return dict(batch_id=batch, inserted=False, samples=len(manifest['samples']), manifest_digest=digest)
            cur.execute('SELECT identity_json FROM segmentation_models WHERE model_id=%s', (model['model_id'],))
            prior_model = cur.fetchone()
            if prior_model:
                if digest_json(json.loads(prior_model['identity_json'])) != model['model_id']:
                    raise ValueError('Stored model identity is inconsistent')
            else:
                cur.execute('INSERT INTO segmentation_models(model_id,name,identity_json,training_json,created_ms) VALUES(%s,%s,%s,%s,%s)',
                            (model['model_id'], model['name'], canonical_json(model['identity']), canonical_json(model.get('training', {})), now))
            cur.execute('''INSERT INTO segmentation_batches(batch_id,model_id,label,task_type,sample_count,manifest_digest,created_ms,imported_ms)
                        VALUES(%s,%s,%s,%s,%s,%s,%s,%s)''',
                        (batch, model['model_id'], manifest['label'], manifest['task_type'], len(manifest['samples']), digest, manifest['created_ms'], now))
            for sample in manifest['samples']:
                sid = sample['sample_id']
                cur.execute('INSERT INTO segmentation_samples(sample_id,batch_id,name,region_id,width,height) VALUES(%s,%s,%s,%s,%s,%s)',
                            (sid, batch, sample['name'], sample['region_id'], sample['width'], sample['height']))
                cur.execute('INSERT INTO segmentation_predictions(sample_id,foreground_pixels,foreground_ratio,component_count,inference_ms) VALUES(%s,%s,%s,%s,%s)',
                            (sid, sample['foreground_pixels'], sample['foreground_ratio'], sample['component_count'], sample['inference_ms']))
                for asset in sample['assets']:
                    cur.execute('INSERT INTO segmentation_assets(asset_id,sample_id,role,path,sha256,byte_size) VALUES(%s,%s,%s,%s,%s,%s)',
                                (asset['asset_id'], sid, asset['role'], asset['path'], asset['sha256'], asset['byte_size']))
            if manifest.get('evaluation'):
                evaluation = manifest['evaluation']
                provenance = {k: v for k, v in evaluation.items() if k not in ('metrics', 'counts')}
                cur.execute('INSERT INTO segmentation_evaluations(batch_id,scope,image_count,metrics_json,counts_json,provenance_json) VALUES(%s,%s,%s,%s,%s,%s)',
                            (batch, evaluation['scope'], evaluation['images'], canonical_json(evaluation['metrics']), canonical_json(evaluation['counts']), canonical_json(provenance)))
        db.commit()
    except Exception:
        db.rollback()
        raise
    return dict(batch_id=batch, inserted=True, samples=len(manifest['samples']), manifest_digest=digest)


def list_batches(db):
    with db.cursor() as cur:
        cur.execute('''SELECT b.*,m.name AS model_name FROM segmentation_batches b
                    JOIN segmentation_models m ON m.model_id=b.model_id ORDER BY b.created_ms DESC LIMIT 100''')
        return cur.fetchall()


def batch_detail(db, batch):
    with db.cursor() as cur:
        cur.execute('''SELECT b.*,m.name AS model_name FROM segmentation_batches b
                    JOIN segmentation_models m ON m.model_id=b.model_id WHERE b.batch_id=%s''', (batch,))
        result = cur.fetchone()
        if not result:
            return None
        cur.execute('''SELECT s.*,p.foreground_pixels,p.foreground_ratio,p.component_count,p.inference_ms
                    FROM segmentation_samples s JOIN segmentation_predictions p ON p.sample_id=s.sample_id
                    WHERE s.batch_id=%s ORDER BY s.name''', (batch,))
        result['samples'] = cur.fetchall()
        cur.execute('''SELECT a.asset_id,a.sample_id,a.role FROM segmentation_assets a
                    JOIN segmentation_samples s ON s.sample_id=a.sample_id WHERE s.batch_id=%s''', (batch,))
        assets = {}
        for asset in cur.fetchall():
            assets.setdefault(asset['sample_id'], {})[asset['role']] = asset['asset_id']
        for sample in result['samples']:
            sample['assets'] = assets.get(sample['sample_id'], {})
        cur.execute('SELECT * FROM segmentation_evaluations WHERE batch_id=%s', (batch,))
        evaluation = cur.fetchone()
        result['evaluation'] = None if evaluation is None else {
            'scope': evaluation['scope'], 'images': evaluation['image_count'],
            'metrics': json.loads(evaluation['metrics_json']),
            'counts': json.loads(evaluation['counts_json']),
            'provenance': json.loads(evaluation['provenance_json'])}
        return result

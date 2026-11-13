"""Explicit read-only SQL demo over public app_data.json facts, never GT labels."""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from pathlib import Path


class StaticFacts:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self._lock = threading.RLock()
        self._signature = None
        self._conn = None

    def _refresh(self):
        info = self.path.stat()
        signature = (info.st_mtime_ns, info.st_size)
        if signature == self._signature:
            return
        data = json.loads(self.path.read_text(encoding='utf-8-sig'))
        conn = sqlite3.connect(':memory:', check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.create_function('JSON_UNQUOTE', 1, lambda x: x)
        conn.create_function('FIELD', -1, lambda value, *values: values.index(value) + 1 if value in values else 0)
        conn.executescript('''
            CREATE TABLE regions(region_id TEXT PRIMARY KEY, short_name TEXT, region_name TEXT,
                total_rows INTEGER, changed_rows INTEGER, review_rows INTEGER);
            CREATE TABLE samples(sample_id TEXT PRIMARY KEY, region_id TEXT, sample_stem TEXT,
                sample_name TEXT, event_key TEXT, event_name TEXT, changed_px INTEGER,
                changed_ratio REAL, object_count INTEGER, grid_position TEXT, review_status TEXT,
                review_required INTEGER, review_reason TEXT, confidence REAL,
                public_facts_json TEXT, quality_flags_json TEXT, main_bbox_json TEXT, report_text TEXT);
            CREATE INDEX sample_region ON samples(region_id);
        ''')
        for region in data.get('regions', []):
            items = region.get('items', [])
            conn.execute('INSERT INTO regions VALUES(?,?,?,?,?,?)', (
                region['id'], region.get('short', region['id']), region.get('name', ''), len(items),
                sum(bool(x.get('changedPx', 0)) for x in items), sum(bool(x.get('reviewRequired')) for x in items)))
            for item in items:
                level = (item.get('maskQuality') or {}).get('review_level') or item.get('reviewLevel') or item.get('reviewStatus') or 'none'
                level = {'高风险复核': 'high', '中风险复核': 'medium', '常规样本': 'low'}.get(level, level)
                # Whitelist public columns: no privateMetrics, GT paths, or raw blobs.
                facts = {'review_level': level}
                conn.execute('INSERT INTO samples VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', (
                    str(item.get('id') or item.get('stem')), region['id'], item.get('stem', ''),
                    item.get('name', ''), item.get('primaryEventType') or item.get('eventType', ''),
                    item.get('primaryEventName') or item.get('eventTypeZh', ''), item.get('changedPx', 0),
                    item.get('changedRatio', 0), item.get('objectCount', 0), item.get('gridPosition', ''),
                    level, int(bool(item.get('reviewRequired'))), item.get('reviewReason', ''), item.get('confidence', 0),
                    json.dumps(facts, ensure_ascii=False), json.dumps(item.get('qualityFlags', []), ensure_ascii=False),
                    json.dumps(item.get('mainBbox', [])), item.get('evidenceSummary', '')))
        conn.commit()
        conn.execute('PRAGMA query_only=ON')

        def authorize(action, arg1, arg2, _db, _trigger):
            allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
            if action not in allowed:
                return sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_READ and arg1 not in {'samples', 'regions'}:
                return sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_FUNCTION and str(arg2 or '').lower() in {'load_extension', 'readfile', 'writefile'}:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        conn.set_authorizer(authorize)
        if self._conn is not None:
            self._conn.close()
        self._conn, self._signature = conn, signature

    def execute(self, query: str, params=(), max_rows: int = 36) -> dict:
        text = str(query or '').strip().rstrip(';')
        if not re.match(r'^(SELECT|WITH)\b', text, re.I):
            raise ValueError('Static facts allow read-only SELECT queries only')
        with self._lock:
            self._refresh()
            deadline = time.monotonic() + 2.0
            self._conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
            try:
                cur = self._conn.execute(text.replace('%s', '?'), tuple(params))
                rows = cur.fetchmany(max_rows + 1)
                return {'sql': text, 'columns': [x[0] for x in cur.description or []],
                        'rows': [dict(x) for x in rows[:max_rows]], 'row_count': min(len(rows), max_rows),
                        'truncated': len(rows) > max_rows, 'dataSource': 'static_snapshot'}
            except sqlite3.Error as exc:
                raise ValueError(f'Static SQL query failed: {exc}') from exc
            finally:
                self._conn.set_progress_handler(None, 0)


_STORES = {}
_LOCK = threading.Lock()


def static_store(path: str | Path) -> StaticFacts:
    key = str(Path(path).resolve())
    with _LOCK:
        if key not in _STORES:
            if len(_STORES) >= 4:
                _STORES.pop(next(iter(_STORES)))
            _STORES[key] = StaticFacts(key)
        return _STORES[key]

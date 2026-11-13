"""Create isolated backend databases using the project's existing local credentials."""
import ast
import json
import os
from pathlib import Path
import secrets
import shlex
import pymysql

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path.home() / '.config/xunbian-backend/backend.env'

def main():
    CONFIG.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    existing = {}
    if CONFIG.exists():
        for line in CONFIG.read_text().splitlines():
            key, value = line.removeprefix('export ').split('=', 1)
            existing[key] = shlex.split(value)[0]
    tree = ast.parse((ROOT / 'tools/import_app_data_to_mysql.py').read_text(encoding='utf-8'))
    declaration = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument'
                       and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == '--password')
    default = next(k.value for k in declaration.keywords if k.arg == 'default')
    password = os.getenv('TRUST_AGENT_MYSQL_PASSWORD', default.args[1].value)
    user = existing.get('XB_DB_USER', 'xb_' + secrets.token_hex(5))
    app_password = existing.get('XB_DB_PASSWORD', secrets.token_urlsafe(32))
    internal_token = existing.get('XB_INTERNAL_TOKEN', secrets.token_urlsafe(40))
    conn = pymysql.connect(unix_socket='/var/run/mysqld/mysqld.sock', user='root', password=password, autocommit=True)
    with conn, conn.cursor() as cursor:
        for database in ('xunbian_backend', 'xunbian_backend_test'):
            cursor.execute(f'CREATE DATABASE IF NOT EXISTS `{database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci')
        if not existing:
            cursor.execute('CREATE USER %s@%s IDENTIFIED BY %s', (user, 'localhost', app_password))
        for database in ('xunbian_backend', 'xunbian_backend_test'):
            cursor.execute(f'GRANT SELECT,INSERT,UPDATE,DELETE,CREATE,ALTER,INDEX,REFERENCES ON `{database}`.* TO %s@%s', (user, 'localhost'))
    jdbc_suffix = '?localSocket=/var/run/mysqld/mysqld.sock'
    values = {'XB_DB_USER': user, 'XB_DB_PASSWORD': app_password, 'XB_INTERNAL_TOKEN': internal_token,
              'XB_DB_URL': 'jdbc:mariadb://localhost/xunbian_backend' + jdbc_suffix,
              'XB_TEST_DB_URL': 'jdbc:mariadb://localhost/xunbian_backend_test' + jdbc_suffix,
              'XB_STORAGE_ROOT': str(Path.home() / '.local/share/xunbian-backend/storage'),
              'XB_STATE_ROOT': str(Path.home() / '.local/share/xunbian-backend/state')}
    temporary = CONFIG.with_suffix('.tmp')
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, 'w') as target:
        target.write(''.join(f'export {k}={shlex.quote(v)}\n' for k, v in values.items()))
    temporary.replace(CONFIG)
    print(json.dumps({'config': str(CONFIG), 'databases': ['xunbian_backend','xunbian_backend_test'], 'transport': 'unix_socket', 'status': 'created'}))

if __name__ == '__main__':
    main()

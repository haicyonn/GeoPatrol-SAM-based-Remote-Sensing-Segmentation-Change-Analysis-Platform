"""Local administrator recovery. Prints a private file path, never the new password."""
import argparse
import json
import os
from pathlib import Path
import secrets
import shlex
import bcrypt
import pymysql

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--username',default='admin')
    parser.add_argument('--database',choices=('xunbian_backend','xunbian_backend_test'),default='xunbian_backend')
    parser.add_argument('--credential-file',type=Path,default=Path.home()/'.config/xunbian-backend/admin-recovery.json')
    args=parser.parse_args();config={}
    for line in (Path.home()/'.config/xunbian-backend/backend.env').read_text().splitlines():
        key,value=line.removeprefix('export ').split('=',1);config[key]=shlex.split(value)[0]
    password=secrets.token_urlsafe(24);hashed=bcrypt.hashpw(password.encode(),bcrypt.gensalt(rounds=12)).decode()
    args.credential_file.parent.mkdir(parents=True,exist_ok=True)
    conn=pymysql.connect(unix_socket='/var/run/mysqld/mysqld.sock',user=config['XB_DB_USER'],password=config['XB_DB_PASSWORD'],database=args.database)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM app_users WHERE username=%s AND role='ADMIN' FOR UPDATE",(args.username,))
            row=cur.fetchone()
            if not row:raise SystemExit('没有找到该管理员账号；未修改任何密码。')
            temporary=args.credential_file.with_suffix('.tmp')
            fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
            with os.fdopen(fd,'w') as target:json.dump({'username':args.username,'password':password,'notice':'本地管理员恢复后生成的密码，请登录后修改。'},target,ensure_ascii=False)
            os.chmod(temporary,0o600);temporary.replace(args.credential_file)
            cur.execute('UPDATE app_users SET password_hash=%s,auth_version=auth_version+1,enabled=TRUE WHERE id=%s',(hashed,row[0]))
        conn.commit()
    except BaseException:conn.rollback();raise
    finally:conn.close()
    print(json.dumps({'status':'reset','username':args.username,'privateCredentialFile':str(args.credential_file)},ensure_ascii=False))

if __name__=='__main__':main()

"""Authentication for local verification scripts; credentials never enter reports."""
import json
import os
from pathlib import Path
import requests

BASE='http://127.0.0.1:8786/api/v1'
SESSION=requests.Session()
_ready=False

def login():
    global _ready
    file=Path(os.environ.get('XB_ADMIN_CREDENTIAL_FILE',str(Path.home()/'.config/xunbian-backend/initial-admin.json')))
    credentials=json.loads(file.read_text());token=SESSION.get(BASE+'/auth/csrf',timeout=10).json()
    response=SESSION.post(BASE+'/auth/login',data={'username':credentials['username'],'password':credentials['password']},headers={token['headerName']:token['token']},timeout=20)
    if response.status_code!=200:raise RuntimeError('验证账号登录失败；若已修改管理员密码，请通过XB_ADMIN_CREDENTIAL_FILE指定私有凭据文件。')
    token=SESSION.get(BASE+'/auth/csrf',timeout=10).json();SESSION.headers[token['headerName']]=token['token'];_ready=True

def response(method,path,**kwargs):
    if not _ready:login()
    positions=[]
    files=kwargs.get('files') or []
    for _,value in (files.items() if isinstance(files,dict) else files):
        stream=value[1] if isinstance(value,tuple) else value
        if hasattr(stream,'tell') and hasattr(stream,'seek'):positions.append((stream,stream.tell()))
    result=SESSION.request(method,BASE+path,**kwargs)
    if result.status_code in (401,403):
        # A restarted server invalidates the session and CSRF. No retry on
        # model/transport timeout or any uncertain execution outcome.
        login()
        for stream,position in positions:stream.seek(position)
        result=SESSION.request(method,BASE+path,**kwargs)
    return result

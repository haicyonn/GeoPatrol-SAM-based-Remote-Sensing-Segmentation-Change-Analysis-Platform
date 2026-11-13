"""A denied multipart request may be replayed after login without losing its file."""
import importlib.util
import io
import json
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

def test_session_reauthentication_rewinds_multipart_files(tmp_path,monkeypatch):
    path=Path(__file__).resolve().parents[1]/'backend-java/scripts/authenticated_http.py'
    spec=importlib.util.spec_from_file_location('test_auth_http_helper',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    credentials=tmp_path/'private.json';credentials.write_text(json.dumps({'username':'fixture','password':'fixture-secret'}));monkeypatch.setenv('XB_ADMIN_CREDENTIAL_FILE',str(credentials))
    attempts=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,status,value):
            body=json.dumps(value).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):self.send(200,{'headerName':'X-CSRF-TOKEN','token':'fixture-token'})
        def do_POST(self):
            body=self.rfile.read(int(self.headers['Content-Length']))
            if self.path.endswith('/auth/login'):self.send(200,{'status':'ok'});return
            attempts.append(body)
            if len(attempts)==1:self.send(403,{'error':'expired session csrf'});return
            self.send(200,{'validFile':b'original-image-payload' in body})
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    module.BASE='http://127.0.0.1:'+str(server.server_port)+'/api/v1'
    try:
        result=module.response('POST','/batches',files=[('before',('same.png',io.BytesIO(b'original-image-payload'),'image/png'))],timeout=5)
        assert result.json()=={'validFile':True}
        assert len(attempts)==2 and b'original-image-payload' in attempts[0]
    finally:module.SESSION.close();server.shutdown();server.server_close();thread.join()

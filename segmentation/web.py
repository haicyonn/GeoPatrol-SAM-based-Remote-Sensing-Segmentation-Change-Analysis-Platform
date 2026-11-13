"""Compatibility redirect; all business operations use the Java console."""
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def create_redirect_server(port=8788):
    """Compatibility URL only. All active UI/data operations belong to Java."""
    class Redirect(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            if self.path=='/health':
                data=json.dumps(dict(application='xunbian-building-segmentation',redirectOnly=True,pid=os.getpid())).encode()
                self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data);return
            self.send_response(302);self.send_header('Location','http://127.0.0.1:8786/index.html?batch=4270335f-a3c7-46e4-baa9-bb679d4339fa');self.send_header('Cache-Control','no-store');self.send_header('Content-Length','0');self.end_headers()
        def do_POST(self):
            self.send_response(410);self.send_header('Content-Length','0');self.end_headers()
    return ThreadingHTTPServer(('127.0.0.1',port),Redirect)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--port', type=int, default=8788)
    parser.add_argument('--database', choices=['xunbian_backend', 'xunbian_backend_test'], default='xunbian_backend')
    args = parser.parse_args()
    server = create_redirect_server(args.port)
    print('Segmentation moved to the shared Java console: http://127.0.0.1:8786/', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()

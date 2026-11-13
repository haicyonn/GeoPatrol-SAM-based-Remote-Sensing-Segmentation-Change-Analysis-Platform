import threading
import unittest
import urllib.error
import urllib.request

from segmentation import web


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args):return None


class RedirectTests(unittest.TestCase):
    def test_old_entry_only_redirects_to_shared_java_console(self):
        self.assertTrue(hasattr(web,'create_redirect_server'))
        server=web.create_redirect_server(0)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
        try:
            with self.assertRaises(urllib.error.HTTPError) as error:
                opener.open(f'http://127.0.0.1:{server.server_port}/')
            self.assertEqual(error.exception.code,302)
            self.assertTrue(error.exception.headers['Location'].startswith('http://127.0.0.1:8786/'))
            with self.assertRaises(urllib.error.HTTPError) as error:
                opener.open(urllib.request.Request(f'http://127.0.0.1:{server.server_port}/api/batches/x/questions',data=b'{}'))
            self.assertEqual(error.exception.code,410)
        finally:server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()

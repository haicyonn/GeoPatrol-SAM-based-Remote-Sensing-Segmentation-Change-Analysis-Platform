const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../web/auth.js'), 'utf8');

async function boot({disabled = true, hostname = '127.0.0.1', page = 'index.html', protocol = 'http:', search = '', hash = ''} = {}) {
  const redirects = [], classes = new Set();
  let fetches = 0;
  const window = {__TRUST_AUTH_DISABLED__: disabled,
    location: {protocol, hostname, pathname: '/' + page,
      href: protocol === 'file:' ? `file:///E:/trust_agent_mainline_clean/web/${page}${search}${hash}` : `http://${hostname}:8766/${page}${search}${hash}`, search, hash, replace: u => redirects.push(u)},
    fetch: async () => { fetches++; return {ok: true, json: async () => ({loggedIn: false})}; }};
  const document = {documentElement: {classList: {add: v => classes.add(v), remove: v => classes.delete(v)}},
    head: {appendChild() {}}, createElement: () => ({}), addEventListener() {}};
  vm.runInNewContext(source, {window, document, localStorage: {getItem: () => null,
    setItem() {}, removeItem() {}}, URL, URLSearchParams, Headers, setTimeout, console});
  await new Promise(resolve => setImmediate(resolve));
  return {window, redirects, classes, fetches};
}

(async () => {
  const demo = await boot();
  assert.equal(demo.redirects.length, 0, 'local demo must not redirect to login');
  assert.equal(demo.window.__TRUST_AUTH_USER__?.username, 'local-demo');
  assert.equal(demo.fetches, 0, 'no token or credential exchange is needed');
  assert.equal(demo.classes.has('auth-pending'), false);
  const oldLogin = await boot({page: 'login.html'});
  assert.equal(new URL(oldLogin.redirects[0]).pathname, '/index.html');
  const protectedSite = await boot({disabled: false});
  assert.equal(new URL(protectedSite.redirects[0]).pathname, '/login.html');
  const remote = await boot({hostname: 'example.com'});
  assert.equal(new URL(remote.redirects[0]).pathname, '/login.html');
  const disk = await boot({protocol: 'file:', disabled: false, hostname: '', page: 'data_analysis.html', search: '?view=chat', hash: '#result'});
  assert.deepEqual(disk.redirects, ['http://127.0.0.1:8766/data_analysis.html?view=chat#result']);
  assert.equal(disk.fetches, 0, 'a file entry must not contact the old auth API');
  assert.equal(disk.classes.has('auth-pending'), false);
  const diskLogin = await boot({protocol: 'file:', disabled: false, hostname: '', page: 'login.html'});
  assert.deepEqual(diskLogin.redirects, ['http://127.0.0.1:8766/index.html']);
  console.log('Disk entries use the local HTTP service; local demo needs no login; normal and remote authentication remain active.');
})().catch(error => {console.error(error);process.exitCode = 1;});

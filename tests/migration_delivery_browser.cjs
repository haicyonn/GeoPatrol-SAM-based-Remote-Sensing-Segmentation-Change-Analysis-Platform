const {chromium}=require('C:/Users/halcy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const credentials=JSON.parse(fs.readFileSync(String.raw`\\wsl.localhost\Ubuntu-22.04\home\halcyon\.config\xunbian-backend\initial-admin.json`,'utf8'));
const out=path.resolve(__dirname,'../evaluation/full_migration_20261102');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'}),p=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
 p.on('pageerror',e=>errors.push(e.message));
 try{
  await p.goto('http://127.0.0.1:8786/');await p.waitForURL('**/manage.html?returnTo=*');await p.locator('#username').fill(credentials.username);await p.locator('#password').fill(credentials.password);await p.locator('#loginButton').click();await p.waitForFunction(()=>!!window.ConsoleWorkspace?.controller.context);
  assert.equal(await p.locator('#consoleBatch option:checked').innerText().then(v=>v.includes('5984组')),true);
  assert.equal(await p.locator('a[href="/"]').count(),2);
  await p.locator('[data-region-id="levir"]').click();await p.locator('.tab[data-tab="review"]').click();
  await p.waitForFunction(()=>!!document.querySelector('#manualEditor [data-stage]')&&!document.querySelector('#approveGeometry').disabled);
  const stage=p.locator('#manualEditor [data-stage]');await stage.scrollIntoViewIfNeeded();await p.locator('#manualEditor [data-image]').evaluate(async i=>{if(!i.complete)await new Promise(r=>i.onload=r);});
  const b=await stage.boundingBox();for(const [x,y] of [[.1,.1],[.35,.1],[.35,.4]])await p.mouse.click(b.x+x*b.width,b.y+y*b.height);
  await p.locator('#consoleReviewNote').fill('未提交的绘制与备注，重载时必须保留');
  await p.locator('#sortMode').selectOption('review');
  assert.equal((await p.locator('#manualEditor [data-svg] polyline').getAttribute('points')).split(' ').length,3);
  assert.equal(await p.locator('#consoleReviewNote').inputValue(),'未提交的绘制与备注，重载时必须保留');
  await p.locator('#consoleReloadReview').click();await p.waitForFunction(()=>document.querySelector('#manualEditor [data-svg] polyline'));
  assert.equal((await p.locator('#manualEditor [data-svg] polyline').getAttribute('points')).split(' ').length,3);
  await p.locator('#manualEditor [data-stage]').press('Enter');await p.locator('#manualEditor').scrollIntoViewIfNeeded();await p.screenshot({path:path.join(out,'restored-original-editor.png')});
  await p.locator('#discardLocalReview').click();await p.waitForFunction(()=>!!document.querySelector('#manualEditor [data-stage]'));
  // Only local unsaved geometry was exercised; no review PUT was made.
  await p.locator('#consoleLogout').click();await p.waitForURL('**/manage.html?returnTo=*');
  assert.deepEqual(errors,[]);const result={passed:true,defaultCorpus5984:true,visibleCorpusLink:true,unfinishedVerticesPreserved:true,notePreserved:true,reloadPreservesInputs:true,noReviewWrite:true,pageErrors:errors};fs.writeFileSync(path.join(out,'delivery-browser.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
 }catch(e){await p.screenshot({path:path.join(out,'delivery-failure.png')}).catch(()=>{});throw e;}
 finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});

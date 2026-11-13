const {chromium}=require('C:/Users/halcy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const privateRoot=String.raw`\\wsl.localhost\Ubuntu-22.04\home\halcyon\.config\xunbian-backend`;
const people=JSON.parse(fs.readFileSync(path.join(privateRoot,'console-merge-qa.json'),'utf8'));
const out=path.resolve(__dirname,'../evaluation/console_merge_20261102');
const origin='http://127.0.0.1:8786';
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',args:['--host-resolver-rules=MAP xunbian.test 127.0.0.1','--no-proxy-server']});
 const errors=[],oldRequests=[],checks=[];const pages=[];
 const record=checks.push.bind(checks);checks.push=(...values)=>{for(const value of values)console.log(value);return record(...values);};
 function track(page){pages.push(page);page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(/127\.0\.0\.1:(8766|8790|8791)/.test(r.url()))oldRequests.push(r.url());});return page;}
 async function open(credentials,route,base=origin){
  const page=track(await browser.newPage({viewport:{width:1440,height:1024},acceptDownloads:true}));
  await page.goto(base+route);await page.waitForURL('**/manage.html?returnTo=*');
  await page.locator('#username').fill(credentials.username);await page.locator('#password').fill(credentials.password);await page.locator('#loginButton').click();
  await page.waitForFunction(()=>!!window.ConsoleWorkspace?.controller.context,undefined,{timeout:30000});return page;
 }
 async function ask(page,text){await page.locator('#consoleQuestion').fill(text);await page.locator('#consoleSend').click();await page.waitForFunction(()=>!document.querySelector('#consoleSend').disabled,undefined,{timeout:180000});return page.locator('#consoleChatLog article.assistant').last();}
 async function api(page,method,route,body){return page.evaluate(async({method,route,body})=>{const t=await(await fetch('/api/v1/auth/csrf')).json();const r=await fetch('/api/v1'+route,{method,headers:{'Content-Type':'application/json',[t.headerName]:t.token},body:body===undefined?undefined:JSON.stringify(body)});return {status:r.status,data:await r.json()};},{method,route,body});}
 try{
  const owner=await open(people.owner,'/index.html?batch='+people.batchId+'&tab=agent');
  assert.equal(new URL(owner.url()).searchParams.get('batch'),people.batchId);
  const greeting=await ask(owner,'你好');assert.match(await greeting.innerText(),/未执行 SQL/);assert.doesNotMatch(await greeting.innerText(),/请选择.*区域|low_risk_samples/);checks.push('login returns original console and greeting remains chat');
  const row=await ask(owner,'本批各区域的样本数是多少？返回region_id和sample_count，按region_id升序。');assert.equal(await row.locator('tbody tr').count(),3);checks.push('real UI query includes all three batch regions');
  await owner.locator('.tab[data-tab="interpret"]').click();assert.equal(await owner.locator('#workPane img').count(),3);await owner.waitForFunction(()=>[...document.querySelectorAll('#workPane img')].every(i=>i.complete&&i.naturalWidth>0));
  assert.ok((await owner.locator('#workPane img').evaluateAll(imgs=>imgs.map(i=>i.src))).every(u=>u.includes('/api/v1/assets/')));checks.push('authorized T1 T2 and actual prediction mask load');
  await owner.locator('.tab[data-tab="review"]').click();await owner.waitForFunction(()=>document.querySelector('#consoleReviewState')?.textContent.includes('记录版本'));
  const editor=await open(people.editor,'/index.html?batch='+people.batchId+'&tab=review');await editor.waitForFunction(()=>document.querySelector('#consoleReviewState')?.textContent.includes('记录版本'));
  await editor.locator('#consoleVerdict').selectOption('confirmed_change');await editor.locator('#consoleReviewNote').fill('双时相影像浏览器验收');await editor.locator('#consoleSaveReview').click();await editor.waitForFunction(()=>document.querySelector('#consoleStatus').textContent.includes('复核已保存'),undefined,{timeout:20000});
  await owner.locator('#consoleReviewNote').fill('版本冲突时应保留的输入');await owner.locator('#consoleSaveReview').click();await owner.waitForFunction(()=>document.querySelector('#consoleReviewState')?.textContent.includes('输入已保留'));
  assert.equal(await owner.locator('#consoleReviewNote').inputValue(),'版本冲突时应保留的输入');await owner.locator('#consoleReloadReview').click();await owner.waitForFunction(()=>document.querySelector('#consoleReviewNote')?.value==='双时相影像浏览器验收');checks.push('concurrent review conflict retains input and loads newer version');
  await owner.locator('#consoleReload').click();await owner.waitForFunction(()=>!!window.ConsoleWorkspace.controller.context&&document.querySelector('#consoleReviewState')?.textContent.includes('记录版本'));
  await owner.locator('.tab[data-tab="report"]').click();await owner.waitForFunction(()=>document.querySelector('.report-summary-box')?.textContent.includes('当前范围共1张样本'));
  const download=owner.waitForEvent('download');await owner.locator('#downloadWordReportBtn').click();const file=await download;assert.match(file.suggestedFilename(),/\.doc$/);await file.saveAs(path.join(out,'browser-region-report.doc'));const doc=fs.readFileSync(path.join(out,'browser-region-report.doc'),'utf8');assert.match(doc,/复核版本/);assert.match(doc,/当前范围共1张样本/);assert.ok(!doc.trimStart().startsWith('{'));checks.push('original Word exporter downloads current scoped server statistics and version');
  await owner.locator('#reportPane').evaluate(el=>el.scrollTop=0);await owner.screenshot({path:path.join(out,'console-desktop.png'),fullPage:true});await owner.setViewportSize({width:390,height:844});
  await owner.waitForFunction(()=>document.querySelector('.workspace-overlay .window').getBoundingClientRect().right<=innerWidth+1);
  await owner.waitForFunction(()=>document.querySelector('.workspace-overlay .window').getBoundingClientRect().top>=document.querySelector('.console-header').getBoundingClientRect().bottom);
  assert.ok((await owner.locator('.report-main').boundingBox()).width<=390);await owner.locator('#reportPane').evaluate(el=>el.scrollTop=0);await owner.screenshot({path:path.join(out,'console-mobile.png'),fullPage:true});assert.ok(await owner.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));await owner.setViewportSize({width:1440,height:1024});checks.push('desktop and390px mobile layout with unclipped workspace');
  const analysis=await open(people.owner,'/analysis.html?batch='+people.batchId);await analysis.locator('#analysisGoal').fill('统计本批全部样本数量，只需一步查询，返回sample_count。');await analysis.locator('#analysisForm button').click();await analysis.waitForFunction(()=>!document.querySelector('#analysisForm button').disabled,undefined,{timeout:180000});assert.equal(await analysis.locator('[data-action="execute"]').count(),1);
  await analysis.locator('[data-action="execute"]').click();await analysis.waitForFunction(()=>!document.querySelector('#analysisForm button').disabled,undefined,{timeout:180000});assert.match(await analysis.locator('#analysisResult').innerText(),/分析简报/);assert.equal(await analysis.locator('#analysisResult .table-wrap tbody tr').count(),1);assert.match(await analysis.locator('#analysisResult').innerText(),/3/);await analysis.screenshot({path:path.join(out,'analysis-desktop.png'),fullPage:true});checks.push('real model analysis plan executes with table and evidence-linked brief');
  // Hold a genuine analysis response, switch context, then let the stale reply return.
  let release,arrived,settled;const intercepted=new Promise(r=>arrived=r),handled=new Promise(r=>settled=r);
  await analysis.route('**/api/v1/batches/*/analysis',async route=>{const response=await route.fetch({timeout:180000});const hold=new Promise(r=>release=r);arrived();await hold;await route.fulfill({response});settled();});
  await analysis.locator('#analysisGoal').fill('统计全部样本数量，只需一步。');await analysis.locator('#analysisForm button').click();await intercepted;await analysis.locator('#consoleReload').click();await analysis.waitForFunction(()=>!!window.ConsoleWorkspace.controller.context&&!document.querySelector('#analysisForm button').disabled);release();await handled;await analysis.unroute('**/api/v1/batches/*/analysis');await analysis.waitForFunction(()=>document.querySelector('#analysisResult').textContent.includes('先描述分析目标'));checks.push('switch during pending model request restores controls and ignores old reply');
  const admin=await open(people.admin,'/index.html?batch='+people.archiveId);assert.equal(await admin.evaluate(()=>window.ConsoleWorkspace.controller.context.app.regions.reduce((n,r)=>n+r.items.length,0)),5984);assert.equal(await admin.evaluate(()=>window.ConsoleWorkspace.controller.context.readOnly),true);await admin.locator('#openWorkspaceBtn').click();await admin.screenshot({path:path.join(out,'historical-console.png'),fullPage:true});checks.push('5984 historical samples remain explicit readonly admin reference');
  const denied=await api(owner,'GET','/batches/'+people.archiveId+'/console');assert.equal(denied.status,403);
  const lan=await open(people.owner,'/index.html?batch='+people.batchId+'&tab=agent','http://xunbian.test:8786');assert.equal(await lan.evaluate(()=>isSecureContext),false);await lan.locator('#consoleNewChat').click();assert.equal(await lan.locator('#consoleChatLog article').count(),0);checks.push('same-origin login and new session work on nonsecure named HTTP origin');
  await owner.locator('#consoleLogout').click();await owner.waitForURL('**/manage.html?returnTo=*');assert.equal(await owner.locator('#loginPanel').isVisible(),true);checks.push('logout leaves authorized console and returns login');
  assert.deepEqual(errors,[]);assert.deepEqual(oldRequests,[]);const result={passed:true,checks,pageErrors:errors,oldPortRequests:oldRequests};fs.writeFileSync(path.join(out,'browser-merge.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
 }catch(e){fs.writeFileSync(path.join(out,'browser-merge-failure.txt'),String(e));for(const [i,page] of pages.entries())await page.screenshot({path:path.join(out,'console-failure-'+i+'.png'),fullPage:true}).catch(()=>{});throw e;}
 finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});

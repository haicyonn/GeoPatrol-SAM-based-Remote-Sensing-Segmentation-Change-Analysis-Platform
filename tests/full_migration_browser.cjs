const {chromium}=require('C:/Users/halcy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const people=JSON.parse(fs.readFileSync(String.raw`\\wsl.localhost\Ubuntu-22.04\home\halcyon\.config\xunbian-backend\console-merge-qa.json`,'utf8'));
const out=path.resolve(__dirname,'../evaluation/full_migration_20261102'),origin='http://127.0.0.1:8786';
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'}),errors=[],checks=[],pages=[];
 async function open(credentials,route){
  const p=await browser.newPage({viewport:{width:1440,height:1100},acceptDownloads:true});pages.push(p);p.on('pageerror',e=>errors.push(e.message));
  await p.goto(origin+route);await p.waitForURL('**/manage.html?returnTo=*');
  await p.locator('#username').fill(credentials.username);await p.locator('#password').fill(credentials.password);await p.locator('#loginButton').click();
  await p.waitForFunction(()=>!!window.ConsoleWorkspace?.controller.context,undefined,{timeout:60000});return p;
 }
 async function api(p,method,url,body){return p.evaluate(async({method,url,body})=>{const t=await(await fetch('/api/v1/auth/csrf')).json();const r=await fetch('/api/v1'+url,{method,headers:{'Content-Type':'application/json',[t.headerName]:t.token},body:body===undefined?undefined:JSON.stringify(body)});if(!r.ok)throw Error('API '+r.status);return r.json();},{method,url,body});}
 async function reviewReady(p){await p.waitForFunction(()=>!!document.querySelector('#manualEditor [data-stage]')&&!document.querySelector('#approveGeometry').disabled,undefined,{timeout:30000});}
 async function save(p,button){const response=p.waitForResponse(r=>r.request().method()==='PUT'&&r.url().includes('/reviews/'),{timeout:45000});await p.locator(button).click();const r=await response;assert.equal(r.status(),200,await r.text());const value=await r.json();await p.waitForFunction(v=>document.querySelector('#consoleReviewState')?.textContent.includes('记录版本 '+v),value.version,{timeout:30000});await reviewReady(p);return value;}
 try{
  const admin=await open(people.admin,'/');
  const context=await admin.evaluate(()=>({source:ConsoleWorkspace.controller.context.sourceType,counts:ConsoleWorkspace.controller.context.app.regions.map(r=>[r.id,r.items.length])}));
  assert.equal(context.source,'ARCHIVE');assert.equal(context.counts.reduce((n,r)=>n+r[1],0),5984);assert.equal(await admin.locator('[data-region-id]').count(),5);checks.push('no batch URL defaults to full five-region corpus');
  await admin.locator('[data-region-id="levir"]').click();await admin.locator('.tab[data-tab="interpret"]').click();
  assert.equal(await admin.locator('.console-gallery-card').count(),24);const first=await admin.locator('.console-gallery-card h3').first().innerText();await admin.locator('#galleryNext').click();assert.notEqual(await admin.locator('.console-gallery-card h3').first().innerText(),first);checks.push('gallery pagination reaches additional samples');
  await admin.locator('#consoleQuality').selectOption('high');assert.ok(await admin.locator('.console-gallery-card').count()>0);checks.push('quality filter retains known source labels');
  await admin.screenshot({path:path.join(out,'full-corpus-gallery.png'),fullPage:true});
  const p=await open(people.owner,'/index.html?tab=review&batch='+people.batchId);await reviewReady(p);
  const start=await p.evaluate(()=>({id:ConsoleLegacy.state.selectedId,px:ConsoleWorkspace.controller.context.app.regions.flatMap(r=>r.items).find(i=>i.id===ConsoleLegacy.state.selectedId).changedPx}));
  await p.locator('#manualEditor [data-clear]').click();
  await p.locator('#manualEditor [data-image]').evaluate(async i=>{if(!i.complete)await new Promise(r=>i.onload=r);});
  const stage=p.locator('#manualEditor [data-stage]'),box=await stage.boundingBox();
  const size=await p.evaluate(()=>ConsoleWorkspace.controller.context.app.regions.flatMap(r=>r.items).find(i=>i.id===ConsoleLegacy.state.selectedId).sourceSize);
  for(const [x,y] of [[10,10],[30,10],[30,30],[10,30]])await p.mouse.click(box.x+x/size[0]*box.width,box.y+y/size[1]*box.height);
  await stage.press('Enter');assert.equal(await p.locator('#manualEditor [data-preview] polygon').count(),1);
  const draft=await save(p,'#saveGeometryDraft');assert.equal(draft.correction.status,'draft');assert.equal(draft.correction.changedPx,441);checks.push('scaled drawing and Enter close produce 441px persisted draft');
  await p.reload();await p.waitForFunction(()=>!!window.ConsoleWorkspace?.controller.context);await reviewReady(p);assert.equal(await p.locator('#manualEditor [data-preview] polygon').count(),1);checks.push('draft survives real page reload');
  await p.locator('#consoleVerdict').selectOption('confirmed_change');const approved=await save(p,'#approveGeometry');assert.equal(approved.approvedCorrection.changedPx,441);
  const effective=await api(p,'POST','/batches/'+people.batchId+'/report-summary',{sampleIds:[start.id],resultBasis:'effective'});assert.equal(effective.report.changedPixelsSum,441);
  await p.screenshot({path:path.join(out,'manual-correction-approved.png'),fullPage:true});
  await p.locator('#consoleResultBasis').selectOption('original');await p.waitForFunction(()=>ConsoleWorkspace.controller.context?.resultBasis==='original');
  await p.waitForFunction(()=>document.querySelector('#approveGeometry')?.disabled);const original=await api(p,'POST','/batches/'+people.batchId+'/report-summary',{sampleIds:[start.id],resultBasis:'original'});assert.equal(original.report.changedPixelsSum,start.px);checks.push('approved correction affects effective statistics while original remains readonly');
  await p.locator('#consoleResultBasis').selectOption('effective');await p.waitForFunction(()=>ConsoleWorkspace.controller.context?.resultBasis==='effective');await reviewReady(p);
  const jsonDownload=p.waitForEvent('download');await p.locator('#exportGeometry').click();const download=await jsonDownload;await download.saveAs(path.join(out,'corrections-export.json'));assert.ok(JSON.parse(fs.readFileSync(path.join(out,'corrections-export.json'))).records.some(r=>r.correction));checks.push('revision export contains geometry');
  await p.locator('.tab[data-tab="report"]').click();await p.waitForFunction(()=>document.querySelector('#downloadWordReportBtn')&&!document.querySelector('#downloadWordReportBtn').disabled,undefined,{timeout:30000});
  const wordDownload=p.waitForEvent('download');await p.locator('#downloadWordReportBtn').click();await(await wordDownload).saveAs(path.join(out,'corrected-report.doc'));assert.match(fs.readFileSync(path.join(out,'corrected-report.doc'),'utf8'),/人工修正|已审批修正/);checks.push('Word export records correction basis');
  await p.locator('.tab[data-tab="review"]').click();await reviewReady(p);await save(p,'#restorePrediction');
  assert.deepEqual(errors,[]);fs.writeFileSync(path.join(out,'browser.json'),JSON.stringify({passed:true,checks,pageErrors:errors},null,2));console.log(JSON.stringify({passed:true,checks}));
 }catch(e){for(const [i,p] of pages.entries())await p.screenshot({path:path.join(out,'failure-'+i+'.png'),fullPage:true}).catch(()=>{});throw e;}
 finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});

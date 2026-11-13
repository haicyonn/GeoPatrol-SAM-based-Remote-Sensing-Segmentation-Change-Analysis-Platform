const {chromium}=require(process.env.XB_PLAYWRIGHT||'C:/Users/halcy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const fixture=JSON.parse(fs.readFileSync(process.env.XB_GROUP_FIXTURE_FILE||String.raw`\\wsl.localhost\Ubuntu-22.04\home\halcyon\.config\xunbian-backend\group-smoke-accounts.json`,'utf8'));
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:process.env.XB_BROWSER||'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});const results=[];
 try{
  const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.addInitScript(()=>Object.defineProperty(Crypto.prototype,'randomUUID',{value:undefined}));
  await page.goto('http://127.0.0.1:8786/');await page.waitForTimeout(300);
  results.push({case:'LAN_without_randomUUID',passed:errors.length===0,errors});await page.close();
  const app=await browser.newPage();await app.goto('http://127.0.0.1:8786/');
  await app.locator('#username').fill(fixture.owner.username);await app.locator('#password').fill(fixture.owner.password);await app.locator('#loginButton').click();
  await app.locator('[data-batch-id="'+fixture.batchId+'"]').click();
  await app.waitForFunction(()=>document.querySelectorAll('#results button').length>=2);
  await app.locator('#results button').nth(0).click();await app.waitForFunction(()=>document.querySelector('#reviewMeta').textContent.includes('记录版本'));
  await app.locator('#reviewNote').fill('note that belongs to the first sample only');
  const state=await app.evaluate(()=>({id:selected,sample:reviewing.sampleId}));
  const batches=await app.evaluate(async id=>(await api('/batches/'+id+'/results?limit=200')).rows,fixture.batchId);
  const second=batches[1].sampleId;let release,entered;
  const held=new Promise(r=>release=r),started=new Promise(r=>entered=r);let first=true;const puts=[];
  await app.route('**/reviews/'+second,async route=>{
    if(route.request().method()==='PUT'){puts.push(route.request().postDataJSON());await route.fulfill({json:{version:1}});return;}
    if(first){first=false;entered();await held;}await route.continue();
  });
  await app.locator('#results button').nth(1).click();await started;
  const disabled=await app.locator('#saveReview').isDisabled();if(!disabled){await app.locator('#saveReview').click();await app.waitForTimeout(100);}
  release();await app.waitForTimeout(350);
  results.push({case:'sample_switch_blocks_stale_review_submit',passed:disabled&&puts.length===0,disabled,interceptedPutCount:puts.length});
  await app.locator('#logout').click();await app.waitForFunction(()=>!document.querySelector('#loginPanel').hidden);
  results.push({case:'logout_clears_previous_private_batch_content',passed:await app.locator('#results').evaluate(el=>el.childElementCount===0)});
  await app.close();
  const offline=await browser.newPage();await offline.route('**/api/v1/models',route=>route.fulfill({status:503,json:{error:'model service offline fixture'}}));
  await offline.goto('http://127.0.0.1:8786/');await offline.locator('#username').fill(fixture.owner.username);await offline.locator('#password').fill(fixture.owner.password);await offline.locator('#loginButton').click();await offline.waitForTimeout(700);
  results.push({case:'model_outage_keeps_saved_batches_accessible',passed:await offline.locator('[data-batch-id="'+fixture.batchId+'"]').count()===1});await offline.close();
 }finally{await browser.close();}
 const out=path.resolve(__dirname,'../evaluation/group_collaboration_20261101');fs.mkdirSync(out,{recursive:true});
 fs.writeFileSync(path.join(out,'ui-regression-latest.json'),JSON.stringify(results,null,2));console.log(JSON.stringify(results));assert.ok(results.every(r=>r.passed));
})().catch(e=>{console.error(e);process.exitCode=1;});

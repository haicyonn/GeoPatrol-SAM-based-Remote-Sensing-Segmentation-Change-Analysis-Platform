const {chromium}=require('C:/Users/halcy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const credentials=JSON.parse(fs.readFileSync(String.raw`\\wsl.localhost\Ubuntu-22.04\home\halcyon\.config\xunbian-backend\initial-admin.json`,'utf8'));
const out=path.resolve(__dirname,'../evaluation/report_numbers_20261102');
const original=JSON.parse(fs.readFileSync(path.join(out,'original-rejected.json'),'utf8'));
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'}),p=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];
 p.on('pageerror',e=>errors.push(e.message));
 const record={kind:'real model / browser regression of the reported request; not general accuracy'};
 try{
  await p.goto('http://127.0.0.1:8786/analysis.html?batch=f197b57a-c15b-3d92-bc14-1c0f6c40d67c&basis=effective');
  await p.waitForURL('**/manage.html?returnTo=*');await p.locator('#username').fill(credentials.username);await p.locator('#password').fill(credentials.password);await p.locator('#loginButton').click();
  await p.waitForFunction(()=>!!window.ConsoleWorkspace?.controller.context);
  await p.locator('#analysisScope').selectOption(original.requestedScopeRegion||'');await p.locator('#analysisGoal').fill(original.goal);
  const planning=p.waitForResponse(r=>r.url().endsWith('/analysis')&&r.request().postDataJSON()?.action==='plan',{timeout:180000});await p.locator('#analysisForm button[type="submit"]').click();const pr=await planning;
  record.plan=await pr.json();fs.writeFileSync(path.join(out,'browser.json'),JSON.stringify(record,null,2));assert.equal(pr.status(),200);assert.equal(record.plan.status,'planned');
  const executing=p.waitForResponse(r=>r.url().endsWith('/analysis')&&r.request().postDataJSON()?.action==='execute',{timeout:180000});await p.locator('[data-action="execute"]').click();const er=await executing;
  record.execution=await er.json();fs.writeFileSync(path.join(out,'browser.json'),JSON.stringify(record,null,2));assert.equal(er.status(),200);
  const r=record.execution;assert.equal(r.status,'completed');assert.equal(r.report.status,'generated');assert.ok(r.report.attempts.length<=2);
  const text=r.report.findings.map(f=>f.text).join('\n');
  for(const value of ['743','2529.54','23.60','13.84','6.77'])assert.ok(text.includes(value),'missing validated statistic '+value);
  // The brief may summarize extremes; the complete source statistics must remain in the result table.
  const stats=r.steps.flatMap(s=>s.rows).find(row=>row.sample_count===743);
  assert.ok(stats);assert.equal(stats.min_changed_px,0);assert.equal(stats.max_changed_px,65534);
  for(const region of ['uav','clcd'])assert.ok(text.toLowerCase().includes(region));assert.deepEqual(errors,[]);
  await p.waitForFunction(()=>document.querySelector('#analysisResult').textContent.includes('分析简报'));
  assert.equal(await p.locator('#analysisError').innerText(),'');
  await p.screenshot({path:path.join(out,'fixed-report.png'),fullPage:true});record.passed=true;record.pageErrors=errors;
  fs.writeFileSync(path.join(out,'browser.json'),JSON.stringify(record,null,2));console.log(JSON.stringify({passed:true,findings:r.report.findings,reportAttempts:r.report.attempts.length}));
 }finally{try{await p.locator('#consoleLogout').click({timeout:3000});}catch{}await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});

const {chromium}=require('C:/Users/halcy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const credentials=JSON.parse(fs.readFileSync(String.raw`\\wsl.localhost\Ubuntu-22.04\home\halcyon\.config\xunbian-backend\initial-admin.json`,'utf8'));
const out=path.resolve(__dirname,'../evaluation/full_migration_20261102');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'}),p=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];
 p.on('pageerror',e=>errors.push(e.message));
 try{
  await p.goto('http://127.0.0.1:8786/analysis.html?batch=f197b57a-c15b-3d92-bc14-1c0f6c40d67c&basis=effective');await p.waitForURL('**/manage.html?returnTo=*');
  await p.locator('#username').fill(credentials.username);await p.locator('#password').fill(credentials.password);await p.locator('#loginButton').click();await p.waitForFunction(()=>!!window.ConsoleWorkspace?.controller.context);
  await p.locator('#analysisGoal').fill('统计各区域样本数量，返回region_id和sample_count，按sample_count降序排列。');
  const planning=p.waitForResponse(r=>r.url().endsWith('/analysis')&&r.request().postDataJSON()?.action==='plan',{timeout:180000});await p.locator('#analysisForm button[type="submit"]').click();const pr=await planning;assert.equal(pr.status(),200);const plan=await pr.json();assert.equal(plan.status,'planned');
  const executing=p.waitForResponse(r=>r.url().endsWith('/analysis')&&r.request().postDataJSON()?.action==='execute',{timeout:180000});await p.locator('[data-action="execute"]').click();const er=await executing;assert.equal(er.status(),200);const result=await er.json();assert.equal(result.status,'completed');
  await p.waitForFunction(()=>document.querySelectorAll('#analysisResult tbody tr').length===5);assert.equal(await p.locator('#analysisError').innerText(),'');assert.equal(result.report.status,'generated');assert.deepEqual(errors,[]);
  await p.screenshot({path:path.join(out,'full-corpus-analysis-fixed.png'),fullPage:true});
  fs.writeFileSync(path.join(out,'io-browser.json'),JSON.stringify({passed:true,sourceSamples:5984,rows:5,planStatus:plan.status,executionStatus:result.status,reportStatus:result.report.status,pageErrors:errors},null,2));
  console.log(JSON.stringify({passed:true,rows:5,reportStatus:result.report.status}));await p.locator('#consoleLogout').click();
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});

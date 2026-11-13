// Live, authenticated checks of the merged console, using existing private QA identities.
const {chromium}=require('C:/Users/halcy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const people=JSON.parse(fs.readFileSync(String.raw`\\wsl.localhost\Ubuntu-22.04\home\halcyon\.config\xunbian-backend\console-merge-qa.json`,'utf8'));
const out=path.resolve(__dirname,'../evaluation/clarification_flow_20261102');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[],responses=[];
 page.on('pageerror',e=>errors.push(e.message));
 page.on('response',async r=>{if(r.request().method()==='POST'&&r.url().endsWith('/questions'))responses.push(await r.json());});
 async function idle(){await page.waitForFunction(()=>document.querySelector('#consoleSend')&&!document.querySelector('#consoleSend').disabled,undefined,{timeout:180000});}
 async function perform(action,predicate){
  const response=page.waitForResponse(r=>r.request().method()==='POST'&&r.url().endsWith('/questions')&&predicate(r.request().postDataJSON()),{timeout:180000});
  await action();const data=await(await response).json();await idle();return data;
 }
 async function ask(text){await page.locator('#consoleQuestion').fill(text);return perform(()=>page.locator('#consoleSend').click(),p=>p.question===text);}
 try{
  await page.goto('http://127.0.0.1:8786/index.html?tab=agent&batch='+people.batchId);
  await page.waitForURL('**/manage.html?returnTo=*');
  await page.locator('#username').fill(people.owner.username);await page.locator('#password').fill(people.owner.password);await page.locator('#loginButton').click();
  await page.waitForFunction(()=>!!window.ConsoleWorkspace?.controller.context);await idle();
  await ask('LEVIR区域空洞比例是多少？');
  assert.equal(await page.locator('[data-choice-id]:enabled').count(),2);
  await page.screenshot({path:path.join(out,'choices.png'),fullPage:true});
  assert.equal((await ask('你好')).usedSQL,false);
  assert.ok(await page.locator('[data-choice-id]:enabled').count()>=2);
  await perform(()=>page.locator('#consoleHistory').click(),p=>p.action==='history');
  assert.equal(await page.locator('[data-choice-id]:enabled').count(),2);
  const area=await perform(()=>page.locator('[data-choice-id="option_2"]:enabled').last().click(),p=>p.optionId==='option_2');
  assert.equal(area.status,'completed');assert.equal(area.usedSQL,true);
  assert.match(area.sqlQuery,/AVG\(hole_ratio\)/i);
  assert.equal(await page.locator('[data-choice-id]:enabled').count(),0);
  await page.screenshot({path:path.join(out,'chosen-result.png'),fullPage:true});
  await page.locator('#consoleNewChat').click();assert.equal(await page.locator('#consoleChatLog article').count(),0);
  await ask('WHU空洞比例是多少？');assert.equal(await page.locator('[data-choice-id]:enabled').count(),2);
  const flag=await perform(()=>page.locator('[data-choice-id="option_1"]:enabled').click(),p=>p.optionId==='option_1');
  assert.equal(flag.status,'completed');assert.match(flag.sqlQuery,/has_holes/i);
  assert.deepEqual(errors,[]);
  const report={passed:true,checks:['two choices visible','chat preserves choices','history restores choices','area mean executes','old buttons disabled','new session resets','flag proportion executes'],pageErrors:errors,responses};
  fs.writeFileSync(path.join(out,'browser.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({passed:true,checks:report.checks}));
 }catch(error){await page.screenshot({path:path.join(out,'browser-failure.png'),fullPage:true}).catch(()=>{});throw error;}
 finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});

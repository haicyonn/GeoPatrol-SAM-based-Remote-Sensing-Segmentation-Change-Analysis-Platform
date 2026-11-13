const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {chromium}=require('C:/Users/halcy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const people=JSON.parse(fs.readFileSync(String.raw`\\wsl.localhost\Ubuntu-22.04\home\halcyon\.config\xunbian-backend\console-merge-qa.json`,'utf8'));
const out=path.resolve(__dirname,'../evaluation/query_intent_20261102');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[],payloads=[],replies=[],checks=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>{if(r.url().endsWith('/questions')&&r.method()==='POST')payloads.push(r.postDataJSON());});
 try{
  await page.goto('http://127.0.0.1:8786/index.html?tab=agent&batch='+people.intentBatchId);await page.locator('#username').fill(people.owner.username);await page.locator('#password').fill(people.owner.password);await page.locator('#loginButton').click();await page.waitForFunction(()=>!!window.ConsoleWorkspace?.controller.context);
  async function choose(rid){await page.locator('#regionScope').click();await page.locator('[data-switch-id="'+rid+'"]').click();}
  async function ask(q){const response=page.waitForResponse(r=>r.url().endsWith('/questions')&&r.request().method()==='POST');await page.locator('#consoleQuestion').fill(q);await page.locator('#consoleSend').click();const value=await(await response).json();replies.push(value);await page.waitForFunction(()=>!document.querySelector('#consoleSend').disabled,undefined,{timeout:180000});return value;}
  const q='你好吗，这个区域有几个图片';await choose('whu');let r=await ask(q);assert.equal(r.status,'completed');assert.equal(r.questionContext.viewRegion,'whu');assert.ok(Object.values(r.sqlRows[0]).includes(2));assert.equal(payloads.at(-1).question,q);assert.equal(payloads.at(-1).viewRegion,'whu');assert.equal(payloads.at(-1).scopeRegion,null);assert.match(await page.locator('#consoleChatLog article.assistant').last().innerText(),/本轮区域指代：WHU/);checks.push('original mixed query sends separate visible region and returnsWHU2');
  await choose('levir');r=await ask(q);assert.equal(r.questionContext.viewRegion,'levir');assert.ok(Object.values(r.sqlRows[0]).includes(1));checks.push('same chat follows visibleLEVIR switch and returns1');
  r=await ask('各区域有几张图片？返回region_id和sample_count，按region_id升序。');assert.deepEqual(r.sqlRows,[{region_id:'levir',sample_count:1},{region_id:'whu',sample_count:2}]);checks.push('visibleLEVIR never limits all-region query');
  r=await ask('这个区域的图片数量怎么算？只解释，不查询数据库。');assert.equal(r.status,'chat_completed');assert.equal(r.usedSQL,false);checks.push('conceptual explanation with same words0SQL');
  await page.goto('http://127.0.0.1:8786/index.html?tab=agent&batch='+people.batchId);await page.waitForFunction(()=>!!window.ConsoleWorkspace?.controller.context);await choose('whu');
  const visibleCount=await page.evaluate(()=>window.ConsoleWorkspace.controller.context.app.regions.find(r=>r.id==='whu').items.length);
  r=await ask(q);assert.equal(r.status,'completed');assert.equal(r.questionContext.viewRegion,'whu');assert.ok(Object.values(r.sqlRows[0]).includes(visibleCount));checks.push('original user console batch answers exact reported question consistently with displayed samples');
  await page.screenshot({path:path.join(out,'mixed-query-desktop.png'),fullPage:true});await page.setViewportSize({width:390,height:844});
  await page.waitForFunction(()=>document.querySelector('.workspace-overlay .window').getBoundingClientRect().top>=document.querySelector('.console-header').getBoundingClientRect().bottom);
  await page.locator('.workspace-overlay .window').evaluate(el=>el.scrollTop=0);await page.screenshot({path:path.join(out,'mixed-query-mobile.png'),fullPage:true});assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(out,'browser.json'),JSON.stringify({passed:true,checks,pageErrors:errors,payloads,replies},null,2));console.log(JSON.stringify({passed:true,checks,pageErrors:errors}));
 }catch(e){await page.screenshot({path:path.join(out,'browser-failure.png'),fullPage:true}).catch(()=>{});throw e;}finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});

// Real local browser/upload/model smoke; dependency path supplied by test runner.
const {chromium}=require(process.env.XB_PLAYWRIGHT||'playwright');
const fs=require('node:fs');const path=require('node:path');const assert=require('node:assert/strict');
(async()=>{
 const root=path.resolve(__dirname,'..'),out=path.join(root,'evaluation/java_backend_20261101');
 fs.mkdirSync(out,{recursive:true});
 const browser=await chromium.launch({headless:true,executablePath:process.env.XB_BROWSER});
 const page=await browser.newPage({viewport:{width:1440,height:1050}});const errors=[];
 page.on('pageerror',error=>errors.push(String(error)));
 try{
  await page.goto('http://127.0.0.1:8786/',{waitUntil:'networkidle'});
  const credentials=JSON.parse(fs.readFileSync(process.env.XB_ADMIN_CREDENTIAL_FILE||String.raw`\\wsl.localhost\Ubuntu-22.04\home\halcyon\.config\xunbian-backend\initial-admin.json`,'utf8'));
  await page.locator('#username').fill(credentials.username);await page.locator('#password').fill(credentials.password);await page.locator('#loginButton').click();
  await page.waitForFunction(()=>document.querySelector('#health').textContent==='服务已连接');
  await page.locator('#label').fill('浏览器真实上传验证');
  await page.locator('#model').selectOption('secor-whu');
  await page.locator('#before').setInputFiles(path.join(root,'web/assets/whu/pre/whucd_01342.jpg'));
  await page.locator('#after').setInputFiles(path.join(root,'web/assets/whu/post/whucd_01342.jpg'));
  await page.locator('#submitUpload').click();
  await page.waitForFunction(()=>document.querySelector('#batchState').textContent.includes('可分析'),undefined,{timeout:180000});
  await page.locator('#results button').first().click();
  await page.waitForFunction(()=>document.querySelectorAll('#images img').length===3&&[...document.querySelectorAll('#images img')].every(im=>im.complete&&im.naturalWidth>0));
  await page.locator('#question').fill('共有多少样本？只返回sample_count。');await page.locator('#send').click();
  await page.waitForFunction(()=>document.querySelector('#send').disabled===false,undefined,{timeout:180000});
  assert.equal(await page.locator('#chat table tbody td').last().textContent(),'1');
  await page.locator('#question').fill('哪个区域空洞比例最高？');await page.locator('#send').click();
  await page.waitForFunction(()=>document.querySelector('#send').disabled===false,undefined,{timeout:180000});
  assert.equal(await page.locator('#chat .chat-item').last().locator('button').count(),2);
  assert.ok(!(await page.locator('#chat .chat-item').last().innerText()).includes('没有符合条件的记录'),'pending clarification is not an empty SQL result');
  await page.screenshot({path:path.join(out,'ui-desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:path.join(out,'ui-mobile.png'),fullPage:true});
  assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
  assert.deepEqual(errors,[]);
  // UI-only transport fixture: an older session's completion must not unlock
  // a newer in-flight request. This is separate from real-model checks above.
  const replies=[];
  await page.route('**/api/v1/batches/*/questions',async route=>{
   await new Promise(resolve=>replies.push(async()=>{await route.fulfill({json:{status:'completed',answer:'UI状态测试',sqlRows:[{n:1}]}});resolve();}));
  });
  await page.locator('#newSession').click();
  await page.locator('#question').fill('UI状态测试一');await page.locator('#send').click();
  await page.waitForFunction(()=>document.querySelector('#send').disabled);
  while(replies.length<1)await new Promise(r=>setTimeout(r,10));
  await page.locator('#newSession').click();
  await page.locator('#question').fill('UI状态测试二');await page.locator('#send').click();
  while(replies.length<2)await new Promise(r=>setTimeout(r,10));
  const oldResponse=page.waitForResponse(r=>r.url().includes('/questions'));
  await replies[0]();await(await oldResponse).finished();
  await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
  assert.equal(await page.locator('#send').isDisabled(),true,'old response must not unlock new session');
  await replies[1]();await page.waitForFunction(()=>!document.querySelector('#send').disabled);
  await page.waitForFunction(()=>document.querySelector('#batches .active')?.textContent.includes('可分析'),undefined,{timeout:5000});
  const result={passed:true,realUpload:true,maskImages:3,realModelCountQuery:1,clarificationOptions:2,sessionRaceIsolation:true,pageErrors:errors};
  fs.writeFileSync(path.join(out,'browser-smoke.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
 }catch(error){
  fs.writeFileSync(path.join(out,'browser-smoke-failure.txt'),await page.locator('body').innerText());
  await page.screenshot({path:path.join(out,'ui-failure.png'),fullPage:true});throw error;
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});

// Local product smoke test; requires Playwright and a compatible installed Chromium.
const {chromium}=require(process.env.CHANGE_CASE_PLAYWRIGHT||'playwright');
const path=require('node:path');const fs=require('node:fs');const assert=require('node:assert/strict');
(async()=>{
  const browser=await chromium.launch({headless:true,executablePath:process.env.CHANGE_CASE_CHROMIUM});
  const page=await browser.newPage({viewport:{width:1400,height:1000}});const errors=[];
  page.on('pageerror',err=>errors.push(String(err)));
  try{
    await page.goto(process.env.CHANGE_CASE_URL||'http://127.0.0.1:8783/',{waitUntil:'networkidle'});
    await page.waitForFunction(()=>document.querySelector('#sample').options.length>100,{timeout:60000});
    await page.locator('#filter').fill('val_000001');
    await page.waitForFunction(()=>document.querySelector('#sample').options.length===1);
    assert.equal(await page.locator('#selected').textContent(),'val:val_000001.png');
    await page.locator('#retrieve').click();
    await page.waitForFunction(()=>document.querySelectorAll('.ref').length===3);
    await page.waitForFunction(()=>[...document.images].every(im=>im.complete&&im.naturalWidth>0));
    assert.equal(await page.locator('.ref').count(),3);
    const output=path.resolve(__dirname,'../outputs/change_cases_20261101');fs.mkdirSync(output,{recursive:true});
    await page.screenshot({path:path.join(output,'ui-desktop.png'),fullPage:true});
    if(process.argv.includes('--answer')||process.argv.includes('--caption')){
      await page.locator('#mode').selectOption('retrieved');await page.locator('#topk').selectOption('3');
      await page.locator('#question').fill('请比较前后影像，说明主要可见变化在哪里。不确定的数量不要猜测。');
      await page.locator(process.argv.includes('--caption')?'#caption':'#answer').click();
      await page.waitForFunction(()=>!document.querySelector('#answer').disabled,{timeout:120000});
      assert.ok((await page.locator('#output').textContent()).length>5);
      assert.equal(await page.locator('#output .error').count(),0);
      assert.equal(await page.locator('#trace details').count(),1);
      console.log(JSON.stringify({answer:await page.locator('#output').textContent(),progress:await page.locator('#progress').textContent()}));
      await page.screenshot({path:path.join(output,'ui-answer.png'),fullPage:true});
    }
    await page.setViewportSize({width:390,height:844});
    await page.screenshot({path:path.join(output,'ui-mobile.png'),fullPage:true});
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
    assert.deepEqual(errors,[]);console.log('Browser smoke passed: filtering, paired images, train references, mobile layout'+(process.argv.includes('--answer')?', real visual answer':''));
  }finally{await browser.close()}
})().catch(error=>{console.error(error);process.exitCode=1});

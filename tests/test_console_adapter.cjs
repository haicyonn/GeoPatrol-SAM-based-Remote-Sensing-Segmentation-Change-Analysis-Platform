const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const file = path.join(__dirname, '../backend-java/src/main/resources/static/console-adapter.js');

test('console adapter exists for actual selected-batch behavior', () => {
  assert.ok(fs.existsSync(file), 'Console adapter has not been implemented');
});

if (fs.existsSync(file)) {
  const adapter = require(file);
  const fixture = (id='a', version=0, sourceType='INFERENCE') => ({
    batchId:id,label:'Batch '+id,sourceType,permission:'OWNER',readOnly:sourceType==='ARCHIVE',
    dataVersion:id+'-v'+version,reviewVersion:version,predictionDigest:'digest-'+id,
    models:[{modelId:'secor-whu',modelVersion:'abc'}],capabilities:{reviewVerdicts:true},
    app:{classes:{},classOrder:[],regions:[{id:'whu',short:'WHU',items:[{id:'task-'+id,name:'same',changedPx:6,changedRatio:.25,objectCount:2,assets:{pre:'/api/v1/assets/a',post:'/api/v1/assets/b',predMask:'/api/v1/assets/c'}}]}]}
  });
  test('untyped prediction facts stay unknown and controlled mask URL survives', () => {
    const context=adapter.normalizeConsole(fixture());
    const item=context.app.regions[0].items[0];
    assert.equal(adapter.facts(item,context).event,'未知');
    assert.equal(adapter.facts(item,context).confidence,'未知');
    assert.equal(adapter.facts(item,context).dimensions,'未知');
    assert.equal(item.assets.mask,'/api/v1/assets/c');
    assert.equal(item.stem,'same');
    assert.equal(item.regionId,'whu');
    assert.equal(adapter.statistics(context.app.regions[0].items).changedPixels,6);
  });
  test('archive is readonly even if capability or permission claims allow edits', () => {
    const context=adapter.normalizeConsole(fixture('archive',0,'ARCHIVE'));
    assert.equal(adapter.canReview(context),false);
  });
  test('readonly membership never relabels an inference batch as historical archive', () => {
    const context=fixture();context.readOnly=true;context.permission='VIEW';
    assert.match(adapter.sourceLabel(context),/实际模型预测/);
    assert.match(adapter.sourceLabel(context),/只读权限/);
    assert.doesNotMatch(adapter.sourceLabel(context),/历史/);
  });
  test('private or uncontrolled asset URLs do not become image sources', () => {
    const raw=fixture();raw.app.regions[0].items[0].assets.pre='http://127.0.0.1:8790/private';
    assert.throws(()=>adapter.normalizeConsole(raw),/资产/);
  });
  test('a slower batch load cannot replace a newer selected batch', async () => {
    let finishFirst;
    const controller=adapter.createController({request:async route=>route.includes('/a/')?new Promise(r=>finishFirst=r):fixture('b')});
    const first=controller.select('a');await controller.select('b');finishFirst(fixture('a'));await first;
    assert.equal(controller.context.batchId,'b');
  });
  test('report posts only selected IDs and rejects mismatched server review version', async () => {
    const requests=[];
    const controller=adapter.createController({request:async(route,options)=>{
      requests.push([route,options]);
      if(route.endsWith('/console'))return fixture();
      return {status:'done',text:'old',dataVersion:'a-v0',reviewVersion:1,report:{}};
    }});
    await controller.select('a');
    await assert.rejects(controller.summary('whu',['task-a']),/版本/);
    const payload=JSON.parse(requests[1][1].body);
    assert.deepEqual(payload,{regionId:'whu',sampleIds:['task-a'],reviewVersion:0,resultBasis:'effective'});
    assert.equal(controller.report,null);
  });
  test('a report arriving after switching account context is ignored', async () => {
    let finish;
    const controller=adapter.createController({request:async route=>route.endsWith('/console')?fixture():new Promise(r=>finish=r)});
    await controller.select('a');const pending=controller.summary('whu',['task-a']);controller.clear();
    finish({status:'done',text:'private',dataVersion:'a-v0',reviewVersion:0});
    assert.equal(await pending,null);assert.equal(controller.report,null);
  });
  test('normal chat is labeled without SQL and preserves pending clarification', () => {
    const html=adapter.renderReply({status:'chat',questionRoute:'chat',usedSQL:false,answer:'你好 <b>',clarification:{id:'c1',options:[{id:'a',label:'面积'}]}});
    assert.match(html,/未执行 SQL/);assert.match(html,/你好 &lt;b&gt;/);assert.doesNotMatch(html,/查询未完成/);
    assert.match(html,/data-choice-id="a"/);
  });
  test('Word text uses server summary and versioned statistics without inventing local semantics', () => {
    const text=adapter.reportText({status:'done',text:'服务端统计：1张样本，变化6像素。',dataVersion:'a-v0',reviewVersion:0,report:{sampleCount:1,changedPixelsSum:6}},fixture(),'全部样本');
    assert.match(text,/服务端统计：1张样本，变化6像素/);
    assert.match(text,/a-v0/);assert.match(text,/复核版本：0/);
    assert.doesNotMatch(text,/小区域建筑|低风险|高可信/);
  });
  test('default full-batch chat leaves scope null even with WHU as visual selection', async () => {
    let sent;
    const controller=adapter.createController({request:async(route,options)=>{
      if(route.endsWith('/console'))return fixture();sent=JSON.parse(options.body);return {status:'chat_completed',questionRoute:'chat',usedSQL:false,answer:'你好'};
    }});
    await controller.select('a');const result=await controller.question('各区域对比');
    assert.equal(sent.scopeRegion,null);assert.equal(sent.mode,'auto');assert.equal(result.usedSQL,false);
  });
  test('visible region is a separate request field and never a hidden SQL scope',async()=>{
    let sent;const controller=adapter.createController({request:async(route,options)=>{if(route.endsWith('/console'))return fixture();sent=JSON.parse(options.body);return {status:'chat_completed',usedSQL:false,answer:'ok'};}});
    await controller.select('a');await controller.question('这个区域有几个图片',null,{viewRegion:'whu'});
    assert.equal(sent.viewRegion,'whu');assert.equal(sent.scopeRegion,null);assert.equal(sent.question,'这个区域有几个图片');
  });
  test('region binding is visible and safely escaped in the assistant reply',()=>{
    const html=adapter.renderReply({answer:'已查询',usedSQL:true,questionContext:{viewRegion:'levir',regionLabel:'LEVIR<script>',originalQuestion:'这个区域有几个图片',resolvedQuestion:'LEVIR样本数量',countUnit:'sample_pair'}});
    assert.match(html,/LEVIR&lt;script&gt;/);assert.match(html,/样本组/);assert.doesNotMatch(html,/<script>/);
  });
  test('a question answered on a newer review snapshot cannot mix with old displayed facts', async () => {
    const controller=adapter.createController({request:async route=>route.endsWith('/console')?fixture():{status:'completed',questionRoute:'sql',usedSQL:true,dataVersion:'a-v1',reviewVersion:1,contextReset:true,answer:'new private values'}});
    await controller.select('a');await assert.rejects(controller.question('统计'),/版本/);
    assert.equal(controller.context.reviewVersion,0);
  });
  test('same batch reload rotates session and invalidates an old analysis response', async () => {
    let finish;let loads=0;
    const controller=adapter.createController({request:async route=>route.endsWith('/console')?fixture('a',loads++):new Promise(r=>finish=r)});
    await controller.select('a');const oldSession=controller.sessionId;
    const pending=controller.analysis({action:'plan',goal:'统计'});await controller.select('a');
    finish({status:'planned',planId:'old',reviewVersion:0});assert.equal(await pending,null);
    assert.notEqual(controller.sessionId,oldSession);
  });
  test('review requests use server version and enforce readonly before sending', async () => {
    const calls=[];const controller=adapter.createController({request:async(route,options)=>{calls.push([route,options]);return fixture('archive',0,'ARCHIVE');}});
    await controller.select('archive');await assert.rejects(controller.saveReview('task-archive',0,'confirmed_change','x'),/只读/);
    assert.equal(calls.length,1);
  });
  test('collaborative save sends the loaded sample version and explicit verdict', async () => {
    let write;
    const controller=adapter.createController({request:async(route,options)=>{
      if(route.endsWith('/console'))return fixture();write={route,options};return {version:5,verdict:'uncertain',note:'保留'};
    }});
    await controller.select('a');const saved=await controller.saveReview('task-a',4,'uncertain','保留');
    assert.equal(write.route,'/batches/a/reviews/task-a');assert.equal(write.options.method,'PUT');
    assert.deepEqual(JSON.parse(write.options.body),{expectedVersion:4,verdict:'uncertain',note:'保留'});
    assert.equal(saved.version,5);
  });
}

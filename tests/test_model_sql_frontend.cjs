const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const app = require('../web/model_sql.js');

assert.equal(app.request('多少条','levir','s').scopeRegion,'levir');
assert.equal(app.request('多少条','','s').sessionKey,'s');
const result={status:'completed',modelId:'test-7B',sqlQuery:'SELECT "<script>" FROM samples',
  sqlColumns:['<img>'],sqlRows:[{'<img>':'<script>alert(1)</script>'}],llmCalls:1,
  attempts:[{status:'executed',sql:'SELECT 1'}],sourceVersion:'abc',latencyMs:25};
const html=app.render(result);
assert.ok(!html.includes('<script>'));
assert.ok(html.includes('&lt;script&gt;'));
assert.ok(html.includes('test-7B'));
assert.ok(html.includes('SELECT'));
assert.ok(app.render({...result,status:'failed',sqlRows:[],error:'wrong <column>'}).includes('wrong &lt;column&gt;'));
assert.ok(app.render({...result,status:'needs_clarification',answer:'请明确区域',sqlRows:[]}).includes('请明确区域'));
const interpreted=app.render({...result,queryContract:{question:'<script>bad</script>',regions:['whu'],columns:['sample_id'],order:[{expression:'changed_px',direction:'DESC'}],limit:3,offset:1},resolvedReference:{position:2,sampleId:'B<bad>'}});
assert.ok(interpreted.includes('本次查询要求'));
assert.ok(interpreted.includes('changed_px'));
assert.ok(interpreted.includes('B&lt;bad&gt;'));
assert.ok(!interpreted.includes('<script>'));
const grounded=app.render({...result,population:{inherited:true,kind:'filtered_population',origin_question:'LEVIR空洞与碎片',filters:{field:'region_id',value:'<unsafe>'}},
  knowledge:{status:'extractive',answer:'核对前后时相 <script>',facts:[{sample_count:6}]},
  citations:[{id:1,title:'质量手册',section:'空洞检查',source:'quality.md',version:'v1',excerpt:'边界 <img>'}]});
assert.ok(grounded.includes('当前样本集合'));
assert.ok(grounded.includes('检查依据'));
assert.ok(grounded.includes('质量手册'));
assert.ok(!grounded.includes('<script>'));
const knowledge=app.render({...result,status:'knowledge_completed',sqlQuery:'',sqlRows:[],knowledge:{status:'extractive',answer:'检查来源'},citations:[]});
assert.ok(knowledge.includes('已检索检查依据'));
assert.ok(!knowledge.includes('没有符合条件的记录'));
const info=app.render({status:'system_info',answer:'当前配置模型是 Qwen/test',modelId:'Qwen/test',llmCalls:0,dataSource:'runtime_configuration'});
assert.ok(info.includes('系统运行信息'));
assert.ok(info.includes('运行配置'));
assert.ok(!info.includes('来源：公开样本快照'));
const generated=app.render({...result,modelGeneratedSQL:'SELECT AVG(changed_px) FROM samples',
  programChanges:['bound_population'],modelUsage:[{stage:'sql',inputTokens:100,outputTokens:20}],
  attempts:[{stage:'sql',status:'executed',modelGeneratedSQL:'SELECT AVG(changed_px) FROM samples',sql:'SELECT AVG(changed_px) FROM scoped_samples'}]});
assert.ok(generated.includes('模型生成的原始 SQL'));
assert.ok(generated.includes('实际执行的 SQL'));
assert.ok(generated.includes('100'));
assert.ok(generated.includes('20'));
const scoped=app.render({...result,scopePlan:{selection:{mode:'reselect',region:'levir'},reference:{scope:'region',region:'levir',aggregate:'avg',metric:'changed_px'}},selectionSampleIds:['L<1>','L2'],programChanges:['scoped_populations'],semanticChecks:['independent_population_scopes']});
assert.ok(scoped.includes('对象与比较基准'));
assert.ok(scoped.includes('LEVIR全部样本'));
assert.ok(scoped.includes('按原选样方法重新选择'));
assert.ok(scoped.includes('L&lt;1&gt;'));
assert.ok(!scoped.includes('L<1>'));
const dialogue=app.render({...result,contextResolution:{previousQuestion:'高风险数量<script>',resolvedQuestion:'总样本最多区域的高风险数量',retainedMeasures:['高风险样本数量<img>'],programBoundMeasures:true},modelUsage:[{stage:'context',inputTokens:90,outputTokens:30}]});
assert.ok(dialogue.includes('本轮如何沿用上下文'));
assert.ok(dialogue.includes('高风险数量&lt;script&gt;'));
assert.ok(dialogue.includes('高风险样本数量&lt;img&gt;'));
assert.ok(dialogue.includes('上下文补全'));
assert.ok(!dialogue.includes('<script>'));
const source=fs.readFileSync(require('node:path').join(__dirname,'../web/data_analysis.html'),'utf8');
assert.ok(source.includes('model_sql.js'));
assert.ok(source.includes('scopeRegion'));
new vm.Script(fs.readFileSync(require('node:path').join(__dirname,'../web/model_sql.js'),'utf8'));
console.log('Model SQL request, honest status, SQL/result escaping and page wiring checks passed.');

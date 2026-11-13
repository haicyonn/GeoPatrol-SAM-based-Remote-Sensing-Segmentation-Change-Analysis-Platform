const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const file=path.join(__dirname,'../web/analysis_tasks.js');
assert.ok(fs.existsSync(file),'Missing task analysis UI');
const app=require(file);
const base={status:'planned',planId:'p',title:'<script>unsafe</script>',goal:'分布分析',scopeRegion:'whu',
  dataVersion:'abc',modelId:'Qwen/test',steps:[{id:'S1',title:'概况',question:'范围与均值',sql:'SELECT COUNT(*) AS n FROM samples'}],limits:[]};
const html=app.render(base);
assert.ok(!html.includes('<script>'));
assert.ok(html.includes('&lt;script&gt;'));
assert.ok(html.includes('执行分析'));
const finished=app.render({...base,status:'completed',steps:[{...base.steps[0],status:'completed',columns:['n'],rows:[{n:2}],attempts:[]}],
  report:{status:'generated',findings:[{text:'共有2张样本。',evidence:['S1:R1']}],nextQuestions:[]},evidence:[{id:'S1:R1',values:{n:2}}]});
assert.ok(finished.includes('S1:R1'));
assert.ok(finished.includes('共有2张样本'));
assert.ok(finished.includes('导出分析证据'));
const rejected=app.render({...base,status:'partial',steps:[{...base.steps[0],status:'failed',error:'bad <column>',rows:[],columns:[],attempts:[]}],
  report:{status:'rejected',findings:[],message:'保留实际结果'},evidence:[]});
assert.ok(rejected.includes('bad &lt;column&gt;'));
assert.ok(!rejected.includes('分析已全部完成'));
const scoped=app.render({...base,scopeRegion:null,targetRegion:'levir',scopeAdjustment:'跨区域比较 <确认>'});
assert.ok(scoped.includes('当前对象：levir'));
assert.ok(scoped.includes('本批次全部区域'));
assert.ok(scoped.includes('跨区域比较 &lt;确认&gt;'));
const empty=app.render({...base,status:'partial',steps:[{...base.steps[0],status:'no_data',rows:[],columns:[],attempts:[{status:'executed',sql:'SELECT 1'}]}],
  report:{status:'facts_only',message:'目标未完整完成',findings:[],unansweredSteps:[{question:'第三高区域',reason:'没有可用结果'}]},evidence:[]});
assert.ok(empty.includes('尚未回答：第三高区域'));
assert.ok(empty.includes('已执行，结果不足'));
assert.ok(!empty.includes('查询已完成'));
const page=fs.readFileSync(path.join(__dirname,'../web/analysis.html'),'utf8');
assert.ok(page.includes('analysis_tasks.js'));
assert.ok(page.includes('analysisGoal'));
assert.ok(!page.includes('空洞'));
new (require('node:vm').Script)(fs.readFileSync(file,'utf8'));
console.log('Analysis plan preview, actual results, rejected report and HTML escaping passed.');

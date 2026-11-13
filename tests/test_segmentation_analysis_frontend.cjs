const {test}=require('node:test'),assert=require('node:assert/strict');
const app=require('../backend-java/src/main/resources/static/analysis-renderer.js');
test('segmentation clarification explains the missing metric and exposes an explicit escaped choice',()=>{
 const html=app.render({status:'needs_clarification',answer:'没有变化比例',originalGoal:'排名第三变化比例的图片',options:[{label:'按建筑覆盖比例继续',goal:'分析第3张图片及建筑量" <script>'}]});
 assert.ok(html.includes('按建筑覆盖比例继续'));assert.ok(html.includes('原问题：排名第三变化比例的图片'));
 assert.ok(html.includes('data-analysis-goal='));assert.ok(!html.includes('<script>'));assert.ok(html.includes('&quot;'));
 assert.ok(!html.includes('执行分析'));
});
test('rank preview distinguishes image entity, metric and ordinal',()=>{
 const html=app.render({status:'planned',title:'图片排名',goal:'建筑覆盖比例第三高',dataVersion:'v1',sampleRank:{field:'foreground_ratio',direction:'DESC',start:3},steps:[],limits:[]});
 assert.ok(html.includes('排名对象：图片'));assert.ok(html.includes('建筑覆盖比例（0—1）'));assert.ok(html.includes('降序第3张'));assert.ok(html.includes('执行分析'));
});

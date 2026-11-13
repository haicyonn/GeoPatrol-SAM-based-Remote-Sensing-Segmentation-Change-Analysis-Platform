const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const {test}=require('node:test');
const folder=require('node:path').join(__dirname,'../backend-java/src/main/resources/static');
test('analysis controls recover on source reload and stale response cannot unlock a newer request',async()=>{
  const nodes=new Map(),button={disabled:false},scope={disabled:false,value:'',innerHTML:''};
  function node(id){if(id==='analysisScope')return scope;if(!nodes.has(id))nodes.set(id,{value:'',innerHTML:'',textContent:'',classList:{toggle(){}},replaceChildren(){this.innerHTML='';}});return nodes.get(id);}
  let finish,epoch=0;
  const controller={context:null,capture:()=>epoch,isCurrent:t=>t===epoch,analysis:()=>new Promise(r=>finish=r)};
  const sandbox={console,URL,location:{href:'http://example.test/analysis.html',search:''},history:{replaceState(){}},document:{body:{dataset:{consolePage:'analysis'}},getElementById:node,querySelectorAll:selector=>selector.includes('analysisForm')?[button]:[]},Headers,AnalysisTasks:{render:()=>'<p>STALE PRIVATE</p>'},ConsoleAdapter:{esc:String,createController:()=>controller}};sandbox.window=sandbox;
  let source=fs.readFileSync(folder+'/console-bootstrap.js','utf8').replace(/  start\(\);\s*\}\)\(\);\s*$/,'  window.testHooks={resetPrivateViews,runAnalysis};\n})();');
  vm.runInNewContext(source,sandbox);
  const context=id=>({batchId:id,label:id,sourceType:'INFERENCE',reviewVersion:0,app:{regions:[]}});
  controller.context=context('old');sandbox.testHooks.resetPrivateViews(controller.context);
  const pending=sandbox.testHooks.runAnalysis({action:'plan'});assert.equal(button.disabled,true);
  epoch++;controller.context=null;sandbox.testHooks.resetPrivateViews(null);assert.equal(button.disabled,true);
  controller.context=context('new');sandbox.testHooks.resetPrivateViews(controller.context);
  assert.equal(button.disabled,false);assert.equal(scope.disabled,false);
  const finishOld=finish;const newer=sandbox.testHooks.runAnalysis({action:'plan'});assert.equal(button.disabled,true);
  finishOld({planId:'old'});await pending;assert.equal(button.disabled,true);assert.doesNotMatch(node('analysisResult').innerHTML,/STALE PRIVATE/);
  finish(null);await newer;assert.equal(button.disabled,false);
});

'use strict';
const $=id=>document.getElementById(id);
function newSessionId(){const bytes=new Uint8Array(16);globalThis.crypto.getRandomValues(bytes);return Array.from(bytes,b=>b.toString(16).padStart(2,'0')).join('');}
let selected=null,batch=null,actor=null,csrf=null,session=newSessionId(),epoch=0,busy=false,polling=false;
let reviewing=null,loadedReview=null,reviewSelection=0,reviewSaving=false;
let registeredModels=[];
function selectTaskType(){
  const seg=$('taskType').value==='building_segmentation';
  $('beforeLabel').textContent=seg?'原始影像（可多选）':'前时相影像（可多选）';
  $('after').hidden=seg;$('afterLabel').hidden=seg;$('after').required=!seg;
  $('inputHelp').textContent=seg?'每张图独立提取建筑区域，无需后时相；每批最多200张。':'按文件名配对，前后时相应使用相同名称，每批最多200组。';
  $('model').replaceChildren();
  for(const model of registeredModels.filter(m=>(m.taskType||'change_detection')===$('taskType').value)){const option=node('option',model.name+(model.available?'':'（权重不可用）'));option.value=model.modelId;option.disabled=!model.available;$('model').append(option);}
}
$('taskType').onchange=selectTaskType;
const statusNames={DRAFT:'待执行',RUNNING:'预测中',PENDING:'待处理',SUCCEEDED:'已完成',COMPLETED:'正在发布',PUBLISHED:'可分析',FAILED:'失败',CANCELLED:'已取消'};
const verdictNames={unreviewed:'未复核',confirmed_change:'确认变化',false_change:'伪变化',uncertain:'暂不确定'};
function node(tag,text,className){const e=document.createElement(tag);if(text!==undefined)e.textContent=String(text);if(className)e.className=className;return e;}
function loggedOut(){
  actor=null;selected=null;batch=null;epoch++;session=newSessionId();busy=false;reviewing=null;loadedReview=null;reviewSelection++;
  $('app').hidden=true;$('loginPanel').hidden=false;$('logout').hidden=true;$('changePassword').hidden=true;$('accountName').textContent='';$('password').value='';
  for(const id of ['chat','images','results','tasks','batches','members','reviewHistory','users'])$(id).replaceChildren();
  for(const id of ['reviewNote','question','currentPassword','newPassword','memberPassword'])$(id).value='';
  $('batchTitle').textContent='选择或创建一个批次';$('batchState').textContent='预测完成后，结果会保存并开放本批问数。';$('progress').value=0;
  $('reviewPanel').hidden=true;$('sharing').hidden=true;$('passwordPanel').hidden=true;$('adminPanel').hidden=true;updateQueryState();updateReviewControls();$('health').textContent='请登录';
}
async function api(path,options={}){
  const headers=new Headers(options.headers||{});
  if(!['GET','HEAD','OPTIONS'].includes(options.method||'GET')&&csrf)headers.set(csrf.headerName,csrf.token);
  const response=await fetch('/api/v1'+path,{...options,headers,credentials:'same-origin',cache:'no-store'});
  let data;try{data=await response.json();}catch{throw new Error('服务返回了无法解析的响应');}
  if(!response.ok){if(response.status===401&&actor)loggedOut();const error=new Error(data.error||'请求失败：'+response.status);error.status=response.status;error.data=data;throw error;}return data;
}
const post=(path,body)=>api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});
async function refreshCsrf(){csrf=await api('/auth/csrf');}
function table(rows){const wrap=node('div',undefined,'scroll'),t=node('table');wrap.append(t);if(!rows?.length){wrap.append(node('p','没有符合条件的记录','muted'));return wrap;}const keys=Object.keys(rows[0]);const head=node('tr');keys.forEach(k=>head.append(node('th',k)));const thead=node('thead');thead.append(head);t.append(thead);const body=node('tbody');rows.forEach(row=>{const tr=node('tr');keys.forEach(k=>tr.append(node('td',row[k]??'未知')));body.append(tr);});t.append(body);return wrap;}
function canEdit(){return batch&&['OWNER','EDIT'].includes(batch.permission);}
function resetSession(){epoch++;session=newSessionId();$('chat').replaceChildren();busy=false;updateQueryState();}
function reviewReady(){return loadedReview&&reviewing&&loadedReview.sampleId===reviewing.sampleId&&loadedReview.batchId===selected&&loadedReview.selection===reviewSelection;}
function updateReviewControls(){const editable=reviewReady()&&canEdit()&&!reviewSaving;$('saveReview').disabled=!editable;$('verdict').disabled=!editable;$('reviewNote').readOnly=!editable;}
function updateQueryState(){
  const enabled=actor&&batch?.status==='PUBLISHED'&&!busy;$('question').disabled=!enabled;$('send').disabled=!enabled;
  $('queryHint').textContent=batch?.status==='PUBLISHED'?'个人会话只分析当前批次；复核数据更新后会明确切换到新版本。':'本批完成并发布后可开始查询。';
  if(batch)document.querySelectorAll('#batches button').forEach(b=>{if(b.dataset.batchId===selected)b.textContent=batch.label+' · '+(statusNames[batch.status]||batch.status);});
}
async function refreshList(){
  const version=epoch,data=await api('/batches');if(version!==epoch||!actor)return;$('batches').replaceChildren();
  for(const row of data.batches){const button=node('button',row.label+' · '+(statusNames[row.status]||row.status),'batch-button'+(row.batchId===selected?' active':''));button.dataset.batchId=row.batchId;button.onclick=()=>selectBatch(row.batchId).catch(showBatchError);$('batches').append(button);}
  if(!data.batches.length)$('batches').append(node('p','暂无可访问批次。创建一个批次或请创建者授权。','muted'));
}
function showBatchError(e){$('batchState').textContent=e.message;if(e.status===403){batch=null;reviewing=null;$('results').replaceChildren();$('images').replaceChildren();$('chat').replaceChildren();$('reviewPanel').hidden=true;$('sharing').hidden=true;updateQueryState();}}
async function selectBatch(id){selected=id;batch=null;reviewing=null;loadedReview=null;reviewSelection++;updateReviewControls();resetSession();$('images').replaceChildren();$('reviewPanel').hidden=true;$('sharing').hidden=true;await refreshSelected();await refreshList();}
async function refreshSelected(){
  if(!selected||!actor)return;const version=epoch,id=selected,value=await api('/batches/'+id);if(version!==epoch)return;batch=value;updateReviewControls();
  $('batchTitle').textContent=value.label;$('batchState').textContent=(statusNames[value.status]||value.status)+' · 完成 '+value.completed+'/'+value.total+'，失败 '+value.failed+' · 复核数据版本 '+value.reviewVersion;
  $('progress').max=value.total;$('progress').value=value.completed;$('retry').hidden=!canEdit()||value.status!=='FAILED';$('cancel').hidden=!canEdit()||!['DRAFT','RUNNING','FAILED'].includes(value.status);
  $('manageMembers').hidden=value.permission!=='OWNER';$('exportReport').hidden=value.status!=='PUBLISHED';$('openConsole').hidden=value.status!=='PUBLISHED';$('openAnalysis').hidden=value.status!=='PUBLISHED';$('tasks').replaceChildren();
  for(const task of value.tasks){const tr=node('tr');[task.sampleName,task.regionId,statusNames[task.status]||task.status,task.attemptNo,task.error||''].forEach(v=>tr.append(node('td',v)));$('tasks').append(tr);}updateQueryState();
  if(value.sourceType==='ARCHIVE'||value.taskType==='building_segmentation'){$('results').replaceChildren(node('p',(value.taskType==='building_segmentation'?'建筑分割':'原全量样本库')+'，共'+value.total+'张影像；请进入同一巡检控制台查看影像、智能问答、人工精修和报告。','note'));}
  else if(value.completed){const result=await api('/batches/'+id+'/results?limit=200');if(version!==epoch)return;renderResults(result.rows);}else $('results').replaceChildren();
}
function renderResults(rows){
  $('results').replaceChildren(node('h3','已保存的检测结果'));
  $('results').append(table(rows.map(r=>({样本:r.sampleName,区域:r.regionId,变化像素:r.changedPx,变化比例:(r.changedRatio*100).toFixed(3)+'%',连通对象数:r.objectCount,人工复核:verdictNames[r.reviewVerdict]||'未复核'}))));
  rows.forEach(r=>{const button=node('button','查看与复核 '+r.sampleName,'small');button.onclick=()=>showImages(r).catch(showBatchError);$('results').append(button);});
}
async function showImages(row){
  reviewing=row;const version=epoch;$('images').replaceChildren();
  for(const [title,key] of [['前时相','beforeAssetId'],['后时相','afterAssetId'],['预测掩膜','predMaskAssetId']]){const figure=node('figure'),image=node('img');image.src='/api/v1/assets/'+encodeURIComponent(row[key]);image.alt=title;figure.append(image,node('figcaption',title));$('images').append(figure);}
  $('reviewPanel').hidden=false;$('reviewTitle').textContent='样本复核 · '+row.sampleName;$('reviewStatus').textContent='';await loadReview(version,row.sampleId);
}
async function loadReview(version=epoch,sample=reviewing?.sampleId){
  if(!sample)return;const id=selected,selection=++reviewSelection;loadedReview=null;updateReviewControls();$('reviewNote').value='';$('verdict').value='uncertain';$('reviewMeta').textContent='正在载入本样本的复核记录…';$('reviewHistory').replaceChildren();
  const value=await api('/batches/'+id+'/reviews/'+sample);if(id!==selected||selection!==reviewSelection||reviewing?.sampleId!==sample)return;
  loadedReview={batchId:id,sampleId:sample,version:value.version,selection};$('verdict').value=value.verdict==='unreviewed'?'uncertain':value.verdict;$('reviewNote').value=value.note;
  $('reviewMeta').textContent='记录版本 '+value.version+' · '+(value.authorName?'上次由 '+value.authorName+' 更新':'尚无人工复核记录');
  updateReviewControls();
  const history=await api('/batches/'+id+'/reviews/'+sample+'/history');if(id!==selected||selection!==reviewSelection||reviewing?.sampleId!==sample)return;
  $('reviewHistory').replaceChildren(table(history.history.map(r=>({版本:r.version,结论:verdictNames[r.verdict],成员:r.authorName,说明:r.note,时间:new Date(r.createdAt).toLocaleString()}))));
}
$('reviewForm').onsubmit=async event=>{
  event.preventDefault();if(!reviewReady()||!canEdit()||reviewSaving)return;const state=loadedReview,version=epoch,sample=state.sampleId,id=state.batchId;reviewSaving=true;updateReviewControls();
  try{await api('/batches/'+id+'/reviews/'+sample,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({expectedVersion:state.version,verdict:$('verdict').value,note:$('reviewNote').value})});
    if(state.selection!==reviewSelection||id!==selected||reviewing?.sampleId!==sample)return;$('reviewStatus').textContent='已保存。之后的问数将使用更新后的复核版本。';await loadReview(version,sample);await refreshSelected();
  }catch(e){if(state.selection===reviewSelection&&id===selected)$('reviewStatus').textContent=e.status===409?e.message+'；您的输入已保留，点击“载入最新版本”后核对。':e.message;}
  finally{reviewSaving=false;updateReviewControls();}
};
$('reloadReview').onclick=()=>loadReview().catch(e=>{$('reviewStatus').textContent=e.message;});
$('manageMembers').onclick=async()=>{
  const version=epoch,id=selected;$('sharing').hidden=!$('sharing').hidden;if($('sharing').hidden)return;
  try{const data=await api('/batches/'+id+'/members');if(version!==epoch)return;$('members').replaceChildren();
    for(const user of data.availableUsers){if(user.id===batch.ownerId)continue;const row=node('div',undefined,'member-row');row.append(node('span',user.displayName+'（'+user.username+'）'));
      const select=node('select');select.setAttribute('aria-label',user.username+'的批次权限');for(const [value,label] of [['NONE','未授权'],['VIEW','只读'],['EDIT','协作']]){const option=node('option',label);option.value=value;select.append(option);}select.value=data.members.find(m=>m.id===user.id)?.permission||'NONE';
      const save=node('button','保存权限','small');save.onclick=async()=>{save.disabled=true;try{await post('/batches/'+id+'/members/'+user.id,{permission:select.value});if(version===epoch)$('sharingStatus').textContent='已更新 '+user.displayName+' 的权限。';}catch(e){if(version===epoch)$('sharingStatus').textContent=e.message;}finally{save.disabled=false;}};row.append(select,save);$('members').append(row);}
  }catch(e){$('sharingStatus').textContent=e.message;}
};
$('exportReport').onclick=()=>{if(selected&&batch?.status==='PUBLISHED')window.location.href='/api/v1/batches/'+selected+'/report';};
$('openConsole').onclick=()=>{if(selected&&batch?.status==='PUBLISHED')location.href='/console.html?batch='+encodeURIComponent(selected);};
$('openAnalysis').onclick=()=>{if(selected&&batch?.status==='PUBLISHED')location.href='/analysis.html?batch='+encodeURIComponent(selected);};
$('upload').onsubmit=async event=>{
  event.preventDefault();const before=$('before').files,seg=$('taskType').value==='building_segmentation',after=seg?[]:$('after').files;if((!seg&&before.length!==after.length)||!before.length){$('uploadStatus').textContent=seg?'请选择原始影像。':'请上传数量相同的前、后时相影像。';return;}if(!$('model').value){$('uploadStatus').textContent='当前没有可用的检测模型。';return;}
  $('submitUpload').disabled=true;const data=new FormData();data.append('taskType',$('taskType').value);data.append('label',$('label').value);data.append('regionId',$('region').value);for(const file of before)data.append('before',file);for(const file of after)data.append('after',file);
  try{$('uploadStatus').textContent='正在上传并检查影像配对…';const created=await api('/batches',{method:'POST',body:data});await post('/batches/'+created.batchId+'/start',{modelId:$('model').value});$('uploadStatus').textContent='批次已保存，后台正在排队预测。';await selectBatch(created.batchId);}catch(e){$('uploadStatus').textContent=e.message;}finally{$('submitUpload').disabled=false;}
};
$('refresh').onclick=()=>refreshList().catch(showBatchError);
$('retry').onclick=async()=>{try{await post('/batches/'+selected+'/retry-failed');await refreshSelected();}catch(e){showBatchError(e);}};
$('cancel').onclick=async()=>{try{await post('/batches/'+selected+'/cancel');await refreshSelected();}catch(e){showBatchError(e);}};
$('newSession').onclick=resetSession;
async function ask(question,extra={}){
  if(busy||batch?.status!=='PUBLISHED')return;busy=true;updateQueryState();const version=epoch,id=selected;const user=node('div',undefined,'chat-item user');user.append(node('div',question,'message'));$('chat').append(user);
  const answer=node('div',undefined,'chat-item');answer.append(node('div','正在等待模型并分析问题…','message'));$('chat').append(answer);
  try{const result=await post('/batches/'+id+'/questions',{sessionId:session,question,...extra});if(version!==epoch)return;
    const label=result.status==='chat_completed'?'本地聊天 · 未查询样本库':result.status==='completed'?'本批查询已完成':result.status==='needs_clarification'?'需要补充条件':'本轮未完成';
    answer.replaceChildren(node('div',label+' · 复核数据版本 '+result.reviewVersion,'chat-meta'),node('div',result.answer||result.message||result.clarification?.question||result.clarification||'请查看查询过程','message'));
    if(result.contextReset)answer.prepend(node('p','复核数据已更新，本次从新版本开始分析；请重新明确需要沿用的筛选条件。','note'));
    if(result.dataChangedDuringQuery)answer.prepend(node('p','处理期间有成员更新了复核，本回答保留上述版本；下一问会切换新版本。','note'));
    if(result.status==='completed'&&result.sqlRows)answer.append(table(result.sqlRows));
    const choices=result.clarification?.options||result.clarificationOptions||[];
    for(const option of choices){const b=node('button',option.label||option.title||option.id);b.onclick=()=>ask(option.label||option.title||'已选择口径',{clarificationId:result.clarification?.id||result.clarificationId,optionId:option.id});answer.append(b);}
    const details=node('details'),summary=node('summary','查看SQL与查询过程');details.append(summary,node('pre',JSON.stringify(result,null,2)));answer.append(details);
  }catch(e){if(version===epoch)answer.replaceChildren(node('div',e.message,'message error'));}finally{if(version===epoch){busy=false;updateQueryState();}}
}
$('query').onsubmit=event=>{event.preventDefault();const q=$('question').value.trim();if(q){$('question').value='';ask(q);}};
document.querySelectorAll('[data-question]').forEach(button=>button.onclick=()=>{$('question').value=button.dataset.question;});
async function loadUsers(){
  const identity=actor?.id,data=await api('/admin/users');if(!actor||identity!==actor.id||actor.role!=='ADMIN')return;$('users').replaceChildren();
  for(const user of data.users){const row=node('div',undefined,'member-row');row.append(node('span',user.displayName+' · '+user.username+' · '+(user.enabled?'可用':'已禁用')));if(user.id!==actor.id){const button=node('button',user.enabled?'禁用':'启用','small');button.onclick=async()=>{try{await post('/admin/users/'+user.id+'/enabled',{enabled:!user.enabled});await loadUsers();}catch(e){$('userStatus').textContent=e.message;}};row.append(button);
    const password=node('input');password.type='password';password.autocomplete='new-password';password.placeholder='重设密码（至少10字符）';password.setAttribute('aria-label',user.username+'的新密码');
    const reset=node('button','重设密码','small');reset.onclick=async()=>{try{await post('/admin/users/'+user.id+'/password',{password:password.value});password.value='';$('userStatus').textContent='密码已重设，原登录已失效。';}catch(e){$('userStatus').textContent=e.message;}};row.append(password,reset);
  }$('users').append(row);}
}
$('createUser').onsubmit=async event=>{event.preventDefault();try{await post('/admin/users',{username:$('memberUsername').value,displayName:$('memberName').value,password:$('memberPassword').value,role:'MEMBER'});$('memberPassword').value='';$('userStatus').textContent='成员账号已创建，请私下交付账号和初始密码。';await loadUsers();}catch(e){$('userStatus').textContent=e.message;}};
$('reloadUsers').onclick=()=>loadUsers().catch(e=>{$('userStatus').textContent=e.message;});
$('changePassword').onclick=()=>{$('passwordPanel').hidden=!$('passwordPanel').hidden;};
$('passwordForm').onsubmit=async event=>{event.preventDefault();try{await post('/auth/password',{currentPassword:$('currentPassword').value,password:$('newPassword').value});$('currentPassword').value='';$('newPassword').value='';loggedOut();await refreshCsrf();$('loginStatus').textContent='密码已更新，请重新登录。';}catch(e){$('passwordStatus').textContent=e.message;}};
$('logout').onclick=async()=>{try{await post('/auth/logout');loggedOut();await refreshCsrf();}catch(e){$('loginStatus').textContent=e.message;}};
function returnToConsole(){const target=new URLSearchParams(location.search).get('returnTo');if(!target)return false;try{const url=new URL(target,location.origin);if(url.origin===location.origin&&['/','/index.html','/console.html','/analysis.html'].includes(url.pathname)){location.replace(url.pathname+url.search+url.hash);return true;}}catch{}return false;}
$('login').onsubmit=async event=>{event.preventDefault();$('loginButton').disabled=true;try{await refreshCsrf();await api('/auth/login',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:new URLSearchParams({username:$('username').value,password:$('password').value})});$('password').value='';await refreshCsrf();await loadWorkspace();}catch(e){$('loginStatus').textContent=e.message;}finally{$('loginButton').disabled=false;}};
async function loadWorkspace(){
  actor=await api('/auth/me');$('loginPanel').hidden=true;$('app').hidden=false;$('logout').hidden=false;$('changePassword').hidden=false;$('accountName').textContent=actor.displayName;$('adminPanel').hidden=actor.role!=='ADMIN';
  if(returnToConsole())return;
  const health=await api('/health');$('health').textContent=health.database==='ok'&&health.python.status==='ok'?'服务已连接':'部分服务未就绪';$('model').replaceChildren();
  try{const data=await api('/models');registeredModels=data.models;selectTaskType();}
  catch(e){if(!actor)return;const option=node('option','模型服务暂不可用');option.value='';$('model').append(option);$('uploadStatus').textContent='模型服务暂不可用；仍可查看已有结果、复核和导出报告。';}
  await refreshList();const initial=new URLSearchParams(location.search).get('batchId');if(initial)await selectBatch(initial);if(actor?.role==='ADMIN')try{await loadUsers();}catch(e){$('userStatus').textContent=e.message;}
}
setInterval(async()=>{if(polling||!selected||!actor)return;polling=true;try{await refreshSelected();}catch(e){showBatchError(e);}finally{polling=false;}},2000);
(async()=>{try{await refreshCsrf();await loadWorkspace();}catch(e){loggedOut();if(e.status!==401)$('loginStatus').textContent=e.message;}})();

(function(root,factory){
  const api=factory();
  if(typeof module==='object'&&module.exports)module.exports=api;
  else root.ConsoleAdapter=api;
})(typeof globalThis!=='undefined'?globalThis:this,function(){
  'use strict';
  const esc=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const verdicts={unreviewed:'未复核',confirmed_change:'确认变化',false_change:'伪变化',accepted_mask:'分割可接受',needs_correction:'需要修正',uncertain:'暂不确定'};
  const finite=value=>typeof value==='number'&&Number.isFinite(value)?value:null;
  const session=()=>globalThis.crypto?.randomUUID?.()||'console-'+Date.now().toString(36)+'-'+Math.random().toString(36).slice(2);
  function normalizeConsole(raw){
    if(!raw||!raw.batchId||!raw.app||!Array.isArray(raw.app.regions))throw new Error('批次控制台数据不完整');
    const context=JSON.parse(JSON.stringify(raw));context.readOnly=!!raw.readOnly;if(raw.sourceType==='ARCHIVE'&&raw.label==='历史样本参考（只读）')context.label='原版全量样本库';
    context.app.regions=context.app.regions.map(region=>({...region,items:(region.items||[]).map(original=>{
      const item={...original,stem:original.stem||original.name||original.id,regionId:region.id,assets:{}};
      for(const [key,url] of Object.entries(original.assets||{})){
        if(!url)continue;
        if(!/^\/api\/v1\/(?:assets|reference-assets|segmentation-assets)\/[A-Za-z0-9_.%-]+$/.test(String(url)))throw new Error('批次资产地址不受当前服务控制');
        item.assets[key]=url;
      }
      if(item.assets.predMask&&!item.assets.mask)item.assets.mask=item.assets.predMask;
      const size=original.sourceSize;
      item.sourceSize=Array.isArray(size)&&size.length===2&&size.every(n=>Number.isInteger(n)&&n>0)?size:null;
      if(context.sourceType==='INFERENCE'){
        for(const key of ['pred','eventType','primaryEventType','eventTypeZh','primaryEventName','morphologyType','morphologyLabel','scale','confidence','gridPosition','mainBbox','objects'])if(!['mainBbox','objects'].includes(key)||!item.hasApprovedCorrection)delete item[key];
      }
      return item;
    })}));
    return context;
  }
  function canReview(context){return !!context&&!context.readOnly&&['OWNER','EDIT'].includes(context.permission)&&context.capabilities?.reviewVerdicts!==false;}
  function sourceLabel(context){return context.sourceType==='ARCHIVE'?'原全量样本库':'实际模型预测与协同复核'+(context.readOnly?' · 只读权限':'');}
  function facts(item,context){
    const historical=context?.sourceType==='ARCHIVE';
    const confidence=finite(item.confidence);
    return {event:historical?(item.primaryEventName||item.eventTypeZh||'未知'):'未知',
      confidence:historical&&confidence!==null?(confidence*100).toFixed(1)+'%':'未知',
      dimensions:item.sourceSize?item.sourceSize.join(' × '):'未知',
      review:verdicts[item.manualReview?.verdict||item.reviewVerdict||'unreviewed']||'未知'};
  }
  function statistics(items){
    return {samples:items.length,changed:items.filter(i=>(finite(i.changedPx)||0)>0).length,
      changedPixels:items.reduce((n,i)=>n+(finite(i.changedPx)||0),0),
      reviewed:items.filter(i=>i.manualReview||i.reviewVerdict&&i.reviewVerdict!=='unreviewed').length};
  }
  function renderReply(data){
    const sql=data.questionRoute==='sql'||data.usedSQL===true;
    const badge=sql?(data.usedSQL?'数据查询 · 已执行 SQL':'数据查询 · 未执行 SQL'):'对话回复 · 未执行 SQL';
    const columns=data.sqlColumns||Object.keys(data.sqlRows?.[0]||{});
    const rows=sql&&data.status==='completed'&&data.sqlRows?`<div class="console-table"><table><thead><tr>${columns.map(c=>`<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>${data.sqlRows.map(row=>`<tr>${columns.map(c=>`<td>${esc(row[c]??'未知')}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`:'';
    const clarification=data.clarification;
    const binding=data.questionContext;
    const contextNote=binding?`<p class="console-meta">${binding.regionLabel?'本轮区域指代：'+esc(binding.regionLabel)+'。':''}${binding.countUnit==='sample_pair'?'图片数量按双时相样本组统计，每组计 1。':''}</p>`:'';
    return `<p class="console-meta">${badge}${data.reviewVersion!=null?' · 复核版本 '+esc(data.reviewVersion):''}</p>${contextNote}<div class="console-answer">${esc(data.answer||data.message||(typeof clarification==='string'?clarification:clarification?.question)||'暂无回复')}</div>${data.contextReset?'<p class="console-note">复核版本已更新，本次使用新版本。</p>':''}${data.dataChangedDuringQuery?'<p class="console-note">处理期间复核已变化；本条回复保留所示历史版本。</p>':''}${rows}${data.countingNotice?`<p class="console-note">${esc(data.countingNotice)}</p>`:""}${clarification?.options?.length?`<div class="console-choices">${clarification.options.map(o=>`<button class="btn ghost" data-clarification-id="${esc(clarification.id)}" data-choice-id="${esc(o.id)}">${esc(o.label||o.title||o.id)}</button>`).join('')}</div>`:''}${sql?`<details><summary>SQL 与查询证据</summary><pre>${esc(JSON.stringify(data,null,2))}</pre></details>`:''}`;
  }
  function reportText(summary,context,scope){
    if(summary?.status!=='done')throw new Error('请先生成当前版本的服务端统计摘要');
    const labels={estimatedBuildingCountSum:'建筑量合计（估计）',filteredFragmentCountSum:'已过滤小块',boundaryBuildingCountSum:'触及边界的候选',countMethod:'计数方法版本',countMinAreaPixels:'最小候选面积（像素）',positiveImages:'含建筑影像',foregroundPixelsSum:'建筑像素合计',meanForegroundRatio:'平均建筑像素占比（0—1）',componentCountSum:'建筑连通区域合计',correctedSamples:'已审批修正样本',resultBasis:'统计口径',sampleCount:'样本数',samples:'样本数',totalCount:'范围样本数',changedSamples:'变化样本数',changedCount:'变化样本数',changedPixelsSum:'变化像素合计',meanChangedRatio:'平均变化比例',reviewedSamples:'已记录复核样本',reviewed:'已记录复核样本',reviewedCount:'已记录复核样本',reviewCount:'明确标记的复核候选数',confirmedChange:'确认变化',confirmedChangeCount:'确认变化',falseChange:'伪变化',falseChangeCount:'伪变化',uncertain:'暂不确定',uncertainCount:'暂不确定',unknownRiskCount:'风险未知样本',regionId:'区域',filterText:'服务端范围',stem:'样本名',id:'样本ID',changedPx:'变化像素',objectCount:'连通对象数',label:'分组',count:'样本数',ratio:'比例',key:'分组标识'};
    const lines=[];
    function append(value,prefix='',depth=0){
      if(depth>3||value==null)return;
      if(Array.isArray(value)){value.slice(0,30).forEach((row,i)=>append(row,prefix+(i+1)+'. ',depth+1));return;}
      if(typeof value==='object')for(const [key,entry] of Object.entries(value)){
        if(['data','items','assets','report','models','generatedBy'].includes(key))continue;
        if(entry&&typeof entry==='object')append(entry,prefix,depth+1);
        else lines.push(prefix+(labels[key]||key)+'：'+String(entry??'未知'));
      }
    }
    append(summary.report);
    return [context.label+(context.taskType==='building_segmentation'?' 建筑分割统计报告':' 巡检统计报告'),'报告范围：'+scope,
      '来源：'+(context.sourceType==='ARCHIVE'?'原全量样本库':'本批模型推理'),
      '批次：'+context.batchId,'数据版本：'+String(summary.dataVersion||context.dataVersion),
      '复核版本：'+String(summary.reviewVersion??context.reviewVersion),
      '结果口径：'+(context.resultBasis==='original'?'原始预测':'人工修正后（无修正时沿用预测）'),'预测摘要：'+String(context.predictionDigest||'未提供'),
      '模型：'+((context.models||[]).map(m=>(m.modelId||'未知')+' '+(m.modelVersion||'版本未知')).join('；')||'历史来源未登记模型版本'),
      '生成时间：'+new Date().toISOString(),'', '服务端统计摘要',summary.text||'',...lines,'',
      context.taskType==='building_segmentation'?'建筑像素按影像累加，未换算地面面积；连通区域数不等于建筑栋数。人工修正与原始预测分别保存。':'变化量单位为影像像素；逐影像累加不代表去重地理面积。对象数为掩膜连通分量，不代表已确认建筑数。人工复核与原始预测分别保存。',
      '本文件为 HTML 格式的 Word 兼容 .doc 文档。'].join('\n');
  }
  function createController({request,onChange=()=>{}}){
    let current=null,epoch=0,sessionId=session(),report=null;
    const notify=()=>onChange(current);
    const capture=()=>({epoch,batchId:current?.batchId,dataVersion:current?.dataVersion,reviewVersion:current?.reviewVersion,resultBasis:current?.resultBasis||'effective'});
    const valid=token=>token.epoch===epoch&&token.batchId===current?.batchId;
    const batchPath=(id,suffix)=>'/batches/'+encodeURIComponent(id)+'/'+suffix;
    const json=(method,body)=>({method,headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    function checkVersion(value,token){
      if(value.reviewVersion!=null&&Number(value.reviewVersion)!==Number(token.reviewVersion)||value.dataVersion!=null&&String(value.dataVersion)!==String(token.dataVersion))throw new Error('数据版本已变化，请点击“刷新批次”后重试；本次不同版本的回复未填入当前视图。');
    }
    const api={
      get context(){return current;},get sessionId(){return sessionId;},get report(){return report;},
      capture,isCurrent:valid,
      clear(){epoch++;current=null;report=null;sessionId=session();notify();},
      async select(id,basis='effective'){
        api.clear();const selectedEpoch=epoch;
        const raw=await request(batchPath(id,'console')+(basis==='original'?'?basis=original':''));
        if(epoch!==selectedEpoch)return null;
        current=normalizeConsole(raw);notify();return current;
      },
      async question(question,scopeRegion,extra={}){
        if(!current)throw new Error('请先选择批次');const token=capture();
        const result=await request(batchPath(token.batchId,'questions'),json('POST',{question,sessionId,scopeRegion:scopeRegion||null,mode:'auto',...extra,resultBasis:token.resultBasis}));
        if(!valid(token))return null;checkVersion(result,token);return result;
      },
      async history(scopeRegion){return api.question(undefined,scopeRegion,{action:'history'});},
      resetSession(){sessionId=session();return sessionId;},
      async summary(regionId,sampleIds){
        if(!current)throw new Error('请先选择批次');const token=capture();report=null;
        const result=await request(batchPath(token.batchId,'report-summary'),json('POST',{regionId,sampleIds,reviewVersion:token.reviewVersion,resultBasis:token.resultBasis}));
        if(!valid(token))return null;checkVersion(result,token);
        report=result;return result;
      },
      async review(sampleId){
        if(!current)return null;const token=capture();
        const [record,history]=await Promise.all([request(batchPath(token.batchId,'reviews/'+encodeURIComponent(sampleId))),request(batchPath(token.batchId,'reviews/'+encodeURIComponent(sampleId)+'/history'))]);
        return valid(token)?{record,history:history.history||[]}:null;
      },
      async saveReview(sampleId,expectedVersion,verdict,note,extra={}){
        if(!canReview(current))throw new Error('当前批次为只读或没有协作权限');const token=capture();
        const result=await request(batchPath(token.batchId,'reviews/'+encodeURIComponent(sampleId)),json('PUT',{expectedVersion,verdict,note,...extra}));
        return valid(token)?result:null;
      },
      async corrections(){if(!current)return null;return request(batchPath(current.batchId,'corrections'));},
      async analysis(payload){
        if(!current)throw new Error('请先选择批次');const token=capture();
        const result=await request(batchPath(token.batchId,'analysis'),json('POST',{...payload,sessionId,reviewVersion:token.reviewVersion,resultBasis:token.resultBasis}));
        if(!valid(token))return null;checkVersion(result,token);return result;
      }
    };return api;
  }
  return {esc,normalizeConsole,canReview,sourceLabel,facts,statistics,renderReply,reportText,createController,verdicts};
});

(function(){
  'use strict';
  const A=window.ConsoleAdapter,$=id=>document.getElementById(id),esc=A.esc;
  const isAnalysis=document.body.dataset.consolePage==='analysis';
  const legacy=window.ConsoleLegacy;
  let actor=null,csrf=null,selection=0,controller;
  let chats=[],chatBusy=false,queryScope='',clarification=null;
  let selectedReview=null,reviewLoad=0,reviewBusy=false,reviewDraft=null;
  let reportState=null,reportBusy=false,reportAttempted='',reportError='',visibleLimit=100;
  let currentPlan=null,analysisBusy=false;
  let resultBasis='effective',editor=null,geometryDraft=null,galleryPage=0,qualityFilter='all',scaleFilter='all',confidenceMin=0;
  function fitChrome(){
    const header=document.querySelector('.console-header'),bar=$('consoleStatus');
    if(header&&bar)document.documentElement.style.setProperty('--console-top',(header.offsetHeight+bar.offsetHeight)+'px');
  }
  function status(message,error=false){$('consoleStatus').textContent=message;$('consoleStatus').classList.toggle('error',error);}
  function login(){
    controller?.clear();
    actor=null;if($('consoleActor'))$('consoleActor').textContent='';
    const returnTo=isAnalysis?'analysis.html':'index.html';
    location.replace('/manage.html?returnTo='+encodeURIComponent(returnTo+location.search));
  }
  async function request(path,options={}){
    if(!/^\/(?:auth|batches|health)(?:\/|$)/.test(path))throw new Error('不受支持的服务地址');
    const headers=new Headers(options.headers||{});
    if(!['GET','HEAD','OPTIONS'].includes(options.method||'GET')&&csrf)headers.set(csrf.headerName,csrf.token);
    const response=await fetch('/api/v1'+path,{...options,headers,credentials:'same-origin',cache:'no-store'});
    let data;try{data=await response.json();}catch{throw new Error('服务响应无法解析');}
    if(!response.ok){
      const error=Object.assign(new Error(data.error||'请求失败：'+response.status),{status:response.status,data});
      if(response.status===401)login();
      else if(response.status===403){controller?.clear();status(error.message,true);}
      throw error;
    }return data;
  }
  const region=()=>controller.context?.app.regions.find(r=>r.id===legacy?.state.regionId)||controller.context?.app.regions[0];
  const isSeg=()=>controller.context?.taskType==='building_segmentation';
  const pixels=i=>Number((isSeg()?i.foregroundPixels:i.changedPx)||0);
  const imageRatio=i=>isSeg()?i.foregroundRatio:i.changedRatio;
  const objectCount=i=>isSeg()?i.componentCount:i.objectCount;
  const buildingCount=i=>Number.isFinite(i?.estimatedBuildingCount)?i.estimatedBuildingCount:null;
  const scopeLabel=()=>region()?.short+' / '+($('searchInput')?.value.trim()?'搜索：'+$('searchInput').value.trim():'全部样本')+((legacy?.state.classFilter||'all')!=='all'?' / '+legacy.state.classFilter:'')+(qualityFilter!=='all'?' / 质量筛选：'+qualityFilter:'')+(scaleFilter!=='all'?' / 规模：'+scaleFilter:'')+(confidenceMin?' / 参考置信度≥'+confidenceMin+'%':'');
  function scopedItems(){
    const query=String(legacy?.state.search||'').trim().toLowerCase(),filter=legacy?.state.classFilter||'all';
    return (region()?.items||[]).filter(item=>matchesQuality(item)&&(scaleFilter==='all'||item.scale===scaleFilter)&&(!confidenceMin||Number(item.confidence)>=confidenceMin/100)&&(filter==='all'||(item.primaryEventName||item.eventTypeZh||'未知')===filter)&&(!query||[item.name,item.stem,item.id,item.primaryEventName,item.eventTypeZh,item.gridPosition,item.evidenceSummary,(item.reviewReasonCodes||[]).join(' ')].join(' ').toLowerCase().includes(query)))
      .sort((a,b)=>legacy?.state.sortMode==='confidence'?Number(b.confidence??-1)-Number(a.confidence??-1):legacy?.state.sortMode==='latest'?(Date.parse(b.addedAt||b.createdAt||'')||0)-(Date.parse(a.addedAt||a.createdAt||'')||0):legacy?.state.sortMode==='review'?Number(!!a.manualReviewComplete)-Number(!!b.manualReviewComplete)||({high:3,medium:2,low:1}[b.reviewLevel]||0)-({high:3,medium:2,low:1}[a.reviewLevel]||0)||pixels(b)-pixels(a):pixels(b)-pixels(a));
  }
  function resetPrivateViews(context){
    chats=[];chatBusy=false;queryScope='';clarification=null;selectedReview=null;reviewDraft=null;reviewBusy=false;reviewLoad++;geometryDraft=null;editor=null;galleryPage=0;qualityFilter='all';scaleFilter='all';confidenceMin=0;
    reportState=null;reportBusy=false;reportAttempted='';reportError='';currentPlan=null;analysisBusy=false;visibleLimit=100;
    if(legacy){legacy.clear();if(context)legacy.setContext(context);}
    for(const id of ['overviewPane','workPane','trustPane','agentPane','reviewPane','reportPane','regionCards','analysisResult'])if($(id))$(id).replaceChildren();
    for(const id of ['searchInput','analysisGoal'])if($(id))$(id).value='';
    if(isAnalysis){document.querySelectorAll('#analysisForm button,#analysisResult button').forEach(button=>button.disabled=!context);$('analysisScope').disabled=!context;}
    if(!context)return;
    document.title=isSeg()?'巡变智判 · 建筑物分割':'遥感变化智能巡检平台';
    const hero=document.querySelector?.('.hero-copy h1');if(hero)hero.textContent=isSeg()?'建筑分割与智能分析':'变化检测可信判读智能体';
    const lead=document.querySelector?.('.hero-copy .lead');if(lead)lead.textContent=isSeg()?'单张影像提取建筑区域，集中查看结果、对话查询、人工精修与分析报告。':'双时相变化检测、结果查询、人工精修与分析报告。';
    const eyebrow=document.querySelector?.('.hero-copy .eyebrow');if(eyebrow)eyebrow.textContent=isSeg()?'Building Segmentation Workspace':'Remote Sensing Change Inspection';
    const create=document.querySelector?.('.hero-actions a[href="manage.html"]');if(create)create.textContent=isSeg()?'新建分割批次':'新建检测批次';
    status(context.label.replace('（只读）','（原始预测保留）')+' · '+(context.sourceType==='ARCHIVE'?'原全量样本库':'实际推理批次')+' · 复核版本 '+context.reviewVersion);
    const url=new URL(location.href);url.searchParams.set('batch',context.batchId);url.searchParams.set('basis',context.resultBasis||'effective');history.replaceState(null,'',url);
    const link=$('consoleAnalysisLink');if(link)link.href='analysis.html?batch='+encodeURIComponent(context.batchId);
    document.querySelectorAll('a[href]').forEach(anchor=>{
      const target=new URL(anchor.getAttribute('href'),location.href);
      if(target.origin===location.origin&&['/index.html','/console.html','/analysis.html'].includes(target.pathname)){
        target.searchParams.set('batch',context.batchId);anchor.href=target.pathname+target.search+target.hash;
      }
    });
    if(isAnalysis)setupAnalysisContext(context);else render();
  }
  controller=A.createController({request,onChange:resetPrivateViews});
  // Exposed for browser acceptance/debugging; contains only current authenticated state.
  window.ConsoleWorkspace={controller,reload:()=>choose(controller.context?.batchId,true),get actor(){return actor;}};
  async function choose(id,preserveView=false){
    if(!id)return;const turn=++selection;status('正在读取授权批次…');
    const view=preserveView&&legacy&&controller.context?.batchId===id?{regionId:legacy.state.regionId,tab:legacy.state.tab,selectedId:legacy.state.selectedId}:null;
    try{const value=await controller.select(id,resultBasis);if(!value||turn!==selection)return;$('consoleBatch').value=value.batchId;
      if(view){const r=legacy.app.regions.find(r=>r.id===view.regionId);if(r){legacy.state.regionId=r.id;legacy.state.selectedId=r.items.some(i=>i.id===view.selectedId)?view.selectedId:null;}legacy.state.tab=view.tab;render();}
    }
    catch(error){if(turn===selection)status(error.message,true);}
  }
  function metric(label,value,note=''){return `<div class="metric-card"><div class="metric-label">${esc(label)}</div><div class="metric-value">${esc(value)}</div><div class="metric-note">${esc(note)}</div></div>`;}
  function provenance(context){
    const models=(context.models||[]).map(m=>(m.modelId||'未知')+' · '+(m.modelVersion||'版本未知').slice(0,12)).join('；');
    return `<div class="console-note"><strong>${context.sourceType==='ARCHIVE'?'原全量样本库 · 原始预测保留':'当前批次 · 保存的模型结果'}</strong><p>${esc(context.label)} · 数据版本 ${esc(String(context.dataVersion||'未知').slice(0,16))} · 复核版本 ${esc(context.reviewVersion)}</p><p>${esc(models||'历史模型版本未登记')}。${isSeg()?"建筑量是小碎片过滤后的候选区域估计数，粘连未拆分、跨图未去重。":"变化量为像素，连通对象数不是已确认建筑数。"}</p></div>`;
  }
  function selectRegion(id,open=false){
    if(!legacy.app.regions.some(r=>r.id===id))return;
    legacy.state.regionId=id;legacy.state.selectedId=null;legacy.state.search='';galleryPage=0;legacy.state.classFilter='all';$('searchInput').value='';visibleLimit=100;reportState=null;reportAttempted='';reportError='';selectedReview=null;reviewLoad++;
    if(open){$('workspaceOverlay').classList.add('open');document.body.classList.add('workspace-active');}
    render();
  }
  function render(){
    const context=controller.context;if(!context||isAnalysis)return;
    const current=region();if(!current){status('本批次尚无完整发布结果。请到账号与批次页面查看执行进度。');return;}
    $('regionCards').innerHTML=context.app.regions.map(r=>`<button class="region-card console-region-card" data-region-id="${esc(r.id)}"><span class="region-tag">${esc(context.sourceType==='ARCHIVE'?'历史参考':'当前批次')}</span><h2>${esc(r.short||r.id)}</h2><p>${esc(r.name||r.short||r.id)}</p><span>${r.items.length.toLocaleString()} 张样本 · 点击进入</span></button>`).join('');
    document.querySelectorAll('[data-region-id]').forEach(button=>button.onclick=()=>selectRegion(button.dataset.regionId,true));
    $('regionScope').textContent=current.short||current.id;
    $('regionSwitchMenu').innerHTML=context.app.regions.map(r=>`<button class="btn ghost" data-switch-id="${esc(r.id)}">${esc(r.short||r.id)}</button>`).join('');
    document.querySelectorAll('[data-switch-id]').forEach(button=>button.onclick=()=>{selectRegion(button.dataset.switchId);$('regionSwitcher').classList.remove('open');});
    $('workspaceTitle').textContent=(current.short||current.id)+' 巡检控制台';
    $('workspaceSub').textContent=context.label+' · '+A.sourceLabel(context);
    const classOptions=context.sourceType==='ARCHIVE'?[...new Set(current.items.map(i=>i.primaryEventName||i.eventTypeZh).filter(Boolean))]:[];
    $('classFilter').innerHTML='<option value="all">全部样本</option>'+classOptions.map(label=>`<option value="${esc(label)}">${esc(label)}</option>`).join('');
    $('classFilter').value=legacy.state.classFilter;$('classFilter').hidden=!classOptions.length;
    $('sortMode').innerHTML='<option value="area">'+(isSeg()?'建筑像素':'变化像素')+'</option><option value="review">未复核优先</option>'+(!isSeg()?'<option value="confidence">参考置信度</option>':'')+'<option value="latest">最近新增</option>';$('sortMode').value=['review','confidence','latest'].includes(legacy.state.sortMode)?legacy.state.sortMode:'area';
    for(const [tab,pane] of Object.entries({overview:'overviewPane',interpret:'workPane',trust:'trustPane',agent:'agentPane',review:'reviewPane',report:'reportPane'})){
      $(pane).classList.toggle('hidden',legacy.state.tab!==tab);$(pane).classList.add('console-managed-pane');
    }
    document.querySelectorAll('.tab').forEach(tab=>{tab.classList.toggle('active',tab.dataset.tab===legacy.state.tab);if(tab.dataset.tab==='trust')tab.textContent=isSeg()?'分割事实':'检测事实';if(tab.dataset.tab==='review')tab.textContent='协同复核';});
    if(legacy.state.tab==='overview')renderOverview();
    else if(['interpret','trust'].includes(legacy.state.tab))renderImages(legacy.state.tab==='trust'?'trustPane':'workPane');
    else if(legacy.state.tab==='agent')renderChat();
    else if(legacy.state.tab==='review')renderReview();
    else if(legacy.state.tab==='report')renderReport();
  }
  function renderOverview(){
    const context=controller.context,items=scopedItems(),stats=A.statistics(items);
    if(isSeg()){
      const positive=items.filter(i=>pixels(i)>0).length,total=items.reduce((s,i)=>s+pixels(i),0),mean=items.length?items.reduce((s,i)=>s+Number(imageRatio(i)),0)/items.length:0;
      $('overviewPane').innerHTML=`<div class="overview"><div><div class="overview-cards">${metric('范围影像',items.length)}${metric('建筑量',items.reduce((n,i)=>n+Number(buildingCount(i)||0),0).toLocaleString(),'小碎片过滤后的候选区域估计数')}${metric('含建筑影像',positive)}${metric('建筑像素合计',total.toLocaleString(),'像素，不是地面面积')}${metric('平均建筑像素占比',(mean*100).toFixed(3)+'%','包含零建筑影像')}</div><section class="panel console-section"><h2>建筑分割结果分析</h2><p>查看原图与建筑掩膜；在同一智能问答中聊天、筛选和统计，在协同复核中描绘修正区域，审批后用于后续统计。</p><button class="btn primary" id="consoleViewImages">查看影像</button></section></div><aside>${provenance(context)}</aside></div>`;
      $('consoleViewImages').onclick=()=>{legacy.state.tab='interpret';render();};return;
    }
    $('overviewPane').innerHTML=`<div class="overview"><div><div class="overview-cards">${metric('范围样本',stats.samples.toLocaleString())}${metric('变化样本',stats.changed.toLocaleString(),'变化像素大于 0')}${metric('变化像素合计',stats.changedPixels.toLocaleString(),'按影像累加，非地面面积')}${metric('已记录复核',stats.reviewed.toLocaleString(),'结论与预测分别保存')}</div><section class="panel console-section"><h2>基于实际结果开展巡检</h2><p>在全图检索中查看 T1、T2 与预测掩膜，在智能问答中聊天或查询本批数据，在协同复核中保存结论和备注。</p><p>${context.sourceType==='INFERENCE'?'本批没有已验证的事件分类、置信度和风险分级，这些字段显示为未知。':'历史事件和置信度仅作为原始参考元数据保留，不代表当前模型的新推理。'}</p><button class="btn primary" id="consoleViewImages">查看影像</button></section></div><aside>${provenance(context)}<section class="panel console-section"><h3>报告与证据</h3><p>报告摘要由服务端按当前筛选重新统计。Word 兼容 .doc 保留所选批次、范围和数据版本。</p><p>人工修正在协同复核中绘制、保存并审批；原始预测始终保留。</p></section></aside></div>`;
    $('consoleViewImages').onclick=()=>{legacy.state.tab='interpret';render();};
  }
  function sampleList(items,current){
    return `<aside class="panel console-sample-list"><div class="panel-head"><div class="panel-title">样本清单</div><span>${items.length} 张</span></div><div class="list">${items.slice(0,visibleLimit).map(item=>`<button class="sample-card console-sample ${item.id===current?.id?'active':''}" data-sample-id="${esc(item.id)}"><b>${esc(item.stem)}</b><span>${pixels(item).toLocaleString()} px · ${esc(A.facts(item,controller.context).review)}</span></button>`).join('')}</div>${items.length>visibleLimit?'<button class="btn ghost" id="consoleMoreSamples">再显示 100 张</button>':''}</aside>`;
  }
  function images(item){
    if(!item)return '<p class="console-note">当前筛选没有样本。</p>';
    const roles=[...(isSeg()?[['source','原始影像']]:[['pre','T1 前时相'],['post','T2 后时相']]),[item.assets.correctedMask&&controller.context.resultBasis!=='original'?'correctedMask':'mask',item.assets.correctedMask&&controller.context.resultBasis!=='original'?'人工修正掩膜':'原始预测掩膜']];
    if(item.assets.overlay&&(!item.assets.correctedMask||controller.context.resultBasis==='original'))roles.push(['overlay','原始叠加图']);
    return `<div class="console-image-grid">${roles.map(([key,label])=>item.assets[key]?`<figure class="evidence"><a href="${esc(item.assets[key])}" target="_blank" rel="noopener"><img loading="lazy" src="${esc(item.assets[key])}" alt="${esc(label+' '+item.stem)}"></a><figcaption>${esc(label)}</figcaption></figure>`:`<div class="console-note">${esc(label)}未提供</div>`).join('')}</div>`;
  }
  function factsHtml(item){
    if(!item)return '';const f=A.facts(item,controller.context);
    if(isSeg())return `<div class="badge-row"><span class="badge">建筑量 ${buildingCount(item)??'未知'} 个</span><span class="badge">建筑像素 ${pixels(item).toLocaleString()}</span><span class="badge">建筑像素占比 ${(Number(imageRatio(item))*100).toFixed(3)}%</span><span class="badge">建筑连通区域 ${objectCount(item)??'未知'}</span><span class="badge">尺寸 ${esc(f.dimensions)}</span></div><p>复核：${esc(f.review)} · 建筑量为候选区域估计数；过滤 ${item.filteredFragmentCount??'未知'} 个小块，${item.boundaryBuildingCount??'未知'} 个候选触及边界。</p>`;
    return `<div class="badge-row"><span class="badge">${Number(item.changedPx||0).toLocaleString()} px</span><span class="badge">变化比例 ${item.changedRatio==null?'未知':(item.changedRatio*100).toFixed(3)+'%'}</span><span class="badge">连通对象 ${item.objectCount??'未知'}</span><span class="badge">尺寸 ${esc(f.dimensions)}</span></div><p>事件：${esc(f.event)} · 置信度：${esc(f.confidence)} · 风险标记：${esc(item.reviewLevel||'unknown')} · 复核：${esc(f.review)}</p>`;
  }
  function buildingEvidence(item){
    if(!isSeg()||!item?.buildingCandidates)return '';
    const [w,h]=item.sourceSize||[],candidates=item.buildingCandidates;
    const marks=candidates.map(c=>{const [x,y,right,bottom]=c.bbox;return `<rect x="${x}" y="${y}" width="${right-x}" height="${bottom-y}" fill="none" stroke="${c.touchesBoundary?'#ff795d':'#ffe04d'}" stroke-width="1.2"/><text x="${x+1}" y="${Math.max(9,y+9)}" fill="#fff" stroke="#000" stroke-width=".25" font-size="9">${esc(c.id)}</text>`;}).join('');
    return `<details><summary>建筑量计数依据：${buildingCount(item)??'未知'}个候选</summary><p class="console-meta">面积至少${item.countMinAreaPixels}像素；过滤${item.filteredFragmentCount}个小块。橙色框为触及图像边界的候选。框是候选外接框，粘连区域尚未拆分。</p><div style="position:relative;max-width:720px"><img src="${esc(item.assets.source)}" style="display:block;width:100%" alt="建筑量候选依据"><svg viewBox="0 0 ${w} ${h}" style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none">${marks}</svg></div><details><summary>候选面积、坐标与计数版本</summary><pre>${esc(JSON.stringify({method:item.countMethod,minArea:item.countMinAreaPixels,maskSha256:item.countMaskSha256,candidates},null,2))}</pre></details></details>`;
  }
  function bindSamples(){
    document.querySelectorAll('[data-sample-id]').forEach(button=>{button.disabled=reviewBusy;button.onclick=()=>{if(reviewBusy)return;geometryDraft=null;legacy.state.selectedId=button.dataset.sampleId;selectedReview=null;reviewDraft=null;reviewLoad++;render();};});
    if($('consoleMoreSamples'))$('consoleMoreSamples').onclick=()=>{visibleLimit+=100;render();};
  }
  function matchesQuality(i){return qualityFilter==='all'||qualityFilter==='unreviewed'&&!i.manualReviewComplete||qualityFilter==='reviewed'&&i.manualReviewComplete||qualityFilter==='high'&&i.reviewLevel==='high'||qualityFilter==='holes'&&(i.qualityFlags?.has_holes||Number(i.maskQuality?.hole_count)>0)||qualityFilter==='fragments'&&(i.qualityFlags?.has_fragments||Number(i.maskQuality?.small_fragment_count)>0)||qualityFilter==='corrected'&&i.hasApprovedCorrection;}
  function filtersHtml(){const scales=[...new Set((region()?.items||[]).map(i=>i.scale).filter(Boolean))];return `<label>筛选 <select id="consoleQuality"><option value="all">全部样本</option><option value="unreviewed">未复核</option><option value="reviewed">已复核</option>${isSeg()?'':'<option value="high">高风险标记</option><option value="holes">空洞线索</option><option value="fragments">碎片线索</option>'}<option value="corrected">已审批修正</option></select></label>${scales.length?`<label>规模 <select id="consoleScale"><option value="all">全部规模</option>${scales.map(s=>`<option value="${esc(s)}">${esc(s)}</option>`).join('')}</select></label>`:''}${controller.context.capabilities?.confidence?'<label>最低参考置信度 <input id="consoleConfidence" type="number" min="0" max="100" value="'+confidenceMin+'" style="width:70px">%</label>':''}`;}
  function filteredItems(){return scopedItems();}
  function bindQuality(){
    if($('consoleQuality')){$('consoleQuality').value=qualityFilter;$('consoleQuality').onchange=e=>{qualityFilter=e.target.value;galleryPage=0;reportState=null;render();};}
    if($('consoleScale')){$('consoleScale').value=scaleFilter;$('consoleScale').onchange=e=>{scaleFilter=e.target.value;galleryPage=0;reportState=null;render();};}
    if($('consoleConfidence'))$('consoleConfidence').onchange=e=>{confidenceMin=Math.max(0,Math.min(100,Number(e.target.value)||0));galleryPage=0;reportState=null;render();};
  }
  function renderImages(pane){
    const items=filteredItems(),pageSize=24,max=Math.max(1,Math.ceil(items.length/pageSize));galleryPage=Math.min(galleryPage,max-1);
    if(pane==='workPane'){
      $(pane).innerHTML=`<section class="panel console-section"><div class="panel-head"><h2>${isSeg()?'建筑分割影像总览':'双时相影像总览'} · ${items.length} ${isSeg()?'张':'组'}</h2>${filtersHtml()}<span>第 ${galleryPage+1} / ${max} 页</span><button class="btn ghost" id="galleryPrev">上一页</button><button class="btn ghost" id="galleryNext">下一页</button></div><div class="console-gallery">${items.slice(galleryPage*pageSize,(galleryPage+1)*pageSize).map(item=>`<article class="console-gallery-card"><h3>${esc(item.stem||item.id)}</h3>${images(item)}${factsHtml(item)}<button class="btn ghost" data-detail-id="${esc(item.id)}">查看事实与证据</button><button class="btn primary" data-edit-id="${esc(item.id)}">复核与精修</button></article>`).join('')}</div></section>`;
      $('galleryPrev').disabled=galleryPage===0;$('galleryNext').disabled=galleryPage>=max-1;
      $('galleryPrev').onclick=()=>{galleryPage--;render();};$('galleryNext').onclick=()=>{galleryPage++;render();};
      document.querySelectorAll('[data-detail-id],[data-edit-id]').forEach(b=>b.onclick=()=>{legacy.state.selectedId=b.dataset.detailId||b.dataset.editId;legacy.state.tab=b.dataset.editId?'review':'trust';selectedReview=null;geometryDraft=null;render();});bindQuality();return;
    }
    const item=items.find(i=>i.id===legacy.state.selectedId)||items[0];if(item)legacy.state.selectedId=item.id;
    $(pane).innerHTML=`<div class="console-two-column">${sampleList(items,item)}<section class="panel console-section"><h2>${esc(item?.stem||'没有样本')}</h2>${factsHtml(item)}${images(item)}${buildingEvidence(item)}${item?`<h3>检测依据与质量线索</h3><pre class="console-facts">${esc(JSON.stringify({source:item.metricSource||'原始预测',bbox:item.mainBbox,objects:item.objects,quality:item.maskQuality,reasons:item.reviewReasonCodes,evidence:item.evidenceSummary,publicFacts:item.publicFacts,checks:item.verifierChecks,report:item.report,originalPrediction:item.originalPrediction},null,2))}</pre><button class="btn primary" id="detailReview">复核与精修</button>`:''}</section></div>`;bindSamples();if($('detailReview'))$('detailReview').onclick=()=>{legacy.state.tab='review';render();};
  }
  function renderChat(){
    const context=controller.context;
    $('agentPane').innerHTML=`<section class="panel console-section console-chat"><div class="panel-head"><h2>智能问答</h2><button class="btn ghost" id="consoleNewChat">新会话</button><button class="btn ghost" id="consoleHistory">载入本会话历史</button></div><label>查询范围（明确选择后生效） <select id="consoleQueryScope"><option value="">整个当前批次</option>${context.app.regions.map(r=>`<option value="${esc(r.id)}">仅 ${esc(r.short||r.id)}</option>`).join('')}</select></label><p class="console-meta">可直接聊天；需要统计时才执行只读 SQL。影像区域选择不会自动限制查询范围。</p><div id="consoleChatLog" class="console-chat-log">${chats.map(row=>`<article class="console-message ${row.role}"><strong>${row.role==='user'?'你':'本地助手'}</strong>${row.result?A.renderReply(row.result):`<div class="console-answer">${esc(row.text)}</div>`}</article>`).join('')||'<p class="console-note">可以说“你好”，或问“各区域有多少样本？”</p>'}</div><form id="consoleChatForm" class="chat-compose"><textarea id="consoleQuestion" required maxlength="4000" placeholder="聊天、查询或继续追问…"></textarea><button class="btn primary" id="consoleSend">发送</button></form><p id="consoleChatState" class="console-meta">${chatBusy?'正在处理…':''}</p></section>`;
    $('consoleQueryScope').value=queryScope;$('consoleQueryScope').disabled=chatBusy;$('consoleQueryScope').onchange=e=>{queryScope=e.target.value;};
    $('consoleSend').disabled=chatBusy||context.capabilities?.questions===false;
    $('consoleNewChat').disabled=chatBusy;$('consoleHistory').disabled=chatBusy;
    $('consoleNewChat').onclick=()=>{controller.resetSession();chats=[];clarification=null;renderChat();};
    $('consoleHistory').onclick=loadHistory;
    $('consoleChatForm').onsubmit=event=>{event.preventDefault();const text=$('consoleQuestion').value.trim();if(text)ask(text);};
    $('consoleQuestion').onkeydown=event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing){event.preventDefault();$('consoleChatForm').requestSubmit();}};
    $('consoleChatLog').querySelectorAll('[data-choice-id]').forEach(button=>{
      button.disabled=chatBusy||button.dataset.clarificationId!==clarification?.id;
      button.onclick=()=>ask(button.textContent,{clarificationId:button.dataset.clarificationId,optionId:button.dataset.choiceId});
    });
    $('consoleChatLog').scrollTop=$('consoleChatLog').scrollHeight;
  }
  async function loadHistory(){
    if(chatBusy)return;const token=controller.capture();chatBusy=true;renderChat();
    try{const result=await controller.history(queryScope);if(!result||!controller.isCurrent(token))return;
      chats=(result.messages||[]).map(row=>({role:row.role,text:row.content||'',result:row.role==='assistant'?{answer:row.content||'',usedSQL:row.usedSQL,questionRoute:row.usedSQL?'sql':'chat'}:null}));
      clarification=result.clarification||null;if(clarification)chats.push({role:'assistant',result:{answer:clarification.question,questionRoute:'sql',usedSQL:false,clarification}});
    }catch(error){if(controller.isCurrent(token))chats.push({role:'assistant',text:error.message});}
    finally{if(controller.isCurrent(token)){chatBusy=false;if(legacy.state.tab==='agent')renderChat();}}
  }
  async function ask(text,extra={}){
    if(chatBusy)return;const token=controller.capture();chatBusy=true;chats.push({role:'user',text});renderChat();
    // Supply visible context separately; the server binds only explicit demonstratives.
    try{const result=await controller.question(text,queryScope,{...extra,viewRegion:region()?.id||null,currentSampleId:legacy.state.selectedId||null});if(!result||!controller.isCurrent(token))return;
      chats.push({role:'assistant',result});
      if(result.questionRoute==='sql'||result.clarification)clarification=result.clarification||null;
    }catch(error){if(controller.isCurrent(token))chats.push({role:'assistant',text:error.message});}
    finally{if(controller.isCurrent(token)){chatBusy=false;if(legacy.state.tab==='agent')renderChat();}}
  }
  function renderReview(){
    const context=controller.context,items=filteredItems(),item=items.find(i=>i.id===legacy.state.selectedId)||items[0];if(item)legacy.state.selectedId=item.id;
    const record=selectedReview?.sampleId===item?.id?selectedReview.record:null,readonly=!A.canReview(context);
    $('reviewPane').innerHTML=`<div class="console-two-column">${sampleList(items,item)}<section class="panel console-section"><div class="panel-head"><h2>样本复核与人工精修 · ${esc(item?.stem||'没有样本')}</h2>${filtersHtml()}</div>${factsHtml(item)}${images(item)}${item?`<form id="consoleReviewForm"><label>结论<select id="consoleVerdict"><option value="unreviewed">草稿／未复核</option>${isSeg()?'<option value="accepted_mask">分割可接受</option><option value="needs_correction">需要修正</option>':'<option value="confirmed_change">确认变化</option><option value="false_change">伪变化</option>'}<option value="uncertain">暂不确定</option></select></label><label>复核备注<textarea id="consoleReviewNote" maxlength="2000"></textarea></label><div class="manual-workbench" id="manualEditor"></div><div class="bbox-actions"><button type="button" class="btn ghost" id="saveGeometryDraft">保存绘制草稿</button><button type="button" class="btn primary" id="approveGeometry">审批修正并生效</button><button class="btn primary" id="consoleSaveReview">仅保存结论</button><button type="button" class="btn ghost" id="restorePrediction">恢复原预测</button><button type="button" class="btn ghost" id="copyGeometry">复制修正</button><button type="button" class="btn ghost" id="exportGeometry">导出修订历史</button><button type="button" class="btn ghost" id="consoleReloadReview">载入最新版本（保留输入）</button><button type="button" class="btn ghost" id="discardLocalReview">放弃本地输入</button></div><p id="consoleReviewState" role="status">${record?'记录版本 '+record.version+' · 草稿不影响统计，审批后生效':'正在读取服务器复核记录…'}</p></form><details><summary>复核与修正历史</summary><pre>${esc(JSON.stringify(selectedReview?.history||[],null,2))}</pre></details>`:''}</section></div>`;
    bindSamples();bindQuality();if(!item)return;
    $('consoleVerdict').value=reviewDraft?.sampleId===item.id?reviewDraft.verdict:record?.verdict||'unreviewed';$('consoleReviewNote').value=reviewDraft?.sampleId===item.id?reviewDraft.note:record?.note||'';
    for(const id of ['consoleSaveReview','saveGeometryDraft','approveGeometry','restorePrediction'])$(id).disabled=readonly||!record||reviewBusy;
    $('consoleVerdict').disabled=readonly||!record;$('consoleReviewNote').readOnly=readonly||!record;
    if(record)editor=ManualEditor.mount($('manualEditor'),{item,record,readonly,initial:geometryDraft?.sampleId===item.id?geometryDraft.geometry:null,initialActive:geometryDraft?.sampleId===item.id?geometryDraft.active:[],onChange:(geometry,active)=>{geometryDraft={sampleId:item.id,geometry,active,points:active.length};}});
    const rememberInput=()=>{reviewDraft={sampleId:item.id,verdict:$('consoleVerdict').value,note:$('consoleReviewNote').value};};$('consoleVerdict').onchange=rememberInput;$('consoleReviewNote').oninput=rememberInput;
    $('consoleReloadReview').onclick=()=>{rememberInput();selectedReview=null;loadReview(item);};$('discardLocalReview').onclick=()=>{geometryDraft=null;reviewDraft=null;selectedReview=null;loadReview(item);};
    $('consoleReviewForm').onsubmit=e=>{e.preventDefault();saveReview(item);};
    $('saveGeometryDraft').onclick=()=>saveReview(item,'draft');$('approveGeometry').onclick=()=>saveReview(item,'approved');$('restorePrediction').onclick=()=>saveReview(item,'restore');
    $('copyGeometry').onclick=()=>navigator.clipboard?.writeText(JSON.stringify({sampleId:item.id,geometry:editor?.geometry()},null,2));
    $('exportGeometry').onclick=async()=>{try{const value=await controller.corrections();downloadJson(value,'corrections-'+context.batchId+'.json');}catch(e){$('consoleReviewState').textContent=e.message;}};
    if(!record&&!selectedReview?.loading)loadReview(item);
  }
  function downloadJson(value,name){const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  async function loadReview(item){
    const token=controller.capture(),load=++reviewLoad;selectedReview={sampleId:item.id,loading:true};
    try{const value=await controller.review(item.id);if(!value||!controller.isCurrent(token)||load!==reviewLoad||legacy.state.selectedId!==item.id)return;selectedReview={sampleId:item.id,...value};renderReview();}
    catch(error){if(controller.isCurrent(token)&&load===reviewLoad){selectedReview={sampleId:item.id,failed:true};if($('consoleReviewState'))$('consoleReviewState').textContent=error.message;}}
  }
  async function saveReview(item,geometryAction){
    if(reviewBusy||!selectedReview?.record||!A.canReview(controller.context))return;
    if(geometryAction&&geometryAction!=='restore'&&geometryDraft?.points){$('consoleReviewState').textContent='请先按 Enter 完成当前轮廓。';return;}
    if(geometryAction==='approved'&&$('consoleVerdict').value==='unreviewed'){$('consoleReviewState').textContent='审批前请选择复核结论。';return;}
    const token=controller.capture(),expectedVersion=selectedReview.record.version,batchId=controller.context.batchId;
    reviewDraft={sampleId:item.id,verdict:$('consoleVerdict').value,note:$('consoleReviewNote').value};reviewBusy=true;
    for(const id of ['consoleSaveReview','saveGeometryDraft','approveGeometry','restorePrediction'])$(id).disabled=true;$('consoleVerdict').disabled=true;$('consoleReviewNote').readOnly=true;$('consoleReloadReview').disabled=true;
    try{const value=await controller.saveReview(item.id,expectedVersion,reviewDraft.verdict,reviewDraft.note,geometryAction?{correction:geometryAction==='restore'?null:{status:geometryAction,geometry:editor.geometry()}}:{});if(!value||!controller.isCurrent(token))return;
      const view={regionId:legacy.state.regionId,tab:legacy.state.tab,selectedId:legacy.state.selectedId};await choose(batchId);if(controller.context?.batchId!==batchId)return;
      Object.assign(legacy.state,view);render();status('复核已保存；报告和后续查询使用新的复核版本。');
    }catch(error){if(controller.isCurrent(token)&&$('consoleReviewState'))$('consoleReviewState').textContent=error.status===409?error.message+'；输入已保留，请载入最新版本后核对再提交。':error.message;}
    finally{if(controller.isCurrent(token)){reviewBusy=false;if($('consoleSaveReview')){for(const id of ['consoleSaveReview','saveGeometryDraft','approveGeometry','restorePrediction'])$(id).disabled=!A.canReview(controller.context);$('consoleVerdict').disabled=!A.canReview(controller.context);$('consoleReviewNote').readOnly=!A.canReview(controller.context);$('consoleReloadReview').disabled=false;}}}
  }
  function reportKey(){return JSON.stringify([controller.context?.batchId,controller.context?.dataVersion,controller.context?.reviewVersion,region()?.id,scopedItems().map(i=>i.id)]);}
  function reportFactsHtml(){
    const facts=reportState?.value.report;if(!facts)return '';
    if(isSeg())return `<div class="overview-cards">${metric('范围影像',facts.totalCount)}${metric('含建筑影像',facts.positiveImages)}${metric('建筑量',facts.estimatedBuildingCountSum,'候选区域估计数')}${metric('已过滤小块',facts.filteredFragmentCountSum)}${metric('触及边界的候选',facts.boundaryBuildingCountSum)}${metric('已审批修正',facts.correctedSamples)}</div>`;
    const total=Number(facts.totalCount??facts.sampleCount??0);
    const card=(label,value)=>`<div class="report-stat"><span>${esc(label)}</span><b>${value==null?'未知':Number(value).toLocaleString()}</b></div>`;
    const risks=Array.isArray(facts.riskBuckets)?facts.riskBuckets:[];
    const events=Array.isArray(facts.eventBuckets)?facts.eventBuckets:[];
    return `<div class="report-sections"><section class="report-section"><h3>人工修正与分布</h3><p>当前口径：${esc(controller.context.resultBasis==='original'?'原始预测':'人工修正后')} · 已审批 ${Number(facts.correctedSamples||0)} 组</p><details><summary>修订记录、规模与质量线索</summary><pre class="console-facts">${esc(JSON.stringify({corrections:facts.correctionEntries,scales:facts.scaleCounts,reasons:facts.reasonCounts},null,2))}</pre></details></section><section class="report-section"><h3>范围与复核统计</h3><div class="report-card-grid">${card('范围样本',facts.totalCount??facts.sampleCount)}${card('变化样本',facts.changedCount)}${card('已记录复核',facts.reviewedCount)}${card('风险未知',facts.unknownRiskCount)}${card('已审批修正',facts.correctedSamples)}</div></section><section class="report-section"><h3>事件与风险分布</h3>${events.length?legacy.reportMeter(events,total,'事件分类未知'):'<p class="console-meta">当前没有可用的事件分类，保持未知。</p>'}${risks.length?legacy.reportMeter(risks,total,'风险分级未知'):'<p class="console-meta">当前没有已校准的风险分级，保持未知。</p>'}</section><section class="report-section"><h3>代表样本</h3><div class="report-list">${(facts.representatives||[]).map(item=>`<button class="report-pick console-report-pick" data-report-sample="${esc(item.id)}"><b>${esc(item.stem||item.id)}</b><span>${Number(item.changedPx||0).toLocaleString()} px · 连通对象 ${esc(item.objectCount??'未知')}</span></button>`).join('')||'<p class="console-meta">当前范围没有代表样本。</p>'}</div></section></div>`;
  }
  function renderReport(){
    const key=reportKey();if(reportState?.key!==key)reportState=null;
    $('reportPane').innerHTML=`<div class="report-shell"><aside class="report-nav"><section class="panel console-section"><h3>报告范围</h3><p>${esc(scopeLabel())}</p><p>${scopedItems().length} 张样本</p><p class="console-meta">服务端重新统计所选样本；不采信浏览器发送的计数。</p></section>${provenance(controller.context)}</aside><section class="panel report-main"><div class="panel-head"><div class="panel-title">巡检统计报告</div><button class="btn primary" id="runRegionReportBtn">更新统计摘要</button></div><div class="console-section"><p class="console-meta">${reportBusy?'正在重新统计…':'统计摘要由服务器生成；模型分析简报请使用分析任务。'}</p><div class="report-summary-box">${esc(reportState?.value.text||'尚未取得当前范围的服务端摘要。')}</div>${reportState?`<details><summary>服务端报告事实与版本</summary><pre>${esc(JSON.stringify(reportState.value.report||{},null,2))}</pre></details>`:''}<p id="consoleReportError" class="error"></p><div class="bbox-actions"><button class="btn ghost" id="copyRegionReportBtn">复制摘要</button><button class="btn primary" id="downloadWordReportBtn">下载 Word 兼容 .doc</button></div><p class="console-meta">导出含当前范围、数据版本、复核版本及单位。文件是 HTML 格式的 .doc。</p></div></section></div>`;
    $('consoleReportError').insertAdjacentHTML('beforebegin',reportFactsHtml());
    document.querySelectorAll('[data-report-sample]').forEach(button=>button.onclick=()=>{
      const id=button.dataset.reportSample,owner=controller.context.app.regions.find(r=>r.items.some(i=>i.id===id));
      if(owner){selectRegion(owner.id);legacy.state.selectedId=id;legacy.state.tab='interpret';render();}
    });
    $('consoleReportError').textContent=reportError;
    $('runRegionReportBtn').disabled=reportBusy;$('downloadWordReportBtn').disabled=reportBusy;$('copyRegionReportBtn').disabled=!reportState||reportBusy;
    $('runRegionReportBtn').onclick=()=>refreshReport();
    $('copyRegionReportBtn').onclick=()=>{if(reportState?.key===reportKey())navigator.clipboard?.writeText(A.reportText(reportState.value,controller.context,scopeLabel()));};
    $('downloadWordReportBtn').onclick=async()=>{if(await refreshReport()){try{legacy.downloadWord();}catch(error){status(error.message,true);}}};
    if(!reportBusy&&reportAttempted!==key){reportAttempted=key;refreshReport();}
  }
  async function refreshReport(){
    if(reportBusy)return false;const token=controller.capture(),key=reportKey(),items=scopedItems();reportBusy=true;reportAttempted=key;reportError='';renderReport();
    try{const value=await controller.summary(region().id,items.map(i=>i.id));if(!value||!controller.isCurrent(token)||key!==reportKey())return false;
      reportState={key,value};return true;
    }catch(error){if(controller.isCurrent(token))reportError=error.message;return false;}
    finally{if(controller.isCurrent(token)){reportBusy=false;if(legacy.state.tab==='report')renderReport();}}
  }
  function setupAnalysisContext(context){
    if(!isSeg()){
      const examples=[['WHU：变化量是否集中在少数样本？','分析WHU的变化像素分布：统计有效样本数量、均值、最小值与最大值；计算变化像素最多的前10张占本区域变化像素总量的百分比。'],['各区域：还有多少未复核样本？','按区域比较已复核和未复核样本的数量及各自占本区域的比例；列出每个区域变化像素最大的3个未复核样本。'],['LEVIR：按变化比例分组分析','分析LEVIR的变化比例分布，统计变化比例为0、大于0且不超过0.1、大于0.1且不超过0.5、大于0.5的样本数量。']];
      document.querySelectorAll('[data-goal]').forEach((button,i)=>{if(examples[i]){button.textContent=examples[i][0];button.dataset.goal=examples[i][1];}});
      $('analysisGoal').placeholder='例如：分析WHU变化量分布，比较已复核与未复核样本。';
      const h=document.querySelector?.('main>header h1');if(h)h.textContent='变化检测结果分析助手';
    }
    if(isSeg()){
      const examples=[['建筑量与覆盖统计','统计本批建筑量、被过滤的小块数及触及图像边界的候选数量；统计建筑像素占比均值。'],['建筑覆盖分组','按建筑像素占比为0、大于0且不超过0.1、大于0.1且不超过0.5、大于0.5分别统计图片数量。'],['未复核影像','统计各区域已复核与未复核的图片数量；列出建筑像素最多的3张未复核图片。']];
      document.querySelectorAll('[data-goal]').forEach((button,i)=>{if(examples[i]){button.textContent=examples[i][0];button.dataset.goal=examples[i][1];}});
      $('analysisGoal').placeholder='例如：分析本批建筑像素分布，计算前10张的贡献占比。';
      const h=document.querySelector('main>header h1');if(h)h.textContent='建筑分割结果分析助手';
    }
    $('analysisScope').innerHTML='<option value="">整个当前批次</option>'+context.app.regions.map(r=>`<option value="${esc(r.id)}">${esc(r.short||r.id)}</option>`).join('');
    $('modelState').textContent='当前批次：'+context.label+' · 复核版本 '+context.reviewVersion;
    $('analysisResult').innerHTML='<p class="console-note">先描述分析目标，再核对方案；执行结果和简报只使用当前批次数据。</p>';
    $('analysisError').textContent='';
  }
  function analysisSetBusy(value,message){
    analysisBusy=value;document.querySelectorAll('#analysisForm button,#analysisResult button').forEach(button=>button.disabled=value);$('analysisScope').disabled=value;
    $('modelState').textContent=message||'当前批次：'+(controller.context?.label||'未选择');
  }
  async function runAnalysis(payload){
    if(analysisBusy||!controller.context)return;const token=controller.capture();analysisSetBusy(true,payload.action==='plan'?'正在生成分析方案…':'正在执行分析并核对证据…');$('analysisError').textContent='';
    try{const value=await controller.analysis(payload);if(!value||!controller.isCurrent(token))return;currentPlan=value;$('analysisResult').innerHTML=window.AnalysisTasks.render(value);}
    catch(error){if(controller.isCurrent(token))$('analysisError').textContent=error.message;}
    finally{if(controller.isCurrent(token))analysisSetBusy(false);}
  }
  function bindAnalysis(){
    $('analysisForm').onsubmit=event=>{event.preventDefault();runAnalysis({action:'plan',goal:$('analysisGoal').value.trim(),scopeRegion:$('analysisScope').value||null});};
    document.querySelectorAll('[data-goal]').forEach(button=>button.onclick=()=>{$('analysisGoal').value=button.dataset.goal;$('analysisGoal').focus();});
    $('analysisScope').onchange=()=>{currentPlan=null;$('analysisResult').innerHTML='<p class="console-note">分析范围已改变，请重新生成方案。</p>';};
    $('analysisResult').onclick=event=>{
      const button=event.target.closest('button');if(!button||analysisBusy)return;
      if(button.dataset.action==='execute'&&currentPlan?.planId)runAnalysis({action:'execute',planId:currentPlan.planId,scopeRegion:$('analysisScope').value||null});
      else if(button.dataset.analysisGoal){$('analysisGoal').value=button.dataset.analysisGoal;runAnalysis({action:'plan',goal:button.dataset.analysisGoal,scopeRegion:$('analysisScope').value||null});}
      else if(button.dataset.question){$('analysisGoal').value=button.dataset.question;$('analysisGoal').focus();}
      else if(button.dataset.action==='export'&&currentPlan){
        const link=document.createElement('a'),url=URL.createObjectURL(new Blob([JSON.stringify({batchId:controller.context.batchId,dataVersion:controller.context.dataVersion,reviewVersion:controller.context.reviewVersion,analysis:currentPlan},null,2)],{type:'application/json;charset=utf-8'}));
        link.href=url;link.download='analysis-'+currentPlan.planId+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
      }
    };
  }
  async function start(){
    try{
      fitChrome();if(typeof ResizeObserver!=='undefined'){const observer=new ResizeObserver(fitChrome);for(const node of [document.querySelector('.console-header'),$('consoleStatus')])if(node)observer.observe(node);}
      csrf=await request('/auth/csrf');actor=await request('/auth/me');$('consoleActor').textContent=actor.displayName||actor.username;
      $('consoleReload').onclick=()=>choose(controller.context?.batchId||$('consoleBatch').value,true);
      $('consoleLogout').onclick=async()=>{try{await request('/auth/logout',{method:'POST'});}finally{actor=null;login();}};
      if(legacy){
        legacy.install({render});
        legacy.setWordProvider({snapshot:()=>{if(!reportState||reportState.key!==reportKey())throw new Error('报告已过期，请重新生成');return {filterText:scopeLabel(),generatedAt:new Date().toISOString()};},text:()=>A.reportText(reportState.value,controller.context,scopeLabel()),scope:scopeLabel});
        $('regionScope').onclick=()=>$('regionSwitcher').classList.toggle('open');
        $('classFilter').onchange=event=>{legacy.state.classFilter=event.target.value;reportState=null;render();};
        $('searchInput').oninput=event=>{legacy.state.search=event.target.value;legacy.state.selectedId=null;visibleLimit=100;reportState=null;render();};
        $('sortMode').onchange=event=>{legacy.state.sortMode=event.target.value;render();};
        document.querySelectorAll('.tab').forEach(tab=>tab.onclick=()=>{legacy.state.tab=tab.dataset.tab;render();});
      }else bindAnalysis();
      const list=await request('/batches');
      $('consoleBatch').replaceChildren();
      for(const batch of list.batches||[]){const option=document.createElement('option');option.value=batch.batchId;option.textContent=batch.label.replace('（只读）','')+(batch.sourceType==='ARCHIVE'?' · 原全量样本库':batch.taskType==='building_segmentation'?' · 建筑分割':' · 变化检测')+' · '+batch.total+(batch.taskType==='building_segmentation'?'张':'组')+(batch.status!=='PUBLISHED'?' · 尚未发布':'');option.disabled=batch.status!=='PUBLISHED';$('consoleBatch').append(option);}
      $('consoleBatch').onchange=()=>choose($('consoleBatch').value);
      const requested=new URL(location.href).searchParams.get('batch'),available=(list.batches||[]).filter(b=>b.status==='PUBLISHED');
      resultBasis=new URL(location.href).searchParams.get('basis')==='original'?'original':'effective';
      const basisLabel=document.createElement('label');basisLabel.innerHTML='<span>结果口径 </span><select id="consoleResultBasis"><option value="effective">人工修正后</option><option value="original">原始预测</option></select>';document.querySelector('.console-header').appendChild(basisLabel);$('consoleResultBasis').value=resultBasis;$('consoleResultBasis').onchange=async e=>{resultBasis=e.target.value;await choose(controller.context?.batchId,true);};
      const initial=requested?available.find(b=>b.batchId===requested):(available.find(b=>b.sourceType==='ARCHIVE')||available[0]);
      if(initial){await choose(initial.batchId);const tab=new URL(location.href).searchParams.get('tab');if(legacy&&['overview','interpret','trust','agent','review','report'].includes(tab)){legacy.state.tab=tab;$('workspaceOverlay').classList.add('open');document.body.classList.add('workspace-active');render();}}
      else status(requested?'指定批次不可访问或尚未发布，请从批次列表重新选择。':'暂无已发布的授权批次，请到“账号与批次”创建任务或等待预测完成。');
    }catch(error){if(error.status!==401)status(error.message,true);}
  }
  start();
})();

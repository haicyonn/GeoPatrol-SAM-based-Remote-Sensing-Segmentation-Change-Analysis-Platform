(function(root,factory){
  const api=factory();
  if(typeof module==='object'&&module.exports)module.exports=api;
  else{root.AnalysisTasks=api;}
})(typeof globalThis!=='undefined'?globalThis:this,function(){
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const anchor=id=>'evidence-'+String(id).replace(/[^A-Za-z0-9_-]/g,'-');
  const columns={region_id:'区域',sample_id:'样本',sample_count:'样本数',total_samples:'有效样本数',n:'数量',mean:'均值',mean_changed_px:'平均变化像素',min_changed_px:'最小变化像素',max_changed_px:'最大变化像素',avg_changed_px:'平均变化像素',total_changed_px:'变化像素合计',reviewed:'已复核标记',share_pct:'占比（%）',top_10_percentage:'前10张贡献占比（%）'};
  Object.assign(columns,{sample_name:'图片',foreground_ratio:'建筑覆盖比例（0—1）',foreground_pixels:'建筑像素',estimated_building_count:'建筑量（估计）',component_count:'原始连通区域数'});
  function render(data){
    if(data.status==='needs_clarification')return `<div class="notice">${esc(data.answer)}</div>${data.originalGoal?`<p class="hint">原问题：${esc(data.originalGoal)}</p>`:''}${(data.options||[]).filter(o=>o.goal).map(o=>`<button class="secondary" data-analysis-goal="${esc(o.goal)}">${esc(o.label)}</button><p class="hint">确认后分析：${esc(o.goal)}</p>`).join('')}`;
    const planned=data.status==='planned';
    return `<div class="result-heading"><div><p class="eyebrow">${planned?'方案预览':'分析结果'}</p><h2>${esc(data.title)}</h2></div><span class="badge">${planned?'等待执行':data.status==='completed'?'查询已完成':'部分查询未完成'}</span></div>
      <p class="task-goal">${esc(data.goal)}</p><div class="meta">${data.targetRegion?'当前对象：'+esc(data.targetRegion)+' · ':''}查询范围：${esc(data.scopeRegion||'本批次全部区域')} · 数据版本 ${esc((data.dataVersion||'').slice(0,12))}</div>
      ${data.sampleRank?`<p class="notice">排名对象：图片 · 指标：${esc(columns[data.sampleRank.field]||data.sampleRank.field)} · ${data.sampleRank.direction==='ASC'?'升序':'降序'}第${esc(data.sampleRank.start)}张；同值按样本ID升序。</p>`:''}
      ${data.scopeAdjustment?`<p class="notice">${esc(data.scopeAdjustment)}</p>`:''}
      ${planned?'<p class="hint">先核对下面每一步的问题和统计口径。需要调整时，修改左侧目标并重新生成方案。</p><button class="primary" data-action="execute">执行分析</button>':''}
      ${data.report?`<section class="brief"><h3>分析简报</h3><p class="hint">${esc(data.report.message||'')}</p>${(data.report.unansweredSteps||[]).map(s=>`<p class="error">尚未回答：${esc(s.question)}。${esc(s.reason)}</p>`).join('')}${data.reportEvidenceIds?.length<data.evidence?.length?'<p class="hint">简报只参考每步的部分结果行；完整查询结果保留在下方表格中。</p>':''}${(data.report.findings||[]).map(f=>`<p class="finding">${esc(f.text)} <span>${(f.evidence||[]).map(id=>`<a href="#${anchor(id)}">[${esc(id)}]</a>`).join(' ')}</span></p>`).join('')}${data.report.status==='rejected'?`<details><summary>查看未采用结论的原因</summary><p>${esc(data.report.error||'')}</p></details>`:''}</section>`:''}
      <div class="steps">${(data.steps||[]).map((step,i)=>`<section class="step"><div class="step-top"><span class="step-number">${i+1}</span><h3>${esc(step.title)}</h3><span class="step-status">${planned?'待执行':step.status==='completed'?'已执行':step.status==='no_data'?'已执行，结果不足':'未完成'}</span></div><p>${esc(step.question)}</p>
        ${step.error&&step.status!=='completed'?`<p class="error">${esc(step.error)}</p>`:''}
        ${step.rows?`<div class="table-wrap"><table><thead><tr><th>证据</th>${(step.columns||[]).map(c=>`<th>${esc(columns[c]||c)}</th>`).join('')}</tr></thead><tbody>${step.rows.map((row,n)=>`<tr id="${anchor(step.id+':R'+(n+1))}"><td><code>${esc(step.id+':R'+(n+1))}</code></td>${(step.columns||[]).map(c=>`<td>${row[c]==null?'<span class="muted">未知</span>':esc(row[c])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>${!step.rows.length?'<p class="hint">本步骤没有返回记录。</p>':''}${step.truncated?'<p class="notice">结果超过50行，表格已截断；不能把展示的行数当成总体数量。</p>':''}`:''}
        ${step.programChanges?.includes('stable_sample_id_tie_break')?'<p class="hint">程序补充同值按样本ID升序，保留模型的指标、筛选与名次。</p>':''}<details><summary>${planned?'查看模型拟定的 SQL':'查看实际 SQL 与执行记录'}</summary>${step.modelGeneratedSQL?`<p class="hint">模型原始SQL</p><pre>${esc(step.modelGeneratedSQL)}</pre><p class="hint">补充稳定排序后的SQL</p>`:''}<pre>${esc(step.sql)}</pre>${(step.attempts||[]).map(a=>`<div class="attempt"><p>${a.status==='executed'?'执行成功':'未通过检查'}${a.error?'：'+esc(a.error):''}</p><pre>${esc(a.sql)}</pre></div>`).join('')}</details></section>`).join('')}</div>
      ${data.report?.nextQuestions?.length?`<section><h3>可以继续分析</h3><div class="followups">${data.report.nextQuestions.map(q=>`<button class="secondary" data-question="${esc(q)}">${esc(q)}</button>`).join('')}</div></section>`:''}
      <div class="limits">${(data.limits||[]).map(s=>`<p>${esc(s)}</p>`).join('')}</div>
      ${!planned?'<button class="secondary" data-action="export">导出分析证据</button>':''}
      <details><summary>模型调用记录 · ${Number(data.llmCalls||0)} 次</summary><p class="hint">${esc(data.modelId)}</p><pre>${esc(JSON.stringify(data.modelUsage||[],null,2))}</pre></details>`;
  }
  return {render};
});

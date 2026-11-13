(function(root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.InspectionEvidence = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function() {
  'use strict';
  const esc = value => String(value == null ? '' : value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const numberIn = (text, value) => value == null || new RegExp('(^|[^0-9])' + String(Number(value)).replace('.', '\\.') + '([^0-9]|$)').test(String(text).replace(/,/g,''));

  function normalizedSample(row) {
    return {id:row.sample_id || row.id, stem:row.sample_stem || row.stem,
      changedPx:row.changed_px ?? row.changedPx, changedRatio:row.changed_ratio ?? row.changedRatio,
      objectCount:row.object_count ?? row.objectCount, gridPosition:row.grid_position ?? row.gridPosition,
      reviewRequired:Boolean(row.review_required ?? row.reviewRequired), mainBbox:row.mainBbox || []};
  }

  function guardAnswer(data, currentItem, legacyGuard) {
    const answer = String(data.answer || '');
    const scope = data.answerScope;
    if (scope === 'knowledge' || scope === 'region' || scope === 'review_plan' || scope === 'review_reference') return answer;
    const rows = Array.isArray(data.evidenceSamples) ? data.evidenceSamples : [];
    if (scope === 'list' || scope === 'mixed') {
      const failed = rows.filter(row => {
        const id = String(row.sample_id || row.sample_stem || '');
        const start = id ? answer.indexOf(id) : -1;
        if (start < 0) return true;
        const nextIds = rows.filter(other => other !== row).map(other => answer.indexOf(String(other.sample_id || other.sample_stem || ''), start + id.length)).filter(pos => pos > start);
        const citationStart = answer.indexOf('\n[', start);
        const end = Math.min(answer.length, ...nextIds, ...(citationStart > start ? [citationStart] : []));
        const block = answer.slice(start, end);
        return !numberIn(block, row.changed_px) || !numberIn(block, row.object_count);
      });
      return failed.length ? answer + '\n\n核对提示：部分样本数值未能与返回记录对应，请查看样本数据。' : answer;
    }
    if (scope === 'sample' && rows.length) return legacyGuard(normalizedSample(rows[0]), answer);
    return legacyGuard(currentItem, answer);
  }

  function citationsHtml(citations) {
    if (!Array.isArray(citations) || !citations.length) return '';
    return '<div class="evidence-citations">' + citations.map(c =>
      `<details class="evidence-source"><summary>[${esc(c.id)}] ${esc(c.title)} · ${esc(c.section)}</summary><p>${esc(c.excerpt)}</p><small>来源：${esc(c.source)} · 版本：${esc(c.version)}</small></details>`).join('') + '</div>';
  }

  function samplePicks(data, items) {
    if (data.answerScope === 'knowledge') return [];
    if (!Array.isArray(data.evidenceSamples)) return null;
    return data.evidenceSamples.map(row => {
      const item = items.find(x => x.id === row.sample_id || x.stem === row.sample_stem);
      if (!item) return null;
      return {id:item.id,stem:item.stem,tab:'agent',reviewOnly:false,
        summary:`${Number(row.changed_px || 0).toLocaleString()}px · ${row.object_count || 0} 个对象${row.review_reason ? ' · '+row.review_reason : ''}`};
    }).filter(Boolean);
  }

  function detailsHtml(data) {
    if (!data) return '';
    const trace = Array.isArray(data.toolTrace) ? data.toolTrace : [];
    const labels = {query_samples:'样本查询',search_knowledge:'资料检索',explain_evidence:'依据解释',report_snapshot:'报告统计',compile_review_request:'理解复核条件',select_review_batch:'约束与覆盖选样',resolve_review_reference:'清单样本引用'};
    const modes = {extractive:'资料摘录',generated:'模型生成',model_unavailable:'模型不可用，显示资料',citation_fallback:'引用校验后显示资料',numeric_fallback:'数值校验后显示资料',no_evidence:'资料不足'};
    const source = {static_snapshot:'静态样本快照',mysql:'MySQL',knowledge:'本地知识库',report_snapshot:'当前报告快照',unavailable:'样本数据不可用'}[data.dataSource] || '';
    const meta = [source,modes[data.generationStatus] || '', data.llmCalls != null ? `模型调用 ${Number(data.llmCalls)} 次` : '',data.latencyMs != null ? `${Number(data.latencyMs)} ms` : ''].filter(Boolean).join(' · ');
    const review = typeof globalThis.ReviewPlanner !== 'undefined' ? globalThis.ReviewPlanner.render(data) : '';
    return review + citationsHtml(data.citations) + (trace.length ? `<details class="evidence-trace"><summary>处理过程${meta ? ' · '+esc(meta) : ''}</summary><ol>${trace.map(t => `<li>${esc(labels[t.tool] || t.tool)}：${esc(t.summary || t.status)}</li>`).join('')}</ol></details>` : '');
  }

  function reportSummary(summary, fingerprint, fallback) {
    if (summary && summary.status === 'done' && summary.snapshotFingerprint === fingerprint) return summary;
    return {text:fallback,citations:[],status:'local'};
  }

  function citationLines(citations) {
    if (!Array.isArray(citations) || !citations.length) return [];
    return ['', '参考资料', ...citations.map(c => `[${c.id}] ${c.title} · ${c.section}；${c.source}；版本 ${c.version}\n${c.excerpt}`)];
  }
  return {guardAnswer,citationsHtml,detailsHtml,reportSummary,citationLines,samplePicks};
});

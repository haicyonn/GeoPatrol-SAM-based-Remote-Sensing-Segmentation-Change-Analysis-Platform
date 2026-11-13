(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.ManualEditor=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
 const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const copy=v=>JSON.parse(JSON.stringify(v));
 function pixel(event,rect,size){return [Math.round(Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width))*size[0]*100)/100,Math.round(Math.max(0,Math.min(1,(event.clientY-rect.top)/rect.height))*size[1]*100)/100];}
 function create(size,initial,unfinished=[]){let polys=copy(initial?.polygons||[]),active=copy(unfinished);return {size,add(p){if(active.length>=512)throw Error('每个轮廓最多512点');active.push(p);},close(){if(active.length<3)return false;if(polys.length>=64)throw Error('最多64个轮廓');polys.push(active);active=[];return true;},undo(){if(active.length)active.pop();else polys.pop();},clear(){polys=[];active=[];},geometry(){return {type:'multi_polygon',polygons:copy(polys)};},get active(){return active;}};}
 function mount(element,{item,record,readonly,onChange,initial,initialActive}){
  const size=item.sourceSize;if(!Array.isArray(size)||size.length!==2){element.innerHTML='<p>影像尺寸未登记，无法精修。</p>';return null;}
  const state=create(size,initial||record?.correction?.geometry||record?.approvedCorrection?.geometry,initialActive||[]);
  const originalMask=item.assets.predMask||item.assets.mask;
  element.innerHTML=`<div class="manual-toolbar"><label>精修底图 <select data-base>${item.assets.source?'<option value="post">原始影像</option>':'<option value="post">T2 后时相</option><option value="pre">T1 前时相</option>'}<option value="system">系统掩膜叠加</option></select></label><button type="button" data-close>完成当前轮廓 Enter</button><button type="button" data-undo>撤销点／上一轮廓</button><button type="button" data-clear>清空目标区域</button></div><p>逐点描出目标区域，按 Enter 闭合；可绘制多块区域。坐标按原图 ${size.join(' × ')} 保存。清空并审批表示没有目标区域。</p><div class="manual-canvases"><div class="manual-stage" tabindex="0" data-stage><img data-image src="${esc((item.assets.source||item.assets.post))}" alt="人工精修底图"><img data-system class="manual-mask" src="${esc(originalMask||'')}" alt="系统预测参考" hidden><svg data-svg viewBox="0 0 ${size[0]} ${size[1]}" preserveAspectRatio="none"></svg></div><div class="manual-stage manual-preview"><img src="${esc((item.assets.source||item.assets.post))}" alt="人工修正叠加预览"><svg data-preview viewBox="0 0 ${size[0]} ${size[1]}" preserveAspectRatio="none"></svg></div></div><p data-state role="status"></p>`;
  const q=s=>element.querySelector(s),stage=q('[data-stage]');
  function draw(){
   const polygons=state.geometry().polygons.map(p=>`<polygon points="${p.map(v=>v.join(',')).join(' ')}" fill="#f3b43b66" stroke="#ffbd43" stroke-width="${Math.max(1,size[0]/300)}"/>`).join('');
   q('[data-preview]').innerHTML=polygons;q('[data-svg]').innerHTML=polygons+(state.active.length?`<polyline points="${state.active.map(v=>v.join(',')).join(' ')}" fill="none" stroke="#5de0ff" stroke-width="${Math.max(1,size[0]/300)}"/>`:'');
   q('[data-state]').textContent=`已完成 ${state.geometry().polygons.length} 块，当前 ${state.active.length} 个点${readonly?' · 只读':''}`;onChange?.(state.geometry(),copy(state.active));
  }
  stage.onclick=e=>{if(readonly)return;stage.focus();try{state.add(pixel(e,stage.getBoundingClientRect(),size));draw();}catch(err){q('[data-state]').textContent=err.message;}};
  function close(){if(readonly)return;if(!state.close()){q('[data-state]').textContent='至少三个点才能完成轮廓。';return;}draw();}
  stage.onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();close();}else if(e.key==='Backspace'){e.preventDefault();state.undo();draw();}};
  q('[data-close]').onclick=close;q('[data-undo]').onclick=()=>{state.undo();draw();};q('[data-clear]').onclick=()=>{state.clear();draw();};
  q('[data-base]').onchange=e=>{q('[data-image]').src=e.target.value==='pre'?item.assets.pre:(item.assets.source||item.assets.post);q('[data-system]').hidden=e.target.value!=='system';};
  element.querySelectorAll('button').forEach(b=>b.disabled=readonly);draw();return state;
 }
 return {pixel,create,mount};
});

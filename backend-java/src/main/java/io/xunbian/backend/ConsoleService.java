package io.xunbian.backend;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.*;
import org.springframework.core.io.ClassPathResource;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

@Service
public class ConsoleService {
    Map<String,Object> segmentationSummary(String id,String basis,AnalysisService.Snapshot snap,List<Map<String,Object>> all,String region){
        long pixels=0,components=0;double ratios=0;int positive=0,reviewed=0,corrected=0;
        for(var i:all){long p=((Number)i.get("foregroundPixels")).longValue();pixels+=p;ratios+=((Number)i.get("foregroundRatio")).doubleValue();components+=((Number)i.get("componentCount")).longValue();if(p>0)positive++;if(Boolean.TRUE.equals(i.get("manualReviewComplete")))reviewed++;if(Boolean.TRUE.equals(i.get("hasApprovedCorrection")))corrected++;}
        var report=new LinkedHashMap<String,Object>();report.put("totalCount",all.size());report.put("positiveImages",positive);report.put("foregroundPixelsSum",pixels);report.put("meanForegroundRatio",all.isEmpty()?null:ratios/all.size());report.put("componentCountSum",components);report.put("reviewedCount",reviewed);report.put("correctedSamples",corrected);report.put("regionId",region);report.put("resultBasis",basis);
        long buildings=0,fragments=0,borders=0;for(var i:all){buildings+=((Number)i.get("estimatedBuildingCount")).longValue();fragments+=((Number)i.get("filteredFragmentCount")).longValue();borders+=((Number)i.get("boundaryBuildingCount")).longValue();}
        report.put("estimatedBuildingCountSum",buildings);report.put("filteredFragmentCountSum",fragments);report.put("boundaryBuildingCountSum",borders);report.put("countMethod",BuildingCountService.METHOD);report.put("countMinAreaPixels",BuildingCountService.MIN_AREA);
        String text="当前范围共"+all.size()+"张影像，"+positive+"张预测含建筑；建筑像素合计"+pixels+"，建筑像素占比均值"+(all.isEmpty()?"无数据":String.format(Locale.ROOT,"%.4f%%",100*ratios/all.size()))+"；掩膜连通区域合计"+components+"。已有"+reviewed+"张人工复核，"+corrected+"张有已审批修正。连通区域不等于建筑栋数，像素量未换算地面面积。";
        text="建筑量合计"+buildings+"个（候选区域估计）；过滤小于"+BuildingCountService.MIN_AREA+"像素的小块"+fragments+"个，保留候选中触及图像边界的有"+borders+"个。"+text+"粘连建筑未拆分，跨图累加未作地理去重。";
        var out=new LinkedHashMap<String,Object>();out.put("status","done");out.put("taskType","building_segmentation");out.put("resultBasis",basis);out.put("text",text);out.put("answer",text);out.put("engine","server_stats");out.put("report",report);out.put("batchId",id);out.put("dataVersion",snap.digest());out.put("reviewVersion",snap.reviewVersion());out.put("predictionDigest",snap.predictionDigest());out.put("models",snap.models());out.put("generatedAt",System.currentTimeMillis());
        if(db.queryForObject("SELECT review_version FROM batches WHERE id=?",Long.class,id)!=snap.reviewVersion())throw new ApiError(409,"统计期间复核版本变化，请刷新后重试");return out;
    }
    final AnalysisService analysis;final ReferenceArchive reference;final BatchAccess access;final JdbcTemplate db;final ObjectMapper json;
    public ConsoleService(AnalysisService analysis,ReferenceArchive reference,BatchAccess access,JdbcTemplate db,ObjectMapper json){this.analysis=analysis;this.reference=reference;this.access=access;this.db=db;this.json=json;}
    Map<String,Object> batch(String id){access.requireRead(id);return db.queryForMap("SELECT label,source_type,provenance_json,task_type FROM batches WHERE id=?",id);}
    Map<String,Object> metadata(){
        try(var input=new ClassPathResource("static/console-config.json").getInputStream()){return json.readValue(input,new TypeReference<>(){});}
        catch(Exception e){return new LinkedHashMap<>(Map.of("title","巡变智判","classes",Map.of(),"classOrder",List.of()));}
    }
    public Map<String,Object> console(String id){return console(id,"effective");}
    public Map<String,Object> console(String id,String basis){
        var batch=batch(id);var snap=analysis.snapshot(id,basis);boolean archive="ARCHIVE".equals(batch.get("source_type"));
        Map<String,Object> data=archive?reference.data(id):snap.data();
        if(archive)analysis.annotations.overlay(id,data,basis,true);
        if(!archive){
            var dimensions=new HashMap<String,Map<String,Object>>();for(var row:db.queryForList("SELECT id,width,height FROM sample_tasks WHERE batch_id=?",id))dimensions.put(row.get("id").toString(),row);
            for(var region:(List<Map<String,Object>>)data.get("regions")){
                var items=(List<Map<String,Object>>)region.get("items");region.put("name",region.getOrDefault("short",region.get("id")));region.put("totalRows",items.size());region.put("displayRows",items.size());
                for(var item:items){
                    item.put("stem",item.getOrDefault("name",item.get("id")));item.put("regionId",region.get("id"));var dim=dimensions.get(item.get("id").toString());if(dim!=null)item.put("sourceSize",List.of(dim.get("width"),dim.get("height")));
                    var urls=new LinkedHashMap<String,Object>();for(var e:((Map<String,Object>)item.getOrDefault("assets",Map.of())).entrySet())urls.put(e.getKey(),e.getValue().toString().startsWith("/api/v1/")?e.getValue():"/api/v1/assets/"+e.getValue());
                    if(urls.containsKey("predMask"))urls.put("mask",urls.get("predMask"));item.put("assets",urls);
                    if(item.get("manualReview") instanceof Map review){item.put("reviewVerdict",review.get("verdict"));item.put("reviewVersion",review.get("version"));}
                    else {item.put("reviewVerdict","unreviewed");item.put("reviewVersion",0);}
                }
            }
        }
        if(!archive){
            var rich=analysis.annotations.source(id);analysis.annotations.overlay(id,rich,basis,true);
            var byId=new HashMap<String,Map<String,Object>>();for(var r:(List<Map<String,Object>>)rich.get("regions"))for(var i:(List<Map<String,Object>>)r.get("items"))byId.put(i.get("id").toString(),i);
            for(var r:(List<Map<String,Object>>)data.get("regions"))for(var i:(List<Map<String,Object>>)r.get("items")){var ri=byId.get(i.get("id").toString());for(String k:List.of("originalPrediction","correction","approvedCorrection","hasApprovedCorrection"))if(ri.containsKey(k))i.put(k,ri.get(k));if("effective".equals(basis)&&ri.get("approvedCorrection") instanceof Map c)((Map)i.get("assets")).put("correctedMask",c.get("maskUrl"));}
        }
        var app=metadata();app.put("regions",data.get("regions"));app.put("qaApiUrl","/api/v1/batches/"+id+"/questions");
        boolean seg="building_segmentation".equals(batch.get("task_type"));
        if(seg){app.put("title","巡变智判 · 建筑物分割");app.put("classes",Map.of());app.put("classOrder",List.of());}
        var out=new LinkedHashMap<String,Object>();out.put("batchId",id);out.put("label",batch.get("label"));out.put("sourceType",batch.get("source_type"));out.put("readOnly","original".equals(basis)||"VIEW".equals(access.permission(id)));out.put("permission",access.permission(id));out.put("dataVersion",snap.digest());out.put("predictionDigest",snap.predictionDigest());out.put("reviewVersion",snap.reviewVersion());out.put("models",snap.models());out.put("app",app);out.put("resultBasis",basis);
        out.put("capabilities",Map.of("questions",true,"reviewVerdicts",true,"reviewGeometry",true,"eventClassification",archive,"confidence",archive,"calibratedRisk",false,"geographicArea",false,"referenceLabels",false));
        out.put("taskType",seg?"building_segmentation":"change_detection");
        try{out.put("provenance",batch.get("provenance_json")==null?Map.of("kind","new_prediction"):json.readValue(batch.get("provenance_json").toString(),Map.class));}catch(Exception e){throw new ApiError(500,"来源信息无法读取");}
        if(db.queryForObject("SELECT review_version FROM batches WHERE id=?",Long.class,id)!=snap.reviewVersion())throw new ApiError(409,"读取期间复核版本变化，请刷新后重试");
        return out;
    }
    public Map<String,Object> summary(String id,Map<String,Object> request){
        var batch=batch(id);String basis=AnnotationLayer.basis(request.get("resultBasis"));var snap=analysis.snapshot(id,basis);
        Object expected=request.get("reviewVersion");if(expected!=null&&(!(expected instanceof Number n)||n.longValue()!=snap.reviewVersion()))throw new ApiError(409,"复核数据已更新，请刷新控制台后重新生成报告",Map.of("reviewVersion",snap.reviewVersion()));
        Object expectedDigest=request.get("dataVersion");if(expectedDigest!=null&&!snap.digest().equals(expectedDigest))throw new ApiError(409,"报告数据版本已变化，请刷新后重试");
        String region=Objects.toString(request.getOrDefault("regionId",request.getOrDefault("scopeRegion","")),"");if(region.equals("all"))region="";
        var all=new ArrayList<Map<String,Object>>();boolean found=region.isEmpty();
        Map<String,Object> source="ARCHIVE".equals(batch.get("source_type"))?reference.data(id):analysis.annotations.source(id);analysis.annotations.overlay(id,source,basis,true);
        for(var r:(List<Map<String,Object>>)source.get("regions")){if(region.isEmpty()||region.equals(r.get("id"))){found=true;all.addAll((List<Map<String,Object>>)r.get("items"));}}
        if(!found)throw new ApiError(400,"报告区域不属于当前批次");
        if(request.containsKey("sampleIds")){
            if(!(request.get("sampleIds") instanceof List<?> list)||list.size()>10000||list.stream().anyMatch(x->!(x instanceof String)))throw new ApiError(400,"报告样本范围无效");
            var wanted=new HashSet<>(list);var known=new HashSet<String>();all.forEach(i->known.add(i.get("id").toString()));if(!known.containsAll(wanted))throw new ApiError(400,"报告包含不属于当前范围的样本");all.removeIf(i->!wanted.contains(i.get("id")));
        }
        if("building_segmentation".equals(batch.get("task_type")))return segmentationSummary(id,basis,snap,all,region);
        int changed=0,reviewed=0,confirmed=0,rejected=0,uncertain=0,needs=0,unknownRisk=0,corrected=0;double px=0;int knownPx=0;
        var risks=new LinkedHashMap<String,Integer>();for(String key:List.of("high","medium","low","none","unknown"))risks.put(key,0);
        var events=new LinkedHashMap<String,Integer>();var scales=new LinkedHashMap<String,Integer>();var reasons=new LinkedHashMap<String,Integer>();var corrections=new ArrayList<Map<String,Object>>();
        for(var item:all){
            if(Boolean.TRUE.equals(item.get("hasApprovedCorrection"))){corrected++;var entry=new LinkedHashMap<String,Object>();entry.put("sampleId",item.get("id"));entry.put("changedPx",item.get("changedPx"));entry.put("correction",item.get("approvedCorrection"));corrections.add(entry);}
            scales.merge(Objects.toString(item.get("scale"),"未知"),1,Integer::sum);if(item.get("reviewReasonCodes") instanceof List codes)for(Object code:codes)reasons.merge(code.toString(),1,Integer::sum);
            if(item.get("changedPx") instanceof Number p){px+=p.doubleValue();knownPx++;if(p.doubleValue()>0)changed++;}
            if(Boolean.TRUE.equals(item.get("reviewRequired")))needs++;
            String risk=Objects.toString(item.get("reviewLevel"),"unknown");if(!risks.containsKey(risk))risk="unknown";risks.put(risk,risks.get(risk)+1);if(risk.equals("unknown"))unknownRisk++;
            Object review=item.get("manualReview");if(review instanceof Map r){reviewed++;switch(Objects.toString(r.get("verdict"),"uncertain")){case "confirmed_change"->confirmed++;case "false_change"->rejected++;default->uncertain++;}}
            else if(Boolean.TRUE.equals(item.get("manualReviewComplete"))||"manual_verified".equals(item.get("reviewStatus")))reviewed++;
            String event=Objects.toString(item.get("primaryEventName"),"");if(!event.isBlank())events.merge(event,1,Integer::sum);
        }
        var reps=new ArrayList<Map<String,Object>>();var sorted=new ArrayList<>(all);sorted.sort(Comparator.comparingDouble((Map<String,Object> i)->i.get("changedPx") instanceof Number n?n.doubleValue():-1).reversed().thenComparing(i->i.get("id").toString()));
        for(var item:sorted.subList(0,Math.min(5,sorted.size()))){var rep=new LinkedHashMap<String,Object>();for(String key:List.of("id","stem","name","changedPx","changedRatio","objectCount","reviewVerdict"))if(item.containsKey(key))rep.put(key,item.get(key));if(!rep.containsKey("stem"))rep.put("stem",item.get("name"));reps.add(rep);}
        var riskBuckets=new ArrayList<Map<String,Object>>();for(var e:risks.entrySet())riskBuckets.add(Map.of("key",e.getKey(),"label",Map.of("high","高风险标记","medium","中风险标记","low","低风险标记","none","无风险标记","unknown","未知").get(e.getKey()),"count",e.getValue(),"ratio",all.isEmpty()?0:1.0*e.getValue()/all.size()));
        var eventBuckets=new ArrayList<Map<String,Object>>();for(var e:events.entrySet())eventBuckets.add(Map.of("key",e.getKey(),"label",e.getKey(),"count",e.getValue(),"ratio",all.isEmpty()?0:1.0*e.getValue()/all.size()));
        var report=new LinkedHashMap<String,Object>();report.put("correctionEntries",corrections);report.put("scaleCounts",scales);report.put("reasonCounts",reasons);report.put("correctedSamples",corrected);report.put("resultBasis",basis);report.put("totalCount",all.size());report.put("changedCount",changed);report.put("knownChangeCount",knownPx);report.put("changedPixelsSum",knownPx==0?null:px);report.put("reviewedCount",reviewed);report.put("reviewCount",needs);report.put("confirmedChangeCount",confirmed);report.put("falseChangeCount",rejected);report.put("uncertainCount",uncertain);report.put("unknownRiskCount",unknownRisk);report.put("riskBuckets",riskBuckets);report.put("eventBuckets",eventBuckets);report.put("representatives",reps);report.put("regionId",region);report.put("filterText",region.isEmpty()?"本批全部区域":region.toUpperCase(Locale.ROOT));
        String text=(basis.equals("original")?"原始预测口径。":"人工修正后口径（无修正样本沿用原预测）。")+"已审批修正"+corrected+"张。当前范围共"+all.size()+"张样本，其中检测到变化"+changed+"张；变化像素仅表示影像中的像素量。已有"+reviewed+"张样本完成人工复核，确认变化"+confirmed+"张、伪变化"+rejected+"张、暂不确定"+uncertain+"张。"+(unknownRisk>0?unknownRisk+"张样本未提供风险等级，不能视为低风险。":"风险等级来自已有质量标记，不代表人工误检真值。")+("INFERENCE".equals(batch.get("source_type"))?"当前检测模型未提供真实地物类别、置信度或实际地面面积。":"该范围来自原版历史样本快照，未在此处重新推理。");
        var out=new LinkedHashMap<String,Object>();out.put("status","done");out.put("resultBasis",basis);out.put("text",text);out.put("answer",text);out.put("engine","server_stats");out.put("report",report);out.put("batchId",id);out.put("dataVersion",snap.digest());out.put("reviewVersion",snap.reviewVersion());out.put("predictionDigest",snap.predictionDigest());out.put("models",snap.models());out.put("sourceType",batch.get("source_type"));out.put("generatedAt",System.currentTimeMillis());if(db.queryForObject("SELECT review_version FROM batches WHERE id=?",Long.class,id)!=snap.reviewVersion())throw new ApiError(409,"统计期间复核版本变化，请刷新后重试");return out;
    }
}

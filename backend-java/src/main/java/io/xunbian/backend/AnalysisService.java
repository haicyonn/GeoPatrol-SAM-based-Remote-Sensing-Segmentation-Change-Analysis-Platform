package io.xunbian.backend;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.*;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.support.TransactionTemplate;

@Service
public class AnalysisService {
    final JdbcTemplate db;final TransactionTemplate tx;final ObjectMapper json;final PythonClient python;final BatchAccess access;final ResourceAdmission admission;final AnnotationLayer annotations;
    public AnalysisService(JdbcTemplate db,TransactionTemplate tx,ObjectMapper json,PythonClient python,BatchAccess access,ResourceAdmission admission,AnnotationLayer annotations){this.db=db;this.tx=tx;this.json=json;this.python=python;this.access=access;this.admission=admission;this.annotations=annotations;}
    public void publishCompleted(){
        for(var row:db.queryForList("SELECT id FROM batches WHERE status='COMPLETED'"))publish(row.get("id").toString());
    }
    void publish(String id){tx.executeWithoutResult(transaction->{
        var batches=db.queryForList("SELECT * FROM batches WHERE id=? FOR UPDATE",id);
        if(batches.isEmpty()||!"COMPLETED".equals(batches.getFirst().get("status")))return;
        boolean seg="building_segmentation".equals(batches.getFirst().get("task_type"));
        var rows=db.queryForList("SELECT p.task_id,t.region_id,t.sample_name,t.width,t.height,t.before_asset,t.after_asset,p.result_json FROM "+(seg?"segmentation_task_results":"prediction_results")+" p JOIN sample_tasks t ON t.id=p.task_id WHERE p.batch_id=? ORDER BY t.region_id,t.id",id);
        if(rows.size()!=((Number)batches.getFirst().get("total")).intValue())throw new ApiError(409,"批次结果尚未完整入库");
        var regions=new TreeMap<String,Map<String,Object>>();
        try{
            for(var row:rows){
                String rid=row.get("region_id").toString();
                var region=regions.computeIfAbsent(rid,k->{var r=new LinkedHashMap<String,Object>();r.put("id",k);r.put("short",k.toUpperCase(Locale.ROOT));r.put("items",new ArrayList<>());return r;});
                var result=json.readValue(row.get("result_json").toString(),new TypeReference<Map<String,Object>>(){});
                var item=new LinkedHashMap<String,Object>();
                item.put("id",row.get("task_id"));item.put("name",row.get("sample_name"));
                for(String key:seg?List.of("foregroundPixels","foregroundRatio","componentCount","maskQuality"):List.of("changedPx","changedRatio","objectCount","maskQuality"))item.put(key,result.get(key));
                if(seg)for(String key:BuildingCountService.FIELDS)item.put(key,result.get(key));
                item.put("sourceSize",List.of(row.get("width"),row.get("height")));
                item.put("reviewLevel","unknown");
                item.put("assets",seg?Map.of("source",row.get("before_asset"),"predMask",result.get("predMaskAssetId")):Map.of("pre",row.get("before_asset"),"post",row.get("after_asset"),"predMask",result.get("predMaskAssetId")));
                if(seg&&result.get("overlayAssetId")!=null){var assets=new LinkedHashMap<String,Object>((Map)item.get("assets"));assets.put("overlay",result.get("overlayAssetId"));item.put("assets",assets);}
                ((List<Map<String,Object>>)region.get("items")).add(item);
            }
            String data=json.writeValueAsString(Map.of("taskType",seg?"building_segmentation":"change_detection","regions",new ArrayList<>(regions.values())));
            String digest=HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(data.getBytes(StandardCharsets.UTF_8)));
            db.update("INSERT INTO analysis_datasets VALUES(?,?,?,?)",id,data,digest,System.currentTimeMillis());
            db.update("UPDATE batches SET status='PUBLISHED',updated_ms=? WHERE id=?",System.currentTimeMillis(),id);
        }catch(ApiError e){throw e;}catch(Exception e){throw new ApiError(500,"分析数据发布失败");}
    });}
    record Snapshot(String datasetId,Map<String,Object> data,String predictionDigest,String digest,long reviewVersion,List<Map<String,Object>> models){}
    Snapshot snapshot(String id){return snapshot(id,"effective");}
    Snapshot snapshot(String id,String basis){
        AnnotationLayer.basis(basis);
        access.requireRead(id);
        return tx.execute(t->{
            var rows=db.queryForList("SELECT d.data_json,d.source_digest,b.review_version FROM analysis_datasets d JOIN batches b ON b.id=d.batch_id WHERE d.batch_id=? AND b.status='PUBLISHED' FOR SHARE",id);
            if(rows.isEmpty())throw new ApiError(409,"本批结果尚未完整发布，请等待预测完成");
            var row=rows.getFirst();long version=((Number)row.get("review_version")).longValue();Map<String,Object> data;
            try{data=json.readValue(row.get("data_json").toString(),new TypeReference<>(){});}catch(Exception e){throw new ApiError(500,"已发布数据无法读取");}
            annotations.overlay(id,data,basis,false);
            data.put("reviewVersion",version);data.put("resultBasis",basis);
            try{
                String digest=HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(json.writeValueAsBytes(data)));
                var models=db.queryForList("SELECT DISTINCT model_id AS modelId,model_version AS modelVersion FROM "+("building_segmentation".equals(data.get("taskType"))?"segmentation_task_results":"prediction_results")+" WHERE batch_id=? ORDER BY model_id,model_version",id);
                if(models.isEmpty()&&data.get("models") instanceof List list)models=(List<Map<String,Object>>)list;
                return new Snapshot(id+"-v3-"+basis+"-r"+version+"-"+digest.substring(0,12),data,row.get("source_digest").toString(),digest,version,models);
            }catch(Exception e){throw new ApiError(500,"分析数据版本生成失败");}
        });
    }
    public Map<String,Object> report(String id){return report(id,"effective");}
    public Map<String,Object> report(String id,String basis){
        Snapshot snap=snapshot(id,basis);var regions=new ArrayList<Map<String,Object>>();int total=0,reviewed=0;
        if("building_segmentation".equals(snap.data.get("taskType"))){
            for(var region:(List<Map<String,Object>>)snap.data.get("regions")){
                var items=(List<Map<String,Object>>)region.get("items");long pixels=0,components=0;double ratios=0;
                for(var item:items){pixels+=((Number)item.get("foregroundPixels")).longValue();components+=((Number)item.get("componentCount")).longValue();ratios+=((Number)item.get("foregroundRatio")).doubleValue();}
                var row=new LinkedHashMap<String,Object>();row.put("regionId",region.get("id"));row.put("images",items.size());row.put("foregroundPixelsSum",pixels);row.put("componentCountSum",components);row.put("meanForegroundRatio",items.isEmpty()?null:ratios/items.size());regions.add(row);total+=items.size();
                for(String field:List.of("estimatedBuildingCount","filteredFragmentCount","boundaryBuildingCount"))row.put(field+"Sum",items.stream().mapToLong(i->((Number)i.get(field)).longValue()).sum());
            }
            return Map.of("taskType","building_segmentation","batchId",id,"dataDigest",snap.digest,"reviewVersion",snap.reviewVersion,"sampleCount",total,"regions",regions,"models",snap.models,"data",snap.data,"units","建筑像素与像素占比；连通区域不是建筑栋数，未换算地面面积");
        }
        for(var region:(List<Map<String,Object>>)snap.data.get("regions")){
            var items=(List<Map<String,Object>>)region.get("items");long pixels=0;double sum=0;int done=0,confirmed=0,rejected=0,uncertain=0;
            int knownRatio=0;for(var item:items){if(item.get("changedPx") instanceof Number n)pixels+=n.longValue();if(item.get("changedRatio") instanceof Number n){sum+=n.doubleValue();knownRatio++;}
                if(item.get("manualReview") instanceof Map r){done++;switch(r.get("verdict").toString()){case "confirmed_change"->confirmed++;case "false_change"->rejected++;default->uncertain++;}}
            }
            var stats=new LinkedHashMap<String,Object>();stats.put("regionId",region.get("id"));stats.put("samples",items.size());stats.put("changedPixelsSum",pixels);stats.put("meanChangedRatio",knownRatio==0?null:sum/knownRatio);stats.put("reviewed",done);stats.put("confirmedChange",confirmed);stats.put("falseChange",rejected);stats.put("uncertain",uncertain);regions.add(stats);total+=items.size();reviewed+=done;
        }
        var out=new LinkedHashMap<String,Object>();out.put("batchId",id);out.put("generatedAt",System.currentTimeMillis());out.put("generatedBy",access.accounts.current().id());out.put("predictionDigest",snap.predictionDigest);out.put("dataDigest",snap.digest);out.put("reviewVersion",snap.reviewVersion);out.put("models",snap.models);out.put("sampleCount",total);out.put("reviewedSamples",reviewed);out.put("regions",regions);out.put("data",snap.data);out.put("units","变化像素按影像累加，不代表去重地理面积；人工结论与模型预测分别保存。");return out;
    }
    public Map<String,Object> question(String id,Map<String,Object> input){
        return assistant(id,input,false);
    }
    public Map<String,Object> analysis(String id,Map<String,Object> input){return assistant(id,input,true);}
    Map<String,Object> assistant(String id,Map<String,Object> input,boolean analysis){
        access.requireRead(id);
        Object session=input.get("sessionId"),question=input.get("question");
        boolean history=!analysis&&"history".equals(input.get("action"));
        if(!(session instanceof String s)||!s.matches("[A-Za-z0-9][A-Za-z0-9_.-]{0,127}"))throw new ApiError(400,"会话格式无效");
        if(!analysis&&!history&&(!(question instanceof String q)||q.isBlank()||q.length()>4000))throw new ApiError(400,"问题格式无效");
        Object viewRegion=input.get("viewRegion");if(viewRegion!=null&&(!(viewRegion instanceof String r)||!r.matches("[a-z][a-z0-9_-]{0,63}")))throw new ApiError(400,"当前页面区域格式无效");
        if(analysis){
            if(!Set.of("plan","execute").contains(Objects.toString(input.get("action"),"")))throw new ApiError(400,"未知分析操作");
            if("plan".equals(input.get("action"))&&(!(input.get("goal") instanceof String g)||g.isBlank()||g.length()>2000))throw new ApiError(400,"请输入有效分析目标");
            if("execute".equals(input.get("action"))&&(!(input.get("planId") instanceof String p)||p.length()>100||!input.containsKey("reviewVersion")))throw new ApiError(400,"分析计划或版本无效，请重新生成方案");
        }
        String basis=AnnotationLayer.basis(input.get("resultBasis"));String user=access.accounts.current().id();String bound;
        try{bound=HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest((user+":"+id+":"+basis+":"+session).getBytes(StandardCharsets.UTF_8)));}
        catch(Exception e){throw new IllegalStateException(e);}
        try(var turn=history?null:admission.enter(user,bound)){
            if(turn!=null)turn.awaitGpu();access.requireRead(id);
            Snapshot snap=snapshot(id,basis);
            if(viewRegion!=null&&((List<Map<String,Object>>)snap.data.get("regions")).stream().noneMatch(r->viewRegion.equals(r.get("id"))))throw new ApiError(400,"当前页面区域不属于本批数据");
            if(analysis&&input.containsKey("reviewVersion")&&(!(input.get("reviewVersion") instanceof Number n)||n.longValue()!=snap.reviewVersion))throw new ApiError(409,"分析数据版本已更新，请重新生成分析方案");
            var previous=db.queryForList("SELECT review_version FROM query_conversations WHERE user_id=? AND batch_id=? AND session_hash=?",user,id,bound);
            boolean reset=!previous.isEmpty()&&((Number)previous.getFirst().get("review_version")).longValue()!=snap.reviewVersion;
            python.call("/internal/datasets",Map.of("datasetId",snap.datasetId,"data",snap.data),10);
            var request=new LinkedHashMap<String,Object>();request.put("datasetId",snap.datasetId);request.put("sessionId",bound);request.put("userId",user);
            for(String key:List.of("question","scopeRegion","viewRegion","currentSampleId","clarificationId","optionId","mode","action","goal","planId"))if(input.containsKey(key))request.put(key,input.get(key));
            var result=new LinkedHashMap<>(python.call(analysis?"/internal/analysis":"/internal/questions",request,180));
            access.requireRead(id);
            if(!history&&!analysis)db.update("INSERT INTO query_conversations VALUES(?,?,?,?,?) ON DUPLICATE KEY UPDATE review_version=VALUES(review_version),updated_ms=VALUES(updated_ms)",user,id,bound,snap.reviewVersion,System.currentTimeMillis());
            if(result.containsKey("dataVersion"))result.put("assistantDataVersion",result.get("dataVersion"));
            result.put("resultBasis",basis);result.put("dataVersion",snap.digest);result.put("batchId",id);result.put("databaseDigest",snap.digest);result.put("predictionDigest",snap.predictionDigest);result.put("reviewVersion",snap.reviewVersion);result.put("contextReset",reset);
            result.put("dataChangedDuringQuery",db.queryForObject("SELECT review_version FROM batches WHERE id=?",Long.class,id)!=snap.reviewVersion);return result;
        }
    }
}

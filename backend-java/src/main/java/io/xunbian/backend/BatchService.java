package io.xunbian.backend;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.*;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.support.TransactionTemplate;
import org.springframework.web.multipart.MultipartFile;

@Service
public class BatchService {
    final JdbcTemplate db;final TransactionTemplate tx;final Storage storage;final ObjectMapper json;final BatchAccess access;final Settings settings;
    public BatchService(JdbcTemplate db,TransactionTemplate tx,Storage storage,ObjectMapper json,BatchAccess access,Settings settings){
        this.db=db;this.tx=tx;this.storage=storage;this.json=json;this.access=access;this.settings=settings;
    }
    static String stem(MultipartFile file){
        String name=Optional.ofNullable(file.getOriginalFilename()).orElse("").replace('\\','/');
        name=name.substring(name.lastIndexOf('/')+1);
        int dot=name.lastIndexOf('.');
        if(dot<=0||dot>150)throw new ApiError(400,"影像名称无效");
        return name.substring(0,dot).toLowerCase(Locale.ROOT);
    }
    static Map<String,MultipartFile> pairMap(List<MultipartFile> files){
        var map=new LinkedHashMap<String,MultipartFile>();
        for(var f:files)if(map.putIfAbsent(stem(f),f)!=null)throw new ApiError(400,"同一时相存在重名样本");
        return map;
    }
    public Map<String,Object> create(String label,String region,String assignments,List<MultipartFile> before,List<MultipartFile> after){
        return create(label,region,assignments,before,after,"change_detection");
    }
    public Map<String,Object> create(String label,String region,String assignments,List<MultipartFile> before,List<MultipartFile> after,String taskType){
        boolean segmentation="building_segmentation".equals(taskType);
        if(!segmentation&&!"change_detection".equals(taskType))throw new ApiError(400,"任务类型未登记");
        if(label==null||label.isBlank()||label.length()>200||region==null||!region.matches("[a-z][a-z0-9_-]{0,63}"))
            throw new ApiError(400,"批次名称或区域标识无效");
        if(before.isEmpty()||before.size()>200||(!segmentation&&after.size()!=before.size())||(segmentation&&!after.isEmpty()))throw new ApiError(400,segmentation?"建筑分割需要1至200张单图，不上传后时相":"需要1至200组配对影像");
        var a=pairMap(before);var b=pairMap(after);
        if(!segmentation&&!a.keySet().equals(b.keySet()))throw new ApiError(400,"前后时相文件名未一一配对");
        Map<String,String> regions;
        try{regions=assignments==null||assignments.isBlank()?Map.of():json.readValue(assignments,new TypeReference<>(){});}
        catch(Exception e){throw new ApiError(400,"区域分配清单格式无效");}
        if(!a.keySet().containsAll(regions.keySet())||regions.values().stream().anyMatch(r->r==null||!r.matches("[a-z][a-z0-9_-]{0,63}")))
            throw new ApiError(400,"区域分配清单与文件不匹配");
        String id=UUID.randomUUID().toString();long now=System.currentTimeMillis();
        var saved=new ArrayList<String>();var inputs=new ArrayList<Map<String,Object>>();
        try{
            for(String name:a.keySet()){
                var first=storage.save(a.get(name));saved.add(first.id());
                var input=new LinkedHashMap<String,Object>();input.put("id",UUID.randomUUID().toString());input.put("name",name);input.put("region",regions.getOrDefault(name,region));input.put("a",first);
                if(!segmentation){var second=storage.save(b.get(name));saved.add(second.id());if(first.width()!=second.width()||first.height()!=second.height())throw new ApiError(400,"样本“"+name+"”前后时相尺寸不一致");input.put("b",second);}
                inputs.add(input);
            }
            String owner=access.accounts.current().id();
            tx.executeWithoutResult(status->{
                db.queryForObject("SELECT id FROM app_users WHERE id=? FOR UPDATE",String.class,owner);
                // The first consistent read must happen after acquiring admission.
                // A pre-lock identity SELECT would freeze an older REPEATABLE READ snapshot.
                access.accounts.current();
                long pending=db.queryForObject("SELECT COUNT(*) FROM sample_tasks t JOIN batches b ON b.id=t.batch_id WHERE b.owner_id=? AND t.status IN ('PENDING','RUNNING')",Long.class,owner);
                if(pending+inputs.size()>settings.maxPendingPerUser)throw new ApiError(429,"每名成员最多保留"+settings.maxPendingPerUser+"个待处理样本，请先完成或取消已有任务");
                db.update("INSERT INTO batches(id,label,status,model_id,total,created_ms,updated_ms,owner_id) VALUES(?,?, 'DRAFT',NULL,?,?,?,?)",id,label,inputs.size(),now,now,owner);
                db.update("UPDATE batches SET task_type=? WHERE id=?",taskType,id);
                for(var input:inputs){
                    String tid=(String)input.get("id");var first=(Storage.Stored)input.get("a");var second=(Storage.Stored)input.get("b");
                    db.update("INSERT INTO sample_tasks(id,batch_id,sample_name,region_id,before_asset,after_asset,width,height,status,created_ms,updated_ms) VALUES(?,?,?,?,?,?,?,?,'PENDING',?,?)",
                        tid,id,input.get("name"),input.get("region"),first.id(),second==null?null:second.id(),first.width(),first.height(),now,now);
                    var entries=new ArrayList<Map.Entry<String,Storage.Stored>>();entries.add(Map.entry(segmentation?"source":"before",first));if(second!=null)entries.add(Map.entry("after",second));
                    for(var entry:entries){
                        var asset=entry.getValue();db.update("INSERT INTO assets VALUES(?,?,?,?,?,?)",asset.id(),id,tid,entry.getKey(),asset.mime(),asset.bytes());
                    }
                }
            });
        }catch(RuntimeException e){storage.removeUncommitted(saved);throw e;}
        return get(id);
    }
    public Map<String,Object> get(String id){
        access.requireRead(id);
        var rows=db.queryForList("SELECT * FROM batches WHERE id=?",id);
        if(rows.isEmpty())throw new ApiError(404,"批次不存在");
        var row=rows.getFirst();var out=new LinkedHashMap<String,Object>();
        out.put("batchId",id);out.put("label",row.get("label"));out.put("status",row.get("status"));out.put("total",row.get("total"));
        out.put("modelId",row.get("model_id"));out.put("createdAt",row.get("created_ms"));
        out.put("permission",access.permission(id));out.put("ownerId",row.get("owner_id"));out.put("reviewVersion",row.get("review_version"));
        out.put("sourceType",row.get("source_type"));
        out.put("taskType",row.get("task_type"));
        out.put("tasks",db.queryForList("SELECT id AS taskId,sample_name AS sampleName,region_id AS regionId,status,attempt_no AS attemptNo,error_message AS error,before_asset AS beforeAssetId,after_asset AS afterAssetId FROM sample_tasks WHERE batch_id=? ORDER BY created_ms,id",id));
        out.put("completed",Set.of("ARCHIVE","SEGMENTATION").contains(row.get("source_type"))?row.get("total"):db.queryForObject("SELECT COUNT(*) FROM sample_tasks WHERE batch_id=? AND status='SUCCEEDED'",Integer.class,id));
        out.put("failed",db.queryForObject("SELECT COUNT(*) FROM sample_tasks WHERE batch_id=? AND status='FAILED'",Integer.class,id));
        return out;
    }
    public List<Map<String,Object>> list(){var a=access.accounts.current();return db.queryForList("SELECT b.id AS batchId,b.label,b.status,b.total,b.task_type AS taskType,b.source_type AS sourceType,b.created_ms AS createdAt,u.display_name AS ownerName FROM batches b LEFT JOIN app_users u ON u.id=b.owner_id WHERE ?=TRUE OR (b.owner_id=? OR EXISTS(SELECT 1 FROM batch_members m WHERE m.batch_id=b.id AND m.user_id=?)) ORDER BY b.created_ms DESC LIMIT 100",a.admin(),a.id(),a.id());}
    public Map<String,Object> start(String id,String model){
        access.requireEdit(id);
        String task=db.queryForObject("SELECT task_type FROM batches WHERE id=?",String.class,id);
        if(!("building_segmentation".equals(task)?model.equals("sam-scfp-whu"):Set.of("secor-whu","secor-sysu").contains(model)))throw new ApiError(400,"模型与任务类型不匹配");
        int changed=db.update("UPDATE batches SET status='RUNNING',model_id=?,updated_ms=? WHERE id=? AND status='DRAFT'",model,System.currentTimeMillis(),id);
        if(changed!=1){get(id);throw new ApiError(409,"当前状态不能开始执行");}return get(id);
    }
    public Map<String,Object> retryFailed(String id){
        access.requireEdit(id);
        tx.executeWithoutResult(status->{
            db.queryForMap("SELECT * FROM execution_slots WHERE slot_id=1 FOR UPDATE");
            var rows=db.queryForList("SELECT status,owner_id FROM batches WHERE id=? FOR UPDATE",id);
            if(rows.isEmpty())throw new ApiError(404,"批次不存在");
            if(!rows.getFirst().get("status").equals("FAILED"))throw new ApiError(409,"当前批次没有可重试的失败任务");
            String owner=rows.getFirst().get("owner_id").toString();
            db.queryForObject("SELECT id FROM app_users WHERE id=? FOR UPDATE",String.class,owner);
            long pending=db.queryForObject("SELECT COUNT(*) FROM sample_tasks t JOIN batches b ON b.id=t.batch_id WHERE b.owner_id=? AND t.status IN ('PENDING','RUNNING')",Long.class,owner);
            long failed=db.queryForObject("SELECT COUNT(*) FROM sample_tasks WHERE batch_id=? AND status='FAILED'",Long.class,id);
            if(pending+failed>settings.maxPendingPerUser)throw new ApiError(429,"该批次创建者的待处理样本已达配额，请先完成或取消其他任务");
            db.update("UPDATE sample_tasks SET status='PENDING',attempt_id=NULL,error_message=NULL,next_eligible_ms=0,transient_failures=0,lease_expirations=0,updated_ms=? WHERE batch_id=? AND status='FAILED'",System.currentTimeMillis(),id);
            db.update("UPDATE batches SET status='RUNNING',updated_ms=? WHERE id=?",System.currentTimeMillis(),id);
        });return get(id);
    }
    public Map<String,Object> cancel(String id){
        access.requireEdit(id);
        tx.executeWithoutResult(transaction->{
            db.queryForMap("SELECT * FROM execution_slots WHERE slot_id=1 FOR UPDATE");
            var rows=db.queryForList("SELECT status FROM batches WHERE id=? FOR UPDATE",id);
            if(rows.isEmpty())throw new ApiError(404,"批次不存在");
            if(Set.of("PUBLISHED","COMPLETED","CANCELLED").contains(rows.getFirst().get("status")))throw new ApiError(409,"当前状态不能取消");
            long now=System.currentTimeMillis();
            db.update("UPDATE task_attempts SET status='CANCELLED',finished_ms=? WHERE status='RUNNING' AND task_id IN (SELECT id FROM sample_tasks WHERE batch_id=?)",now,id);
            db.update("UPDATE execution_slots SET task_id=NULL,attempt_id=NULL,lease_until=NULL WHERE task_id IN (SELECT id FROM sample_tasks WHERE batch_id=?)",id);
            db.update("UPDATE sample_tasks SET status='CANCELLED',lease_until=NULL,updated_ms=? WHERE batch_id=? AND status IN ('PENDING','RUNNING')",now,id);
            db.update("UPDATE batches SET status='CANCELLED',updated_ms=? WHERE id=?",now,id);
        });return get(id);
    }
    public List<Map<String,Object>> results(String id,int offset,int limit){
        get(id);if(offset<0||limit<1||limit>200)throw new ApiError(400,"分页参数无效");
        if("building_segmentation".equals(db.queryForObject("SELECT task_type FROM batches WHERE id=?",String.class,id))){
            var row=db.queryForList("SELECT data_json FROM analysis_datasets WHERE batch_id=?",id);
            if(row.isEmpty())return List.of();
            try{
                var data=json.readValue(row.getFirst().get("data_json").toString(),Map.class);var all=new ArrayList<Map<String,Object>>();
                for(var r:(List<Map<String,Object>>)data.get("regions"))for(var i:(List<Map<String,Object>>)r.get("items")){var value=new LinkedHashMap<String,Object>(i);value.put("sampleId",i.get("id"));value.put("regionId",r.get("id"));all.add(value);}
                return all.subList(Math.min(offset,all.size()),Math.min(offset+limit,all.size()));
            }catch(Exception e){throw new ApiError(500,"分割结果无法读取");}
        }
        return db.queryForList("SELECT p.task_id AS sampleId,t.sample_name AS sampleName,t.region_id AS regionId,p.changed_px AS changedPx,p.changed_ratio AS changedRatio,p.object_count AS objectCount,p.pred_mask_asset AS predMaskAssetId,t.before_asset AS beforeAssetId,t.after_asset AS afterAssetId,p.model_id AS modelId,p.model_version AS modelVersion,COALESCE(r.version,0) AS reviewVersion,COALESCE(JSON_UNQUOTE(JSON_EXTRACT(r.data_json,'$.verdict')),'unreviewed') AS reviewVerdict FROM prediction_results p JOIN sample_tasks t ON t.id=p.task_id LEFT JOIN sample_annotations r ON r.sample_id=p.task_id AND r.batch_id=p.batch_id WHERE p.batch_id=? ORDER BY t.sample_name LIMIT ? OFFSET ?",id,limit,offset);
    }
}

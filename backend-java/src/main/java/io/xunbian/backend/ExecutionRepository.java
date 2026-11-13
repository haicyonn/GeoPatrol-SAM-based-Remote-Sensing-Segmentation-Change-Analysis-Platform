package io.xunbian.backend;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Files;
import java.util.*;
import javax.imageio.ImageIO;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.support.TransactionTemplate;

@Repository
public class ExecutionRepository {
    final JdbcTemplate db;final TransactionTemplate tx;final Settings settings;final Storage storage;final ObjectMapper json;final BuildingCountService counts;
    public ExecutionRepository(JdbcTemplate db,TransactionTemplate tx,Settings settings,Storage storage,ObjectMapper json,SchemaMigration migration,BuildingCountService counts){
        this.db=db;this.tx=tx;this.settings=settings;this.storage=storage;this.json=json;this.counts=counts;
        if(settings.leaseSeconds<3||settings.predictionTimeoutSeconds<1)throw new IllegalArgumentException("invalid task timing");
    }
    public record Attempt(String taskId,String batchId,String attemptId,String modelId,String beforeAssetId,String afterAssetId,int width,int height){
        public Map<String,Object> payload(){var p=new LinkedHashMap<String,Object>();p.put("taskId",taskId);p.put("attemptId",attemptId);p.put("modelId",modelId);p.put("beforeAssetId",beforeAssetId);if(afterAssetId!=null)p.put("afterAssetId",afterAssetId);return p;}
    }
    Map<String,Object> lockSlot(){return db.queryForMap("SELECT * FROM execution_slots WHERE slot_id=1 FOR UPDATE");}
    public Attempt claim(){return tx.execute(transaction->{
        var slot=lockSlot();if(slot.get("attempt_id")!=null)return null;
        long now=System.currentTimeMillis();if(((Number)slot.get("available_after_ms")).longValue()>now)return null;
        var rows=db.queryForList("SELECT t.*,b.model_id FROM sample_tasks t JOIN batches b ON b.id=t.batch_id WHERE t.status='PENDING' AND b.status='RUNNING' AND t.next_eligible_ms<=? ORDER BY b.last_dispatch_ms,b.created_ms,b.id,t.created_ms,t.id LIMIT 1 FOR UPDATE",now);
        if(rows.isEmpty())return null;
        var row=rows.getFirst();String attempt=UUID.randomUUID().toString();long until=now+settings.leaseSeconds*1000L;
        String task=row.get("id").toString();
        db.update("UPDATE sample_tasks SET status='RUNNING',attempt_id=?,attempt_no=attempt_no+1,lease_until=?,updated_ms=? WHERE id=?",attempt,until,now,task);
        db.update("INSERT INTO task_attempts(id,task_id,status,started_ms) VALUES(?,?,'RUNNING',?)",attempt,task,now);
        db.update("UPDATE execution_slots SET task_id=?,attempt_id=?,lease_until=? WHERE slot_id=1",task,attempt,until);
        db.update("UPDATE batches SET last_dispatch_ms=? WHERE id=?",now,row.get("batch_id"));
        return new Attempt(task,row.get("batch_id").toString(),attempt,row.get("model_id").toString(),row.get("before_asset").toString(),row.get("after_asset")==null?null:row.get("after_asset").toString(),((Number)row.get("width")).intValue(),((Number)row.get("height")).intValue());
    });}
    public boolean heartbeat(Attempt a){return Boolean.TRUE.equals(tx.execute(transaction->{
        var slot=lockSlot();long now=System.currentTimeMillis();
        if(!a.attemptId.equals(slot.get("attempt_id"))||((Number)slot.get("lease_until")).longValue()<now)return false;
        long until=now+settings.leaseSeconds*1000L;
        int n=db.update("UPDATE sample_tasks SET lease_until=?,updated_ms=? WHERE id=? AND attempt_id=? AND status='RUNNING'",until,now,a.taskId,a.attemptId);
        if(n==1)db.update("UPDATE execution_slots SET lease_until=? WHERE slot_id=1",until);
        return n==1;
    }));}
    void refresh(String batch){
        long pending=db.queryForObject("SELECT COUNT(*) FROM sample_tasks WHERE batch_id=? AND status IN ('PENDING','RUNNING')",Long.class,batch);
        if(pending!=0)return;
        long failed=db.queryForObject("SELECT COUNT(*) FROM sample_tasks WHERE batch_id=? AND status<>'SUCCEEDED'",Long.class,batch);
        db.update("UPDATE batches SET status=?,updated_ms=? WHERE id=? AND status='RUNNING'",failed==0?"COMPLETED":"FAILED",System.currentTimeMillis(),batch);
    }
    public void recoverExpired(){tx.executeWithoutResult(transaction->{
        var slot=lockSlot();long now=System.currentTimeMillis();
        if(slot.get("attempt_id")==null||((Number)slot.get("lease_until")).longValue()>=now)return;
        String attempt=slot.get("attempt_id").toString(),task=slot.get("task_id").toString();
        var rows=db.queryForList("SELECT batch_id FROM sample_tasks WHERE id=?",task);
        db.update("UPDATE task_attempts SET status='EXPIRED',finished_ms=?,error_message='execution lease expired' WHERE id=? AND status='RUNNING'",now,attempt);
        db.update("UPDATE sample_tasks SET status=IF(lease_expirations>=2,'FAILED','PENDING'),lease_expirations=lease_expirations+1,attempt_id=NULL,lease_until=NULL,error_message='执行中断，租约已过期',updated_ms=? WHERE id=? AND attempt_id=? AND status='RUNNING'",now,task,attempt);
        db.update("UPDATE execution_slots SET task_id=NULL,attempt_id=NULL,lease_until=NULL WHERE slot_id=1");
        if(!rows.isEmpty())refresh(rows.getFirst().get("batch_id").toString());
    });}
    public boolean workerProbeDue(){return db.queryForObject("SELECT COUNT(*) FROM execution_slots WHERE slot_id=1 AND available_after_ms>0 AND available_after_ms<=?",Integer.class,System.currentTimeMillis())>0;}
    public void workerProbeFailed(){db.update("UPDATE execution_slots SET available_after_ms=? WHERE slot_id=1",System.currentTimeMillis()+5000);}
    public void workerProbePassed(){db.update("UPDATE execution_slots SET available_after_ms=0 WHERE slot_id=1");}
    public void retryLater(Attempt a,String message){tx.executeWithoutResult(t->{
        var slot=lockSlot();if(!a.attemptId.equals(slot.get("attempt_id")))return;
        var rows=db.queryForList("SELECT transient_failures FROM sample_tasks WHERE id=? AND status='RUNNING' AND attempt_id=? FOR UPDATE",a.taskId,a.attemptId);
        if(rows.isEmpty())return;int count=((Number)rows.getFirst().get("transient_failures")).intValue()+1;
        long now=System.currentTimeMillis(),after=now+5000L*(1L<<Math.min(count-1,3));boolean exhausted=count>=4;
        db.update("UPDATE sample_tasks SET status=?,transient_failures=?,next_eligible_ms=?,attempt_id=NULL,lease_until=NULL,error_message=?,updated_ms=? WHERE id=?",exhausted?"FAILED":"PENDING",count,after,message,now,a.taskId);
        db.update("UPDATE task_attempts SET status=?,finished_ms=?,error_message=? WHERE id=?",exhausted?"FAILED":"RETRY_WAIT",now,message,a.attemptId);
        db.update("UPDATE execution_slots SET task_id=NULL,attempt_id=NULL,lease_until=NULL,available_after_ms=? WHERE slot_id=1",after);
        refresh(a.batchId);
    });}
    static double number(Map<String,Object> result,String key){
        Object v=result.get(key);if(!(v instanceof Number n)||!Double.isFinite(n.doubleValue()))throw new ApiError(502,"模型统计字段无效："+key);
        return n.doubleValue();
    }
    void validate(Attempt a,Map<String,Object> result){
        if(!a.taskId.equals(result.get("taskId"))||!a.attemptId.equals(result.get("attemptId"))||!a.modelId.equals(result.get("modelId")))throw new ApiError(502,"模型结果不属于当前任务");
        Object ver=result.get("modelVersion");if(!(ver instanceof String s)||!s.matches("[0-9a-f]{64}"))throw new ApiError(502,"模型版本标识无效");
        String asset="pred-"+a.taskId+"-"+a.attemptId+".png";
        if(!asset.equals(result.get("predMaskAssetId")))throw new ApiError(502,"预测资产标识不匹配");
        boolean seg=a.modelId.equals("sam-scfp-whu");
        double px=number(result,seg?"foregroundPixels":"changedPx"),ratio=number(result,seg?"foregroundRatio":"changedRatio"),objects=number(result,seg?"componentCount":"objectCount");
        if(px<0||px!=Math.rint(px)||objects<0||objects!=Math.rint(objects)||objects>px||ratio<0||ratio>1
            ||number(result,"width")!=a.width||number(result,"height")!=a.height)throw new ApiError(502,"预测统计范围无效");
        try{
            var image=ImageIO.read(storage.asset(asset).toFile());
            if(image==null||image.getWidth()!=a.width||image.getHeight()!=a.height)throw new ApiError(502,"预测掩膜尺寸不匹配");
            long count=0;
            for(int y=0;y<a.height;y++)for(int x=0;x<a.width;x++){
                int value=image.getRaster().getSample(x,y,0);
                if(value!=0&&value!=255)throw new ApiError(502,"预测掩膜不是二值图像");
                if(value==255)count++;
            }
            if(count!=(long)px||Math.abs(ratio-count/((double)a.width*a.height))>1e-10)throw new ApiError(502,"掩膜像素与统计结果不一致");
        }catch(ApiError e){throw e;}catch(Exception e){throw new ApiError(502,"预测掩膜无法核对");}
    }
    public boolean complete(Attempt a,Map<String,Object> result){
        validate(a,result);
        var enriched=new LinkedHashMap<String,Object>(result);
        if(a.modelId.equals("sam-scfp-whu"))counts.add(enriched,storage.asset(result.get("predMaskAssetId").toString()));
        return Boolean.TRUE.equals(tx.execute(transaction->{
            var slot=lockSlot();long now=System.currentTimeMillis();
            if(!a.attemptId.equals(slot.get("attempt_id"))||((Number)slot.get("lease_until")).longValue()<now)return false;
            var row=db.queryForMap("SELECT t.status,t.attempt_id,b.status AS batch_status FROM sample_tasks t JOIN batches b ON b.id=t.batch_id WHERE t.id=? FOR UPDATE",a.taskId);
            if(!"RUNNING".equals(row.get("status"))||!"RUNNING".equals(row.get("batch_status"))||!a.attemptId.equals(row.get("attempt_id")))return false;
            String serialized;
            try{serialized=json.writeValueAsString(enriched);}catch(Exception e){throw new ApiError(502,"模型结果无法保存");}
            boolean seg=a.modelId.equals("sam-scfp-whu");
            db.update("INSERT INTO "+(seg?"segmentation_task_results":"prediction_results")+" VALUES(?,?,?,?,?,?,?,?,?,?,?)",a.taskId,a.batchId,a.attemptId,a.modelId,result.get("modelVersion"),(long)number(result,seg?"foregroundPixels":"changedPx"),number(result,seg?"foregroundRatio":"changedRatio"),(int)number(result,seg?"componentCount":"objectCount"),result.get("predMaskAssetId"),serialized,now);
            try{db.update("INSERT INTO assets VALUES(?,?,?,?,?,?)",result.get("predMaskAssetId"),a.batchId,a.taskId,"predMask","image/png",Files.size(storage.asset(result.get("predMaskAssetId").toString())));}
            catch(java.io.IOException e){throw new ApiError(502,"预测资产不可读");}
            if(seg&&result.get("overlayAssetId")!=null){
                String overlay="overlay-"+a.taskId+"-"+a.attemptId+".png";
                if(!overlay.equals(result.get("overlayAssetId")))throw new ApiError(502,"叠加资产标识不一致");
                try{var image=ImageIO.read(storage.asset(overlay).toFile());if(image==null||image.getWidth()!=a.width||image.getHeight()!=a.height)throw new ApiError(502,"叠加图尺寸不一致");db.update("INSERT INTO assets VALUES(?,?,?,?,?,?)",overlay,a.batchId,a.taskId,"overlay","image/png",Files.size(storage.asset(overlay)));}
                catch(java.io.IOException e){throw new ApiError(502,"叠加图不可读");}
            }
            db.update("UPDATE sample_tasks SET status='SUCCEEDED',lease_until=NULL,updated_ms=? WHERE id=? AND attempt_id=?",now,a.taskId,a.attemptId);
            db.update("UPDATE task_attempts SET status='SUCCEEDED',finished_ms=? WHERE id=?",now,a.attemptId);
            db.update("UPDATE execution_slots SET task_id=NULL,attempt_id=NULL,lease_until=NULL WHERE slot_id=1");
            refresh(a.batchId);return true;
        }));
    }
    public void fail(Attempt a,String message){tx.executeWithoutResult(transaction->{
        var slot=lockSlot();if(!a.attemptId.equals(slot.get("attempt_id")))return;
        long now=System.currentTimeMillis();String error=message==null?"执行失败":message.substring(0,Math.min(500,message.length()));
        int n=db.update("UPDATE sample_tasks SET status='FAILED',lease_until=NULL,error_message=?,updated_ms=? WHERE id=? AND attempt_id=? AND status='RUNNING'",error,now,a.taskId,a.attemptId);
        if(n==1)db.update("UPDATE task_attempts SET status='FAILED',finished_ms=?,error_message=? WHERE id=?",now,error,a.attemptId);
        db.update("UPDATE execution_slots SET task_id=NULL,attempt_id=NULL,lease_until=NULL WHERE slot_id=1");refresh(a.batchId);
    });}
}

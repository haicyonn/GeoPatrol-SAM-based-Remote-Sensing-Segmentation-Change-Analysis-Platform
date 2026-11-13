package io.xunbian.backend;
import java.util.*;
import java.nio.file.*;
import javax.imageio.ImageIO;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.support.TransactionTemplate;

@Service
public class ReviewService {
 final JdbcTemplate db;final TransactionTemplate tx;final BatchAccess access;final AnnotationLayer layer;final PythonClient python;final Storage storage;
 public ReviewService(JdbcTemplate db,TransactionTemplate tx,BatchAccess access,AnnotationLayer layer,PythonClient python,Storage storage){this.db=db;this.tx=tx;this.access=access;this.layer=layer;this.python=python;this.storage=storage;}
 public Map<String,Object> get(String batch,String sample){access.requireRead(batch);layer.sample(batch,sample);return layer.current(batch,sample);}
 public Object history(String batch,String sample){access.requireRead(batch);layer.sample(batch,sample);var list=new ArrayList<Map<String,Object>>();for(var row:db.queryForList("SELECT * FROM sample_annotation_history WHERE batch_id=? AND sample_id=? ORDER BY version DESC LIMIT 200",batch,sample))list.add(layer.record(row));return Map.of("history",list);}
 public Object export(String batch){access.requireRead(batch);return Map.of("batchId",batch,"records",layer.exports(batch));}
 public Map<String,Object> save(String batch,String sample,Map<String,Object> body){
  access.requireEdit(batch);var original=layer.sample(batch,sample);Object version=body.get("expectedVersion");
  if(!(version instanceof Number n)||!Double.isFinite(n.doubleValue())||n.doubleValue()<0||n.doubleValue()!=Math.rint(n.doubleValue())||n.doubleValue()>Long.MAX_VALUE)throw new ApiError(400,"复核版本无效");
  long expected=((Number)version).longValue();String verdict=Objects.toString(body.get("verdict"),"");Object rawNote=body.getOrDefault("note","");
  boolean seg="building_segmentation".equals(db.queryForObject("SELECT task_type FROM batches WHERE id=?",String.class,batch));
  if(!(seg?Set.of("unreviewed","accepted_mask","needs_correction","uncertain"):Set.of("unreviewed","confirmed_change","false_change","uncertain")).contains(verdict)||!(rawNote instanceof String note)||note.length()>2000)throw new ApiError(400,"复核结论或备注无效");
  Map<String,Object> correction=null;String aid=null;
  if(body.get("correction")!=null){
   if(!(body.get("correction") instanceof Map c)||!Set.of("draft","approved").contains(c.get("status"))||!(c.get("geometry") instanceof Map))throw new ApiError(400,"修正几何或状态无效");
   if(verdict.equals("unreviewed")&&!c.get("status").equals("draft"))throw new ApiError(400,"审批请提供复核结论");
   Object size=original.get("sourceSize");if(!(size instanceof List dims)||dims.size()!=2)throw new ApiError(409,"原影像尺寸未登记，不能绘制");
   String cid=UUID.randomUUID().toString();aid="correction-"+cid+".png";
   try{
    correction=new LinkedHashMap<>(python.call("/internal/corrections",Map.of("correctionId",cid,"width",dims.get(0),"height",dims.get(1),"geometry",c.get("geometry")),30));
    if(!aid.equals(correction.get("maskAssetId")))throw new ApiError(502,"修正资产标识不一致");
    var image=ImageIO.read(storage.asset(aid).toFile());if(image==null||image.getWidth()!=((Number)dims.get(0)).intValue()||image.getHeight()!=((Number)dims.get(1)).intValue())throw new ApiError(502,"修正掩膜尺寸不一致");
    long count=0;for(int y=0;y<image.getHeight();y++)for(int x=0;x<image.getWidth();x++)if(image.getRaster().getSample(x,y,0)>0)count++;
    if(!(correction.get("changedPx") instanceof Number px)||px.longValue()!=count)throw new ApiError(502,"修正掩膜像素统计不一致");
    if(!(correction.get("changedRatio") instanceof Number ratio)||Math.abs(ratio.doubleValue()-count/(double)(image.getWidth()*image.getHeight()))>1e-10)throw new ApiError(502,"修正掩膜比例不一致");
    correction.put("status",c.get("status"));correction.put("maskUrl","/api/v1/assets/"+aid);
    if(seg){correction.put("foregroundPixels",correction.remove("changedPx"));correction.put("foregroundRatio",correction.remove("changedRatio"));correction.put("componentCount",correction.remove("objectCount"));layer.counts.add(correction,storage.asset(aid));}
   }catch(RuntimeException e){storage.removeUncommitted(List.of(aid));throw e;}catch(Exception e){storage.removeUncommitted(List.of(aid));throw new ApiError(502,"修正掩膜读取失败");}
  }else if(verdict.equals("unreviewed"))throw new ApiError(400,"未复核状态仅用于保存绘制草稿");
  final Map<String,Object> rendered=correction;final String asset=aid;String author=access.accounts.current().id();
  try{return tx.execute(t->{
   db.queryForObject("SELECT review_version FROM batches WHERE id=? FOR UPDATE",Long.class,batch);access.requireEdit(batch);
   var current=layer.current(batch,sample);if(((Number)current.get("version")).longValue()!=expected)throw new ApiError(409,"样本已被其他成员更新，请载入最新版本；输入已保留",Map.of("current",current));
   Object approved=current.get("approvedCorrection"),draft=current.get("correction");
   if(body.containsKey("correction")){draft=rendered;if(rendered==null)approved=null;else if("approved".equals(rendered.get("status")))approved=rendered;}
   var record=new LinkedHashMap<String,Object>();record.put("verdict",rendered!=null&&"draft".equals(rendered.get("status"))&&approved!=null&&verdict.equals("unreviewed")?current.get("verdict"):verdict);record.put("note",rawNote);record.put("correction",draft);
   long now=System.currentTimeMillis(),next=expected+1;
   if(asset!=null){try{db.update("INSERT INTO assets(asset_id,batch_id,task_id,role,mime_type,byte_size) VALUES(?,?,NULL,'manual_mask','image/png',?)",asset,batch,Files.size(storage.asset(asset)));}catch(java.io.IOException e){throw new ApiError(500,"资产登记失败");}}
   String value=layer.encode(record),ap=approved==null?null:layer.encode(approved);
   db.update("INSERT INTO sample_annotations VALUES(?,?,?,?,?,?,?) ON DUPLICATE KEY UPDATE version=VALUES(version),data_json=VALUES(data_json),approved_json=VALUES(approved_json),author_id=VALUES(author_id),updated_ms=VALUES(updated_ms)",batch,sample,next,value,ap,author,now);
   db.update("INSERT INTO sample_annotation_history VALUES(?,?,?,?,?,?,?)",batch,sample,next,value,ap,author,now);
   db.update("UPDATE batches SET review_version=review_version+1,updated_ms=? WHERE id=?",now,batch);return layer.current(batch,sample);
  });}catch(RuntimeException e){if(asset!=null)storage.removeUncommitted(List.of(asset));throw e;}
 }
}

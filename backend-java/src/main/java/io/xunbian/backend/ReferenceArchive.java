package io.xunbian.backend;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.*;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.support.TransactionTemplate;

/** Explicit readonly historical reference. No fabricated inference tasks or weight versions. */
@Service
public class ReferenceArchive {
    private static final org.slf4j.Logger LOG=org.slf4j.LoggerFactory.getLogger(ReferenceArchive.class);
    final JdbcTemplate db;final ObjectMapper json;final BatchAccess access;
    static final Set<String> SAMPLE_FIELDS=Set.of("id","name","stem","areaName","sampleTime","captureTime","createdAt","addedAt","pred","eventType","eventTypeZh","morphologyType","morphologyLabel","primaryEventType","primaryEventName","scale","changedPx","changedRatio","objectCount","mainBbox","gridPosition","confidence","reviewRequired","reviewStatus","reviewReason","reviewLevel","reviewReasonCodes","qualityFlags","objects","evidenceSummary","maskQuality","sourceSize","manualReviewComplete");
    static final Set<String> ASSET_ROLES=Set.of("pre","post","mask","predMask","overlay","predOverlay");
    public ReferenceArchive(JdbcTemplate db,ObjectMapper json,BatchAccess access,AccountService accounts,TransactionTemplate tx,
        @Value("${xunbian.archive-file:}") String configured){
        this.db=db;this.json=json;this.access=access;
        if(configured==null||configured.isBlank())return;
        try{
            Path source=Path.of(configured).toAbsolutePath().normalize();if(!Files.isRegularFile(source))throw new IllegalStateException("配置的历史样本文件不存在");
            if(Files.size(source)>120_000_000)throw new IllegalStateException("历史数据超出导入大小限制");
            byte[] bytes=Files.readAllBytes(source);String digest=sha(bytes);
            String id=UUID.nameUUIDFromBytes(("reference:"+source+":"+digest).getBytes(StandardCharsets.UTF_8)).toString();
            if(db.queryForObject("SELECT COUNT(*) FROM reference_sources WHERE batch_id=?",Integer.class,id)>0){supplement(id,bytes);return;}
            Map<String,Object> raw=json.readValue(bytes,new TypeReference<>(){});
            var regions=new ArrayList<Map<String,Object>>();var facts=new ArrayList<Map<String,Object>>();var assets=new LinkedHashMap<String,String[]>();var ids=new HashSet<String>();int total=0;
            for(var region:(List<Map<String,Object>>)raw.get("regions")){
                String rid=Objects.toString(region.get("id"),"");if(!rid.matches("[a-z][a-z0-9_-]{0,63}"))throw new IllegalArgumentException("历史区域标识无效");
                var projected=new LinkedHashMap<String,Object>();for(String key:List.of("id","short","name","role","status","sceneGuide","areaName","eventKey","eventName"))if(region.containsKey(key))projected.put(key,safe(region.get(key),0));
                var items=new ArrayList<Map<String,Object>>();var sqlItems=new ArrayList<Map<String,Object>>();
                for(var original:(List<Map<String,Object>>)region.get("items")){
                    String sid=Objects.toString(original.get("id"),"");if(sid.isBlank()||sid.length()>191||!ids.add(sid))throw new IllegalArgumentException("历史样本标识缺失或重复");
                    var item=new LinkedHashMap<String,Object>();for(String key:SAMPLE_FIELDS)if(original.containsKey(key))item.put(key,safe(original.get(key),0));
                    var urls=new LinkedHashMap<String,Object>();Object rawAssets=original.get("assets");
                    if(rawAssets instanceof Map<?,?> map)for(String role:ASSET_ROLES){
                        String relative=Objects.toString(map.get(role),"").replace('\\','/');if(!allowed(relative))continue;
                        String aid=sha((id+":"+relative).getBytes(StandardCharsets.UTF_8));String mime=mime(relative);if(mime==null)continue;
                        assets.put(aid,new String[]{relative,mime});urls.put(role,"/api/v1/reference-assets/"+aid);
                    }
                    if(urls.containsKey("mask"))urls.put("predMask",urls.get("mask"));else if(urls.containsKey("predMask"))urls.put("mask",urls.get("predMask"));
                    item.put("assets",urls);items.add(item);
                    var sql=new LinkedHashMap<String,Object>();for(String key:List.of("id","name","stem","changedPx","changedRatio","objectCount","reviewLevel","reviewRequired","reviewStatus","manualReviewComplete","maskQuality","qualityFlags","reviewReasonCodes"))if(item.containsKey(key))sql.put(key,item.get(key));sqlItems.add(sql);
                }
                projected.put("items",items);projected.put("totalRows",items.size());projected.put("displayRows",items.size());regions.add(projected);
                facts.add(Map.of("id",rid,"short",region.getOrDefault("short",rid.toUpperCase(Locale.ROOT)),"items",sqlItems));total+=items.size();
            }
            if(total<1||total>10000)throw new IllegalArgumentException("历史样本数量无效");
            String factJson=json.writeValueAsString(Map.of("regions",facts)),consoleJson=json.writeValueAsString(Map.of("regions",regions));
            if(factJson.getBytes(StandardCharsets.UTF_8).length>7_500_000)throw new IllegalStateException("历史问数投影超过模型服务预算");
            String owner=db.queryForObject("SELECT id FROM app_users WHERE role='ADMIN' ORDER BY created_ms LIMIT 1",String.class);int count=total;long now=System.currentTimeMillis();
            String provenance=json.writeValueAsString(Map.of("kind","historical_reference","sourceFile",source.getFileName().toString(),"sourceSha256",digest,"samples",total,"assetIndexes",assets.size(),"note","原版样本快照的只读参考；没有在此处重新推理。"));
            LOG.info("Import historical reference: {} samples, {} asset indexes; facts {} bytes, console {} bytes",total,assets.size(),factJson.getBytes(StandardCharsets.UTF_8).length,consoleJson.getBytes(StandardCharsets.UTF_8).length);
            tx.executeWithoutResult(t->{
                db.update("INSERT INTO batches(id,label,status,total,created_ms,updated_ms,owner_id,source_type,provenance_json) VALUES(?,?,'PUBLISHED',?,?,?,?, 'ARCHIVE',?)",id,"历史样本参考（只读）",count,now,now,owner,provenance);
                db.update("INSERT INTO analysis_datasets VALUES(?,?,?,?)",id,factJson,digest,now);
                db.update("INSERT INTO reference_sources VALUES(?,?,?)",id,source.getParent().toString(),consoleJson);
                var entries=new ArrayList<>(assets.entrySet());
                // Bounded multi-row statements avoid driver-specific batch pipelining on MySQL.
                for(int start=0;start<entries.size();start+=200){
                    var chunk=entries.subList(start,Math.min(start+200,entries.size()));var args=new ArrayList<Object>();
                    for(var entry:chunk)Collections.addAll(args,entry.getKey(),id,entry.getValue()[0],entry.getValue()[1]);
                    db.update("INSERT INTO reference_assets VALUES"+String.join(",",Collections.nCopies(chunk.size(),"(?,?,?,?)")),args.toArray());
                }
            });
            LOG.info("Historical reference committed: {} samples, {} asset indexes",total,assets.size());
            supplement(id,bytes);
        }catch(RuntimeException e){throw e;}catch(Exception e){throw new IllegalStateException("历史参考数据接入失败",e);}
    }
    static String sha(byte[] data){try{return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(data));}catch(Exception e){throw new IllegalStateException(e);}}
    static Object safe(Object v,int depth){
        if(depth>6)return null;
        if(v instanceof String s)return s.length()>4000?s.substring(0,4000):s;
        if(v instanceof Map<?,?> m){var result=new LinkedHashMap<String,Object>();for(var e:m.entrySet()){String k=e.getKey().toString(),low=k.toLowerCase(Locale.ROOT);if(low.startsWith("gt")||low.contains("groundtruth")||low.contains("private")||Set.of("maskf1","maskiou","maskprecision","maskrecall","f1","iou","precision","recall").contains(low))continue;result.put(k,safe(e.getValue(),depth+1));}return result;}
        if(v instanceof List<?> a)return a.stream().limit(30).map(x->safe(x,depth+1)).toList();return v;
    }
    static boolean allowed(String value){
        if(!(value.startsWith("assets/")||value.startsWith("deployed_assets/"))||value.contains(":"))return false;
        for(String p:value.split("/"))if(Set.of("..",".","gt","ground_truth","gt_mask","gt_overlay").contains(p.toLowerCase(Locale.ROOT)))return false;
        return value.length()<1024;
    }
    static String mime(String path){String low=path.toLowerCase(Locale.ROOT);if(low.endsWith(".png"))return "image/png";if(low.endsWith(".jpg")||low.endsWith(".jpeg"))return "image/jpeg";if(low.endsWith(".webp"))return "image/webp";return null;}
    void supplement(String id,byte[] bytes) throws Exception {
        if(db.queryForObject("SELECT COUNT(*) FROM reference_details WHERE batch_id=?",Integer.class,id)>0)return;
        var raw=json.readValue(bytes,Map.class);var details=new LinkedHashMap<String,Object>();
        for(var region:(List<Map<String,Object>>)raw.get("regions"))for(var item:(List<Map<String,Object>>)region.get("items")){
            var extra=new LinkedHashMap<String,Object>();for(String key:List.of("report","evidencePack","publicFacts","verifierChecks","road_changed","road_region_count","road_regions","road_location","road_reason","building_context"))if(item.containsKey(key))extra.put(key,safe(item.get(key),0));
            if(!extra.isEmpty())details.put(item.get("id").toString(),extra);
        }
        db.update("INSERT IGNORE INTO reference_details VALUES(?,?)",id,json.writeValueAsString(details));
    }
    Map<String,Object> data(String id){
        access.requireRead(id);var rows=db.queryForList("SELECT console_json FROM reference_sources WHERE batch_id=?",id);if(rows.isEmpty())throw new ApiError(404,"历史参考数据不存在");
        try{
            Map<String,Object> data=json.readValue(rows.getFirst().get("console_json").toString(),new TypeReference<>(){});
            var extras=db.queryForList("SELECT details_json FROM reference_details WHERE batch_id=?",id);
            if(!extras.isEmpty()){
                Map<String,Object> detail=json.readValue(extras.getFirst().get("details_json").toString(),Map.class);
                for(var r:(List<Map<String,Object>>)data.get("regions"))for(var i:(List<Map<String,Object>>)r.get("items"))if(detail.get(i.get("id")) instanceof Map extra)i.putAll(extra);
            }return data;
        }catch(Exception e){throw new ApiError(500,"历史参考数据无法读取");}
    }
    public ResponseEntity<?> asset(String id){
        var rows=db.queryForList("SELECT a.*,s.source_root FROM reference_assets a JOIN reference_sources s ON s.batch_id=a.batch_id WHERE a.asset_id=?",id);if(rows.isEmpty())throw new ApiError(404,"历史影像未登记");var row=rows.getFirst();access.requireRead(row.get("batch_id").toString());
        try{Path root=Path.of(row.get("source_root").toString()).toRealPath(),path=root.resolve(row.get("relative_path").toString()).toRealPath();if(!path.startsWith(root)||!Files.isRegularFile(path)||!allowed(row.get("relative_path").toString()))throw new ApiError(404,"历史影像不可用");return ResponseEntity.ok().contentType(MediaType.parseMediaType(row.get("mime_type").toString())).header("X-Content-Type-Options","nosniff").body(new FileSystemResource(path));}
        catch(ApiError e){throw e;}catch(Exception e){throw new ApiError(404,"历史影像文件不存在");}
    }
}

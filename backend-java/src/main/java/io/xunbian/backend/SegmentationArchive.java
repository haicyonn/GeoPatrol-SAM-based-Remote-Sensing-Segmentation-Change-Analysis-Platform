package io.xunbian.backend;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;
import org.springframework.transaction.support.TransactionTemplate;

/** Register immutable existing segmentation facts; never create fake inference tasks. */
@Component @org.springframework.core.annotation.Order(0)
public class SegmentationArchive implements ApplicationRunner {
    final JdbcTemplate db;final TransactionTemplate tx;final ObjectMapper json;final BuildingCountService counts;
    public SegmentationArchive(JdbcTemplate db,TransactionTemplate tx,ObjectMapper json,SchemaMigration migration,BuildingCountService counts){this.db=db;this.tx=tx;this.json=json;this.counts=counts;}
    public void run(ApplicationArguments args){
        sync();
    }
    @org.springframework.scheduling.annotation.Scheduled(fixedDelay=10000,initialDelay=10000)
    public void sync(){
        if(db.queryForObject("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema=DATABASE() AND table_name='segmentation_batches'",Integer.class)==0)return;
        var owners=db.queryForList("SELECT id FROM app_users WHERE role='ADMIN' AND enabled=TRUE ORDER BY created_ms LIMIT 1");if(owners.isEmpty())return;
        for(var b:db.queryForList("SELECT * FROM segmentation_batches ORDER BY created_ms")){
            String id=b.get("batch_id").toString();
            tx.executeWithoutResult(t->{
                if(db.queryForObject("SELECT COUNT(*) FROM batches WHERE id=?",Integer.class,id)>0)return;
                try{
                    var regions=new TreeMap<String,Map<String,Object>>();
                    for(var s:db.queryForList("SELECT s.*,p.foreground_pixels,p.foreground_ratio,p.component_count FROM segmentation_samples s JOIN segmentation_predictions p ON p.sample_id=s.sample_id WHERE s.batch_id=? ORDER BY s.region_id,s.name",id)){
                        String rid=s.get("region_id").toString();
                        var r=regions.computeIfAbsent(rid,k->{var v=new LinkedHashMap<String,Object>();v.put("id",k);v.put("short",k.toUpperCase(Locale.ROOT));v.put("items",new ArrayList<>());return v;});
                        var item=new LinkedHashMap<String,Object>();item.put("id",s.get("sample_id"));item.put("name",s.get("name"));item.put("sourceSize",List.of(s.get("width"),s.get("height")));
                        item.put("foregroundPixels",s.get("foreground_pixels"));item.put("foregroundRatio",s.get("foreground_ratio"));item.put("componentCount",s.get("component_count"));
                        var assets=new LinkedHashMap<String,Object>();
                        for(var a:db.queryForList("SELECT asset_id,role FROM segmentation_assets WHERE sample_id=?",s.get("sample_id"))){String role=a.get("role").toString();assets.put(role.equals("prediction_mask")?"mask":role,"/api/v1/segmentation-assets/"+a.get("asset_id"));}
                        item.put("assets",assets);((List<Map<String,Object>>)r.get("items")).add(item);
                    }
                    if(regions.values().stream().mapToInt(r->((List)r.get("items")).size()).sum()!=((Number)b.get("sample_count")).intValue())throw new ApiError(409,"已有分割批次结果不完整");
                    var data=new LinkedHashMap<String,Object>();data.put("taskType","building_segmentation");data.put("regions",new ArrayList<>(regions.values()));data.put("models",List.of(Map.of("modelId","sam-scfp-whu","modelVersion",b.get("model_id"))));
                    counts.enrichData(data);
                    String encoded=json.writeValueAsString(data);String digest=HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(encoded.getBytes(java.nio.charset.StandardCharsets.UTF_8)));
                    long now=System.currentTimeMillis();
                    db.update("INSERT INTO batches(id,label,status,model_id,total,created_ms,updated_ms,owner_id,source_type,provenance_json,task_type) VALUES(?,?,'PUBLISHED','sam-scfp-whu',?,?,?,?, 'SEGMENTATION',?,'building_segmentation')",id,b.get("label"),b.get("sample_count"),b.get("created_ms"),now,owners.getFirst().get("id"),json.writeValueAsString(Map.of("kind","segmentation_import","manifestDigest",b.get("manifest_digest"),"originalBatchId",id)));
                    db.update("INSERT INTO analysis_datasets VALUES(?,?,?,?)",id,encoded,digest,now);
                }catch(ApiError e){throw e;}catch(Exception e){throw new ApiError(500,"分割批次登记失败："+e.getClass().getSimpleName());}
            });
        }
    }
    static Path localPath(String value){
        if(value.matches("^[A-Za-z]:[\\\\/].*")){String normalized=value.replace('\\','/');return Path.of("/mnt/"+Character.toLowerCase(value.charAt(0))+normalized.substring(2));}
        return Path.of(value);
    }
}

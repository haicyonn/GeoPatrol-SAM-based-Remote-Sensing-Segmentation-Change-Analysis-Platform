package io.xunbian.backend;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.*;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

/** Separate human records; immutable input datasets/prediction assets are never overwritten. */
@Service
public class AnnotationLayer {
    final JdbcTemplate db;final ObjectMapper json;final BuildingCountService counts;
    public AnnotationLayer(JdbcTemplate db,ObjectMapper json,BuildingCountService counts){this.db=db;this.json=json;this.counts=counts;}
    static String basis(Object value){String b=Objects.toString(value,"effective");if(!Set.of("effective","original").contains(b))throw new ApiError(400,"结果口径无效");return b;}
    Map<String,Object> decode(Object raw){try{return raw==null?new LinkedHashMap<>():json.readValue(raw.toString(),Map.class);}catch(Exception e){throw new ApiError(500,"人工修正记录无法读取");}}
    String encode(Object value){try{return json.writeValueAsString(value);}catch(Exception e){throw new ApiError(400,"人工记录格式无效");}}
    Map<String,Object> record(Map<String,Object> row){
        var value=decode(row.get("data_json"));value.put("version",row.get("version"));value.put("authorId",row.get("author_id"));value.put("updatedAt",row.get("updated_ms"));
        value.put("approvedCorrection",row.get("approved_json")==null?null:decode(row.get("approved_json")));return value;
    }
    Map<String,Object> current(String batch,String sample){var rows=db.queryForList("SELECT * FROM sample_annotations WHERE batch_id=? AND sample_id=?",batch,sample);return rows.isEmpty()?new LinkedHashMap<>(Map.of("version",0L,"verdict","unreviewed","note","")):record(rows.getFirst());}
    Map<String,Object> source(String batch){
        var rows=db.queryForList("SELECT data_json FROM analysis_datasets WHERE batch_id=?",batch);if(rows.isEmpty())throw new ApiError(409,"批次尚未发布");return decode(rows.getFirst().get("data_json"));
    }
    Map<String,Object> sample(String batch,String sample){
        var archive=db.queryForList("SELECT console_json FROM reference_sources WHERE batch_id=?",batch);
        var data=archive.isEmpty()?source(batch):decode(archive.getFirst().get("console_json"));
        for(var region:(List<Map<String,Object>>)data.get("regions"))for(var item:(List<Map<String,Object>>)region.get("items"))if(sample.equals(item.get("id"))){
            if(!item.containsKey("sourceSize")){
                var dims=db.queryForList("SELECT width,height FROM sample_tasks WHERE batch_id=? AND id=?",batch,sample);
                if(!dims.isEmpty())item.put("sourceSize",List.of(dims.getFirst().get("width"),dims.getFirst().get("height")));
            }
            return item;
        }
        throw new ApiError(404,"本批不存在该样本");
    }
    void overlay(String batch,Map<String,Object> data,String basis,boolean rich){
        counts.enrichData(data);
        var metricFields=new ArrayList<String>(List.of("foregroundPixels","foregroundRatio","componentCount","changedPx","changedRatio","objectCount","maskQuality","mainBbox","objects","reviewLevel","qualityFlags","reviewReasonCodes"));metricFields.addAll(BuildingCountService.FIELDS);
        var records=new HashMap<String,Map<String,Object>>();
        for(var r:db.queryForList("SELECT * FROM sample_annotations WHERE batch_id=? ORDER BY sample_id",batch))records.put(r.get("sample_id").toString(),record(r));
        for(var region:(List<Map<String,Object>>)data.get("regions"))for(var item:(List<Map<String,Object>>)region.get("items")){
            var r=records.get(item.get("id").toString());if(r==null)continue;
            boolean done=!"unreviewed".equals(r.get("verdict"));item.put("manualReviewComplete",done);item.put("reviewVerdict",r.get("verdict"));item.put("reviewVersion",r.get("version"));
            var review=new LinkedHashMap<String,Object>();for(String key:List.of("version","verdict","note","authorId","updatedAt"))review.put(key,r.get(key));if(done)item.put("manualReview",review);else item.remove("manualReview");
            if(rich){item.put("correction",r.get("correction"));item.put("approvedCorrection",r.get("approvedCorrection"));}
            if(r.get("approvedCorrection") instanceof Map correction){
                if("building_segmentation".equals(data.get("taskType"))&&!BuildingCountService.METHOD.equals(correction.get("countMethod")))counts.add(correction,counts.maskPath(correction.get("maskAssetId")));
                if(rich){var original=new LinkedHashMap<String,Object>();for(String key:metricFields)if(item.containsKey(key))original.put(key,item.get(key));item.put("originalPrediction",original);}
                item.put("hasApprovedCorrection",true);
                if("effective".equals(basis)){
                    for(String key:metricFields)if(correction.containsKey(key))item.put(key,correction.get(key));
                    item.put("mainBbox",correction.get("bbox"));item.put("reviewLevel","unknown");item.put("reviewRequired",false);item.put("metricSource","manual_polygon");
                    if(rich){var assets=new LinkedHashMap<String,Object>((Map)item.getOrDefault("assets",Map.of()));assets.put("correctedMask",correction.get("maskUrl"));item.put("assets",assets);}
                }
            }
        }
    }
    List<Map<String,Object>> exports(String batch){var out=new ArrayList<Map<String,Object>>();for(var row:db.queryForList("SELECT * FROM sample_annotation_history WHERE batch_id=? ORDER BY sample_id,version",batch)){var r=record(row);r.put("sampleId",row.get("sample_id"));out.add(r);}return out;}
}

package io.xunbian.backend;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.awt.image.BufferedImage;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;
import javax.imageio.ImageIO;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.core.annotation.Order;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

/** Versioned counting of candidate footprints; never rewrites a prediction mask. */
@Service @Order(10)
public class BuildingCountService implements ApplicationRunner {
    static final org.slf4j.Logger LOG=org.slf4j.LoggerFactory.getLogger(BuildingCountService.class);
    static final String METHOD="cc8-min10-v1";
    static final int MIN_AREA=10;
    static final List<String> FIELDS=List.of("estimatedBuildingCount","filteredFragmentCount","boundaryBuildingCount","countMinAreaPixels","countMethod","countMaskSha256","buildingCandidates");
    final JdbcTemplate db;final Storage storage;final ObjectMapper json;
    public BuildingCountService(JdbcTemplate db,Storage storage,ObjectMapper json,SchemaMigration migration){this.db=db;this.storage=storage;this.json=json;}

    static Map<String,Object> measure(BufferedImage image){
        int w=image.getWidth(),h=image.getHeight(),size=Math.multiplyExact(w,h);
        if(w<1||h<1||size>16_777_216)throw new ApiError(502,"计数影像尺寸超限");
        boolean[] seen=new boolean[size];int[] queue=new int[size];
        var candidates=new ArrayList<Map<String,Object>>();int raw=0,filtered=0,boundary=0;
        for(int origin=0;origin<size;origin++){
            if(seen[origin]||image.getRaster().getSample(origin%w,origin/w,0)==0)continue;
            raw++;int head=0,tail=1,minX=w,minY=h,maxX=0,maxY=0;long sumX=0,sumY=0;queue[0]=origin;seen[origin]=true;
            while(head<tail){
                int pixel=queue[head++],x=pixel%w,y=pixel/w;
                minX=Math.min(minX,x);minY=Math.min(minY,y);maxX=Math.max(maxX,x);maxY=Math.max(maxY,y);sumX+=x;sumY+=y;
                for(int yy=Math.max(0,y-1);yy<=Math.min(h-1,y+1);yy++)for(int xx=Math.max(0,x-1);xx<=Math.min(w-1,x+1);xx++){
                    int p=yy*w+xx;if(!seen[p]&&image.getRaster().getSample(xx,yy,0)>0){seen[p]=true;queue[tail++]=p;}
                }
            }
            if(tail<MIN_AREA){filtered++;continue;}
            boolean touches=minX==0||minY==0||maxX==w-1||maxY==h-1;if(touches)boundary++;
            var item=new LinkedHashMap<String,Object>();item.put("id",String.format(Locale.ROOT,"B%04d",candidates.size()+1));
            item.put("pixelArea",tail);item.put("bbox",List.of(minX,minY,maxX+1,maxY+1));
            item.put("centroid",List.of(sumX/(double)tail,sumY/(double)tail));item.put("touchesBoundary",touches);candidates.add(item);
        }
        var out=new LinkedHashMap<String,Object>();out.put("rawComponentCount",raw);out.put("estimatedBuildingCount",candidates.size());
        out.put("filteredFragmentCount",filtered);out.put("boundaryBuildingCount",boundary);out.put("countMinAreaPixels",MIN_AREA);out.put("countMethod",METHOD);out.put("buildingCandidates",candidates);return out;
    }

    Path maskPath(Object reference){
        String value=Objects.toString(reference,"");String legacy="/api/v1/segmentation-assets/";
        if(value.startsWith(legacy)){
            var rows=db.queryForList("SELECT path FROM segmentation_assets WHERE asset_id=?",value.substring(legacy.length()));
            if(rows.size()!=1)throw new ApiError(409,"计数掩膜未登记");return SegmentationArchive.localPath(rows.getFirst().get("path").toString());
        }
        return storage.asset(value.startsWith("/api/v1/assets/")?value.substring("/api/v1/assets/".length()):value);
    }
    void add(Map<String,Object> item,Path path){
        try{
            byte[] bytes=Files.readAllBytes(path);var image=ImageIO.read(new java.io.ByteArrayInputStream(bytes));
            if(image==null)throw new ApiError(409,"计数掩膜无法读取");
            var counted=measure(image);
            if(item.get("componentCount") instanceof Number n&&n.intValue()!=((Number)counted.get("rawComponentCount")).intValue())throw new ApiError(409,"计数掩膜与原始连通域统计不一致");
            for(String key:FIELDS)if(counted.containsKey(key))item.put(key,counted.get(key));
            item.put("countMaskSha256",HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes)));
        }catch(ApiError e){throw e;}catch(Exception e){throw new ApiError(409,"建筑量计算失败："+e.getClass().getSimpleName());}
    }
    boolean enrichData(Map<String,Object> data){
        if(!"building_segmentation".equals(data.get("taskType")))return false;
        boolean changed=false;
        for(var region:(List<Map<String,Object>>)data.get("regions"))for(var item:(List<Map<String,Object>>)region.get("items")){
            if(METHOD.equals(item.get("countMethod"))&&item.get("estimatedBuildingCount") instanceof Number)continue;
            var assets=(Map<String,Object>)item.get("assets");add(item,maskPath(assets.getOrDefault("predMask",assets.get("mask"))));changed=true;
        }return changed;
    }
    public void run(ApplicationArguments args){
        // Add derived facts without changing masks, model metrics or review history.
        for(var row:db.queryForList("SELECT d.batch_id,d.data_json,d.source_digest FROM analysis_datasets d JOIN batches b ON b.id=d.batch_id WHERE b.task_type='building_segmentation'")){
            try{
                Map<String,Object> data=json.readValue(row.get("data_json").toString(),Map.class);
                if(!enrichData(data))continue;
                String encoded=json.writeValueAsString(data),digest=HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(encoded.getBytes(java.nio.charset.StandardCharsets.UTF_8)));
                db.update("UPDATE analysis_datasets SET data_json=?,source_digest=? WHERE batch_id=? AND source_digest=?",encoded,digest,row.get("batch_id"),row.get("source_digest"));
            }catch(Exception e){LOG.warn("Building count backfill pending for batch {}: {}",row.get("batch_id"),e.getMessage());}
        }
        for(var row:db.queryForList("SELECT task_id,result_json,pred_mask_asset FROM segmentation_task_results")){
            try{Map<String,Object> result=json.readValue(row.get("result_json").toString(),Map.class);if(METHOD.equals(result.get("countMethod")))continue;add(result,storage.asset(row.get("pred_mask_asset").toString()));db.update("UPDATE segmentation_task_results SET result_json=? WHERE task_id=?",json.writeValueAsString(result),row.get("task_id"));}
            catch(Exception e){LOG.warn("Building count backfill pending for task {}: {}",row.get("task_id"),e.getMessage());}
        }
    }
}

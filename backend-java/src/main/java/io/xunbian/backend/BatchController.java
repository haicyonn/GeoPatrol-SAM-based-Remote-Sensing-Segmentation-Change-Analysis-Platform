package io.xunbian.backend;

import java.nio.file.Files;
import java.util.*;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.*;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/api/v1")
public class BatchController {
    final BatchService batches;final AnalysisService analysis;final PythonClient python;final JdbcTemplate db;final Storage storage;
    public BatchController(BatchService batches,AnalysisService analysis,PythonClient python,JdbcTemplate db,Storage storage){this.batches=batches;this.analysis=analysis;this.python=python;this.db=db;this.storage=storage;}
    @PostMapping(value="/batches",consumes=MediaType.MULTIPART_FORM_DATA_VALUE)
    ResponseEntity<?> create(@RequestParam String label,@RequestParam(defaultValue="custom") String regionId,
        @RequestParam(required=false) String assignments,@RequestParam(defaultValue="change_detection") String taskType,
        @RequestParam List<MultipartFile> before,@RequestParam(required=false) List<MultipartFile> after){
        return ResponseEntity.status(201).body(batches.create(label,regionId,assignments,before,after==null?List.of():after,taskType));
    }
    @GetMapping("/batches") Object list(){return Map.of("batches",batches.list());}
    @GetMapping("/batches/{id}") Object get(@PathVariable String id){return batches.get(id);}
    @PostMapping("/batches/{id}/start") Object start(@PathVariable String id,@RequestBody Map<String,Object> body){
        Object model=body.get("modelId");if(!(model instanceof String s))throw new ApiError(400,"请选择登记模型");return batches.start(id,s);
    }
    @PostMapping("/batches/{id}/retry-failed") Object retry(@PathVariable String id){return batches.retryFailed(id);}
    @PostMapping("/batches/{id}/cancel") Object cancel(@PathVariable String id){return batches.cancel(id);}
    @GetMapping("/batches/{id}/results") Object results(@PathVariable String id,@RequestParam(defaultValue="0") int offset,@RequestParam(defaultValue="50") int limit){return Map.of("batchId",id,"rows",batches.results(id,offset,limit));}
    @PostMapping("/batches/{id}/questions") Object question(@PathVariable String id,@RequestBody Map<String,Object> body){return analysis.question(id,body);}
    @GetMapping("/batches/{id}/members") Object members(@PathVariable String id){return batches.access.members(id);}
    @PostMapping("/batches/{id}/members/{user}") Object grant(@PathVariable String id,@PathVariable String user,@RequestBody Map<String,Object> body){
        if(!(body.get("permission") instanceof String p))throw new ApiError(400,"请指定协作权限");batches.access.grant(id,user,p);return Map.of("status","ok");
    }
    @GetMapping("/models") Object models(){return python.models();}
    @GetMapping("/health") Object health(){
        var status=new LinkedHashMap<String,Object>();
        status.put("application","xunbian-backend");
        try{status.put("database",db.queryForObject("SELECT 1",Integer.class)==1?"ok":"unavailable");}catch(Exception e){status.put("database","unavailable");}
        try{status.put("python",python.health());}catch(Exception e){status.put("python",Map.of("status","unavailable"));}
        status.put("status","ok");return status;
    }
    @GetMapping("/segmentation-assets/{assetId}") ResponseEntity<?> segmentationAsset(@PathVariable String assetId){
        var rows=db.queryForList("SELECT a.path,a.sha256,a.byte_size,s.batch_id FROM segmentation_assets a JOIN segmentation_samples s ON s.sample_id=a.sample_id WHERE a.asset_id=?",assetId);
        if(rows.isEmpty())throw new ApiError(404,"分割资产未登记");var row=rows.getFirst();batches.access.requireRead(row.get("batch_id").toString());
        var path=SegmentationArchive.localPath(row.get("path").toString());
        try{
            if(!Files.isRegularFile(path)||Files.size(path)!=((Number)row.get("byte_size")).longValue())throw new ApiError(409,"登记影像文件缺失或已改变");
            var digest=java.security.MessageDigest.getInstance("SHA-256");try(var in=Files.newInputStream(path)){byte[] buf=new byte[65536];int n;while((n=in.read(buf))!=-1)digest.update(buf,0,n);}
            if(!HexFormat.of().formatHex(digest.digest()).equals(row.get("sha256")))throw new ApiError(409,"登记影像完整性校验失败");
            String mime=Files.probeContentType(path);if(mime==null||!Set.of("image/png","image/jpeg","image/tiff","image/bmp").contains(mime))throw new ApiError(415,"影像格式不支持");
            return ResponseEntity.ok().contentType(MediaType.parseMediaType(mime)).header("X-Content-Type-Options","nosniff").body(new FileSystemResource(path));
        }catch(ApiError e){throw e;}catch(Exception e){throw new ApiError(503,"登记影像暂时无法读取");}
    }
    @GetMapping("/assets/{assetId}") ResponseEntity<?> asset(@PathVariable String assetId){
        var rows=db.queryForList("SELECT mime_type,batch_id FROM assets WHERE asset_id=?",assetId);
        if(rows.isEmpty())throw new ApiError(404,"资产未登记");
        batches.access.requireRead(rows.getFirst().get("batch_id").toString());
        var file=storage.asset(assetId);if(!Files.isRegularFile(file))throw new ApiError(404,"资产文件不存在");
        return ResponseEntity.ok().contentType(MediaType.parseMediaType(rows.getFirst().get("mime_type").toString()))
            .header("X-Content-Type-Options","nosniff").body(new FileSystemResource(file));
    }
}

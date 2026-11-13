package io.xunbian.backend;

import java.nio.file.*;
import java.util.*;
import org.junit.jupiter.api.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.http.*;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.util.LinkedMultiValueMap;
import static org.assertj.core.api.Assertions.*;

@SpringBootTest(webEnvironment=SpringBootTest.WebEnvironment.RANDOM_PORT,
 properties={"spring.datasource.url=${XB_TEST_DB_URL}","xunbian.execution-enabled=false","xunbian.question-queue-seconds=1"})
class ConsoleIntegrationTest {
    @Test void singleImageSegmentationUsesSharedBatchConsoleReviewAndReports() throws Exception {
        var owner=GroupTestClient.login(port,GroupTestClient.member(accounts));
        var form=new LinkedMultiValueMap<String,Object>();form.add("label","segmentation integration fixture");form.add("regionId","whu");form.add("taskType","building_segmentation");form.add("before",BackendIntegrationTest.image("building.png",32,32));
        var headers=new HttpHeaders();headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        var created=owner.postForEntity("/api/v1/batches",new HttpEntity<>(form,headers),Map.class);
        assertThat(created.getStatusCode()).isEqualTo(HttpStatus.CREATED);String id=created.getBody().get("batchId").toString();
        assertThat(owner.postForEntity("/api/v1/batches/"+id+"/start",Map.of("modelId","secor-whu"),Map.class).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(owner.postForEntity("/api/v1/batches/"+id+"/start",Map.of("modelId","sam-scfp-whu"),Map.class).getStatusCode()).isEqualTo(HttpStatus.OK);
        var attempt=tasks.claim();assertThat(attempt.afterAssetId()).isNull();
        String asset="pred-"+attempt.taskId()+"-"+attempt.attemptId()+".png";
        var image=new java.awt.image.BufferedImage(32,32,java.awt.image.BufferedImage.TYPE_BYTE_GRAY);image.getRaster().setSample(0,0,0,255);javax.imageio.ImageIO.write(image,"png",FakePython.ROOT.resolve(asset).toFile());
        var prediction=Map.<String,Object>of("taskId",attempt.taskId(),"attemptId",attempt.attemptId(),"modelId","sam-scfp-whu","modelVersion","a".repeat(64),"predMaskAssetId",asset,"width",32,"height",32,"foregroundPixels",1,"foregroundRatio",1.0/1024,"componentCount",1);
        assertThat(tasks.complete(attempt,prediction)).isTrue();analysis.publishCompleted();
        assertThat(db.queryForObject("SELECT COUNT(*) FROM prediction_results WHERE batch_id=?",Integer.class,id)).isZero();
        var console=owner.getForObject("/api/v1/batches/"+id+"/console",Map.class);assertThat(console).containsEntry("taskType","building_segmentation");
        var item=((List<Map>)((List<Map>)((Map)console.get("app")).get("regions")).getFirst().get("items")).getFirst();
        assertThat(item).containsKeys("foregroundPixels","foregroundRatio","componentCount").doesNotContainKey("changedPx");
        assertThat(((Number)item.get("estimatedBuildingCount")).intValue()).isZero();assertThat(((Number)item.get("filteredFragmentCount")).intValue()).isEqualTo(1);
        assertThat(db.queryForObject("SELECT result_json FROM segmentation_task_results WHERE task_id=?",String.class,attempt.taskId())).contains("estimatedBuildingCount");
        assertThat((Map)item.get("assets")).containsKeys("source","mask").doesNotContainKeys("pre","post");
        String review="/api/v1/batches/"+id+"/reviews/"+attempt.taskId();
        var geometry=Map.of("type","multi_polygon","polygons",List.of(List.of(List.of(2,2),List.of(6,2),List.of(6,6),List.of(2,6))));
        assertThat(owner.exchange(review,HttpMethod.PUT,new HttpEntity<>(Map.of("expectedVersion",0,"verdict","accepted_mask","note","分割修正","correction",Map.of("status","approved","geometry",geometry))),Map.class).getStatusCode()).isEqualTo(HttpStatus.OK);
        var effective=owner.postForObject("/api/v1/batches/"+id+"/report-summary",Map.of(),Map.class);assertThat(((Number)((Map)effective.get("report")).get("foregroundPixelsSum")).longValue()).isEqualTo(25);
        assertThat(((Number)((Map)effective.get("report")).get("estimatedBuildingCountSum")).longValue()).isEqualTo(1);
        var original=owner.postForObject("/api/v1/batches/"+id+"/report-summary",Map.of("resultBasis","original"),Map.class);assertThat(((Number)((Map)original.get("report")).get("foregroundPixelsSum")).longValue()).isEqualTo(1);
        assertThat(((Number)((Map)original.get("report")).get("estimatedBuildingCountSum")).longValue()).isZero();
        var other=GroupTestClient.login(port,GroupTestClient.member(accounts));assertThat(other.getForEntity("/api/v1/batches/"+id+"/console",Map.class).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
    }
    @Test void manualCorrectionDraftApprovalAndOriginalViewAreSeparate() throws Exception {
        var owner=GroupTestClient.login(port,GroupTestClient.member(accounts));String id=published(owner);
        String sample=db.queryForObject("SELECT id FROM sample_tasks WHERE batch_id=?",String.class,id);
        String route="/api/v1/batches/"+id+"/reviews/"+sample;
        var geometry=Map.of("type","multi_polygon","polygons",List.of(List.of(List.of(2,2),List.of(6,2),List.of(6,6),List.of(2,6))));
        var draft=owner.exchange(route,HttpMethod.PUT,new HttpEntity<>(Map.of("expectedVersion",0,"verdict","unreviewed","note","草稿","correction",Map.of("status","draft","geometry",geometry))),Map.class);
        assertThat(draft.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(draft.getBody()).containsKey("correction");
        var before=owner.postForObject("/api/v1/batches/"+id+"/report-summary",Map.of(),Map.class);
        assertThat(((Number)((Map)before.get("report")).get("changedPixelsSum")).intValue()).isEqualTo(4);
        var approved=owner.exchange(route,HttpMethod.PUT,new HttpEntity<>(Map.of("expectedVersion",1,"verdict","confirmed_change","note","审批","correction",Map.of("status","approved","geometry",geometry))),Map.class);
        assertThat(approved.getStatusCode()).isEqualTo(HttpStatus.OK);
        var listed=(List<Map>)owner.getForObject("/api/v1/batches/"+id+"/results",Map.class).get("rows");
        assertThat(listed.getFirst()).containsEntry("reviewVerdict","confirmed_change");
        var effective=owner.postForObject("/api/v1/batches/"+id+"/report-summary",Map.of("resultBasis","effective"),Map.class);
        assertThat(((Number)((Map)effective.get("report")).get("changedPixelsSum")).intValue()).isEqualTo(25);
        owner.postForObject("/api/v1/batches/"+id+"/questions",Map.of("question","变化像素合计","sessionId","corrected"),Map.class);
        var sqlItem=((List<Map>)((List<Map>)FakePython.dataset.get("regions")).getFirst().get("items")).getFirst();
        assertThat(((Number)sqlItem.get("changedPx")).intValue()).isEqualTo(25);
        var original=owner.postForObject("/api/v1/batches/"+id+"/report-summary",Map.of("resultBasis","original"),Map.class);
        assertThat(((Number)((Map)original.get("report")).get("changedPixelsSum")).intValue()).isEqualTo(4);
        assertThat(owner.exchange(route,HttpMethod.PUT,new HttpEntity<>(Map.of("expectedVersion",1,"verdict","confirmed_change","note","过期")),Map.class).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        var restore=new LinkedHashMap<String,Object>();restore.put("expectedVersion",2);restore.put("verdict","confirmed_change");restore.put("note","恢复");restore.put("correction",null);
        assertThat(owner.exchange(route,HttpMethod.PUT,new HttpEntity<>(restore),Map.class).getStatusCode()).isEqualTo(HttpStatus.OK);
        var reverted=owner.postForObject("/api/v1/batches/"+id+"/report-summary",Map.of(),Map.class);
        assertThat(((Number)((Map)reverted.get("report")).get("changedPixelsSum")).intValue()).isEqualTo(4);
    }
    @Test void historicalSamplesSupportSharedAnnotationsWithoutInventingPredictions() throws Exception {
        var administrator=admin();String id=db.queryForObject("SELECT batch_id FROM reference_sources ORDER BY batch_id LIMIT 1",String.class);
        var actor=GroupTestClient.member(accounts);var member=GroupTestClient.login(port,actor);
        String shared="/api/v1/batches/"+id+"/members/"+actor.actor().id();
        assertThat(administrator.postForEntity(shared,Map.of("permission","EDIT"),Map.class).getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(member.getForObject("/api/v1/batches",Map.class).toString()).contains(id);
        String route="/api/v1/batches/"+id+"/reviews/legacy-b";
        var old=member.getForObject(route,Map.class);
        var geometry=Map.of("type","multi_polygon","polygons",List.of(List.of(List.of(2,2),List.of(6,2),List.of(6,6),List.of(2,6))));
        var response=member.exchange(route,HttpMethod.PUT,new HttpEntity<>(Map.of("expectedVersion",old.get("version"),"verdict","confirmed_change","note","历史库修正","correction",Map.of("status","approved","geometry",geometry))),Map.class);
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(member.getForObject("/api/v1/batches/"+id+"/console",Map.class)).containsEntry("readOnly",false);
        assertThat(member.getForObject("/api/v1/batches/"+id+"/console?basis=original",Map.class)).containsEntry("readOnly",true);
        assertThat(db.queryForObject("SELECT COUNT(*) FROM prediction_results WHERE batch_id=?",Integer.class,id)).isZero();
        assertThat(db.queryForObject("SELECT COUNT(*) FROM sample_tasks WHERE batch_id=?",Integer.class,id)).isZero();
        administrator.postForEntity(shared,Map.of("permission","NONE"),Map.class);
        assertThat(member.getForEntity(route,Map.class).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
    }
    @LocalServerPort int port;
    @Autowired AccountService accounts;@Autowired JdbcTemplate db;@Autowired ExecutionRepository tasks;
    @Autowired AnalysisService analysis;@Autowired PythonClient python;
    static String archiveFile(){
        try{
            Path root=FakePython.ROOT.resolve("legacy-web");Files.createDirectories(root.resolve("assets/uav"));
            for(String name:List.of("pre.png","post.png","mask.png"))Files.write(root.resolve("assets/uav/"+name),BackendIntegrationTest.image(name,32,32).getByteArray());
            var sample=new LinkedHashMap<String,Object>();sample.putAll(Map.of("id","legacy-a","stem","legacy-a","changedPx",4,"changedRatio",4.0/1024,"objectCount",1,"sourceSize",List.of(32,32),"confidence",.8,"reviewLevel","high"));
            sample.put("assets",Map.of("pre","assets/uav/pre.png","post","assets/uav/post.png","mask","assets/uav/mask.png","gtMask","../../secret.png"));
            sample.put("privateMetrics",Map.of("maskF1",.99));
            var second=new LinkedHashMap<>(sample);second.put("id","legacy-b");second.put("stem","legacy-b");
            Files.writeString(root.resolve("app_data.json"),FakePython.JSON.writeValueAsString(Map.of("title","历史参考","regions",List.of(Map.of("id","uav","short","UAV","items",List.of(sample,second))))));
            return root.resolve("app_data.json").toString();
        }catch(Exception e){throw new RuntimeException(e);}
    }
    @DynamicPropertySource static void adapter(DynamicPropertyRegistry p){
        p.add("xunbian.storage-root",()->FakePython.ROOT.toString());p.add("xunbian.python-url",FakePython::url);
        p.add("xunbian.bootstrap-file",()->FakePython.ROOT.resolve("initial-test-admin.json").toString());p.add("xunbian.archive-file",ConsoleIntegrationTest::archiveFile);
    }
    @BeforeEach void clearQueue(){
        FakePython.MODE.set(0);db.update("UPDATE execution_slots SET task_id=NULL,attempt_id=NULL,lease_until=NULL,available_after_ms=0 WHERE slot_id=1");
        db.update("UPDATE sample_tasks SET status='CANCELLED' WHERE status IN ('PENDING','RUNNING')");db.update("UPDATE batches SET status='CANCELLED' WHERE status='RUNNING'");
    }
    TestRestTemplate admin(){
        GroupTestClient.as(accounts.loadUserByUsername("admin"));
        try{var user=accounts.create("admin_"+UUID.randomUUID().toString().substring(0,12),"console test admin",GroupTestClient.PASSWORD,"ADMIN");return GroupTestClient.login(port,accounts.loadUserByUsername(user.get("username").toString()));}
        finally{org.springframework.security.core.context.SecurityContextHolder.clearContext();}
    }
    String published(TestRestTemplate client) throws Exception {
        var form=new LinkedMultiValueMap<String,Object>();form.add("label","console actual fixture");form.add("regionId","whu");form.add("before",BackendIntegrationTest.image("one.png",32,32));form.add("after",BackendIntegrationTest.image("one.png",32,32));
        var headers=new HttpHeaders();headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        var response=client.postForEntity("/api/v1/batches",new HttpEntity<>(form,headers),Map.class);assertThat(response.getStatusCode()).isEqualTo(HttpStatus.CREATED);
        String id=response.getBody().get("batchId").toString();client.postForEntity("/api/v1/batches/"+id+"/start",Map.of("modelId","secor-whu"),Map.class);
        var a=tasks.claim();assertThat(a.batchId()).isEqualTo(id);assertThat(tasks.complete(a,python.predict(a))).isTrue();analysis.publishCompleted();return id;
    }
    @Test void consoleUsesOnlyAuthorizedBatchWithActualDimensionsAndUnknownSemanticFacts() throws Exception {
        var owner=GroupTestClient.login(port,GroupTestClient.member(accounts));String id=published(owner);
        var response=owner.getForEntity("/api/v1/batches/"+id+"/console",Map.class);assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        var data=response.getBody();assertThat(data).containsEntry("sourceType","INFERENCE").containsEntry("readOnly",false);
        var regions=(List<Map>)((Map)data.get("app")).get("regions");assertThat(regions).hasSize(1);
        var item=((List<Map>)regions.getFirst().get("items")).getFirst();assertThat(item.get("sourceSize")).isEqualTo(List.of(32,32));assertThat(item).doesNotContainKeys("confidence","primaryEventType");assertThat(item.get("reviewLevel")).isEqualTo("unknown");
        var other=GroupTestClient.login(port,GroupTestClient.member(accounts));assertThat(other.getForEntity("/api/v1/batches/"+id+"/console",Map.class).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(GroupTestClient.anonymous(port).getForEntity("/api/v1/batches/"+id+"/console",Map.class).getStatusCode()).isEqualTo(HttpStatus.UNAUTHORIZED);
    }
    @Test void historicalReferenceIsReadOnlyAdminOnlyAndDoesNotForgeInferenceRows(){
        var administrator=admin();var list=(List<Map>)administrator.getForObject("/api/v1/batches",Map.class).get("batches");
        var archives=list.stream().filter(b->"ARCHIVE".equals(b.get("sourceType"))).toList();assertThat(archives).isNotEmpty();String id=archives.getFirst().get("batchId").toString();
        var dto=administrator.getForObject("/api/v1/batches/"+id+"/console",Map.class);assertThat(dto).containsEntry("sourceType","ARCHIVE").containsEntry("readOnly",false);
        String json=FakePython.JSON.valueToTree(dto).toString();assertThat(json).doesNotContain("gtMask","maskF1","secret.png");
        assertThat(db.queryForObject("SELECT COUNT(*) FROM sample_tasks WHERE batch_id=?",Integer.class,id)).isZero();assertThat(db.queryForObject("SELECT COUNT(*) FROM prediction_results WHERE batch_id=?",Integer.class,id)).isZero();
        var member=GroupTestClient.login(port,GroupTestClient.member(accounts));assertThat(member.getForEntity("/api/v1/batches/"+id+"/console",Map.class).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(administrator.postForEntity("/api/v1/batches/"+id+"/start",Map.of("modelId","secor-whu"),Map.class).getStatusCode().is4xxClientError()).isTrue();
        var regions=(List<Map>)((Map)dto.get("app")).get("regions");var sample=((List<Map>)regions.getFirst().get("items")).getFirst();String asset=((Map)sample.get("assets")).get("pre").toString();
        assertThat(administrator.getForEntity(asset,byte[].class).getStatusCode()).isEqualTo(HttpStatus.OK);assertThat(member.getForEntity(asset,byte[].class).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
    }
    @Test void reportRecomputesFactsAndRejectsStaleVersionOrForeignSamples() throws Exception {
        var owner=GroupTestClient.login(port,GroupTestClient.member(accounts));String id=published(owner);String route="/api/v1/batches/"+id+"/report-summary";
        var result=owner.postForEntity(route,Map.of("regionId","whu","reviewVersion",0,"totalCount",999),Map.class);assertThat(result.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat((Map)result.getBody().get("report")).containsEntry("totalCount",1);assertThat(result.getBody().get("text").toString()).doesNotContain("999");
        var riskBuckets=(List<Map>)((Map)result.getBody().get("report")).get("riskBuckets");
        assertThat(((Number)riskBuckets.stream().filter(b->"unknown".equals(b.get("key"))).findFirst().orElseThrow().get("ratio")).doubleValue()).isEqualTo(1.0);
        assertThat(owner.postForEntity(route,Map.of("reviewVersion",1),Map.class).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        assertThat(owner.postForEntity(route,Map.of("sampleIds",List.of("foreign-sample"),"reviewVersion",0),Map.class).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        var empty=owner.postForObject(route,Map.of("sampleIds",List.of(),"reviewVersion",0),Map.class);assertThat((Map)empty.get("report")).containsEntry("totalCount",0);
    }
    @Test void chatHistoryDoesNotRequireQuestionOrGpuAdmission() throws Exception {
        var owner=GroupTestClient.login(port,GroupTestClient.member(accounts));String id=published(owner);
        analysis.admission.gpu.acquire();
        try{var response=owner.postForEntity("/api/v1/batches/"+id+"/questions",Map.of("action","history","sessionId","same"),Map.class);assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);assertThat(response.getBody()).containsEntry("status","history");}
        finally{analysis.admission.gpu.release();}
    }
    @Test void visibleRegionIsValidatedInsideAuthorizedBatchAndForwardedSeparately() throws Exception {
        var owner=GroupTestClient.login(port,GroupTestClient.member(accounts));String id=published(owner),route="/api/v1/batches/"+id+"/questions";
        var response=owner.postForEntity(route,Map.of("question","这个区域有几个图片","sessionId","visible","viewRegion","whu"),Map.class);
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);assertThat(FakePython.questionRequest).containsEntry("viewRegion","whu").doesNotContainKey("scopeRegion");
        assertThat(owner.postForEntity(route,Map.of("question","这个区域有几个图片","sessionId","visible2","viewRegion","levir"),Map.class).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(owner.postForEntity(route,Map.of("question","这个区域有几个图片","sessionId","visible3","viewRegion",List.of("whu")),Map.class).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
    }
}

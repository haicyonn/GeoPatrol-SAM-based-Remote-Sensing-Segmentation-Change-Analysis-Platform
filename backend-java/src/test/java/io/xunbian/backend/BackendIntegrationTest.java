package io.xunbian.backend;

import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.util.Map;
import java.util.List;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.*;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.test.annotation.DirtiesContext;
import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest(webEnvironment=SpringBootTest.WebEnvironment.RANDOM_PORT,
    properties={"spring.datasource.url=${XB_TEST_DB_URL}"})
@DirtiesContext(classMode=DirtiesContext.ClassMode.AFTER_CLASS)
class BackendIntegrationTest {
    TestRestTemplate http;
    @Autowired AccountService accounts;
    @LocalServerPort int port;
    @BeforeEach void login(){http=GroupTestClient.login(port,GroupTestClient.member(accounts));}
    @Autowired JdbcTemplate db;
    @DynamicPropertySource static void adapter(DynamicPropertyRegistry properties){
        properties.add("xunbian.storage-root",()->FakePython.ROOT.toString());
        properties.add("xunbian.python-url",FakePython::url);
        properties.add("xunbian.bootstrap-file",()->FakePython.ROOT.resolve("initial-test-admin.json").toString());
    }
    @AfterEach void reset(){FakePython.MODE.set(0);}

    static ByteArrayResource image(String name, int width, int height) throws Exception {
        var out=new ByteArrayOutputStream();
        ImageIO.write(new BufferedImage(width,height,BufferedImage.TYPE_INT_RGB),"png",out);
        return new ByteArrayResource(out.toByteArray()) { @Override public String getFilename(){return name;} };
    }
    ResponseEntity<Map> upload(String before, String after, int afterWidth) throws Exception {
        return upload(before,after,afterWidth,null);
    }
    ResponseEntity<Map> upload(String before, String after, int afterWidth,String origin) throws Exception {
        var form=new LinkedMultiValueMap<String,Object>();
        form.add("label","integration fixture"); form.add("regionId","whu");
        form.add("before",image(before,32,32)); form.add("after",image(after,afterWidth,32));
        var headers=new HttpHeaders(); headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        if(origin!=null)headers.setOrigin(origin);
        return http.postForEntity("/api/v1/batches",new HttpEntity<>(form,headers),Map.class);
    }
    @Test void matchingImagePairCreatesDurableDraft() throws Exception {
        var response=upload("same.png","same.png",32);
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.CREATED);
        var id=response.getBody().get("batchId");
        var batch=http.getForEntity("/api/v1/batches/"+id,Map.class);
        assertThat(batch.getBody()).containsEntry("status","DRAFT").containsEntry("total",1);
    }
    @Test void mismatchedNamesAreRejectedBeforePrediction() throws Exception {
        assertThat(upload("a.png","b.png",32).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
    }
    @Test void mismatchedImageDimensionsAreRejected() throws Exception {
        assertThat(upload("same.png","same.png",33).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
    }
    @Test void unpublishedBatchCannotBeQueried() throws Exception {
        var id=upload("same.png","same.png",32).getBody().get("batchId");
        var response=http.postForEntity("/api/v1/batches/"+id+"/questions",Map.of("sessionId","one","question","共有多少样本"),Map.class);
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
    }
    Map waitFor(String id,String status) throws Exception {
        long deadline=System.nanoTime()+10_000_000_000L;Map value=null;
        while(System.nanoTime()<deadline){
            value=http.getForObject("/api/v1/batches/"+id,Map.class);
            if(status.equals(value.get("status")))return value;
            Thread.sleep(80);
        }
        throw new AssertionError("batch did not reach "+status+": "+value);
    }
    String start() throws Exception {
        String id=upload("same.png","same.png",32).getBody().get("batchId").toString();
        assertThat(http.postForEntity("/api/v1/batches/"+id+"/start",Map.of("modelId","secor-whu"),Map.class).getStatusCode()).isEqualTo(HttpStatus.OK);
        return id;
    }
    @Test void completedResultIsSavedAndFeedsPublishedDataset() throws Exception {
        String id=start();assertThat(waitFor(id,"PUBLISHED")).containsEntry("completed",1);
        var response=http.postForEntity("/api/v1/batches/"+id+"/questions",Map.of("sessionId","same-session","question","共有多少样本"),Map.class);
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(response.getBody()).containsEntry("batchId",id);
        var regions=(java.util.List<Map>)FakePython.dataset.get("regions");
        var samples=(java.util.List<Map>)regions.getFirst().get("items");
        assertThat(samples).hasSize(1);
        assertThat(((Number)samples.getFirst().get("changedPx")).intValue()).isEqualTo(4);
        assertThat(db.queryForObject("SELECT COUNT(*) FROM prediction_results WHERE batch_id=?",Integer.class,id)).isEqualTo(1);
    }
    @Test void modelFailureDoesNotPublishAndOnlyFailedTaskIsRetried() throws Exception {
        FakePython.MODE.set(1);String id=start();waitFor(id,"FAILED");
        assertThat(db.queryForObject("SELECT COUNT(*) FROM prediction_results WHERE batch_id=?",Integer.class,id)).isZero();
        FakePython.MODE.set(0);
        assertThat(http.postForEntity("/api/v1/batches/"+id+"/retry-failed",null,Map.class).getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(waitFor(id,"PUBLISHED")).containsEntry("completed",1);
        assertThat(db.queryForObject("SELECT MAX(attempt_no) FROM sample_tasks WHERE batch_id=?",Integer.class,id)).isEqualTo(2);
    }
    @Test void inconsistentMaskStatisticsCannotEnterDatabase() throws Exception {
        FakePython.MODE.set(2);String id=start();waitFor(id,"FAILED");
        assertThat(db.queryForObject("SELECT COUNT(*) FROM prediction_results WHERE batch_id=?",Integer.class,id)).isZero();
        assertThat(db.queryForObject("SELECT COUNT(*) FROM analysis_datasets WHERE batch_id=?",Integer.class,id)).isZero();
    }
    @Test void unknownAssetDoesNotExposeFilesystem() {
        assertThat(http.getForEntity("/api/v1/assets/not-registered.png",byte[].class).getStatusCode()).isEqualTo(HttpStatus.NOT_FOUND);
    }
    @Test void cancelledDraftCannotBeExecuted() throws Exception {
        String id=upload("same.png","same.png",32).getBody().get("batchId").toString();
        assertThat(http.postForEntity("/api/v1/batches/"+id+"/cancel",null,Map.class).getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(http.getForObject("/api/v1/batches/"+id,Map.class)).containsEntry("status","CANCELLED");
        assertThat(http.postForEntity("/api/v1/batches/"+id+"/start",Map.of("modelId","secor-whu"),Map.class).getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
    }
    @Test void crossOriginPageCannotCreateLocalPredictionBatch() throws Exception {
        assertThat(upload("same.png","same.png",32,"https://example.invalid").getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
    }
    @Test void advertisedTwoHundredPairsFitMultipartParserLimit() throws Exception {
        var form=new LinkedMultiValueMap<String,Object>();form.add("label","200-pair limit check");form.add("regionId","whu");form.add("assignments","{}");
        for(int i=0;i<200;i++){String name="pair-"+i+".png";form.add("before",image(name,32,32));form.add("after",image(name,32,32));}
        var headers=new HttpHeaders();headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        var response=http.postForEntity("/api/v1/batches",new HttpEntity<>(form,headers),Map.class);
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.CREATED);
        assertThat(response.getBody()).containsEntry("total",200);
    }
    @Test void duplicateStemsAndCorruptFilesCannotCreateBatch() throws Exception {
        for(boolean duplicate:List.of(true,false)){
            var form=new LinkedMultiValueMap<String,Object>();form.add("label","invalid input");form.add("regionId","whu");
            if(duplicate){form.add("before",image("same.png",32,32));form.add("before",image("same.png",32,32));form.add("after",image("same.png",32,32));form.add("after",image("other.png",32,32));}
            else {form.add("before",new ByteArrayResource(new byte[]{1,2,3}){@Override public String getFilename(){return "same.png";}});form.add("after",image("same.png",32,32));}
            var headers=new HttpHeaders();headers.setContentType(MediaType.MULTIPART_FORM_DATA);
            assertThat(http.postForEntity("/api/v1/batches",new HttpEntity<>(form,headers),Map.class).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        }
    }
    @Test void validButOversizedPixelDimensionsAreRejectedBeforePersistence() throws Exception {
        int before=db.queryForObject("SELECT COUNT(*) FROM batches",Integer.class);
        var resource=image("big.png",4097,4096);
        var form=new LinkedMultiValueMap<String,Object>();form.add("label","pixel limit");form.add("regionId","whu");form.add("before",resource);form.add("after",resource);
        var headers=new HttpHeaders();headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        assertThat(http.postForEntity("/api/v1/batches",new HttpEntity<>(form,headers),Map.class).getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(db.queryForObject("SELECT COUNT(*) FROM batches",Integer.class)).isEqualTo(before);
    }
}

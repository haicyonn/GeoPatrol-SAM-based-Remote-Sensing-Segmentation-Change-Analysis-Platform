package io.xunbian.backend;

import org.junit.jupiter.api.Test;
import java.util.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.http.HttpStatus;
import org.springframework.http.*;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest(webEnvironment=SpringBootTest.WebEnvironment.RANDOM_PORT,
    properties={"spring.datasource.url=${XB_TEST_DB_URL}","xunbian.execution-enabled=false"})
class GroupAccessIntegrationTest {
    @Autowired TestRestTemplate http;
    @Autowired AccountService accounts;
    @Autowired JdbcTemplate db;
    @LocalServerPort int port;
    @DynamicPropertySource static void adapter(DynamicPropertyRegistry p){
        p.add("xunbian.storage-root",()->FakePython.ROOT.toString());
        p.add("xunbian.python-url",FakePython::url);
        p.add("xunbian.bootstrap-file",()->FakePython.ROOT.resolve("initial-test-admin.json").toString());
    }
    @Test void anonymousCannotReadResearchBatchesOrImageAssets(){
        assertThat(http.getForEntity("/api/v1/batches",String.class).getStatusCode()).isEqualTo(HttpStatus.UNAUTHORIZED);
        assertThat(http.getForEntity("/api/v1/assets/unknown.png",String.class).getStatusCode()).isEqualTo(HttpStatus.UNAUTHORIZED);
    }
    String upload(TestRestTemplate client) throws Exception {
        var form=new LinkedMultiValueMap<String,Object>();form.add("label","private group fixture");form.add("regionId","whu");
        form.add("before",BackendIntegrationTest.image("one.png",32,32));form.add("after",BackendIntegrationTest.image("one.png",32,32));
        var headers=new HttpHeaders();headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        var response=client.postForEntity("/api/v1/batches",new HttpEntity<>(form,headers),Map.class);
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.CREATED);return response.getBody().get("batchId").toString();
    }
    @Test void ownerViewerEditorAndRevokedMemberAreEnforcedOnRealHttpAndAssets() throws Exception {
        var a=GroupTestClient.member(accounts);var b=GroupTestClient.member(accounts);
        var owner=GroupTestClient.login(port,a);var other=GroupTestClient.login(port,b);
        String id=upload(owner);String asset=db.queryForObject("SELECT before_asset FROM sample_tasks WHERE batch_id=?",String.class,id);
        assertThat(other.getForEntity("/api/v1/batches/"+id,Map.class).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(other.getForEntity("/api/v1/assets/"+asset,byte[].class).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(other.getForObject("/api/v1/batches",Map.class).get("batches")).isEqualTo(List.of());
        String share="/api/v1/batches/"+id+"/members/"+b.actor().id();
        assertThat(owner.postForEntity(share,Map.of("permission","VIEW"),Map.class).getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(other.getForEntity("/api/v1/assets/"+asset,byte[].class).getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(other.postForEntity("/api/v1/batches/"+id+"/cancel",null,Map.class).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(other.postForEntity(share,Map.of("permission","EDIT"),Map.class).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        owner.postForEntity(share,Map.of("permission","EDIT"),Map.class);
        assertThat(other.postForEntity("/api/v1/batches/"+id+"/cancel",null,Map.class).getStatusCode()).isEqualTo(HttpStatus.OK);
        owner.postForEntity(share,Map.of("permission","NONE"),Map.class);
        assertThat(other.getForEntity("/api/v1/batches/"+id,Map.class).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
    }
    @Test void authenticatedMutationsRequireCsrfAndDisabledAccountLosesExistingCookie(){
        var member=GroupTestClient.member(accounts);var client=GroupTestClient.login(port,member);
        client.getRestTemplate().getInterceptors().clear();
        assertThat(client.postForEntity("/api/v1/auth/password",Map.of("password","new-valid-password","currentPassword",GroupTestClient.PASSWORD),Map.class).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        GroupTestClient.as(accounts.loadUserByUsername("admin"));
        try{accounts.enabled(member.actor().id(),false);}finally{org.springframework.security.core.context.SecurityContextHolder.clearContext();}
        assertThat(client.getForEntity("/api/v1/auth/me",Map.class).getStatusCode()).isEqualTo(HttpStatus.UNAUTHORIZED);
    }
    String published(TestRestTemplate owner) throws Exception {
        String id=upload(owner);
        db.update("UPDATE batches SET status='PUBLISHED' WHERE id=?",id);
        String task=db.queryForObject("SELECT id FROM sample_tasks WHERE batch_id=?",String.class,id);
        String asset=db.queryForObject("SELECT before_asset FROM sample_tasks WHERE batch_id=?",String.class,id);
        db.update("UPDATE sample_tasks SET status='SUCCEEDED' WHERE batch_id=?",id);
        db.update("INSERT INTO prediction_results VALUES(?,?,?,?,?,?,?,?,?,?,?)",task,id,UUID.randomUUID().toString(),"secor-whu","0".repeat(64),4,4.0/1024,1,asset,"{}",System.currentTimeMillis());
        db.update("INSERT INTO analysis_datasets VALUES(?,?,?,?)",id,"{\"regions\":[{\"id\":\"whu\",\"items\":[{\"id\":\""+task+"\",\"changedPx\":4,\"changedRatio\":0.00390625,\"objectCount\":1}]}]}","a".repeat(64),System.currentTimeMillis());
        return id;
    }
    @Test void callerCannotForgeUserIdentityAndEqualClientSessionsRemainSeparate() throws Exception {
        var a=GroupTestClient.member(accounts);var b=GroupTestClient.member(accounts);var owner=GroupTestClient.login(port,a);var other=GroupTestClient.login(port,b);
        String id=published(owner);owner.postForEntity("/api/v1/batches/"+id+"/members/"+b.actor().id(),Map.of("permission","VIEW"),Map.class);
        var body=Map.of("question","共有多少样本","sessionId","same-browser-id","userId","forged-user");
        var first=owner.postForObject("/api/v1/batches/"+id+"/questions",body,Map.class);
        var second=other.postForObject("/api/v1/batches/"+id+"/questions",body,Map.class);
        assertThat(first.get("trustedUser")).isEqualTo(a.actor().id());
        assertThat(second.get("trustedUser")).isEqualTo(b.actor().id());
        assertThat(first.get("trustedSession")).isNotEqualTo(second.get("trustedSession"));
    }
    @Test void overlappingTurnsInSameConversationAreRejectedWithoutAdvancingAgain() throws Exception {
        var p=GroupTestClient.member(accounts);var client=GroupTestClient.login(port,p);String id=published(client);
        FakePython.MODE.set(4);
        try(var pool=java.util.concurrent.Executors.newVirtualThreadPerTaskExecutor()){
            var body=Map.of("question","共有多少样本","sessionId","same-turn");
            var first=pool.submit(()->client.postForEntity("/api/v1/batches/"+id+"/questions",body,Map.class));
            Thread.sleep(300);
            var second=client.postForEntity("/api/v1/batches/"+id+"/questions",body,Map.class);
            assertThat(second.getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
            assertThat(first.get().getStatusCode()).isEqualTo(HttpStatus.OK);
        }finally{FakePython.MODE.set(0);}
    }
    @Test void simultaneousReviewsKeepOneVersionAndReportConflictingCurrentValue() throws Exception {
        var a=GroupTestClient.member(accounts);var b=GroupTestClient.member(accounts);var owner=GroupTestClient.login(port,a);var editor=GroupTestClient.login(port,b);
        String id=published(owner),sample=db.queryForObject("SELECT id FROM sample_tasks WHERE batch_id=?",String.class,id);
        owner.postForEntity("/api/v1/batches/"+id+"/members/"+b.actor().id(),Map.of("permission","EDIT"),Map.class);
        String route="/api/v1/batches/"+id+"/reviews/"+sample;
        try(var pool=java.util.concurrent.Executors.newVirtualThreadPerTaskExecutor()){
            var one=pool.submit(()->owner.exchange(route,HttpMethod.PUT,new HttpEntity<>(Map.of("expectedVersion",0,"verdict","confirmed_change","note","first assessment")),Map.class));
            var two=pool.submit(()->editor.exchange(route,HttpMethod.PUT,new HttpEntity<>(Map.of("expectedVersion",0,"verdict","false_change","note","second assessment")),Map.class));
            var x=one.get();var y=two.get();assertThat(List.of(x.getStatusCode().value(),y.getStatusCode().value())).containsExactlyInAnyOrder(200,409);
            var conflict=x.getStatusCode().value()==409?x:y;assertThat((Map)conflict.getBody().get("current")).containsEntry("version",1);
        }
        assertThat(db.queryForObject("SELECT COUNT(*) FROM sample_annotation_history WHERE sample_id=?",Integer.class,sample)).isEqualTo(1);
        var report=owner.getForObject("/api/v1/batches/"+id+"/report",Map.class);
        assertThat(report).containsEntry("reviewVersion",1).containsEntry("reviewedSamples",1);
    }
    @Test void reviewChangeStartsFreshModelDatasetAndReadOnlyMemberCannotReview() throws Exception {
        var a=GroupTestClient.member(accounts);var b=GroupTestClient.member(accounts);var owner=GroupTestClient.login(port,a);var viewer=GroupTestClient.login(port,b);
        String id=published(owner),sample=db.queryForObject("SELECT id FROM sample_tasks WHERE batch_id=?",String.class,id);
        owner.postForEntity("/api/v1/batches/"+id+"/members/"+b.actor().id(),Map.of("permission","VIEW"),Map.class);
        var body=Map.of("sessionId","same-after-review","question","共有多少样本");String questions="/api/v1/batches/"+id+"/questions";
        var before=owner.postForObject(questions,body,Map.class);
        String route="/api/v1/batches/"+id+"/reviews/"+sample;
        var change=new HttpEntity<>(Map.of("expectedVersion",0,"verdict","confirmed_change","note","checked"));
        assertThat(viewer.exchange(route,HttpMethod.PUT,change,Map.class).getStatusCode()).isEqualTo(HttpStatus.FORBIDDEN);
        assertThat(owner.exchange(route,HttpMethod.PUT,change,Map.class).getStatusCode()).isEqualTo(HttpStatus.OK);
        var after=owner.postForObject(questions,body,Map.class);
        assertThat(after.get("datasetId")).isNotEqualTo(before.get("datasetId"));
        assertThat(after).containsEntry("reviewVersion",1).containsEntry("contextReset",true);
        var items=(List<Map>)((List<Map>)FakePython.dataset.get("regions")).getFirst().get("items");
        assertThat(items.getFirst()).containsEntry("manualReviewComplete",true);
        assertThat(viewer.getForEntity("/api/v1/batches/"+id+"/report",Map.class).getStatusCode()).isEqualTo(HttpStatus.OK);
    }
    @Test void administratorPasswordResetInvalidatesMemberSession() {
        var member=GroupTestClient.member(accounts);var client=GroupTestClient.login(port,member);
        GroupTestClient.as(accounts.loadUserByUsername("admin"));
        AccountService.Principal operator;
        try{var created=accounts.create("a"+UUID.randomUUID().toString().substring(0,15),"test admin",GroupTestClient.PASSWORD,"ADMIN");operator=accounts.loadUserByUsername(created.get("username").toString());}
        finally{org.springframework.security.core.context.SecurityContextHolder.clearContext();}
        var admin=GroupTestClient.login(port,operator);
        assertThat(admin.postForEntity("/api/v1/admin/users/"+member.actor().id()+"/password",Map.of("password","replacement-passphrase-123"),Map.class).getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(client.getForEntity("/api/v1/auth/me",Map.class).getStatusCode()).isEqualTo(HttpStatus.UNAUTHORIZED);
        assertThat(GroupTestClient.login(port,member.getUsername(),"replacement-passphrase-123").getForEntity("/api/v1/auth/me",Map.class).getStatusCode()).isEqualTo(HttpStatus.OK);
    }
    @Test void repeatedIncorrectLoginIsThrottledBeforeFurtherPasswordChecks(){
        var anonymous=GroupTestClient.anonymous(port);var csrf=anonymous.getForObject("/api/v1/auth/csrf",Map.class);
        var headers=new HttpHeaders();headers.setContentType(MediaType.APPLICATION_FORM_URLENCODED);headers.set(csrf.get("headerName").toString(),csrf.get("token").toString());
        var form=new LinkedMultiValueMap<String,String>();form.add("username","absent-"+UUID.randomUUID().toString().substring(0,10));form.add("password","invalid-password");
        for(int i=0;i<8;i++)assertThat(anonymous.postForEntity("/api/v1/auth/login",new HttpEntity<>(form,headers),Map.class).getStatusCode()).isEqualTo(HttpStatus.UNAUTHORIZED);
        assertThat(anonymous.postForEntity("/api/v1/auth/login",new HttpEntity<>(form,headers),Map.class).getStatusCode()).isEqualTo(HttpStatus.TOO_MANY_REQUESTS);
    }
}

package io.xunbian.backend;

import java.awt.image.BufferedImage;
import java.util.*;
import java.util.concurrent.*;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import static org.assertj.core.api.Assertions.*;

@SpringBootTest(properties={"spring.datasource.url=${XB_TEST_DB_URL}","xunbian.execution-enabled=false"})
class ExecutionRepositoryTest {
    @Autowired BatchService batches;
    @Autowired ExecutionRepository tasks;
    @Autowired JdbcTemplate db;
    @Autowired Settings settings;
    @Autowired AnalysisService analysis;
    @Autowired PythonClient python;
    @Autowired AccountService accounts;
    @DynamicPropertySource static void adapter(DynamicPropertyRegistry p){p.add("xunbian.storage-root",()->FakePython.ROOT.toString());p.add("xunbian.python-url",FakePython::url);p.add("xunbian.bootstrap-file",()->FakePython.ROOT.resolve("initial-test-admin.json").toString());}
    @BeforeEach void isolateFixtureQueue(){
        GroupTestClient.as(GroupTestClient.member(accounts));
        db.update("UPDATE execution_slots SET task_id=NULL,attempt_id=NULL,lease_until=NULL,available_after_ms=0 WHERE slot_id=1");
        db.update("UPDATE sample_tasks SET status='CANCELLED' WHERE status IN ('PENDING','RUNNING')");
        db.update("UPDATE batches SET status='CANCELLED' WHERE status='RUNNING'");
    }
    @AfterEach void clearIdentity(){org.springframework.security.core.context.SecurityContextHolder.clearContext();}
    String batch() throws Exception {
        var a=BackendIntegrationTest.image("same.png",32,32);var b=BackendIntegrationTest.image("same.png",32,32);
        var first=new org.springframework.mock.web.MockMultipartFile("before",a.getFilename(),"image/png",a.getByteArray());
        var second=new org.springframework.mock.web.MockMultipartFile("after",b.getFilename(),"image/png",b.getByteArray());
        String id=batches.create("lease test","whu",null,List.of(first),List.of(second)).get("batchId").toString();
        batches.start(id,"secor-whu");return id;
    }
    Map<String,Object> result(ExecutionRepository.Attempt a) throws Exception {
        String id="pred-"+a.taskId()+"-"+a.attemptId()+".png";
        var image=new BufferedImage(32,32,BufferedImage.TYPE_BYTE_GRAY);image.getRaster().setSample(0,0,0,255);
        ImageIO.write(image,"png",FakePython.ROOT.resolve(id).toFile());
        var map=new LinkedHashMap<>(a.payload());map.putAll(Map.of("modelVersion","0".repeat(64),"predMaskAssetId",id,"width",32,"height",32,"changedPx",1,"changedRatio",1.0/1024,"objectCount",1));return map;
    }
    @Test void expiredAttemptCannotOverwriteReclaimedExecution() throws Exception {
        String batch=batch();var old=tasks.claim();
        db.update("UPDATE execution_slots SET lease_until=1 WHERE slot_id=1");
        db.update("UPDATE sample_tasks SET lease_until=1 WHERE id=?",old.taskId());
        tasks.recoverExpired();var current=tasks.claim();
        assertThat(current.attemptId()).isNotEqualTo(old.attemptId());
        assertThat(tasks.complete(old,result(old))).isFalse();
        assertThat(tasks.complete(current,result(current))).isTrue();
        assertThat(db.queryForObject("SELECT attempt_id FROM prediction_results WHERE batch_id=?",String.class,batch)).isEqualTo(current.attemptId());
    }
    @Test void concurrentClaimersShareOneDatabaseExecutionSlot() throws Exception {
        batch();batch();
        try(var pool=Executors.newVirtualThreadPerTaskExecutor()){
            var calls=List.<Callable<ExecutionRepository.Attempt>>of(tasks::claim,tasks::claim,tasks::claim);
            int claimed=0;for(var future:pool.invokeAll(calls))if(future.get()!=null)claimed++;
            assertThat(claimed).isEqualTo(1);
        }
    }
    @Test void transactionFailureDoesNotCommitPredictionOrSuccessState() throws Exception {
        String id=batch();var a=tasks.claim();var result=result(a);
        db.update("INSERT INTO assets VALUES(?,?,?,?,?,?)",result.get("predMaskAssetId"),id,a.taskId(),"fixture","image/png",1);
        assertThatThrownBy(()->tasks.complete(a,result)).isInstanceOf(org.springframework.dao.DataIntegrityViolationException.class);
        assertThat(db.queryForObject("SELECT COUNT(*) FROM prediction_results WHERE batch_id=?",Integer.class,id)).isZero();
        assertThat(db.queryForObject("SELECT status FROM sample_tasks WHERE id=?",String.class,a.taskId())).isEqualTo("RUNNING");
    }
    @Test void gracefulShutdownLeavesInterruptedTaskRecoverable() throws Exception {
        String id=batch();int before=FakePython.PREDICTIONS.get();FakePython.MODE.set(3);
        var coordinator=new TaskCoordinator(tasks,analysis,python,settings);
        try{
            settings.executionEnabled=true;coordinator.dispatch();settings.executionEnabled=false;
            long until=System.nanoTime()+3_000_000_000L;
            while(FakePython.PREDICTIONS.get()==before&&System.nanoTime()<until)Thread.sleep(20);
            assertThat(FakePython.PREDICTIONS.get()).isGreaterThan(before);
            coordinator.stop();assertThat(coordinator.executor.awaitTermination(3,TimeUnit.SECONDS)).isTrue();
            assertThat(db.queryForObject("SELECT status FROM sample_tasks WHERE batch_id=?",String.class,id)).isEqualTo("RUNNING");
            db.update("UPDATE execution_slots SET lease_until=1 WHERE slot_id=1");tasks.recoverExpired();
            assertThat(db.queryForObject("SELECT status FROM sample_tasks WHERE batch_id=?",String.class,id)).isEqualTo("PENDING");
        }finally{settings.executionEnabled=false;FakePython.MODE.set(0);coordinator.stop();}
    }
    @Test void pendingBatchesTakeTurnsInsteadOfDrainingOldestBatch() throws Exception {
        String first=batch();
        String task=db.queryForObject("SELECT id FROM sample_tasks WHERE batch_id=?",String.class,first);
        db.update("INSERT INTO sample_tasks(id,batch_id,sample_name,region_id,before_asset,after_asset,width,height,status,created_ms,updated_ms) SELECT ?,batch_id,'second','whu',before_asset,after_asset,width,height,'PENDING',created_ms+1,updated_ms FROM sample_tasks WHERE id=?",UUID.randomUUID().toString(),task);
        db.update("UPDATE batches SET total=2 WHERE id=?",first);
        var one=tasks.claim();assertThat(one.batchId()).isEqualTo(first);
        String second=batch();assertThat(tasks.complete(one,result(one))).isTrue();
        var next=tasks.claim();assertThat(next.batchId()).isEqualTo(second);
    }
    @Test void temporaryWorkerOutageRetainsPendingWorkAndRecordsRetryAttempt() throws Exception {
        String id=batch();FakePython.MODE.set(5);settings.executionEnabled=true;
        var coordinator=new TaskCoordinator(tasks,analysis,python,settings);
        try{
            coordinator.dispatch();long until=System.nanoTime()+3_000_000_000L;
            while(db.queryForObject("SELECT COUNT(*) FROM task_attempts a JOIN sample_tasks t ON t.id=a.task_id WHERE t.batch_id=? AND a.status='RETRY_WAIT'",Integer.class,id)==0&&System.nanoTime()<until)Thread.sleep(25);
            assertThat(db.queryForObject("SELECT status FROM sample_tasks WHERE batch_id=?",String.class,id)).isEqualTo("PENDING");
            assertThat(db.queryForObject("SELECT status FROM batches WHERE id=?",String.class,id)).isEqualTo("RUNNING");
            assertThat(db.queryForObject("SELECT COUNT(*) FROM task_attempts a JOIN sample_tasks t ON t.id=a.task_id WHERE t.batch_id=? AND a.status='RETRY_WAIT'",Integer.class,id)).isEqualTo(1);
        }finally{settings.executionEnabled=false;FakePython.MODE.set(0);coordinator.stop();}
    }
    @Test void perMemberPendingLimitRejectsExcessUploadAndLeavesNoExtraBatch() throws Exception {
        int previous=settings.maxPendingPerUser;settings.maxPendingPerUser=1;
        try{batch();assertThatThrownBy(this::batch).isInstanceOf(ApiError.class).hasMessageContaining("待处理");}
        finally{settings.maxPendingPerUser=previous;}
    }
    @Test void readTimeoutIsUncertainAndNeverAutomaticallyRetried() throws Exception {
        String id=batch();int original=settings.predictionTimeoutSeconds;settings.predictionTimeoutSeconds=1;FakePython.MODE.set(6);settings.executionEnabled=true;
        var coordinator=new TaskCoordinator(tasks,analysis,python,settings);
        try{
            coordinator.dispatch();long deadline=System.nanoTime()+4_000_000_000L;
            while(!"FAILED".equals(db.queryForObject("SELECT status FROM sample_tasks WHERE batch_id=?",String.class,id))&&System.nanoTime()<deadline)Thread.sleep(30);
            assertThat(db.queryForObject("SELECT error_message FROM sample_tasks WHERE batch_id=?",String.class,id)).contains("尚不确定");
            coordinator.dispatch();assertThat(db.queryForObject("SELECT attempt_no FROM sample_tasks WHERE batch_id=?",Integer.class,id)).isEqualTo(1);
        }finally{settings.executionEnabled=false;settings.predictionTimeoutSeconds=original;FakePython.MODE.set(0);coordinator.stop();}
    }
    @Test void retryingFailedBatchCannotExceedOwnersPendingQuota() throws Exception {
        int previous=settings.maxPendingPerUser;settings.maxPendingPerUser=1;
        try{
            String failed=batch();var attempt=tasks.claim();tasks.fail(attempt,"fixture failure");batch();
            assertThatThrownBy(()->batches.retryFailed(failed)).isInstanceOf(ApiError.class).hasMessageContaining("待处理");
            assertThat(db.queryForObject("SELECT status FROM batches WHERE id=?",String.class,failed)).isEqualTo("FAILED");
        }finally{settings.maxPendingPerUser=previous;}
    }
    @Test void concurrentUploadAndRetryShareTheSameOwnerAdmissionLimit() throws Exception {
        int previous=settings.maxPendingPerUser;settings.maxPendingPerUser=1;
        var principal=accounts.loadUserByUsername(accounts.current().username());
        try{
            String failed=batch();tasks.fail(tasks.claim(),"fixture failure");
            try(var pool=Executors.newVirtualThreadPerTaskExecutor()){
                Callable<Integer> retry=()->{GroupTestClient.as(principal);try{batches.retryFailed(failed);return 200;}catch(ApiError e){return e.status;}finally{org.springframework.security.core.context.SecurityContextHolder.clearContext();}};
                Callable<Integer> upload=()->{GroupTestClient.as(principal);try{batch();return 200;}catch(ApiError e){return e.status;}finally{org.springframework.security.core.context.SecurityContextHolder.clearContext();}};
                var outcomes=pool.invokeAll(List.of(retry,upload));assertThat(List.of(outcomes.get(0).get(),outcomes.get(1).get())).containsExactlyInAnyOrder(200,429);
            }
            assertThat(db.queryForObject("SELECT COUNT(*) FROM sample_tasks t JOIN batches b ON b.id=t.batch_id WHERE b.owner_id=? AND t.status IN ('PENDING','RUNNING')",Integer.class,principal.actor().id())).isEqualTo(1);
        }finally{settings.maxPendingPerUser=previous;}
    }
}

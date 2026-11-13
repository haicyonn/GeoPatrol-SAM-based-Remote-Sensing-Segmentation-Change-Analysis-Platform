package io.xunbian.backend;

import jakarta.annotation.PreDestroy;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

@Component
public class TaskCoordinator {
    private static final Logger log=LoggerFactory.getLogger(TaskCoordinator.class);
    final ExecutionRepository tasks;final AnalysisService analysis;final PythonClient python;final Settings settings;
    final ExecutorService executor=Executors.newSingleThreadExecutor();
    final AtomicReference<ExecutionRepository.Attempt> active=new AtomicReference<>();
    volatile boolean stopping=false;
    public TaskCoordinator(ExecutionRepository tasks,AnalysisService analysis,PythonClient python,Settings settings){this.tasks=tasks;this.analysis=analysis;this.python=python;this.settings=settings;}
    @Scheduled(fixedDelay=500,initialDelay=500)
    public void dispatch(){
        if(!settings.executionEnabled||stopping)return;
        try{
            tasks.recoverExpired();analysis.publishCompleted();
            if(active.get()!=null)return;
            if(tasks.workerProbeDue()){
                try{python.health();tasks.workerProbePassed();}catch(Exception e){tasks.workerProbeFailed();return;}
            }
            if(!analysis.admission.tryPrediction())return;
            ExecutionRepository.Attempt a;
            try{a=tasks.claim();}catch(RuntimeException e){analysis.admission.predictionFinished();throw e;}
            if(a==null){analysis.admission.predictionFinished();return;}active.set(a);
            executor.submit(()->{
                try{boolean accepted=tasks.complete(a,python.predict(a));analysis.admission.metrics.counter("xunbian.prediction.outcomes","status",accepted?"accepted":"stale").increment();}
                catch(Exception e){
                    // On shutdown leave the durable attempt leased. The next
                    // process recovers it after expiry instead of turning a
                    // recoverable interruption into a permanent sample failure.
                    if(!stopping)try{
                        if(e instanceof PythonClient.ModelFailure f&&f.retryable){tasks.retryLater(a,e.getMessage());analysis.admission.metrics.counter("xunbian.prediction.outcomes","status","retry_wait").increment();}
                        else {tasks.fail(a,e.getMessage());analysis.admission.metrics.counter("xunbian.prediction.outcomes","status","failed").increment();}
                        log.warn("Prediction attempt ended task={} attempt={} cause={}",a.taskId(),a.attemptId(),e.getClass().getSimpleName());
                    }catch(Exception persistence){log.error("Cannot record task failure {}",a.taskId());}
                }
                finally{active.compareAndSet(a,null);analysis.admission.predictionFinished();}
            });
        }catch(Exception e){log.error("Task scheduler failed: {}",e.getClass().getSimpleName());}
    }
    @Scheduled(fixedDelay=1000,initialDelay=1000)
    public void heartbeat(){
        var a=active.get();if(!stopping&&a!=null)try{tasks.heartbeat(a);}catch(Exception e){log.warn("Task heartbeat unavailable: {}",e.getClass().getSimpleName());}
    }
    @PreDestroy public void stop(){stopping=true;executor.shutdownNow();}
}

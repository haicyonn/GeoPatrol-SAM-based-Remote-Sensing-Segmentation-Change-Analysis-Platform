package io.xunbian.backend;

import java.util.Set;
import java.util.concurrent.*;
import io.micrometer.core.instrument.MeterRegistry;
import org.springframework.stereotype.Component;

/** Bounds waiting work as well as active model calls; the Python lock is the final GPU guard. */
@Component
public class ResourceAdmission {
    final Semaphore gpu=new Semaphore(1,true), requests;
    final Set<String> conversations=ConcurrentHashMap.newKeySet(),users=ConcurrentHashMap.newKeySet();
    final Settings settings;final MeterRegistry metrics;
    public ResourceAdmission(Settings settings,MeterRegistry metrics){
        if(settings.maxQuestions<1||settings.maxQuestions>20||settings.questionQueueSeconds<1||settings.maxPendingPerUser<1)throw new IllegalArgumentException("模型准入配置无效");
        this.settings=settings;this.metrics=metrics;requests=new Semaphore(settings.maxQuestions);
        metrics.gauge("xunbian.questions.admitted",users,Set::size);
    }
    public Turn enter(String user,String conversation){
        if(!conversations.add(conversation))throw new ApiError(409,"本会话上一条问题仍在处理，请等待结果");
        if(!users.add(user)){conversations.remove(conversation);throw new ApiError(429,"您已有一个分析请求在处理，请稍后再试");}
        if(!requests.tryAcquire()){users.remove(user);conversations.remove(conversation);throw new ApiError(429,"模型分析队列已满，请稍后再试");}
        return new Turn(user,conversation);
    }
    public boolean tryPrediction(){return !gpu.hasQueuedThreads()&&gpu.tryAcquire();}
    public void predictionFinished(){gpu.release();}
    public class Turn implements AutoCloseable {
        final String user,conversation;boolean held=false;
        Turn(String user,String conversation){this.user=user;this.conversation=conversation;}
        public void awaitGpu(){
            var started=io.micrometer.core.instrument.Timer.start(metrics);
            try{held=gpu.tryAcquire(settings.questionQueueSeconds,TimeUnit.SECONDS);}
            catch(InterruptedException e){Thread.currentThread().interrupt();throw new ApiError(503,"分析等待已中断");}
            finally{started.stop(metrics.timer("xunbian.questions.queue","outcome",held?"admitted":"rejected"));}
            if(!held)throw new ApiError(503,"模型正在处理其他任务，本次等待已结束，请稍后重试");
        }
        public void close(){if(held)gpu.release();requests.release();users.remove(user);conversations.remove(conversation);}
    }
}

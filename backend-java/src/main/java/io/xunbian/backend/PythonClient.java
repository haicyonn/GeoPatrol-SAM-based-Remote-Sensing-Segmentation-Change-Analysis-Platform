package io.xunbian.backend;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.net.*;
import java.net.http.*;
import java.time.Duration;
import java.util.*;
import org.springframework.stereotype.Component;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;

@Component
public class PythonClient {
    private static final org.slf4j.Logger LOG=org.slf4j.LoggerFactory.getLogger(PythonClient.class);
    final Settings settings;final ObjectMapper json;final MeterRegistry metrics;
    final HttpClient client=HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(3)).build();
    public PythonClient(Settings settings,ObjectMapper json,MeterRegistry metrics){
        this.settings=settings;this.json=json;this.metrics=metrics;
        URI uri=URI.create(settings.pythonUrl);
        if(!"http".equals(uri.getScheme())||!Set.of("127.0.0.1","localhost","[::1]").contains(uri.getHost()))
            throw new IllegalArgumentException("Python adapter must use a loopback HTTP endpoint");
    }
    Map<String,Object> call(String route,Map<String,Object> body,int timeout){
        var sample=Timer.start(metrics);String outcome="error";
        try{
            var builder=HttpRequest.newBuilder(URI.create(settings.pythonUrl+route))
                .timeout(Duration.ofSeconds(timeout)).header("X-Xunbian-Token",settings.internalToken);
            if(body==null)builder.GET();else builder.header("Content-Type","application/json").POST(HttpRequest.BodyPublishers.ofString(json.writeValueAsString(body)));
            var response=client.send(builder.build(),HttpResponse.BodyHandlers.ofString());
            if(response.statusCode()!=200){
                if(response.statusCode()==400||response.statusCode()==409){
                    var detail=json.readValue(response.body(),Map.class);throw new ApiError(response.statusCode(),Objects.toString(detail.get("error"),"请求条件无效"));
                }
                throw new ModelFailure(Set.of(429,502,503,504).contains(response.statusCode()),false,"模型服务返回错误（"+response.statusCode()+"）");
            }
            if(response.body().length()>8_000_000)throw new ApiError(502,"模型响应超过大小限制");
            Map<String,Object> result=json.readValue(response.body(),new TypeReference<>(){});outcome="ok";return result;
        }catch(ApiError e){throw e;}
        catch(InterruptedException e){Thread.currentThread().interrupt();throw new ApiError(503,"模型请求已中断");}
        catch(HttpConnectTimeoutException|ConnectException e){throw new ModelFailure(true,false,"模型服务暂时无法连接");}
        catch(HttpTimeoutException e){throw new ModelFailure(false,true,"模型请求超时，后台执行结果尚不确定；未自动重复推理，请核查后重试");}
        catch(Exception e){LOG.warn("Internal Python request failed: route={}",route,e);throw new ApiError(502,"Python服务连接或响应失败："+e.getClass().getSimpleName());}
        finally{sample.stop(metrics.timer("xunbian.python.call","route",route,"outcome",outcome));}
    }
    static class ModelFailure extends ApiError {
        final boolean retryable,uncertain;
        ModelFailure(boolean retryable,boolean uncertain,String message){super(uncertain?504:502,message);this.retryable=retryable;this.uncertain=uncertain;}
    }
    Map<String,Object> predict(ExecutionRepository.Attempt attempt){return call("/internal/predict",attempt.payload(),settings.predictionTimeoutSeconds);}
    Map<String,Object> health(){return call("/internal/health",null,3);}
    Map<String,Object> models(){return call("/internal/models",null,3);}
}

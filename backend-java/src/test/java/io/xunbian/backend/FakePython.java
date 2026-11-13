package io.xunbian.backend;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpServer;
import java.awt.image.BufferedImage;
import java.nio.file.*;
import java.net.InetSocketAddress;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import javax.imageio.ImageIO;

/** Deterministic HTTP dependency for fault injection, not model accuracy evidence. */
final class FakePython {
    static final ObjectMapper JSON=new ObjectMapper();
    static final Path ROOT;
    static final HttpServer SERVER;
    static final AtomicInteger MODE=new AtomicInteger();
    static final AtomicInteger PREDICTIONS=new AtomicInteger();
    static volatile Map<String,Object> dataset;
    static volatile Map<String,Object> questionRequest;
    static {
        try{
            ROOT=Files.createTempDirectory("xunbian-java-test-");
            SERVER=HttpServer.create(new InetSocketAddress("127.0.0.1",0),0);
            SERVER.createContext("/internal/",exchange->{
                try{
                    Map<String,Object> input=exchange.getRequestMethod().equals("POST")?JSON.readValue(exchange.getRequestBody(),Map.class):Map.of();
                    Map<String,Object> output;
                    if(exchange.getRequestURI().getPath().endsWith("predict")){
                        PREDICTIONS.incrementAndGet();
                        if(MODE.get()==3||MODE.get()==6)Thread.sleep(2000);
                        if(MODE.get()==1||MODE.get()==5){byte[] b="{\"error\":\"injected model failure\"}".getBytes();exchange.sendResponseHeaders(MODE.get()==5?503:500,b.length);exchange.getResponseBody().write(b);exchange.close();return;}
                        String asset="pred-"+input.get("taskId")+"-"+input.get("attemptId")+".png";
                        var image=new BufferedImage(32,32,BufferedImage.TYPE_BYTE_GRAY);
                        for(int x=0;x<4;x++)image.getRaster().setSample(x,0,0,255);
                        ImageIO.write(image,"png",ROOT.resolve(asset).toFile());
                        output=new LinkedHashMap<>(input);
                        output.putAll(Map.of("predMaskAssetId",asset,"width",32,"height",32,
                            "changedPx",MODE.get()==2?5:4,"changedRatio",4.0/1024,"objectCount",1,
                            "modelVersion","0".repeat(64),"maskQuality",Map.of("hole_ratio",0,"hole_count",0)));
                    }else if(exchange.getRequestURI().getPath().endsWith("corrections")){
                        String asset="correction-"+input.get("correctionId")+".png";
                        var image=new BufferedImage(32,32,BufferedImage.TYPE_BYTE_GRAY);
                        for(int y=2;y<=6;y++)for(int x=2;x<=6;x++)image.getRaster().setSample(x,y,0,255);
                        ImageIO.write(image,"png",ROOT.resolve(asset).toFile());
                        output=new LinkedHashMap<>(input);output.putAll(Map.of("maskAssetId",asset,"changedPx",25,"changedRatio",25.0/1024,"objectCount",1,"maskQuality",Map.of("hole_ratio",0,"hole_count",0),"bbox",List.of(2,2,7,7)));
                    }else if(exchange.getRequestURI().getPath().endsWith("datasets")){
                        dataset=(Map<String,Object>)input.get("data");output=Map.of("datasetId",input.get("datasetId"));
                    }else if(exchange.getRequestURI().getPath().endsWith("questions")){
                        questionRequest=input;
                        if(MODE.get()==4)Thread.sleep(2000);
                        if("history".equals(input.get("action")))output=Map.of("status","history","messages",List.of());
                        else output=Map.of("status","completed","sqlQuery","SELECT COUNT(*) AS n FROM samples",
                            "sqlRows",List.of(Map.of("n",1)),"datasetId",input.get("datasetId"),
                            "trustedUser",Objects.toString(input.get("userId"),"missing"),"trustedSession",input.get("sessionId"));
                    }else output=Map.of("status","ok","models",List.of(Map.of("modelId","secor-whu","available",true)));
                    byte[] bytes=JSON.writeValueAsBytes(output);
                    exchange.getResponseHeaders().set("Content-Type","application/json");
                    exchange.sendResponseHeaders(200,bytes.length);exchange.getResponseBody().write(bytes);exchange.close();
                }catch(Exception error){exchange.sendResponseHeaders(500,-1);exchange.close();}
            });
            SERVER.setExecutor(java.util.concurrent.Executors.newVirtualThreadPerTaskExecutor());SERVER.start();
        }catch(Exception e){throw new ExceptionInInitializerError(e);}
    }
    static String url(){return "http://127.0.0.1:"+SERVER.getAddress().getPort();}
}

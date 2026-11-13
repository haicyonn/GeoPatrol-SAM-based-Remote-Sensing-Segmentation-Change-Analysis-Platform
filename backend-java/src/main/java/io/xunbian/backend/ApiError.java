package io.xunbian.backend;

import java.util.Map;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MaxUploadSizeExceededException;

public class ApiError extends RuntimeException {
    final int status;
    final Map<String,Object> details;
    public ApiError(int status,String message){this(status,message,Map.of());}
    public ApiError(int status,String message,Map<String,Object> details){super(message);this.status=status;this.details=details;}

    @RestControllerAdvice
    public static class Handler {
        @ExceptionHandler(ApiError.class) ResponseEntity<?> api(ApiError e){
            var body=new java.util.LinkedHashMap<String,Object>();body.put("error",e.getMessage());body.putAll(e.details);return ResponseEntity.status(e.status).body(body);
        }
        @ExceptionHandler(MaxUploadSizeExceededException.class) ResponseEntity<?> size(Exception e){
            return ResponseEntity.status(413).body(Map.of("error","文件或整批请求超过大小限制"));
        }
        @ExceptionHandler(IllegalArgumentException.class) ResponseEntity<?> invalid(Exception e){
            return ResponseEntity.badRequest().body(Map.of("error","请求参数无效"));
        }
    }
}

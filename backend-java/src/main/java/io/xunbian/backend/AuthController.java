package io.xunbian.backend;

import java.util.*;
import jakarta.servlet.http.HttpServletRequest;
import org.springframework.security.web.csrf.CsrfToken;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1")
public class AuthController {
    final AccountService accounts;
    public AuthController(AccountService accounts){this.accounts=accounts;}
    @GetMapping("/auth/csrf") Object csrf(CsrfToken token){return Map.of("token",token.getToken(),"headerName",token.getHeaderName());}
    @GetMapping("/auth/me") Object me(){return accounts.current();}
    @PostMapping("/auth/password") Object password(@RequestBody Map<String,Object> body,HttpServletRequest request){
        accounts.changePassword((String)body.get("currentPassword"),(String)body.get("password"));request.getSession().invalidate();return Map.of("status","ok");
    }
    @GetMapping("/admin/users") Object users(){return Map.of("users",accounts.list());}
    @PostMapping("/admin/users") Object create(@RequestBody Map<String,Object> body){return accounts.create((String)body.get("username"),(String)body.get("displayName"),(String)body.get("password"),Objects.toString(body.getOrDefault("role","MEMBER")));}
    @PostMapping("/admin/users/{id}/enabled") Object enabled(@PathVariable String id,@RequestBody Map<String,Object> body){
        if(!(body.get("enabled") instanceof Boolean value))throw new ApiError(400,"enabled必须为布尔值");accounts.enabled(id,value);return Map.of("status","ok");
    }
    @PostMapping("/admin/users/{id}/password") Object reset(@PathVariable String id,@RequestBody Map<String,Object> body){accounts.resetPassword(id,(String)body.get("password"));return Map.of("status","ok");}
}

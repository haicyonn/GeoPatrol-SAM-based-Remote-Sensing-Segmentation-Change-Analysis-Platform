package io.xunbian.backend;

import java.net.*;
import java.net.http.HttpClient;
import java.util.*;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.http.*;
import org.springframework.http.client.JdkClientHttpRequestFactory;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.web.util.DefaultUriBuilderFactory;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;

/** Real session cookies and CSRF, never a production authentication bypass. */
final class GroupTestClient {
    static final String PASSWORD="test-passphrase-735a";
    static void as(AccountService.Principal p){SecurityContextHolder.getContext().setAuthentication(new UsernamePasswordAuthenticationToken(p,p.hash(),p.getAuthorities()));}
    static AccountService.Principal member(AccountService accounts){
        as(accounts.loadUserByUsername("admin"));String name="t"+UUID.randomUUID().toString().replace("-","").substring(0,18);
        try{accounts.create(name,name,PASSWORD,"MEMBER");return accounts.loadUserByUsername(name);}finally{SecurityContextHolder.clearContext();}
    }
    static TestRestTemplate anonymous(int port){
        return anonymous(port,new CookieManager(null,CookiePolicy.ACCEPT_ALL));
    }
    static TestRestTemplate anonymous(int port,CookieManager cookies){
        var client=HttpClient.newBuilder().cookieHandler(cookies).build();
        var http=new TestRestTemplate(new RestTemplateBuilder().requestFactory(()->new JdkClientHttpRequestFactory(client)));
        http.setUriTemplateHandler(new DefaultUriBuilderFactory("http://127.0.0.1:"+port));return http;
    }
    static TestRestTemplate login(int port,String username,String password){
        return login(anonymous(port),username,password);
    }
    static TestRestTemplate login(TestRestTemplate http,String username,String password){
        var csrf=http.getForObject("/api/v1/auth/csrf",Map.class);
        var form=new LinkedMultiValueMap<String,String>();form.add("username",username);form.add("password",password);
        var headers=new HttpHeaders();headers.setContentType(MediaType.APPLICATION_FORM_URLENCODED);headers.set(csrf.get("headerName").toString(),csrf.get("token").toString());
        var response=http.postForEntity("/api/v1/auth/login",new HttpEntity<>(form,headers),Map.class);
        if(response.getStatusCode()!=HttpStatus.OK)throw new AssertionError("real login failed: "+response);
        var fresh=http.getForObject("/api/v1/auth/csrf",Map.class);
        http.getRestTemplate().getInterceptors().add((request,body,execution)->{
            if(!Set.of(HttpMethod.GET,HttpMethod.HEAD,HttpMethod.OPTIONS).contains(request.getMethod()))request.getHeaders().set(fresh.get("headerName").toString(),fresh.get("token").toString());
            return execution.execute(request,body);
        });return http;
    }
    static TestRestTemplate login(int port,AccountService.Principal p){return login(port,p.getUsername(),PASSWORD);}
}

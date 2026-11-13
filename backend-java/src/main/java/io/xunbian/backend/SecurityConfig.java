package io.xunbian.backend;

import java.io.IOException;
import java.util.Map;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.servlet.*;
import jakarta.servlet.http.*;
import org.springframework.context.annotation.*;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.*;
import org.springframework.security.web.access.intercept.AuthorizationFilter;
import org.springframework.web.filter.OncePerRequestFilter;

@Configuration
public class SecurityConfig {
    static void reply(HttpServletResponse response,int status,String message) throws IOException {
        response.setStatus(status);response.setContentType("application/json;charset=UTF-8");
        response.getWriter().write(new ObjectMapper().writeValueAsString(Map.of("error",message)));
    }
    @Bean PasswordEncoder passwordEncoder(AccountService accounts){return accounts.passwords;}
    @Bean SecurityFilterChain security(HttpSecurity http,AccountService accounts) throws Exception {
        http.authorizeHttpRequests(a->a.requestMatchers("/","/index.html","/console.html","/manage.html","/analysis.html","/*.js","/*.css","/console-assets/**","/ui-assets/**","/favicon.ico","/api/v1/auth/csrf","/api/v1/auth/login","/api/v1/health","/error").permitAll()
            .requestMatchers("/actuator/**","/api/v1/admin/**").hasRole("ADMIN").anyRequest().authenticated())
            .formLogin(f->f.loginProcessingUrl("/api/v1/auth/login")
                .successHandler((req,res,auth)->{res.setContentType("application/json;charset=UTF-8");res.getWriter().write("{\"status\":\"ok\"}");})
                .failureHandler((req,res,e)->reply(res,401,"账号或密码错误，或账号已禁用")))
            .logout(l->l.logoutUrl("/api/v1/auth/logout").logoutSuccessHandler((req,res,a)->{res.setContentType("application/json;charset=UTF-8");res.getWriter().write("{\"status\":\"ok\"}");}))
            .exceptionHandling(e->e.authenticationEntryPoint((req,res,error)->reply(res,401,"请先登录"))
                .accessDeniedHandler((req,res,error)->reply(res,403,"没有操作权限或请求校验已失效，请刷新后重试")))
            .requestCache(c->c.disable())
            .addFilterBefore(new LoginThrottle(),org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter.class)
            .addFilterBefore(new OncePerRequestFilter(){
                @Override protected void doFilterInternal(HttpServletRequest req,HttpServletResponse res,FilterChain chain) throws ServletException,IOException {
                    var auth=SecurityContextHolder.getContext().getAuthentication();
                    if(auth!=null&&auth.getPrincipal() instanceof AccountService.Principal p){
                        boolean valid;
                        try{var latest=accounts.loadUserByUsername(p.actor().username());valid=latest.active()&&latest.actor().version()==p.actor().version();}
                        catch(org.springframework.security.core.userdetails.UsernameNotFoundException e){valid=false;}
                        if(!valid){SecurityContextHolder.clearContext();var session=req.getSession(false);if(session!=null)session.invalidate();reply(res,401,"账号状态已变化，请重新登录");return;}
                    }
                    chain.doFilter(req,res);
                }
            },AuthorizationFilter.class);
        return http.build();
    }
}

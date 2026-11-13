package io.xunbian.backend;

import java.io.IOException;
import java.util.Locale;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import jakarta.servlet.*;
import jakarta.servlet.http.*;
import org.springframework.web.filter.OncePerRequestFilter;

/** Small single-instance login guard, independent of GPU admission. */
public class LoginThrottle extends OncePerRequestFilter {
    record Window(long started,AtomicInteger failures){}
    final ConcurrentHashMap<String,Window> windows=new ConcurrentHashMap<>();
    final Semaphore checking=new Semaphore(4);
    @Override protected void doFilterInternal(HttpServletRequest req,HttpServletResponse res,FilterChain chain) throws ServletException,IOException {
        if(!"POST".equals(req.getMethod())||!"/api/v1/auth/login".equals(req.getServletPath())){chain.doFilter(req,res);return;}
        long now=System.nanoTime();windows.entrySet().removeIf(e->now-e.getValue().started()>60_000_000_000L);
        String username=java.util.Objects.toString(req.getParameter("username"),"");
        if(username.length()>32){SecurityConfig.reply(res,401,"账号或密码错误，或账号已禁用");return;}
        String key=req.getRemoteAddr()+":"+username.toLowerCase(Locale.ROOT);
        if(!windows.containsKey(key)&&windows.size()>=2048){SecurityConfig.reply(res,429,"登录请求较多，请稍后再试");return;}
        var window=windows.computeIfAbsent(key,k->new Window(now,new AtomicInteger()));
        if(window.failures().get()>=8){SecurityConfig.reply(res,429,"登录失败次数较多，请一分钟后再试");return;}
        if(!checking.tryAcquire()){SecurityConfig.reply(res,429,"登录服务忙碌，请稍后再试");return;}
        try{chain.doFilter(req,res);if(res.getStatus()==401)window.failures().incrementAndGet();else if(res.getStatus()==200)windows.remove(key);}
        finally{checking.release();}
    }
}

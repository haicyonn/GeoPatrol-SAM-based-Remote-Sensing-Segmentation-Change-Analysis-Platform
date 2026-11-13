package io.xunbian.backend;

import jakarta.servlet.*;
import jakarta.servlet.http.*;
import java.io.IOException;
import java.net.URI;
import java.util.Set;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

/** Reject cross-site browser writes, allowing the application's configured same origin. */
@Component
public class LocalRequestFilter extends OncePerRequestFilter {
    @Override protected void doFilterInternal(HttpServletRequest request,HttpServletResponse response,FilterChain chain) throws ServletException,IOException {
        String origin=request.getHeader("Origin");
        if(origin!=null&&!Set.of("GET","HEAD","OPTIONS").contains(request.getMethod())){
            boolean allowed=false;
            try{
                URI uri=URI.create(origin);
                int port=uri.getPort()<0?("https".equals(uri.getScheme())?443:80):uri.getPort();
                allowed=request.getScheme().equals(uri.getScheme())&&request.getServerName().equalsIgnoreCase(uri.getHost())&&port==request.getServerPort();
            }catch(Exception ignored){}
            if(!allowed){response.setStatus(403);response.setContentType("application/json;charset=UTF-8");response.getWriter().write("{\"error\":\"请通过本地巡变页面提交请求\"}");return;}
        }
        chain.doFilter(request,response);
    }
}

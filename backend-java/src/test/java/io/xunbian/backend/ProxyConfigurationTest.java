package io.xunbian.backend;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.http.*;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import static org.assertj.core.api.Assertions.assertThat;

@ActiveProfiles("proxy")
@SpringBootTest(webEnvironment=SpringBootTest.WebEnvironment.RANDOM_PORT,
    properties={"spring.datasource.url=${XB_TEST_DB_URL}","xunbian.execution-enabled=false","server.servlet.session.cookie.secure=false"})
class ProxyConfigurationTest {
    @LocalServerPort int port;
    @Autowired AccountService accounts;
    @DynamicPropertySource static void adapter(DynamicPropertyRegistry p){p.add("xunbian.storage-root",()->FakePython.ROOT.toString());p.add("xunbian.python-url",FakePython::url);p.add("xunbian.bootstrap-file",()->FakePython.ROOT.resolve("initial-test-admin.json").toString());}
    @Test void trustedLocalProxyPreservesExternalHttpsOriginAndNondefaultPort() throws Exception {
        var user=GroupTestClient.member(accounts);var cookies=new java.net.CookieManager(null,java.net.CookiePolicy.ACCEPT_ALL);var http=GroupTestClient.anonymous(port,cookies);
        http.getRestTemplate().getInterceptors().add((request,body,execution)->{
            request.getHeaders().set("X-Forwarded-Proto","https");request.getHeaders().set("X-Forwarded-Host","xunbian.example");request.getHeaders().set("X-Forwarded-Port","9443");
            // This test uses plain HTTP between client/proxy and the embedded app.
            // A real browser uses HTTPS and sends these Secure cookies normally.
            var cookie=cookies.getCookieStore().getCookies().stream().map(c->c.getName()+"="+c.getValue()).collect(java.util.stream.Collectors.joining("; "));
            if(!cookie.isEmpty())request.getHeaders().set("Cookie",cookie);
            if(request.getMethod()==HttpMethod.POST)request.getHeaders().setOrigin("https://xunbian.example:9443");
            return execution.execute(request,body);
        });
        GroupTestClient.login(http,user.getUsername(),GroupTestClient.PASSWORD);
        assertThat(http.getForEntity("/api/v1/auth/me",java.util.Map.class).getStatusCode()).isEqualTo(HttpStatus.OK);
        var form=new org.springframework.util.LinkedMultiValueMap<String,Object>();form.add("label","proxy-origin fixture");form.add("regionId","whu");form.add("before",BackendIntegrationTest.image("same.png",32,32));form.add("after",BackendIntegrationTest.image("same.png",32,32));
        var headers=new HttpHeaders();headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        assertThat(http.postForEntity("/api/v1/batches",new HttpEntity<>(form,headers),java.util.Map.class).getStatusCode()).isEqualTo(HttpStatus.CREATED);
    }
}

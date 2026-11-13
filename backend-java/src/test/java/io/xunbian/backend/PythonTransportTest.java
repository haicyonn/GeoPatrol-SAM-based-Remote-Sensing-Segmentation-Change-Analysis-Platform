package io.xunbian.backend;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import org.junit.jupiter.api.*;
import static org.assertj.core.api.Assertions.*;

/** Real Java HttpClient against our Python HTTP server, not the HTTP/1.1 FakePython. */
class PythonTransportTest {
    @Test void largeRegistrationThenSmallPostsDoNotReuseClosedPythonConnections() throws Exception {
        Path python=Path.of(System.getProperty("user.home"),".venvs/trust-agent-demo/bin/python");
        Assumptions.assumeTrue(Files.isExecutable(python),"WSL integration runtime required");
        Path root=Path.of("..").toAbsolutePath().normalize(),temp=Files.createTempDirectory("python-http-contract-");
        String token="test-service-transport-token-32-characters";
        String program="""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1])/'tools'))
from java_backend_bridge import Bridge,create_server
folder=Path(sys.argv[2]);bridge=Bridge(folder/'assets',folder/'state',generate=None)
server=create_server(bridge,sys.argv[3],0)
print(server.server_port,flush=True)
server.serve_forever()
""";
        Process server=new ProcessBuilder(python.toString(),"-u","-c",program,root.toString(),temp.toString(),token).redirectError(temp.resolve("server-error.log").toFile()).start();
        try(var reader=new BufferedReader(new InputStreamReader(server.getInputStream()))){
            int port=Integer.parseInt(reader.readLine());var settings=new Settings();settings.pythonUrl="http://127.0.0.1:"+port;settings.internalToken=token;
            var client=new PythonClient(settings,new ObjectMapper(),new SimpleMeterRegistry());
            var data=Map.of("regions",List.of(Map.of("id","whu","short","WHU","items",List.of(Map.of("id","one","changedPx",1,"changedRatio",0.1,"objectCount",1)))),"padding","x".repeat(6_100_000));
            for(int i=0;i<12;i++){
                assertThat(client.call("/internal/datasets",Map.of("datasetId","transport-data","data",data),20)).containsEntry("datasetId","transport-data");
                assertThat(client.call("/internal/questions",Map.of("datasetId","transport-data","sessionId","transport-session","userId","test-member","action","history"),20)).containsEntry("status","history");
            }
        }finally{server.destroy();if(!server.waitFor(5,java.util.concurrent.TimeUnit.SECONDS))server.destroyForcibly();}
    }
}

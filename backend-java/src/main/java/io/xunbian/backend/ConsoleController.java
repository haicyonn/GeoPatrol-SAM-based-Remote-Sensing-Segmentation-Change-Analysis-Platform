package io.xunbian.backend;

import java.util.Map;
import org.springframework.web.bind.annotation.*;
import org.springframework.http.ResponseEntity;

@RestController
@RequestMapping("/api/v1")
public class ConsoleController {
    final ConsoleService console;final ReferenceArchive reference;final AnalysisService analysis;
    public ConsoleController(ConsoleService console,ReferenceArchive reference,AnalysisService analysis){this.console=console;this.reference=reference;this.analysis=analysis;}
    @GetMapping("/batches/{id}/console") Object console(@PathVariable String id,@RequestParam(defaultValue="effective") String basis){return console.console(id,basis);}
    @PostMapping("/batches/{id}/report-summary") Object report(@PathVariable String id,@RequestBody Map<String,Object> body){return console.summary(id,body);}
    @PostMapping("/batches/{id}/analysis") Object analysis(@PathVariable String id,@RequestBody Map<String,Object> body){return analysis.analysis(id,body);}
    @GetMapping("/reference-assets/{id}") ResponseEntity<?> asset(@PathVariable String id){return reference.asset(id);}
}

package io.xunbian.backend;

import java.util.Map;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/v1/batches/{id}")
public class ReviewController {
    final ReviewService reviews;final AnalysisService analysis;
    public ReviewController(ReviewService reviews,AnalysisService analysis){this.reviews=reviews;this.analysis=analysis;}
    @GetMapping("/reviews/{sample}") Object get(@PathVariable String id,@PathVariable String sample){return reviews.get(id,sample);}
    @GetMapping("/reviews/{sample}/history") Object history(@PathVariable String id,@PathVariable String sample){return reviews.history(id,sample);}
    @PutMapping("/reviews/{sample}") Object save(@PathVariable String id,@PathVariable String sample,@RequestBody Map<String,Object> body){return reviews.save(id,sample,body);}
    @GetMapping("/corrections") Object corrections(@PathVariable String id){return reviews.export(id);}
    @GetMapping("/report") ResponseEntity<?> report(@PathVariable String id,@RequestParam(defaultValue="effective") String basis){
        return ResponseEntity.ok().header("Content-Disposition","attachment; filename=\"xunbian-report-"+id+".json\"").body(analysis.report(id,basis));
    }
}

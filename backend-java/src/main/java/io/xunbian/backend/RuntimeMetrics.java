package io.xunbian.backend;

import io.micrometer.core.instrument.MeterRegistry;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

@Component
public class RuntimeMetrics {
    public RuntimeMetrics(JdbcTemplate db,MeterRegistry meters,SchemaMigration migration){
        for(String status:new String[]{"PENDING","RUNNING","FAILED"})
            io.micrometer.core.instrument.Gauge.builder("xunbian.tasks",db,d->d.queryForObject("SELECT COUNT(*) FROM sample_tasks WHERE status=?",Double.class,status)).tag("status",status).register(meters);
    }
}

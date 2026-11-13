package io.xunbian.backend;

import java.util.Map;
import org.springframework.boot.sql.init.dependency.DependsOnDatabaseInitialization;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

/** Additive upgrade of the already installed v1 database; no existing facts are rewritten. */
@Component
@DependsOnDatabaseInitialization
public class SchemaMigration {
    public SchemaMigration(JdbcTemplate db){
        add(db,"batches","owner_id","VARCHAR(36) NULL");
        add(db,"batches","last_dispatch_ms","BIGINT NOT NULL DEFAULT 0");
        add(db,"batches","review_version","BIGINT NOT NULL DEFAULT 0");
        add(db,"sample_tasks","next_eligible_ms","BIGINT NOT NULL DEFAULT 0");
        add(db,"sample_tasks","transient_failures","INT NOT NULL DEFAULT 0");
        add(db,"sample_tasks","lease_expirations","INT NOT NULL DEFAULT 0");
        add(db,"execution_slots","available_after_ms","BIGINT NOT NULL DEFAULT 0");
        add(db,"batches","source_type","VARCHAR(16) NOT NULL DEFAULT 'INFERENCE'");
        add(db,"batches","provenance_json","LONGTEXT NULL");
        add(db,"batches","task_type","VARCHAR(32) NOT NULL DEFAULT 'change_detection'");
        if(db.queryForObject("SELECT COUNT(*) FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name='sample_tasks' AND column_name='after_asset' AND is_nullable='NO'",Integer.class)>0)
            db.execute("ALTER TABLE sample_tasks MODIFY after_asset VARCHAR(191) NULL");
        db.execute("CREATE TABLE IF NOT EXISTS segmentation_task_results (task_id VARCHAR(36) PRIMARY KEY,batch_id VARCHAR(36) NOT NULL,attempt_id VARCHAR(36) NOT NULL,model_id VARCHAR(64) NOT NULL,model_version CHAR(64) NOT NULL,foreground_pixels BIGINT NOT NULL,foreground_ratio DOUBLE NOT NULL,component_count INT NOT NULL,pred_mask_asset VARCHAR(191) NOT NULL,result_json LONGTEXT NOT NULL,created_ms BIGINT NOT NULL,FOREIGN KEY(task_id) REFERENCES sample_tasks(id),KEY segmentation_result_batch(batch_id)) ENGINE=InnoDB");
        db.update("INSERT IGNORE INTO sample_annotations SELECT batch_id,task_id,version,JSON_OBJECT('verdict',verdict,'note',note),NULL,author_id,updated_ms FROM sample_reviews");
        db.update("INSERT IGNORE INTO sample_annotation_history SELECT batch_id,task_id,version,JSON_OBJECT('verdict',verdict,'note',note),NULL,author_id,created_ms FROM review_history");
    }
    private void add(JdbcTemplate db,String table,String column,String definition){
        if(db.queryForObject("SELECT COUNT(*) FROM information_schema.columns WHERE table_schema=DATABASE() AND table_name=? AND column_name=?",Integer.class,table,column)==0)
            db.execute("ALTER TABLE "+table+" ADD COLUMN "+column+" "+definition);
    }
}

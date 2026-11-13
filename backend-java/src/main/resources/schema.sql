CREATE TABLE IF NOT EXISTS batches (
 id VARCHAR(36) PRIMARY KEY, label VARCHAR(200) NOT NULL, status VARCHAR(24) NOT NULL,
 model_id VARCHAR(64), total INT NOT NULL, created_ms BIGINT NOT NULL, updated_ms BIGINT NOT NULL
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS sample_tasks (
 id VARCHAR(36) PRIMARY KEY, batch_id VARCHAR(36) NOT NULL, sample_name VARCHAR(200) NOT NULL,
 region_id VARCHAR(64) NOT NULL, before_asset VARCHAR(191) NOT NULL, after_asset VARCHAR(191) NOT NULL,
 width INT NOT NULL, height INT NOT NULL, status VARCHAR(24) NOT NULL,
 attempt_id VARCHAR(36), attempt_no INT NOT NULL DEFAULT 0, lease_until BIGINT, error_message TEXT,
 created_ms BIGINT NOT NULL, updated_ms BIGINT NOT NULL,
 UNIQUE KEY unique_batch_sample(batch_id,sample_name), KEY task_queue(status,created_ms),
 FOREIGN KEY(batch_id) REFERENCES batches(id)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS task_attempts (
 id VARCHAR(36) PRIMARY KEY, task_id VARCHAR(36) NOT NULL, status VARCHAR(24) NOT NULL,
 started_ms BIGINT NOT NULL, finished_ms BIGINT, error_message TEXT,
 FOREIGN KEY(task_id) REFERENCES sample_tasks(id)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS assets (
 asset_id VARCHAR(191) PRIMARY KEY, batch_id VARCHAR(36) NOT NULL, task_id VARCHAR(36),
 role VARCHAR(24) NOT NULL, mime_type VARCHAR(64) NOT NULL, byte_size BIGINT NOT NULL,
 FOREIGN KEY(batch_id) REFERENCES batches(id)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS prediction_results (
 task_id VARCHAR(36) PRIMARY KEY, batch_id VARCHAR(36) NOT NULL, attempt_id VARCHAR(36) NOT NULL,
 model_id VARCHAR(64) NOT NULL, model_version VARCHAR(64) NOT NULL,
 changed_px BIGINT NOT NULL, changed_ratio DOUBLE NOT NULL, object_count INT NOT NULL,
 pred_mask_asset VARCHAR(191) NOT NULL, result_json LONGTEXT NOT NULL, created_ms BIGINT NOT NULL,
 FOREIGN KEY(task_id) REFERENCES sample_tasks(id), KEY result_batch(batch_id)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS execution_slots (
 slot_id INT PRIMARY KEY, task_id VARCHAR(36), attempt_id VARCHAR(36), lease_until BIGINT
) ENGINE=InnoDB;
INSERT IGNORE INTO execution_slots(slot_id) VALUES(1);
CREATE TABLE IF NOT EXISTS analysis_datasets (
 batch_id VARCHAR(36) PRIMARY KEY, data_json LONGTEXT NOT NULL, source_digest CHAR(64) NOT NULL,
 created_ms BIGINT NOT NULL, FOREIGN KEY(batch_id) REFERENCES batches(id)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS app_users (
 id VARCHAR(36) PRIMARY KEY, username VARCHAR(32) NOT NULL UNIQUE,
 display_name VARCHAR(60) NOT NULL, password_hash VARCHAR(100) NOT NULL,
 role VARCHAR(16) NOT NULL, enabled BOOLEAN NOT NULL DEFAULT TRUE,
 auth_version BIGINT NOT NULL DEFAULT 1, created_ms BIGINT NOT NULL
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS app_bootstrap (id INT PRIMARY KEY) ENGINE=InnoDB;
INSERT IGNORE INTO app_bootstrap VALUES(1);
CREATE TABLE IF NOT EXISTS batch_members (
 batch_id VARCHAR(36) NOT NULL, user_id VARCHAR(36) NOT NULL, permission VARCHAR(8) NOT NULL,
 PRIMARY KEY(batch_id,user_id), FOREIGN KEY(batch_id) REFERENCES batches(id),
 FOREIGN KEY(user_id) REFERENCES app_users(id)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS sample_reviews (
 task_id VARCHAR(36) PRIMARY KEY, batch_id VARCHAR(36) NOT NULL,
 version BIGINT NOT NULL, verdict VARCHAR(24) NOT NULL, note TEXT NOT NULL,
 author_id VARCHAR(36) NOT NULL, updated_ms BIGINT NOT NULL,
 FOREIGN KEY(task_id) REFERENCES sample_tasks(id), FOREIGN KEY(author_id) REFERENCES app_users(id)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS review_history (
 task_id VARCHAR(36) NOT NULL, version BIGINT NOT NULL, batch_id VARCHAR(36) NOT NULL,
 verdict VARCHAR(24) NOT NULL, note TEXT NOT NULL, author_id VARCHAR(36) NOT NULL,
 created_ms BIGINT NOT NULL, PRIMARY KEY(task_id,version),
 FOREIGN KEY(task_id) REFERENCES sample_tasks(id), FOREIGN KEY(author_id) REFERENCES app_users(id)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS query_conversations (
 user_id VARCHAR(36) NOT NULL, batch_id VARCHAR(36) NOT NULL, session_hash CHAR(64) NOT NULL,
 review_version BIGINT NOT NULL, updated_ms BIGINT NOT NULL,
 PRIMARY KEY(user_id,batch_id,session_hash), FOREIGN KEY(user_id) REFERENCES app_users(id),
 FOREIGN KEY(batch_id) REFERENCES batches(id)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS reference_sources (
 batch_id VARCHAR(36) PRIMARY KEY, source_root TEXT NOT NULL, console_json LONGTEXT NOT NULL,
 FOREIGN KEY(batch_id) REFERENCES batches(id)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS reference_assets (
 asset_id CHAR(64) PRIMARY KEY, batch_id VARCHAR(36) NOT NULL, relative_path VARCHAR(1024) NOT NULL,
 mime_type VARCHAR(64) NOT NULL, FOREIGN KEY(batch_id) REFERENCES batches(id), KEY reference_batch(batch_id)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS sample_annotations (
 batch_id VARCHAR(36) NOT NULL,sample_id VARCHAR(191) NOT NULL,version BIGINT NOT NULL,
 data_json LONGTEXT NOT NULL,approved_json LONGTEXT NULL,author_id VARCHAR(36) NOT NULL,updated_ms BIGINT NOT NULL,
 PRIMARY KEY(batch_id,sample_id),FOREIGN KEY(batch_id) REFERENCES batches(id),FOREIGN KEY(author_id) REFERENCES app_users(id)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS sample_annotation_history (
 batch_id VARCHAR(36) NOT NULL,sample_id VARCHAR(191) NOT NULL,version BIGINT NOT NULL,
 data_json LONGTEXT NOT NULL,approved_json LONGTEXT NULL,author_id VARCHAR(36) NOT NULL,updated_ms BIGINT NOT NULL,
 PRIMARY KEY(batch_id,sample_id,version),FOREIGN KEY(batch_id) REFERENCES batches(id),FOREIGN KEY(author_id) REFERENCES app_users(id)
) ENGINE=InnoDB;
CREATE TABLE IF NOT EXISTS reference_details (
 batch_id VARCHAR(36) PRIMARY KEY,details_json LONGTEXT NOT NULL,
 FOREIGN KEY(batch_id) REFERENCES batches(id)
) ENGINE=InnoDB;

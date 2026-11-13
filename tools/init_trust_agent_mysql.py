#!/usr/bin/env python3
"""Initialize MySQL tables for the trust-agent web API."""

from __future__ import annotations

import argparse
import os
from typing import Any


DDL = [
    """
    CREATE TABLE IF NOT EXISTS regions (
      id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
      region_id VARCHAR(64) NOT NULL,
      short_name VARCHAR(64) NULL,
      region_name VARCHAR(255) NULL,
      role_text VARCHAR(255) NULL,
      status_text VARCHAR(128) NULL,
      scene_guide TEXT NULL,
      total_rows INT NULL,
      display_rows INT NULL,
      changed_rows INT NULL,
      display_changed_rows INT NULL,
      review_rows INT NULL,
      full_review_rows INT NULL,
      avg_confidence DOUBLE NULL,
      counts_json LONGTEXT NULL,
      display_counts_json LONGTEXT NULL,
      metrics_json LONGTEXT NULL,
      region_json LONGTEXT NULL,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
      PRIMARY KEY (id),
      UNIQUE KEY uniq_regions_region_id (region_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS samples (
      id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
      sample_id VARCHAR(191) NOT NULL,
      region_id VARCHAR(64) NULL,
      sample_name VARCHAR(255) NULL,
      sample_stem VARCHAR(255) NULL,
      event_key VARCHAR(128) NULL,
      event_name VARCHAR(128) NULL,
      changed_px BIGINT NULL,
      changed_ratio DOUBLE NULL,
      object_count INT NULL,
      grid_position VARCHAR(64) NULL,
      review_status VARCHAR(64) NULL,
      review_required TINYINT(1) NULL,
      review_reason TEXT NULL,
      confidence DOUBLE NULL,
      mask_precision DOUBLE NULL,
      mask_recall DOUBLE NULL,
      mask_f1 DOUBLE NULL,
      mask_iou DOUBLE NULL,
      pred_area BIGINT NULL,
      main_bbox_json LONGTEXT NULL,
      quality_flags_json LONGTEXT NULL,
      report_text LONGTEXT NULL,
      display_policy VARCHAR(64) NULL,
      public_facts_json LONGTEXT NULL,
      private_metrics_json LONGTEXT NULL,
      evidence_hash CHAR(64) NULL,
      evidence_json LONGTEXT NULL,
      assets_json LONGTEXT NULL,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
      PRIMARY KEY (id),
      UNIQUE KEY uniq_samples_sample_id (sample_id),
      KEY idx_samples_region_event (region_id, event_key),
      KEY idx_samples_review (review_status),
      KEY idx_samples_evidence_hash (evidence_hash)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS image_assets (
      id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
      asset_id VARCHAR(191) NOT NULL,
      sample_id VARCHAR(191) NULL,
      region_id VARCHAR(64) NULL,
      asset_role VARCHAR(64) NOT NULL,
      storage_backend VARCHAR(32) NOT NULL DEFAULT 'local',
      uri TEXT NULL,
      fs_path TEXT NULL,
      mime_type VARCHAR(64) NULL,
      width INT NULL,
      height INT NULL,
      file_size BIGINT NULL,
      sha256 CHAR(64) NULL,
      visibility VARCHAR(32) NOT NULL DEFAULT 'public',
      include_in_llm TINYINT(1) NOT NULL DEFAULT 0,
      version INT NOT NULL DEFAULT 1,
      is_active TINYINT(1) NOT NULL DEFAULT 1,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
      PRIMARY KEY (id),
      UNIQUE KEY uniq_image_asset_id (asset_id),
      KEY idx_image_assets_sample_role (sample_id, asset_role, is_active),
      KEY idx_image_assets_region (region_id),
      KEY idx_image_assets_visibility (visibility, include_in_llm),
      KEY idx_image_assets_hash (sha256)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS model_runs (
      id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
      run_id VARCHAR(128) NOT NULL,
      region_id VARCHAR(64) NULL,
      model_name VARCHAR(128) NULL,
      checkpoint_path TEXT NULL,
      config_json LONGTEXT NULL,
      metrics_json LONGTEXT NULL,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
      PRIMARY KEY (id),
      UNIQUE KEY uniq_model_run (run_id),
      KEY idx_model_runs_region (region_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS prediction_results (
      id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
      prediction_id VARCHAR(191) NOT NULL,
      sample_id VARCHAR(191) NOT NULL,
      region_id VARCHAR(64) NULL,
      run_id VARCHAR(128) NULL,
      pred_mask_asset_id VARCHAR(191) NULL,
      pred_overlay_asset_id VARCHAR(191) NULL,
      changed_px BIGINT NULL,
      changed_ratio DOUBLE NULL,
      object_count INT NULL,
      main_bbox_json LONGTEXT NULL,
      grid_position VARCHAR(64) NULL,
      event_key VARCHAR(128) NULL,
      event_name VARCHAR(128) NULL,
      review_required TINYINT(1) NULL,
      review_reason TEXT NULL,
      confidence DOUBLE NULL,
      public_facts_json LONGTEXT NULL,
      private_metrics_json LONGTEXT NULL,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
      PRIMARY KEY (id),
      UNIQUE KEY uniq_prediction_result (prediction_id),
      KEY idx_prediction_sample (sample_id),
      KEY idx_prediction_region (region_id),
      KEY idx_prediction_run (run_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS sample_objects (
      id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
      sample_id VARCHAR(191) NOT NULL,
      region_id VARCHAR(64) NULL,
      object_index INT NOT NULL,
      object_key VARCHAR(128) NULL,
      area_px BIGINT NULL,
      bbox_json LONGTEXT NULL,
      center_json LONGTEXT NULL,
      grid_position VARCHAR(64) NULL,
      shape_text VARCHAR(128) NULL,
      object_json LONGTEXT NULL,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
      PRIMARY KEY (id),
      UNIQUE KEY uniq_sample_object (sample_id, object_index),
      KEY idx_sample_objects_region (region_id),
      KEY idx_sample_objects_area (area_px)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS ai_judgements (
      id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
      sample_id VARCHAR(191) NOT NULL,
      judgement_type VARCHAR(64) NOT NULL,
      evidence_hash CHAR(64) NOT NULL,
      prompt_version VARCHAR(128) NOT NULL,
      model_id VARCHAR(255) NULL,
      answer LONGTEXT NULL,
      engine VARCHAR(255) NULL,
      verify_status VARCHAR(64) NULL,
      verify_json LONGTEXT NULL,
      generated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
      PRIMARY KEY (id),
      UNIQUE KEY uniq_ai_judgement (sample_id, judgement_type, evidence_hash, prompt_version),
      KEY idx_ai_judgement_sample (sample_id),
      KEY idx_ai_judgement_verify (verify_status)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS manual_corrections (
      id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
      sample_id VARCHAR(191) NOT NULL,
      region_id VARCHAR(64) NULL,
      geometry_type VARCHAR(64) NOT NULL DEFAULT 'polygon',
      geometry_json LONGTEXT NULL,
      source_size_json LONGTEXT NULL,
      note TEXT NULL,
      corrected_event VARCHAR(128) NULL,
      evidence_hash_before CHAR(64) NULL,
      created_by VARCHAR(128) NULL,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY (id),
      KEY idx_manual_correction_sample (sample_id),
      KEY idx_manual_correction_region (region_id),
      KEY idx_manual_correction_created_at (created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS qa_sessions (
      id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
      session_key VARCHAR(191) NOT NULL,
      region_id VARCHAR(64) NULL,
      sample_id VARCHAR(191) NULL,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
      PRIMARY KEY (id),
      UNIQUE KEY uniq_qa_session_key (session_key),
      KEY idx_qa_session_region (region_id),
      KEY idx_qa_session_sample (sample_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS qa_messages (
      id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
      session_id BIGINT UNSIGNED NOT NULL,
      role VARCHAR(32) NOT NULL,
      message LONGTEXT NULL,
      engine VARCHAR(255) NULL,
      evidence_hash CHAR(64) NULL,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY (id),
      KEY idx_qa_message_session (session_id, id),
      KEY idx_qa_message_evidence_hash (evidence_hash)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS region_reports (
      id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
      region_id VARCHAR(64) NOT NULL,
      report_type VARCHAR(64) NOT NULL DEFAULT 'region',
      evidence_hash CHAR(64) NULL,
      model_id VARCHAR(255) NULL,
      answer LONGTEXT NULL,
      engine VARCHAR(255) NULL,
      generated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
      PRIMARY KEY (id),
      UNIQUE KEY uniq_region_report (region_id, report_type, evidence_hash),
      KEY idx_region_reports_region (region_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS auth_users (
      id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
      username VARCHAR(64) NOT NULL,
      display_name VARCHAR(128) NULL,
      password_salt CHAR(32) NOT NULL,
      password_hash CHAR(64) NOT NULL,
      is_active TINYINT(1) NOT NULL DEFAULT 1,
      last_login_at TIMESTAMP NULL DEFAULT NULL,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
      PRIMARY KEY (id),
      UNIQUE KEY uniq_auth_users_username (username)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    """
    CREATE TABLE IF NOT EXISTS auth_sessions (
      id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
      user_id BIGINT UNSIGNED NOT NULL,
      username VARCHAR(64) NOT NULL,
      session_token_hash CHAR(64) NOT NULL,
      expires_at TIMESTAMP NOT NULL,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
      PRIMARY KEY (id),
      UNIQUE KEY uniq_auth_sessions_token_hash (session_token_hash),
      KEY idx_auth_sessions_user_id (user_id),
      KEY idx_auth_sessions_expires_at (expires_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
]


EXTRA_SAMPLE_COLUMNS = [
    ("review_required", "TINYINT(1) NULL"),
    ("review_reason", "TEXT NULL"),
    ("confidence", "DOUBLE NULL"),
    ("mask_precision", "DOUBLE NULL"),
    ("mask_recall", "DOUBLE NULL"),
    ("mask_f1", "DOUBLE NULL"),
    ("mask_iou", "DOUBLE NULL"),
    ("pred_area", "BIGINT NULL"),
    ("main_bbox_json", "LONGTEXT NULL"),
    ("quality_flags_json", "LONGTEXT NULL"),
    ("report_text", "LONGTEXT NULL"),
    ("display_policy", "VARCHAR(64) NULL"),
    ("public_facts_json", "LONGTEXT NULL"),
    ("private_metrics_json", "LONGTEXT NULL"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create trust-agent MySQL database and tables.")
    parser.add_argument("--host", default=os.getenv("TRUST_AGENT_MYSQL_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("TRUST_AGENT_MYSQL_PORT", "3306")))
    parser.add_argument("--user", default=os.getenv("TRUST_AGENT_MYSQL_USER", "root"))
    parser.add_argument("--password", default=os.getenv("TRUST_AGENT_MYSQL_PASSWORD", ""))
    parser.add_argument("--database", default=os.getenv("TRUST_AGENT_MYSQL_DATABASE", "trust_change_agent"))
    parser.add_argument("--print_sql", action="store_true", help="Print DDL instead of executing it.")
    return parser.parse_args()


def connect(args: argparse.Namespace, database: str | None = None) -> Any:
    try:
        import pymysql  # type: ignore

        return pymysql.connect(
            host=args.host,
            port=args.port,
            user=args.user,
            password=args.password,
            database=database,
            charset="utf8mb4",
            autocommit=True,
        )
    except ImportError:
        try:
            import mysql.connector  # type: ignore

            return mysql.connector.connect(
                host=args.host,
                port=args.port,
                user=args.user,
                password=args.password,
                database=database,
                autocommit=True,
            )
        except ImportError as exc:
            raise RuntimeError("Install pymysql or mysql-connector-python first.") from exc


def ensure_extra_sample_columns(conn: Any, database: str) -> None:
    """Add optional sample columns on older MySQL versions without IF NOT EXISTS."""

    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT COLUMN_NAME
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
            """,
            (database, "samples"),
        )
        existing = {str(row[0]) for row in cur.fetchall()}
        for column_name, column_sql in EXTRA_SAMPLE_COLUMNS:
            if column_name not in existing:
                cur.execute(f"ALTER TABLE samples ADD COLUMN {column_name} {column_sql}")
    finally:
        cur.close()


def main() -> None:
    args = parse_args()
    create_db = (
        f"CREATE DATABASE IF NOT EXISTS `{args.database}` "
        "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    )

    if args.print_sql:
        print(create_db + ";")
        for sql in DDL:
            print(sql.strip() + ";")
        return

    conn = connect(args, None)
    try:
        cur = conn.cursor()
        cur.execute(create_db)
        cur.close()
    finally:
        conn.close()

    conn = connect(args, args.database)
    try:
        cur = conn.cursor()
        for sql in DDL:
            cur.execute(sql)
        cur.close()
        ensure_extra_sample_columns(conn, args.database)
    finally:
        conn.close()

    print(
        {
            "event": "mysql_schema_ready",
            "host": args.host,
            "port": args.port,
            "database": args.database,
            "tables": [
                "regions",
                "samples",
                "image_assets",
                "model_runs",
                "prediction_results",
                "sample_objects",
                "ai_judgements",
                "manual_corrections",
                "qa_sessions",
                "qa_messages",
                "region_reports",
                "auth_users",
                "auth_sessions",
            ],
        }
    )


if __name__ == "__main__":
    main()

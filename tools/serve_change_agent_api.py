#!/usr/bin/env python3
"""Local evidence-grounded QA API for the trust agent web app.

The web page is static, so this small HTTP service is optional. Without it the
front-end falls back to deterministic evidence rules. In the current demo mode,
--use_deepseek loads a local DeepSeek text model and grounds the answer in
structured change-detection facts.
"""

from __future__ import annotations

import argparse
import base64
import difflib
import hashlib
import io
import json
import math
import os
import re
import secrets
import subprocess
import time
import traceback
import threading
from datetime import datetime, timedelta
from urllib.parse import parse_qs, urlsplit
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from inspection_knowledge import KnowledgeIndex, route_question, data_question, public_citations, extractive_answer, valid_citations
from static_facts import static_store
from inspection_agent import explain as explain_with_evidence


MODEL = None
PROCESSOR = None
TOKENIZER = None
PROCESS_VISION_INFO = None
TORCH = None
MODEL_GENERATION_LOCK = threading.RLock()
ARGS: argparse.Namespace
WEB_ROOT: Path | None = None
DEPLOY_JOBS: dict[str, dict[str, Any]] = {}
DEFAULT_DEPLOY_PATHS = {
    "t1": "/media/dataset/GZCD/T1/P_GZ_test1_2013_1221_Level_18.tif",
    "t2": "/media/dataset/GZCD/T2/P_GZ_test1_2018_0119_Level_18.tif",
    "gt": "/media/dataset/GZCD/labels_change/P_GZ_test1_2013_2018.png",
}
DEFAULT_ADD_SAMPLE_MODEL = (
    "/media/CS备份/LCD/LCD-Net/LCD-Net/rctf-net/"
    "runs/gzcd_deploy_lcdnet_rcrm_scr3_cdir_gzcd_custom_mqarzzot_iter_100_lr_0.00025/best_model.pth"
)
DEFAULT_ADD_SAMPLE_PATHS = {
    "t1": "/media/dataset/levir/val/time1/test_2_x256_y0.png",
    "t2": "/media/dataset/levir/val/time2/test_2_x256_y0.png",
}
QA_CORPUS: dict[str, Any] = {"intents": []}
QA_EXACT: dict[str, dict[str, Any]] = {}
QA_ENTRIES: list[tuple[str, frozenset[str], dict[str, Any]]] = []
TRANSIENT_QA_MESSAGES: dict[str, list[dict[str, Any]]] = {}
TRANSIENT_SESSION_CONTEXTS: dict[str, dict[str, Any]] = {}
AUTH_SCHEMA_READY = False
AUTH_SCHEMA_ERROR = ""
AUTH_SESSION_DAYS = 7
DEFAULT_AUTH_USERNAME = "admin"
DEFAULT_AUTH_PASSWORD = ""  # Export copy: set runtime credentials explicitly.
DEFAULT_AUTH_DISPLAY_NAME = "系统管理员"
DEFAULT_AUTH_ADMIN_KEY = os.getenv("TRUST_AGENT_ADMIN_KEY", "admin-key-2026")
LOCAL_AUTH_SESSIONS: dict[str, dict[str, Any]] = {}
MYSQL_STATUS_CACHE: dict[str, Any] = {"ok": None, "checked_at": 0.0, "error": "", "transport": ""}
MYSQL_STATUS_TTL_SECONDS = 5.0
DEFAULT_MYSQL_SOCKET_CANDIDATES = (
    "/var/run/mysqld/mysqld.sock",
    "/run/mysqld/mysqld.sock",
    "/tmp/mysql.sock",
)
AUTH_SCHEMA_SQL = [
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
def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve evidence-grounded QA for trust_agent_web_app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument(
        "--web_data",
        default="runs/event_cls_unified_v2_aux_mg_siam_mbv2/trust_agent_web_app/app_data.json",
        help="Path to app_data.json. Used to resolve sample metadata and image paths.",
    )
    parser.add_argument("--use_qwen", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--use_deepseek", action="store_true", help="Load local DeepSeek text LLM for chat/report generation.")
    parser.add_argument("--model_id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--static_sql", action="store_true", help="Enable explicit read-only SQL demo using public app_data.json facts.")
    parser.add_argument("--no_auth", action="store_true", help="Skip login for a loopback-only static demo without a configured MySQL connection.")
    parser.add_argument("--knowledge_dir", default=str(Path(__file__).resolve().parents[1] / "knowledge"))
    parser.add_argument("--enable_agent_workflow", action="store_true", help="Use LangGraph task plans, clarification and local checkpoints.")
    parser.add_argument("--enable_model_sql", action="store_true", help="Enable model-authored read-only SQL over the public snapshot.")
    parser.add_argument("--dense_retrieval", action="store_true", help="Enable locally cached BGE Chinese dense retrieval and RRF.")
    parser.add_argument("--rerank", action="store_true", help="Enable locally cached BGE cross-encoder reranking.")
    parser.add_argument("--retrieval_cache", default=str(Path(__file__).resolve().parents[1] / ".model-cache"))
    parser.add_argument("--agent_checkpoint", default=str(Path(__file__).resolve().parents[1] / "logs/agent_checkpoints.sqlite"))
    parser.add_argument("--load_in_4bit", action="store_true", help="Load text LLM with bitsandbytes 4bit quantization.")
    parser.add_argument("--query_update_adapter", help="Optional local PEFT adapter, enabled only for typed query-state updates.")
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--max_pixels", type=int, default=512 * 512)
    parser.add_argument("--prompt_version", default="trust-agent-v9-deepseek-judge")
    parser.add_argument(
        "--qa_corpus",
        default="web/assistant_corpus.json",
        help="Generated assistant intent corpus used before evidence-rule/model fallback.",
    )
    parser.add_argument(
        "--enable_local_intent_corpus",
        action="store_true",
        default=env_flag("TRUST_AGENT_ENABLE_LOCAL_INTENT_CORPUS", False),
        help="Enable assistant_corpus local intent replies before model-backed QA.",
    )
    parser.add_argument(
        "--enable_rule_answer",
        action="store_true",
        default=env_flag("TRUST_AGENT_ENABLE_RULE_ANSWER", False),
        help="Enable local rule_answer() template fallback when model-backed QA is unavailable.",
    )
    parser.add_argument("--mysql_host", default=os.getenv("TRUST_AGENT_MYSQL_HOST"))
    parser.add_argument("--mysql_port", type=int, default=int(os.getenv("TRUST_AGENT_MYSQL_PORT", "3306")))
    parser.add_argument("--mysql_user", default=os.getenv("TRUST_AGENT_MYSQL_USER"))
    parser.add_argument("--mysql_password", default=os.getenv("TRUST_AGENT_MYSQL_PASSWORD", ""))
    parser.add_argument("--mysql_database", default=os.getenv("TRUST_AGENT_MYSQL_DATABASE", "trust_change_agent"))
    parser.add_argument("--mysql_socket", default=os.getenv("TRUST_AGENT_MYSQL_SOCKET", ""))
    parser.add_argument(
        "--repo_root",
        default=os.getenv("TRUST_AGENT_REPO_ROOT", ""),
        help="RCTF-Net repository root used by deployment training endpoints.",
    )
    parser.add_argument(
        "--training_python",
        default=os.getenv("TRUST_AGENT_TRAINING_PYTHON", "/home/halcyon/anaconda3/envs/amt/bin/python"),
        help="Python interpreter used to launch local WSL training jobs.",
    )
    parser.add_argument(
        "--deploy_crop_script",
        default=os.getenv("TRUST_AGENT_DEPLOY_CROP_SCRIPT", "/media/dataset/crop_deploy_dataset.py"),
        help="Crop script used by /api/deploy/crop.",
    )
    parser.add_argument(
        "--deploy_output_root",
        default=os.getenv("TRUST_AGENT_DEPLOY_OUTPUT_ROOT", "/media/dataset/deploy_uploads"),
        help="Output root used by /api/deploy/crop.",
    )
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(_json_safe(payload), ensure_ascii=False, allow_nan=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Auth-Token")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Cache-Control", "no-store, max-age=0")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def mysql_configured() -> bool:
    return bool(getattr(ARGS, "mysql_host", None) and getattr(ARGS, "mysql_user", None))


def mysql_socket_candidates() -> list[str]:
    candidates: list[str] = []
    explicit = str(getattr(ARGS, "mysql_socket", "") or "").strip()
    if explicit:
        candidates.append(explicit)
    for path_text in DEFAULT_MYSQL_SOCKET_CANDIDATES:
        if path_text not in candidates and os.path.exists(path_text):
            candidates.append(path_text)
    return candidates


def mysql_local_socket_allowed() -> bool:
    host = str(getattr(ARGS, "mysql_host", "") or "").strip().lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def mysql_enabled() -> bool:
    return bool(mysql_status().get("ok"))


def ensure_transient_session(session_key: str) -> None:
    if not session_key:
        return
    TRANSIENT_QA_MESSAGES.setdefault(session_key, [])
    TRANSIENT_SESSION_CONTEXTS.setdefault(session_key, {})


def transient_session_enabled(payload: dict[str, Any] | None = None, session_key: str = "") -> bool:
    if payload and payload.get("transientSession"):
        key = str(payload.get("sessionKey") or session_key or "")
        ensure_transient_session(key)
        return True
    return bool(session_key and session_key in TRANSIENT_QA_MESSAGES)


def local_intent_corpus_enabled() -> bool:
    args = globals().get("ARGS")
    return bool(getattr(args, "enable_local_intent_corpus", False))


def rule_answer_enabled() -> bool:
    args = globals().get("ARGS")
    return bool(getattr(args, "enable_rule_answer", False))


def mysql_connect():
    if not mysql_configured():
        return None
    last_exc: Exception | None = None
    transport_attempts: list[tuple[str, dict[str, Any]]] = []
    if mysql_local_socket_allowed():
        for socket_path in mysql_socket_candidates():
            transport_attempts.append((f"socket:{socket_path}", {"unix_socket": socket_path}))
    transport_attempts.append(("tcp", {"host": ARGS.mysql_host, "port": ARGS.mysql_port}))
    try:
        import pymysql  # type: ignore

        base_kwargs = {
            "user": ARGS.mysql_user,
            "password": ARGS.mysql_password,
            "database": ARGS.mysql_database,
            "charset": "utf8mb4",
            "autocommit": True,
            "cursorclass": pymysql.cursors.DictCursor,
            "connect_timeout": 1,
            "read_timeout": 3,
            "write_timeout": 3,
        }
        for transport, extra_kwargs in transport_attempts:
            try:
                conn = pymysql.connect(**base_kwargs, **extra_kwargs)
                setattr(conn, "_trust_transport", transport)
                return conn
            except Exception as exc:  # pragma: no cover - environment dependent
                last_exc = exc
    except ImportError:
        try:
            import mysql.connector  # type: ignore

            base_kwargs = {
                "user": ARGS.mysql_user,
                "password": ARGS.mysql_password,
                "database": ARGS.mysql_database,
                "autocommit": True,
                "connection_timeout": 1,
            }
            for transport, extra_kwargs in transport_attempts:
                try:
                    conn = mysql.connector.connect(**base_kwargs, **extra_kwargs)
                    setattr(conn, "_trust_transport", transport)
                    return conn
                except Exception as exc:  # pragma: no cover - environment dependent
                    last_exc = exc
        except ImportError as exc:
            raise RuntimeError("Install pymysql or mysql-connector-python to enable MySQL persistence.") from exc
    if last_exc is not None:
        raise last_exc
    return None


def mysql_status(force: bool = False) -> dict[str, Any]:
    now = time.time()
    cached_ok = MYSQL_STATUS_CACHE.get("ok")
    cached_at = float(MYSQL_STATUS_CACHE.get("checked_at") or 0.0)
    if not force and cached_ok is not None and (now - cached_at) < MYSQL_STATUS_TTL_SECONDS:
        return dict(MYSQL_STATUS_CACHE)
    if not mysql_configured():
        MYSQL_STATUS_CACHE.update({"ok": False, "checked_at": now, "error": "MySQL not configured", "transport": ""})
        return dict(MYSQL_STATUS_CACHE)
    conn = None
    try:
        conn = mysql_connect()
        if conn is None:
            MYSQL_STATUS_CACHE.update({"ok": False, "checked_at": now, "error": "mysql_connect returned None", "transport": ""})
        else:
            MYSQL_STATUS_CACHE.update(
                {
                    "ok": True,
                    "checked_at": now,
                    "error": "",
                    "transport": str(getattr(conn, "_trust_transport", "tcp")),
                }
            )
    except Exception as exc:
        MYSQL_STATUS_CACHE.update(
            {
                "ok": False,
                "checked_at": now,
                "error": f"{type(exc).__name__}: {exc}",
                "transport": "",
            }
        )
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    return dict(MYSQL_STATUS_CACHE)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_corpus_text(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", str(text or "").lower())


def char_bigrams(text: str) -> frozenset[str]:
    if len(text) < 2:
        return frozenset({text}) if text else frozenset()
    return frozenset(text[index : index + 2] for index in range(len(text) - 1))


def load_qa_corpus(path_text: str | None) -> None:
    global QA_CORPUS, QA_EXACT, QA_ENTRIES
    QA_CORPUS = {"intents": []}
    QA_EXACT = {}
    QA_ENTRIES = []
    if not path_text:
        return
    path = Path(path_text)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists() and WEB_ROOT:
        candidate = (WEB_ROOT / Path(path_text).name).resolve()
        if candidate.exists():
            path = candidate
    if not path.exists():
        print(f"qa_corpus=missing:{path}")
        return
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    for intent in data.get("intents") or []:
        for utterance in intent.get("utterances") or []:
            key = normalize_corpus_text(utterance)
            if not key:
                continue
            QA_EXACT.setdefault(key, intent)
            QA_ENTRIES.append((key, char_bigrams(key), intent))
    QA_CORPUS = data
    print(f"qa_corpus={path}")
    print(f"qa_corpus_intents={len(data.get('intents') or [])}")
    print(f"qa_corpus_utterances={len(QA_ENTRIES)}")


def corpus_match(question: str) -> tuple[dict[str, Any] | None, float]:
    if not local_intent_corpus_enabled():
        return None, 0.0
    key = normalize_corpus_text(question)
    if not key:
        return None, 0.0
    exact = QA_EXACT.get(key)
    if exact:
        return exact, 1.0
    # Very short follow-ups are intentionally exact-only to avoid mapping
    # ambiguous inputs such as "几个？" or "准吗？" to the wrong intent.
    if len(key) <= 4:
        return None, 0.0
    query_bigrams = char_bigrams(key)
    best_intent = None
    best_score = 0.0
    for candidate, candidate_bigrams, intent in QA_ENTRIES:
        overlap = len(query_bigrams & candidate_bigrams)
        if overlap == 0:
            continue
        dice = (2.0 * overlap) / max(1, len(query_bigrams) + len(candidate_bigrams))
        ratio = difflib.SequenceMatcher(None, key, candidate).ratio()
        score = 0.58 * dice + 0.42 * ratio
        if min(len(key), len(candidate)) >= 5 and (key in candidate or candidate in key):
            score = max(score, 0.88)
        anchors = [normalize_corpus_text(x) for x in intent.get("anchor_keywords") or []]
        if any(anchor and anchor in key for anchor in anchors):
            score += 0.05
        if score > best_score:
            best_score = score
            best_intent = intent
    if best_score < 0.78:
        return None, best_score
    return best_intent, min(best_score, 1.0)


def evidence_hash(payload: dict[str, Any]) -> str:
    sample = payload.get("sample") or {}
    compact = {
        "sample": {
            "id": sample.get("id"),
            "name": sample.get("name"),
            "pred": sample.get("pred"),
            "event": sample.get("event"),
            "eventType": sample.get("eventType"),
            "eventTypeZh": sample.get("eventTypeZh"),
            "morphologyType": sample.get("morphologyType"),
            "morphologyLabel": sample.get("morphologyLabel"),
            "changeLevel": sample.get("changeLevel"),
            "riskFlags": sample.get("riskFlags") or [],
            "changedPx": sample.get("changedPx"),
            "changedRatio": sample.get("changedRatio"),
            "objectCount": sample.get("objectCount"),
            "mainBbox": sample.get("mainBbox"),
            "gridPosition": sample.get("gridPosition"),
            "reviewStatus": sample.get("reviewStatus"),
            "reviewReason": sample.get("reviewReason"),
            "objects": sample.get("objects") or [],
            "qualityFlags": sample.get("qualityFlags") or [],
            "verifierChecks": sample.get("verifierChecks") or [],
        },
        "region": payload.get("region") or {},
    }
    return hashlib.sha256(canonical_json(compact).encode("utf-8")).hexdigest()


def db_execute(sql: str, params: tuple[Any, ...]) -> None:
    try:
        conn = mysql_connect()
    except Exception as exc:
        print(f"mysql_execute_connect_error={type(exc).__name__}: {exc}")
        return
    if conn is None:
        return
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        cur.close()
    except Exception as exc:
        print(f"mysql_execute_error={type(exc).__name__}: {exc}")
    finally:
        conn.close()


def db_fetch_one(sql: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
    try:
        conn = mysql_connect()
    except Exception as exc:
        print(f"mysql_fetch_connect_error={type(exc).__name__}: {exc}")
        return None
    if conn is None:
        return None
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        row = cur.fetchone()
        columns = [desc[0] for desc in cur.description or []]
        cur.close()
        if row is None:
            return None
        if isinstance(row, dict):
            return row
        return dict(zip(columns, row))
    except Exception as exc:
        print(f"mysql_fetch_error={type(exc).__name__}: {exc}")
        return None
    finally:
        conn.close()


def db_fetch_all(sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    try:
        conn = mysql_connect()
    except Exception as exc:
        print(f"mysql_fetch_all_connect_error={type(exc).__name__}: {exc}")
        return []
    if conn is None:
        return []
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description or []]
        cur.close()
        if not rows:
            return []
        if isinstance(rows[0], dict):
            return list(rows)
        return [dict(zip(columns, row)) for row in rows]
    except Exception as exc:
        print(f"mysql_fetch_all_error={type(exc).__name__}: {exc}")
        return []
    finally:
        conn.close()


def ensure_auth_tables() -> bool:
    global AUTH_SCHEMA_READY, AUTH_SCHEMA_ERROR
    if AUTH_SCHEMA_READY:
        try:
            ensure_default_auth_user()
        except Exception:
            pass
        return True
    if not mysql_enabled():
        AUTH_SCHEMA_ERROR = str(mysql_status().get("error") or "MySQL unavailable")
        return False
    try:
        conn = mysql_connect()
    except Exception as exc:
        AUTH_SCHEMA_ERROR = f"{type(exc).__name__}: {exc}"
        print(f"auth_schema_connect_error={type(exc).__name__}: {exc}")
        return False
    if conn is None:
        AUTH_SCHEMA_ERROR = "mysql_connect returned None"
        return False
    try:
        cur = conn.cursor()
        for sql in AUTH_SCHEMA_SQL:
            cur.execute(sql)
        cur.execute("SHOW COLUMNS FROM auth_users LIKE 'role_code'")
        role_column = cur.fetchone()
        if not role_column:
            cur.execute(
                """
                ALTER TABLE auth_users
                ADD COLUMN role_code VARCHAR(16) NOT NULL DEFAULT 'user' AFTER password_hash
                """
            )
        cur.close()
        ensure_default_auth_user()
        AUTH_SCHEMA_READY = True
        AUTH_SCHEMA_ERROR = ""
        return True
    except Exception as exc:
        AUTH_SCHEMA_ERROR = f"{type(exc).__name__}: {exc}"
        print(f"auth_schema_error={type(exc).__name__}: {exc}")
        return False
    finally:
        conn.close()


def auth_trim(value: Any, limit: int = 128) -> str:
    return str(value or "").strip()[:limit]


def auth_username(value: Any) -> str:
    return auth_trim(value, 64)


def auth_hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        200_000,
    ).hex()


def auth_token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def auth_role_code(value: Any) -> str:
    return "admin" if str(value or "").strip().lower() == "admin" else "user"


def auth_public_user(row: dict[str, Any]) -> dict[str, Any]:
    role_code = auth_role_code(row.get("role_code"))
    return {
        "userId": int(row.get("user_id") or row.get("id") or 0),
        "username": str(row.get("username") or ""),
        "displayName": str(row.get("display_name") or row.get("username") or ""),
        "roleCode": role_code,
        "isAdmin": role_code == "admin",
    }


def default_auth_user_row() -> dict[str, Any]:
    return {
        "id": 0,
        "user_id": 0,
        "username": DEFAULT_AUTH_USERNAME,
        "display_name": DEFAULT_AUTH_DISPLAY_NAME,
        "role_code": "admin",
    }


def ensure_default_auth_user() -> None:
    if not AUTH_SCHEMA_READY:
        return
    exists = db_fetch_one(
        "SELECT id, username, display_name, role_code FROM auth_users WHERE username = %s LIMIT 1",
        (DEFAULT_AUTH_USERNAME,),
    )
    if exists:
        updates: list[str] = []
        params: list[Any] = []
        if auth_role_code(exists.get("role_code")) != "admin":
            updates.append("role_code = %s")
            params.append("admin")
        if not str(exists.get("display_name") or "").strip():
            updates.append("display_name = %s")
            params.append(DEFAULT_AUTH_DISPLAY_NAME)
        if updates:
            db_execute(
                f"UPDATE auth_users SET {', '.join(updates)} WHERE id = %s",
                tuple(params + [int(exists.get("id") or 0)]),
            )
        return
    salt = secrets.token_hex(16)
    password_hash = auth_hash_password(DEFAULT_AUTH_PASSWORD, salt)
    conn = mysql_connect()
    if conn is None:
        raise RuntimeError("auth persistence unavailable")
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO auth_users (username, display_name, password_salt, password_hash, role_code)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (DEFAULT_AUTH_USERNAME, DEFAULT_AUTH_DISPLAY_NAME, salt, password_hash, "admin"),
        )
        cur.close()
    finally:
        conn.close()


def auth_create_local_session(user_row: dict[str, Any]) -> dict[str, Any]:
    token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(days=AUTH_SESSION_DAYS)
    LOCAL_AUTH_SESSIONS[token] = {
        "expiresAt": expires_at.strftime("%Y-%m-%d %H:%M:%S"),
        "user": auth_public_user(user_row),
    }
    return {
        "ok": True,
        "token": token,
        "expiresAt": expires_at.strftime("%Y-%m-%d %H:%M:%S"),
        "user": auth_public_user(user_row),
        "fallback": True,
    }


def auth_create_session(user_row: dict[str, Any]) -> dict[str, Any]:
    if not ensure_auth_tables():
        raise RuntimeError("auth persistence unavailable")
    token = secrets.token_urlsafe(32)
    token_hash = auth_token_hash(token)
    expires_at = datetime.utcnow() + timedelta(days=AUTH_SESSION_DAYS)
    conn = mysql_connect()
    if conn is None:
        raise RuntimeError("auth persistence unavailable")
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO auth_sessions (user_id, username, session_token_hash, expires_at)
            VALUES (%s, %s, %s, %s)
            """,
            (
                int(user_row.get("id") or user_row.get("user_id") or 0),
                str(user_row.get("username") or ""),
                token_hash,
                expires_at.strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        cur.execute(
            "UPDATE auth_users SET last_login_at = CURRENT_TIMESTAMP WHERE id = %s",
            (int(user_row.get("id") or user_row.get("user_id") or 0),),
        )
        cur.close()
    finally:
        conn.close()
    return {
        "ok": True,
        "token": token,
        "expiresAt": expires_at.strftime("%Y-%m-%d %H:%M:%S"),
        "user": auth_public_user(user_row),
    }


def auth_lookup_session(token: str) -> dict[str, Any] | None:
    token_text = str(token or "").strip()
    if not token_text:
        return None
    local_session = LOCAL_AUTH_SESSIONS.get(token_text)
    if local_session:
        expires_text = str(local_session.get("expiresAt") or "")
        try:
            if datetime.strptime(expires_text, "%Y-%m-%d %H:%M:%S") <= datetime.utcnow():
                LOCAL_AUTH_SESSIONS.pop(token_text, None)
                return None
        except Exception:
            LOCAL_AUTH_SESSIONS.pop(token_text, None)
            return None
        return {
            "sessionId": 0,
            "expiresAt": expires_text,
            "user": local_session.get("user") or auth_public_user(default_auth_user_row()),
        }
    if not ensure_auth_tables():
        return None
    row = db_fetch_one(
        """
        SELECT
          s.id AS session_id,
          s.user_id,
          s.username,
          s.expires_at,
          u.display_name,
          u.is_active,
          u.role_code
        FROM auth_sessions s
        JOIN auth_users u ON u.id = s.user_id
        WHERE s.session_token_hash = %s
          AND s.expires_at > CURRENT_TIMESTAMP
          AND u.is_active = 1
        LIMIT 1
        """,
        (auth_token_hash(token_text),),
    )
    if not row:
        return None
    return {
        "sessionId": int(row.get("session_id") or 0),
        "expiresAt": str(row.get("expires_at") or ""),
        "user": auth_public_user(row),
    }


def parse_auth_token(handler: BaseHTTPRequestHandler) -> str:
    bearer = str(handler.headers.get("Authorization") or "").strip()
    if bearer.lower().startswith("bearer "):
        return bearer[7:].strip()
    return str(handler.headers.get("X-Auth-Token") or "").strip()


def demo_no_auth_enabled() -> bool:
    args = globals().get("ARGS")
    return bool(args and getattr(args, "no_auth", False)
                and getattr(args, "host", "") in {"127.0.0.1", "localhost", "::1"}
                and getattr(args, "static_sql", False) and not mysql_configured())


def demo_anonymous_session() -> dict[str, Any]:
    return {"sessionId": 0, "authMode": "local_demo", "user": {
        "id": 0, "username": "local-demo", "displayName": "本地演示", "isAdmin": False}}


def require_auth(handler: BaseHTTPRequestHandler) -> dict[str, Any] | None:
    if demo_no_auth_enabled():
        return demo_anonymous_session()
    session = auth_lookup_session(parse_auth_token(handler))
    if session:
        return session
    json_response(
        handler,
        401,
        {
            "ok": False,
            "error": "auth required",
            "authRequired": True,
        },
    )
    return None


def require_admin(handler: BaseHTTPRequestHandler) -> dict[str, Any] | None:
    session = require_auth(handler)
    if session is None:
        return None
    user = session.get("user") or {}
    if bool(user.get("isAdmin")):
        return session
    json_response(
        handler,
        403,
        {
            "ok": False,
            "error": "admin required",
            "adminRequired": True,
        },
    )
    return None


def handle_auth_session(token: str) -> dict[str, Any]:
    if demo_no_auth_enabled():
        return {"ok": True, "loggedIn": True, "authRequired": False,
                "authMode": "local_demo", "user": demo_anonymous_session()["user"]}
    session = auth_lookup_session(token)
    if not session:
        return {"ok": True, "loggedIn": False}
    return {
        "ok": True,
        "loggedIn": True,
        "user": session.get("user") or {},
        "expiresAt": session.get("expiresAt") or "",
    }


def handle_auth_register(payload: dict[str, Any]) -> dict[str, Any]:
    if not mysql_enabled():
        return {"ok": False, "error": "MySQL unavailable"}
    if not ensure_auth_tables():
        return {"ok": False, "error": AUTH_SCHEMA_ERROR or "auth schema unavailable"}
    username = auth_username(payload.get("username"))
    display_name = auth_trim(payload.get("displayName") or payload.get("display_name") or username, 128)
    password = str(payload.get("password") or "")
    admin_key = auth_trim(payload.get("adminKey") or payload.get("admin_key"), 128)
    role_code = "user"
    if len(username) < 3:
        return {"ok": False, "error": "用户名至少需要 3 个字符"}
    if len(password) < 6:
        return {"ok": False, "error": "密码至少需要 6 个字符"}
    if admin_key:
        if admin_key != DEFAULT_AUTH_ADMIN_KEY:
            return {"ok": False, "error": "管理员 Key 不正确"}
        role_code = "admin"
    exists = db_fetch_one("SELECT id FROM auth_users WHERE username = %s LIMIT 1", (username,))
    if exists:
        return {"ok": False, "error": "该用户名已存在"}
    salt = secrets.token_hex(16)
    password_hash = auth_hash_password(password, salt)
    conn = mysql_connect()
    if conn is None:
        return {"ok": False, "error": "auth persistence unavailable"}
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO auth_users (username, display_name, password_salt, password_hash, role_code)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (username, display_name or username, salt, password_hash, role_code),
        )
        user_id = int(getattr(cur, "lastrowid", 0) or 0)
        cur.close()
    finally:
        conn.close()
    user_row = db_fetch_one(
        """
        SELECT id, username, display_name, role_code
        FROM auth_users
        WHERE id = %s
        LIMIT 1
        """,
        (user_id,),
    ) or {"id": user_id, "username": username, "display_name": display_name or username, "role_code": role_code}
    result = auth_create_session(user_row)
    result["registered"] = True
    return result


def handle_auth_login(payload: dict[str, Any]) -> dict[str, Any]:
    if not mysql_enabled():
        username = auth_username(payload.get("username"))
        password = str(payload.get("password") or "")
        if username == DEFAULT_AUTH_USERNAME and password == DEFAULT_AUTH_PASSWORD:
            return auth_create_local_session(default_auth_user_row())
        return {"ok": False, "error": "MySQL unavailable"}
    if not ensure_auth_tables():
        username = auth_username(payload.get("username"))
        password = str(payload.get("password") or "")
        if username == DEFAULT_AUTH_USERNAME and password == DEFAULT_AUTH_PASSWORD:
            return auth_create_local_session(default_auth_user_row())
        return {"ok": False, "error": AUTH_SCHEMA_ERROR or "auth schema unavailable"}
    username = auth_username(payload.get("username"))
    password = str(payload.get("password") or "")
    if not username or not password:
        return {"ok": False, "error": "请输入用户名和密码"}
    user_row = db_fetch_one(
        """
        SELECT id, username, display_name, password_salt, password_hash, is_active, role_code
        FROM auth_users
        WHERE username = %s
        LIMIT 1
        """,
        (username,),
    )
    if not user_row or not int(user_row.get("is_active") or 0):
        return {"ok": False, "error": "用户名或密码错误"}
    expected = str(user_row.get("password_hash") or "")
    actual = auth_hash_password(password, str(user_row.get("password_salt") or ""))
    if not secrets.compare_digest(actual, expected):
        return {"ok": False, "error": "用户名或密码错误"}
    return auth_create_session(user_row)


def handle_auth_logout(token: str) -> dict[str, Any]:
    token_text = str(token or "").strip()
    if token_text:
        LOCAL_AUTH_SESSIONS.pop(token_text, None)
    if token_text and ensure_auth_tables():
        db_execute(
            "DELETE FROM auth_sessions WHERE session_token_hash = %s",
            (auth_token_hash(token_text),),
        )
    return {"ok": True, "loggedOut": True}


def admin_user_stats() -> dict[str, int]:
    if not ensure_auth_tables():
        return {"total": 0, "active": 0, "disabled": 0, "admin": 0, "user": 0}
    row = db_fetch_one(
        """
        SELECT
          COUNT(*) AS total,
          SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END) AS active_total,
          SUM(CASE WHEN is_active = 0 THEN 1 ELSE 0 END) AS disabled_total,
          SUM(CASE WHEN role_code = 'admin' THEN 1 ELSE 0 END) AS admin_total,
          SUM(CASE WHEN role_code <> 'admin' OR role_code IS NULL THEN 1 ELSE 0 END) AS user_total
        FROM auth_users
        """,
        tuple(),
    ) or {}
    return {
        "total": int(row.get("total") or 0),
        "active": int(row.get("active_total") or 0),
        "disabled": int(row.get("disabled_total") or 0),
        "admin": int(row.get("admin_total") or 0),
        "user": int(row.get("user_total") or 0),
    }


def handle_admin_users(payload: dict[str, Any]) -> dict[str, Any]:
    if not mysql_enabled():
        return {"ok": False, "error": "MySQL unavailable"}
    if not ensure_auth_tables():
        return {"ok": False, "error": AUTH_SCHEMA_ERROR or "auth schema unavailable"}
    keyword = auth_trim(payload.get("keyword") or payload.get("search") or "", 128)
    status = auth_trim(payload.get("status") or "all", 16).lower()
    limit = admin_coerce_int(payload.get("limit"), default=20, minimum=10, maximum=200)
    offset = admin_coerce_int(payload.get("offset"), default=0, minimum=0, maximum=100000)
    where_parts: list[str] = []
    params: list[Any] = []
    if status == "active":
        where_parts.append("u.is_active = 1")
    elif status == "disabled":
        where_parts.append("u.is_active = 0")
    if keyword:
        where_parts.append("(u.username LIKE %s OR COALESCE(u.display_name, '') LIKE %s)")
        like = f"%{keyword}%"
        params.extend([like, like])
    where_sql = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
    count_row = db_fetch_one(f"SELECT COUNT(*) AS total FROM auth_users u{where_sql}", tuple(params))
    rows = db_fetch_all(
        f"""
        SELECT
          u.id,
          u.username,
          u.display_name,
          u.role_code,
          u.is_active,
          u.last_login_at,
          u.created_at,
          u.updated_at,
          (
            SELECT COUNT(*)
            FROM auth_sessions s
            WHERE s.user_id = u.id
              AND s.expires_at > CURRENT_TIMESTAMP
          ) AS active_sessions
        FROM auth_users u
        {where_sql}
        ORDER BY
          CASE WHEN u.username = %s THEN 0 ELSE 1 END,
          u.updated_at DESC,
          u.id DESC
        LIMIT %s OFFSET %s
        """,
        tuple(params + [DEFAULT_AUTH_USERNAME, limit, offset]),
    )
    return {
        "ok": True,
        "rows": admin_normalize_rows(rows),
        "total": int((count_row or {}).get("total") or 0),
        "offset": offset,
        "limit": limit,
        "keyword": keyword,
        "status": status,
        "stats": admin_user_stats(),
    }


def handle_admin_user_save(payload: dict[str, Any]) -> dict[str, Any]:
    if not mysql_enabled():
        return {"ok": False, "error": "MySQL unavailable"}
    if not ensure_auth_tables():
        return {"ok": False, "error": AUTH_SCHEMA_ERROR or "auth schema unavailable"}
    user_id = admin_coerce_int(payload.get("id"), default=0, minimum=0, maximum=10**12)
    username = auth_username(payload.get("username"))
    display_name = auth_trim(payload.get("displayName") or payload.get("display_name") or username, 128)
    password = str(payload.get("password") or "")
    role_code = auth_role_code("admin" if payload.get("isAdmin") else payload.get("roleCode") or payload.get("role_code"))
    is_active = 1 if bool(payload.get("isActive", True)) else 0
    if len(username) < 3:
        return {"ok": False, "error": "username must be at least 3 characters"}
    if not display_name:
        display_name = username
    if user_id <= 0 and len(password) < 6:
        return {"ok": False, "error": "password must be at least 6 characters"}

    existing = None
    if user_id > 0:
        existing = db_fetch_one(
            "SELECT id, username, display_name, is_active, role_code FROM auth_users WHERE id = %s LIMIT 1",
            (user_id,),
        )
        if not existing:
            return {"ok": False, "error": "user not found"}
        if str(existing.get("username") or "") == DEFAULT_AUTH_USERNAME:
            username = DEFAULT_AUTH_USERNAME
            is_active = 1
            role_code = "admin"

    duplicate = db_fetch_one(
        "SELECT id FROM auth_users WHERE username = %s AND id <> %s LIMIT 1",
        (username, user_id if user_id > 0 else 0),
    )
    if duplicate:
        return {"ok": False, "error": "username already exists"}

    conn = mysql_connect()
    if conn is None:
        return {"ok": False, "error": "auth persistence unavailable"}
    try:
        cur = conn.cursor()
        if user_id > 0:
            if password:
                if len(password) < 6:
                    return {"ok": False, "error": "password must be at least 6 characters"}
                salt = secrets.token_hex(16)
                password_hash = auth_hash_password(password, salt)
                cur.execute(
                    """
                    UPDATE auth_users
                    SET username = %s,
                        display_name = %s,
                        role_code = %s,
                        is_active = %s,
                        password_salt = %s,
                        password_hash = %s
                    WHERE id = %s
                    """,
                    (username, display_name, role_code, is_active, salt, password_hash, user_id),
                )
            else:
                cur.execute(
                    """
                    UPDATE auth_users
                    SET username = %s,
                        display_name = %s,
                        role_code = %s,
                        is_active = %s
                    WHERE id = %s
                    """,
                    (username, display_name, role_code, is_active, user_id),
                )
        else:
            salt = secrets.token_hex(16)
            password_hash = auth_hash_password(password, salt)
            cur.execute(
                """
                INSERT INTO auth_users (username, display_name, password_salt, password_hash, role_code, is_active)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (username, display_name, salt, password_hash, role_code, is_active),
            )
            user_id = int(getattr(cur, "lastrowid", 0) or 0)
        cur.close()
    finally:
        conn.close()

    row = db_fetch_one(
        """
        SELECT
          id,
          username,
          display_name,
          role_code,
          is_active,
          last_login_at,
          created_at,
          updated_at
        FROM auth_users
        WHERE id = %s
        LIMIT 1
        """,
        (user_id,),
    )
    return {
        "ok": True,
        "saved": True,
        "user": admin_normalize_row(row) or {},
        "stats": admin_user_stats(),
    }


def handle_admin_user_password(payload: dict[str, Any]) -> dict[str, Any]:
    if not mysql_enabled():
        return {"ok": False, "error": "MySQL unavailable"}
    if not ensure_auth_tables():
        return {"ok": False, "error": AUTH_SCHEMA_ERROR or "auth schema unavailable"}
    user_id = admin_coerce_int(payload.get("id"), default=0, minimum=1, maximum=10**12)
    password = str(payload.get("password") or "")
    if len(password) < 6:
        return {"ok": False, "error": "password must be at least 6 characters"}
    row = db_fetch_one("SELECT id, username FROM auth_users WHERE id = %s LIMIT 1", (user_id,))
    if not row:
        return {"ok": False, "error": "user not found"}
    salt = secrets.token_hex(16)
    password_hash = auth_hash_password(password, salt)
    db_execute(
        """
        UPDATE auth_users
        SET password_salt = %s,
            password_hash = %s
        WHERE id = %s
        """,
        (salt, password_hash, user_id),
    )
    return {
        "ok": True,
        "saved": True,
        "user": {
            "id": int(row.get("id") or 0),
            "username": str(row.get("username") or ""),
        },
    }


def db_number(value: Any, default: int | float = 0) -> int | float:
    if value is None:
        return default
    try:
        if isinstance(value, float):
            return value
        if isinstance(value, int):
            return value
        text = str(value)
        return float(text) if "." in text else int(text)
    except Exception:
        return default


LIVE_REGION_ORDER = ("levir", "whu", "clcd", "gzcd", "uav")
PRIVATE_LIVE_ASSET_KEYS = {"gt", "gtMask", "gtOverlay", "gt_mask", "gt_overlay"}
MANUAL_REVIEW_TAG_LABELS = {
    "hole": "空洞",
    "tiny_target": "极小目标",
    "small_area": "小面积",
    "fragmented": "边界破碎",
    "irregular": "边界不规则",
    "manual_ok": "人工确认正常",
}
MANUAL_MIN_COMPONENT_AREA_PX = 16
MANUAL_MIN_HOLE_AREA_PX = 16

ADMIN_REGION_ORDER_SQL = "CASE region_id " + " ".join(
    f"WHEN '{region_id}' THEN {index + 1}" for index, region_id in enumerate(LIVE_REGION_ORDER)
) + " ELSE 99 END, updated_at DESC"

ADMIN_TABLES: dict[str, dict[str, Any]] = {
    "regions": {
        "label": "区域",
        "from_sql": "regions",
        "key_field": "region_id",
        "region_field": "region_id",
        "order_by": ADMIN_REGION_ORDER_SQL,
        "columns": [
            ("region_id", "区域ID"),
            ("short_name", "简称"),
            ("region_name", "区域名称"),
            ("total_rows", "样本数"),
            ("changed_rows", "变化样本"),
            ("review_rows", "复核样本"),
            ("updated_at", "更新时间"),
        ],
        "selects": [
            "region_id",
            "short_name",
            "region_name",
            "total_rows",
            "changed_rows",
            "review_rows",
            "updated_at",
        ],
        "search_fields": ["region_id", "short_name", "region_name", "role_text", "status_text"],
        "detail_sql": "SELECT * FROM regions WHERE region_id=%s LIMIT 1",
    },
    "samples": {
        "label": "样本",
        "from_sql": "samples",
        "key_field": "sample_id",
        "region_field": "region_id",
        "order_by": "updated_at DESC, sample_id ASC",
        "columns": [
            ("sample_id", "样本ID"),
            ("region_id", "区域"),
            ("sample_name", "样本名"),
            ("event_name", "主事件"),
            ("changed_px", "变化像素"),
            ("object_count", "对象数"),
            ("review_status", "风险级别"),
            ("review_required", "进入复核"),
            ("updated_at", "更新时间"),
        ],
        "selects": [
            "sample_id",
            "region_id",
            "COALESCE(sample_name, sample_stem, sample_id) AS sample_name",
            "event_name",
            "changed_px",
            "object_count",
            "review_status",
            "review_required",
            "updated_at",
        ],
        "search_fields": ["sample_id", "sample_name", "sample_stem", "event_name", "review_reason"],
        "detail_sql": "SELECT * FROM samples WHERE sample_id=%s LIMIT 1",
        "sample_id_field": "sample_id",
    },
    "prediction_results": {
        "label": "预测结果",
        "from_sql": "prediction_results",
        "key_field": "prediction_id",
        "region_field": "region_id",
        "order_by": "updated_at DESC, prediction_id ASC",
        "columns": [
            ("prediction_id", "预测ID"),
            ("sample_id", "样本ID"),
            ("region_id", "区域"),
            ("event_name", "主事件"),
            ("changed_px", "变化像素"),
            ("object_count", "对象数"),
            ("review_required", "进入复核"),
            ("updated_at", "更新时间"),
        ],
        "selects": [
            "prediction_id",
            "sample_id",
            "region_id",
            "event_name",
            "changed_px",
            "object_count",
            "review_required",
            "updated_at",
        ],
        "search_fields": ["prediction_id", "sample_id", "event_name", "review_reason"],
        "detail_sql": "SELECT * FROM prediction_results WHERE prediction_id=%s LIMIT 1",
        "sample_id_field": "sample_id",
    },
    "image_assets": {
        "label": "图片资产",
        "from_sql": "image_assets",
        "key_field": "asset_id",
        "region_field": "region_id",
        "order_by": "updated_at DESC, asset_id ASC",
        "columns": [
            ("asset_id", "资产ID"),
            ("sample_id", "样本ID"),
            ("region_id", "区域"),
            ("asset_role", "角色"),
            ("uri", "URI"),
            ("width", "宽"),
            ("height", "高"),
            ("updated_at", "更新时间"),
        ],
        "selects": [
            "asset_id",
            "sample_id",
            "region_id",
            "asset_role",
            "uri",
            "width",
            "height",
            "updated_at",
        ],
        "search_fields": ["asset_id", "sample_id", "asset_role", "uri", "fs_path"],
        "detail_sql": "SELECT * FROM image_assets WHERE asset_id=%s LIMIT 1",
        "sample_id_field": "sample_id",
    },
    "sample_objects": {
        "label": "对象明细",
        "from_sql": "sample_objects",
        "key_field": "id",
        "region_field": "region_id",
        "order_by": "updated_at DESC, id DESC",
        "columns": [
            ("id", "记录ID"),
            ("sample_id", "样本ID"),
            ("region_id", "区域"),
            ("object_index", "对象序号"),
            ("area_px", "面积"),
            ("grid_position", "位置"),
            ("shape_text", "形状"),
            ("updated_at", "更新时间"),
        ],
        "selects": [
            "id",
            "sample_id",
            "region_id",
            "object_index",
            "area_px",
            "grid_position",
            "shape_text",
            "updated_at",
        ],
        "search_fields": ["sample_id", "object_key", "grid_position", "shape_text"],
        "detail_sql": "SELECT * FROM sample_objects WHERE id=%s LIMIT 1",
        "sample_id_field": "sample_id",
    },
    "manual_corrections": {
        "label": "人工修订",
        "from_sql": "manual_corrections",
        "key_field": "id",
        "region_field": "region_id",
        "order_by": "created_at DESC, id DESC",
        "columns": [
            ("id", "记录ID"),
            ("sample_id", "样本ID"),
            ("region_id", "区域"),
            ("geometry_type", "几何类型"),
            ("corrected_event", "修订事件"),
            ("created_by", "创建人"),
            ("created_at", "创建时间"),
        ],
        "selects": [
            "id",
            "sample_id",
            "region_id",
            "geometry_type",
            "corrected_event",
            "created_by",
            "created_at",
        ],
        "search_fields": ["sample_id", "region_id", "corrected_event", "note", "created_by"],
        "detail_sql": "SELECT * FROM manual_corrections WHERE id=%s LIMIT 1",
        "sample_id_field": "sample_id",
    },
    "ai_judgements": {
        "label": "AI判读",
        "from_sql": "ai_judgements",
        "key_field": "id",
        "region_field": None,
        "order_by": "generated_at DESC, id DESC",
        "columns": [
            ("id", "记录ID"),
            ("sample_id", "样本ID"),
            ("judgement_type", "判读类型"),
            ("engine", "引擎"),
            ("verify_status", "校验状态"),
            ("generated_at", "生成时间"),
        ],
        "selects": [
            "id",
            "sample_id",
            "judgement_type",
            "engine",
            "verify_status",
            "generated_at",
        ],
        "search_fields": ["sample_id", "judgement_type", "engine", "verify_status"],
        "detail_sql": "SELECT * FROM ai_judgements WHERE id=%s LIMIT 1",
        "sample_id_field": "sample_id",
    },
    "qa_sessions": {
        "label": "问答会话",
        "from_sql": "qa_sessions",
        "key_field": "id",
        "region_field": "region_id",
        "order_by": "updated_at DESC, id DESC",
        "columns": [
            ("id", "会话ID"),
            ("session_key", "会话键"),
            ("region_id", "区域"),
            ("sample_id", "样本ID"),
            ("updated_at", "更新时间"),
            ("created_at", "创建时间"),
        ],
        "selects": [
            "id",
            "session_key",
            "region_id",
            "sample_id",
            "updated_at",
            "created_at",
        ],
        "search_fields": ["session_key", "region_id", "sample_id"],
        "detail_sql": "SELECT * FROM qa_sessions WHERE id=%s LIMIT 1",
        "sample_id_field": "sample_id",
    },
    "qa_messages": {
        "label": "问答消息",
        "from_sql": "qa_messages m LEFT JOIN qa_sessions s ON s.id = m.session_id",
        "key_field": "message_id",
        "region_field": "s.region_id",
        "order_by": "m.id DESC",
        "columns": [
            ("message_id", "消息ID"),
            ("session_id", "会话ID"),
            ("session_key", "会话键"),
            ("region_id", "区域"),
            ("sample_id", "样本ID"),
            ("role", "角色"),
            ("engine", "引擎"),
            ("created_at", "创建时间"),
        ],
        "selects": [
            "m.id AS message_id",
            "m.session_id",
            "s.session_key",
            "s.region_id",
            "s.sample_id",
            "m.role",
            "m.engine",
            "m.created_at",
        ],
        "search_fields": ["s.session_key", "s.sample_id", "m.role", "m.engine", "m.message"],
        "detail_sql": """
            SELECT
              m.id AS message_id,
              m.session_id,
              s.session_key,
              s.region_id,
              s.sample_id,
              m.role,
              m.message,
              m.engine,
              m.evidence_hash,
              m.created_at
            FROM qa_messages m
            LEFT JOIN qa_sessions s ON s.id = m.session_id
            WHERE m.id=%s
            LIMIT 1
        """,
        "sample_id_field": "sample_id",
    },
}


def admin_escape_like(text: str) -> str:
    return str(text or "").replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def admin_coerce_int(value: Any, default: int = 0, minimum: int = 0, maximum: int | None = None) -> int:
    try:
        result = int(float(str(value)))
    except Exception:
        result = default
    if result < minimum:
        result = minimum
    if maximum is not None and result > maximum:
        result = maximum
    return result


def admin_maybe_expand_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    parsed = parse_json_maybe(text)
    return parsed if parsed is not None else value


def admin_normalize_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    out: dict[str, Any] = {}
    for key, value in row.items():
        out[str(key)] = _json_safe(admin_maybe_expand_json(value))
    return out


def admin_normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows or []:
        normalized = admin_normalize_row(row)
        if normalized is not None:
            out.append(normalized)
    return out


def admin_build_where(config: dict[str, Any], region_id: str, keyword: str) -> tuple[str, tuple[Any, ...]]:
    clauses: list[str] = []
    params: list[Any] = []
    region_field = str(config.get("region_field") or "").strip()
    if region_id and region_field:
        clauses.append(f"{region_field} = %s")
        params.append(region_id)
    search_fields = list(config.get("search_fields") or [])
    if keyword and search_fields:
        like_text = f"%{admin_escape_like(keyword[:80])}%"
        search_clauses: list[str] = []
        for field in search_fields:
            search_clauses.append(f"CAST({field} AS CHAR) LIKE %s")
            params.append(like_text)
        clauses.append("(" + " OR ".join(search_clauses) + ")")
    if not clauses:
        return "", tuple()
    return " WHERE " + " AND ".join(clauses), tuple(params)


def admin_table_options() -> list[dict[str, str]]:
    return [{"name": name, "label": str(config.get("label") or name)} for name, config in ADMIN_TABLES.items()]


def admin_region_rows() -> list[dict[str, Any]]:
    rows = db_fetch_all(
        f"""
        SELECT
          region_id,
          COALESCE(short_name, region_name, region_id) AS label,
          region_name,
          total_rows,
          changed_rows,
          review_rows,
          updated_at
        FROM regions
        ORDER BY {ADMIN_REGION_ORDER_SQL}
        """,
        tuple(),
    )
    return admin_normalize_rows(rows)


def admin_overview() -> dict[str, Any]:
    if not mysql_enabled():
        return {"ok": False, "error": "MySQL unavailable"}
    table_counts: dict[str, int] = {}
    for table_name in ADMIN_TABLES:
        count_from = "qa_messages" if table_name == "qa_messages" else table_name
        row = db_fetch_one(f"SELECT COUNT(*) AS total FROM {count_from}", tuple())
        table_counts[table_name] = admin_coerce_int((row or {}).get("total"), 0)
    recent_corrections = db_fetch_all(
        """
        SELECT id, sample_id, region_id, corrected_event, created_by, created_at
        FROM manual_corrections
        ORDER BY created_at DESC, id DESC
        LIMIT 8
        """,
        tuple(),
    )
    recent_sessions = db_fetch_all(
        """
        SELECT id, session_key, region_id, sample_id, updated_at
        FROM qa_sessions
        ORDER BY updated_at DESC, id DESC
        LIMIT 8
        """,
        tuple(),
    )
    return {
        "ok": True,
        "tables": admin_table_options(),
        "tableCounts": table_counts,
        "userStats": admin_user_stats(),
        "system": {
            "mysql": mysql_status(),
            "deepseekEnabled": bool(getattr(ARGS, "use_deepseek", False)),
            "authReady": ensure_auth_tables(),
        },
        "regions": admin_region_rows(),
        "recentCorrections": admin_normalize_rows(recent_corrections),
        "recentSessions": admin_normalize_rows(recent_sessions),
        "updatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def handle_admin_query(payload: dict[str, Any]) -> dict[str, Any]:
    if not mysql_enabled():
        return {"ok": False, "error": "MySQL unavailable"}
    table_name = str(payload.get("table") or "samples").strip().lower()
    config = ADMIN_TABLES.get(table_name)
    if not config:
        return {"ok": False, "error": f"unsupported table: {table_name}"}
    region_id = str(payload.get("regionId") or payload.get("region_id") or "").strip()
    keyword = str(payload.get("keyword") or payload.get("search") or "").strip()
    limit = admin_coerce_int(payload.get("limit"), default=50, minimum=10, maximum=200)
    offset = admin_coerce_int(payload.get("offset"), default=0, minimum=0, maximum=100000)
    where_sql, params = admin_build_where(config, region_id, keyword)
    count_row = db_fetch_one(f"SELECT COUNT(*) AS total FROM {config['from_sql']}{where_sql}", params)
    rows = db_fetch_all(
        f"SELECT {', '.join(config['selects'])} FROM {config['from_sql']}{where_sql} ORDER BY {config['order_by']} LIMIT %s OFFSET %s",
        params + (limit, offset),
    )
    return {
        "ok": True,
        "table": table_name,
        "label": str(config.get("label") or table_name),
        "keyField": str(config.get("key_field") or "id"),
        "columns": [{"key": key, "label": label} for key, label in config.get("columns") or []],
        "rows": admin_normalize_rows(rows),
        "total": admin_coerce_int((count_row or {}).get("total"), 0),
        "offset": offset,
        "limit": limit,
        "regionId": region_id,
        "keyword": keyword,
    }


def admin_region_bundle(region_id: str) -> dict[str, Any]:
    if not region_id:
        return {}
    summary = db_fetch_one("SELECT * FROM regions WHERE region_id=%s LIMIT 1", (region_id,))
    latest_samples = db_fetch_all(
        """
        SELECT sample_id, event_name, changed_px, object_count, review_status, review_required, updated_at
        FROM samples
        WHERE region_id=%s
        ORDER BY updated_at DESC, sample_id ASC
        LIMIT 12
        """,
        (region_id,),
    )
    review_samples = db_fetch_all(
        """
        SELECT sample_id, event_name, changed_px, review_status, review_reason, updated_at
        FROM samples
        WHERE region_id=%s AND review_required=1
        ORDER BY updated_at DESC, sample_id ASC
        LIMIT 12
        """,
        (region_id,),
    )
    corrections = db_fetch_all(
        """
        SELECT id, sample_id, corrected_event, created_by, created_at
        FROM manual_corrections
        WHERE region_id=%s
        ORDER BY created_at DESC, id DESC
        LIMIT 12
        """,
        (region_id,),
    )
    return {
        "summary": admin_normalize_row(summary) or {},
        "latestSamples": admin_normalize_rows(latest_samples),
        "reviewSamples": admin_normalize_rows(review_samples),
        "manualCorrections": admin_normalize_rows(corrections),
    }


def admin_sample_bundle(sample_id: str) -> dict[str, Any]:
    if not sample_id:
        return {}
    sample = db_fetch_one("SELECT * FROM samples WHERE sample_id=%s LIMIT 1", (sample_id,))
    predictions = db_fetch_all(
        "SELECT * FROM prediction_results WHERE sample_id=%s ORDER BY updated_at DESC, id DESC LIMIT 8",
        (sample_id,),
    )
    assets = db_fetch_all(
        """
        SELECT asset_id, asset_role, uri, fs_path, mime_type, width, height, file_size, is_active, updated_at
        FROM image_assets
        WHERE sample_id=%s
        ORDER BY is_active DESC, asset_role ASC, id DESC
        LIMIT 24
        """,
        (sample_id,),
    )
    objects = db_fetch_all(
        """
        SELECT object_index, object_key, area_px, bbox_json, center_json, grid_position, shape_text
        FROM sample_objects
        WHERE sample_id=%s
        ORDER BY object_index ASC
        LIMIT 64
        """,
        (sample_id,),
    )
    corrections = db_fetch_all(
        """
        SELECT id, geometry_type, corrected_event, note, created_by, created_at
        FROM manual_corrections
        WHERE sample_id=%s
        ORDER BY created_at DESC, id DESC
        LIMIT 16
        """,
        (sample_id,),
    )
    judgements = db_fetch_all(
        """
        SELECT id, judgement_type, engine, verify_status, generated_at
        FROM ai_judgements
        WHERE sample_id=%s
        ORDER BY generated_at DESC, id DESC
        LIMIT 16
        """,
        (sample_id,),
    )
    sessions = db_fetch_all(
        """
        SELECT id, session_key, region_id, updated_at, created_at
        FROM qa_sessions
        WHERE sample_id=%s
        ORDER BY updated_at DESC, id DESC
        LIMIT 12
        """,
        (sample_id,),
    )
    messages = db_fetch_all(
        """
        SELECT
          m.id AS message_id,
          m.session_id,
          m.role,
          m.engine,
          m.message,
          m.created_at
        FROM qa_messages m
        JOIN qa_sessions s ON s.id = m.session_id
        WHERE s.sample_id=%s
        ORDER BY m.id DESC
        LIMIT 20
        """,
        (sample_id,),
    )
    return {
        "sample": admin_normalize_row(sample) or {},
        "predictionResults": admin_normalize_rows(predictions),
        "imageAssets": admin_normalize_rows(assets),
        "sampleObjects": admin_normalize_rows(objects),
        "manualCorrections": admin_normalize_rows(corrections),
        "aiJudgements": admin_normalize_rows(judgements),
        "qaSessions": admin_normalize_rows(sessions),
        "qaMessages": admin_normalize_rows(messages),
    }


def handle_admin_detail(payload: dict[str, Any]) -> dict[str, Any]:
    if not mysql_enabled():
        return {"ok": False, "error": "MySQL unavailable"}
    table_name = str(payload.get("table") or "samples").strip().lower()
    config = ADMIN_TABLES.get(table_name)
    if not config:
        return {"ok": False, "error": f"unsupported table: {table_name}"}
    key_field = str(config.get("key_field") or "id")
    key_value = payload.get("key")
    if key_value is None:
        key_value = payload.get(key_field)
    key_text = str(key_value or "").strip()
    if not key_text:
        return {"ok": False, "error": f"missing key for {table_name}"}
    record = db_fetch_one(str(config.get("detail_sql") or ""), (key_text,))
    normalized_record = admin_normalize_row(record)
    if not normalized_record:
        return {"ok": False, "error": "record not found"}
    sample_id_field = str(config.get("sample_id_field") or "sample_id")
    region_id = str(normalized_record.get("region_id") or "")
    sample_id = str(normalized_record.get(sample_id_field) or normalized_record.get("sample_id") or "")
    bundle: dict[str, Any] = {}
    if sample_id:
        bundle = admin_sample_bundle(sample_id)
    elif table_name == "regions" and region_id:
        bundle = admin_region_bundle(region_id)
    elif region_id:
        bundle = {"region": admin_region_bundle(region_id)}
    return {
        "ok": True,
        "table": table_name,
        "label": str(config.get("label") or table_name),
        "record": normalized_record,
        "bundle": bundle,
    }


def parse_request_path(path_text: str) -> tuple[str, dict[str, list[str]]]:
    parsed = urlsplit(path_text or "")
    return parsed.path.rstrip("/"), parse_qs(parsed.query, keep_blank_values=True)


def live_public_assets(value: Any) -> dict[str, Any]:
    assets = parse_json_maybe(value) or {}
    if not isinstance(assets, dict):
        return {}
    out: dict[str, Any] = {}
    for key, asset_value in assets.items():
        name = str(key or "")
        if not name or name in PRIVATE_LIVE_ASSET_KEYS or not asset_value:
            continue
        out[name] = asset_value
    if out.get("predOverlay") and not out.get("overlay"):
        out["overlay"] = out["predOverlay"]
    return out


def live_scale_from_event(event_key: str, changed_px: int) -> str:
    key = str(event_key or "")
    if key in {"large_area_change", "large_building_hardening_change"}:
        return "large"
    if key in {"local_change", "small_building_facility_change", "land_cover_change"}:
        return "local"
    if key in {"weak_change", "no_change_or_weak_change"}:
        return "weak"
    return "none" if changed_px <= 0 else "local"


def live_objects(value: Any) -> list[dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "id": row.get("id") or f"obj_{idx + 1:03d}",
                "areaPx": int(db_number(row.get("area_px") or row.get("areaPx"), 0)),
                "bbox": parse_json_maybe(row.get("bbox")) or row.get("bbox") or [],
                "center": parse_json_maybe(row.get("center")) or row.get("center") or [],
                "grid": row.get("grid") or row.get("grid_position") or row.get("gridPosition") or "",
                "shape": row.get("shape") or "connected_component",
            }
        )
    return out


def live_item_from_sample_row(row: dict[str, Any], region: dict[str, Any]) -> dict[str, Any]:
    public_facts = parse_json_maybe(row.get("public_facts_json")) or parse_json_maybe(row.get("evidence_json")) or {}
    if not isinstance(public_facts, dict):
        public_facts = {}
    mask_quality = public_facts.get("mask_quality") if isinstance(public_facts.get("mask_quality"), dict) else {}
    changed_px = int(db_number(row.get("changed_px"), db_number(public_facts.get("changed_px"), 0)))
    changed_ratio = float(db_number(row.get("changed_ratio"), db_number(public_facts.get("changed_ratio"), 0.0)))
    object_count = int(db_number(row.get("object_count"), db_number(public_facts.get("object_count"), 0)))
    review_required = bool(row.get("review_required"))
    review_level = str(public_facts.get("review_level") or mask_quality.get("review_level") or "").lower()
    if review_level not in {"none", "low", "medium", "high"}:
        review_level = "medium" if review_required else ("low" if changed_px > 0 else "none")
    event_key = str(row.get("event_key") or public_facts.get("event_key") or "no_primary_event")
    event_name = str(row.get("event_name") or public_facts.get("event_name") or "未分类事件")
    assets = live_public_assets(row.get("assets_json"))
    main_bbox = (
        public_facts.get("main_bbox")
        or parse_json_maybe(row.get("main_bbox_json"))
        or mask_quality.get("pred_bbox")
        or []
    )
    quality_flags = public_facts.get("quality_flags")
    if not isinstance(quality_flags, list):
        quality_flags = parse_json_maybe(row.get("quality_flags_json")) or []
    objects = live_objects(public_facts.get("objects"))[:6]
    evidence_summary = str(public_facts.get("evidence_summary") or public_facts.get("evidenceSummary") or "").strip()
    if not evidence_summary:
        if changed_px > 0:
            evidence_summary = f"系统检测到 {object_count} 个变化对象，主要位置为{row.get('grid_position') or '未定位'}，变化面积约 {changed_px:,} px。"
        else:
            evidence_summary = "当前样本未形成稳定有效变化。"
    public_facts_compact = {
        "target_type": public_facts.get("target_type"),
        "manual_review_complete": bool(public_facts.get("manual_review_complete")),
    }
    return {
        "id": row.get("sample_id"),
        "name": row.get("sample_name") or row.get("sample_stem") or row.get("sample_id"),
        "stem": row.get("sample_stem") or row.get("sample_name") or row.get("sample_id"),
        "areaName": region.get("areaName") or "XX省",
        "sampleTime": "",
        "captureTime": "",
        "createdAt": "",
        "addedAt": "",
        "pred": event_key,
        "eventType": event_key,
        "eventTypeZh": event_name,
        "targetType": public_facts.get("target_type") or "",
        "morphologyType": event_key,
        "morphologyLabel": event_name,
        "primaryEventType": region.get("eventKey") or "",
        "primaryEventName": region.get("eventName") or "",
        "scale": str(public_facts.get("change_scale") or live_scale_from_event(event_key, changed_px)),
        "changedPx": changed_px,
        "changedRatio": changed_ratio,
        "objectCount": object_count,
        "mainBbox": main_bbox,
        "gridPosition": row.get("grid_position") or public_facts.get("grid_position") or "未定位",
        "confidence": float(db_number(row.get("confidence"), 0.0)),
        "reviewRequired": review_required,
        "reviewStatus": row.get("review_status") or review_level,
        "reviewReason": row.get("review_reason") or public_facts.get("review_reason") or "",
        "reviewLevel": review_level,
        "manualReviewComplete": bool(public_facts.get("manual_review_complete")),
        "reviewReasonCodes": public_facts.get("review_reason_codes") or mask_quality.get("review_reason_codes") or [],
        "qualityFlags": quality_flags,
        "objects": objects,
        "assets": assets,
        "evidenceSummary": evidence_summary,
        "report": row.get("report_text") or public_facts.get("report") or "",
        "evidencePack": public_facts_compact,
        "publicFacts": public_facts_compact,
        "maskQuality": mask_quality,
        "verifierChecks": [],
        "sourceSize": public_facts.get("source_size") or [256, 256],
        "road_changed": public_facts.get("road_changed"),
        "road_region_count": public_facts.get("road_region_count"),
        "road_regions": public_facts.get("road_regions"),
        "road_location": public_facts.get("road_location"),
        "road_reason": public_facts.get("road_reason"),
        "building_context": public_facts.get("building_context"),
    }


def live_region_rows(region_id: str | None = None) -> list[dict[str, Any]]:
    if not mysql_enabled():
        return []
    region_where = ""
    params: list[Any] = []
    if region_id:
        region_where = " WHERE region_id=%s"
        params.append(region_id)
    region_rows = db_fetch_all(
        f"""
        SELECT region_id, short_name, region_name, role_text, status_text, scene_guide,
               total_rows, display_rows, changed_rows, display_changed_rows,
               review_rows, full_review_rows, counts_json, metrics_json
        FROM regions
        {region_where}
        ORDER BY region_id
        """,
        tuple(params),
    )
    if not region_rows:
        return []
    by_region: dict[str, dict[str, Any]] = {}
    for row in region_rows:
        rid = str(row.get("region_id") or "")
        counts = parse_json_maybe(row.get("counts_json")) or {}
        metrics = parse_json_maybe(row.get("metrics_json")) or {}
        by_region[rid] = {
            "id": rid,
            "short": row.get("short_name") or rid.upper(),
            "name": row.get("region_name") or row.get("short_name") or rid,
            "role": row.get("role_text") or "",
            "status": row.get("status_text") or "",
            "sceneGuide": row.get("scene_guide") or row.get("role_text") or "",
            "areaName": "XX省",
            "totalRows": int(db_number(row.get("total_rows"), 0)),
            "displayRows": int(db_number(row.get("display_rows"), 0)),
            "changedRows": int(db_number(row.get("changed_rows"), 0)),
            "displayChangedRows": int(db_number(row.get("display_changed_rows"), 0)),
            "reviewRows": int(db_number(row.get("review_rows"), 0)),
            "fullReviewRows": int(db_number(row.get("full_review_rows"), 0)),
            "counts": counts if isinstance(counts, dict) else {},
            "metrics": metrics if isinstance(metrics, dict) else {},
            "items": [],
            "visualCount": int(db_number(row.get("display_rows"), db_number(row.get("total_rows"), 0))),
            "isDeployment": True,
            "updatedAt": "",
        }
    sample_rows = db_fetch_all(
        f"""
        SELECT sample_id, region_id, sample_name, sample_stem, event_key, event_name,
               changed_px, changed_ratio, object_count, grid_position, review_status,
               review_required, review_reason, confidence,
               main_bbox_json, quality_flags_json, report_text,
               public_facts_json, evidence_json, assets_json
        FROM samples
        {region_where}
        ORDER BY region_id, COALESCE(changed_px,0) DESC, sample_stem ASC
        """,
        tuple(params),
    )
    for row in sample_rows:
        rid = str(row.get("region_id") or "")
        region = by_region.get(rid)
        if not region:
            continue
        region["items"].append(live_item_from_sample_row(row, region))
    ordered_ids = [rid for rid in LIVE_REGION_ORDER if rid in by_region]
    ordered_ids.extend(rid for rid in sorted(by_region) if rid not in ordered_ids)
    return [by_region[rid] for rid in ordered_ids]


def sample_lookup_values(sample: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for key in ("id", "sampleId", "name", "stem"):
        value = str(sample.get(key) or "").strip()
        if value and value not in values:
            values.append(value)
    return tuple(values)


def database_scope_question(text: str) -> bool:
    text = str(text or "")
    return bool(
        re.search(
            r"(一共|总共|总计|共有|多少条|几条|多少张|几张|数据量|样本数|全量|全部|整体|统计|分布|"
            r"当前区域|这个区域|本区域|区域级|复核队列|优先复核|哪些样本|最大|最小|排名|top|count|total|all|samples|review queue)",
            text,
            flags=re.I,
        )
    )


def db_context_for_prompt(payload: dict[str, Any]) -> dict[str, Any]:
    if not mysql_enabled():
        return {}
    question = str(payload.get("question") or "")
    sample = payload.get("sample") or {}
    region = payload.get("region") or {}
    region_id = str(region.get("id") or sample.get("regionId") or "").strip()
    sample_keys = sample_lookup_values(sample)
    wants_region = database_scope_question(question) or not sample_keys
    context: dict[str, Any] = {
        "source": "mysql",
        "question_scope": "region_or_global" if wants_region else "sample",
    }
    try:
        if region_id:
            region_row = db_fetch_one(
                """
                SELECT region_id, short_name, region_name, role_text, status_text, scene_guide,
                       total_rows, display_rows, changed_rows, display_changed_rows,
                       review_rows, full_review_rows, avg_confidence, counts_json,
                       display_counts_json, metrics_json
                FROM regions WHERE region_id=%s LIMIT 1
                """,
                (region_id,),
            )
            if region_row:
                context["region"] = {
                    "region_id": region_row.get("region_id"),
                    "short": region_row.get("short_name"),
                    "name": region_row.get("region_name"),
                    "role": region_row.get("role_text"),
                    "status": region_row.get("status_text"),
                    "scene_guide": region_row.get("scene_guide"),
                    "total_rows": db_number(region_row.get("total_rows")),
                    "display_rows": db_number(region_row.get("display_rows")),
                    "changed_rows": db_number(region_row.get("changed_rows")),
                    "display_changed_rows": db_number(region_row.get("display_changed_rows")),
                    "review_rows": db_number(region_row.get("review_rows")),
                    "full_review_rows": db_number(region_row.get("full_review_rows")),
                    "avg_confidence": db_number(region_row.get("avg_confidence"), 0.0),
                    "counts": parse_json_maybe(region_row.get("counts_json")) or {},
                    "display_counts": parse_json_maybe(region_row.get("display_counts_json")) or {},
                    "metrics": parse_json_maybe(region_row.get("metrics_json")) or {},
                }
            stats = db_fetch_one(
                """
                SELECT COUNT(*) AS loaded_samples,
                       SUM(CASE WHEN COALESCE(changed_px,0) > 0 THEN 1 ELSE 0 END) AS loaded_changed_samples,
                       SUM(CASE WHEN COALESCE(review_required,0) = 1 THEN 1 ELSE 0 END) AS loaded_review_samples,
                       SUM(COALESCE(changed_px,0)) AS loaded_changed_px,
                       AVG(confidence) AS avg_confidence,
                       MAX(changed_px) AS max_changed_px
                FROM samples WHERE region_id=%s
                """,
                (region_id,),
            )
            if stats:
                context["loaded_region_stats"] = {key: db_number(value) for key, value in stats.items()}
            context["event_distribution"] = db_fetch_all(
                """
                SELECT event_key, event_name, COUNT(*) AS count_rows
                FROM samples WHERE region_id=%s
                GROUP BY event_key, event_name
                ORDER BY count_rows DESC
                LIMIT 12
                """,
                (region_id,),
            )
            context["review_queue_top"] = db_fetch_all(
                """
                SELECT sample_id, sample_stem, event_name, changed_px, object_count,
                       grid_position, review_reason
                FROM samples
                WHERE region_id=%s AND COALESCE(review_required,0)=1
                ORDER BY COALESCE(changed_px,0) DESC
                LIMIT 8
                """,
                (region_id,),
            )
            context["top_area_samples"] = db_fetch_all(
                """
                SELECT sample_id, sample_stem, event_name, changed_px, object_count,
                       grid_position, review_required
                FROM samples
                WHERE region_id=%s
                ORDER BY COALESCE(changed_px,0) DESC
                LIMIT 8
                """,
                (region_id,),
            )
            correction_count = db_fetch_one(
                "SELECT COUNT(*) AS count_rows FROM manual_corrections WHERE region_id=%s",
                (region_id,),
            )
            context["manual_corrections"] = {"count": db_number(correction_count.get("count_rows") if correction_count else 0)}

        if sample_keys:
            placeholders = ",".join(["%s"] * len(sample_keys))
            sample_row = db_fetch_one(
                f"""
                SELECT sample_id, region_id, sample_name, sample_stem, event_key, event_name,
                       changed_px, changed_ratio, object_count, grid_position,
                       review_status, review_required, review_reason, confidence,
                       pred_area,
                       main_bbox_json, quality_flags_json, report_text, display_policy,
                       evidence_hash, public_facts_json, evidence_json, assets_json
                FROM samples
                WHERE sample_id IN ({placeholders}) OR sample_name IN ({placeholders}) OR sample_stem IN ({placeholders})
                LIMIT 1
                """,
                sample_keys * 3,
            )
            if sample_row:
                context["sample_from_db"] = {
                    "sample_id": sample_row.get("sample_id"),
                    "region_id": sample_row.get("region_id"),
                    "name": sample_row.get("sample_name"),
                    "stem": sample_row.get("sample_stem"),
                    "event_key": sample_row.get("event_key"),
                    "event_name": sample_row.get("event_name"),
                    "changed_px": db_number(sample_row.get("changed_px")),
                    "changed_ratio": db_number(sample_row.get("changed_ratio"), 0.0),
                    "object_count": db_number(sample_row.get("object_count")),
                    "grid_position": sample_row.get("grid_position"),
                    "review_status": sample_row.get("review_status"),
                    "review_required": bool(sample_row.get("review_required")),
                    "review_reason": sample_row.get("review_reason"),
                    "confidence": db_number(sample_row.get("confidence"), 0.0),
                    "main_bbox": parse_json_maybe(sample_row.get("main_bbox_json")) or [],
                    "quality_flags": parse_json_maybe(sample_row.get("quality_flags_json")) or [],
                    "display_policy": sample_row.get("display_policy"),
                    "evidence_hash": sample_row.get("evidence_hash"),
                    "public_facts": parse_json_maybe(sample_row.get("public_facts_json"))
                    or parse_json_maybe(sample_row.get("evidence_json"))
                    or {},
                }
                context["sample_objects"] = db_fetch_all(
                    """
                    SELECT object_index, object_key, area_px, bbox_json, center_json,
                           grid_position, shape_text
                    FROM sample_objects WHERE sample_id=%s
                    ORDER BY object_index ASC LIMIT 12
                    """,
                    (str(sample_row.get("sample_id")),),
                )
                correction_rows = db_fetch_all(
                    """
                    SELECT geometry_type, note, corrected_event, created_at
                    FROM manual_corrections WHERE sample_id=%s
                    ORDER BY created_at DESC LIMIT 5
                    """,
                    (str(sample_row.get("sample_id")),),
                )
                context["sample_manual_corrections"] = correction_rows

        if not region_id or re.search(r"(全部数据集|所有区域|全局|all regions|global)", question, flags=re.I):
            context["global_regions"] = db_fetch_all(
                """
                SELECT r.region_id, r.short_name, r.total_rows, r.display_rows,
                       r.changed_rows, r.review_rows, r.full_review_rows,
                       COUNT(s.sample_id) AS loaded_samples,
                       SUM(CASE WHEN COALESCE(s.review_required,0)=1 THEN 1 ELSE 0 END) AS loaded_review_samples
                FROM regions
                LEFT JOIN samples s ON s.region_id=r.region_id
                GROUP BY r.region_id, r.short_name, r.total_rows, r.display_rows,
                         r.changed_rows, r.review_rows, r.full_review_rows
                ORDER BY r.region_id
                """,
                (),
            )
    except Exception as exc:
        context["mysql_error"] = f"{type(exc).__name__}: {exc}"
    return context


def db_upsert_sample(payload: dict[str, Any], ev_hash: str) -> None:
    sample = payload.get("sample") or {}
    region = payload.get("region") or {}
    sample_id = str(sample.get("id") or sample.get("name") or sample.get("stem") or "")
    if not sample_id or not mysql_enabled():
        return
    public_sample = compact_sample_for_prompt(sample) or {}
    public_assets = {
        key: value
        for key, value in (sample.get("assets") or {}).items()
        if key not in {"gtMask", "gtOverlay", "gt_mask", "gt_overlay"}
    }
    db_execute(
        """
        INSERT INTO samples
          (sample_id, region_id, sample_name, sample_stem, event_key, event_name,
           changed_px, changed_ratio, object_count, grid_position, review_status,
           evidence_hash, public_facts_json, evidence_json, assets_json)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
          region_id=VALUES(region_id),
          sample_name=VALUES(sample_name),
          sample_stem=VALUES(sample_stem),
          event_key=VALUES(event_key),
          event_name=VALUES(event_name),
          changed_px=VALUES(changed_px),
          changed_ratio=VALUES(changed_ratio),
          object_count=VALUES(object_count),
          grid_position=VALUES(grid_position),
          review_status=VALUES(review_status),
          evidence_hash=VALUES(evidence_hash),
          public_facts_json=VALUES(public_facts_json),
          evidence_json=VALUES(evidence_json),
          assets_json=VALUES(assets_json)
        """,
        (
            sample_id,
            str(region.get("id") or sample.get("regionId") or ""),
            sample.get("name"),
            sample.get("stem"),
            sample_event_key(sample),
            sample_event_name(sample),
            sample.get("changedPx"),
            sample.get("changedRatio"),
            sample.get("objectCount"),
            sample.get("gridPosition"),
            sample.get("reviewStatus"),
            ev_hash,
            canonical_json(public_sample),
            canonical_json(public_sample),
            canonical_json(public_assets),
        ),
    )


def db_get_judgement(sample_id: str, ev_hash: str, judgement_type: str) -> dict[str, Any] | None:
    if not sample_id or not mysql_enabled():
        return None
    return db_fetch_one(
        """
        SELECT answer, engine, verify_status, verify_json, generated_at, updated_at
        FROM ai_judgements
        WHERE sample_id=%s AND judgement_type=%s AND evidence_hash=%s AND prompt_version=%s
        LIMIT 1
        """,
        (sample_id, judgement_type, ev_hash, ARGS.prompt_version),
    )


def db_save_judgement(
    sample_id: str,
    judgement_type: str,
    ev_hash: str,
    answer: str,
    engine: str,
    verify_status: str,
    verify_json: dict[str, Any],
) -> None:
    if not sample_id or not mysql_enabled():
        return
    db_execute(
        """
        INSERT INTO ai_judgements
          (sample_id, judgement_type, evidence_hash, prompt_version, model_id,
           answer, engine, verify_status, verify_json)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
          model_id=VALUES(model_id),
          answer=VALUES(answer),
          engine=VALUES(engine),
          verify_status=VALUES(verify_status),
          verify_json=VALUES(verify_json)
        """,
        (
            sample_id,
            judgement_type,
            ev_hash,
            ARGS.prompt_version,
            ARGS.model_id if ARGS.use_qwen else None,
            answer,
            engine,
            verify_status,
            canonical_json(verify_json),
        ),
    )


def db_save_correction(payload: dict[str, Any], ev_hash: str) -> None:
    sample = payload.get("sample") or {}
    region = payload.get("region") or {}
    geometry = payload.get("geometry") or {}
    sample_id = str(sample.get("id") or sample.get("name") or sample.get("stem") or "")
    if not sample_id or not mysql_enabled():
        return
    manual_tags = [str(tag or "").strip() for tag in (payload.get("manualReviewTags") or []) if str(tag or "").strip()]
    manual_note = str(payload.get("manualReviewNote") or "").strip()
    note_parts = [str(payload.get("note") or "").strip()]
    if manual_tags:
        labels = [MANUAL_REVIEW_TAG_LABELS.get(tag, tag) for tag in manual_tags]
        note_parts.append("manual_tags=" + ",".join(labels))
    if manual_note:
        note_parts.append(manual_note)
    persisted_note = " | ".join(part for part in note_parts if part)
    db_upsert_sample(payload, ev_hash)
    db_execute(
        """
        INSERT INTO manual_corrections
          (sample_id, region_id, geometry_type, geometry_json, source_size_json,
           note, corrected_event, evidence_hash_before, created_by)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (
            sample_id,
            str(region.get("id") or sample.get("regionId") or ""),
            geometry.get("type") or "polygon",
            canonical_json(
                geometry.get("polygons")
                if isinstance(geometry, dict) and geometry.get("type") == "multi_polygon"
                else (geometry.get("points") if isinstance(geometry, dict) else geometry)
            ),
            canonical_json(payload.get("sourceSize") or sample.get("sourceSize") or []),
            persisted_note,
            sample_event_name(sample),
            ev_hash,
            payload.get("createdBy") or "reviewer",
        ),
    )


def db_save_review_override(payload: dict[str, Any], ev_hash: str) -> None:
    sample = payload.get("sample") or {}
    region = payload.get("region") or {}
    sample_id = str(payload.get("sampleId") or sample.get("id") or sample.get("name") or sample.get("stem") or "")
    region_id = str(region.get("id") or sample.get("regionId") or "").strip()
    if not sample_id or not mysql_enabled():
        return
    queued = bool(payload.get("queued"))
    reason = str(payload.get("reason") or ("人工加入复核队列，建议优先检查变化叠加图与范围连续性。" if queued else "")).strip()
    db_upsert_sample(payload, ev_hash)
    current = db_fetch_one(
        "SELECT public_facts_json FROM samples WHERE sample_id=%s LIMIT 1",
        (sample_id,),
    ) or {}
    public_facts = parse_json_maybe(current.get("public_facts_json")) or {}
    if not isinstance(public_facts, dict):
        public_facts = {}
    public_facts["review_required"] = queued
    public_facts["review_reason"] = reason
    if queued:
        public_facts["review_level"] = "medium" if str(public_facts.get("review_level") or "").lower() not in {"medium", "high"} else str(public_facts.get("review_level")).lower()
    db_execute(
        """
        UPDATE samples
        SET review_required=%s,
            review_reason=%s,
            review_status=%s,
            public_facts_json=%s,
            evidence_json=%s
        WHERE sample_id=%s
        """,
        (
            1 if queued else 0,
            reason,
            "manual_review" if queued else "normal",
            canonical_json(public_facts),
            canonical_json(public_facts),
            sample_id,
        ),
    )
    if region_id:
        review_count_row = db_fetch_one(
            "SELECT COUNT(*) AS count_rows FROM samples WHERE region_id=%s AND COALESCE(review_required,0)=1",
            (region_id,),
        ) or {}
        db_execute(
            "UPDATE regions SET review_rows=%s WHERE region_id=%s",
            (int(db_number(review_count_row.get("count_rows"), 0)), region_id),
        )


def resolve_asset_write_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    raw = str(path_text).strip()
    if not raw:
        return None
    parsed = urlsplit(raw)
    if parsed.scheme in {"http", "https", "file"} and parsed.path:
        raw = parsed.path
    raw = raw.split("?", 1)[0].split("#", 1)[0].strip()
    if not raw:
        return None
    path = Path(raw)
    if WEB_ROOT and raw.startswith("/deployed_assets/"):
        return (WEB_ROOT / raw.lstrip("/")).resolve()
    if path.is_absolute():
        return path
    if WEB_ROOT:
        return WEB_ROOT / path
    return None


def manual_review_tag_labels(tags: list[str]) -> list[str]:
    return [MANUAL_REVIEW_TAG_LABELS.get(tag, tag) for tag in tags if str(tag or "").strip()]


def manual_review_reason(tags: list[str], note: str, changed_px: int) -> str:
    labels = manual_review_tag_labels(tags)
    if "manual_ok" in tags:
        base = "人工复核完成：已确认当前预测范围可接受。"
    elif labels:
        base = f"人工复核完成：已按{'、'.join(labels[:3])}完成修正并回写。"
    elif changed_px > 0:
        base = "人工复核完成：已回写修正后的变化范围。"
    else:
        base = "人工复核完成：当前样本未保留有效变化范围。"
    note = str(note or "").strip()
    if note:
        base += f" 备注：{note[:120]}"
    return base


def manual_grid_position(bbox: list[int], width: int, height: int) -> str:
    if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return "未定位"
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    x_band = min(2, max(0, int(cx * 3 / max(width, 1))))
    y_band = min(2, max(0, int(cy * 3 / max(height, 1))))
    labels = {
        (0, 0): "左上",
        (1, 0): "上中",
        (2, 0): "右上",
        (0, 1): "左中",
        (1, 1): "中心",
        (2, 1): "右中",
        (0, 2): "左下",
        (1, 2): "下中",
        (2, 2): "右下",
    }
    return labels.get((x_band, y_band), "中心")


def manual_bbox_from_binary(binary: Any) -> list[int]:
    import numpy as np  # type: ignore

    ys, xs = np.where(binary)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def manual_hole_stats(binary: Any) -> tuple[int, int, float]:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    changed_px = int(binary.sum())
    background = (~binary).astype(np.uint8)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(background, connectivity=4)
    border_labels = set(np.unique(np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))).tolist())
    areas: list[int] = []
    for label in range(1, labels_count):
        if label in border_labels:
            continue
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= MANUAL_MIN_HOLE_AREA_PX:
            areas.append(area)
    hole_area_px = int(sum(areas))
    hole_ratio = float(hole_area_px / max(changed_px + hole_area_px, 1))
    return len(areas), hole_area_px, hole_ratio


def manual_objects_from_binary(binary: Any, width: int, height: int) -> tuple[list[dict[str, Any]], int, int, int, int, float]:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    objects: list[dict[str, Any]] = []
    raw_component_count = max(0, labels_count - 1)
    small_fragment_count = 0
    small_fragment_area_px = 0
    main_component_area_px = 0
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < MANUAL_MIN_COMPONENT_AREA_PX:
            small_fragment_count += 1
            small_fragment_area_px += area
            continue
        bbox = [x, y, x + w, y + h]
        main_component_area_px = max(main_component_area_px, area)
        objects.append(
            {
                "id": f"obj_{len(objects) + 1:03d}",
                "area_px": area,
                "bbox": bbox,
                "center": [int(round(x + w / 2.0)), int(round(y + h / 2.0))],
                "grid_position": manual_grid_position(bbox, width, height),
                "shape": "manual_component",
            }
        )
    objects.sort(key=lambda row: int(row.get("area_px") or 0), reverse=True)
    changed_px = int(binary.sum())
    main_component_ratio = float(main_component_area_px / changed_px) if changed_px > 0 else 0.0
    return objects, raw_component_count, small_fragment_count, small_fragment_area_px, main_component_area_px, main_component_ratio


def manual_mask_from_payload(payload: dict[str, Any]) -> tuple[Any, Any, int, int]:
    from PIL import Image, ImageDraw
    import numpy as np  # type: ignore

    sample = payload.get("sample") or {}
    geometry = payload.get("geometry") or {}
    source_size = payload.get("sourceSize") or sample.get("sourceSize") or [256, 256]
    width = max(1, exact_int(source_size[0] if len(source_size) > 0 else 256, 256))
    height = max(1, exact_int(source_size[1] if len(source_size) > 1 else 256, 256))
    polygons = geometry.get("polygons") if isinstance(geometry, dict) else None
    if not isinstance(polygons, list) or not polygons:
        points = geometry.get("points") if isinstance(geometry, dict) else None
        polygons = [points] if isinstance(points, list) and points else []
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    for poly in polygons:
        if not isinstance(poly, list) or len(poly) < 3:
            continue
        pts = []
        for point in poly:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            x = max(0, min(width, exact_int(point[0], 0)))
            y = max(0, min(height, exact_int(point[1], 0)))
            pts.append((x, y))
        if len(pts) >= 3:
            draw.polygon(pts, fill=255)
    binary = np.array(mask_img) > 0
    return mask_img, binary, width, height


def write_manual_overlay(post_path: Path | None, overlay_path: Path | None, mask_img: Any) -> None:
    if not post_path or not overlay_path or not post_path.exists():
        return
    from PIL import Image

    with Image.open(post_path).convert("RGB") as post_image:
        base = post_image.copy()
    mask = mask_img.resize(base.size, resample=Image.Resampling.NEAREST)
    tint = Image.new("RGB", base.size, (244, 182, 75))
    tinted = Image.blend(base, tint, 0.42)
    composed = Image.composite(tinted, base, mask)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    ext = overlay_path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        composed.save(overlay_path, quality=95)
    else:
        composed.save(overlay_path)


def manual_public_facts(
    payload: dict[str, Any],
    current_public_facts: dict[str, Any],
    binary: Any,
    width: int,
    height: int,
    manual_tags: list[str],
    manual_note: str,
) -> tuple[dict[str, Any], int, float, int, list[int], str, list[dict[str, Any]], list[dict[str, Any]], str]:
    sample = payload.get("sample") or {}
    changed_px = int(binary.sum())
    changed_ratio = float(changed_px / max(width * height, 1))
    main_bbox = manual_bbox_from_binary(binary)
    grid_position = manual_grid_position(main_bbox, width, height)
    objects, raw_component_count, small_fragment_count, small_fragment_area_px, main_component_area_px, main_component_ratio = manual_objects_from_binary(binary, width, height)
    hole_count, hole_area_px, hole_ratio = manual_hole_stats(binary)
    bbox_area = max(0, main_bbox[2] - main_bbox[0]) * max(0, main_bbox[3] - main_bbox[1])
    bbox_fill_ratio = float(changed_px / bbox_area) if bbox_area > 0 else 0.0
    small_fragment_area_ratio = float(small_fragment_area_px / changed_px) if changed_px > 0 else 0.0
    review_level = "none" if changed_px <= 0 else "low"
    review_reason_codes: list[str] = []
    review_reason = manual_review_reason(manual_tags, manual_note, changed_px)
    event_key = str(current_public_facts.get("event_key") or sample.get("pred") or "local_change")
    event_name = str(current_public_facts.get("event_name") or sample.get("eventTypeZh") or "人工修正变化")
    if changed_px <= 0:
        event_key, event_name = "no_obvious_change", "无明显变化"
    evidence_summary = (
        f"系统检测到 {len(objects)} 个变化对象，主要位置为{grid_position}，变化面积约 {changed_px} px。"
        if changed_px > 0
        else "当前样本未形成稳定有效变化范围。"
    )
    quality_flags = [
        {
            "code": "manual_review_complete",
            "name": "人工复核完成",
            "text": review_reason,
            "level": "info",
        }
    ]
    mask_quality = {
        "schema_version": "mask_quality_facts_v1",
        "metric_source": "manual_polygon_writeback",
        "pred_changed": bool(changed_px > 0),
        "pred_changed_px": changed_px,
        "pred_changed_ratio": round(changed_ratio, 6),
        "pred_bbox": main_bbox,
        "component_count": len(objects),
        "raw_component_count": raw_component_count,
        "main_component_area_px": main_component_area_px,
        "main_component_ratio": round(main_component_ratio, 6),
        "small_fragment_count": small_fragment_count,
        "small_fragment_area_px": small_fragment_area_px,
        "small_fragment_area_ratio": round(small_fragment_area_ratio, 6),
        "hole_area_px": hole_area_px,
        "hole_count": hole_count,
        "hole_ratio": round(hole_ratio, 6),
        "bbox_fill_ratio": round(bbox_fill_ratio, 6),
        "border_contact": bool(changed_px > 0 and (main_bbox[0] <= 0 or main_bbox[1] <= 0 or main_bbox[2] >= width or main_bbox[3] >= height)),
        "area_too_large": False,
        "area_too_small": bool(0 < changed_ratio < 0.001),
        "review_level": review_level,
        "review_reason_codes": review_reason_codes,
        "manual_review_complete": True,
        "manual_review_tags": manual_tags,
    }
    public_facts = dict(current_public_facts)
    public_facts.update(
        {
            "schema_version": current_public_facts.get("schema_version") or "trust_agent_public_facts_v2",
            "sample_id": sample.get("id") or sample.get("sampleId") or sample.get("stem") or sample.get("name"),
            "sample_name": sample.get("name") or sample.get("stem") or sample.get("id"),
            "region_id": sample.get("regionId") or payload.get("region", {}).get("id"),
            "target_type": current_public_facts.get("target_type") or "变化检测",
            "change_detected": bool(changed_px > 0),
            "changed_px": changed_px,
            "changed_ratio": round(changed_ratio, 6),
            "object_count": len(objects),
            "main_bbox": main_bbox,
            "grid_position": grid_position,
            "event_key": event_key,
            "event_name": event_name,
            "objects": objects[:32],
            "mask_quality": mask_quality,
            "quality_flags": quality_flags,
            "review_required": False,
            "review_level": review_level,
            "review_reason_codes": review_reason_codes,
            "review_reason": review_reason,
            "review_status": "manual_verified",
            "manual_review_tags": manual_tags,
            "manual_review_note": manual_note,
            "source_size": [width, height],
            "evidence_summary": evidence_summary,
            "report": review_reason,
        }
    )
    return public_facts, changed_px, changed_ratio, len(objects), main_bbox, grid_position, objects, quality_flags, review_reason


def db_complete_review(payload: dict[str, Any], ev_hash: str) -> None:
    sample = payload.get("sample") or {}
    region = payload.get("region") or {}
    sample_id = str(payload.get("sampleId") or sample.get("id") or sample.get("name") or sample.get("stem") or "")
    region_id = str(region.get("id") or sample.get("regionId") or "").strip()
    if not sample_id or not mysql_enabled():
        return
    db_upsert_sample(payload, ev_hash)
    db_save_correction(payload, ev_hash)

    manual_tags = [str(tag or "").strip() for tag in (payload.get("manualReviewTags") or []) if str(tag or "").strip()]
    manual_note = str(payload.get("manualReviewNote") or "").strip()
    assets = sample.get("assets") or {}
    mask_img, binary, width, height = manual_mask_from_payload(payload)
    mask_paths: list[Path] = []
    for asset_key in ("mask", "predMask", "pred_mask"):
        mask_path = resolve_asset_write_path(assets.get(asset_key))
        if mask_path and mask_path not in mask_paths:
            mask_paths.append(mask_path)
    for mask_path in mask_paths:
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask_img.save(mask_path)
    overlay_paths: list[Path] = []
    for asset_key in ("overlay", "predOverlay", "pred_overlay"):
        overlay_path = resolve_asset_write_path(assets.get(asset_key))
        if overlay_path and overlay_path not in overlay_paths:
            overlay_paths.append(overlay_path)
    post_path = resolve_asset(assets.get("post"))
    for overlay_path in overlay_paths:
        write_manual_overlay(post_path, overlay_path, mask_img)

    current = db_fetch_one(
        "SELECT public_facts_json FROM samples WHERE sample_id=%s LIMIT 1",
        (sample_id,),
    ) or {}
    current_public_facts = parse_json_maybe(current.get("public_facts_json")) or {}
    if not isinstance(current_public_facts, dict):
        current_public_facts = {}
    public_facts, changed_px, changed_ratio, object_count, main_bbox, grid_position, objects, quality_flags, review_reason = manual_public_facts(
        payload,
        current_public_facts,
        binary,
        width,
        height,
        manual_tags,
        manual_note,
    )
    db_execute(
        """
        UPDATE samples
        SET changed_px=%s,
            changed_ratio=%s,
            object_count=%s,
            grid_position=%s,
            review_status=%s,
            review_required=0,
            review_reason=%s,
            main_bbox_json=%s,
            quality_flags_json=%s,
            report_text=%s,
            public_facts_json=%s,
            evidence_json=%s,
            assets_json=%s
        WHERE sample_id=%s
        """,
        (
            changed_px,
            changed_ratio,
            object_count,
            grid_position,
            "manual_verified",
            review_reason,
            canonical_json(main_bbox),
            canonical_json(quality_flags),
            review_reason,
            canonical_json(public_facts),
            canonical_json(public_facts),
            canonical_json(assets),
            sample_id,
        ),
    )
    if region_id:
        review_count_row = db_fetch_one(
            "SELECT COUNT(*) AS count_rows FROM samples WHERE region_id=%s AND COALESCE(review_required,0)=1",
            (region_id,),
        ) or {}
        db_execute(
            "UPDATE regions SET review_rows=%s WHERE region_id=%s",
            (int(db_number(review_count_row.get("count_rows"), 0)), region_id),
        )


def db_session_key(payload: dict[str, Any]) -> str:
    region = payload.get("region") or {}
    sample = payload.get("sample") or {}
    region_id = str(region.get("id") or region.get("short") or "region")
    sample_id = str(sample.get("id") or sample.get("name") or sample.get("stem") or "region")
    session_key = str(payload.get("sessionKey") or f"{region_id}:{sample_id}")
    if payload.get("transientSession"):
        ensure_transient_session(session_key)
    return session_key


def db_save_qa_exchange(payload: dict[str, Any], answer: str, engine: str, ev_hash: str | None = None) -> None:
    session_key = db_session_key(payload)
    if transient_session_enabled(payload, session_key):
        messages = TRANSIENT_QA_MESSAGES.setdefault(session_key, [])
        question = str(payload.get("question") or "")
        timestamp = datetime.utcnow().isoformat(timespec="seconds")
        messages.append({"role": "user", "message": question, "engine": None, "created_at": timestamp})
        messages.append({"role": "assistant", "message": answer, "engine": engine, "created_at": timestamp})
        if len(messages) > 40:
            del messages[:-40]
        return
    if not mysql_enabled():
        return
    region = payload.get("region") or {}
    sample = payload.get("sample") or {}
    sample_id = str(sample.get("id") or sample.get("name") or sample.get("stem") or "")
    region_id = str(region.get("id") or region.get("short") or "")
    if sample_id:
        db_upsert_sample(payload, ev_hash or evidence_hash(payload))
    db_execute(
        """
        INSERT INTO qa_sessions (session_key, region_id, sample_id)
        VALUES (%s,%s,%s)
        ON DUPLICATE KEY UPDATE
          region_id=VALUES(region_id),
          sample_id=VALUES(sample_id),
          updated_at=CURRENT_TIMESTAMP
        """,
        (session_key, region_id or None, sample_id or None),
    )
    row = db_fetch_one("SELECT id FROM qa_sessions WHERE session_key=%s LIMIT 1", (session_key,))
    if not row:
        return
    session_id = row.get("id")
    question = str(payload.get("question") or "")
    db_execute(
        "INSERT INTO qa_messages (session_id, role, message, engine, evidence_hash) VALUES (%s,%s,%s,%s,%s)",
        (session_id, "user", question, None, ev_hash),
    )
    db_execute(
        "INSERT INTO qa_messages (session_id, role, message, engine, evidence_hash) VALUES (%s,%s,%s,%s,%s)",
        (session_id, "assistant", answer, engine, ev_hash),
    )


def fmt(n: Any) -> str:
    try:
        return f"{float(n):,.0f}".replace(",", ",")
    except Exception:
        return str(n)


def class_name(sample: dict[str, Any] | None) -> str:
    if not sample:
        return "未选择样本"
    value = str(sample.get("event") or sample.get("pred") or "未知事件")
    return {
        "no_change_or_weak_change": "无明显变化/弱变化",
        "regional_significant_change": "成片/区域显著变化",
        "local_change": "局部变化",
        "complex_or_uncertain_change": "复杂/不确定变化",
    }.get(value, value)


def has_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(key in text for key in keywords)


def is_small_talk(text: str) -> bool:
    q = text.strip().lower().strip(" !！.。?")
    return q in {"你好", "您好", "hello", "hi", "hey", "在吗", "你是谁", "介绍一下", "谢谢", "thanks", "thank you"}


def small_talk_answer(payload: dict[str, Any]) -> str:
    region = payload.get("region") or {}
    sample = payload.get("sample") or {}
    current = f"当前选中样本是 {sample.get('stem')}。" if sample else "你也可以先在左侧选择一个样本。"
    return (
        f"你好，我是 {region.get('short', '当前区域')} 的变化判读助手。"
        "你可以问我三类问题：区域里哪些样本优先复核、当前样本发生了什么变化、如何生成巡检报告。"
        + current
    )


def is_small_talk(text: str) -> bool:
    q = text.strip().lower()
    q = re.sub(r"[\s!！?？。,.，、~～]+", "", q)
    if not q:
        return False
    exact = {
        "你好",
        "您好",
        "hello",
        "hi",
        "hey",
        "在吗",
        "你在吗",
        "谢谢",
        "多谢",
        "thanks",
        "thankyou",
        "你好吗",
        "你还好吗",
        "最近怎么样",
        "你在干嘛",
        "你正在干嘛",
        "你是谁",
        "介绍一下",
        "自我介绍",
        "你是什么模型",
        "你用的什么模型",
        "你能做什么",
    }
    if q in exact:
        return True
    casual_patterns = (
        r"^(你)?好呀+$",
        r"^你.*(好吗|怎么样)$",
        r"^你.*(是谁|什么模型|哪个模型|能做什么)$",
        r"^你.*(在干嘛|正在做什么|忙什么)$",
    )
    return any(re.search(pattern, q) for pattern in casual_patterns)


def is_weather_question(text: str) -> bool:
    lower = text.lower()
    return has_any(text, ("天气", "温度", "下雨", "气象")) or has_any(
        lower, ("weather", "temperature", "rain")
    )


def is_project_question(text: str) -> bool:
    q = text.lower()
    evidence_anchor = r"(当前|样本|这张|这幅|本图|区域|复核|对象|面积|位置|范围|变了|有几个|多少|哪里|哪儿|报告|漏检|过检|检测结果|变化对象|变化位置|变化面积)"
    if re.search(r"(写|创作|编|生成).{0,12}(诗|故事|作文|文案|代码|邮件|短文)", text) and not re.search(evidence_anchor, text):
        return False
    if re.search(r"(是什么|介绍|解释|原理|区别|综述|论文)", text) and not re.search(evidence_anchor, text):
        return False
    if re.search(r"(变了|变动|变化).{0,16}(多少|几个|哪些|什么|哪里|哪儿|位置|对象|东西|面积|范围|区域|目标)", text):
        return True
    if re.search(r"(多少|几个|哪些).{0,10}(变化|对象|东西|区域|目标|样本)", text):
        return True
    project_keywords = (
        "变化",
        "样本",
        "当前区域",
        "区域",
        "图像",
        "前时相",
        "后时相",
        "判读",
        "复核",
        "误差",
        "对象",
        "数量",
        "面积",
        "位置",
        "范围",
        "报告",
        "结论",
        "漏检",
        "过检",
        "可信",
        "证据",
        "mask",
        "overlay",
        "bbox",
        "f1",
        "iou",
        "levir",
        "whu",
        "sysu",
        "clcd",
        "uav",
        "低空",
        "施工",
        "设施",
        "地表",
        "覆盖",
        "道路",
        "线性",
        "遥感",
        "变化检测",
    )
    return any(key in q for key in project_keywords)


def small_talk_answer(payload: dict[str, Any]) -> str:
    question = str(payload.get("question") or "")
    region = payload.get("region") or {}
    sample = payload.get("sample") or {}
    current = f"当前选中样本是 {sample.get('stem')}。" if sample else "你也可以先在左侧选择一个样本。"
    if re.search(r"什么模型|哪个模型", question):
        return (
            "我是这个变化检测可信判读系统里的问答助手。当前主线后端使用 DeepSeek-R1-Distill-Qwen-7B 文本模型做问答和报告生成；"
            "图像变化定位由变化检测模型完成，DeepSeek 读取后台结构化 facts 后回答。没有启用模型时会回退到本地证据规则。"
        )
    if re.search(r"在干嘛|做什么|忙什么", question):
        return (
            "我正在等待你的问题。闲聊问题我可以正常回复；只要问题涉及当前样本、区域、变化位置、对象数量、复核或报告，"
            "我就会回到后台证据上回答。"
        )
    if re.search(r"好吗|怎么样", question):
        return "我运行正常。你可以直接问闲聊问题，也可以问当前遥感变化样本的判读、复核和报告问题。"
    return (
        f"你好，我是 {region.get('short', '当前区域')} 的变化判读助手。"
        "闲聊我会按普通对话回复；涉及项目的判断会围绕当前影像、变化范围和后台证据回答。"
        + current
    )


def is_project_question(text: str) -> bool:
    raw = str(text or "")
    q = raw.lower()
    if not q.strip():
        return False
    evidence_anchor = re.search(
        r"(当前|样本|这张|这幅|本图|区域|复核|对象|面积|位置|范围|变化|报告|检测结果|巡检|mask|overlay|bbox|levir|whu|sysu|clcd|uav|建筑|道路|地表|低空)",
        raw,
        flags=re.I,
    )
    if re.search(r"(写|生成).{0,12}(诗|故事|作文|文案|代码|邮件|短文)", raw) and not evidence_anchor:
        return False
    if re.search(r"(介绍|解释|原理|区别|综述|论文)", raw) and not evidence_anchor:
        return False
    return bool(
        re.search(
            r"(变化|样本|区域|复核|对象|数量|面积|位置|范围|报告|巡检|问答|mask|overlay|bbox|levir|whu|sysu|clcd|uav|建筑|道路|地表|低空)",
            raw,
            flags=re.I,
        )
    )


PUBLIC_LEAK_PATTERNS = [
    re.compile(r"\bGT\b", re.I),
    re.compile(r"ground\s*truth", re.I),
    re.compile(r"真值|标注|标签|测试集答案|真实\s*mask", re.I),
    re.compile(r"gt[_-]?(mask|overlay)", re.I),
    re.compile(r"mask\s*(F1|IoU|precision|recall)", re.I),
    re.compile(r"\b(F1|IoU)\b", re.I),
    re.compile(r"(和|与)?后台.{0,12}(MASK|mask|变化范围).{0,12}不匹配", re.I),
    re.compile(r"(MASK|mask|变化范围).{0,8}不匹配", re.I),
]


def strip_public_leakage_sentences(text: str) -> str:
    pieces = re.split(r"(?<=[。！？!?；;])|\n+", str(text or ""))
    kept: list[str] = []
    for piece in pieces:
        part = piece.strip()
        if not part:
            continue
        if any(pattern.search(part) for pattern in PUBLIC_LEAK_PATTERNS):
            continue
        kept.append(part)
    return "\n".join(kept).strip() if "\n" in str(text or "") else "".join(kept).strip()


def sanitize_user_answer(text: str) -> str:
    replacements = {
        "evidence JSON": "后台证据",
        "Evidence JSON": "后台证据",
        "JSON object": "结构化对象",
        "JSON": "结构化数据",
        "Verifier": "可信校验",
        "F1": "检测质量",
        "IoU": "重叠质量",
        "预测掩膜": "变化范围",
        "掩膜": "变化范围",
        "mask": "变化范围",
        "Mask": "变化范围",
        "overlay": "变化叠加图",
        "Overlay": "变化叠加图",
        "证据包": "后台证据",
        "证据绑定": "后台检查",
        "面积一致性": "面积核对",
        "对象一致性": "数量核对",
        "事件（event）": "系统判定的变化类型",
        "event": "变化类型",
        "local change": "局部变化",
        "对象级证据核验与报告生成": "后续判读和报告生成",
        "质量标志": "质量提示",
    }
    cleaned = strip_public_leakage_sentences(strip_deepseek_reasoning(text))
    for src, dst in replacements.items():
        cleaned = cleaned.replace(src, dst)
    cleaned = re.sub(r"具体位置在[0-9,，\s]+坐标处[，,]?", "", cleaned)
    return cleaned


def sample_event_key(sample: dict[str, Any]) -> str:
    return str(sample.get("eventType") or sample.get("pred") or "no_primary_event")


def sample_event_name(sample: dict[str, Any]) -> str:
    return str(sample.get("eventTypeZh") or class_name(sample))


def normalize_named_entries(entries: Any) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    if not isinstance(entries, list):
        return normalized
    for entry in entries:
        if isinstance(entry, dict):
            name = str(entry.get("name") or entry.get("code") or entry.get("label") or "").strip()
            text = str(entry.get("text") or entry.get("label") or name).strip()
            code = str(entry.get("code") or name).strip()
            level = str(entry.get("level") or "").strip()
        else:
            text = str(entry or "").strip()
            name = text
            code = text
            level = ""
        if not any((name, text, code)):
            continue
        normalized.append(
            {
                "name": name or code or text,
                "text": text or name or code,
                "code": code or name or text,
                "level": level,
            }
        )
    return normalized


def compact_sample_for_prompt(sample: dict[str, Any] | None) -> dict[str, Any] | None:
    if not sample:
        return None
    objects = []
    for idx, obj in enumerate(sample.get("objects") or []):
        if idx >= 8:
            break
        objects.append(
            {
                "id": obj.get("id") or f"obj_{idx + 1:03d}",
                "areaPx": obj.get("areaPx"),
                "grid": obj.get("grid") or sample.get("gridPosition"),
                "shape": obj.get("shape"),
            }
        )
    return {
        "sampleName": sample.get("stem") or sample.get("name") or sample.get("id"),
        "eventKey": sample_event_key(sample),
        "eventName": sample_event_name(sample),
        "morphologyType": sample.get("morphologyType") or sample.get("pred"),
        "morphologyName": sample.get("morphologyLabel") or class_name(sample),
        "changeLevel": sample.get("changeLevel"),
        "riskFlags": sample.get("riskFlags") or [],
        "displayPolicy": sample.get("displayPolicy"),
        "evidenceSummary": sample.get("evidenceSummary"),
        "changedPx": sample.get("changedPx"),
        "changedRatio": sample.get("changedRatio"),
        "objectCount": sample.get("objectCount"),
        "gridPosition": sample.get("gridPosition"),
        "reviewStatus": sample.get("reviewStatus"),
        "reviewReason": sample.get("reviewReason"),
        "confidence": sample.get("confidence"),
        "qualityHints": [flag.get("name") for flag in normalize_named_entries(sample.get("qualityFlags")) if flag.get("name")],
        "objects": objects,
        "hardRules": {
            "changedPx": "If mentioned, copy this total value exactly; do not estimate another area.",
            "objectCount": "If mentioned, copy this count exactly.",
            "gridPosition": "If mentioned, use this coarse position exactly.",
            "reviewStatus": "Use this only as a recommendation signal; do not claim final field verification.",
            "temporalDirection": "T1/time1/pre is before; T2/time2/post is after. Do not reverse them.",
        },
    }


def region_scene_guide(region: dict[str, Any]) -> str:
    text = " ".join(str(region.get(k) or "") for k in ("id", "short", "name", "role", "status")).lower()
    if "clcd" in text or "地表" in text or "覆盖" in text:
        return "当前数据域偏向地表覆盖变化，描述时优先使用耕地、林地、草地、水体、裸地、建设用地等地表覆盖语汇。"
    if "whu" in text or "levir" in text or "建筑" in text or "建设" in text:
        return "当前数据域偏向建筑/建设用地变化，描述时优先使用新增建筑、建筑建成、硬化地表、建设扩张等语汇。"
    if "sysu" in text or "复杂" in text or "域外" in text:
        return "当前数据域偏向复杂或域外变化，描述时保持谨慎，优先说明变化范围、对象数量和不确定性。"
    return "根据当前区域说明选择合适语汇，不要套用与数据域无关的地物类别。"


def postprocess_sample_answer(payload: dict[str, Any], answer: str) -> str:
    sample = payload.get("sample") or {}
    region = payload.get("region") or {}
    if not sample or not payload.get("singleSampleOnly"):
        return answer
    text = answer.strip()
    text = re.sub(
        r"每个[^。；;\n]*(?:面积|大小)[^。；;\n]*(?:\d+[\s,，]*(?:-|~|至|到)[\s,，]*\d+|\d+)[^。；;\n]*(?:像素|px)[。；;]?",
        "单个对象面积以对象明细为准，AI 不做额外估计。",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"各个[^。；;\n]*(?:面积|大小)[^。；;\n]*(?:\d+[\s,，]*(?:-|~|至|到)[\s,，]*\d+|\d+)[^。；;\n]*(?:像素|px)[。；;]?",
        "单个对象面积以对象明细为准，AI 不做额外估计。",
        text,
        flags=re.I,
    )
    text = text.replace("不需要人工复核", "不需要优先进入人工复核，可按抽检规则查看")
    text = text.replace("无需人工复核", "无需优先进入人工复核，可按抽检规则查看")
    text = text.replace("不需要进一步的人工复核", "不需要优先进入人工复核，可按抽检规则查看")
    text = text.replace("无需进一步的人工复核", "无需优先进入人工复核，可按抽检规则查看")
    text = text.replace("因为检测质量达标", "依据后台变化范围和质量提示")
    text = text.replace("因为系统检测并通过", "依据后台变化范围和质量提示")
    text = re.sub(r"(占据|覆盖)了?整个图像的?大部分区域", "变化范围较大", text)
    text = text.replace("从图中可以看到，", "")
    text = text.replace("从图中可以看出，", "")
    text = text.replace("根据图像可以看到，", "")
    region_text = " ".join(str(region.get(k) or "") for k in ("id", "short", "name", "role")).lower()
    building_domain = any(token in region_text for token in ("levir", "whu", "建筑", "建设"))
    if building_domain and re.search(r"(建筑|建筑物|大型建筑|硬化).{0,40}(拆除|被拆除|移除|消失|只剩|留下).{0,40}(空地|裸地|地块)", text):
        text = re.sub(
            r"结论[:：]?[^\n。]*(拆除|被拆除|移除|消失)[^\n。]*[。]?",
            "结论：后时相相较前时相出现新增建筑或硬化设施，表现为建成/建设扩张。",
            text,
        )
        text = re.sub(
            r"(主要表现为|表现为)[^\n。]*(拆除|被拆除|移除|消失)[^\n。]*[。]?",
            "主要表现为后时相新增建筑或硬化设施。",
            text,
        )
        text = re.sub(
            r"原本有[^\n。]*(建筑|建筑物|大型建筑)[^\n。]*(空地|裸地|地块)[^\n。]*[。]?",
            "前时相对应区域建设痕迹不明显，后时相出现新增建筑或硬化设施。",
            text,
        )
        text = re.sub(
            r"[^\n。]*(建筑|建筑物|大型建筑)[^\n。]*(拆除|被拆除|移除|消失)[^\n。]*(空地|裸地|地块)[^\n。]*[。]?",
            "后时相相较前时相出现新增建筑或硬化设施。",
            text,
        )
    text = re.sub(
        r"(是否需要复核|复核建议)[:：]\s*不需要优先进入人工复核，可按抽检规则查看[^。\n]*[。]?",
        "复核建议：不需要优先进入人工复核，可按抽检规则查看。",
        text,
    )

    missing = []
    object_count = sample.get("objectCount")
    changed_px = sample.get("changedPx")
    grid_position = sample.get("gridPosition")
    if isinstance(object_count, int) and str(object_count) not in text:
        missing.append(f"对象数量 {object_count} 个")
    if isinstance(changed_px, (int, float)) and str(int(changed_px)) not in text:
        missing.append(f"总变化面积 {int(changed_px)}px")
    if grid_position and str(grid_position) not in text:
        missing.append(f"主要位置 {grid_position}")
    if missing:
        text += "\n\n后台核对：" + "，".join(missing) + "。"
    return text


def mask_score(sample: dict[str, Any]) -> str:
    f1 = sample.get("maskF1")
    iou = sample.get("maskIoU")
    if isinstance(f1, (int, float)):
        if f1 >= 0.85 and (not isinstance(iou, (int, float)) or iou >= 0.65):
            return "检测质量高"
        if f1 >= 0.70:
            return "检测质量中等，建议抽检"
        return "检测质量偏低，建议复核"
    return "无离线参考指标，按部署质量规则判断"


def top_review_lines(items: list[dict[str, Any]]) -> str:
    if not items:
        return "当前没有候选样本。"
    lines = []
    for idx, item in enumerate(items[:8], 1):
        lines.append(
            f"{idx}. {item.get('stem') or item.get('name')}: "
            f"{item.get('reviewStatus')}, {mask_score(item)}, "
            f"{fmt(item.get('changedPx', 0))}px, {class_name(item)}。"
            f"{item.get('reviewReason') or item.get('evidenceSummary') or ''}"
        )
    return "\n".join(lines)


def first_value(source: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in source and source.get(key) is not None:
            return source.get(key)
    return default


def percent_text(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "暂无参考指标"
    number = float(value)
    if 0.0 <= number <= 1.0:
        number *= 100.0
    return f"{number:.1f}%"


def sample_review_action(sample: dict[str, Any]) -> str:
    if sample.get("reviewRequired"):
        return "建议进入人工复核，重点检查漏检、过检或边界偏移。"
    if sample.get("pred") == "complex_or_uncertain_change":
        return "变化形态较复杂，建议作为重点案例抽检。"
    if sample.get("pred") == "no_change_or_weak_change":
        return "未发现稳定变化线索，可低优先级归档。"
    return "当前未触发强制复核，可按抽检规则查看。"


def sample_human_confidence(sample: dict[str, Any]) -> str:
    if sample.get("reviewRequired"):
        return "需要复核"
    f1 = sample.get("maskF1")
    if isinstance(f1, (int, float)):
        if f1 >= 0.85:
            return "高可信"
        if f1 >= 0.70:
            return "建议抽检"
        return "需要复核"
    confidence = sample.get("confidence")
    if isinstance(confidence, (int, float)):
        if confidence >= 0.90:
            return "高可信"
        if confidence >= 0.75:
            return "建议抽检"
    return "需要复核"


def sample_object_summary(sample: dict[str, Any]) -> str:
    objects = sample.get("objects") or []
    if not objects:
        count = int(sample.get("objectCount") or 0)
        return "当前没有稳定变化对象。" if count == 0 else f"主要位置为 {sample.get('gridPosition') or '未知'}。"
    parts = []
    for index, obj in enumerate(objects[:6], 1):
        parts.append(
            f"{index}号对象约 {fmt(obj.get('areaPx', 0))}px，位于"
            f"{obj.get('grid') or obj.get('gridPosition') or sample.get('gridPosition') or '未知位置'}"
        )
    return "；".join(parts) + "。"


def sample_quality_summary(sample: dict[str, Any]) -> str:
    parts = []
    if sample.get("reviewReason"):
        parts.append(str(sample["reviewReason"]))
    for item in normalize_named_entries(sample.get("qualityFlags"))[:5]:
        text = item.get("text") or item.get("name")
        if text:
            parts.append(str(text))
    if not parts:
        parts.append("未触发明显质量异常，仍建议对照 T1/T2 与变化叠加图核对。")
    return " ".join(parts)


def road_context_available(sample: dict[str, Any]) -> bool:
    keys = {
        "road_changed",
        "roadChanged",
        "road_region_count",
        "roadRegionCount",
        "road_change_count",
        "road_regions",
        "roadRegions",
        "road_reason",
        "roadReason",
    }
    return any(key in sample for key in keys)


def corpus_context(payload: dict[str, Any]) -> dict[str, str]:
    sample = payload.get("sample") or {}
    region = payload.get("region") or {}
    review_items = payload.get("reviewCandidates") or []
    changed_ratio = sample.get("changedRatio")
    road_changed = first_value(sample, "road_changed", "roadChanged")
    if road_changed is True:
        road_changed_text = "检测到道路变化"
    elif road_changed is False:
        road_changed_text = "未检测到明显道路变化"
    else:
        road_changed_text = "道路变化状态未知"
    road_count = first_value(
        sample,
        "road_region_count",
        "roadRegionCount",
        "road_change_count",
        default=len(first_value(sample, "road_regions", "roadRegions", default=[]) or []),
    )
    counts = region.get("counts") or {}
    total = int(region.get("totalRows") or 0)
    changed = int(region.get("changedRows") or 0)
    reviews = int(region.get("reviewRows") or 0)
    region_summary = (
        f"当前区域为 {region.get('short') or region.get('name') or '未知区域'}，"
        f"主要场景是 {region.get('role') or '遥感变化巡检'}，状态为 {region.get('status') or '未配置'}。"
        f"当前后台统计包含 {total} 条样本，其中 {changed} 条存在变化，"
        f"{reviews} 条进入或建议进入复核。"
    )
    workload = (
        f"{region.get('short') or '当前区域'} 当前后台统计有 {reviews} 条复核任务，"
        f"总样本数 {total}，变化样本数 {changed}。"
    )
    if counts:
        distribution = "、".join(f"{key} {value} 条" for key, value in counts.items())
        region_summary += f" 事件分布为：{distribution}。"
    return {
        "sample.stem": str(sample.get("stem") or sample.get("name") or sample.get("id") or "当前样本"),
        "sample.event": sample_event_name(sample),
        "sample.morphology": str(sample.get("morphologyLabel") or class_name(sample)),
        "sample.changeLevel": str(sample.get("changeLevel") or ""),
        "sample.riskFlags": "、".join(str(x) for x in (sample.get("riskFlags") or [])),
        "sample.evidenceSummary": str(sample.get("evidenceSummary") or "暂无补充证据摘要。"),
        "sample.objectCount": str(int(sample.get("objectCount") or 0)),
        "sample.changedPx": str(int(sample.get("changedPx") or 0)),
        "sample.changedRatioPct": percent_text(changed_ratio),
        "sample.gridPosition": str(sample.get("gridPosition") or "未知位置"),
        "sample.reviewStatus": str(sample.get("reviewStatus") or ("需要复核" if sample.get("reviewRequired") else "未触发强制复核")),
        "sample.reviewReason": str(sample.get("reviewReason") or "未触发明确复核原因。"),
        "sample.reviewAction": sample_review_action(sample),
        "sample.qualitySummary": sample_quality_summary(sample),
        "sample.objectSummary": sample_object_summary(sample),
        "sample.humanConfidence": sample_human_confidence(sample),
        "sample.maskF1Pct": "该指标仅用于离线实验，不在业务前台展示",
        "sample.maskIoUPct": "该指标仅用于离线实验，不在业务前台展示",
        "sample.report": str(sample.get("report") or sample.get("evidenceSummary") or "当前样本暂无报告草稿。"),
        "sample.roadChangedText": road_changed_text,
        "sample.roadRegionCount": str(int(road_count or 0)),
        "sample.roadLocation": str(first_value(sample, "road_location", "roadLocation", default="未知位置")),
        "sample.roadReason": str(first_value(sample, "road_reason", "roadReason", default="暂无道路判定解释。")),
        "sample.buildingContext": str(first_value(sample, "building_context", "buildingContext", default="暂无建筑干扰说明。")),
        "region.reviewQueueSummary": top_review_lines(review_items),
        "region.summary": region_summary,
        "region.workloadSummary": workload,
    }


def render_corpus_template(template: str, payload: dict[str, Any]) -> str:
    context = corpus_context(payload)

    def replace(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        return context.get(key, "暂无对应证据")

    return re.sub(r"\{\{\s*([a-zA-Z0-9_.]+)\s*\}\}", replace, str(template or "")).strip()


def corpus_answer(payload: dict[str, Any], intent: dict[str, Any], question: str) -> str:
    required = set(intent.get("required_context") or [])
    sample = payload.get("sample") or {}
    region = payload.get("region") or {}
    missing = (
        ("sample" in required and not sample)
        or ("region" in required and not region)
        or ("road" in required and not road_context_available(sample))
    )
    if missing:
        return str(intent.get("missing_context_answer") or "当前上下文不足，请先选择对应区域或样本。")
    templates = intent.get("answer_templates") or []
    if not templates:
        return "当前意图尚未配置回答模板。"
    digest = hashlib.sha1(f"{intent.get('intent_id')}:{question}".encode("utf-8")).hexdigest()
    template = templates[int(digest[:8], 16) % len(templates)]
    return render_corpus_template(str(template), payload)


def rule_answer(payload: dict[str, Any]) -> str:
    question = str(payload.get("question") or "").strip()
    region = payload.get("region") or {}
    sample = payload.get("sample")
    review_items = payload.get("reviewCandidates") or []

    if is_small_talk(question):
        return small_talk_answer(payload)

    lower_question = question.lower()
    if has_any(question, ("天气", "温度", "下雨", "气象")) or has_any(lower_question, ("weather", "temperature", "rain")):
        return "当前服务只读取变化检测证据，不接入实时天气工具，因此不能回答天气问题。"

    if has_any(question, ("当前区域", "这个区域", "哪些", "最应该", "优先", "复核队列", "低F1", "低 F1", "漏检", "过检")) or has_any(
        lower_question, ("current region", "which", "priority", "review", "low f1", "miss", "false positive", "over-detection")
    ):
        return (
            f"{region.get('short', '当前区域')} 的优先复核建议如下。"
            "排序依据为检测质量偏低、变化面积较大或证据质量异常：\n"
            f"{top_review_lines(review_items)}"
        )

    if not sample:
        return (
            f"当前区域为 {region.get('short', '未知区域')}。"
            "请选择一个样本后，可以继续追问变化位置、对象数量、复核原因或报告结论。"
        )

    if has_any(question, ("哪里", "位置", "定位", "框", "bbox", "九宫格")) or has_any(lower_question, ("where", "location", "position", "bbox")):
        return (
            f"{sample.get('stem')} 的主变化位置为 {sample.get('gridPosition')}，"
            "可在复核图中查看系统标出的变化范围。"
            "该定位来自后台几何统计，不由大模型自由猜测。"
        )

    if has_any(question, ("几个", "多少", "数量", "计数", "对象")) or has_any(lower_question, ("how many", "count", "number", "object")):
        objects = sample.get("objects") or []
        object_text = "；".join(
            f"{idx + 1}号对象约 {fmt(obj.get('areaPx', 0))}px，位于{obj.get('grid') or sample.get('gridPosition')}"
            for idx, obj in enumerate(objects[:5])
        )
        object_count = int(sample.get("objectCount") or 0)
        if object_text:
            detail_text = object_text
        elif object_count > 0:
            detail_text = f"主要位置为{sample.get('gridPosition')}，后台证据未展开单个对象明细。"
        else:
            detail_text = "当前没有稳定变化对象。"
        return (
            f"{sample.get('stem')} 共检测到 {object_count} 个变化对象。"
            f"{detail_text} "
            "计数来自系统变化范围统计，不是大模型目测估计。"
        )

    if has_any(question, ("复核", "为什么", "原因", "队列", "可靠", "质量", "过检", "漏检", "幻觉", "校验")) or has_any(
        lower_question, ("why", "reason", "review", "quality", "reliable", "hallucination", "verify")
    ):
        flags = normalize_named_entries(sample.get("qualityFlags"))
        checks = normalize_named_entries(sample.get("verifierChecks"))
        reasons = [sample.get("reviewReason") or ""]
        reasons += [f"{x.get('name')}: {x.get('text')}" for x in checks]
        reasons += [f"{x.get('name')}: {x.get('text')}" for x in flags]
        reasons = [x for x in reasons if x]
        return (
            f"{sample.get('stem')} 当前状态为 {sample.get('reviewStatus')}，"
            f"事件类型为 {class_name(sample)}，检测质量为 {mask_score(sample)}。\n"
            + ("\n".join(reasons) if reasons else "未触发明显质量异常，可作为抽检样本。")
        )

    if has_any(question, ("报告", "结论", "草稿", "生成")) or has_any(lower_question, ("report", "conclusion", "draft", "generate")):
        return str(sample.get("report") or sample.get("evidenceSummary") or "该样本暂无报告草稿。")

    return (
        f"{sample.get('stem')} 的系统结论为“{class_name(sample)}”。"
        f"{sample.get('evidenceSummary', '')} "
        f"变化面积 {fmt(sample.get('changedPx', 0))}px，"
        f"对象数 {sample.get('objectCount', 0)}，"
        f"主位置 {sample.get('gridPosition')}。"
    )


def resolve_asset(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    if path.is_absolute() and path.exists():
        return path
    if WEB_ROOT:
        candidate = WEB_ROOT / path_text
        if candidate.exists():
            return candidate
    return None


def parse_json_maybe(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if not isinstance(value, str) or not value.strip():
        return None
    value = (
        value.strip()
        .replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("main_变化范围_full", "main_bbox_full")
        .replace("change_变化范围_full", "change_bbox_full")
    )
    value = re.sub(r"\bTrue\b", "true", value)
    value = re.sub(r"\bFalse\b", "false", value)
    value = re.sub(r"\bNone\b", "null", value)
    value = re.sub(r",\s*([}\]])", r"\1", value)
    try:
        return json.loads(value)
    except Exception:
        return None


def extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.I | re.S)
    if fenced:
        parsed = parse_json_maybe(fenced.group(1))
        if isinstance(parsed, dict):
            return parsed
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    parsed = parse_json_maybe(cleaned)
    if isinstance(parsed, dict):
        return parsed
    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                parsed = parse_json_maybe(cleaned[start : idx + 1])
                return parsed if isinstance(parsed, dict) else None
    return None


def exact_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(round(float(value)))
    text = str(value or "").replace(",", "").replace("，", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return default
    try:
        return int(round(float(match.group(0))))
    except Exception:
        return default


def exact_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").replace(",", "").replace("%", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return default
    try:
        return float(match.group(0))
    except Exception:
        return default


def sample_source_size(sample: dict[str, Any]) -> tuple[int, int]:
    raw = sample.get("sourceSize") or []
    if isinstance(raw, list) and len(raw) >= 2:
        width, height = exact_int(raw[0], 256), exact_int(raw[1], 256)
        if width > 0 and height > 0:
            return width, height
    cards = sample.get("objectCards") or []
    for card in cards:
        coord = card.get("coordinateSystem") or {}
        width, height = exact_int(coord.get("imageWidth")), exact_int(coord.get("imageHeight"))
        if width > 0 and height > 0:
            return width, height
    return 256, 256


def normalize_bbox(value: Any, sample: dict[str, Any] | None = None) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return []
    vals = [exact_int(v) for v in value[:4]]
    x1, y1, x2, y2 = min(vals[0], vals[2]), min(vals[1], vals[3]), max(vals[0], vals[2]), max(vals[1], vals[3])
    if sample is not None:
        width, height = sample_source_size(sample)
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return []
    return [x1, y1, x2, y2]


def normalize_point(value: Any, sample: dict[str, Any] | None = None) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return []
    x, y = exact_int(value[0]), exact_int(value[1])
    if sample is not None:
        width, height = sample_source_size(sample)
        x = max(0, min(width, x))
        y = max(0, min(height, y))
    return [x, y]


def bbox_union(boxes: list[list[int]]) -> list[int]:
    valid = [b for b in boxes if len(b) == 4]
    if not valid:
        return []
    return [
        min(b[0] for b in valid),
        min(b[1] for b in valid),
        max(b[2] for b in valid),
        max(b[3] for b in valid),
    ]


def detector_bbox(sample: dict[str, Any]) -> list[int]:
    bbox = normalize_bbox(sample.get("mainBbox"), sample)
    if bbox:
        return bbox
    boxes = [normalize_bbox(obj.get("bbox"), sample) for obj in (sample.get("objects") or [])]
    return bbox_union([box for box in boxes if box])


def bbox_center(bbox: list[int]) -> list[int]:
    if len(bbox) != 4:
        return []
    return [int(round((bbox[0] + bbox[2]) / 2)), int(round((bbox[1] + bbox[3]) / 2))]


def point_in_bbox(point: list[int], bbox: list[int], pad: int = 0) -> bool:
    if len(point) != 2 or len(bbox) != 4:
        return False
    return bbox[0] - pad <= point[0] <= bbox[2] + pad and bbox[1] - pad <= point[1] <= bbox[3] + pad


def bbox_iou(a: list[int], b: list[int]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def detector_mask_contains_point(sample: dict[str, Any], point: list[int], radius: int = 2) -> bool | None:
    if len(point) != 2:
        return None
    assets = sample.get("assets") or {}
    mask_path = resolve_asset(assets.get("mask"))
    if not mask_path:
        return None
    try:
        from PIL import Image

        image = Image.open(mask_path).convert("L")
    except Exception:
        return None
    width, height = sample_source_size(sample)
    mask_w, mask_h = image.size
    if width <= 1 or height <= 1 or mask_w <= 0 or mask_h <= 0:
        return None
    mx = int(round(point[0] / max(1, width - 1) * max(0, mask_w - 1)))
    my = int(round(point[1] / max(1, height - 1) * max(0, mask_h - 1)))
    pix = image.load()
    for yy in range(max(0, my - radius), min(mask_h, my + radius + 1)):
        for xx in range(max(0, mx - radius), min(mask_w, mx + radius + 1)):
            if pix[xx, yy] >= 40:
                return True
    return False


def detector_facts(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_name": sample.get("stem") or sample.get("name") or sample.get("id"),
        "changed_px": exact_int(sample.get("changedPx")),
        "object_count": exact_int(sample.get("objectCount")),
        "main_bbox_full": detector_bbox(sample),
        "grid_position": str(sample.get("gridPosition") or ""),
    }


def compact_object_cards_for_prompt(sample: dict[str, Any]) -> list[dict[str, Any]]:
    cards = []
    for card in (sample.get("objectCards") or [])[:4]:
        cards.append(
            {
                "objectId": card.get("objectId"),
                "areaPx": card.get("areaPx"),
                "bboxFull": card.get("bboxFull"),
                "centerFull": card.get("centerFull"),
                "gridPosition": card.get("gridPosition"),
                "shape": card.get("shape"),
                "cropBoxFull": card.get("cropBoxFull"),
                "referenceBoxFull": card.get("referenceBoxFull"),
                "panelOrder": ["T1 crop", "T2 crop", "diff crop", "mask crop", "overlay crop", "no-change reference crop"],
            }
        )
    return cards


def object_image_paths(sample: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    assets = sample.get("assets") or {}
    for key in ("pre", "post", "overlay", "mask"):
        if assets.get(key):
            paths.append(str(assets[key]))
    for card in (sample.get("objectCards") or [])[:2]:
        card_assets = card.get("assets") or {}
        if card_assets.get("card"):
            paths.append(str(card_assets["card"]))
    return paths


def build_object_judgement_prompt(payload: dict[str, Any]) -> str:
    region = payload.get("region") or {}
    sample = payload.get("sample") or {}
    evidence_pack = parse_json_maybe(sample.get("evidencePack")) or {}
    domain_card = evidence_pack.get("domain_card") or {}
    compact = {
        "region": {
            "id": region.get("id"),
            "short": region.get("short"),
            "sceneGuide": region_scene_guide(region),
        },
        "detector_facts": detector_facts(sample),
        "objects": (sample.get("objects") or [])[:8],
        "object_cards": compact_object_cards_for_prompt(sample),
        "domain_card": domain_card,
        "quality": {
            "reviewRequired": sample.get("reviewRequired"),
            "reviewStatus": sample.get("reviewStatus"),
            "reviewReason": sample.get("reviewReason"),
            "qualityFlags": sample.get("qualityFlags") or [],
        },
    }
    facts = detector_facts(sample)
    return (
        "你是遥感变化对象视觉核验助手。只判读当前单张样本，不总结数据集。\n"
        "输入图像顺序：全图T1、全图T2、全图overlay、全图mask，然后是对象卡。对象卡6格顺序：T1 crop、T2 crop、diff crop、mask crop、overlay crop、no-change reference crop。\n"
        "只输出一个紧凑JSON对象，不要Markdown，不要代码块，不要解释。\n"
        "必须逐字复制 detector_facts："
        + json.dumps(facts, ensure_ascii=False, separators=(",", ":"))
        + "\nJSON字段固定为：object_level_judgement_version, detector_facts, qwen_grounding, visual_reasoning, mask_consistency_audit, semantic_adjudication, risk_control, final_report_zh。\n"
        "枚举约束：grounding_confidence=low/medium/high；alignment_with_visible_change=good/partial/poor/uncertain；"
        "semantic candidate=unknown/suspected_water/suspected_vegetation_or_cropland/suspected_bareland/suspected_impervious/suspected_building_related/suspected_road_or_linear_surface；"
        "claim_strength=unknown/suspected/supported_by_semantic_evidence。没有CLCD栅格/语义模型/人工标签时，claim_strength只能是unknown或suspected。\n"
        "change_point_full/change_bbox_full必须使用全图像素坐标；无法可靠定位就填[]并设low。不要编造面积、数量、位置，不要说违法、灾害、拆迁、确定建筑或确定道路。\n"
        "当前证据：\n"
        + json.dumps(compact, ensure_ascii=False, indent=2)
    )


def evidence_prompt(payload: dict[str, Any]) -> str:
    region = payload.get("region") or {}
    sample = payload.get("sample")
    single_sample = bool(payload.get("singleSampleOnly"))
    review_items = [] if single_sample else (payload.get("reviewCandidates") or [])
    compact = {
        "question": payload.get("question"),
        "region": {
            "id": region.get("id"),
            "short": region.get("short"),
            "name": region.get("name"),
            "role": region.get("role"),
            "status": region.get("status"),
            "sceneGuide": region_scene_guide(region),
        },
        "sample": compact_sample_for_prompt(sample),
    }
    if not single_sample:
        compact["review_candidates"] = review_items[:8]
    task_scope = (
        "本次任务是单图智能判读：只允许判读 sample 字段对应的当前这一张样本图，必须围绕当前样本的 T1/T2、变化叠加图、变化范围和结构化证据回答。\n"
        "严禁总结多个样本、验证集、数据集、区域整体或复核队列；不要使用“多个验证样本”“样本区域内”“大多数样本”“所有样本”等区域级表述。\n"
        "如果需要说范围，只能说当前这一张图中的变化范围；如果证据不足，只说当前样本证据不足。\n"
        if single_sample
        else "本次任务可以结合当前问题判断是样本级问答还是区域级问答；只有用户问区域/队列/报告时，才使用 review_candidates。\n"
    )
    return (
        "你是遥感变化检测复核助手。你的任务不是重新替代检测模型判图，而是把检测结果和后台证据解释给巡检人员。\n"
        "必须只依据给定证据回答，不允许编造面积、对象数量、单个对象面积、bbox 或位置。\n"
        + task_scope
        + "必须优先使用结构化证据中的 eventName、evidenceSummary、changedPx、objectCount、gridPosition、reviewStatus。\n"
        "时间方向必须正确：T1/time1/pre 是前时相，T2/time2/post 是后时相。若 T2 出现新建筑/硬化设施而 T1 不明显，应描述为新增、建成或建设扩张；只有 T1 有而 T2 消失时，才可描述为拆除或消失。\n"
        "必须结合 region.sceneGuide 选择措辞：建筑变化域讲建筑/建设/硬化，地表覆盖域讲农田、林地、水体、裸地、建设用地等覆盖变化，复杂域保持谨慎。\n"
        "硬性约束：changedPx 是总变化面积，若提及必须逐字使用该数值；objectCount 是对象数量，若提及必须逐字使用该数值；gridPosition 是粗略位置，若提及必须逐字使用该位置。\n"
        "严禁把总面积拆成“每个对象面积约多少”，除非 objects 列表逐个给出了对应对象面积；无法逐个确认时，只说总面积和对象数量。\n"
        "严禁使用“系统已经证明真实变化”“无需任何人工检查”这类绝对表述；检测质量高时应说“不需要优先进入人工复核，可按抽检规则查看”。\n"
        "不要根据图像自由猜测具体场景或地物名称；除非结构化证据明确给出，否则不要说高尔夫球场、树木、车辆、水体、道路、建筑等具体对象。\n"
        "如果证据不足，请明确说证据不足并建议人工复核。\n"
        "回答要面向非技术巡检人员，简洁自然，优先按以下小标题输出：结论、依据、位置与数量、面积依据、复核建议。复核建议只写一句，原因要简单。\n"
        "不要写“占据整个图像的大部分区域”这类夸张表述；如果范围较大，只说变化范围较大或变化区域较连续。\n"
        "除非用户明确要求技术细节，否则不要主动提及 JSON、Verifier、F1、IoU、mask、overlay、bbox、坐标、证据包、预测掩膜、英文类别标签等内部术语；可改说后台证据、变化范围、变化叠加图。\n"
        "下面是结构化证据 JSON：\n"
        + json.dumps(compact, ensure_ascii=False, indent=2)
    )


def ensure_qwen_loaded() -> None:
    global MODEL, PROCESSOR, PROCESS_VISION_INFO, TORCH
    if MODEL is not None and PROCESSOR is not None:
        return
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration

        model_cls = Qwen2_5_VLForConditionalGeneration
    except Exception:
        from transformers import AutoModelForVision2Seq

        model_cls = AutoModelForVision2Seq

    TORCH = torch
    PROCESS_VISION_INFO = process_vision_info
    PROCESSOR = AutoProcessor.from_pretrained(ARGS.model_id, trust_remote_code=True, local_files_only=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    MODEL = model_cls.from_pretrained(
        ARGS.model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        local_files_only=True,
    )
    if not torch.cuda.is_available():
        MODEL.eval()


def qwen_answer(payload: dict[str, Any]) -> str:
    ensure_qwen_loaded()
    assert MODEL is not None and PROCESSOR is not None and PROCESS_VISION_INFO is not None and TORCH is not None

    sample = payload.get("sample") or {}
    assets = sample.get("assets") or {}
    image_items = []
    for key in ("pre", "post", "overlay", "mask"):
        image_path = resolve_asset(assets.get(key))
        if image_path:
            image_items.append({"type": "image", "image": str(image_path), "max_pixels": ARGS.max_pixels})

    messages = [{"role": "user", "content": image_items + [{"type": "text", "text": evidence_prompt(payload)}]}]
    text = PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = PROCESS_VISION_INFO(messages)
    inputs = PROCESSOR(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    if hasattr(MODEL, "device"):
        inputs = inputs.to(MODEL.device)
    with TORCH.no_grad():
        generated = MODEL.generate(**inputs, max_new_tokens=ARGS.max_new_tokens)
    trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated)]
    return PROCESSOR.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def qwen_generate_multimodal(prompt: str, image_paths: list[str]) -> str:
    ensure_qwen_loaded()
    assert MODEL is not None and PROCESSOR is not None and PROCESS_VISION_INFO is not None and TORCH is not None

    image_items = []
    for path_text in image_paths:
        image_path = resolve_asset(path_text)
        if image_path:
            image_items.append({"type": "image", "image": str(image_path), "max_pixels": ARGS.max_pixels})
    messages = [{"role": "user", "content": image_items + [{"type": "text", "text": prompt}]}]
    text = PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = PROCESS_VISION_INFO(messages)
    inputs = PROCESSOR(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    if hasattr(MODEL, "device"):
        inputs = inputs.to(MODEL.device)
    with TORCH.no_grad():
        generated = MODEL.generate(**inputs, max_new_tokens=max(ARGS.max_new_tokens, 720), do_sample=False)
    trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated)]
    return PROCESSOR.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def compact_free_text(text: str, limit: int = 360) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def looks_like_structured_output(text: str) -> bool:
    return bool(
        re.search(
            r"```|\{\s*\"|object_level_judgement_version|detector_facts|qwen_grounding|semantic_adjudication|changed_px|object_count|bbox",
            str(text or ""),
            flags=re.I,
        )
    )


def extract_qwen_embedded_observation(raw: str) -> str:
    text = str(raw or "")
    keys = ("main_difference", "before_observation", "after_observation", "qwen_grounding", "final_report_zh", "final_report")
    for key in keys:
        pattern = rf'"{key}"\s*:\s*"((?:\\.|[^"\\])*)"'
        match = re.search(pattern, text, flags=re.S)
        if not match:
            continue
        value = match.group(1)
        try:
            value = json.loads('"' + value + '"')
        except Exception:
            value = value.replace("\\n", " ").replace('\\"', '"')
        value = compact_free_text(value, 420)
        if len(value) >= 12:
            return value
    dangling = re.search(r'"(?:qwen_grounding|main_difference|final_report_zh|final_report)"\s*:\s*"(.+)$', text, flags=re.S)
    if dangling:
        value = dangling.group(1)
        value = re.sub(r"\s*```.*$", "", value, flags=re.S)
        value = value.strip().strip('"').strip()
        value = compact_free_text(value, 420)
        if len(value) >= 12:
            return value
    return ""


def safe_visual_observation_from_text(text: str) -> str:
    raw = str(text or "")
    lower = raw.lower()
    if any(key in lower for key in ("building", "road", "residential", "commercial", "vehicle")) or any(
        key in raw for key in ("建筑", "道路", "居民", "车辆", "硬化")
    ):
        return "Qwen 观察到对象附近可能存在人工地表、建筑或道路纹理；由于缺少独立语义证据，系统仅将其作为疑似视觉线索。"
    if any(key in lower for key in ("water", "river", "pond")) or any(key in raw for key in ("水体", "河流", "湖")):
        return "Qwen 观察到对象附近可能存在水体或水边纹理；由于缺少独立语义证据，系统仅将其作为疑似视觉线索。"
    if any(key in lower for key in ("vegetation", "field", "cropland", "grass")) or any(
        key in raw for key in ("植被", "农田", "耕地", "草地")
    ):
        return "Qwen 观察到对象附近可能存在植被或农田纹理；由于缺少独立语义证据，系统仅将其作为疑似视觉线索。"
    cleaned = compact_free_text(raw, 260)
    if cleaned:
        return "Qwen 对对象卡进行了视觉描述；后台已过滤不能与证据绑定的内容，仅保留为辅助判读线索。"
    return "Qwen 未返回稳定的视觉观察，本次判读以后台检测证据为准。"


def public_free_text_observation(raw: str) -> str:
    embedded = extract_qwen_embedded_observation(raw)
    if embedded:
        return safe_visual_observation_from_text(embedded)
    if looks_like_structured_output(raw):
        return "Qwen 已围绕对象卡进行视觉核验；后台仅保留其中可与检测证据对齐的判读信息，并校正面积、数量和位置。"
    text = compact_free_text(raw)
    if not text:
        return "Qwen 未返回稳定的视觉观察，本次判读以后台检测证据为准。"
    return safe_visual_observation_from_text(text)


def free_text_semantic_candidate(text: str) -> tuple[str, list[str]]:
    lower = text.lower()
    checks = [
        ("suspected_water", ("水体", "河流", "湖", "pond", "water", "river")),
        ("suspected_vegetation_or_cropland", ("农田", "耕地", "植被", "草地", "林地", "vegetation", "cropland", "field")),
        ("suspected_bareland", ("裸地", "空地", "bare", "soil")),
        ("suspected_building_related", ("建筑", "房屋", "屋顶", "建设", "building", "roof", "residential", "commercial")),
        ("suspected_road_or_linear_surface", ("道路", "公路", "线性", "road", "linear")),
        ("suspected_impervious", ("硬化", "广场", "impervious", "paved")),
    ]
    for candidate, keys in checks:
        if any(key in lower or key in text for key in keys):
            return candidate, [f"Qwen 自由文本中出现与 {candidate} 相关的视觉词，但未作为确认语义使用。"]
    return "unknown", ["Qwen 没有给出可稳定绑定到证据协议的具体语义。"]


def qwen_free_text_object_judgement(payload: dict[str, Any], raw: str) -> dict[str, Any]:
    sample = payload.get("sample") or {}
    facts = detector_facts(sample)
    bbox = facts.get("main_bbox_full") or []
    point = bbox_center(bbox)
    changed = facts["changed_px"] > 0 and facts["object_count"] > 0
    public_observation = public_free_text_observation(raw)
    candidate_text = extract_qwen_embedded_observation(raw) or ("" if looks_like_structured_output(raw) else public_observation)
    candidate, basis = free_text_semantic_candidate(candidate_text)
    if candidate == "unknown":
        claim_strength = "unknown"
    else:
        claim_strength = "suspected"
    review_required = bool(sample.get("reviewRequired")) or not changed
    review_reasons = []
    if sample.get("reviewReason"):
        review_reasons.append(str(sample.get("reviewReason")))
    return {
        "object_level_judgement_version": "v1",
        "detector_facts": facts,
        "qwen_grounding": {
            "visible_change_exists": changed,
            "change_point_full": point,
            "change_bbox_full": bbox,
            "grounding_confidence": "low",
        },
        "visual_reasoning": {
            "before_observation": "",
            "after_observation": "",
            "main_difference": public_observation,
        },
        "mask_consistency_audit": {
            "alignment_with_visible_change": "uncertain",
            "possible_oversegmentation": False,
            "possible_undersegmentation": False,
            "possible_false_positive": False,
            "reason": "Qwen 未按结构化格式返回；系统保留其自由文本视觉观察，并由后台证据补齐数值、位置和复核字段。",
        },
        "semantic_adjudication": {
            "candidate": candidate,
            "claim_strength": claim_strength,
            "semantic_source": "qwen_free_text_visual_judgement",
            "confidence": "low",
            "basis": basis,
        },
        "risk_control": {
            "review_required": review_required,
            "review_priority": "high" if review_required else "low",
            "review_reasons": review_reasons,
        },
        "final_report_zh": public_observation,
        "_qwen_parse_mode": "free_text_fallback",
    }


def qwen_object_judgement(payload: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
    sample = payload.get("sample") or {}
    prompt = build_object_judgement_prompt(payload)
    raw = qwen_generate_multimodal(prompt, object_image_paths(sample))
    parsed = extract_json_object(raw)
    if not isinstance(parsed, dict):
        return qwen_free_text_object_judgement(payload, raw), f"Qwen free-text examiner + verifier: {ARGS.model_id}", raw
    return parsed, f"Qwen object-level examiner: {ARGS.model_id}", raw


def qwen_text_answer(prompt: str) -> str:
    ensure_qwen_loaded()
    assert MODEL is not None and PROCESSOR is not None and TORCH is not None
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = PROCESSOR(text=[text], padding=True, return_tensors="pt")
    if hasattr(MODEL, "device"):
        inputs = inputs.to(MODEL.device)
    with TORCH.no_grad():
        generated = MODEL.generate(**inputs, max_new_tokens=min(ARGS.max_new_tokens, 180))
    trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated)]
    return PROCESSOR.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def ensure_deepseek_loaded() -> None:
    global MODEL, TOKENIZER, TORCH
    if MODEL is not None and TOKENIZER is not None:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    TORCH = torch
    TOKENIZER = AutoTokenizer.from_pretrained(ARGS.model_id, trust_remote_code=True, local_files_only=True)
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": True,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    if ARGS.load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
    MODEL = AutoModelForCausalLM.from_pretrained(ARGS.model_id, **kwargs)
    if getattr(ARGS,'query_update_adapter',None):
        from peft import PeftModel
        MODEL=PeftModel.from_pretrained(MODEL,ARGS.query_update_adapter,is_trainable=False)
    MODEL.eval()


def strip_deepseek_reasoning(text: str) -> str:
    cleaned = str(text or "")
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.S | re.I)
    if "</think>" in cleaned.lower():
        cleaned = re.split(r"</think>", cleaned, flags=re.I)[-1]
    cleaned = cleaned.replace("**", "")
    cleaned = re.sub(r"```(?:json|text|markdown)?", "", cleaned, flags=re.I).replace("```", "")
    if "结论：" in cleaned:
        cleaned = cleaned[cleaned.rfind("结论：") :]
    elif "结论:" in cleaned:
        cleaned = cleaned[cleaned.rfind("结论:") :]
    else:
        for marker in ("回答：", "当前样本", "该样本", "本次"):
            idx = cleaned.find(marker)
            if idx >= 0:
                cleaned = cleaned[idx:]
                break

    process_line = re.compile(
        r"^\s*(首先|接下来|然后|最后|总结一下|我需要|用户希望|用户给出|我的任务是|这里需要|"
        r"根据这些信息|根据这些结构化|在回答时|现在[，,]?我|检查是否|接下来，我|最后，检查)"
    )
    kept: list[str] = []
    for para in re.split(r"\n{2,}", cleaned):
        part = para.strip()
        if not part:
            continue
        if process_line.search(part):
            continue
        kept.append(part)
    cleaned = "\n\n".join(kept).strip()
    for bad, good in (
        ("GT", "后台数据"),
        ("ground truth", "后台数据"),
        ("真值", "后台数据"),
        ("标签", "后台数据"),
        ("标注", "后台数据"),
        ("mask", "变化范围"),
        ("Mask", "变化范围"),
        ("掩膜", "变化范围"),
    ):
        cleaned = cleaned.replace(bad, good)
    return cleaned.strip()


def deepseek_project_prompt(payload: dict[str, Any]) -> str:
    region = payload.get("region") or {}
    sample = payload.get("sample") or {}
    mysql_context = db_context_for_prompt(payload)
    compact = {
        "user_question": payload.get("question"),
        "region": {
            "id": region.get("id"),
            "name": region.get("name"),
            "short": region.get("short"),
            "task_hint": region_scene_guide(region),
        },
        "sample": {
            "id": sample.get("stem") or sample.get("name") or sample.get("id"),
            "event": sample_event_name(sample),
            "event_type": sample.get("eventType"),
            "changed_px": sample.get("changedPx"),
            "changed_ratio": sample.get("changedRatio"),
            "object_count": sample.get("objectCount"),
            "main_location": sample.get("gridPosition"),
            "main_bbox": sample.get("mainBbox") or sample.get("bbox"),
            "review_required": sample.get("reviewRequired"),
            "review_status": sample.get("reviewStatus"),
            "review_reason": sample.get("reviewReason"),
            "evidence_summary": sample.get("evidenceSummary"),
            "quality_hint": sample.get("qualityHint") or sample.get("qualityStatus"),
        },
        "mysql_context": mysql_context,
    }
    q = str(payload.get("question") or "")
    bbox_note = ""
    if re.search(r"bbox|quad|坐标|范围|框", q, flags=re.I):
        bbox_note = (
            "\n用户问到 bbox/quad/坐标时，要说明它们是系统根据变化范围和连通域换算出的显示坐标，"
            "不是 DeepSeek 自己看图定位；是否可信应以检测结果和人工复核为准。"
        )
    return (
        "你是遥感变化巡检系统的中文终端助手。只输出最终答案，禁止输出思考过程、分析过程、检查过程、计划或自我总结。\n"
        "回答必须直接以“结论：”开头，最多 5 行。不要使用这些句式开头：首先、接下来、然后、最后、总结一下、我需要、用户希望、根据这些信息。\n"
        "你不能直接看图，只能根据下面的结构化结果回答；不得编造面积、数量、位置、bbox、变化原因或新增/拆除方向。\n"
        "如果结构化结果中包含 mysql_context，并且用户询问“一共、总共、多少条、区域、全部、复核队列、统计、排名”等问题，必须优先使用 mysql_context 回答；不要把 sample.object_count 当作区域样本总数。\n"
        "如果 mysql_context.region 同时给出 total_rows 和 display_rows，要区分“全量样本数”和“当前前端载入/可判读样本数”。\n"
        "如果问题涉及当前样本，必须使用 changed_px、object_count、main_location 的原值；如果证据不足，就说需要结合图像人工确认。\n"
        "复核建议要克制：只说明是否优先复核和原因，不要说绝对真实或绝对无误。"
        + bbox_note
        + "\n结构化结果：\n"
        + json.dumps(compact, ensure_ascii=False, indent=2, default=str)
        + "\n请直接给出最终答案。"
    )


def deepseek_generate_raw(prompt: str, max_tokens: int = 768, *, stop_event=None, deadline=None, system_prompt=None, sampling=None, max_input_tokens=None, history_messages=None,use_query_adapter=False) -> dict:
    """Unmodified text for machine-readable plans, with real usage and stop reason."""
    started = time.monotonic()
    with MODEL_GENERATION_LOCK:
        ensure_deepseek_loaded()
        assert MODEL is not None and TOKENIZER is not None and TORCH is not None
        prior=history_messages or []
        if not isinstance(prior,list) or len(prior)>12 or any(not isinstance(m,dict) or set(m)!={'role','content'} or m['role'] not in {'user','assistant'} or not isinstance(m['content'],str) for m in prior):
            raise ValueError('Invalid conversational message history')
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + prior + [{"role": "user", "content": prompt}]
        text = TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = TOKENIZER([text], return_tensors="pt")
        if max_input_tokens is not None and inputs['input_ids'].shape[-1] > max_input_tokens:
            raise ValueError('模型输入超过token预算，请缩短问题或开始新会话。')
        if hasattr(MODEL, "device"):
            inputs = inputs.to(MODEL.device)
        options = {}
        if stop_event is not None or deadline is not None:
            from transformers import StoppingCriteria, StoppingCriteriaList
            class StopReview(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    return bool((stop_event is not None and stop_event.is_set()) or
                                (deadline is not None and time.monotonic() >= deadline))
            options['stopping_criteria'] = StoppingCriteriaList([StopReview()])
        from contextlib import nullcontext
        adapter_context=MODEL.disable_adapter() if getattr(ARGS,'query_update_adapter',None) and not use_query_adapter else nullcontext()
        with TORCH.no_grad(),adapter_context:
            generated = MODEL.generate(**inputs, max_new_tokens=max_tokens, **(sampling or {'do_sample': False}),
                                       pad_token_id=TOKENIZER.eos_token_id, **options)
        trimmed = generated[0, inputs['input_ids'].shape[-1]:]
        eos = TOKENIZER.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        reason = 'stop' if len(trimmed) and int(trimmed[-1]) in eos_ids else 'length' if len(trimmed) >= max_tokens else 'stop'
        if stop_event is not None and stop_event.is_set(): reason = 'cancelled'
        elif deadline is not None and time.monotonic() >= deadline: reason = 'timeout'
        return {'text': TOKENIZER.decode(trimmed, skip_special_tokens=True),
                'inputTokens': int(inputs['input_ids'].shape[-1]), 'outputTokens': len(trimmed),
                'finishReason': reason, 'latencyMs': round((time.monotonic() - started) * 1000, 2)}


def deepseek_text_answer(prompt: str, max_new_tokens: int | None = None) -> str:
    system = (
        "你是遥感变化检测系统中的中文判读与问答助手。"
        "如果用户问题涉及当前样本、变化位置、对象数量、面积、复核或报告，必须只根据给定文本中的结构化信息回答；"
        "如果是普通闲聊，可以自然回答。"
        "不要输出思考过程、分析过程、检查过程或 <think>。"
        "不要声称自己看到了图像；没有图片理解能力时，应表述为基于后台结构化结果。"
        "回答要简洁、自然、面向业务人员。"
    )
    output = deepseek_generate_raw(prompt, max_new_tokens or ARGS.max_new_tokens, system_prompt=system)
    return strip_deepseek_reasoning(output['text'])


def casual_prompt(payload: dict[str, Any]) -> str:
    return (
        "你是遥感变化检测可信判读系统中的中文助手。用户现在问的是通用对话问题，"
        "可能是闲聊、常识、写作、天气或系统身份问题。不要强行引用遥感证据，不要编造当前样本结论。"
        "能直接回答的通用问题自然回答；涉及实时天气、实时新闻、联网检索等需要外部工具的问题，"
        "说明当前没有对应实时工具，只能给一般性建议。"
        f"如果用户问你是什么模型，说明当前主线后端使用 {ARGS.model_id} 文本模型作为问答与报告生成引擎；"
        "图像变化定位由变化检测模型完成，DeepSeek 只读取后台结构化 facts。回答要自然、简短，不要输出思考过程。\n"
        f"用户问题：{payload.get('question')}"
    )


def deepseek_casual_prompt(payload: dict[str, Any]) -> str:
    return (
        "你是遥感变化智能巡检平台中的中文助手。"
        "普通闲聊自然回答；涉及当前样本、区域、变化位置、对象数量、面积、复核或报告时，提醒用户需要结合后台结构化结果回答。"
        f"如果用户问你是什么模型，回答：当前后端使用 {ARGS.model_id} 文本模型作为问答与报告生成引擎，"
        "它不直接看图，图像变化定位由前面的变化检测模型完成。"
        "回答简洁，不要输出思考过程。\n"
        f"用户问题：{payload.get('question')}"
    )


def verify_answer(sample: dict[str, Any] | None, answer: str) -> tuple[str, dict[str, Any]]:
    if not sample:
        return "pass", {"checks": []}
    checks: list[dict[str, Any]] = []
    text = answer or ""
    object_count = sample.get("objectCount")
    grid_position = str(sample.get("gridPosition") or "")
    changed_px = sample.get("changedPx")
    if isinstance(object_count, int) and object_count >= 0:
        has_count = str(object_count) in text or (
            object_count == 0 and any(token in text for token in ("没有", "无明显", "未形成"))
        )
        checks.append({"name": "object_count", "ok": has_count, "expected": object_count})
    if grid_position:
        checks.append({"name": "grid_position", "ok": grid_position in text, "expected": grid_position})
    if isinstance(changed_px, (int, float)):
        checks.append({"name": "changed_px", "ok": str(int(changed_px)) in text, "expected": int(changed_px)})
    status = "pass" if all(check.get("ok") for check in checks) else "warn"
    return status, {"checks": checks}


def rule_object_judgement(payload: dict[str, Any]) -> dict[str, Any]:
    sample = payload.get("sample") or {}
    region = payload.get("region") or {}
    facts = detector_facts(sample)
    bbox = facts.get("main_bbox_full") or []
    point = bbox_center(bbox)
    changed = facts["changed_px"] > 0 and facts["object_count"] > 0
    region_text = " ".join(str(region.get(k) or "") for k in ("id", "short", "name", "role")).lower()
    if changed and any(token in region_text for token in ("levir", "whu", "建筑", "建设")):
        candidate = "suspected_building_related"
        semantic_basis = ["当前数据域偏建筑/建设变化，且变化范围呈面状或块状。"]
    elif changed:
        candidate = "unknown"
        semantic_basis = ["没有额外语义监督，只保留变化范围结论。"]
    else:
        candidate = "unknown"
        semantic_basis = ["检测范围中没有稳定变化对象。"]
    review_required = bool(sample.get("reviewRequired")) or not changed
    return {
        "object_level_judgement_version": "v1",
        "detector_facts": facts,
        "qwen_grounding": {
            "visible_change_exists": changed,
            "change_point_full": point,
            "change_bbox_full": bbox,
            "grounding_confidence": "medium" if changed else "low",
        },
        "visual_reasoning": {
            "before_observation": "以前时相图像为准。",
            "after_observation": "以后时相图像和变化范围为准。",
            "main_difference": "本地规则仅复述检测证据，不做开放视觉语义判断。",
        },
        "mask_consistency_audit": {
            "alignment_with_visible_change": "uncertain",
            "possible_oversegmentation": False,
            "possible_undersegmentation": False,
            "possible_false_positive": not changed,
            "reason": "未调用 Qwen 视觉核验，仅基于后台几何证据生成。",
        },
        "semantic_adjudication": {
            "candidate": candidate,
            "claim_strength": "suspected" if candidate != "unknown" else "unknown",
            "semantic_source": "detector_evidence_only",
            "confidence": "low",
            "basis": semantic_basis,
        },
        "risk_control": {
            "review_required": review_required,
            "review_priority": "high" if review_required else "low",
            "review_reasons": [sample.get("reviewReason") or "按后台复核规则处理。"],
        },
        "final_report_zh": "",
    }


def semantic_label(candidate: str, claim_strength: str) -> str:
    labels = {
        "unknown": "暂不确定具体地物类型",
        "suspected_water": "疑似水体相关变化",
        "suspected_vegetation_or_cropland": "疑似植被或农田覆盖变化",
        "suspected_bareland": "疑似裸地相关变化",
        "suspected_impervious": "疑似硬化地表变化",
        "suspected_building_related": "疑似建筑或建设相关变化",
        "suspected_road_or_linear_surface": "疑似道路或线性地表变化",
    }
    text = labels.get(candidate, "暂不确定具体地物类型")
    if claim_strength == "unknown":
        return "暂不确定具体地物类型"
    return text


def verify_object_judgement(
    sample: dict[str, Any],
    judgement: dict[str, Any],
    raw_text: str = "",
) -> tuple[str, dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    actions: list[str] = []
    fixed = judgement if isinstance(judgement, dict) else {}
    fixed["object_level_judgement_version"] = "v1"

    expected = detector_facts(sample)
    got_facts = fixed.get("detector_facts") if isinstance(fixed.get("detector_facts"), dict) else {}
    fact_checks = (
        ("changed_px", expected["changed_px"]),
        ("object_count", expected["object_count"]),
        ("grid_position", expected["grid_position"]),
    )
    for name, expected_value in fact_checks:
        got = got_facts.get(name)
        ok = exact_int(got) == expected_value if isinstance(expected_value, int) else str(got or "") == str(expected_value)
        checks.append({"name": f"detector_fact_{name}", "ok": ok, "expected": expected_value, "got": got})
        if not ok:
            actions.append(f"已把 {name} 校正为检测证据值。")
    got_bbox = normalize_bbox(got_facts.get("main_bbox_full"), sample)
    bbox_ok = got_bbox == expected["main_bbox_full"]
    checks.append({"name": "detector_fact_main_bbox", "ok": bbox_ok, "expected": expected["main_bbox_full"], "got": got_bbox})
    if not bbox_ok:
        actions.append("已把主变化范围校正为检测证据中的主范围。")
    fixed["detector_facts"] = expected

    grounding = fixed.get("qwen_grounding") if isinstance(fixed.get("qwen_grounding"), dict) else {}
    detector_main = expected.get("main_bbox_full") or []
    point = normalize_point(grounding.get("change_point_full"), sample)
    qbbox = normalize_bbox(grounding.get("change_bbox_full"), sample)
    if point:
        in_detector = point_in_bbox(point, detector_main, pad=3) if detector_main else True
        checks.append({"name": "grounding_point_in_detector_bbox", "ok": in_detector, "expected": detector_main, "got": point})
        mask_hit = detector_mask_contains_point(sample, point)
        if mask_hit is not None:
            checks.append({"name": "grounding_point_in_change_range", "ok": mask_hit, "expected": True, "got": point})
        if not in_detector or mask_hit is False:
            actions.append("Qwen 给出的定位点与检测变化范围不一致，已降低定位可信度并触发复核。")
            grounding["grounding_confidence"] = "low"
    else:
        checks.append({"name": "grounding_point_present", "ok": expected["object_count"] == 0, "expected": "point or no object", "got": []})
        if expected["object_count"] > 0:
            actions.append("Qwen 未给出可靠定位点，已触发复核。")
            grounding["grounding_confidence"] = "low"
    if qbbox and detector_main:
        iou = bbox_iou(qbbox, detector_main)
        ok = iou >= 0.15 or point_in_bbox(bbox_center(qbbox), detector_main, pad=8)
        checks.append({"name": "grounding_bbox_overlap_detector", "ok": ok, "expected": detector_main, "got": qbbox, "iou": round(iou, 4)})
        if not ok:
            actions.append("Qwen 给出的定位框与检测主范围偏离，已触发复核。")
            grounding["grounding_confidence"] = "low"
    elif expected["object_count"] > 0:
        checks.append({"name": "grounding_bbox_present", "ok": False, "expected": detector_main, "got": qbbox})
        actions.append("Qwen 未给出可靠定位框，已触发复核。")
        grounding["grounding_confidence"] = "low"
    grounding["change_point_full"] = point
    grounding["change_bbox_full"] = qbbox
    if grounding.get("grounding_confidence") not in {"low", "medium", "high"}:
        grounding["grounding_confidence"] = "low"
    fixed["qwen_grounding"] = grounding

    semantic = fixed.get("semantic_adjudication") if isinstance(fixed.get("semantic_adjudication"), dict) else {}
    allowed_candidates = {
        "unknown",
        "suspected_water",
        "suspected_vegetation_or_cropland",
        "suspected_bareland",
        "suspected_impervious",
        "suspected_building_related",
        "suspected_road_or_linear_surface",
    }
    candidate = str(semantic.get("candidate") or "unknown")
    if candidate not in allowed_candidates:
        actions.append(f"未识别的语义候选 {candidate} 已降级为 unknown。")
        candidate = "unknown"
    claim_strength = str(semantic.get("claim_strength") or "unknown")
    semantic_source = str(semantic.get("semantic_source") or "qwen_visual_judgement_only")
    has_semantic_source = semantic_source not in {"", "none", "qwen_visual_judgement_only", "detector_evidence_only"}
    if claim_strength == "supported_by_semantic_evidence" and not has_semantic_source:
        actions.append("没有外部语义监督来源，已把语义强度从确认支持降级为疑似。")
        claim_strength = "suspected" if candidate != "unknown" else "unknown"
    if candidate == "unknown":
        claim_strength = "unknown"
    overclaim_text = " ".join(
        [
            raw_text or "",
            str(fixed.get("final_report_zh") or ""),
            str(semantic.get("basis") or ""),
        ]
    )
    overclaim = bool(re.search(r"(违法|非法|灾害|洪水|火灾|产权|拆迁|确定为|确认是|确认为|一定是)", overclaim_text))
    checks.append({"name": "semantic_overclaim", "ok": not overclaim, "expected": "no unsupported hard claim", "got": bool(overclaim)})
    if overclaim:
        actions.append("检测到无证据强语义或业务结论，已降级为疑似/未知并触发复核。")
        if candidate != "unknown":
            claim_strength = "suspected"
    semantic["candidate"] = candidate
    semantic["claim_strength"] = claim_strength
    semantic["semantic_source"] = semantic_source or "qwen_visual_judgement_only"
    semantic["confidence"] = semantic.get("confidence") if semantic.get("confidence") in {"low", "medium", "high"} else "low"
    if not isinstance(semantic.get("basis"), list):
        semantic["basis"] = [str(semantic.get("basis") or "")]
    fixed["semantic_adjudication"] = semantic

    consistency = fixed.get("mask_consistency_audit") if isinstance(fixed.get("mask_consistency_audit"), dict) else {}
    if consistency.get("alignment_with_visible_change") not in {"good", "partial", "poor", "uncertain"}:
        consistency["alignment_with_visible_change"] = "uncertain"
    fixed["mask_consistency_audit"] = consistency

    failed = [check for check in checks if not check.get("ok")]
    risk = fixed.get("risk_control") if isinstance(fixed.get("risk_control"), dict) else {}
    review_reasons = risk.get("review_reasons") if isinstance(risk.get("review_reasons"), list) else []
    if sample.get("reviewRequired"):
        review_reasons.append(sample.get("reviewReason") or "样本触发系统复核规则。")
    if failed or overclaim:
        review_reasons.extend(action for action in actions if action)
    risk["review_required"] = bool(risk.get("review_required")) or bool(sample.get("reviewRequired")) or bool(failed) or overclaim
    risk["review_priority"] = risk.get("review_priority") if risk.get("review_priority") in {"low", "medium", "high"} else ("high" if risk["review_required"] else "low")
    risk["review_reasons"] = list(dict.fromkeys(str(x) for x in review_reasons if x))
    fixed["risk_control"] = risk

    status = "pass" if not failed and not actions else "warn"
    return status, {"checks": checks, "verifier_actions": actions, "object_judgement": fixed}


def format_object_judgement_report(sample: dict[str, Any], judgement: dict[str, Any], verify_json: dict[str, Any]) -> str:
    sections = object_judgement_sections(sample, judgement, verify_json)
    return (
        f"结论：{sections['conclusion']}\n\n"
        f"位置与数量：{sections['location_quantity']}\n\n"
        f"复核建议：{sections['review_suggestion']}"
    ).strip()


def normalize_judgement_answer(sample: dict[str, Any], raw_answer: str, judgement: dict[str, Any], verify_json: dict[str, Any]) -> str:
    cleaned = strip_deepseek_reasoning(raw_answer).replace("\r", "").strip()
    if not cleaned:
        return format_object_judgement_report(sample, judgement, verify_json)
    section_patterns = {
        "conclusion": ("结论",),
        "location_quantity": ("位置与数量", "位置和数量"),
        "review_suggestion": ("复核建议", "建议动作"),
    }
    extracted: dict[str, str] = {}
    for key, labels in section_patterns.items():
        value = ""
        for label in labels:
            match = re.search(
                rf"{label}[：:]\s*(.+?)(?=\n(?:结论|位置与数量|位置和数量|复核建议|建议动作)[：:]|\Z)",
                cleaned,
                flags=re.S,
            )
            if match:
                value = re.sub(r"\s+", " ", match.group(1)).strip()
                break
        extracted[key] = value
    fallback = object_judgement_sections(sample, judgement, verify_json)
    conclusion = extracted.get("conclusion") or fallback["conclusion"]
    location_quantity = extracted.get("location_quantity") or fallback["location_quantity"]
    review_suggestion = extracted.get("review_suggestion") or fallback["review_suggestion"]
    return (
        f"结论：{conclusion}\n\n"
        f"位置与数量：{location_quantity}\n\n"
        f"复核建议：{review_suggestion}"
    ).strip()
def _judgement_change_signature(sample: dict[str, Any], facts: dict[str, Any]) -> str:
    event_name = sample_event_name(sample)
    object_count = exact_int(facts.get("object_count"))
    changed_px = exact_int(facts.get("changed_px"))
    changed_ratio = exact_float(facts.get("changed_ratio") or sample.get("changedRatio") or 0.0)
    if object_count <= 0 or changed_px <= 0:
        return f"{event_name}，当前未形成稳定有效变化"
    if changed_ratio >= 0.45 or changed_px >= 30000:
        pattern = "大面积连续变化"
    elif object_count >= 8:
        pattern = "多目标分散变化"
    elif object_count <= 2:
        pattern = "少量主体变化"
    else:
        pattern = "局部多点变化"
    if re.search(r"地表|覆盖", event_name):
        event_hint = "，以地表覆盖替换或扰动为主"
    elif re.search(r"道路|线性", event_name):
        event_hint = "，以线性带状区域为主"
    elif re.search(r"建筑|设施|硬化", event_name):
        event_hint = "，以建筑或设施主体为主"
    else:
        event_hint = ""
    return f"{event_name}{event_hint}，整体呈{pattern}"


def object_judgement_sections(sample: dict[str, Any], judgement: dict[str, Any], verify_json: dict[str, Any]) -> dict[str, str]:
    facts = judgement.get("detector_facts") or detector_facts(sample)
    risk = judgement.get("risk_control") or {}
    object_count = exact_int(facts.get("object_count"))
    changed_px = exact_int(facts.get("changed_px"))
    grid_position = str(facts.get("grid_position") or sample.get("gridPosition") or "")
    review_required = bool(risk.get("review_required"))
    review_text = "建议进入人工复核。" if review_required else "可按常规抽检查看。"
    reasons = risk.get("review_reasons") if isinstance(risk.get("review_reasons"), list) else []
    public_reasons: list[str] = []
    for reason in reasons:
        reason = str(reason or "").strip()
        if not reason:
            continue
        if re.search(r"Qwen|JSON|object|bbox|语义|结构化", reason, flags=re.I):
            continue
        reason = reason.replace("验证集通过，", "").replace("检测质量达标，不列入误差复核", "当前未触发优先复核")
        public_reasons.append(reason)
    reason_text = "；".join(public_reasons[:2]).rstrip("。；; ")
    reason_clause = f" {reason_text}。" if reason_text else ""
    actions = verify_json.get("verifier_actions") or []
    verify_note = " 请以后端结构化证据和变化叠加图为准。" if actions else ""
    signature = _judgement_change_signature(sample, facts)
    if object_count <= 0 or changed_px <= 0:
        conclusion = f"当前样本以{sample_event_name(sample)}为主，暂未检测到稳定有效变化。"
        location_quantity = "系统未检测到稳定有效变化对象，当前无明确主体位置，总变化面积约 0 px。"
    else:
        conclusion = f"当前样本表现为{signature}。"
        location_quantity = f"系统检测到 {object_count} 个变化对象，主要位置为{grid_position}，总变化面积约 {changed_px} px。"
    review_suggestion = f"{review_text}{reason_clause}{verify_note}".strip()
    return {
        "conclusion": conclusion,
        "location_quantity": location_quantity,
        "review_suggestion": review_suggestion,
    }


def deepseek_judgement_prompt(payload: dict[str, Any], judgement: dict[str, Any], verify_json: dict[str, Any]) -> str:
    sample = payload.get("sample") or {}
    region = payload.get("region") or {}
    sections = object_judgement_sections(sample, judgement, verify_json)
    compact = {
        "region": {
            "id": region.get("id"),
            "short": region.get("short"),
            "name": region.get("name"),
            "role": region.get("role"),
            "status": region.get("status"),
        },
        "sample": {
            "id": sample.get("id") or sample.get("stem") or sample.get("name"),
            "event_name": sample_event_name(sample),
            "grid_position": judgement.get("detector_facts", {}).get("grid_position"),
            "changed_px": judgement.get("detector_facts", {}).get("changed_px"),
            "changed_ratio": judgement.get("detector_facts", {}).get("changed_ratio"),
            "object_count": judgement.get("detector_facts", {}).get("object_count"),
            "review_required": judgement.get("risk_control", {}).get("review_required"),
            "review_reasons": judgement.get("risk_control", {}).get("review_reasons"),
            "change_signature": _judgement_change_signature(sample, judgement.get("detector_facts") or {}),
        },
        "fallback_sections": sections,
    }
    return (
        "你是遥感变化巡检平台中的中文判读助手。你只能基于后台结构化证据回答，不直接看图，不判断预测是否真实正确。\n"
        "请只输出 3 段，不要输出 Markdown 列表，不要输出判断依据、疑似语义、后台校验、JSON、GT、F1、IoU、bbox、坐标。\n"
        "输出格式固定为：\n"
        "结论：<一句话概括当前这一张样本的变化形态，必须明确它更像大面积连续变化、局部多点变化、多目标分散变化、少量主体变化或无明显稳定变化；不要只复述类别名>\n"
        "位置与数量：<对象数量、主要位置、总变化面积>\n"
        "复核建议：<一句话说明是否建议人工复核及建议动作>\n"
        "如果不需要优先复核，只能写为可按常规抽检查看，不要写绝对正确。\n"
        "下面是后台结构化证据：\n"
        + json.dumps(compact, ensure_ascii=False, indent=2)
    )


def generate_project_answer(payload: dict[str, Any]) -> tuple[str, str]:
    if ARGS.use_qwen:
        answer = sanitize_user_answer(qwen_answer(payload))
        return postprocess_sample_answer(payload, answer), f"Qwen 多模态模型: {ARGS.model_id}"
    if getattr(ARGS, "use_deepseek", False):
        answer = sanitize_user_answer(deepseek_text_answer(deepseek_project_prompt(payload), max_new_tokens=min(max(ARGS.max_new_tokens, 512), 768)))
        return postprocess_sample_answer(payload, answer), f"DeepSeek text LLM: {ARGS.model_id}"
    if rule_answer_enabled():
        answer = sanitize_user_answer(rule_answer(payload))
        return postprocess_sample_answer(payload, answer), "local evidence rules"
    raise RuntimeError("本地固定模板回答已关闭，请启动 DeepSeek，或显式开启 --enable_rule_answer。")


def generate_casual_answer(payload: dict[str, Any]) -> tuple[str, str]:
    if ARGS.use_qwen:
        return sanitize_user_answer(qwen_text_answer(casual_prompt(payload))), f"Qwen text chat: {ARGS.model_id}"
    if getattr(ARGS, "use_deepseek", False):
        return sanitize_user_answer(deepseek_text_answer(deepseek_casual_prompt(payload), max_new_tokens=min(ARGS.max_new_tokens, 220))), f"DeepSeek text chat: {ARGS.model_id}"
    return sanitize_user_answer(small_talk_answer(payload)), "dialog boundary rules"


def general_fallback_answer(payload: dict[str, Any]) -> str:
    question = str(payload.get("question") or "")
    if is_small_talk(question) or re.search(r"什么模型|哪个模型|你是谁|能做什么", question):
        return small_talk_answer(payload)
    if is_weather_question(question):
        return (
            "这个问题不需要变化检测证据。当前后端没有接入实时天气工具，"
            "所以不能给出实时天气结论；即使启用 DeepSeek，也只能做一般性说明，实时天气仍建议接入天气工具。"
        )
    return (
        "这个问题不属于当前样本或区域的变化检测证据问答。当前后端未启用 DeepSeek 通用对话，"
        "这里只做本地占位回复；启动 --use_deepseek 后，这类通用问题会交给 DeepSeek 文本模型回答。"
    )


def generate_general_answer(payload: dict[str, Any]) -> tuple[str, str]:
    if ARGS.use_qwen:
        return sanitize_user_answer(qwen_text_answer(casual_prompt(payload))), f"Qwen general chat: {ARGS.model_id}"
    if getattr(ARGS, "use_deepseek", False):
        return sanitize_user_answer(deepseek_text_answer(deepseek_casual_prompt(payload), max_new_tokens=min(ARGS.max_new_tokens, 260))), f"DeepSeek general chat: {ARGS.model_id}"
    return sanitize_user_answer(general_fallback_answer(payload)), "local general fallback"


def database_scope_question(text: str) -> bool:
    text = str(text or "")
    return bool(
        re.search(
            r"(一共|总共|总计|共有|多少条|几条|多少张|几张|数据量|样本数|全量|全部|整体|统计|分布|"
            r"当前区域|这个区域|本区域|区域级|复核队列|优先复核|需要复核|待复核|哪些样本|哪些图|哪几张|前\d+个|最大|最小|排名|top|count|total|all|samples|review queue)",
            text,
            flags=re.I,
        )
    )


def db_context_for_prompt(payload: dict[str, Any]) -> dict[str, Any]:
    if not mysql_enabled():
        return {}
    question = str(payload.get("question") or "")
    sample = payload.get("sample") or {}
    region = payload.get("region") or {}
    region_id = str(region.get("id") or sample.get("regionId") or "").strip()
    sample_keys = sample_lookup_values(sample)
    wants_region = database_scope_question(question) or not sample_keys
    context: dict[str, Any] = {
        "source": "mysql",
        "question_scope": "region_or_global" if wants_region else "sample",
    }
    try:
        if region_id:
            region_row = db_fetch_one(
                """
                SELECT region_id, short_name, region_name, role_text, status_text, scene_guide,
                       total_rows, display_rows, changed_rows, display_changed_rows,
                       review_rows, full_review_rows, avg_confidence, counts_json,
                       display_counts_json, metrics_json
                FROM regions WHERE region_id=%s LIMIT 1
                """,
                (region_id,),
            )
            if region_row:
                context["region"] = {
                    "region_id": region_row.get("region_id"),
                    "short": region_row.get("short_name"),
                    "name": region_row.get("region_name"),
                    "role": region_row.get("role_text"),
                    "status": region_row.get("status_text"),
                    "scene_guide": region_row.get("scene_guide"),
                    "total_rows": db_number(region_row.get("total_rows")),
                    "display_rows": db_number(region_row.get("display_rows")),
                    "changed_rows": db_number(region_row.get("changed_rows")),
                    "display_changed_rows": db_number(region_row.get("display_changed_rows")),
                    "review_rows": db_number(region_row.get("review_rows")),
                    "full_review_rows": db_number(region_row.get("full_review_rows")),
                    "avg_confidence": db_number(region_row.get("avg_confidence"), 0.0),
                    "counts": parse_json_maybe(region_row.get("counts_json")) or {},
                    "display_counts": parse_json_maybe(region_row.get("display_counts_json")) or {},
                    "metrics": parse_json_maybe(region_row.get("metrics_json")) or {},
                }
            if wants_region:
                stats = db_fetch_one(
                    """
                    SELECT COUNT(*) AS loaded_samples,
                           SUM(CASE WHEN COALESCE(changed_px,0) > 0 THEN 1 ELSE 0 END) AS loaded_changed_samples,
                           SUM(CASE WHEN COALESCE(review_required,0) = 1 THEN 1 ELSE 0 END) AS loaded_review_samples,
                           SUM(COALESCE(changed_px,0)) AS loaded_changed_px,
                           AVG(confidence) AS avg_confidence,
                           MAX(changed_px) AS max_changed_px
                    FROM samples WHERE region_id=%s
                    """,
                    (region_id,),
                )
                if stats:
                    context["loaded_region_stats"] = {key: db_number(value) for key, value in stats.items()}
                context["event_distribution"] = db_fetch_all(
                    """
                    SELECT event_key, event_name, COUNT(*) AS count_rows
                    FROM samples WHERE region_id=%s
                    GROUP BY event_key, event_name
                    ORDER BY count_rows DESC
                    LIMIT 12
                    """,
                    (region_id,),
                )
                context["review_queue_top"] = db_fetch_all(
                    """
                    SELECT sample_id, sample_stem, event_name, changed_px, object_count,
                           grid_position, review_reason
                    FROM samples
                    WHERE region_id=%s AND COALESCE(review_required,0)=1
                    ORDER BY COALESCE(changed_px,0) DESC
                    LIMIT 12
                    """,
                    (region_id,),
                )
                context["top_area_samples"] = db_fetch_all(
                    """
                    SELECT sample_id, sample_stem, event_name, changed_px, object_count,
                           grid_position, review_required
                    FROM samples
                    WHERE region_id=%s
                    ORDER BY COALESCE(changed_px,0) DESC
                    LIMIT 12
                    """,
                    (region_id,),
                )
                correction_count = db_fetch_one(
                    "SELECT COUNT(*) AS count_rows FROM manual_corrections WHERE region_id=%s",
                    (region_id,),
                )
                context["manual_corrections"] = {
                    "count": db_number(correction_count.get("count_rows") if correction_count else 0)
                }

        if sample_keys:
            placeholders = ",".join(["%s"] * len(sample_keys))
            sample_row = db_fetch_one(
                f"""
                SELECT sample_id, region_id, sample_name, sample_stem, event_key, event_name,
                       changed_px, changed_ratio, object_count, grid_position,
                       review_status, review_required, review_reason, confidence,
                       pred_area, main_bbox_json, quality_flags_json, report_text, display_policy,
                       evidence_hash, public_facts_json, evidence_json, assets_json
                FROM samples
                WHERE sample_id IN ({placeholders}) OR sample_name IN ({placeholders}) OR sample_stem IN ({placeholders})
                LIMIT 1
                """,
                sample_keys * 3,
            )
            if sample_row:
                context["sample_from_db"] = {
                    "sample_id": sample_row.get("sample_id"),
                    "region_id": sample_row.get("region_id"),
                    "name": sample_row.get("sample_name"),
                    "stem": sample_row.get("sample_stem"),
                    "event_key": sample_row.get("event_key"),
                    "event_name": sample_row.get("event_name"),
                    "changed_px": db_number(sample_row.get("changed_px")),
                    "changed_ratio": db_number(sample_row.get("changed_ratio"), 0.0),
                    "object_count": db_number(sample_row.get("object_count")),
                    "grid_position": sample_row.get("grid_position"),
                    "review_status": sample_row.get("review_status"),
                    "review_required": bool(sample_row.get("review_required")),
                    "review_reason": sample_row.get("review_reason"),
                    "confidence": db_number(sample_row.get("confidence"), 0.0),
                    "main_bbox": parse_json_maybe(sample_row.get("main_bbox_json")) or [],
                    "quality_flags": parse_json_maybe(sample_row.get("quality_flags_json")) or [],
                    "display_policy": sample_row.get("display_policy"),
                    "evidence_hash": sample_row.get("evidence_hash"),
                    "public_facts": parse_json_maybe(sample_row.get("public_facts_json"))
                    or parse_json_maybe(sample_row.get("evidence_json"))
                    or {},
                }
                context["sample_objects"] = db_fetch_all(
                    """
                    SELECT
                        COALESCE(object_key, CONCAT('obj_', LPAD(COALESCE(object_index, 0), 3, '0'))) AS object_id,
                        area_px,
                        bbox_json,
                        NULL AS center_x,
                        NULL AS center_y,
                        grid_position,
                        shape_text AS label_text
                    FROM sample_objects
                    WHERE sample_id=%s
                    ORDER BY area_px DESC, COALESCE(object_index, 0) ASC, id ASC
                    LIMIT 24
                    """,
                    (sample_row.get("sample_id"),),
                )
    except Exception as exc:
        context["error"] = f"{type(exc).__name__}: {exc}"
    return context


def is_region_level_question(text: str) -> bool:
    if database_scope_question(text):
        return True
    return bool(
        re.search(
            r"(当前区域|这个区域|本区域|区域级|复核队列|哪些样本|哪些图|哪几张|最应该|优先复核|需要复核|待复核|前\d+个|全部|整体|总体|区域报告|所有样本|样本分布|统计|排名|top)",
            str(text or ""),
            flags=re.I,
        )
    )


def project_payload_scope(payload: dict[str, Any]) -> dict[str, Any]:
    question = str(payload.get("question") or "")
    if payload.get("sample") and not is_region_level_question(question):
        return {**payload, "singleSampleOnly": True, "reviewCandidates": []}
    return payload


def handle_judgement(payload: dict[str, Any]) -> dict[str, Any]:
    sample = payload.get("sample") or {}
    sample_id = str(sample.get("id") or sample.get("name") or sample.get("stem") or "")
    force = bool(payload.get("force"))
    ev_hash = evidence_hash(payload)
    if sample_id:
        db_upsert_sample(payload, ev_hash)
    if sample_id and not force:
        cached = db_get_judgement(sample_id, ev_hash, "sample")
        if cached:
            cached_engine = str(cached.get("engine") or "")
            if getattr(ARGS, "use_deepseek", False) and "deepseek" not in cached_engine.lower():
                cached = None
        if cached:
            verify_json = parse_json_maybe(cached.get("verify_json")) or {}
            return {
                "answer": cached.get("answer") or "",
                "engine": cached.get("engine") or "mysql cache",
                "cached": True,
                "persisted": True,
                "verifyStatus": cached.get("verify_status") or "pending",
                "verify": verify_json,
                "objectJudgement": verify_json.get("object_judgement") if isinstance(verify_json, dict) else None,
                "evidenceHash": ev_hash,
            }
    payload = {**payload, "singleSampleOnly": True, "reviewCandidates": []}
    raw_text = ""
    model_error = None
    try:
        if ARGS.use_qwen:
            judgement, engine, raw_text = qwen_object_judgement(payload)
        elif getattr(ARGS, "use_deepseek", False):
            judgement = rule_object_judgement(payload)
            raw_text = deepseek_text_answer(
                deepseek_judgement_prompt(payload, judgement, {"verifier_actions": []}),
                max_new_tokens=min(max(ARGS.max_new_tokens, 196), 320),
            )
            engine = f"DeepSeek text LLM: {ARGS.model_id}"
        else:
            judgement = rule_object_judgement(payload)
            engine = "local object-level evidence rules"
    except Exception as exc:
        traceback.print_exc()
        model_error = f"{type(exc).__name__}: {exc}"
        judgement = rule_object_judgement(payload)
        engine = f"local object rules after model error: {type(exc).__name__}"
        raw_text = str(exc)
    verify_status, verify_json = verify_object_judgement(sample, judgement, raw_text)
    if raw_text:
        verify_json["model_raw_excerpt"] = compact_free_text(raw_text, 900)
    object_judgement = verify_json.get("object_judgement") or judgement
    if getattr(ARGS, "use_deepseek", False) and not ARGS.use_qwen and not model_error:
        answer = normalize_judgement_answer(sample, raw_text, object_judgement, verify_json)
    else:
        answer = format_object_judgement_report(sample, object_judgement, verify_json)
    if model_error:
        verify_json["model_error"] = model_error
        answer = f"{answer}\n\n提示：本次模型不可用，已显示本地证据回退结果。"
    if sample_id and not model_error:
        db_save_judgement(sample_id, "sample", ev_hash, answer, engine, verify_status, verify_json)
    return {
        "answer": answer,
        "engine": engine,
        "cached": False,
        "persisted": mysql_enabled() and not model_error,
        "verifyStatus": verify_status,
        "verify": verify_json,
        "objectJudgement": object_judgement,
        "qwenError": bool(model_error),
        "error": model_error,
        "evidenceHash": ev_hash,
    }


def handle_correction(payload: dict[str, Any]) -> dict[str, Any]:
    ev_hash = evidence_hash(payload)
    db_save_correction(payload, ev_hash)
    return {
        "ok": True,
        "persisted": mysql_enabled(),
        "evidenceHash": ev_hash,
        "message": "manual correction saved" if mysql_enabled() else "manual correction accepted without MySQL persistence",
    }


def handle_review_override(payload: dict[str, Any]) -> dict[str, Any]:
    ev_hash = evidence_hash(payload)
    db_save_review_override(payload, ev_hash)
    region = payload.get("region") or {}
    sample = payload.get("sample") or {}
    region_id = str(payload.get("regionId") or region.get("id") or sample.get("regionId") or "").strip()
    return {
        "ok": True,
        "persisted": mysql_enabled(),
        "queued": bool(payload.get("queued")),
        "evidenceHash": ev_hash,
        "updatedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
        "regions": live_region_rows(region_id or None),
    }


def handle_review_complete(payload: dict[str, Any]) -> dict[str, Any]:
    ev_hash = evidence_hash(payload)
    db_complete_review(payload, ev_hash)
    region = payload.get("region") or {}
    sample = payload.get("sample") or {}
    region_id = str(payload.get("regionId") or region.get("id") or sample.get("regionId") or "").strip()
    return {
        "ok": True,
        "persisted": mysql_enabled(),
        "evidenceHash": ev_hash,
        "updatedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
        "regions": live_region_rows(region_id or None),
    }


def deploy_repo_root() -> Path:
    configured = str(getattr(ARGS, "repo_root", "") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    here = Path(__file__).resolve()
    parents = list(here.parents)
    if len(parents) >= 5:
        return parents[4]
    return Path.cwd().resolve()


def deploy_crop_script() -> Path:
    configured = Path(str(getattr(ARGS, "deploy_crop_script", "/media/dataset/crop_deploy_dataset.py"))).expanduser()
    if configured.exists():
        return configured
    repo_script = deploy_repo_root() / "tools" / "crop_deploy_dataset.py"
    if repo_script.exists():
        return repo_script
    return configured


def deploy_paths(payload: dict[str, Any]) -> dict[str, str]:
    paths = payload.get("paths") if isinstance(payload.get("paths"), dict) else {}
    return {
        "t1": str(paths.get("t1") or DEFAULT_DEPLOY_PATHS["t1"]).strip(),
        "t2": str(paths.get("t2") or DEFAULT_DEPLOY_PATHS["t2"]).strip(),
        "gt": str(paths.get("gt") or DEFAULT_DEPLOY_PATHS["gt"]).strip(),
    }


def handle_deploy_crop(payload: dict[str, Any]) -> dict[str, Any]:
    paths = deploy_paths(payload)
    if not paths["t1"] or not paths["t2"]:
        raise ValueError("裁剪需要填写 T1/T2 的 WSL 绝对路径。")
    for key in ("t1", "t2"):
        if not Path(paths[key]).exists():
            raise FileNotFoundError(f"{key} image not found: {paths[key]}")
    if paths["gt"] and not Path(paths["gt"]).exists():
        raise FileNotFoundError(f"mask image not found: {paths['gt']}")

    region_id = str(payload.get("regionId") or f"deploy_{int(time.time())}").strip()
    crop = payload.get("crop") if isinstance(payload.get("crop"), dict) else {}
    ratios = payload.get("ratios") if isinstance(payload.get("ratios"), dict) else {}
    training = payload.get("training") if isinstance(payload.get("training"), dict) else {}
    output_root = str(getattr(ARGS, "deploy_output_root", "/media/dataset/deploy_uploads"))
    script = deploy_crop_script()
    if not script.exists():
        raise FileNotFoundError(f"crop script not found: {script}")

    cmd = [
        str(getattr(ARGS, "training_python", "/home/halcyon/anaconda3/envs/amt/bin/python")),
        str(script),
        "--t1",
        paths["t1"],
        "--t2",
        paths["t2"],
        "--output_root",
        output_root,
        "--region_id",
        region_id,
        "--tile_size",
        str(crop.get("size") or 256),
        "--stride",
        str(crop.get("stride") or 256),
        "--train_ratio",
        str(ratios.get("train") or 70),
        "--val_ratio",
        str(ratios.get("val") or 15),
        "--test_ratio",
        str(ratios.get("test") or 15),
        "--dir_a",
        str(training.get("dir_a") or "time1"),
        "--dir_b",
        str(training.get("dir_b") or "time2"),
        "--dir_label",
        str(training.get("dir_label") or "label"),
        "--include_empty",
    ]
    if paths["gt"]:
        cmd.extend(["--mask", paths["gt"]])

    result = subprocess.run(
        cmd,
        cwd=str(deploy_repo_root()),
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"crop failed: {result.returncode}")

    output_dir = Path(output_root) / region_id
    metadata_path = output_dir / "metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    split_counts = metadata.get("split_counts") or {}
    split_stats = deployment_manifest_all_stats(str(output_dir))
    total_changed = sum(safe_int(v.get("clues"), 0) for v in split_stats.values()) if split_stats else safe_int(metadata.get("changed_tiles"), 0)
    total_review = sum(safe_int(v.get("review"), 0) for v in split_stats.values()) if split_stats else safe_int(metadata.get("review_tiles"), 0)
    total_changed_px = sum(safe_int(v.get("totalChangedPx"), 0) for v in split_stats.values()) if split_stats else safe_int(metadata.get("total_changed_px"), 0)
    test_stats = split_stats.get("test", {}) if split_stats else {}
    return {
        "ok": True,
        "status": "cropped",
        "command": cmd,
        "stdout": result.stdout[-4000:],
        "outputDir": str(output_dir),
        "metadataPath": str(metadata_path),
        "sourceSize": metadata.get("source_size", []),
        "totalTiles": metadata.get("total_tiles", 0),
        "changedTiles": total_changed,
        "reviewTiles": total_review,
        "totalChangedPx": total_changed_px,
        "testChangedTiles": safe_int(test_stats.get("clues"), 0),
        "testReviewTiles": safe_int(test_stats.get("review"), 0),
        "testChangedPx": safe_int(test_stats.get("totalChangedPx"), 0),
        "trainTiles": split_counts.get("train", 0),
        "valTiles": split_counts.get("val", 0),
        "testTiles": split_counts.get("test", 0),
        "splitStats": split_stats,
    }


def deploy_training_roots(payload: dict[str, Any]) -> dict[str, str]:
    training = payload.get("training") if isinstance(payload.get("training"), dict) else {}
    crop_output = str(payload.get("cropOutputDir") or "").strip()
    if crop_output:
        return {
            "file_root": crop_output,
            "train_root": str(Path(crop_output) / "train"),
            "val_root": str(Path(crop_output) / "test"),
        }
    return {
        "file_root": str(training.get("file_root") or "/mnt/h/dataset/LEVIR-MCI-dataset/images"),
        "train_root": str(training.get("train_root") or "/mnt/h/dataset/LEVIR-MCI-dataset/images/train"),
        "val_root": str(training.get("val_root") or "/mnt/h/dataset/LEVIR-MCI-dataset/images/test"),
    }


def handle_deploy_train(payload: dict[str, Any]) -> dict[str, Any]:
    training = payload.get("training") if isinstance(payload.get("training"), dict) else {}
    roots = deploy_training_roots(payload)
    for key in ("file_root", "train_root", "val_root"):
        if not Path(roots[key]).is_dir():
            raise FileNotFoundError(
                f"{key} not found: {roots[key]}. 请先在部署页完成裁剪，再启动本地训练。"
            )
    repo = deploy_repo_root()
    python_bin = str(getattr(ARGS, "training_python", "/home/halcyon/anaconda3/envs/amt/bin/python"))
    job_id = hashlib.sha1(f"{payload.get('regionId')}:{time.time()}".encode("utf-8")).hexdigest()[:12]
    job_dir = repo / "runs" / "deploy_jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / "train.log"
    cmd = [
        python_bin,
        "aggl/train_lcdnet_aegl.py",
        "--file_root",
        roots["file_root"],
        "--train_root",
        roots["train_root"],
        "--val_root",
        roots["val_root"],
        "--dir_a",
        str(training.get("dir_a") or "time1"),
        "--dir_b",
        str(training.get("dir_b") or "time2"),
        "--dir_label",
        str(training.get("dir_label") or "label"),
        "--inWidth",
        str(training.get("inWidth") or 256),
        "--inHeight",
        str(training.get("inHeight") or 256),
        "--batch_size",
        str(training.get("batch_size") or 16),
        "--num_workers",
        str(training.get("num_workers") or 4),
        "--lr",
        str(training.get("lr") or "2.5e-4"),
        "--max_steps",
        str(training.get("max_steps") or 80051),
        "--metric_gt_mode",
        str(training.get("metric_gt_mode") or "binary05"),
        "--savedir",
        str(training.get("savedir") or "runs/gzcd_deploy_lcdnet_rcrm_scr3_cdir"),
        "--backbone_width_mult",
        str(training.get("backbone_width_mult") or "1.0"),
        "--backbone_pretrained_path",
        str(training.get("backbone_pretrained_path") or "weights/mobilenet_v2-b0353104.pth"),
        "--use_rmsf",
        "--rmsf_mode",
        "scr3",
        "--use_cdir",
        "--use_crm",
        "--use_osmr",
        "--osmr_mode",
        "lite",
        "--osmr_stages",
        "l2,l1",
        "--cdir_gate_mode",
        "ones",
        "--cdir_disable_diff_head",
        "true",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"[deploy] job_id={job_id}\n")
        log_file.write(f"[deploy] cwd={repo}\n")
        log_file.write("[deploy] " + " ".join(cmd) + "\n")
        log_file.flush()
        process = subprocess.Popen(
            cmd,
            cwd=str(repo),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    DEPLOY_JOBS[job_id] = {
        "process": process,
        "pid": process.pid,
        "log_path": str(log_path),
        "command": cmd,
        "started_at": time.time(),
        "payload": payload,
    }
    return {
        "ok": True,
        "status": "submitted",
        "jobId": job_id,
        "pid": process.pid,
        "logPath": str(log_path),
        "command": cmd,
        "cwd": str(repo),
    }


def read_tail(path: str, max_chars: int = 6000) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    text = p.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def deployment_manifest_stats(crop_output_dir: str, split: str = "test") -> dict[str, int | str]:
    root = Path(str(crop_output_dir or ""))
    manifest_path = root / "manifest.jsonl"
    metadata_path = root / "metadata.json"
    if not manifest_path.exists():
        return {}
    tile_size = 256
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            tile_size = safe_int(metadata.get("tile_size"), 256)
        except Exception:
            tile_size = 256
    review_threshold = max(16, int(tile_size * tile_size * 0.005))
    samples = 0
    clues = 0
    review = 0
    total_changed_px = 0
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("split") or "") != split:
                continue
            samples += 1
            changed_px = safe_int(row.get("changed_px"), 0)
            total_changed_px += changed_px
            if changed_px > 0:
                clues += 1
            if 0 < changed_px < review_threshold:
                review += 1
    return {
        "samples": samples,
        "clues": clues,
        "review": review,
        "totalChangedPx": total_changed_px,
        "source": f"{split}_manifest",
    }


def deployment_manifest_all_stats(crop_output_dir: str) -> dict[str, Any]:
    root = Path(str(crop_output_dir or ""))
    manifest_path = root / "manifest.jsonl"
    metadata_path = root / "metadata.json"
    if not manifest_path.exists():
        return {}
    tile_size = 256
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            tile_size = safe_int(metadata.get("tile_size"), 256)
        except Exception:
            tile_size = 256
    review_threshold = max(16, int(tile_size * tile_size * 0.005))
    stats = {
        split: {"samples": 0, "clues": 0, "review": 0, "totalChangedPx": 0}
        for split in ("train", "val", "test")
    }
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            split = str(row.get("split") or "")
            if split not in stats:
                continue
            changed_px = safe_int(row.get("changed_px"), 0)
            stats[split]["samples"] += 1
            stats[split]["totalChangedPx"] += changed_px
            if changed_px > 0:
                stats[split]["clues"] += 1
            if 0 < changed_px < review_threshold:
                stats[split]["review"] += 1
    return stats


def deployment_stats_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    manifest_stats = deployment_manifest_stats(str(payload.get("cropOutputDir") or ""))
    if manifest_stats:
        return manifest_stats  # type: ignore[return-value]
    crop = payload.get("crop") if isinstance(payload.get("crop"), dict) else {}
    return {
        "samples": int(crop.get("test") or crop.get("tiles") or 0),
        "clues": int(payload.get("testChangedTiles") or payload.get("changedTiles") or crop.get("changedTiles") or 0),
        "review": int(payload.get("testReviewTiles") or payload.get("reviewTiles") or crop.get("reviewTiles") or 0),
        "source": "payload",
    }


def deploy_web_root() -> Path:
    if WEB_ROOT:
        return WEB_ROOT
    return Path(__file__).resolve().parents[1] / "web"


def deploy_asset_root(region_id: str) -> Path:
    clean = re.sub(r"[^0-9A-Za-z._-]+", "_", str(region_id or "deploy_region")).strip("_") or "deploy_region"
    return deploy_web_root() / "deployed_assets" / clean


def grid_position_from_bbox(bbox: list[Any], width: int = 256, height: int = 256) -> str:
    if len(bbox) != 4:
        return "无明显变化"
    x1, y1, x2, y2 = [float(v or 0) for v in bbox]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    col = "左" if cx < width / 3 else ("右" if cx > width * 2 / 3 else "中")
    row = "上" if cy < height / 3 else ("下" if cy > height * 2 / 3 else "中")
    if col == "中" and row == "中":
        return "中心"
    if col == "中":
        return f"{row}部"
    if row == "中":
        return f"{col}侧"
    return f"{col}{row}"


def deploy_event_type(payload: dict[str, Any]) -> str:
    event_key = str(payload.get("eventKey") or "").strip()
    return event_key or "deployed_change"


def deploy_event_label(payload: dict[str, Any]) -> str:
    event_name = str(payload.get("eventName") or "").strip()
    return event_name or "自定义变化"


def deploy_item_from_manifest(row: dict[str, Any], payload: dict[str, Any], asset_prefix: str) -> dict[str, Any]:
    objects = row.get("objects") if isinstance(row.get("objects"), list) else []
    normalized_objects: list[dict[str, Any]] = []
    for idx, obj in enumerate(objects):
        if not isinstance(obj, dict):
            continue
        bbox = obj.get("bbox") if isinstance(obj.get("bbox"), list) else []
        normalized_objects.append(
            {
                "id": f"obj_{idx + 1:03d}",
                "areaPx": safe_int(obj.get("area_px"), 0),
                "bbox": bbox,
                "center": [
                    round((float(bbox[0]) + float(bbox[2])) / 2.0, 1),
                    round((float(bbox[1]) + float(bbox[3])) / 2.0, 1),
                ]
                if len(bbox) == 4
                else [],
                "grid": grid_position_from_bbox(bbox),
                "shape": "connected_component",
            }
        )
    main_bbox = normalized_objects[0]["bbox"] if normalized_objects else []
    pred_px = safe_int(row.get("changed_px_pred"), 0)
    gt_px = safe_int(row.get("changed_px_gt"), 0)
    object_count = safe_int(row.get("object_count_pred"), len(normalized_objects))
    changed = pred_px > 0 or object_count > 0
    has_reference_mask = not bool(payload.get("noGt") or payload.get("sampleAppend"))
    mismatch = has_reference_mask and ((pred_px > 0) != (gt_px > 0))
    review_required = bool(changed and (mismatch or 0 < pred_px < 64 or object_count > 6))
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    assets = row.get("assets") if isinstance(row.get("assets"), dict) else {}
    def finite_metric(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def asset(name: str, fallback: str = "") -> str:
        value = str(assets.get(name) or fallback or "").strip().replace("\\", "/")
        if not value:
            return ""
        prefix = str(asset_prefix or "").strip().replace("\\", "/").strip("/")
        value = value.lstrip("./")
        if value.startswith(("http://", "https://", "data:", "blob:")):
            return value
        deployed_idx = value.lower().find("deployed_assets/")
        if deployed_idx >= 0:
            return value[deployed_idx:]
        if prefix and value.startswith(prefix + "/"):
            return value
        return f"{prefix}/{value}" if prefix else value

    stem = str(row.get("stem") or Path(str(row.get("name") or "")).stem or "sample")
    ratio = pred_px / float(256 * 256)
    location = grid_position_from_bbox(main_bbox)
    if not changed:
        scale_type = "no_obvious_change"
        scale_label = "无明显变化"
        scale = "none"
    elif pred_px < 128 or ratio < 0.002:
        scale_type = "weak_change"
        scale_label = "弱变化"
        scale = "weak"
    elif ratio >= 0.05:
        scale_type = "large_area_change"
        scale_label = "大区域变化"
        scale = "large"
    else:
        scale_type = "local_change"
        scale_label = "小区域变化"
        scale = "local"
    primary_event_type = deploy_event_type(payload)
    primary_event_label = deploy_event_label(payload)
    item_id = str(payload.get("sampleId") or f"{payload.get('regionId')}_{stem}")
    return {
        "id": item_id,
        "name": str(row.get("name") or f"{stem}.png"),
        "stem": stem,
        "areaName": str(payload.get("areaName") or payload.get("locationName") or "XX省"),
        "captureTime": str(payload.get("captureTime") or payload.get("sampleTime") or "未填写时间"),
        "addedAt": str(payload.get("addedAt") or ""),
        "isUserAdded": bool(payload.get("sampleAppend")),
        "pred": scale_type,
        "eventType": scale_type,
        "eventTypeZh": scale_label,
        "primaryEventType": primary_event_type,
        "primaryEventName": primary_event_label,
        "morphologyType": scale_type,
        "morphologyLabel": scale_label,
        "changedPx": pred_px,
        "changedRatio": ratio,
        "scale": scale,
        "objectCount": object_count,
        "mainBbox": main_bbox,
        "gridPosition": location,
        "confidence": 0.88 if changed and not review_required else (0.62 if review_required else 0.8),
        "reviewRequired": review_required,
        "reviewStatus": "建议复核" if review_required else "检测通过",
        "reviewReason": "变化范围质量规则触发复核，建议人工确认。" if mismatch else ("变化碎片较小或对象较多，建议抽检。" if review_required else "检测结果可作为巡检线索。"),
        "evidenceSummary": (
            f"系统检测到 {object_count} 个变化对象，主要位置为{location}，变化面积约 {pred_px:,} px。"
            if changed
            else "当前样本未检测到明显有效变化。"
        ),
        "report": (
            f"该样本检测到{primary_event_label}线索，变化规模为{scale_label}，主要位于{location}，面积约{pred_px:,} px。"
            if changed
            else "该样本未检测到明显有效变化。"
        ),
        "objects": normalized_objects,
        "assets": {
            "pre": asset("pre"),
            "post": asset("post"),
            "overlay": asset("pred_overlay"),
            "mask": asset("pred_mask"),
            "predOverlay": asset("pred_overlay"),
            "bboxOverlay": asset("bbox_overlay"),
        },
        "assetRootUrl": asset_prefix,
        "maskF1": finite_metric(metrics.get("F1")),
        "maskIoU": finite_metric(metrics.get("IoU")),
        "sourceSize": [256, 256],
        "qualityFlags": (
            [
                {
                    "name": "prediction_quality_review",
                    "code": "prediction_quality_review",
                    "text": "预测范围存在质量风险，建议人工复核。",
                    "level": "warn",
                }
            ]
            if mismatch
            else []
        ),
        "verifierChecks": [],
    }


def clean_name(value: Any, fallback: str = "sample") -> str:
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value or fallback)).strip("_")
    return text or fallback


def handle_deploy_preview_image(payload: dict[str, Any]) -> dict[str, Any]:
    data_url = str(payload.get("dataUrl") or "")
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    if not data_url:
        raise ValueError("dataUrl is required")
    try:
        raw = base64.b64decode(data_url, validate=False)
    except Exception as exc:
        raise ValueError(f"invalid image data: {exc}") from exc
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(f"Pillow is required to preview TIFF images: {exc}") from exc
    with Image.open(io.BytesIO(raw)) as img:
        img = img.convert("RGB")
        width, height = img.size
        img.thumbnail((900, 900))
        out = io.BytesIO()
        img.save(out, format="PNG")
    return {
        "ok": True,
        "width": width,
        "height": height,
        "dataUrl": "data:image/png;base64," + base64.b64encode(out.getvalue()).decode("ascii"),
    }


def add_sample_model_path(payload: dict[str, Any]) -> Path:
    repo = deploy_repo_root()
    requested = str(payload.get("modelPath") or "").strip()
    region_id = clean_name(payload.get("regionId"), "region")
    if requested:
        path = Path(requested).expanduser()
        if path.exists():
            return path
        if not path.is_absolute() and (repo / path).exists():
            return repo / path
        raise FileNotFoundError(f"modelPath not found: {requested}")

    deploy_root = repo / "runs" / "deploy_regions"
    exact = deploy_root / region_id / "best_model.pth"
    if exact.exists():
        return exact
    candidates = list(deploy_root.glob(f"{region_id}*/best_model.pth"))
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    fallback = Path(DEFAULT_ADD_SAMPLE_MODEL).expanduser()
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"modelPath not found for region: {region_id}")


def add_sample_paths(payload: dict[str, Any]) -> dict[str, str]:
    paths = payload.get("paths") if isinstance(payload.get("paths"), dict) else {}
    resolved = {
        "t1": str(paths.get("t1") or DEFAULT_ADD_SAMPLE_PATHS["t1"]).strip(),
        "t2": str(paths.get("t2") or DEFAULT_ADD_SAMPLE_PATHS["t2"]).strip(),
    }
    uploads = payload.get("uploads") if isinstance(payload.get("uploads"), dict) else {}
    if not uploads:
        return resolved

    region_id = clean_name(payload.get("regionId"), "region")
    sample_id = clean_name(payload.get("sampleId") or "added_sample", "added_sample")
    upload_root = Path("/media/dataset/deploy_added_samples") / "_raw_inputs" / region_id / sample_id
    upload_root.mkdir(parents=True, exist_ok=True)
    for key in ("t1", "t2"):
        blob = uploads.get(key)
        if not isinstance(blob, dict):
            continue
        data_url = str(blob.get("dataUrl") or "").strip()
        if not data_url:
            continue
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        filename = re.sub(r"[^0-9A-Za-z._-]+", "_", str(blob.get("filename") or f"{key}.bin")).strip("._") or f"{key}.bin"
        out_path = upload_root / f"{key}_{filename}"
        try:
            out_path.write_bytes(base64.b64decode(data_url, validate=False))
        except Exception as exc:
            raise ValueError(f"invalid upload for {key}: {exc}") from exc
        resolved[key] = str(out_path)
    return resolved


def make_added_sample_dataset(payload: dict[str, Any]) -> tuple[Path, str]:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(f"Pillow is required to prepare added sample: {exc}") from exc

    region_id = clean_name(payload.get("regionId"), "region")
    sample_id = clean_name(payload.get("sampleId") or f"added_{int(time.time())}", "added_sample")
    paths = add_sample_paths(payload)
    t1_path = Path(str(paths.get("t1") or DEFAULT_ADD_SAMPLE_PATHS["t1"]).strip()).expanduser()
    t2_path = Path(str(paths.get("t2") or DEFAULT_ADD_SAMPLE_PATHS["t2"]).strip()).expanduser()
    if not t1_path.exists():
        raise FileNotFoundError(f"T1 image not found: {t1_path}")
    if not t2_path.exists():
        raise FileNotFoundError(f"T2 image not found: {t2_path}")

    crop = payload.get("crop") if isinstance(payload.get("crop"), dict) else {}
    x = max(0, safe_int(crop.get("x"), 0))
    y = max(0, safe_int(crop.get("y"), 0))
    size = max(32, safe_int(crop.get("size"), 256))
    out_root = Path("/media/dataset/deploy_added_samples") / region_id / sample_id
    for sub in ("sample/time1", "sample/time2", "sample/label"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    name = sample_id + ".png"
    with Image.open(t1_path) as im1, Image.open(t2_path) as im2:
        im1 = im1.convert("RGB")
        im2 = im2.convert("RGB")
        width = min(im1.width, im2.width)
        height = min(im1.height, im2.height)
        x = min(x, max(0, width - 1))
        y = min(y, max(0, height - 1))
        x2 = min(width, x + size)
        y2 = min(height, y + size)
        if x2 <= x or y2 <= y:
            raise ValueError("invalid crop box for added sample")
        im1.crop((x, y, x2, y2)).resize((256, 256)).save(out_root / "sample" / "time1" / name)
        im2.crop((x, y, x2, y2)).resize((256, 256)).save(out_root / "sample" / "time2" / name)
    Image.new("L", (256, 256), 0).save(out_root / "sample" / "label" / name)

    metadata = {
        "region_id": region_id,
        "sample_id": sample_id,
        "source_t1": str(t1_path),
        "source_t2": str(t2_path),
        "crop_box": [x, y, x2, y2],
        "output_size": [256, 256],
        "split": "sample",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_root, name


def handle_deploy_add_sample(payload: dict[str, Any]) -> dict[str, Any]:
    region_id = clean_name(payload.get("regionId"), "region")
    sample_id = clean_name(payload.get("sampleId") or f"added_{int(time.time())}", "added_sample")
    added_at = str(payload.get("addedAt") or time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    payload = {**payload, "regionId": region_id, "sampleId": sample_id, "noGt": True, "sampleAppend": True, "addedAt": added_at}
    dataset_root, image_name = make_added_sample_dataset(payload)
    model_path = add_sample_model_path(payload)
    repo = deploy_repo_root()
    python_bin = str(getattr(ARGS, "training_python", "/home/halcyon/anaconda3/envs/amt/bin/python"))
    variant = "l2l1__gate_ones_disable_diff_head"
    asset_root = deploy_asset_root(region_id) / "added_samples" / sample_id
    metrics_path = asset_root / "eval_metrics.json"
    cmd = [
        python_bin,
        "tools/eval_lcdnet_deploy_assets.py",
        "--load",
        str(model_path),
        "--test_root",
        str(dataset_root),
        "--split",
        "sample",
        "--batchsize",
        "1",
        "--num_workers",
        "0",
        "--threshold",
        "0.5",
        "--metric_gt_mode",
        "binary05",
        "--dir_a",
        "time1",
        "--dir_b",
        "time2",
        "--dir_label",
        "label",
        "--backbone_width_mult",
        "1.0",
        "--backbone_pretrained_path",
        "weights/mobilenet_v2-b0353104.pth",
        "--use_rmsf",
        "--rmsf_mode",
        "scr3",
        "--use_cdir",
        "--use_crm",
        "--use_osmr",
        "--osmr_mode",
        "lite",
        "--osmr_stages",
        "l2,l1",
        "--variant_key",
        variant,
        "--deploy_osmr_stages",
        "l2,l1",
        "--deploy_cdir_simplified",
        "--cdir_gate_mode",
        "ones",
        "--cdir_disable_diff_head",
        "true",
        "--save_json",
        str(metrics_path),
        "--save_vis_dir",
        str(asset_root),
        "--save_vis_max",
        "1",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    asset_root.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        cmd,
        cwd=str(repo),
        env=env,
        text=True,
        capture_output=True,
        timeout=900,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout[-2000:] or f"add sample prediction failed: {result.returncode}")

    manifest = asset_root / variant / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"visual manifest not generated: {manifest}")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("added sample prediction produced no manifest rows")
    asset_prefix = f"deployed_assets/{deploy_asset_root(region_id).name}/added_samples/{sample_id}"
    item = deploy_item_from_manifest(rows[0], payload, asset_prefix)
    item["name"] = image_name
    item["stem"] = sample_id
    return {
        "ok": True,
        "status": "completed",
        "regionId": region_id,
        "sampleId": sample_id,
        "datasetRoot": str(dataset_root),
        "assetRoot": str(asset_root),
        "assetRootUrl": asset_prefix,
        "metricsPath": str(metrics_path),
        "modelPath": str(model_path),
        "stdout": result.stdout[-4000:],
        "item": item,
    }


def read_deploy_visual_manifest(region_id: str, payload: dict[str, Any], variant: str = "l2l1__gate_ones_disable_diff_head") -> tuple[list[dict[str, Any]], str]:
    asset_root = deploy_asset_root(region_id)
    manifest = asset_root / variant / "manifest.jsonl"
    if not manifest.exists():
        return [], ""
    asset_prefix = f"deployed_assets/{asset_root.name}"
    items = []
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            items.append(deploy_item_from_manifest(row, payload, asset_prefix))
    return items, f"{asset_prefix}/{variant}/manifest.jsonl"


def resolve_deploy_model_path(region_id: str, payload: dict[str, Any], repo: Path) -> Path:
    requested = Path(str(payload.get("modelPath") or repo / "runs" / "deploy_regions" / region_id / "best_model.pth"))
    if requested.exists():
        return requested
    deploy_root = repo / "runs" / "deploy_regions"
    candidates = list(deploy_root.glob(f"{region_id}*/best_model.pth"))
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]
    training = payload.get("training") if isinstance(payload.get("training"), dict) else {}
    savedir = str(training.get("savedir") or "").strip()
    if savedir:
        savedir_path = Path(savedir)
        if not savedir_path.is_absolute():
            savedir_path = repo / savedir_path
        candidates = list(savedir_path.parent.glob(f"{savedir_path.name}*/best_model.pth"))
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return candidates[0]
    raise FileNotFoundError(
        "trained model not found. Tried {} and {}*/best_model.pth".format(
            requested,
            deploy_root / region_id,
        )
    )


def handle_deploy_evaluate(payload: dict[str, Any]) -> dict[str, Any]:
    region_id = str(payload.get("regionId") or "").strip()
    if not region_id:
        raise ValueError("regionId is required")
    crop_output = str(payload.get("cropOutputDir") or "").strip()
    if not crop_output or not Path(crop_output).is_dir():
        raise FileNotFoundError(f"cropOutputDir not found: {crop_output}")
    training = payload.get("training") if isinstance(payload.get("training"), dict) else {}
    repo = deploy_repo_root()
    python_bin = str(getattr(ARGS, "training_python", "/home/halcyon/anaconda3/envs/amt/bin/python"))
    model_path = resolve_deploy_model_path(region_id, payload, repo)

    asset_root = deploy_asset_root(region_id)
    asset_root.mkdir(parents=True, exist_ok=True)
    job_id = hashlib.sha1(f"eval:{region_id}:{time.time()}".encode("utf-8")).hexdigest()[:12]
    job_dir = repo / "runs" / "deploy_jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / "evaluate.log"
    metrics_path = asset_root / "eval_metrics.json"
    variant = "l2l1__gate_ones_disable_diff_head"
    cmd = [
        python_bin,
        "tools/eval_lcdnet_deploy_assets.py",
        "--load",
        str(model_path),
        "--test_root",
        crop_output,
        "--split",
        "test",
        "--batchsize",
        "8",
        "--num_workers",
        "4",
        "--threshold",
        "0.5",
        "--metric_gt_mode",
        str(training.get("metric_gt_mode") or "binary05"),
        "--dir_a",
        str(training.get("dir_a") or "time1"),
        "--dir_b",
        str(training.get("dir_b") or "time2"),
        "--dir_label",
        str(training.get("dir_label") or "label"),
        "--backbone_width_mult",
        str(training.get("backbone_width_mult") or "1.0"),
        "--backbone_pretrained_path",
        str(training.get("backbone_pretrained_path") or "weights/mobilenet_v2-b0353104.pth"),
        "--use_rmsf",
        "--rmsf_mode",
        "scr3",
        "--use_cdir",
        "--use_crm",
        "--use_osmr",
        "--osmr_mode",
        "lite",
        "--osmr_stages",
        "l2,l1",
        "--variant_key",
        variant,
        "--deploy_osmr_stages",
        "l2,l1",
        "--deploy_cdir_simplified",
        "--cdir_gate_mode",
        "ones",
        "--cdir_disable_diff_head",
        "true",
        "--save_json",
        str(metrics_path),
        "--save_vis_dir",
        str(asset_root),
        "--save_vis_max",
        "999999",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"[eval] job_id={job_id}\n")
        log_file.write(f"[eval] cwd={repo}\n")
        log_file.write("[eval] " + " ".join(cmd) + "\n")
        log_file.flush()
        process = subprocess.Popen(
            cmd,
            cwd=str(repo),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    DEPLOY_JOBS[job_id] = {
        "kind": "evaluate",
        "process": process,
        "pid": process.pid,
        "log_path": str(log_path),
        "command": cmd,
        "started_at": time.time(),
        "payload": payload,
        "region_id": region_id,
        "asset_root": str(asset_root),
        "metrics_path": str(metrics_path),
        "variant": variant,
    }
    return {
        "ok": True,
        "status": "submitted",
        "jobId": job_id,
        "pid": process.pid,
        "logPath": str(log_path),
        "assetRoot": str(asset_root),
        "assetRootUrl": f"deployed_assets/{asset_root.name}",
        "metricsPath": str(metrics_path),
        "modelPath": str(model_path),
    }


def eval_job_payload(job: dict[str, Any], status: str) -> dict[str, Any]:
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    region_id = str(job.get("region_id") or payload.get("regionId") or "")
    variant = str(job.get("variant") or "l2l1__gate_ones_disable_diff_head")
    asset_root = Path(str(job.get("asset_root") or deploy_asset_root(region_id)))
    metrics_path = Path(str(job.get("metrics_path") or asset_root / "eval_metrics.json"))
    metrics: dict[str, Any] = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            metrics = {}
    items, manifest_url = read_deploy_visual_manifest(region_id, payload, variant)
    stats = {
        "samples": len(items),
        "clues": sum(1 for item in items if safe_int(item.get("changedPx"), 0) > 0),
        "review": sum(1 for item in items if item.get("reviewRequired")),
        "source": "eval_visual_manifest" if items else "eval_pending",
    }
    return {
        "assetRoot": str(asset_root),
        "assetRootUrl": f"deployed_assets/{asset_root.name}" if region_id else "",
        "assetManifestUrl": manifest_url,
        "metricsPath": str(metrics_path),
        "metrics": metrics,
        "items": items,
        "visualCount": len(items),
        "stats": stats,
    }


def handle_deploy_status(payload: dict[str, Any]) -> dict[str, Any]:
    job_id = str(payload.get("jobId") or "").strip()
    if not job_id:
        raise ValueError("jobId is required")
    job = DEPLOY_JOBS.get(job_id)
    if not job:
        return {
            "ok": False,
            "status": "unknown",
            "jobId": job_id,
            "message": "job not found in current API process",
        }
    process = job.get("process")
    returncode = process.poll() if process is not None else None
    status = "running" if returncode is None else ("completed" if int(returncode) == 0 else "failed")
    source_payload = job.get("payload") if isinstance(job.get("payload"), dict) else payload
    stats = deployment_stats_from_payload(source_payload)
    response = {
        "ok": True,
        "status": status,
        "jobId": job_id,
        "pid": job.get("pid"),
        "returnCode": returncode,
        "logPath": job.get("log_path"),
        "logTail": read_tail(str(job.get("log_path") or "")),
        "stats": stats,
        "elapsedSec": round(time.time() - float(job.get("started_at") or time.time()), 1),
    }
    if job.get("kind") == "evaluate":
        response.update(eval_job_payload(job, status))
    return response


SQL_AGENT_CONTEXT_TABLE_READY = False
SQL_AGENT_MAX_ROWS = 36
SQL_AGENT_MAX_PROMPT_ROWS = 12
SQL_AGENT_MAX_CELL_CHARS = 320
SQL_AGENT_SQL_MAX_NEW_TOKENS = 320
SQL_AGENT_ANSWER_MAX_NEW_TOKENS = 420
SQL_AGENT_COMPARE_PATTERN = re.compile(r"(和|跟).*(比|比较|对比)|区别|差异", re.I)
SQL_AGENT_FOLLOWUP_PATTERN = re.compile(r"(第[一二三四五六七八九十两0-9]+|刚才|上一|上个|前一个|那个|这一个|继续|再比较|再对比|和它比)", re.I)
SQL_AGENT_TOP_CHANGED_PATTERN = re.compile(r"(变化幅度最大|变化面积最大|最大变化|变化最大的图|面积最大的图)", re.I)
SQL_AGENT_TOP_REVIEW_PATTERN = re.compile(r"(优先复核|需要复核|待复核|复核队列|高风险复核)", re.I)
SQL_AGENT_GENERIC_BAD_ANSWER = re.compile(r"(无法回答|无法直接读取|请提供更多|无法理解图片|无法根据当前信息|请提供具体问题)", re.I)
SQL_AGENT_CHANGED_METRIC_PATTERN = re.compile(r"(像素|面积|变化像素|变化面积|变化范围|变化幅度|changed_px)", re.I)
SQL_AGENT_OBJECT_METRIC_PATTERN = re.compile(r"(对象数|对象数量|多少个对象|几个对象|对象最多|最多对象|目标最多|最多目标|object_count)", re.I)
SQL_AGENT_GENERIC_RANK_PATTERN = re.compile(r"(第\s*[0-9]{1,2}|第\s*[一二三四五六七八九十两]|最大|最小|最多|最少|排行|排名|前\s*[0-9]{1,2}|top\s*[0-9]{1,2}|最高|最低)", re.I)
SQL_AGENT_CHANGED_RANK_PATTERN = re.compile(r"(第\s*[0-9]{1,2}|第\s*[一二三四五六七八九十两]|最大|最小|排行|排名|前\s*[0-9]{1,2}|top\s*[0-9]{1,2}|最高|最低)", re.I)
SQL_AGENT_GAP_PATTERN = re.compile(r"(差距|相差|高多少|低多少|两者|相比|相较|比第|跟第|和第)", re.I)
SQL_AGENT_SAME_SAMPLE_PATTERN = re.compile(r"(是不是同一张|是否同一张|是不是同个样本|是否同个样本|是不是同一样本|是否同一样本)", re.I)
SQL_AGENT_SCHEMA_TEXT = """
可用的 MySQL 表与主字段：
1. regions
- region_id, short_name, region_name, role_text, status_text, scene_guide
- total_rows, display_rows, changed_rows, display_changed_rows
- review_rows, full_review_rows, avg_confidence
- counts_json, display_counts_json, metrics_json

2. samples
- sample_id, region_id, sample_name, sample_stem
- event_key, event_name
- changed_px, changed_ratio, object_count, grid_position
- review_status, review_required, review_reason, confidence
- pred_area, main_bbox_json, quality_flags_json
- report_text, display_policy, public_facts_json, private_metrics_json
- evidence_json, assets_json

3. sample_objects
- sample_id, region_id, object_index, object_key
- area_px, bbox_json, center_json, grid_position, shape_text, object_json

4. prediction_results
- prediction_id, sample_id, region_id, run_id
- changed_px, changed_ratio, object_count, grid_position
- event_key, event_name, review_required, review_reason, confidence
- main_bbox_json, public_facts_json, private_metrics_json

5. qa_sessions / qa_messages / qa_session_contexts
- qa_sessions.session_key, region_id, sample_id
- qa_messages.role, message, engine, created_at
- qa_session_contexts.memory_json 用于存放最近一次查询的样本顺序和引用记忆

关系：
- samples.region_id -> regions.region_id
- sample_objects.sample_id -> samples.sample_id
- prediction_results.sample_id -> samples.sample_id

常见查询建议：
- 单图事实或比较优先查 samples
- 对象级细节再查 sample_objects
- 区域统计查 samples + regions
- 列表类问题默认 LIMIT 12
""".strip()
SQL_AGENT_SQL_EXAMPLES = """
SQL 示例 1
问题：LEVIR对象数最多的是哪个图？
SQL：
SELECT sample_id, sample_stem, sample_name, event_key, event_name, changed_px, changed_ratio,
       object_count, grid_position, review_required, review_reason, confidence
FROM samples
WHERE region_id='levir'
ORDER BY COALESCE(object_count,0) DESC, COALESCE(changed_px,0) DESC, sample_id ASC
LIMIT 1

SQL 示例 2
问题：LEVIR前10个变化像素最高的图里，哪个对象最多？
SQL：
WITH top_changed AS (
  SELECT sample_id, sample_stem, sample_name, event_key, event_name, changed_px, changed_ratio,
         object_count, grid_position, review_required, review_reason, confidence
  FROM samples
  WHERE region_id='levir'
  ORDER BY COALESCE(changed_px,0) DESC, sample_id ASC
  LIMIT 10
)
SELECT *
FROM top_changed
ORDER BY COALESCE(object_count,0) DESC, COALESCE(changed_px,0) DESC, sample_id ASC
LIMIT 1

SQL 示例 3
问题：LEVIR变化面积最大的10张图里，对象数最多的是哪张？
SQL：
WITH top_changed AS (
  SELECT sample_id, sample_stem, sample_name, event_key, event_name, changed_px, changed_ratio,
         object_count, grid_position, review_required, review_reason, confidence
  FROM samples
  WHERE region_id='levir'
  ORDER BY COALESCE(changed_px,0) DESC, sample_id ASC
  LIMIT 10
)
SELECT *
FROM top_changed
ORDER BY COALESCE(object_count,0) DESC, COALESCE(changed_px,0) DESC, sample_id ASC
LIMIT 1

SQL 示例 4
问题：LEVIR第二大的图和对象数最多的图是不是同一张？
SQL：
WITH second_changed AS (
  SELECT sample_id, sample_stem, sample_name, event_key, event_name, changed_px, changed_ratio,
         object_count, grid_position, review_required, review_reason, confidence
  FROM samples
  WHERE region_id='levir'
  ORDER BY COALESCE(changed_px,0) DESC, sample_id ASC
  LIMIT 1 OFFSET 1
),
top_objects AS (
  SELECT sample_id, sample_stem, sample_name, event_key, event_name, changed_px, changed_ratio,
         object_count, grid_position, review_required, review_reason, confidence
  FROM samples
  WHERE region_id='levir'
  ORDER BY COALESCE(object_count,0) DESC, COALESCE(changed_px,0) DESC, sample_id ASC
  LIMIT 1
)
SELECT * FROM second_changed
UNION ALL
SELECT * FROM top_objects
""".strip()


def sql_agent_enabled() -> bool:
    return bool(mysql_enabled() or sql_agent_static_enabled())


def sql_agent_static_enabled() -> bool:
    return bool(getattr(ARGS, "static_sql", False) and Path(getattr(ARGS, "web_data", "")).is_file())


def sql_agent_fetch_all(query: str, params=()) -> list[dict[str, Any]]:
    if mysql_enabled():
        return db_fetch_all(query, params)
    if sql_agent_static_enabled():
        return static_store(ARGS.web_data).execute(query, params)["rows"]
    return []


def sql_agent_requested_filter(question: str) -> str:
    if re.search(r"无需复核|不需要复核|不用复核|非待复核|排除待复核", question):
        return "not_review"
    if re.search(r"非高风险|排除高风险|不含高风险", question):
        return "not_high"
    if re.search(r"非中风险|排除中风险|不含中风险", question):
        return "not_medium"
    if "高风险" in question and "中风险" not in question:
        return "high"
    if "中风险" in question and "高风险" not in question:
        return "medium"
    if re.search(r"待复核|需要复核|优先复核|高风险|中风险|复核队列", question):
        return "review"
    return ""


def sql_agent_filter_sql(filter_kind: str) -> str:
    if filter_kind in {"not_high", "not_medium"}:
        level = filter_kind.removeprefix("not_")
        return f" AND {sql_agent_review_level_expr()} <> '{level}' "
    if filter_kind in {"high", "medium"}:
        return f" AND {sql_agent_review_level_expr()} = '{filter_kind}' "
    if filter_kind == "review":
        return f" AND ({sql_agent_review_level_expr()} IN ('high','medium') OR COALESCE(review_required,0)=1) "
    if filter_kind == "not_review":
        return f" AND ({sql_agent_review_level_expr()} NOT IN ('high','medium') AND COALESCE(review_required,0)=0) "
    return " "


SQL_NUMERIC_CONDITION = re.compile(r"(变化像素|变化面积|对象数量|对象数|目标数|面积)\s*(大于等于|小于等于|不少于|不小于|不超过|不大于|至少|至多|超过|大于|小于|低于|等于|>=|<=|>|<|=|≥|≤)\s*([0-9]+(?:\.[0-9]+)?)")


def sql_agent_intent_question(question: str) -> str:
    matches = list(SQL_NUMERIC_CONDITION.finditer(question))
    intent = SQL_NUMERIC_CONDITION.sub("", question)
    if matches and not re.search(r"变化像素|变化面积|对象数量|对象数|目标数|面积", intent):
        intent = matches[0].group(1) + " " + intent
    return intent


def sql_agent_scope_sql(question: str) -> str:
    result = sql_agent_filter_sql(sql_agent_requested_filter(question))
    operators = {"大于等于": ">=", "不少于": ">=", "不小于": ">=", "至少": ">=", "≥": ">=",
                 "小于等于": "<=", "不超过": "<=", "不大于": "<=", "至多": "<=", "≤": "<=",
                 "超过": ">", "大于": ">", "小于": "<", "低于": "<", "等于": "="}
    for match in SQL_NUMERIC_CONDITION.finditer(question):
        field = "object_count" if match.group(1) in {"对象数量", "对象数", "目标数"} else "changed_px"
        op = operators.get(match.group(2), match.group(2))
        if re.match(r"\s*(平方米|平方公里|公顷|m²)", question[match.end():]):
            raise ValueError("当前数据仅有像素面积，不能直接按地面面积筛选。")
        result += f" AND COALESCE({field},0) {op} {match.group(3)} "
    return result


def sql_agent_validate_query(query: str) -> None:
    # Constrain this user-facing query tool; admin and persistence code use
    # separate functions and retain their original capabilities.
    clean = re.sub(r"'(?:''|\\.|[^'])*'|\"(?:\\.|[^\"])*\"", "''", query)
    clean = re.sub(r"/\*.*?\*/|--[^\n]*|#[^\n]*", " ", clean, flags=re.S).replace('`', '')
    if re.search(r"\b(update|delete|insert|drop|alter|create|attach|pragma|call|grant|truncate|into|outfile|dumpfile|sleep|benchmark|load_file|get_lock)\b", clean, re.I):
        raise ValueError("查询工具仅允许读取公共变化事实。")
    if re.search(r"\b(private_metrics_json|metrics_json|password_hash|password_salt|assets_json)\b", clean, re.I):
        raise ValueError("实验私有指标、认证字段和原始资产字段不能用于用户问答。")
    allowed = {"samples", "regions", "sample_objects", "prediction_results"}
    aliases = set(re.findall(r"(?:\bwith|,)\s*([a-z_]\w*)\s+as\s*\(", clean, re.I))
    tokens = re.findall(r"[a-z_]\w*|[(),.]", clean.lower())
    tables = []
    depth = 0
    from_depths = set()
    for index, token in enumerate(tokens):
        if token == '(':
            depth += 1
            continue
        if token == ')':
            from_depths.discard(depth)
            depth -= 1
            continue
        if token in {'where', 'group', 'order', 'having', 'limit', 'union', 'except', 'intersect'}:
            from_depths.discard(depth)
        reference = token in {'from', 'join'} or (token == ',' and depth in from_depths)
        if token == 'from':
            from_depths.add(depth)
        if reference and index + 1 < len(tokens) and tokens[index + 1] != '(':
            tables.append(tokens[index + 1])
            if index + 2 < len(tokens) and tokens[index + 2] == '.':
                raise ValueError("查询工具不允许跨数据库访问。")
    if not tables or any(table.lower() not in allowed | {x.lower() for x in aliases} for table in tables):
        raise ValueError("查询工具只能访问已允许的公共事实表。")


def ensure_sql_agent_context_table() -> None:
    global SQL_AGENT_CONTEXT_TABLE_READY
    if SQL_AGENT_CONTEXT_TABLE_READY or not mysql_enabled():
        return
    try:
        conn = mysql_connect()
    except Exception as exc:
        print(f"sql_agent_context_connect_error={type(exc).__name__}: {exc}")
        return
    if conn is None:
        return
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS qa_session_contexts (
              session_key varchar(191) COLLATE utf8mb4_unicode_ci NOT NULL,
              memory_json longtext COLLATE utf8mb4_unicode_ci NOT NULL,
              created_at timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
              updated_at timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
              PRIMARY KEY (session_key)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
        )
        cur.close()
        SQL_AGENT_CONTEXT_TABLE_READY = True
    except Exception as exc:
        print(f"sql_agent_context_create_error={type(exc).__name__}: {exc}")
    finally:
        conn.close()


def db_recent_session_messages(session_key: str, limit: int = 8) -> list[dict[str, Any]]:
    if transient_session_enabled(session_key=session_key):
        safe_limit = max(1, min(int(limit or 8), 12))
        return list(TRANSIENT_QA_MESSAGES.get(session_key) or [])[-safe_limit:]
    if not mysql_enabled() or not session_key:
        return []
    safe_limit = max(1, min(int(limit or 8), 12))
    rows = db_fetch_all(
        f"""
        SELECT m.role, m.message, m.engine, m.created_at
        FROM qa_messages m
        JOIN qa_sessions s ON s.id = m.session_id
        WHERE s.session_key=%s
        ORDER BY m.id DESC
        LIMIT {safe_limit}
        """,
        (session_key,),
    )
    rows.reverse()
    return rows


def db_load_session_context(session_key: str) -> dict[str, Any]:
    if transient_session_enabled(session_key=session_key):
        return dict(TRANSIENT_SESSION_CONTEXTS.get(session_key) or {})
    ensure_sql_agent_context_table()
    if not mysql_enabled() or not session_key:
        return {}
    row = db_fetch_one(
        "SELECT memory_json FROM qa_session_contexts WHERE session_key=%s LIMIT 1",
        (session_key,),
    )
    if not row:
        return {}
    return parse_json_maybe(row.get("memory_json")) or {}


def db_save_session_context(session_key: str, memory: dict[str, Any]) -> None:
    if transient_session_enabled(session_key=session_key):
        TRANSIENT_SESSION_CONTEXTS[session_key] = dict(memory or {})
        return
    ensure_sql_agent_context_table()
    if not mysql_enabled() or not session_key:
        return
    db_execute(
        """
        INSERT INTO qa_session_contexts (session_key, memory_json)
        VALUES (%s,%s)
        ON DUPLICATE KEY UPDATE
          memory_json=VALUES(memory_json),
          updated_at=CURRENT_TIMESTAMP
        """,
        (session_key, canonical_json(memory or {})),
    )


def sql_agent_normalize_key(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", str(text or "").lower())


def sql_agent_question_ordinal(text: str) -> int | None:
    raw = str(text or "")
    match = re.search(r"第\s*([0-9]{1,2}|十[一二三四五六七八九]?|[一二三四五六七八九十两])", raw)
    if not match:
        return None
    token = match.group(1)
    if token.isdigit():
        return int(token)
    mapping = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
        "十一": 11,
        "十二": 12,
        "十三": 13,
        "十四": 14,
        "十五": 15,
    }
    return mapping.get(token)


def sql_agent_question_limit(text: str, default: int = 8) -> int:
    raw = str(text or "")
    match = re.search(r"(前|top)\s*([0-9]{1,2})", raw, flags=re.I)
    if match:
        return max(1, min(int(match.group(2)), 12))
    subset_limit = sql_agent_question_subset_limit(raw)
    if subset_limit:
        return subset_limit
    ordinal = sql_agent_question_ordinal(raw)
    if ordinal:
        return max(1, min(int(ordinal), 12))
    return max(1, min(int(default), 12))


def sql_agent_question_subset_limit(text: str) -> int | None:
    raw = str(text or "")
    patterns = (
        r"前\s*([0-9]{1,2})\s*(?:个|张)?[^。？，,；;]{0,24}(?:里|中|当中|里面)",
        r"(?:最大|最高|面积最大|变化面积最大|变化像素最大|变化范围最大)[^0-9]{0,8}([0-9]{1,2})\s*(?:张图|个样本|张样本|个图|条样本)[^。？，,；;]{0,6}(?:里|中|当中|里面)",
        r"([0-9]{1,2})\s*(?:张图|个样本|张样本|个图|条样本)[^。？，,；;]{0,6}(?:里|中|当中|里面)",
    )
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.I)
        if match:
            return max(1, min(int(match.group(1)), 12))
    return None


def sql_agent_has_explicit_topk(text: str) -> bool:
    return bool(re.search(r"(前|top)\s*([0-9]{1,2})", str(text or ""), flags=re.I))


def sql_agent_question_metrics(text: str) -> list[str]:
    raw = str(text or "")
    metrics: list[str] = []
    implied_changed_rank = bool(re.search(r"(第\s*[0-9]{1,2}|第\s*[一二三四五六七八九十两]).{0,3}大(?:的)?图", raw))
    if SQL_AGENT_CHANGED_METRIC_PATTERN.search(raw) or SQL_AGENT_TOP_CHANGED_PATTERN.search(raw) or implied_changed_rank:
        metrics.append("changed_px")
    if SQL_AGENT_OBJECT_METRIC_PATTERN.search(raw):
        metrics.append("object_count")
    if SQL_AGENT_TOP_REVIEW_PATTERN.search(raw) or re.search(r"(复核|风险|高风险|中风险)", raw):
        metrics.append("review_level")
    return metrics


def sql_agent_is_changed_rank_question(question: str) -> bool:
    raw = str(question or "")
    implied_changed_rank = bool(re.search(r"(第\s*[0-9]{1,2}|第\s*[一二三四五六七八九十两]).{0,3}大(?:的)?图", raw))
    return bool(
        SQL_AGENT_TOP_CHANGED_PATTERN.search(raw)
        or (SQL_AGENT_CHANGED_METRIC_PATTERN.search(raw) and SQL_AGENT_CHANGED_RANK_PATTERN.search(raw))
        or implied_changed_rank
    )


def sql_agent_is_object_rank_question(question: str) -> bool:
    raw = str(question or "")
    return bool(SQL_AGENT_OBJECT_METRIC_PATTERN.search(raw) and SQL_AGENT_GENERIC_RANK_PATTERN.search(raw))


def sql_agent_is_compound_subset_question(question: str) -> bool:
    raw = str(question or "")
    metrics = sql_agent_question_metrics(raw)
    has_changed = "changed_px" in metrics
    has_object = "object_count" in metrics
    nested_subset = bool(sql_agent_question_subset_limit(raw))
    scoped_words = bool(re.search(r"(里|中|当中|里面)", raw))
    staged_words = bool(re.search(r"(按|先|再|筛|排序|排名)", raw))
    if has_changed and has_object and (nested_subset or scoped_words or staged_words):
        return True
    return bool(
        re.search(
            r"(前\s*[0-9]{1,2}|top\s*[0-9]{1,2}|第\s*[0-9]{1,2}|第\s*[一二三四五六七八九十两]).{0,18}(里|中|当中|里面).{0,18}(对象|复核|风险|面积|像素)",
            raw,
            flags=re.I,
        )
    )


def sql_agent_is_metric_rank_followup(question: str, session_memory: dict[str, Any] | None = None) -> bool:
    raw = str(question or "")
    if sql_agent_is_changed_rank_question(raw) or sql_agent_is_object_rank_question(raw):
        return True
    resolver = (session_memory or {}).get("last_rank_resolver") or {}
    return bool(
        SQL_AGENT_FOLLOWUP_PATTERN.search(raw)
        and str(resolver.get("kind") or "").strip() in {"changed_px_desc", "object_count_desc"}
    )


def sql_agent_is_strict_db_question(question: str) -> bool:
    raw = str(question or "")
    return bool(
        re.search(
            r"(哪张图|哪个图|哪个样本|第\s*[0-9]{1,2}|第\s*[一二三四五六七八九十两]|前\s*[0-9]{1,2}|top\s*[0-9]{1,2}|最多|最少|差多少|相差|是不是同一张|是否同一张|排名|排序)",
            raw,
            flags=re.I,
        )
    )


def sql_agent_question_hints(question: str, resolved_refs: dict[str, Any]) -> dict[str, Any]:
    raw = str(question or "")
    return {
        "metrics": sql_agent_question_metrics(raw),
        "ordinal": sql_agent_question_ordinal(raw),
        "list_limit": sql_agent_question_limit(raw, 8),
        "subset_limit": sql_agent_question_subset_limit(raw),
        "has_explicit_topk": sql_agent_has_explicit_topk(raw),
        "is_changed_rank": sql_agent_is_changed_rank_question(raw),
        "is_object_rank": sql_agent_is_object_rank_question(raw),
        "is_compound_subset": sql_agent_is_compound_subset_question(raw),
        "asks_same_sample": bool(SQL_AGENT_SAME_SAMPLE_PATTERN.search(raw)),
        "resolved_references_present": bool(
            resolved_refs.get("referenced_sample_id") or resolved_refs.get("comparison_sample_ids")
        ),
    }


def sql_agent_is_changed_rank_followup(question: str, session_memory: dict[str, Any] | None = None) -> bool:
    raw = str(question or "")
    if sql_agent_is_changed_rank_question(raw):
        return True
    resolver = (session_memory or {}).get("last_rank_resolver") or {}
    return bool(
        SQL_AGENT_FOLLOWUP_PATTERN.search(raw)
        and str(resolver.get("kind") or "").strip() == "changed_px_desc"
    )


def sql_agent_extract_sample_ids(rows: list[dict[str, Any]]) -> list[str]:
    sample_ids: list[str] = []
    for row in rows:
        sample_id = str(row.get("sample_id") or row.get("sampleId") or "").strip()
        if sample_id and sample_id not in sample_ids:
            sample_ids.append(sample_id)
    return sample_ids


def sql_agent_compact_messages(messages: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for row in messages[-limit:]:
        compacted.append(
            {
                "role": row.get("role"),
                "message": compact_free_text(str(row.get("message") or ""), 180),
                "engine": compact_free_text(str(row.get("engine") or ""), 80),
            }
        )
    return compacted


def sql_agent_infer_rank_resolver(question: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    filter_question = str(payload.get("_scopeQuestion") or question)
    question = sql_agent_intent_question(question)
    region = payload.get("region") or {}
    region_id = str(region.get("id") or "").strip()
    if not region_id:
        return None
    if sql_agent_is_compound_subset_question(question):
        return None
    if sql_agent_is_changed_rank_question(question):
        return {"kind": "changed_px_desc", "region_id": region_id, "review_filter": sql_agent_requested_filter(filter_question), "filter_question": filter_question}
    if sql_agent_is_object_rank_question(question):
        return {"kind": "object_count_desc", "region_id": region_id, "review_filter": sql_agent_requested_filter(filter_question), "filter_question": filter_question}
    if SQL_AGENT_TOP_REVIEW_PATTERN.search(question):
        return {"kind": "review_priority", "region_id": region_id, "review_filter": sql_agent_requested_filter(filter_question), "filter_question": filter_question}
    return None


def sql_agent_ranked_ids_by_resolver(resolver: dict[str, Any] | None, limit: int = 12) -> list[str]:
    if not resolver:
        return []
    region_id = str(resolver.get("region_id") or "").strip()
    if not region_id:
        return []
    safe_limit = max(1, min(int(limit or 12), 20))
    kind = str(resolver.get("kind") or "")
    if kind == "changed_px_desc":
        rows = db_fetch_all(
            f"""
            SELECT sample_id
            FROM samples
            WHERE region_id=%s
            ORDER BY COALESCE(changed_px,0) DESC, sample_id ASC
            LIMIT {safe_limit}
            """,
            (region_id,),
        )
        return [str(row.get("sample_id") or "") for row in rows if row.get("sample_id")]
    if kind == "object_count_desc":
        rows = db_fetch_all(
            f"""
            SELECT sample_id
            FROM samples
            WHERE region_id=%s
            ORDER BY COALESCE(object_count,0) DESC, COALESCE(changed_px,0) DESC, sample_id ASC
            LIMIT {safe_limit}
            """,
            (region_id,),
        )
        return [str(row.get("sample_id") or "") for row in rows if row.get("sample_id")]
    if kind == "review_priority":
        rows = db_fetch_all(
            f"""
            SELECT sample_id
            FROM samples
            WHERE region_id=%s AND COALESCE(review_required,0)=1
            ORDER BY COALESCE(changed_px,0) DESC, sample_id ASC
            LIMIT {safe_limit}
            """,
            (region_id,),
        )
        sample_ids = [str(row.get("sample_id") or "") for row in rows if row.get("sample_id")]
        if sample_ids:
            return sample_ids
        rows = db_fetch_all(
            f"""
            SELECT sample_id
            FROM samples
            WHERE region_id=%s
            ORDER BY COALESCE(changed_px,0) DESC, sample_id ASC
            LIMIT {safe_limit}
            """,
            (region_id,),
        )
        return [str(row.get("sample_id") or "") for row in rows if row.get("sample_id")]
    return []


def sql_agent_resolve_references(
    question: str,
    payload: dict[str, Any],
    session_memory: dict[str, Any],
    recent_messages: list[dict[str, Any]],
) -> dict[str, Any]:
    sample = payload.get("sample") or {}
    region = payload.get("region") or {}
    current_sample_id = str(sample.get("id") or sample.get("sampleId") or sample.get("name") or sample.get("stem") or "").strip()
    region_id = str(region.get("id") or sample.get("regionId") or "").strip()
    resolved: dict[str, Any] = {
        "current_sample_id": current_sample_id,
        "region_id": region_id,
        "recent_messages": sql_agent_compact_messages(recent_messages),
    }
    previous_ids = list(session_memory.get("last_ranked_sample_ids") or [])
    ordinal_followup = sql_agent_question_ordinal(question)
    if ordinal_followup and "last_ranked_sample_ids" in session_memory and not SQL_AGENT_CHANGED_METRIC_PATTERN.search(question) and not SQL_AGENT_OBJECT_METRIC_PATTERN.search(question):
        if ordinal_followup > len(previous_ids):
            raise ValueError(f"上次列表只有 {len(previous_ids)} 个样本，没有第 {ordinal_followup} 个")
        target = previous_ids[ordinal_followup - 1]
        resolved.update(ranked_sample_ids=previous_ids, referenced_sample_id=target, ordinal=ordinal_followup,
                        rank_resolver=session_memory.get("last_rank_resolver"))
        if current_sample_id and SQL_AGENT_COMPARE_PATTERN.search(question):
            resolved["comparison_sample_ids"] = [current_sample_id, target]
        return resolved
    rank_resolver = sql_agent_infer_rank_resolver(question, payload) or session_memory.get("last_rank_resolver")
    ranked_ids = list(session_memory.get("last_ranked_sample_ids") or [])
    limit = sql_agent_question_limit(question, 8)
    if not ranked_ids and rank_resolver:
        ranked_ids = sql_agent_ranked_ids_by_resolver(rank_resolver, max(limit, 12))
    if ranked_ids:
        resolved["ranked_sample_ids"] = ranked_ids[:12]
    if rank_resolver:
        resolved["rank_resolver"] = rank_resolver

    rank_question = sql_agent_is_metric_rank_followup(question, session_memory)
    gap_question = bool(SQL_AGENT_GAP_PATTERN.search(question) or SQL_AGENT_COMPARE_PATTERN.search(question))

    if rank_question and rank_resolver:
        target_ids = sql_agent_ranked_ids_by_resolver(rank_resolver, 12)
        if target_ids:
            resolved["ranked_sample_ids"] = target_ids[:12]
        ordinal = sql_agent_question_ordinal(question)
        if ordinal and ordinal <= len(target_ids):
            resolved["referenced_sample_id"] = target_ids[ordinal - 1]
            resolved["ordinal"] = ordinal
            if gap_question and ordinal > 1:
                resolved["comparison_sample_ids"] = [target_ids[0], target_ids[ordinal - 1]]
        elif re.search(r"下一|另一个|另外一个", question) and current_sample_id and target_ids and target_ids[0] == current_sample_id and len(target_ids) > 1:
            resolved["referenced_sample_id"] = target_ids[1]

    ordinal = sql_agent_question_ordinal(question)
    if ordinal and not resolved.get("referenced_sample_id"):
        ranked = list(resolved.get("ranked_sample_ids") or [])
        if ordinal <= len(ranked):
            resolved["referenced_sample_id"] = ranked[ordinal - 1]
            resolved["ordinal"] = ordinal

    if (
        current_sample_id
        and resolved.get("referenced_sample_id")
        and SQL_AGENT_COMPARE_PATTERN.search(question)
    ):
        resolved["comparison_sample_ids"] = [current_sample_id, str(resolved["referenced_sample_id"])]

    return resolved


def sql_agent_build_sql_prompt(
    payload: dict[str, Any],
    session_memory: dict[str, Any],
    recent_messages: list[dict[str, Any]],
    resolved_refs: dict[str, Any],
) -> str:
    region = payload.get("region") or {}
    sample = payload.get("sample") or {}
    compact = {
        "question": payload.get("question"),
        "session_key": db_session_key(payload),
        "current_region": {
            "region_id": region.get("id"),
            "short": region.get("short"),
            "name": region.get("name"),
            "role": region.get("role"),
        },
        "current_sample": {
            "sample_id": sample.get("id") or sample.get("name") or sample.get("stem"),
            "sample_stem": sample.get("stem") or sample.get("name"),
            "changed_px": sample.get("changedPx"),
            "changed_ratio": sample.get("changedRatio"),
            "object_count": sample.get("objectCount"),
            "grid_position": sample.get("gridPosition"),
            "review_required": sample.get("reviewRequired"),
            "review_reason": sample.get("reviewReason"),
        },
        "recent_messages": sql_agent_compact_messages(recent_messages),
        "session_memory": session_memory or {},
        "resolved_references": resolved_refs,
    }
    return (
        "你是本地 MySQL 8 查询规划助手，只负责生成一条可执行 SQL。\n"
        "要求：\n"
        "1. 只能输出一条 SELECT 或 WITH ... SELECT 查询，不要输出解释，不要输出 markdown。\n"
        "2. 优先使用 samples；对象级细节再用 sample_objects；区域统计可用 regions。\n"
        "3. 列表类问题默认 LIMIT 12；如果用户明确要前 N 个，N 不超过 12。\n"
        "4. 如果当前样本 sample_id 已给出，涉及“这张图/当前样本”时优先按该 sample_id 查询。\n"
        "5. 如果 resolved_references 里已经给出 referenced_sample_id 或 comparison_sample_ids，直接使用，不要重新猜。\n"
        "6. 比较问题优先返回两个样本的同一组字段：sample_id, sample_stem, event_name, changed_px, changed_ratio, object_count, grid_position, review_required, review_reason, confidence。\n"
        "7. 不要写 UPDATE/DELETE/INSERT/ALTER/DROP。\n\n"
        f"{SQL_AGENT_SCHEMA_TEXT}\n\n"
        "上下文 JSON：\n"
        + json.dumps(compact, ensure_ascii=False, indent=2, default=str)
        + "\n\n现在只输出 SQL："
    )


def sql_agent_build_sql_prompt_v2(
    payload: dict[str, Any],
    session_memory: dict[str, Any],
    recent_messages: list[dict[str, Any]],
    resolved_refs: dict[str, Any],
) -> str:
    region = payload.get("region") or {}
    sample = payload.get("sample") or {}
    hints = sql_agent_question_hints(str(payload.get("question") or ""), resolved_refs)
    compact = {
        "question": payload.get("question"),
        "session_key": db_session_key(payload),
        "current_region": {
            "region_id": region.get("id"),
            "short": region.get("short"),
            "name": region.get("name"),
            "role": region.get("role"),
        },
        "current_sample": {
            "sample_id": sample.get("id") or sample.get("name") or sample.get("stem"),
            "sample_stem": sample.get("stem") or sample.get("name"),
            "changed_px": sample.get("changedPx"),
            "changed_ratio": sample.get("changedRatio"),
            "object_count": sample.get("objectCount"),
            "grid_position": sample.get("gridPosition"),
            "review_required": sample.get("reviewRequired"),
            "review_reason": sample.get("reviewReason"),
        },
        "recent_messages": sql_agent_compact_messages(recent_messages),
        "session_memory": session_memory or {},
        "resolved_references": resolved_refs,
        "question_hints": hints,
    }
    return (
        "你是本地 MySQL 8 查询规划助手，只负责生成一条可执行 SQL。\n"
        "要求：\n"
        "1. 只能输出一条 SELECT 或 WITH ... SELECT 查询，不要输出解释，不要输出 markdown。\n"
        "2. 优先使用 samples；对象级细节再用 sample_objects；区域统计可用 regions。\n"
        "3. 列表类问题默认 LIMIT 12；如果用户明确要前 N 个，N 不超过 12。\n"
        "4. 如果当前样本 sample_id 已给出，涉及“这张图/当前样本”时优先按该 sample_id 查询。\n"
        "5. 如果 resolved_references 里已经给出 referenced_sample_id 或 comparison_sample_ids，直接使用，不要重新猜。\n"
        "6. 如果用户在问变化像素、变化面积、变化范围的第 i 高、第 i 大、排名第 i、前 N 名，默认按 samples.changed_px 排序：ORDER BY COALESCE(changed_px,0) DESC, sample_id ASC。\n"
        "7. 如果用户在问对象数最多、对象数量排名、前 N 个对象最多的样本，默认按 samples.object_count 排序：ORDER BY COALESCE(object_count,0) DESC, COALESCE(changed_px,0) DESC, sample_id ASC。\n"
        "8. 如果 resolved_references.ordinal=i 已给出，并且 referenced_sample_id 为空，可以使用 LIMIT 1 OFFSET i-1；如果 referenced_sample_id 已给出，就直接按 sample_id 查询，不要重算。\n"
        "9. 组合问题优先拆成两步：先用子查询或 CTE 筛出子集，再在子集内按第二指标排序。例如“前10个变化面积最大的图里，哪个对象最多”。\n"
        "10. 如果问题在问“哪个图/哪张图/哪个样本”，必须返回样本行，不要返回 event_name 聚合计数，不要把 counts_json 里的事件统计当成对象数。\n"
        "11. 比较问题优先返回两个样本的同一组字段：sample_id, sample_stem, event_name, changed_px, changed_ratio, object_count, grid_position, review_required, review_reason, confidence。\n"
        "12. 不要写 UPDATE/DELETE/INSERT/ALTER/DROP。\n\n"
        f"{SQL_AGENT_SCHEMA_TEXT}\n\n"
        "参考 SQL 示例：\n"
        f"{SQL_AGENT_SQL_EXAMPLES}\n\n"
        "上下文 JSON：\n"
        + json.dumps(compact, ensure_ascii=False, indent=2, default=str)
        + "\n\n现在只输出 SQL："
    )


def sql_agent_retry_prompt_v2(
    payload: dict[str, Any],
    previous_sql: str,
    error_text: str,
    resolved_refs: dict[str, Any],
) -> str:
    hints = sql_agent_question_hints(str(payload.get("question") or ""), resolved_refs)
    compact = {
        "question": payload.get("question"),
        "previous_sql": previous_sql,
        "error": error_text,
        "resolved_references": resolved_refs,
        "question_hints": hints,
    }
    return (
        "上一条 SQL 执行失败。请只输出一条修正后的 MySQL 8 SELECT 查询，不要解释。\n"
        "仍然只能使用 SELECT 或 WITH ... SELECT。\n"
        "如果问题在问变化像素/变化面积的第 i 高、第 i 大、前 N 名，请优先按 samples.changed_px 降序排序。\n"
        "如果问题在问对象数最多、对象数量排名，请优先按 samples.object_count 降序排序，并用 changed_px 作为第二排序键。\n"
        "如果是组合问题，请用 CTE 或子查询先筛子集，再在子集内排序。\n"
        "如果已经给出 referenced_sample_id、comparison_sample_ids、ordinal 或 subset_limit，就直接利用这些引用，不要再输出自然语言。\n"
        "不要返回 event_name 聚合计数来回答“哪个图/哪张图/哪个样本”。\n"
        f"参考 SQL 示例：\n{SQL_AGENT_SQL_EXAMPLES}\n"
        + json.dumps(compact, ensure_ascii=False, indent=2, default=str)
        + "\nSQL："
    )


def sql_agent_extract_sql(raw_text: str) -> str:
    cleaned = strip_deepseek_reasoning(str(raw_text or "")).strip()
    block = re.search(r"```(?:sql)?\s*([\s\S]+?)```", cleaned, flags=re.I)
    if block:
        cleaned = block.group(1).strip()
    cleaned = re.sub(r"^\s*SQL\s*[:：]\s*", "", cleaned, flags=re.I)
    match = re.search(r"(?is)\b(with|select)\b[\s\S]*", cleaned)
    sql = match.group(0).strip() if match else cleaned.strip("` \n\r\t")
    if ";" in sql:
        sql = sql.split(";")[0].strip() + ";"
    return sql.strip()


def sql_agent_normalize_cell(value: Any) -> Any:
    if value is None or isinstance(value, (int, float, bool, str)):
        if isinstance(value, str) and len(value) > SQL_AGENT_MAX_CELL_CHARS:
            return value[: SQL_AGENT_MAX_CELL_CHARS - 3] + "..."
        return value
    if isinstance(value, bytes):
        text = value.decode("utf-8", "ignore")
        return text[: SQL_AGENT_MAX_CELL_CHARS - 3] + "..." if len(text) > SQL_AGENT_MAX_CELL_CHARS else text
    if hasattr(value, "isoformat"):
        return value.isoformat(sep=" ")
    text = str(value)
    parsed = parse_json_maybe(text)
    if parsed is not None and text[:1] in "{[":
        return parsed
    return text[: SQL_AGENT_MAX_CELL_CHARS - 3] + "..." if len(text) > SQL_AGENT_MAX_CELL_CHARS else text


def sql_agent_execute_sql(sql: str, max_rows: int = SQL_AGENT_MAX_ROWS) -> dict[str, Any]:
    query = sql_agent_extract_sql(sql)
    lowered = query.lstrip().lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise ValueError("only SELECT/CTE queries are supported")
    if query.count(";") > 1:
        raise ValueError("only a single SQL statement is supported")
    sql_agent_validate_query(query)
    if not mysql_enabled() and sql_agent_static_enabled():
        return static_store(ARGS.web_data).execute(query, max_rows=max_rows)
    conn = mysql_connect()
    if conn is None:
        raise RuntimeError("mysql connection is not available")
    try:
        cur = conn.cursor()
        cur.execute(query)
        if cur.description is None:
            raise ValueError("query did not return rows")
        columns = [desc[0] for desc in cur.description or []]
        rows = cur.fetchmany(max_rows + 1)
        truncated = len(rows) > max_rows
        if truncated:
            rows = rows[:max_rows]
        normalized_rows: list[dict[str, Any]] = []
        for row in rows:
            if isinstance(row, dict):
                normalized_rows.append({key: sql_agent_normalize_cell(value) for key, value in row.items()})
            else:
                normalized_rows.append(
                    {key: sql_agent_normalize_cell(value) for key, value in zip(columns, row)}
                )
        cur.close()
        return {
            "sql": query,
            "columns": columns,
            "rows": normalized_rows,
            "row_count": len(normalized_rows),
            "truncated": truncated,
        }
    finally:
        conn.close()


def sql_agent_retry_prompt(
    payload: dict[str, Any],
    previous_sql: str,
    error_text: str,
    resolved_refs: dict[str, Any],
) -> str:
    compact = {
        "question": payload.get("question"),
        "previous_sql": previous_sql,
        "error": error_text,
        "resolved_references": resolved_refs,
    }
    return (
        "上一条 SQL 执行失败。请只输出一条修正后的 MySQL 8 SELECT 查询，不要解释。\n"
        "仍然只能使用 SELECT 或 WITH ... SELECT。\n"
        + json.dumps(compact, ensure_ascii=False, indent=2, default=str)
        + "\nSQL："
    )


def sql_agent_task_type(question: str, rows: list[dict[str, Any]], resolved_refs: dict[str, Any]) -> str:
    sample_ids = sql_agent_extract_sample_ids(rows)
    if resolved_refs.get("comparison_sample_ids") or (
        SQL_AGENT_COMPARE_PATTERN.search(question) and len(sample_ids) >= 2
    ):
        return "compare_samples"
    if len(sample_ids) >= 3:
        return "ranked_samples"
    if len(sample_ids) == 1:
        return "single_sample"
    return "aggregate"


def sql_agent_build_answer_prompt(
    payload: dict[str, Any],
    session_memory: dict[str, Any],
    recent_messages: list[dict[str, Any]],
    resolved_refs: dict[str, Any],
    query_result: dict[str, Any],
) -> str:
    rows = query_result.get("rows") or []
    task_type = sql_agent_task_type(str(payload.get("question") or ""), rows, resolved_refs)
    hints = sql_agent_question_hints(str(payload.get("question") or ""), resolved_refs)
    compact = {
        "question": payload.get("question"),
        "task_type": task_type,
        "question_hints": hints,
        "current_sample_id": resolved_refs.get("current_sample_id"),
        "resolved_references": resolved_refs,
        "recent_messages": sql_agent_compact_messages(recent_messages),
        "session_memory": session_memory or {},
        "executed_sql": query_result.get("sql"),
        "query_result": rows[:SQL_AGENT_MAX_PROMPT_ROWS],
        "truncated": bool(query_result.get("truncated")),
    }
    return (
        "你是遥感变化智能巡检平台的中文数据库问答助手，只能根据下面给出的 MySQL 查询结果回答。\n"
        "不要说自己看到了图像，不要编造数据库没有返回的数字，不要输出 SQL。\n"
        "回答要求：\n"
        "- task_type=ranked_samples：用 1. 2. 3. 列表列出样本，写 sample_stem、changed_px、object_count、简短复核原因。\n"
        "- task_type=compare_samples：先写一句结论，再写“当前样本”和“对比样本”的面积、对象数、位置、复核状态差异。\n"
        "- task_type=single_sample：按“结论 / 位置与数量 / 复核建议”三段写。\n"
        "- task_type=aggregate：直接回答统计结论。\n"
        "- 如果 question_hints.is_compound_subset=true，优先明确“先按第一指标筛子集，再在子集内按第二指标排序”的结论，不要把结果误说成第一指标的冠军样本。\n"
        "- 如果问题在问“哪个图/哪张图/哪个样本”，答案里必须出现 sample_id 或 sample_stem，不要只回答事件名。\n"
        "- 如果问题在问“是否同一张”，先明确回答“是同一张”或“不是同一张”。\n"
        "- 如果数据库无结果，就明确说“本地数据库未查到匹配记录”。\n"
        "- 严格复制 rows 里的 sample_stem、changed_px、object_count、grid_position、review_reason。\n\n"
        "输入 JSON：\n"
        + json.dumps(compact, ensure_ascii=False, indent=2, default=str)
        + "\n\n现在给出最终中文回答："
    )


def sql_agent_fallback_answer(
    payload: dict[str, Any],
    query_result: dict[str, Any],
    resolved_refs: dict[str, Any],
) -> str:
    question = str(payload.get("question") or "")
    rows = list(query_result.get("rows") or [])
    if not rows:
        return "本地数据库未查到匹配记录。"
    task_type = sql_agent_task_type(question, rows, resolved_refs)
    if task_type == "ranked_samples":
        lines = ["结论：本地数据库查询到以下优先样本："]
        for idx, row in enumerate(rows[: min(len(rows), 12)], 1):
            lines.append(
                f"{idx}. {row.get('sample_stem') or row.get('sample_id')}："
                f"变化面积 {fmt(row.get('changed_px') or 0)}px，"
                f"对象 {row.get('object_count') or 0} 个，"
                f"{row.get('review_reason') or ('建议复核' if row.get('review_required') else '常规抽检')}"
            )
        return "\n".join(lines)
    if task_type == "compare_samples" and len(rows) >= 2:
        row_a, row_b = rows[0], rows[1]
        current_sample_id = str(resolved_refs.get("current_sample_id") or "")
        if current_sample_id and str(row_b.get("sample_id") or "") == current_sample_id and str(row_a.get("sample_id") or "") != current_sample_id:
            row_a, row_b = row_b, row_a
        return (
            f"结论：当前样本 {row_a.get('sample_stem') or row_a.get('sample_id')} 与对比样本 "
            f"{row_b.get('sample_stem') or row_b.get('sample_id')} 相比，"
            f"变化面积分别为 {fmt(row_a.get('changed_px') or 0)}px 和 {fmt(row_b.get('changed_px') or 0)}px，"
            f"对象数分别为 {row_a.get('object_count') or 0} 个和 {row_b.get('object_count') or 0} 个。\n"
            f"当前样本位置：{row_a.get('grid_position') or '未知'}；对比样本位置：{row_b.get('grid_position') or '未知'}。\n"
            f"当前样本复核提示：{row_a.get('review_reason') or ('建议复核' if row_a.get('review_required') else '常规抽检')}；"
            f"对比样本复核提示：{row_b.get('review_reason') or ('建议复核' if row_b.get('review_required') else '常规抽检')}。"
        )
    row = rows[0]
    return (
        f"结论：样本 {row.get('sample_stem') or row.get('sample_id')} 的主要变化类型为 {row.get('event_name') or '未分类'}。\n"
        f"位置与数量：变化面积 {fmt(row.get('changed_px') or 0)}px，对象 {row.get('object_count') or 0} 个，主要位置 {row.get('grid_position') or '未知'}。\n"
        f"复核建议：{row.get('review_reason') or ('建议复核' if row.get('review_required') else '可按常规抽检查看')}。"
    )


def sql_agent_metric_rank_answer(
    payload: dict[str, Any],
    query_result: dict[str, Any],
    resolved_refs: dict[str, Any],
) -> str | None:
    question = str(payload.get("question") or "")
    rank_resolver = resolved_refs.get("rank_resolver") or {}
    resolver_kind = str(rank_resolver.get("kind") or "").strip()
    if resolver_kind not in {"changed_px_desc", "object_count_desc"}:
        return None
    if sql_agent_is_compound_subset_question(question):
        return None
    if not (
        sql_agent_is_changed_rank_question(question)
        or sql_agent_is_object_rank_question(question)
        or resolved_refs.get("ordinal")
        or SQL_AGENT_GAP_PATTERN.search(question)
        or SQL_AGENT_COMPARE_PATTERN.search(question)
    ):
        return None
    rows = list(query_result.get("rows") or [])
    if not rows:
        return "本地数据库未查到匹配记录。"
    region = payload.get("region") or {}
    region_label = str(region.get("short") or region.get("name") or region.get("id") or "当前区域")
    ordinal = int(resolved_refs.get("ordinal") or 0) if str(resolved_refs.get("ordinal") or "").isdigit() else 0
    explicit_rank_list = re.search(r"(前|top)\s*([0-9]{1,2})", question, flags=re.I)
    metric_field = "changed_px" if resolver_kind == "changed_px_desc" else "object_count"
    metric_label = "变化像素" if resolver_kind == "changed_px_desc" else "对象数量"
    secondary_field = "object_count" if resolver_kind == "changed_px_desc" else "changed_px"
    secondary_label = "对象数量" if resolver_kind == "changed_px_desc" else "变化像素"
    max_label = "最大的样本" if resolver_kind == "changed_px_desc" else "对象数最多的样本"
    ordinal_label = f"第 {ordinal} 大" if resolver_kind == "changed_px_desc" else f"第 {ordinal} 多"

    if explicit_rank_list:
        lines = [f"结论：{region_label} 区域按{metric_label}从高到低的结果如下："]
        for idx, row in enumerate(rows, 1):
            lines.append(
                f"{idx}. {row.get('sample_id') or row.get('sample_stem')}："
                f"{metric_label} {fmt(row.get(metric_field) or 0)}"
                f"{'px' if metric_field == 'changed_px' else ' 个'}，"
                f"{secondary_label} {fmt(row.get(secondary_field) or 0)}"
                f"{' 个' if secondary_field == 'object_count' else 'px'}，"
                f"主要位置 {row.get('grid_position') or '未知'}。"
            )
        return "\n".join(lines)

    if len(rows) >= 2 and (SQL_AGENT_GAP_PATTERN.search(question) or SQL_AGENT_COMPARE_PATTERN.search(question)):
        row_a, row_b = rows[0], rows[1]
        diff_metric = abs(int(db_number(row_a.get(metric_field), 0)) - int(db_number(row_b.get(metric_field), 0)))
        rank_text = ordinal_label if ordinal > 0 else "对比样本"
        return (
            f"结论：{region_label} 区域{max_label}是 {row_a.get('sample_id') or row_a.get('sample_stem')}，"
            f"{metric_label} {fmt(row_a.get(metric_field) or 0)}"
            f"{'px' if metric_field == 'changed_px' else ' 个'}；{rank_text}的样本是 "
            f"{row_b.get('sample_id') or row_b.get('sample_stem')}，{metric_label} {fmt(row_b.get(metric_field) or 0)}"
            f"{'px' if metric_field == 'changed_px' else ' 个'}。\n"
            f"两者相差 {fmt(diff_metric)}{'px' if metric_field == 'changed_px' else ' 个'}。对象数量分别为 {row_a.get('object_count') or 0} 个和 {row_b.get('object_count') or 0} 个，"
            f"主要位置分别为 {row_a.get('grid_position') or '未知'} 和 {row_b.get('grid_position') or '未知'}。"
        )

    row = rows[0]
    if ordinal > 0:
        return (
            f"结论：{region_label} 区域{ordinal_label}的样本是 "
            f"{row.get('sample_id') or row.get('sample_stem')}，{metric_label} {fmt(row.get(metric_field) or 0)}"
            f"{'px' if metric_field == 'changed_px' else ' 个'}。\n"
            f"变化像素 {fmt(row.get('changed_px') or 0)}px，对象数量 {row.get('object_count') or 0} 个，主要位置 {row.get('grid_position') or '未知'}。"
        )

    text = (
        f"结论：{region_label} 区域{max_label}是 "
        f"{row.get('sample_id') or row.get('sample_stem')}，{metric_label} {fmt(row.get(metric_field) or 0)}"
        f"{'px' if metric_field == 'changed_px' else ' 个'}。\n"
        f"变化像素 {fmt(row.get('changed_px') or 0)}px，对象数量 {row.get('object_count') or 0} 个，主要位置 {row.get('grid_position') or '未知'}。"
    )
    if len(rows) >= 2:
        second = rows[1]
        diff_metric = abs(int(db_number(row.get(metric_field), 0)) - int(db_number(second.get(metric_field), 0)))
        text += (
            f"\n排名第二的是 {second.get('sample_id') or second.get('sample_stem')}，"
            f"{metric_label} {fmt(second.get(metric_field) or 0)}"
            f"{'px' if metric_field == 'changed_px' else ' 个'}，两者相差 {fmt(diff_metric)}"
            f"{'px' if metric_field == 'changed_px' else ' 个'}。"
        )
    return text


def sql_agent_compound_subset_answer(
    payload: dict[str, Any],
    query_result: dict[str, Any],
    resolved_refs: dict[str, Any],
) -> str | None:
    question = str(payload.get("question") or "")
    if not sql_agent_is_compound_subset_question(question):
        return None
    rows = list(query_result.get("rows") or [])
    if not rows:
        return "本地数据库未查到匹配记录。"
    row = rows[0]
    region = payload.get("region") or {}
    region_label = str(region.get("short") or region.get("name") or region.get("id") or "当前区域")
    subset_limit = sql_agent_question_subset_limit(question) or sql_agent_question_limit(question, 10)
    has_explicit_topk = sql_agent_has_explicit_topk(question) or bool(sql_agent_question_subset_limit(question))
    metrics = sql_agent_question_metrics(question)
    if "changed_px" in metrics and "object_count" in metrics:
        scope_text = (
            f"按变化像素排序的前 {subset_limit} 个样本里"
            if has_explicit_topk
            else "按变化像素排序后"
        )
        return (
            f"结论：{region_label} 区域{scope_text}，对象数量最多的是 "
            f"{row.get('sample_id') or row.get('sample_stem')}。\n"
            f"对象数量 {row.get('object_count') or 0} 个，变化像素 {fmt(row.get('changed_px') or 0)}px，"
            f"主要位置 {row.get('grid_position') or '未知'}。"
        )
    return None


def sql_agent_same_sample_answer(
    payload: dict[str, Any],
    query_result: dict[str, Any],
) -> str | None:
    question = str(payload.get("question") or "")
    if not SQL_AGENT_SAME_SAMPLE_PATTERN.search(question):
        return None
    rows = list(query_result.get("rows") or [])
    if len(rows) < 2:
        return None
    row_a, row_b = rows[0], rows[1]
    sample_a = str(row_a.get("sample_id") or row_a.get("sample_stem") or "")
    sample_b = str(row_b.get("sample_id") or row_b.get("sample_stem") or "")
    same = sample_a and sample_a == sample_b
    if SQL_AGENT_CHANGED_METRIC_PATTERN.search(question) and SQL_AGENT_OBJECT_METRIC_PATTERN.search(question):
        return (
            f"结论：{'是' if same else '不是'}同一张。\n"
            f"按本次数据库查询，变化像素排序中的目标样本是 {sample_a or '未知样本'}，"
            f"对象数量排序中的目标样本是 {sample_b or '未知样本'}，因此两者"
            f"{'一致' if same else '不一致'}。"
        )
    return (
        f"结论：{'是' if same else '不是'}同一张。\n"
        f"本次查询返回的两个候选样本分别是 {sample_a or '未知样本'} 和 {sample_b or '未知样本'}。"
    )


def sql_agent_finalize_answer(
    payload: dict[str, Any],
    answer: str,
    query_result: dict[str, Any],
    resolved_refs: dict[str, Any],
) -> str:
    text = sanitize_user_answer(str(answer or "").strip())
    rows = list(query_result.get("rows") or [])
    task_type = sql_agent_task_type(str(payload.get("question") or ""), rows, resolved_refs)
    deterministic_compare_answer = sql_agent_same_sample_answer(payload, query_result)
    if deterministic_compare_answer:
        return deterministic_compare_answer
    deterministic_compound_answer = sql_agent_compound_subset_answer(payload, query_result, resolved_refs)
    if deterministic_compound_answer:
        return deterministic_compound_answer
    deterministic_rank_answer = sql_agent_metric_rank_answer(payload, query_result, resolved_refs)
    if deterministic_rank_answer:
        return deterministic_rank_answer
    if not text or SQL_AGENT_GENERIC_BAD_ANSWER.search(text):
        text = sql_agent_fallback_answer(payload, query_result, resolved_refs)
    if task_type == "single_sample":
        row = rows[0] if rows else {}
        required_parts = []
        if row.get("changed_px") is not None and str(int(float(row.get("changed_px") or 0))) not in text:
            required_parts.append(f"总变化面积 {fmt(row.get('changed_px') or 0)}px")
        if row.get("object_count") is not None and str(row.get("object_count")) not in text:
            required_parts.append(f"对象数量 {row.get('object_count')} 个")
        if row.get("grid_position") and str(row.get("grid_position")) not in text:
            required_parts.append(f"主要位置 {row.get('grid_position')}")
        if required_parts:
            text += "\n\n数据库核对：" + "；".join(required_parts) + "。"
    return text


def sql_agent_build_session_memory(
    payload: dict[str, Any],
    session_memory: dict[str, Any],
    resolved_refs: dict[str, Any],
    query_result: dict[str, Any],
) -> dict[str, Any]:
    memory = dict(session_memory or {})
    question = str(payload.get("question") or "")
    region = payload.get("region") or {}
    sample = payload.get("sample") or {}
    rows = list(query_result.get("rows") or [])
    task_type = sql_agent_task_type(question, rows, resolved_refs)
    ranked_ids = sql_agent_extract_sample_ids(rows)
    memory.update(
        {
            "last_question": question,
            "last_sql": query_result.get("sql"),
            "last_query_task": task_type,
            "last_result_row_count": query_result.get("row_count"),
            "last_region_id": region.get("id"),
            "current_sample_id": sample.get("id") or sample.get("name") or sample.get("stem"),
            "last_result_preview": rows[:6],
            "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
        }
    )
    if task_type == "ranked_samples" or (sql_agent_infer_rank_resolver(question, payload) and not sql_agent_question_ordinal(question)):
        ranked_memory_ids = [str(v or "").strip() for v in (resolved_refs.get("ranked_sample_ids") or ranked_ids) if str(v or "").strip()]
        memory["last_ranked_sample_ids"] = ranked_memory_ids[:20]
    if task_type == "compare_samples" and resolved_refs.get("comparison_sample_ids"):
        memory["last_compare_sample_ids"] = list(resolved_refs.get("comparison_sample_ids") or [])
    rank_resolver = sql_agent_infer_rank_resolver(question, payload) or resolved_refs.get("rank_resolver")
    if rank_resolver:
        memory["last_rank_resolver"] = rank_resolver
    return memory


def sql_agent_review_level_expr(table_alias: str = "") -> str:
    prefix = f"{table_alias}." if table_alias else ""
    return (
        "COALESCE("
        f"NULLIF(JSON_UNQUOTE(JSON_EXTRACT({prefix}public_facts_json, '$.review_level')), ''), "
        f"CASE WHEN COALESCE({prefix}review_required,0)=1 THEN 'medium' ELSE 'none' END"
        ")"
    )


def sql_agent_review_order_expr(table_alias: str = "") -> str:
    prefix = f"{table_alias}." if table_alias else ""
    level_expr = sql_agent_review_level_expr(table_alias)
    return (
        f"CASE {level_expr} "
        "WHEN 'high' THEN 4 "
        "WHEN 'medium' THEN 3 "
        "WHEN 'low' THEN 2 "
        "WHEN 'none' THEN 1 "
        "ELSE 0 END DESC, "
        f"COALESCE({prefix}changed_ratio,0) DESC, "
        f"COALESCE({prefix}changed_px,0) DESC, "
        f"{prefix}sample_id ASC"
    )


def sql_agent_quote(value: Any) -> str:
    raw = str(value or "")
    return "'" + raw.replace("\\", "\\\\").replace("'", "''") + "'"


def sql_agent_core_sample_select(table_alias: str = "") -> str:
    prefix = f"{table_alias}." if table_alias else ""
    level_expr = sql_agent_review_level_expr(table_alias)
    return (
        f"{prefix}sample_id, "
        f"{prefix}sample_stem, "
        f"{prefix}sample_name, "
        f"{prefix}event_key, "
        f"{prefix}event_name, "
        f"{prefix}changed_px, "
        f"{prefix}changed_ratio, "
        f"{prefix}object_count, "
        f"{prefix}grid_position, "
        f"{prefix}review_required, "
        f"{prefix}review_reason, "
        f"{prefix}confidence, "
        f"{level_expr} AS review_level"
    )


def sql_agent_metric_order_expr(metric: str, table_alias: str = "") -> str:
    prefix = f"{table_alias}." if table_alias else ""
    if metric == "object_count":
        return (
            f"COALESCE({prefix}object_count,0) DESC, "
            f"COALESCE({prefix}changed_px,0) DESC, "
            f"{prefix}sample_id ASC"
        )
    return f"COALESCE({prefix}changed_px,0) DESC, {prefix}sample_id ASC"


def sql_agent_template_query(payload: dict[str, Any], resolved_refs: dict[str, Any]) -> str | None:
    question = str(payload.get("question") or "").strip()
    sample = payload.get("sample") or {}
    region = payload.get("region") or {}
    region_id = str(resolved_refs.get("region_id") or region.get("id") or sample.get("regionId") or "").strip()
    current_sample_id = str(
        resolved_refs.get("current_sample_id")
        or sample.get("id")
        or sample.get("sampleId")
        or sample.get("name")
        or sample.get("stem")
        or ""
    ).strip()
    referenced_sample_id = str(resolved_refs.get("referenced_sample_id") or "").strip()
    comparison_sample_ids = [str(v or "").strip() for v in (resolved_refs.get("comparison_sample_ids") or []) if str(v or "").strip()]
    safe_limit = max(1, min(sql_agent_question_limit(question, 8), 12))
    scope_filter = sql_agent_scope_sql(str(payload.get("_scopeQuestion") or question))
    question = sql_agent_intent_question(question)
    fields = sql_agent_core_sample_select()
    review_level_expr = sql_agent_review_level_expr()
    review_order_expr = sql_agent_review_order_expr()
    metrics = sql_agent_question_metrics(question)
    has_changed_metric = "changed_px" in metrics
    has_object_metric = "object_count" in metrics
    rank_question = sql_agent_is_changed_rank_question(question)
    object_rank_question = sql_agent_is_object_rank_question(question)
    compound_subset_question = sql_agent_is_compound_subset_question(question)
    explicit_topk = sql_agent_has_explicit_topk(question)
    subset_limit = sql_agent_question_subset_limit(question)
    ordinal = int(resolved_refs.get("ordinal") or 0) if str(resolved_refs.get("ordinal") or "").isdigit() else (sql_agent_question_ordinal(question) or 0)
    same_sample_compare = bool(SQL_AGENT_SAME_SAMPLE_PATTERN.search(question))

    if region_id and has_changed_metric and has_object_metric:
        if same_sample_compare or (SQL_AGENT_COMPARE_PATTERN.search(question) and ordinal > 0):
            changed_offset = max(0, (ordinal or 1) - 1)
            return (
                f"WITH changed_pick AS ("
                f"SELECT {fields} "
                f"FROM samples "
                f"WHERE region_id={sql_agent_quote(region_id)} {scope_filter}"
                f"ORDER BY {sql_agent_metric_order_expr('changed_px')} "
                f"LIMIT 1 OFFSET {changed_offset}"
                f"), top_objects AS ("
                f"SELECT {fields} "
                f"FROM samples "
                f"WHERE region_id={sql_agent_quote(region_id)} {scope_filter}"
                f"ORDER BY {sql_agent_metric_order_expr('object_count')} "
                f"LIMIT 1"
                f") "
                f"SELECT * FROM changed_pick "
                f"UNION ALL "
                f"SELECT * FROM top_objects"
            )
        if subset_limit or explicit_topk:
            scoped_limit = subset_limit or safe_limit
            return (
                f"WITH top_changed AS ("
                f"SELECT {fields} "
                f"FROM samples "
                f"WHERE region_id={sql_agent_quote(region_id)} {scope_filter}"
                f"ORDER BY {sql_agent_metric_order_expr('changed_px')} "
                f"LIMIT {scoped_limit}"
                f") "
                f"SELECT * FROM top_changed "
                f"ORDER BY {sql_agent_metric_order_expr('object_count')} "
                f"LIMIT 1"
            )
        if re.search(r"(按|排序|排名)", question):
            return (
                f"SELECT {fields} "
                f"FROM samples "
                f"WHERE region_id={sql_agent_quote(region_id)} {scope_filter}"
                f"ORDER BY {sql_agent_metric_order_expr('object_count')} "
                f"LIMIT 1"
            )
    elif compound_subset_question:
        return None

    if comparison_sample_ids:
        ordered_ids = []
        for sample_id in comparison_sample_ids:
            if sample_id and sample_id not in ordered_ids:
                ordered_ids.append(sample_id)
        if len(ordered_ids) >= 2:
            id_list = ", ".join(sql_agent_quote(sample_id) for sample_id in ordered_ids)
            return (
                f"SELECT {fields} "
                f"FROM samples "
                f"WHERE sample_id IN ({id_list}) "
                f"ORDER BY FIELD(sample_id, {id_list})"
            )

    if rank_question and region_id and not referenced_sample_id:
        rank_limit = safe_limit if explicit_topk and not ordinal else max(3, min(max(ordinal or 1, 3), 12))
        return (
            f"SELECT {fields} "
            "FROM samples "
            f"WHERE region_id={sql_agent_quote(region_id)} {scope_filter}"
            "ORDER BY COALESCE(changed_px,0) DESC, sample_id ASC "
            f"LIMIT {rank_limit}"
        )

    if object_rank_question and region_id and not referenced_sample_id:
        rank_limit = safe_limit if explicit_topk and not ordinal else max(3, min(max(ordinal or 1, 3), 12))
        return (
            f"SELECT {fields} "
            "FROM samples "
            f"WHERE region_id={sql_agent_quote(region_id)} {scope_filter}"
            "ORDER BY COALESCE(object_count,0) DESC, COALESCE(changed_px,0) DESC, sample_id ASC "
            f"LIMIT {rank_limit}"
        )

    if referenced_sample_id and not SQL_AGENT_COMPARE_PATTERN.search(question):
        return (
            f"SELECT {fields} "
            f"FROM samples "
            f"WHERE sample_id={sql_agent_quote(referenced_sample_id)} "
            "LIMIT 1"
        )

    if current_sample_id and not is_region_level_question(question):
        return (
            f"SELECT {fields} "
            f"FROM samples "
            f"WHERE sample_id={sql_agent_quote(current_sample_id)} "
            "LIMIT 1"
        )

    if not region_id:
        return None

    review_question = bool(
        SQL_AGENT_TOP_REVIEW_PATTERN.search(question)
        or re.search(r"(复核|风险|重点|优先|值得看|建议看|异常|高风险|中风险)", question)
    )
    if review_question:
        return (
            f"SELECT {fields} "
            "FROM samples "
            f"WHERE region_id={sql_agent_quote(region_id)} {scope_filter}"
            + (f"AND {review_level_expr} IN ('high', 'medium') " if not sql_agent_requested_filter(question) else "")
            + f"ORDER BY {review_order_expr} "
            f"LIMIT {safe_limit}"
        )

    if SQL_AGENT_TOP_CHANGED_PATTERN.search(question):
        return (
            f"SELECT {fields} "
            "FROM samples "
            f"WHERE region_id={sql_agent_quote(region_id)} {scope_filter}"
            "ORDER BY COALESCE(changed_px,0) DESC, sample_id ASC "
            f"LIMIT {safe_limit}"
        )

    if database_scope_question(question) and re.search(r"(统计|分布|总览|概况|整体|总数|数量)", question):
        return (
            "SELECT "
            f"{review_level_expr} AS review_level, "
            "COUNT(*) AS sample_count, "
            "SUM(CASE WHEN COALESCE(changed_px,0) > 0 THEN 1 ELSE 0 END) AS changed_sample_count, "
            "MAX(COALESCE(changed_px,0)) AS max_changed_px "
            "FROM samples "
            f"WHERE region_id={sql_agent_quote(region_id)} {scope_filter}"
            "GROUP BY review_level "
            "ORDER BY CASE review_level "
            "WHEN 'high' THEN 4 "
            "WHEN 'medium' THEN 3 "
            "WHEN 'low' THEN 2 "
            "WHEN 'none' THEN 1 "
            "ELSE 0 END DESC"
        )

    return None


def sql_agent_task_type(question: str, rows: list[dict[str, Any]], resolved_refs: dict[str, Any]) -> str:
    sample_ids = sql_agent_extract_sample_ids(rows)
    if resolved_refs.get("comparison_sample_ids") or (
        SQL_AGENT_COMPARE_PATTERN.search(question) and len(sample_ids) >= 2
    ):
        return "compare_samples"
    if resolved_refs.get("rank_resolver") or resolved_refs.get("ranked_sample_ids"):
        if len(sample_ids) >= 2:
            return "ranked_samples"
    if len(sample_ids) >= 3:
        return "ranked_samples"
    if len(sample_ids) == 1:
        return "single_sample"
    return "aggregate"


def sql_agent_ranked_ids_by_resolver(resolver: dict[str, Any] | None, limit: int = 12) -> list[str]:
    if not resolver:
        return []
    region_id = str(resolver.get("region_id") or "").strip()
    if not region_id:
        return []
    safe_limit = max(1, min(int(limit or 12), 20))
    kind = str(resolver.get("kind") or "")
    scope_filter = sql_agent_scope_sql(str(resolver.get("filter_question") or "")) if resolver.get("filter_question") else sql_agent_filter_sql(str(resolver.get("review_filter") or ""))
    if kind in {"changed_px_desc", "object_count_desc"}:
        rows = sql_agent_fetch_all(
            f"""
            SELECT sample_id
            FROM samples
            WHERE region_id=%s {scope_filter}
            ORDER BY {sql_agent_metric_order_expr("object_count" if kind == "object_count_desc" else "changed_px")}
            LIMIT {safe_limit}
            """,
            (region_id,),
        )
        return [str(row.get("sample_id") or "") for row in rows if row.get("sample_id")]
    if kind == "review_priority":
        review_level_expr = sql_agent_review_level_expr()
        rows = sql_agent_fetch_all(
            f"""
            SELECT sample_id
            FROM samples
            WHERE region_id=%s {scope_filter}
              AND {review_level_expr} IN ('high', 'medium')
            ORDER BY {sql_agent_review_order_expr()}
            LIMIT {safe_limit}
            """,
            (region_id,),
        )
        sample_ids = [str(row.get("sample_id") or "") for row in rows if row.get("sample_id")]
        if sample_ids:
            return sample_ids
        rows = sql_agent_fetch_all(
            f"""
            SELECT sample_id
            FROM samples
            WHERE region_id=%s {scope_filter}
            ORDER BY {sql_agent_metric_order_expr("object_count" if kind == "object_count_desc" else "changed_px")}
            LIMIT {safe_limit}
            """,
            (region_id,),
        )
        return [str(row.get("sample_id") or "") for row in rows if row.get("sample_id")]
    return []


def sql_agent_should_handle(
    payload: dict[str, Any],
    session_memory: dict[str, Any],
    recent_messages: list[dict[str, Any]],
) -> bool:
    if not sql_agent_enabled():
        return False
    question = str(payload.get("question") or "").strip()
    if not question:
        return False
    if is_project_question(question):
        return True
    normalized = sql_agent_normalize_key(question)
    if not normalized:
        return False
    has_context = bool(
        recent_messages
        or session_memory.get("last_ranked_sample_ids")
        or session_memory.get("last_rank_resolver")
        or session_memory.get("last_compare_sample_ids")
        or payload.get("sample")
        or payload.get("region")
    )
    if SQL_AGENT_FOLLOWUP_PATTERN.search(question) and has_context:
        return True
    return False


def sql_agent_deterministic_answer(payload, query_result, resolved_refs) -> str:
    """Format returned facts before considering an optional generation call."""
    for handler in (sql_agent_same_sample_answer, sql_agent_compound_subset_answer, sql_agent_metric_rank_answer):
        answer = handler(payload, query_result) if handler is sql_agent_same_sample_answer else handler(payload, query_result, resolved_refs)
        if answer:
            return answer
    rows = query_result.get("rows") or []
    if not rows:
        return "本次数据查询未找到符合条件的记录。"
    if all(row.get("sample_id") or row.get("sample_stem") for row in rows):
        return sql_agent_fallback_answer(payload, query_result, resolved_refs)
    return "查询统计：\n" + "\n".join("；".join(f"{key}：{value}" for key, value in row.items()) for row in rows[:12])


def sql_agent_run_structured(payload: dict[str, Any], plan: dict) -> dict[str, Any]:
    """Compile a restricted query plan, preserving filter and sort separately."""
    started = time.perf_counter()
    metric = plan.get("metric")
    if metric not in {"changed_px", "object_count", "review_priority"}:
        raise ValueError("不支持的排序指标")
    region_id = str(plan.get("region_id") or "")
    if not region_id:
        raise ValueError("缺少数据区域")
    limit = plan.get("limit", 3)
    if type(limit) is not int or not 1 <= limit <= 12:
        raise ValueError("返回数量应为1到12")
    scope = sql_agent_scope_sql(str(plan.get("question") or ""))
    original_question = str(plan.get("question") or "")
    positive_review_text = re.sub(r"无需复核|不需要复核|不用复核|非待复核|排除待复核", "", original_question)
    if re.search(r"待复核|需要复核|优先复核", positive_review_text) and sql_agent_requested_filter(original_question) != "review":
        scope += sql_agent_filter_sql("review")
    if metric == "review_priority":
        order = sql_agent_review_order_expr()
    else:
        order = sql_agent_metric_order_expr(metric)
    direction = plan.get("direction", "desc")
    if direction not in {"asc", "desc"}:
        raise ValueError("不支持的排序方向")
    if direction == "asc":
        order = order.replace(" DESC", " ASC")
    sql = (f"SELECT {sql_agent_core_sample_select()} FROM samples "
           f"WHERE region_id={sql_agent_quote(region_id)} {scope} ORDER BY {order} LIMIT {limit}")
    result = sql_agent_execute_sql(sql)
    rows = result.get("rows") or []
    answer = '\n'.join(f"{i}. {row.get('sample_id', '样本')}：变化 {row.get('changed_px', '未知')} 像素，对象 {row.get('object_count', '未知')} 个，复核等级 {row.get('review_level', '未知')}。"
                       for i, row in enumerate(rows, 1)) or "当前条件下未找到样本。"
    memory = {"last_question": plan.get("question"), "last_sql": sql,
              "last_query_task": "ranked_samples", "last_region_id": region_id,
              "last_ranked_sample_ids": [str(row["sample_id"]) for row in rows],
              "last_result_preview": rows[:6], "last_result_row_count": len(rows)}
    db_save_session_context(db_session_key(payload), memory)
    source = result.get("dataSource") or "mysql"
    return {"answer": answer, "engine": f"Structured SQL / {source}", "sqlAgent": True,
            "sqlPlanSource": "structured", "sqlQuery": sql, "sqlRows": rows,
            "sqlPreviewRows": rows[:6], "sqlRowCount": len(rows),
            "sessionMemory": memory, "dataSource": source, "grounded": True,
            "answerScope": "sample" if len(rows) == 1 else "list", "evidenceSamples": rows,
            "llmCalls": 0, "latencyMs": round((time.perf_counter() - started) * 1000, 2),
            "toolTrace": [{"tool": "query_samples", "status": "ok", "source": source,
                           "summary": f"按结构化条件查询，返回 {len(rows)} 条记录"}]}


def sql_agent_run(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("_structuredTaskPlan"):
        return sql_agent_run_structured(payload, payload["_structuredTaskPlan"])
    started = time.perf_counter()
    llm_calls = 0
    question = str(payload.get("question") or "")
    chinese_numbers = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6,
                       "七": 7, "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12}
    question = re.sub(r"前\s*(十二|十一|十|[一二两三四五六七八九])", lambda m: "前" + str(chinese_numbers[m.group(1)]), question)
    question = re.sub(r"(最大|最多)的?\s*(\d{1,2})\s*([个张])", lambda m: m.group(1) + "的前" + m.group(2) + m.group(3), question)
    payload = {**payload, "question": sql_agent_intent_question(question), "_scopeQuestion": question}
    session_key = db_session_key(payload)
    recent_messages = db_recent_session_messages(session_key, 8)
    session_memory = db_load_session_context(session_key)
    resolved_refs = sql_agent_resolve_references(
        str(payload.get("question") or ""),
        payload,
        session_memory,
        recent_messages,
    )
    raw_sql = ""
    plan_source = "model"
    template_sql = sql_agent_template_query(payload, resolved_refs)
    if template_sql:
        raw_sql = template_sql
        sql_text = sql_agent_extract_sql(template_sql)
        try:
            query_result = sql_agent_execute_sql(sql_text)
            plan_source = "template"
        except Exception:
            template_sql = None
    if not template_sql:
        if not getattr(ARGS, "use_deepseek", False):
            raise ValueError("当前未启用大模型，问题未匹配可执行的查询模板；请明确区域、指标或样本。")
        sql_prompt = sql_agent_build_sql_prompt_v2(payload, session_memory, recent_messages, resolved_refs)
        llm_calls += 1
        raw_sql = deepseek_text_answer(sql_prompt, max_new_tokens=SQL_AGENT_SQL_MAX_NEW_TOKENS)
        sql_text = sql_agent_extract_sql(raw_sql)
        try:
            query_result = sql_agent_execute_sql(sql_text)
        except Exception as exc:
            retry_prompt = sql_agent_retry_prompt_v2(payload, sql_text, str(exc), resolved_refs)
            llm_calls += 1
            retry_raw_sql = deepseek_text_answer(retry_prompt, max_new_tokens=SQL_AGENT_SQL_MAX_NEW_TOKENS)
            retry_sql = sql_agent_extract_sql(retry_raw_sql)
            try:
                query_result = sql_agent_execute_sql(retry_sql)
            except Exception as retry_exc:
                raise RuntimeError(
                    "sql planning failed | "
                    f"first_sql={compact_free_text(sql_text, 500)} | "
                    f"retry_sql={compact_free_text(retry_sql, 500)} | "
                    f"retry_error={retry_exc}"
                ) from retry_exc
            raw_sql = retry_raw_sql
            sql_text = retry_sql
    query_result["sql"] = sql_text
    answer = sql_agent_deterministic_answer(payload, query_result, resolved_refs)
    next_memory = sql_agent_build_session_memory(payload, session_memory, resolved_refs, query_result)
    db_save_session_context(session_key, next_memory)
    rows = list(query_result.get("rows") or [])
    sample_rows = [row for row in rows if row.get("sample_id") or row.get("sample_stem")]
    scope = "sample" if len(sample_rows) == 1 else ("list" if sample_rows else "region")
    source = query_result.get("dataSource") or "mysql"
    return {
        "answer": answer,
        "engine": f"SQL Agent / {source} / {plan_source}",
        "mode": "project_grounded_sql_agent",
        "grounded": True,
        "persisted": mysql_enabled(),
        "sqlAgent": True,
        "sqlPlanSource": plan_source,
        "sqlQuery": sql_text,
        "sqlPreviewRows": query_result.get("rows")[:6],
        "sqlRowCount": query_result.get("row_count"),
        "sqlTruncated": bool(query_result.get("truncated")),
        "sessionMemory": next_memory,
        "sqlRows": rows[:12],
        "dataSource": source,
        "answerScope": scope,
        "evidenceSamples": sample_rows[:12],
        "llmCalls": llm_calls,
        "latencyMs": round((time.perf_counter() - started) * 1000, 2),
        "toolTrace": [{"tool": "query_samples", "status": "ok", "source": source,
                       "summary": f"返回 {len(rows)} 条记录；查询来源 {plan_source}"}],
    }


def postprocess_sample_answer(payload: dict[str, Any], answer: str) -> str:
    sample = payload.get("sample") or {}
    if not sample or not payload.get("singleSampleOnly"):
        return answer
    text = sanitize_user_answer(str(answer or "").strip())
    changed_px = sample.get("changedPx")
    object_count = sample.get("objectCount")
    grid_position = sample.get("gridPosition")
    missing = []
    if isinstance(changed_px, (int, float)) and str(int(changed_px)) not in text:
        missing.append(f"总变化面积 {int(changed_px)}px")
    if isinstance(object_count, int) and str(object_count) not in text:
        missing.append(f"对象数量 {object_count} 个")
    if grid_position and str(grid_position) not in text:
        missing.append(f"主要位置 {grid_position}")
    if missing:
        text += "\n\n后端核对：" + "；".join(missing) + "。"
    return text


KNOWLEDGE_INDEXES: dict[str, KnowledgeIndex] = {}


def inspection_knowledge_index() -> KnowledgeIndex:
    folder = str(getattr(ARGS, "knowledge_dir", Path(__file__).resolve().parents[1] / "knowledge"))
    dense, rerank = bool(getattr(ARGS, "dense_retrieval", False)), bool(getattr(ARGS, "rerank", False))
    cache = str(getattr(ARGS, "retrieval_cache", Path(__file__).resolve().parents[1] / ".model-cache"))
    key = json.dumps([folder, dense, rerank, cache])
    if key not in KNOWLEDGE_INDEXES:
        if len(KNOWLEDGE_INDEXES) >= 4:
            KNOWLEDGE_INDEXES.pop(next(iter(KNOWLEDGE_INDEXES)))
        if dense or rerank:
            from hybrid_retrieval import HybridKnowledgeIndex
            KNOWLEDGE_INDEXES[key] = HybridKnowledgeIndex(folder, dense=dense, rerank=rerank, cache_dir=cache)
        else:
            KNOWLEDGE_INDEXES[key] = KnowledgeIndex(folder)
    return KNOWLEDGE_INDEXES[key]


def inspection_generator():
    if getattr(ARGS, "use_deepseek", False):
        return lambda prompt: deepseek_text_answer(prompt, max_new_tokens=min(getattr(ARGS, "max_new_tokens", 768), 512))
    return None


def handle_knowledge_chat(payload: dict[str, Any], route: str) -> dict[str, Any]:
    started = time.perf_counter()
    question = str(payload.get("question") or "")
    data_response: dict[str, Any] = {}
    trace = []
    data_error = None
    if route == "mixed":
        try:
            if not sql_agent_enabled():
                raise ValueError("样本数据源未启用")
            data_response = sql_agent_run({**payload, "question": data_question(question)})
            trace.extend(data_response.get("toolTrace") or [])
        except Exception as exc:
            data_error = str(exc)
            trace.append({"tool": "query_samples", "status": "unavailable", "summary": "未取得本次样本事实"})
    facts = data_response.get("sqlRows") or []
    retrieval_query = question + " " + " ".join(str(r.get("review_reason") or "") for r in facts)
    hits = inspection_knowledge_index().search(retrieval_query, 3)
    trace.append({"tool": "search_knowledge", "status": "ok" if hits else "empty",
                  "source": "bm25-cjk-bigram", "summary": f"检索到 {len(hits)} 条资料"})
    explanation = explain_with_evidence(question, hits, facts, inspection_generator())
    supported_citations = public_citations(hits, explanation)
    trace.append({"tool": "explain_evidence", "status": explanation["generationStatus"],
                  "summary": "模型生成" if explanation["generationStatus"] == "generated" else "使用知识原文摘录"})
    parts = []
    if data_response.get("answer"):
        parts.append(str(data_response["answer"]))
    elif route == "mixed":
        parts.append((data_error + "。" if data_error and data_error.startswith("上次列表") else "本次未取得对应样本数据。") + "以下仅为通用资料说明，不能作为该样本的判读结论。")
    parts.append(explanation["answer"])
    return {**explanation, "answer": "\n\n".join(parts), "mode": f"project_grounded_{route}",
            "questionRoute": route, "answerScope": "mixed" if route == "mixed" else "knowledge",
            "grounded": bool(supported_citations) and (route != "mixed" or bool(facts)),
            "evidenceBound": bool(facts), "citations": supported_citations,
            "evidenceSamples": data_response.get("evidenceSamples") or [],
            "dataSource": data_response.get("dataSource", "unavailable" if route == "mixed" else "knowledge"),
            "dataError": data_error, "toolTrace": trace,
            "llmCalls": explanation["llmCalls"] + data_response.get("llmCalls", 0),
            "latencyMs": round((time.perf_counter() - started) * 1000, 2),
            "retrieval": {"method": "bm25-cjk-bigram", "hitCount": len(hits)},
            "persisted": mysql_enabled()}


def handle_chat_legacy(payload: dict[str, Any]) -> dict[str, Any]:
    question = str(payload.get("question") or "").strip()
    ev_hash = evidence_hash(payload) if payload.get("sample") else None
    session_key = db_session_key(payload)
    recent_messages = db_recent_session_messages(session_key, 8)
    session_memory = db_load_session_context(session_key)
    route = route_question(question, bool(recent_messages or session_memory or payload.get("sample")))
    if route in {"knowledge", "mixed"}:
        response = handle_knowledge_chat(payload, route)
        db_save_qa_exchange(payload, str(response.get("answer") or ""), str(response.get("engine") or ""), ev_hash)
        return response
    intent, match_score = corpus_match(question)
    if intent and not (ARGS.use_qwen or getattr(ARGS, "use_deepseek", False)):
        answer = sanitize_user_answer(corpus_answer(payload, intent, question))
        response = {
            "answer": answer,
            "engine": "local intent corpus",
            "mode": "corpus_answer",
            "grounded": False,
            "evidenceBound": intent.get("scope") != "static",
            "intentId": intent.get("intent_id"),
            "intentLabel": intent.get("label"),
            "matchScore": round(match_score, 4),
            "persisted": mysql_enabled(),
        }
        db_save_qa_exchange(payload, answer, "local intent corpus", ev_hash)
        return response
    if sql_agent_should_handle(payload, session_memory, recent_messages):
        try:
            sql_response = sql_agent_run(payload)
            db_save_qa_exchange(payload, str(sql_response.get("answer") or ""), str(sql_response.get("engine") or ""), ev_hash)
            return sql_response
        except Exception as exc:
            traceback.print_exc()
            if sql_agent_is_strict_db_question(question):
                response = {
                    "answer": "数据库问答未生成可执行 SQL。请换一种更明确的问法，或稍后重试。",
                    "engine": f"sql agent error: {type(exc).__name__}",
                    "mode": "project_grounded_sql_error",
                    "grounded": False,
                    "persisted": mysql_enabled(),
                    "sqlAgent": False,
                    "sqlError": str(exc),
                }
                db_save_qa_exchange(payload, str(response.get("answer") or ""), str(response.get("engine") or ""), ev_hash)
                return response
            try:
                fallback_answer, fallback_engine = generate_project_answer(project_payload_scope(payload))
                response = {
                    "answer": fallback_answer,
                    "engine": f"{fallback_engine} | sql agent fallback after {type(exc).__name__}",
                    "mode": "project_grounded_fallback",
                    "grounded": True,
                    "persisted": mysql_enabled(),
                    "sqlAgent": False,
                    "sqlError": str(exc),
                }
            except Exception as fallback_exc:
                traceback.print_exc()
                response = {
                    "answer": f"DeepSeek SQL Agent 调用失败，且项目问答回退也失败：{fallback_exc}",
                    "engine": f"sql agent error: {type(exc).__name__} / fallback error: {type(fallback_exc).__name__}",
                    "mode": "project_grounded_fallback",
                    "grounded": False,
                    "persisted": mysql_enabled(),
                    "sqlAgent": False,
                    "sqlError": str(exc),
                    "error": str(fallback_exc),
                    "qwenError": True,
                    "modelError": True,
                }
            db_save_qa_exchange(payload, str(response.get("answer") or ""), str(response.get("engine") or ""), ev_hash)
            return response
    if is_project_question(question):
        try:
            scoped_payload = project_payload_scope(payload)
            answer, engine = generate_project_answer(scoped_payload)
            response = {"answer": answer, "engine": engine, "mode": "project_grounded", "grounded": True, "persisted": mysql_enabled()}
        except Exception as exc:
            traceback.print_exc()
            response = {
                "answer": f"大模型项目证据问答调用失败：{exc}",
                "engine": f"model error: {type(exc).__name__}",
                "mode": "project_grounded",
                "grounded": False,
                "qwenError": True,
                "modelError": True,
                "error": str(exc),
                "persisted": mysql_enabled(),
            }
        db_save_qa_exchange(payload, str(response.get("answer") or ""), str(response.get("engine") or ""), ev_hash)
        return response
    try:
        answer, engine = generate_general_answer(payload)
        response = {"answer": answer, "engine": engine, "mode": "general_chat", "grounded": False, "persisted": mysql_enabled()}
    except Exception as exc:
        traceback.print_exc()
        response = {
            "answer": f"大模型通用对话调用失败：{exc}",
            "engine": f"model error: {type(exc).__name__}",
            "mode": "general_chat",
            "grounded": False,
            "qwenError": True,
            "modelError": True,
            "error": str(exc),
            "persisted": mysql_enabled(),
        }
    db_save_qa_exchange(payload, str(response.get("answer") or ""), str(response.get("engine") or ""), ev_hash)
    return response


AGENT_WORKFLOWS: dict[str, Any] = {}
AGENT_WORKFLOW_LOCK = threading.RLock()
REVIEW_TASK_SERVICES: dict[str, Any] = {}
REVIEW_SERVICE_LOCK = threading.RLock()
MODEL_SQL_SERVICES: dict[str, Any] = {}
MODEL_SQL_SERVICE_LOCK = threading.RLock()
ANALYSIS_TASK_SERVICES: dict[str, Any] = {}


def model_sql_service():
    from contextlib import closing
    import sqlite3
    from model_sql import ModelSQL, QueryError, read_reviewed_ids
    if not getattr(ARGS, 'enable_model_sql', False):
        raise QueryError('当前未启用模型问数，请使用带--model-sql的本地演示启动方式。')
    if not getattr(ARGS, 'static_sql', False) or mysql_configured():
        raise QueryError('模型问数只连接独立公开快照，不连接原MySQL。')
    checkpoint = Path(getattr(ARGS, 'agent_checkpoint', Path(__file__).resolve().parents[1] / 'logs/agent_checkpoints.sqlite'))
    key = json.dumps([str(ARGS.web_data), str(checkpoint), str(ARGS.model_id), bool(getattr(ARGS, 'use_deepseek', False))])
    with MODEL_SQL_SERVICE_LOCK:
        if key not in MODEL_SQL_SERVICES:
            def confirmed_ids():
                review_db = checkpoint.with_name('review_tasks.sqlite')
                version = hashlib.sha256(Path(ARGS.web_data).read_bytes()).hexdigest()
                return read_reviewed_ids(review_db, version)

            def generate(prompt, **kwargs):
                # Follow R1's recommended sampling for this explicit comparison;
                # short code-instruct SQL uses deterministic decoding.
                reasoning = 'deepseek-r1' in str(ARGS.model_id).lower()
                sampling = {'do_sample': True, 'temperature': 0.6, 'top_p': 0.95} if reasoning else {'do_sample': False}
                return deepseek_generate_raw(prompt, sampling=sampling, max_input_tokens=8192, **kwargs)

            if len(MODEL_SQL_SERVICES) >= 4: MODEL_SQL_SERVICES.pop(next(iter(MODEL_SQL_SERVICES)))
            MODEL_SQL_SERVICES[key] = ModelSQL(ARGS.web_data, checkpoint.with_name('model_sql.sqlite'),
                generate=generate if getattr(ARGS, 'use_deepseek', False) else None,
                model_id=ARGS.model_id, reviewed_provider=confirmed_ids, contract_mode=True,query_update_adapter=bool(getattr(ARGS,'query_update_adapter',None)),
                retrieve=lambda question: inspection_knowledge_index().search(question,3))
        service = MODEL_SQL_SERVICES[key]
    return service


def handle_model_sql(payload):
    return model_sql_service().run(payload)


CONVERSATION_SERVICES: dict[int, Any] = {}


def handle_conversation(payload):
    from conversation_assistant import ConversationAssistant
    sql=model_sql_service();key=id(sql)
    with MODEL_SQL_SERVICE_LOCK:
        if key not in CONVERSATION_SERVICES:
            if len(CONVERSATION_SERVICES)>=4:CONVERSATION_SERVICES.pop(next(iter(CONVERSATION_SERVICES)))
            CONVERSATION_SERVICES[key]=ConversationAssistant(sql,sql.history_path.with_name('assistant_conversations.sqlite'))
        service=CONVERSATION_SERVICES[key]
    return service.run(payload)


def handle_analysis_task(payload):
    from analysis_tasks import AnalysisTasks
    sql_service = model_sql_service()
    key = str(sql_service.history_path) + '|' + sql_service.model_id
    with MODEL_SQL_SERVICE_LOCK:
        if key not in ANALYSIS_TASK_SERVICES:
            if len(ANALYSIS_TASK_SERVICES) >= 4:
                ANALYSIS_TASK_SERVICES.pop(next(iter(ANALYSIS_TASK_SERVICES)))
            ANALYSIS_TASK_SERVICES[key] = AnalysisTasks(sql_service.source_path,
                sql_service.history_path.with_name('analysis_tasks.sqlite'),
                generate=sql_service.generate,model_id=sql_service.model_id,
                reviewed_provider=sql_service.reviewed_provider)
        service = ANALYSIS_TASK_SERVICES[key]
    return service.handle(payload)


def close_review_tasks():
    with REVIEW_SERVICE_LOCK:
        for service in REVIEW_TASK_SERVICES.values(): service.close()
        REVIEW_TASK_SERVICES.clear()


def review_tasks_service():
    from review_tasks import ReviewTasks, ReviewError
    if not getattr(ARGS, 'static_sql', False) or mysql_configured():
        raise ReviewError('复核清单当前使用快照与独立本地记录；请使用免登录本地演示启动方式。', 409)
    checkpoint = Path(getattr(ARGS, 'agent_checkpoint', Path(__file__).resolve().parents[1] / 'logs/agent_checkpoints.sqlite'))
    key = json.dumps([str(getattr(ARGS, 'web_data', '')), str(checkpoint), bool(getattr(ARGS, 'use_deepseek', False))])
    with REVIEW_SERVICE_LOCK:
        if key not in REVIEW_TASK_SERVICES:
            REVIEW_TASK_SERVICES[key] = ReviewTasks(ARGS.web_data, checkpoint.with_name('review_tasks.sqlite'),
                generate=deepseek_generate_raw if getattr(ARGS, 'use_deepseek', False) else None,
                retrieve=lambda question: inspection_knowledge_index().search(question))
        return REVIEW_TASK_SERVICES[key]


def handle_review_task(payload):
    from review_tasks import ReviewError
    service = review_tasks_service()
    owner = service.owner(payload)
    action = payload.get('action')
    if action == 'from_query':
        try:selection=model_sql_service().review_selection(payload)
        except ValueError as exc:raise ReviewError(str(exc),409) from exc
        return service.adopt_query(selection,payload)
    if action == 'status': return service.get(payload.get('jobId'), owner)
    if action == 'cancel': return service.cancel(payload.get('jobId'), owner)
    if action == 'confirm': return service.confirm(payload, owner)
    raise ReviewError('未知复核任务操作。')


def close_agent_workflows() -> None:
    with AGENT_WORKFLOW_LOCK:
        for agent in AGENT_WORKFLOWS.values():
            agent.close()
        AGENT_WORKFLOWS.clear()


def workflow_region_options() -> list[dict]:
    if mysql_enabled():
        return live_region_rows()
    source = Path(getattr(ARGS, "web_data", ""))
    if not source.is_file():
        return []
    data = json.loads(source.read_text(encoding="utf-8-sig"))
    return [{key: value for key, value in region.items() if key in {"id", "short", "name", "regionName"}}
            for region in data.get("regions", [])]


def handle_chat(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get('queryMode') == 'model_sql':
        return handle_model_sql(payload)
    if getattr(ARGS, 'enable_agent_workflow', False):
        from review_planning import is_review_request, NUM
        question = str(payload.get('question') or '')
        if is_review_request(question, bool(payload.get('reviewTaskId'))):
            return review_tasks_service().submit(payload)
        if payload.get('reviewTaskId') and re.search(r'第\s*' + NUM + r'\s*(?:张|个|幅)', question):
            return review_tasks_service().reference(payload)
    if not getattr(ARGS, "enable_agent_workflow", False):
        return handle_chat_legacy(payload)
    from inspection_workflow import InspectionWorkflow
    key = json.dumps([str(getattr(ARGS, name, "")) for name in
        ("web_data", "knowledge_dir", "agent_checkpoint", "dense_retrieval", "rerank", "retrieval_cache", "use_deepseek", "model_id")])
    with AGENT_WORKFLOW_LOCK:
        if key not in AGENT_WORKFLOWS:
            if len(AGENT_WORKFLOWS) >= 4:
                AGENT_WORKFLOWS.pop(next(iter(AGENT_WORKFLOWS))).close()

            def query(request):
                if request.get("_workflowMemory"):
                    db_save_session_context(db_session_key(request), request["_workflowMemory"])
                return sql_agent_run(request)

            def general(request):
                answer, engine = generate_general_answer(request)
                return {"answer": answer, "engine": engine,
                        "llmCalls": int(bool(getattr(ARGS, "use_deepseek", False)))}

            AGENT_WORKFLOWS[key] = InspectionWorkflow(
                query=query, retrieve=lambda question: inspection_knowledge_index().search(question),
                explain=lambda question, hits, facts: explain_with_evidence(question, hits, facts, inspection_generator()),
                session_key=db_session_key, regions=workflow_region_options,
                planner=inspection_generator(), general=general,
                retrieval_status=lambda: inspection_knowledge_index().status(),
                checkpoint_path=getattr(ARGS, "agent_checkpoint", Path(__file__).resolve().parents[1] / "logs/agent_checkpoints.sqlite"))
        response = AGENT_WORKFLOWS[key].run(payload)
        persisted_payload = {**payload, "sessionKey": response.pop("_sessionKey", db_session_key(payload))}
        db_save_qa_exchange(persisted_payload, str(response.get("answer") or ""), str(response.get("engine") or "LangGraph inspection"))
        response["persisted"] = mysql_enabled()
        return response


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt_text: str, *args: Any) -> None:
        print(f"[qa-api] {self.address_string()} - {fmt_text % args}")

    def do_OPTIONS(self) -> None:
        json_response(self, 200, {"ok": True})

    def do_GET(self) -> None:
        path, query = parse_request_path(self.path)
        if path == "/api/health":
            mysql_state = mysql_status()
            json_response(
                self,
                200,
                {
                    "ok": True,
                    "useQwen": bool(ARGS.use_qwen),
                    "useDeepSeek": bool(getattr(ARGS, "use_deepseek", False)),
                    "modelBackend": "qwen" if ARGS.use_qwen else ("deepseek" if getattr(ARGS, "use_deepseek", False) else "local"),
                    "modelId": ARGS.model_id,
                    "modelLoaded": MODEL is not None,
                    "mysql": "enabled" if mysql_state.get("ok") else ("configured" if mysql_configured() else "disabled"),
                    "mysqlConfigured": bool(mysql_configured()),
                    "mysqlReachable": bool(mysql_state.get("ok")),
                    "mysqlTransport": str(mysql_state.get("transport") or ""),
                    "mysqlError": str(mysql_state.get("error") or ""),
                    "authEnabled": not demo_no_auth_enabled(),
                    "authMode": "local_demo" if demo_no_auth_enabled() else "login",
                    "serviceVersion": "inspection-agentic-rag-v3",
                    "agentWorkflowEnabled": bool(getattr(ARGS, "enable_agent_workflow", False)),
                    "knowledge": inspection_knowledge_index().status(),
                    "staticSqlEnabled": sql_agent_static_enabled(),
                    "modelSqlEnabled": bool(getattr(ARGS, 'enable_model_sql', False)),
                    "queryUpdateAdapterEnabled": bool(getattr(ARGS,'query_update_adapter',None)),
                    "analysisTasksEnabled": bool(getattr(ARGS, 'enable_model_sql', False)),
                    "modelSqlRegions": workflow_region_options() if getattr(ARGS, 'enable_model_sql', False) else [],
                    "sqlAgentEnabled": bool(sql_agent_enabled()),
                    "sqlSessionMemoryTable": "qa_session_contexts",
                    "localIntentCorpusEnabled": local_intent_corpus_enabled(),
                    "ruleAnswerEnabled": rule_answer_enabled(),
                    "qaCorpusIntents": len(QA_CORPUS.get("intents") or []),
                    "qaCorpusUtterances": len(QA_ENTRIES),
                },
            )
            return
        if path == "/api/auth/session":
            json_response(self, 200, handle_auth_session(parse_auth_token(self)))
            return
        if path == "/api/admin/overview":
            if require_admin(self) is None:
                return
            json_response(self, 200, admin_overview())
            return
        if path == "/api/live_regions":
            if require_auth(self) is None:
                return
            region_id = str((query.get("region_id") or [""])[0] or "").strip()
            regions = live_region_rows(region_id or None)
            if not regions:
                json_response(
                    self,
                    503,
                    {
                        "ok": False,
                        "error": "live region data unavailable",
                        "mysql": "enabled" if mysql_enabled() else "disabled",
                        "regions": [],
                    },
                )
                return
            json_response(
                self,
                200,
                {
                    "ok": True,
                    "source": "mysql",
                    "regionCount": len(regions),
                    "updatedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "regions": regions,
                },
            )
            return
        json_response(self, 404, {"error": "not found"})

    def do_POST(self) -> None:
        path, _query = parse_request_path(self.path)
        path = path or "/"
        protected_routes = {"/api/chat", "/api/assistant", "/api/nl_sql", "/api/analysis", "/api/review_tasks", "/api/report_summary", "/api/judgement", "/api/correction", "/api/review_override", "/api/review_complete", "/api/deploy/crop", "/api/deploy/train", "/api/deploy/evaluate", "/api/deploy/add_sample", "/api/deploy/preview_image", "/api/deploy/status", "/api/admin/query", "/api/admin/detail", "/api/admin/users", "/api/admin/user_save", "/api/admin/user_password"}
        auth_routes = {"/api/auth/register", "/api/auth/login", "/api/auth/logout"}
        if path in protected_routes | auth_routes:
            try:
                length = int(self.headers.get("Content-Length") or "0")
                raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
                payload = json.loads(raw_body or "{}")
                if path == "/api/auth/register":
                    json_response(self, 200, handle_auth_register(payload))
                elif path == "/api/auth/login":
                    json_response(self, 200, handle_auth_login(payload))
                elif path == "/api/auth/logout":
                    json_response(self, 200, handle_auth_logout(parse_auth_token(self)))
                elif path in {"/api/admin/query", "/api/admin/detail", "/api/admin/users", "/api/admin/user_save", "/api/admin/user_password"} and require_admin(self) is None:
                    return
                elif (authenticated_user := require_auth(self)) is None:
                    return
                elif path == "/api/chat":
                    principal = authenticated_user.get("user") or {}
                    payload["_authenticatedUser"] = str(principal.get("username") or principal.get("id") or "authenticated")
                    response = handle_chat(payload)
                    json_response(self, 202 if response.get("reviewJob") else 200, response)
                elif path == "/api/review_tasks":
                    principal = authenticated_user.get("user") or {}
                    payload["_authenticatedUser"] = str(principal.get("username") or principal.get("id") or "authenticated")
                    json_response(self, 200, handle_review_task(payload))
                elif path == "/api/nl_sql":
                    principal = authenticated_user.get("user") or {}
                    payload["_authenticatedUser"] = str(principal.get("username") or principal.get("id") or "authenticated")
                    json_response(self, 200, handle_model_sql(payload))
                elif path == "/api/assistant":
                    principal = authenticated_user.get("user") or {}
                    payload["_authenticatedUser"] = str(principal.get("username") or principal.get("id") or "authenticated")
                    json_response(self, 200, handle_conversation(payload))
                elif path == "/api/analysis":
                    principal = authenticated_user.get("user") or {}
                    payload["_authenticatedUser"] = str(principal.get("username") or principal.get("id") or "authenticated")
                    json_response(self, 200, handle_analysis_task(payload))
                elif path == "/api/report_summary":
                    json_response(self, 200, handle_report_summary(payload))
                elif path == "/api/judgement":
                    json_response(self, 200, handle_judgement(payload))
                elif path == "/api/admin/query":
                    json_response(self, 200, handle_admin_query(payload))
                elif path == "/api/admin/detail":
                    json_response(self, 200, handle_admin_detail(payload))
                elif path == "/api/admin/users":
                    json_response(self, 200, handle_admin_users(payload))
                elif path == "/api/admin/user_save":
                    json_response(self, 200, handle_admin_user_save(payload))
                elif path == "/api/admin/user_password":
                    json_response(self, 200, handle_admin_user_password(payload))
                elif path == "/api/correction":
                    json_response(self, 200, handle_correction(payload))
                elif path == "/api/review_override":
                    json_response(self, 200, handle_review_override(payload))
                elif path == "/api/review_complete":
                    json_response(self, 200, handle_review_complete(payload))
                elif path == "/api/deploy/crop":
                    json_response(self, 200, handle_deploy_crop(payload))
                elif path == "/api/deploy/train":
                    json_response(self, 200, handle_deploy_train(payload))
                elif path == "/api/deploy/evaluate":
                    json_response(self, 200, handle_deploy_evaluate(payload))
                elif path == "/api/deploy/add_sample":
                    json_response(self, 200, handle_deploy_add_sample(payload))
                elif path == "/api/deploy/preview_image":
                    json_response(self, 200, handle_deploy_preview_image(payload))
                else:
                    json_response(self, 200, handle_deploy_status(payload))
            except Exception as exc:
                traceback.print_exc()
                json_response(
                    self,
                    getattr(exc, "status", 500),
                    {
                        "error": str(exc),
                        "errorType": type(exc).__name__,
                        "answer": f"后端接口异常：{exc}",
                        "engine": "api exception",
                        "persisted": False,
                        "qwenError": True,
                    },
                )
            return
        json_response(self, 404, {"error": "not found"})


def local_report_summary(snapshot: dict[str, Any]) -> str:
    total = int(snapshot.get("totalCount") or 0)
    changed = int(snapshot.get("changedCount") or 0)
    review = int(snapshot.get("reviewCount") or 0)
    dominant_event = str(snapshot.get("dominantEvent") or "暂无主导主事件")
    dominant_scale = str(snapshot.get("dominantScale") or "暂无主导变化层级")
    reason_rows = snapshot.get("reasonBuckets") or []
    top_reason = str((reason_rows[0] or {}).get("label") or "暂无集中风险原因") if reason_rows else "暂无集中风险原因"
    reps = [
        row for row in (snapshot.get("representatives") or [])
        if str((row or {}).get("stem") or "").strip()
    ][:3]
    change_ratio = f"{(changed / total * 100):.1f}%" if total > 0 else "0.0%"
    risk_map = {str((row or {}).get("key") or ""): int((row or {}).get("count") or 0) for row in (snapshot.get("riskBuckets") or [])}
    high_count = risk_map.get("high", 0)
    medium_count = risk_map.get("medium", 0)
    rep_text = "、".join(
        f"{str((row or {}).get('stem') or '').strip()}（{int((row or {}).get('changedPx') or 0)}px，{int((row or {}).get('objectCount') or 0)}对象）"
        for row in reps
    )
    correction_count = int(snapshot.get("correctionCount", len(snapshot.get("correctionEntries") or [])) or 0)
    sentences = [
        f"区域结论：当前范围共 {total} 张切片，其中变化切片 {changed} 张，占 {change_ratio}；整体以{dominant_event}为主，变化层级以{dominant_scale}为主。",
    ]
    if review > 0:
        sentences.append(f"复核方面共有 {review} 张优先复核样本，其中高风险 {high_count} 张、中风险 {medium_count} 张，主要触发原因集中在{top_reason}。")
    else:
        sentences.append("当前范围内暂无优先复核样本，整体结果以常规变化与无变化切片为主。")
    if rep_text:
        sentences.append(f"代表样本建议优先查看 {rep_text}。")
    else:
        sentences.append("当前范围内暂无需要突出展示的代表样本，可结合主事件分布和面积排序继续筛查。")
    if correction_count > 0:
        sentences.append(f"当前已有 {correction_count} 条人工修订回流，建议优先对照已修正样本复查同类问题。")
    else:
        sentences.append("当前暂无人工修订回流，建议先从高风险样本与面积靠前样本开始复核。")
    return "".join(sentences[:4]).strip()


def report_summary_prompt(payload: dict[str, Any]) -> str:
    region = payload.get("region") or {}
    snapshot = payload.get("reportSnapshot") or {}
    event_rows = snapshot.get("eventBuckets") or []
    risk_rows = snapshot.get("riskBuckets") or []
    reason_rows = snapshot.get("reasonBuckets") or []
    rep_rows = snapshot.get("representatives") or []
    event_text = "；".join(
        f"{row.get('label')} {int(row.get('count') or 0)}张"
        for row in event_rows[:5]
        if row.get("label")
    ) or "暂无主事件分布"
    risk_text = "；".join(
        f"{row.get('label')} {int(row.get('count') or 0)}张"
        for row in risk_rows[:4]
        if row.get("label")
    ) or "暂无风险分布"
    reason_text = "；".join(
        f"{row.get('label')} {int(row.get('count') or 0)}张"
        for row in reason_rows[:5]
        if row.get("label")
    ) or "暂无风险原因"
    rep_text = "；".join(
        f"{row.get('stem')}｜{row.get('status')}｜{int(row.get('changedPx') or 0)}px"
        for row in rep_rows[:5]
        if row.get("stem")
    ) or "暂无代表样本"
    return (
        "你是遥感变化巡检平台的中文报告助手。\n"
        "你只能根据下面提供的事实生成最终区域结论。\n"
        "只输出 3 到 4 句中文正文，不要输出思考过程、提示词复述、输入说明、JSON、字段名、规则条款、<think>、representatives、report_snapshot、mysql_context。\n"
        "第一句必须以“区域结论：”开头。\n"
        "必须覆盖整体变化态势、主导主事件或主导层级、复核风险、优先查看样本。\n"
        "如果没有代表样本，就不要硬写样本名。\n"
        f"区域：{region.get('short') or region.get('name') or region.get('id') or '当前区域'}\n"
        f"总切片：{int(snapshot.get('totalCount') or 0)}\n"
        f"变化切片：{int(snapshot.get('changedCount') or 0)}\n"
        f"优先复核样本：{int(snapshot.get('reviewCount') or 0)}\n"
        f"主导主事件：{snapshot.get('dominantEvent') or '暂无'}\n"
        f"主导变化层级：{snapshot.get('dominantScale') or '暂无'}\n"
        f"主事件分布：{event_text}\n"
        f"风险分布：{risk_text}\n"
        f"风险原因分布：{reason_text}\n"
        f"代表样本：{rep_text}\n"
        "现在直接输出最终区域结论："
    )


def finalize_report_summary_answer(text: str, snapshot: dict[str, Any]) -> str:
    cleaned = strip_deepseek_reasoning(sanitize_user_answer(text or "")).replace("\r", "\n").strip()
    if not cleaned:
        return local_report_summary(snapshot)
    drop_markers = [
        "总长度控制",
        "如果代表样本为空",
        "输入 JSON",
        "输出要求",
        "只保留",
        "不要输出",
        "report_snapshot",
        "mysql_context",
        "query_result",
        "representatives",
        "buckets",
        "distribution",
        "用户提供",
        "下面给出",
        "五个样本",
        "开头",
        "3到4句",
        "3 到 4 句",
    ]
    kept: list[str] = []
    for piece in re.split(r"(?<=[。！？])\s+|[\n]+", cleaned):
        sentence = piece.strip(" -*•\t")
        if not sentence:
            continue
        lowered = sentence.lower()
        if any(marker.lower() in lowered for marker in drop_markers):
            continue
        if re.match(r"^(首先|接下来|然后|最后|现在我需要|我将|我现在|用户)", sentence):
            continue
        if not re.search(r"(区域|切片|变化|复核|风险|样本|主导|事件|层级|建议|张)", sentence):
            continue
        kept.append(sentence if sentence.endswith(("。", "！", "？")) else sentence + "。")
        if len(kept) >= 4:
            break
    if not kept:
        return local_report_summary(snapshot)
    body = "".join(kept).strip()
    if body.startswith("结论："):
        body = "区域结论：" + body[len("结论：") :].lstrip()
    elif body.startswith("结论:"):
        body = "区域结论：" + body[len("结论:") :].lstrip()
    elif not body.startswith("区域结论："):
        body = "区域结论：" + body
    if any(marker in body for marker in ("总长度控制", "report_snapshot", "mysql_context", "representatives")):
        return local_report_summary(snapshot)
    return body.strip()


def handle_report_summary(payload: dict[str, Any]) -> dict[str, Any]:
    snapshot = payload.get("reportSnapshot") or {}
    updated_at = datetime.utcnow().isoformat(timespec="seconds")
    fallback = local_report_summary(snapshot)
    question = "根据区域报告的复核风险，说明检查依据和操作建议。 " + " ".join(
        str(r.get("label") or "") for r in snapshot.get("reasonBuckets") or [])
    hits = inspection_knowledge_index().search(question, 3)
    explanation = explain_with_evidence(question, hits, snapshot, inspection_generator())
    supported_citations = public_citations(hits, explanation)
    return {**explanation, "ok": True, "answer": fallback + "\n\n" + explanation["answer"],
            "answerScope": "region", "citations": supported_citations, "updatedAt": updated_at,
            "reportFingerprint": hashlib.sha256(canonical_json(snapshot).encode("utf-8")).hexdigest(),
            "dataSource": "report_snapshot", "grounded": bool(supported_citations),
            "toolTrace": [{"tool": "report_snapshot", "status": "ok", "summary": "采用当前报告统计快照"},
                          {"tool": "search_knowledge", "status": "ok" if hits else "empty", "summary": f"检索到 {len(hits)} 条资料"}]}


def main() -> None:
    global ARGS, WEB_ROOT
    ARGS = parse_args()
    if ARGS.no_auth and not demo_no_auth_enabled():
        raise SystemExit("--no_auth requires a loopback host, --static_sql, and no configured MySQL connection.")
    if ARGS.use_qwen:
        raise SystemExit("This clean package disables the legacy Qwen VLM backend. Use --use_deepseek or local mode.")
    web_data = Path(ARGS.web_data).resolve()
    WEB_ROOT = web_data.parent if web_data.exists() else Path.cwd()
    load_qa_corpus(ARGS.qa_corpus)
    server = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    print(f"qa_api=http://{ARGS.host}:{ARGS.port}/api/chat")
    print(f"judgement_api=http://{ARGS.host}:{ARGS.port}/api/judgement")
    print(f"correction_api=http://{ARGS.host}:{ARGS.port}/api/correction")
    print(f"deploy_crop_api=http://{ARGS.host}:{ARGS.port}/api/deploy/crop")
    print(f"deploy_train_api=http://{ARGS.host}:{ARGS.port}/api/deploy/train")
    print(f"deploy_status_api=http://{ARGS.host}:{ARGS.port}/api/deploy/status")
    print(f"engine={'qwen' if ARGS.use_qwen else ('deepseek' if getattr(ARGS, 'use_deepseek', False) else 'local evidence rules')}")
    print(f"local_intent_corpus={'enabled' if local_intent_corpus_enabled() else 'disabled'}")
    print(f"rule_answer={'enabled' if rule_answer_enabled() else 'disabled'}")
    print(f"mysql={'enabled' if mysql_enabled() else 'disabled'}")
    print(f"web_root={WEB_ROOT}")
    server.serve_forever()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Import trust-agent web metadata into MySQL with public/private separation.

Images stay on disk. MySQL stores deterministic image asset indexes and
structured facts. Evaluation-only fields such as GT masks and F1/IoU are kept
out of public facts so DeepSeek prompts do not leak benchmark/test labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any


PRIVATE_ASSET_KEYS = {"gtMask", "gtOverlay", "gt_mask", "gt_overlay"}
PRIVATE_METRIC_KEYS = {
    "maskF1",
    "maskIoU",
    "maskPrecision",
    "maskRecall",
    "f1",
    "iou",
    "precision",
    "recall",
}
ASSET_ROLE_META = {
    "pre": ("t1", "public", 0),
    "post": ("t2", "public", 0),
    "overlay": ("pred_overlay", "public", 0),
    "predOverlay": ("pred_overlay", "public", 0),
    "mask": ("pred_mask", "public", 0),
    "gtMask": ("gt_mask", "audit_private", 0),
    "gtOverlay": ("gt_overlay", "audit_private", 0),
}
LEAK_PATTERNS = [
    re.compile(r"\bGT\b", re.I),
    re.compile(r"ground\s*truth", re.I),
    re.compile(r"真值|标注|标签|测试集答案|真实\s*mask", re.I),
    re.compile(r"gt[_-]?(mask|overlay)", re.I),
    re.compile(r"mask\s*(F1|IoU|precision|recall)", re.I),
    re.compile(r"\b(F1|IoU)\b", re.I),
    re.compile(r"(和|与)?后台.{0,12}(MASK|mask|变化范围).{0,12}不匹配", re.I),
    re.compile(r"(MASK|mask|变化范围).{0,8}不匹配", re.I),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import sanitized trust-agent metadata into MySQL.")
    parser.add_argument("--app_data", default="web/app_data.json")
    parser.add_argument("--index_html", default="", help="Optional web/index.html containing embedded APP data.")
    parser.add_argument("--host", default=os.getenv("TRUST_AGENT_MYSQL_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("TRUST_AGENT_MYSQL_PORT", "3306")))
    parser.add_argument("--user", default=os.getenv("TRUST_AGENT_MYSQL_USER", "root"))
    parser.add_argument("--password", default=os.getenv("TRUST_AGENT_MYSQL_PASSWORD", ""))
    parser.add_argument("--database", default=os.getenv("TRUST_AGENT_MYSQL_DATABASE", "trust_change_agent"))
    parser.add_argument("--clear", action="store_true", help="Delete imported metadata before importing.")
    parser.add_argument("--stat_assets", action="store_true", help="Read asset file sizes. Slower on network paths.")
    parser.add_argument("--hash_assets", action="store_true", help="Compute sha256 for asset files. Slower.")
    parser.add_argument("--image_dims", action="store_true", help="Read asset image dimensions. Slower on large folders.")
    return parser.parse_args()


def connect(args: argparse.Namespace):
    try:
        import pymysql  # type: ignore

        return pymysql.connect(
            host=args.host,
            port=args.port,
            user=args.user,
            password=args.password,
            database=args.database,
            charset="utf8mb4",
            autocommit=False,
            cursorclass=pymysql.cursors.DictCursor,
        )
    except ImportError as exc:
        raise RuntimeError("Install pymysql first: pip install pymysql") from exc


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def embedded_app_from_index(index_path: Path) -> dict[str, Any] | None:
    if not index_path.exists():
        return None
    text = index_path.read_text(encoding="utf-8-sig", errors="replace")
    match = re.search(r'const\s+APP\s*=\s*JSON\.parse\(new\s+TextDecoder\(\)\.decode\(Uint8Array\.from\(atob\("([^"]+)"\)', text)
    if not match:
        return None
    import base64

    decoded = base64.b64decode(match.group(1))
    return json.loads(decoded.decode("utf-8"))


def load_app_data(app_path: Path, index_path: Path | None) -> tuple[dict[str, Any], str]:
    data = load_json(app_path)
    regions = data.get("regions") or []
    has_items = any(region.get("items") or region.get("samples") for region in regions)
    if not has_items and index_path:
        embedded = embedded_app_from_index(index_path)
        if embedded:
            return embedded, str(index_path)
    return data, str(app_path)


def strip_leaky_sentences(text: str) -> str:
    cleaned = str(text or "")
    if not cleaned:
        return ""
    pieces = re.split(r"(?<=[。！？!?；;])|\n+", cleaned)
    kept: list[str] = []
    for piece in pieces:
        part = piece.strip()
        if not part:
            continue
        if any(pattern.search(part) for pattern in LEAK_PATTERNS):
            continue
        kept.append(part)
    cleaned = "".join(kept) if kept else ""
    replacements = {
        "预测掩膜": "模型输出的变化范围",
        "掩膜": "变化范围",
        "mask": "变化范围",
        "Mask": "变化范围",
        "MASK": "变化范围",
        "overlay": "变化叠加图",
        "Overlay": "变化叠加图",
        "bbox": "变化范围",
        "证据包": "后台结构化结果",
        "evidence JSON": "后台结构化结果",
        "JSON": "结构化数据",
    }
    for src, dst in replacements.items():
        cleaned = cleaned.replace(src, dst)
    return cleaned.strip()


def safe_text(value: Any) -> str:
    return strip_leaky_sentences(str(value or ""))


def sample_event_key(sample: dict[str, Any]) -> str:
    return str(sample.get("eventType") or sample.get("pred") or "no_primary_event")


def sample_event_name(sample: dict[str, Any]) -> str:
    return str(sample.get("eventTypeZh") or sample.get("morphologyLabel") or sample.get("pred") or "未分类事件")


def public_object(obj: dict[str, Any], fallback_grid: Any = None) -> dict[str, Any]:
    return {
        "id": obj.get("id"),
        "area_px": obj.get("areaPx") or obj.get("area_px"),
        "bbox": obj.get("bbox") or obj.get("bboxFull") or [],
        "center": obj.get("center") or obj.get("centerFull") or [],
        "grid_position": obj.get("grid") or obj.get("gridPosition") or fallback_grid,
        "shape": obj.get("shape") or obj.get("shapeType"),
    }


def public_assets(assets: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in (assets or {}).items() if key not in PRIVATE_ASSET_KEYS}


def private_metrics(sample: dict[str, Any], private_asset_ids: dict[str, str]) -> dict[str, Any]:
    metrics = {key: sample.get(key) for key in PRIVATE_METRIC_KEYS if key in sample and sample.get(key) is not None}
    if private_asset_ids:
        metrics["audit_asset_ids"] = private_asset_ids
    if sample.get("verifierChecks"):
        metrics["verifier_checks_raw"] = sample.get("verifierChecks")
    return metrics


def public_sample_facts(region: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    objects = [public_object(obj, sample.get("gridPosition")) for obj in (sample.get("objects") or [])]
    facts = {
        "schema_version": "trust_agent_public_facts_v1",
        "sample_id": sample.get("id") or sample.get("sampleId") or sample.get("name") or sample.get("stem"),
        "sample_name": sample.get("name") or sample.get("stem"),
        "region_id": region.get("id") or region.get("short"),
        "region_name": region.get("name") or region.get("short"),
        "target_type": sample.get("primaryEventName") or region.get("eventName") or region.get("role"),
        "event_key": sample_event_key(sample),
        "event_name": sample_event_name(sample),
        "change_scale": sample.get("scale"),
        "change_detected": bool((sample.get("changedPx") or 0) > 0 or (sample.get("objectCount") or 0) > 0),
        "changed_px": sample.get("changedPx"),
        "changed_ratio": sample.get("changedRatio"),
        "object_count": sample.get("objectCount") or 0,
        "main_bbox": sample.get("mainBbox") or [],
        "grid_position": sample.get("gridPosition"),
        "review_required": bool(sample.get("reviewRequired")),
        "review_status": sample.get("reviewStatus"),
        "review_reason": safe_text(sample.get("reviewReason")),
        "quality_flags": sample.get("qualityFlags") or [],
        "objects": objects,
        "evidence_summary": safe_text(sample.get("evidenceSummary")),
        "report": safe_text(sample.get("report")),
    }
    return {key: value for key, value in facts.items() if value not in (None, "")}


def evidence_hash(region: dict[str, Any], sample: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(public_sample_facts(region, sample)).encode("utf-8")).hexdigest()


def asset_id_for(sample_id: str, asset_role: str, uri: str) -> str:
    digest = hashlib.sha256(f"{sample_id}:{asset_role}:{uri}".encode("utf-8")).hexdigest()
    return digest[:32]


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def asset_fs_path(app_root: Path, uri: str) -> Path:
    path = Path(uri)
    if path.is_absolute():
        return path
    return app_root / path


def asset_file_meta(path: Path, stat_assets: bool, hash_assets: bool, image_dims: bool) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "mime_type": mimetypes.guess_type(str(path))[0],
        "file_size": path.stat().st_size if stat_assets and path.exists() and path.is_file() else None,
        "sha256": sha256_file(path) if hash_assets else None,
    }
    if not image_dims:
        meta["width"], meta["height"] = None, None
        return meta
    try:
        from PIL import Image  # type: ignore

        if path.exists() and path.is_file():
            with Image.open(path) as img:
                meta["width"], meta["height"] = img.size
        else:
            meta["width"], meta["height"] = None, None
    except Exception:
        meta["width"], meta["height"] = None, None
    return meta


def execute(cur: Any, sql: str, params: tuple[Any, ...] = ()) -> None:
    cur.execute(sql, params)


def upsert_asset(
    cur: Any,
    app_root: Path,
    sample_id: str,
    region_id: str,
    source_key: str,
    uri: str,
    stat_assets: bool,
    hash_assets: bool,
    image_dims: bool,
) -> tuple[str, str, str, int]:
    asset_role, visibility, include_in_llm = ASSET_ROLE_META.get(source_key, (source_key, "public", 0))
    fs_path = asset_fs_path(app_root, str(uri))
    asset_id = asset_id_for(sample_id, asset_role, str(uri))
    meta = asset_file_meta(fs_path, stat_assets, hash_assets, image_dims)
    execute(
        cur,
        """
        INSERT INTO image_assets
          (asset_id, sample_id, region_id, asset_role, storage_backend, uri, fs_path,
           mime_type, width, height, file_size, sha256, visibility, include_in_llm)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
          sample_id=VALUES(sample_id),
          region_id=VALUES(region_id),
          asset_role=VALUES(asset_role),
          storage_backend=VALUES(storage_backend),
          uri=VALUES(uri),
          fs_path=VALUES(fs_path),
          mime_type=VALUES(mime_type),
          width=VALUES(width),
          height=VALUES(height),
          file_size=VALUES(file_size),
          sha256=VALUES(sha256),
          visibility=VALUES(visibility),
          include_in_llm=VALUES(include_in_llm),
          is_active=1
        """,
        (
            asset_id,
            sample_id,
            region_id,
            asset_role,
            "local",
            str(uri),
            str(fs_path),
            meta["mime_type"],
            meta["width"],
            meta["height"],
            meta["file_size"],
            meta["sha256"],
            visibility,
            include_in_llm,
        ),
    )
    return source_key, asset_id, visibility, include_in_llm


def main() -> None:
    args = parse_args()
    app_path = Path(args.app_data)
    if not app_path.is_absolute():
        app_path = (Path.cwd() / app_path).resolve()
    index_path = Path(args.index_html) if args.index_html else app_path.parent / "index.html"
    if not index_path.is_absolute():
        index_path = (Path.cwd() / index_path).resolve()

    data, data_source = load_app_data(app_path, index_path)
    regions = data.get("regions") or []
    app_root = app_path.parent

    conn = connect(args)
    try:
        cur = conn.cursor()
        if args.clear:
            for table in ("prediction_results", "image_assets", "sample_objects", "samples", "regions"):
                cur.execute(f"DELETE FROM {table}")

        total_samples = 0
        total_objects = 0
        total_assets = 0
        private_assets = 0
        by_region: list[dict[str, Any]] = []
        for region in regions:
            region_id = str(region.get("id") or region.get("short") or "")
            if not region_id:
                continue
            items = region.get("items") or region.get("samples") or []
            execute(
                cur,
                """
                INSERT INTO regions
                  (region_id, short_name, region_name, role_text, status_text, scene_guide,
                   total_rows, display_rows, changed_rows, display_changed_rows,
                   review_rows, full_review_rows, avg_confidence,
                   counts_json, display_counts_json, metrics_json, region_json)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE
                  short_name=VALUES(short_name),
                  region_name=VALUES(region_name),
                  role_text=VALUES(role_text),
                  status_text=VALUES(status_text),
                  scene_guide=VALUES(scene_guide),
                  total_rows=VALUES(total_rows),
                  display_rows=VALUES(display_rows),
                  changed_rows=VALUES(changed_rows),
                  display_changed_rows=VALUES(display_changed_rows),
                  review_rows=VALUES(review_rows),
                  full_review_rows=VALUES(full_review_rows),
                  avg_confidence=VALUES(avg_confidence),
                  counts_json=VALUES(counts_json),
                  display_counts_json=VALUES(display_counts_json),
                  metrics_json=VALUES(metrics_json),
                  region_json=VALUES(region_json)
                """,
                (
                    region_id,
                    region.get("short"),
                    region.get("name"),
                    region.get("role"),
                    region.get("status"),
                    region.get("sceneGuide"),
                    region.get("totalRows"),
                    region.get("displayRows") or len(items),
                    region.get("changedRows"),
                    region.get("displayChangedRows"),
                    region.get("reviewRows"),
                    region.get("fullReviewRows"),
                    region.get("avgConfidence"),
                    canonical_json(region.get("counts") or {}),
                    canonical_json(region.get("displayCounts") or {}),
                    canonical_json(region.get("metrics") or {}),
                    canonical_json({k: v for k, v in region.items() if k != "items"}),
                ),
            )

            region_samples = 0
            for sample in items:
                sample_id = str(sample.get("id") or sample.get("sampleId") or sample.get("name") or sample.get("stem") or "")
                if not sample_id:
                    continue
                assets = sample.get("assets") or {}
                asset_ids: dict[str, str] = {}
                private_asset_ids: dict[str, str] = {}
                for source_key, uri in assets.items():
                    if not uri:
                        continue
                    source_key, asset_id, visibility, _include = upsert_asset(
                        cur,
                        app_root,
                        sample_id,
                        region_id,
                        str(source_key),
                        str(uri),
                        args.stat_assets,
                        args.hash_assets,
                        args.image_dims,
                    )
                    asset_ids[source_key] = asset_id
                    total_assets += 1
                    if visibility != "public":
                        private_asset_ids[source_key] = asset_id
                        private_assets += 1

                public_facts = public_sample_facts(region, sample)
                private_facts = private_metrics(sample, private_asset_ids)
                ev_hash = evidence_hash(region, sample)
                visible_assets = public_assets(assets)
                report_text = safe_text(sample.get("report"))
                review_reason = safe_text(sample.get("reviewReason"))
                execute(
                    cur,
                    """
                    INSERT INTO samples
                      (sample_id, region_id, sample_name, sample_stem, event_key, event_name,
                       changed_px, changed_ratio, object_count, grid_position,
                       review_status, review_required, review_reason, confidence,
                       mask_precision, mask_recall, mask_f1, mask_iou, pred_area,
                       main_bbox_json, quality_flags_json, report_text, display_policy,
                       public_facts_json, private_metrics_json, evidence_hash, evidence_json, assets_json)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
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
                      review_required=VALUES(review_required),
                      review_reason=VALUES(review_reason),
                      confidence=VALUES(confidence),
                      mask_precision=VALUES(mask_precision),
                      mask_recall=VALUES(mask_recall),
                      mask_f1=VALUES(mask_f1),
                      mask_iou=VALUES(mask_iou),
                      pred_area=VALUES(pred_area),
                      main_bbox_json=VALUES(main_bbox_json),
                      quality_flags_json=VALUES(quality_flags_json),
                      report_text=VALUES(report_text),
                      display_policy=VALUES(display_policy),
                      public_facts_json=VALUES(public_facts_json),
                      private_metrics_json=VALUES(private_metrics_json),
                      evidence_hash=VALUES(evidence_hash),
                      evidence_json=VALUES(evidence_json),
                      assets_json=VALUES(assets_json)
                    """,
                    (
                        sample_id,
                        region_id,
                        sample.get("name"),
                        sample.get("stem"),
                        sample_event_key(sample),
                        sample_event_name(sample),
                        sample.get("changedPx"),
                        sample.get("changedRatio"),
                        sample.get("objectCount"),
                        sample.get("gridPosition"),
                        sample.get("reviewStatus"),
                        1 if sample.get("reviewRequired") else 0,
                        review_reason,
                        sample.get("confidence"),
                        sample.get("maskPrecision"),
                        sample.get("maskRecall"),
                        sample.get("maskF1"),
                        sample.get("maskIoU"),
                        sample.get("predArea"),
                        canonical_json(sample.get("mainBbox") or []),
                        canonical_json(sample.get("qualityFlags") or []),
                        report_text,
                        sample.get("displayPolicy"),
                        canonical_json(public_facts),
                        canonical_json(private_facts),
                        ev_hash,
                        canonical_json(public_facts),
                        canonical_json(visible_assets),
                    ),
                )

                pred_id = f"{sample_id}:latest"
                execute(
                    cur,
                    """
                    INSERT INTO prediction_results
                      (prediction_id, sample_id, region_id, run_id, pred_mask_asset_id,
                       pred_overlay_asset_id, changed_px, changed_ratio, object_count,
                       main_bbox_json, grid_position, event_key, event_name, review_required,
                       review_reason, confidence, public_facts_json, private_metrics_json)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON DUPLICATE KEY UPDATE
                      region_id=VALUES(region_id),
                      pred_mask_asset_id=VALUES(pred_mask_asset_id),
                      pred_overlay_asset_id=VALUES(pred_overlay_asset_id),
                      changed_px=VALUES(changed_px),
                      changed_ratio=VALUES(changed_ratio),
                      object_count=VALUES(object_count),
                      main_bbox_json=VALUES(main_bbox_json),
                      grid_position=VALUES(grid_position),
                      event_key=VALUES(event_key),
                      event_name=VALUES(event_name),
                      review_required=VALUES(review_required),
                      review_reason=VALUES(review_reason),
                      confidence=VALUES(confidence),
                      public_facts_json=VALUES(public_facts_json),
                      private_metrics_json=VALUES(private_metrics_json)
                    """,
                    (
                        pred_id,
                        sample_id,
                        region_id,
                        "web_import_latest",
                        asset_ids.get("mask"),
                        asset_ids.get("predOverlay") or asset_ids.get("overlay"),
                        sample.get("changedPx"),
                        sample.get("changedRatio"),
                        sample.get("objectCount"),
                        canonical_json(sample.get("mainBbox") or []),
                        sample.get("gridPosition"),
                        sample_event_key(sample),
                        sample_event_name(sample),
                        1 if sample.get("reviewRequired") else 0,
                        review_reason,
                        sample.get("confidence"),
                        canonical_json(public_facts),
                        canonical_json(private_facts),
                    ),
                )

                cur.execute("DELETE FROM sample_objects WHERE sample_id=%s", (sample_id,))
                objects = sample.get("objects") or []
                for index, obj in enumerate(objects, 1):
                    pub_obj = public_object(obj, sample.get("gridPosition"))
                    execute(
                        cur,
                        """
                        INSERT INTO sample_objects
                          (sample_id, region_id, object_index, object_key, area_px,
                           bbox_json, center_json, grid_position, shape_text, object_json)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """,
                        (
                            sample_id,
                            region_id,
                            index,
                            pub_obj.get("id"),
                            pub_obj.get("area_px"),
                            canonical_json(pub_obj.get("bbox") or []),
                            canonical_json(pub_obj.get("center") or []),
                            pub_obj.get("grid_position"),
                            pub_obj.get("shape"),
                            canonical_json(pub_obj),
                        ),
                    )
                total_samples += 1
                total_objects += len(objects)
                region_samples += 1
            by_region.append({"region_id": region_id, "samples": region_samples})

        cur.close()
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    print(
        json.dumps(
            {
                "event": "app_data_imported",
                "app_data": str(app_path),
                "data_source": data_source,
                "database": args.database,
                "regions": len(by_region),
                "samples": total_samples,
                "objects": total_objects,
                "image_assets": total_assets,
                "private_audit_assets": private_assets,
                "policy": "public facts exclude GT/label/F1/IoU and mask-mismatch wording",
                "by_region": by_region,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

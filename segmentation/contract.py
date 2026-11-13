"""Versioned artifact contract shared by Windows inference and WSL storage."""
import hashlib
import json
import math
import os
import re
import uuid
from pathlib import Path, PureWindowsPath


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest_json(value):
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def local_path(value):
    if not isinstance(value, str) or '://' in value or '\x00' in value:
        raise ValueError('Only local artifact paths are supported')
    if re.match(r'^[A-Za-z]:[\\/]', value):
        if os.name != 'nt':
            path = PureWindowsPath(value)
            return Path('/mnt', path.drive[0].lower(), *path.parts[1:])
        return Path(value)
    path = Path(value)
    if not path.is_absolute():
        raise ValueError('Artifact paths must be absolute')
    return path


def asset_record(path, role):
    path = Path(path).resolve(strict=True)
    return dict(asset_id=str(uuid.uuid4()), role=role, path=str(path),
                sha256=file_sha256(path), byte_size=path.stat().st_size)


def validate_asset(asset, *, width, height):
    from PIL import Image
    if asset.get('role') not in ('source', 'prediction_mask', 'overlay'):
        raise ValueError('Unknown asset role')
    path = local_path(asset.get('path'))
    if not path.is_file() or path.stat().st_size != asset.get('byte_size'):
        raise ValueError('Asset is missing or its size changed')
    if file_sha256(path) != asset.get('sha256'):
        raise ValueError('Asset SHA256 mismatch')
    with Image.open(path) as image:
        image.load()
        if image.size != (width, height):
            raise ValueError('Asset dimensions differ from the source')


def mask_statistics(mask):
    import cv2
    import numpy as np
    mask = np.asarray(mask)
    if mask.ndim != 2 or not mask.size or not np.isin(mask, (0, 255)).all():
        raise ValueError('Prediction mask must be a nonempty binary 0/255 image')
    height, width = mask.shape
    binary = (mask > 0).astype(np.uint8)
    pixels = int(binary.sum())
    return dict(width=width, height=height, foreground_pixels=pixels,
                foreground_ratio=pixels / (width * height),
                component_count=int(cv2.connectedComponents(binary, connectivity=8)[0] - 1))


def segmentation_metrics(counts):
    tp, fp, fn, tn = (float(counts[k]) for k in ('tp', 'fp', 'fn', 'tn'))
    if any(not math.isfinite(x) or x < 0 for x in (tp, fp, fn, tn)):
        raise ValueError('Invalid confusion matrix')
    divide = lambda a, b: a / b if b else 0.0
    biou = divide(tp, tp + fp + fn)
    return dict(building_iou=biou, f1=divide(2 * tp, 2 * tp + fp + fn),
                precision=divide(tp, tp + fp), recall=divide(tp, tp + fn),
                binary_miou=.5 * (biou + divide(tn, tn + fp + fn)))


def validate_manifest(data, *, check_files=True):
    from PIL import Image
    import numpy as np
    if not isinstance(data, dict) or data.get('schema_version') != 1:
        raise ValueError('Unsupported segmentation manifest')
    try:
        uuid.UUID(data['batch_id'])
        model = data['model']
        if model['model_id'] != digest_json(model['identity']):
            raise ValueError('Model identity digest mismatch')
        if data['task_type'] != 'building_segmentation':
            raise ValueError('This model only predicts building masks')
        if not isinstance(data['label'], str) or not 1 <= len(data['label']) <= 200:
            raise ValueError('Batch label is invalid')
        samples = data['samples']
        if not isinstance(samples, list) or not 1 <= len(samples) <= 10000:
            raise ValueError('Invalid sample count')
        ids, names, asset_ids = set(), set(), set()
        for sample in samples:
            sid = sample['sample_id']; uuid.UUID(sid)
            if sid in ids or sample['name'] in names:
                raise ValueError('Duplicate sample identity')
            ids.add(sid); names.add(sample['name'])
            if not isinstance(sample['name'], str) or not 1 <= len(sample['name']) <= 200:
                raise ValueError('Sample name is invalid')
            width, height = sample['width'], sample['height']
            pixels, ratio, components = (sample[k] for k in ('foreground_pixels', 'foreground_ratio', 'component_count'))
            if any(type(x) is not int for x in (width, height, pixels, components)):
                raise ValueError('Integer mask statistics required')
            if width < 1 or height < 1 or width * height > 16_777_216 or not 0 <= pixels <= width * height:
                raise ValueError('Mask statistics out of range')
            if not math.isfinite(ratio) or abs(ratio - pixels / (width * height)) > 1e-12 or not 0 <= components <= pixels:
                raise ValueError('Mask ratio or component count is inconsistent')
            if not 0 <= float(sample['inference_ms']) < 3_600_000:
                raise ValueError('Inference duration is invalid')
            if not re.fullmatch(r'[a-z0-9_-]{1,64}', sample['region_id']):
                raise ValueError('Region identity is invalid')
            assets = sample['assets']
            if len(assets) != 3 or {a['role'] for a in assets} != {'source', 'prediction_mask', 'overlay'}:
                raise ValueError('Each sample needs source, mask and overlay')
            for asset in assets:
                uuid.UUID(asset['asset_id'])
                if asset['asset_id'] in asset_ids:
                    raise ValueError('Duplicate asset identity')
                asset_ids.add(asset['asset_id'])
                if check_files:
                    validate_asset(asset, width=width, height=height)
                    if asset['role'] == 'prediction_mask':
                        with Image.open(local_path(asset['path'])) as image:
                            actual = mask_statistics(np.asarray(image))
                        if any(actual[k] != sample[k] for k in actual):
                            raise ValueError('Persisted mask and statistics differ')
        evaluation = data.get('evaluation')
        if evaluation:
            if evaluation['scope'] not in ('validation_subset', 'labeled_images') or evaluation['images'] != len(samples):
                raise ValueError('Evaluation scope does not match this batch')
            actual = segmentation_metrics(evaluation['counts'])
            if sum(evaluation['counts'].values()) != sum(x['width'] * x['height'] for x in samples):
                raise ValueError('Evaluation pixel count differs from source images')
            if any(abs(actual[k] - evaluation['metrics'][k]) > 1e-12 for k in actual):
                raise ValueError('Evaluation metrics do not match confusion counts')
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('Incomplete segmentation manifest') from exc
    return digest_json(data)

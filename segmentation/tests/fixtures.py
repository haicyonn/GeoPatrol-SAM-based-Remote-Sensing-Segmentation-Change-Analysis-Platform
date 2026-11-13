import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from segmentation.contract import asset_record, digest_json, mask_statistics


def manifest_fixture(directory):
    root = Path(directory)
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(root / 'source.png')
    mask = np.zeros((4, 4), dtype=np.uint8); mask[1:3, 1:3] = 255
    Image.fromarray(mask).save(root / 'mask.png')
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(root / 'overlay.png')
    identity = {'weights_sha256': '0' * 64, 'test_fixture': True}
    return dict(schema_version=1, batch_id=str(uuid.uuid4()), label='segmentation automated test',
                created_ms=1, task_type='building_segmentation',
                model=dict(model_id=digest_json(identity), name='test model', identity=identity,
                           training={'trainable_parameters': 1}),
                samples=[dict(sample_id=str(uuid.uuid4()), name='test.png', region_id='test',
                              inference_ms=1.0, **mask_statistics(mask), assets=[
                                  asset_record(root / 'source.png', 'source'),
                                  asset_record(root / 'mask.png', 'prediction_mask'),
                                  asset_record(root / 'overlay.png', 'overlay')])])

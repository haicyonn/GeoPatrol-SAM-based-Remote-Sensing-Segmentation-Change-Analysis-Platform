"""Automatic single-image building masks from a pinned local SCFP release."""
import argparse
import json
import sys
import time
import uuid
from pathlib import Path

from .contract import asset_record, canonical_json, digest_json, file_sha256, mask_statistics, segmentation_metrics, validate_manifest

DEFAULT_RUN = Path('H:/RSAM-Seg-save/scfp_whu500_100_f1_20261112_01')


def select_images(directory, split_file=None, split='val', limit=100):
    root = Path(directory).resolve(strict=True)
    if not 1 <= limit <= 10000:
        raise ValueError('Image limit must be 1..10000')
    if split_file:
        names = json.loads(Path(split_file).read_text(encoding='utf-8'))[split][:limit]
        if not names or any(Path(name).name != name or '/' in name or '\\' in name for name in names):
            raise ValueError('Invalid split image filename')
        images = [root / name for name in names]
    else:
        extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
        images = sorted(x for x in root.iterdir() if x.is_file() and x.suffix.lower() in extensions)[:limit]
    if not images or any(not path.is_file() for path in images):
        raise ValueError('Input folder is empty or a requested image is missing')
    if any(not path.resolve().is_relative_to(root) for path in images):
        raise ValueError('Image path escapes the input folder')
    return images


def dump_json(path, value):
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(canonical_json(value), encoding='utf-8')
    temporary.replace(path)


def validate_base_state(missing, unexpected):
    allowed = ('prompt_generator.', 'Space_Adapter.', 'MLP_Adapter.',
               'Depth_Adapter.', 'pe_layer.', 'no_mask_embed.')
    if any(not any(key in name for key in allowed) for name in missing):
        raise ValueError('Unexpected missing base weights')
    if any(not name.startswith('prompt_encoder.') for name in unexpected):
        raise ValueError('Unexpected base checkpoint tensors')


def build_release_identity(source, config_path, base_hash, adapter_hash, device, runtime):
    source = Path(source)
    return dict(architecture='RSAM-Seg/SAM-ViT-L', variant='SCFP',
                base_sha256=base_hash, adapter_sha256=adapter_hash,
                config_sha256=file_sha256(config_path),
                source_sha256={str(p.relative_to(source)).replace('\\', '/'): file_sha256(p)
                               for p in sorted(source.rglob('*.py')) if '__pycache__' not in p.parts},
                inference_sha256={name: file_sha256(Path(__file__).with_name(name))
                                  for name in ('predict.py', 'contract.py')},
                runtime=dict(runtime, device=device, precision='bfloat16' if device == 'cuda' else 'float32'),
                preprocessing=dict(input_size=1024, mean=[.485, .456, .406], std=[.229, .224, .225],
                                   interpolation='PIL bilinear input; bilinear logits align_corners=False',
                                   threshold=.5, output='original image dimensions'))


def make_evaluation(counts, image_count, reference, reference_matches):
    metrics = segmentation_metrics(counts)
    result = dict(scope='validation_subset' if reference_matches else 'labeled_images',
                  images=image_count, counts=counts, metrics=metrics)
    if reference_matches:
        result.update(saved_validation_f1=reference['f1'],
                      absolute_f1_delta_from_saved=abs(metrics['f1'] - reference['f1']))
    return result


def load_release(run, device):
    import cv2
    import numpy as np
    import PIL
    import torch
    import torchvision
    import yaml
    run = Path(run).resolve(strict=True)
    config_path = run / 'scfp_500_100_f1.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    source = run / 'source_snapshot'
    sys.path.insert(0, str(source))
    import models
    from train_module_screen import configure_plan_a, seed_all
    seed_all(config['seed'])
    torch.set_num_threads(4)
    from .contract import local_path
    base_path = local_path(config['sam_checkpoint'])
    adapter_path = run / 'training/best_adapters.pth'
    file_hashes = {str(p): file_sha256(p) for p in (base_path, adapter_path)}
    identity = build_release_identity(source, config_path, file_hashes[str(base_path)],
                                      file_hashes[str(adapter_path)], device,
                                      {'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
                                       'cuda_runtime': torch.version.cuda, 'numpy': str(np.__version__),
                                       'pillow': str(PIL.__version__), 'opencv': str(cv2.__version__)})
    model = models.make(config['model'])
    base = torch.load(base_path, map_location='cpu', weights_only=True, mmap=True)
    loaded = model.load_state_dict(base, strict=False)
    del base
    validate_base_state(loaded.missing_keys, loaded.unexpected_keys)
    # The locally trained checkpoint stores torch.__version__ as TorchVersion.
    # Allow that string subclass while retaining the weights-only loader.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        state = torch.load(adapter_path, map_location='cpu', weights_only=True, mmap=True)
    if state.get('format') != 'sam_adapters_v1' or state['run_info']['variant'] != config['variant']:
        raise ValueError('Adapter checkpoint does not match release configuration')
    configure_plan_a(model)
    required = {n for n, p in model.named_parameters() if p.requires_grad}
    required.update(n for n, _ in model.named_buffers())
    if not required.issubset(state['model']):
        raise ValueError('Adapter checkpoint is incomplete')
    if model.load_state_dict(state['model'], strict=False).unexpected_keys:
        raise ValueError('Adapter checkpoint contains unknown tensors')
    model.requires_grad_(False)
    model = model.to(device).eval()
    for path, expected in file_hashes.items():
        if file_sha256(path) != expected:
            raise ValueError('Weights changed while loading; refusing mixed-version inference')
    if identity != build_release_identity(source, config_path, file_hashes[str(base_path)],
                                          file_hashes[str(adapter_path)], device, identity['runtime']):
        raise ValueError('Inference source changed while loading')
    metadata = dict(model_id=digest_json(identity), name='SCFP / SAM ViT-L 建筑物分割', identity=identity,
                    training=dict(state['run_info'], base_checkpoint=str(base_path),
                                  adapter_checkpoint=str(adapter_path), source_snapshot=str(source),
                                  checkpoint_epoch=state['epoch']))
    return model, metadata, state.get('metrics', {})


def predict_batch(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms
    images = select_images(args.input, args.split_file, 'val', args.limit)
    if args.label_dir and any(not (Path(args.label_dir) / path.name).is_file() for path in images):
        raise ValueError('A validation label is missing')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu explicitly')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for folder in ('masks', 'overlays'):
        (output / folder).mkdir()
    print(json.dumps({'stage': 'loading_release', 'images': len(images)}, ensure_ascii=False), flush=True)
    model, metadata, reference = load_release(args.release, args.device)
    normalize = transforms.Compose([
        transforms.Resize((1024, 1024)), transforms.ToTensor(),
        transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])
    manifest = dict(schema_version=1, batch_id=str(uuid.uuid4()), label=args.label,
                    created_ms=int(time.time() * 1000), task_type='building_segmentation',
                    model=metadata, samples=[])
    confusion = dict(tp=0, fp=0, fn=0, tn=0)
    audit_file = Path(args.release) / 'data_audit.json'
    reference_audit = {r['name']: r for r in json.loads(audit_file.read_text(encoding='utf-8')) if r['split'] == 'val'}
    reference_matches = bool(args.split_file and file_sha256(args.split_file) == metadata['training']['split_sha256']
                             and len(images) == metadata['training']['val_images'])
    for number, path in enumerate(images, 1):
        initial_hash = file_sha256(path)
        with Image.open(path) as opened:
            if opened.width * opened.height > 16_777_216:
                raise ValueError('Source image exceeds 16777216 pixels')
            image = opened.convert('RGB')
        width, height = image.size
        tensor = normalize(image).unsqueeze(0).to(args.device)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(args.device, dtype=torch.bfloat16, enabled=args.device == 'cuda'):
            logits = model.infer(tensor)
        logits = F.interpolate(logits.float(), (height, width), mode='bilinear', align_corners=False)
        mask = ((logits[0, 0] > 0).cpu().numpy().astype(np.uint8) * 255)
        if args.device == 'cuda':
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        sid = str(uuid.uuid4())
        mask_path = output / 'masks' / f'{sid}.png'
        overlay_path = output / 'overlays' / f'{sid}.png'
        Image.fromarray(mask).save(mask_path)
        rgb = np.array(image, dtype=np.uint8)
        selected = mask > 0
        rgb[selected] = (rgb[selected].astype(np.float32) * .55 + np.array([255, 70, 40]) * .45).astype(np.uint8)
        Image.fromarray(rgb).save(overlay_path)
        source_asset = asset_record(path, 'source')
        if source_asset['sha256'] != initial_hash:
            raise ValueError('Input image changed during prediction')
        sample = dict(sample_id=sid, name=path.name, region_id=args.region,
                      inference_ms=elapsed_ms, **mask_statistics(mask), assets=[
                          source_asset, asset_record(mask_path, 'prediction_mask'), asset_record(overlay_path, 'overlay')])
        manifest['samples'].append(sample)
        if args.label_dir:
            with Image.open(Path(args.label_dir) / path.name) as label:
                truth = np.asarray(label.convert('L'))
            if truth.shape != mask.shape or not np.isin(truth, (0, 255)).all():
                raise ValueError('Validation label dimensions or values are invalid')
            truth = truth > 0
            known = reference_audit.get(path.name)
            import hashlib
            if not known or hashlib.sha256(np.array(image).tobytes()).hexdigest() != known['image_rgb_sha256'] or hashlib.sha256((truth.astype(np.uint8) * 255).tobytes()).hexdigest() != known['mask_sha256']:
                reference_matches = False
            for key, value in zip(('tp', 'fp', 'fn', 'tn'), (
                    (selected & truth).sum(), (selected & ~truth).sum(),
                    (~selected & truth).sum(), (~selected & ~truth).sum())):
                confusion[key] += int(value)
        if number % 10 == 0 or number == len(images):
            print(json.dumps({'stage': 'predicting', 'completed': number, 'total': len(images)}, ensure_ascii=False), flush=True)
    if args.label_dir:
        manifest['evaluation'] = make_evaluation(confusion, len(images), reference, reference_matches)
        manifest['evaluation'].update(label_directory=str(Path(args.label_dir).resolve()),
                                      split_sha256=file_sha256(args.split_file) if args.split_file else None)
    validate_manifest(manifest)
    dump_json(output / 'manifest.json', manifest)
    print(json.dumps(dict(stage='complete', output=str(output / 'manifest.json'), batch_id=manifest['batch_id'],
                          samples=len(images), evaluation=manifest.get('evaluation')), ensure_ascii=False), flush=True)
    return output / 'manifest.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', default=str(DEFAULT_RUN))
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split-file')
    parser.add_argument('--label-dir')
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--region', default='whu-building')
    parser.add_argument('--label', default='WHU建筑物分割 · SCFP验证子集')
    args = parser.parse_args()
    predict_batch(args)


if __name__ == '__main__':
    main()

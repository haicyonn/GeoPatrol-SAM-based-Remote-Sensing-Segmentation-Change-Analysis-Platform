"""Single-seed B-A/R1/R2 training with a common 15-epoch schedule.

Run to --stop-epoch 3 for screening, then resume the selected arm to epoch 15.
Checkpoints store only trained model tensors/buffers plus optimizer and RNG.
The original SAM checkpoint remains required to reconstruct frozen weights.
"""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

import datasets
import models
import utils
from models.sam import _iou_loss


def dump_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def configure_plan_a(model):
    model.requires_grad_(False)
    model.mask_decoder.requires_grad_(True)
    model.no_mask_embed.requires_grad_(True)
    model.image_encoder.prompt_generator.requires_grad_(True)
    for block in model.image_encoder.blocks:
        block.Space_Adapter.requires_grad_(True)
        block.MLP_Adapter.requires_grad_(True)
        block.Depth_Adapter.requires_grad_(False)
    for name, p in model.image_encoder.named_parameters():
        if p.requires_grad:
            assert name.startswith('prompt_generator.') or '.Space_Adapter.' in name or '.MLP_Adapter.' in name, name


def make_loader(spec, shuffle, generator=None):
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    return dataset, DataLoader(dataset, batch_size=1, shuffle=shuffle,
                               num_workers=0, pin_memory=True, generator=generator)


def checkpoint_model_state(model):
    keys = {name for name, p in model.named_parameters() if p.requires_grad}
    keys.update(name for name, _ in model.named_buffers())
    return {name: value.detach().cpu() for name, value in model.state_dict().items() if name in keys}


def common_initial_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(checkpoint_model_state(model).items()):
        if 'prior_extractor.' not in name:
            digest.update(name.encode())
            digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def prior_values(model):
    prior = model.image_encoder.prompt_generator.prior_extractor
    return {} if prior is None else prior.learned_filter_values()


def checkpoint_score(values, metric):
    """Select the configured validation score; reject invalid checkpoint rankings."""
    if metric not in ('building_iou', 'f1') or metric not in values:
        raise ValueError(f'Invalid checkpoint selection metric: {metric}')
    score = float(values[metric])
    if not np.isfinite(score):
        raise ValueError(f'Nonfinite checkpoint selection metric: {metric}')
    return score


def structure_prompt_loss(aux_logits, gt):
    """Native512 body occupancy and morphological edge occupancy; training only."""
    native = F.interpolate(gt.float(), (512, 512), mode='nearest')
    edge = (F.max_pool2d(native, 3, 1, 1) + F.max_pool2d(-native, 3, 1, 1)).clamp(0, 1)
    terms = []
    for logits in aux_logits:
        body_target = F.interpolate(native, logits.shape[-2:], mode='area')
        edge_target = F.interpolate(edge, logits.shape[-2:], mode='area')
        pos_weight = ((1 - edge_target).sum() / edge_target.sum().clamp_min(1)).clamp(1, 10).detach()
        body_loss = F.binary_cross_entropy_with_logits(logits[:, :1].float(), body_target)
        edge_loss = F.binary_cross_entropy_with_logits(logits[:, 1:].float(), edge_target, pos_weight=pos_weight)
        terms.append(0.5 * (body_loss + edge_loss))
    return torch.stack(terms).mean()


@torch.no_grad()
def evaluate(loader, model, amp_dtype, max_batches=None):
    model.eval()
    counts = dict(tp=0., fp=0., fn=0., tn=0.)
    macro_iou, macro_f1 = [], []
    boundary = dict(matched_pred=0., pred=0., matched_gt=0., gt=0.)
    started = time.perf_counter()
    for index, batch in enumerate(loader):
        inp = batch['inp'].cuda(non_blocking=True)
        gt = batch['gt'].cuda(non_blocking=True)
        with torch.autocast('cuda', dtype=amp_dtype):
            logits = model.infer(inp)
        logits = F.interpolate(logits.float(), size=(512, 512), mode='bilinear', align_corners=False)
        true = F.interpolate(gt.float(), size=(512, 512), mode='nearest') > 0.5
        pred = logits > 0
        tp = (pred & true).sum().item()
        fp = (pred & ~true).sum().item()
        fn = (~pred & true).sum().item()
        tn = (~pred & ~true).sum().item()
        for key, value in zip(('tp', 'fp', 'fn', 'tn'), (tp, fp, fn, tn)):
            counts[key] += value
        macro_iou.append((tp + 1e-6) / (tp + fp + fn + 1e-6))
        macro_f1.append((2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6))
        # Inner 3x3 contour; BF1 matching tolerance is 2 native 512-image pixels.
        pb = pred.float() + F.max_pool2d(-pred.float(), 3, 1, 1)
        gb = true.float() + F.max_pool2d(-true.float(), 3, 1, 1)
        boundary['matched_pred'] += (pb * F.max_pool2d(gb, 5, 1, 2)).sum().item()
        boundary['matched_gt'] += (gb * F.max_pool2d(pb, 5, 1, 2)).sum().item()
        boundary['pred'] += pb.sum().item()
        boundary['gt'] += gb.sum().item()
        if max_batches is not None and index + 1 >= max_batches:
            break
    torch.cuda.synchronize()
    tp, fp, fn, tn = [counts[key] for key in ('tp', 'fp', 'fn', 'tn')]
    foreground_iou = tp / max(tp + fp + fn, 1)
    background_iou = tn / max(tn + fp + fn, 1)
    bp = boundary['matched_pred'] / max(boundary['pred'], 1)
    br = boundary['matched_gt'] / max(boundary['gt'], 1)
    return dict(building_iou=foreground_iou,
                binary_miou=(foreground_iou + background_iou) / 2,
                f1=2 * tp / max(2 * tp + fp + fn, 1),
                precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                macro_building_iou=float(np.mean(macro_iou)),
                macro_f1=float(np.mean(macro_f1)),
                boundary_f1=2 * bp * br / max(bp + br, 1e-12),
                val_seconds=time.perf_counter() - started, val_images=len(macro_iou))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--stop-epoch', type=int, default=3)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    selection_metric = config.get('selection_metric', 'building_iou')
    checkpoint_score({selection_metric: 0.}, selection_metric)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'last_state.pth').exists() and not args.resume:
        raise FileExistsError('Existing run requires explicit --resume')
    seed = config['seed']
    seed_all(seed)
    generator = torch.Generator().manual_seed(seed)
    train_set, train_loader = make_loader(config['train_dataset'], True, generator)
    val_set, val_loader = make_loader(config['val_dataset'], False)
    if 'required_dataset_sizes' in config:
        actual_sizes = dict(train=len(train_set), val=len(val_set))
        if actual_sizes != config['required_dataset_sizes']:
            raise ValueError(f'Dataset size mismatch: {actual_sizes} != {config["required_dataset_sizes"]}')
    manifest_path = Path(config['train_dataset']['dataset']['args']['split_file'])
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    # Build on CPU first, preserving the same initialization RNG in all arms.
    model = models.make(config['model'])
    base_path = Path(config['sam_checkpoint']).resolve()
    base = torch.load(base_path, map_location='cpu', weights_only=True, mmap=True)
    missing, unexpected = model.load_state_dict(base, strict=False)
    allowed_missing = ('prompt_generator.', 'Space_Adapter.', 'MLP_Adapter.',
                       'Depth_Adapter.', 'pe_layer.', 'no_mask_embed.')
    assert all(any(part in name for part in allowed_missing) for name in missing), missing
    assert all(name.startswith('prompt_encoder.') for name in unexpected), unexpected
    del base
    configure_plan_a(model)
    initial_hash = common_initial_hash(model)
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    default_extra = 0 if config['variant'] == 'B-A' else 3
    expected = 29402404 + config.get('expected_prior_parameters', default_extra)
    assert count == expected, (count, expected)
    model.cuda()
    optimizer = utils.make_optimizer((p for p in model.parameters() if p.requires_grad), config['optimizer'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config['schedule_epochs'], eta_min=config.get('lr_min', 1e-7))
    amp_dtype = torch.bfloat16
    assert torch.cuda.is_bf16_supported(), 'This experiment requires BF16 support'
    info = dict(variant=config['variant'], seed=seed, train_images=len(train_set),
                val_images=len(val_set), trainable_parameters=count,
                total_parameters=sum(p.numel() for p in model.parameters()),
                common_initial_sha256=initial_hash, split_sha256=manifest_hash,
                precision='autocast_bfloat16; FP32 FFT/Haar/loss/optimizer states',
                base_checkpoint=str(base_path), schedule_epochs=config['schedule_epochs'],
                auxiliary_prompt_weight=config.get('auxiliary_prompt_weight', 0.),
                selection_metric=selection_metric,
                metric_resolution=512, threshold=0.5,
                augmentation='legacy TrainDataset horizontal flip probability 0.5',
                torch_version=torch.__version__, gpu=torch.cuda.get_device_name(0))
    dump_json(out / 'run_info.json', info)
    print(json.dumps({'start': info}, ensure_ascii=False), flush=True)
    start_epoch, best = 1, -1.
    metrics = []
    if args.resume:
        state = torch.load(out / 'last_state.pth', map_location='cpu', weights_only=False)
        for key in ('variant', 'seed', 'split_sha256', 'schedule_epochs', 'common_initial_sha256'):
            assert state['run_info'][key] == info[key], key
        assert state['run_info'].get('auxiliary_prompt_weight', 0.) == info['auxiliary_prompt_weight']
        assert state['run_info'].get('selection_metric', 'building_iou') == selection_metric
        assert state['trainable_names'] == trainable_names
        loaded = model.load_state_dict(state['model'], strict=False)
        assert not loaded.unexpected_keys
        assert set(state['model']) == set(checkpoint_model_state(model))
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch, best = state['epoch'] + 1, state['best']
        random.setstate(state['rng']['python'])
        np.random.set_state(state['rng']['numpy'])
        torch.set_rng_state(state['rng']['torch'])
        torch.cuda.set_rng_state_all(state['rng']['cuda'])
        generator.set_state(state['rng']['loader'])
        metrics = json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
        del state
        print(f"RESUMED epoch={start_epoch} scheduler_T_max={scheduler.T_max}", flush=True)

    stop_epoch = min(args.stop_epoch, config['schedule_epochs'])
    for epoch in range(start_epoch, stop_epoch + 1):
        model.train()
        losses, step_times, auxiliary_losses = [], [], []
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        epoch_started = time.perf_counter()
        first_audit = None
        for step, batch in enumerate(train_loader, 1):
            model.set_input(batch['inp'], batch['gt'])
            optimizer.zero_grad(set_to_none=True)
            audit_step = epoch == 1 and step == 1
            if audit_step:
                snapshots = {
                    'space': model.image_encoder.blocks[0].Space_Adapter.D_fc2.weight.detach().clone(),
                    'mlp': model.image_encoder.blocks[0].MLP_Adapter.D_fc2.weight.detach().clone(),
                }
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast('cuda', dtype=amp_dtype):
                model.forward()
            logits = model.pred_mask.float()
            loss = F.binary_cross_entropy_with_logits(logits, model.gt_mask) + _iou_loss(logits, model.gt_mask)
            segmentation_loss_value = loss.item() if audit_step else None
            auxiliary_weight = config.get('auxiliary_prompt_weight', 0.)
            if auxiliary_weight:
                aux_loss = structure_prompt_loss(model.image_encoder.prompt_generator.structure_aux_logits, model.gt_mask)
                loss = loss + auxiliary_weight * aux_loss
                auxiliary_losses.append(aux_loss.item())
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss epoch={epoch} step={step}')
            loss.backward()
            if audit_step:
                first_audit = {'segmentation_loss_before_auxiliary': segmentation_loss_value}
                for i in (0, 23):
                    for key in ('Space_Adapter', 'MLP_Adapter'):
                        grad = getattr(model.image_encoder.blocks[i], key).D_fc2.weight.grad
                        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
                        first_audit[f'block_{i}_{key}_grad_norm'] = grad.norm().item()
                assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
                prior = model.image_encoder.prompt_generator.prior_extractor
                if prior is not None:
                    active_grads = 0
                    for name, p in prior.named_parameters():
                        assert p.grad is not None and torch.isfinite(p.grad).all(), name
                        norm = p.grad.norm().item()
                        active_grads += int(norm > 0)
                        first_audit[f'prior_{name}_grad_norm'] = norm
                    assert active_grads > 0
                    # Zero-initialized structure heads intentionally delay gradients
                    # to their upstream private branch and route scales by one step.
            optimizer.step()
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - started)
            losses.append(loss.item())
            if audit_step:
                for key, module in (('space', model.image_encoder.blocks[0].Space_Adapter),
                                    ('mlp', model.image_encoder.blocks[0].MLP_Adapter)):
                    delta = (module.D_fc2.weight.detach() - snapshots[key]).abs().max().item()
                    assert delta > 0
                    first_audit[f'{key}_weight_max_update'] = delta
                dump_json(out / 'first_step_audit.json', first_audit)
                del snapshots
                print(json.dumps({'first_step_passed': first_audit}), flush=True)
            if step == 1 or step % 30 == 0 or step == len(train_loader):
                print(f"{config['variant']} epoch={epoch} step={step}/{len(train_loader)} "
                      f"loss={np.mean(losses):.5f} step_s={np.mean(step_times):.3f}", flush=True)
            if args.smoke:
                break
        train_seconds = time.perf_counter() - epoch_started
        allocated = torch.cuda.max_memory_allocated() / 2**20
        reserved = torch.cuda.max_memory_reserved() / 2**20
        # Drop retained output graphs before evaluation/checkpoint serialization.
        model.pred_mask = None
        model.features = None
        model.image_encoder.prompt_generator.structure_aux_logits = []
        optimizer.zero_grad(set_to_none=True)
        values = evaluate(val_loader, model, amp_dtype, max_batches=1 if args.smoke else None)
        row = dict(epoch=epoch, loss=float(np.mean(losses)), lr=optimizer.param_groups[0]['lr'],
                   train_seconds=train_seconds, train_step_seconds=float(np.mean(step_times)),
                   peak_allocated_mib=allocated, peak_reserved_mib=reserved,
                   prior=prior_values(model), **values)
        if auxiliary_losses:
            row['structure_aux_loss'] = float(np.mean(auxiliary_losses))
        print(json.dumps({'epoch_result': row}, ensure_ascii=False), flush=True)
        if args.smoke:
            dump_json(out / 'smoke_result.json', dict(run_info=info, audit=first_audit, result=row))
            return
        scheduler.step()
        metrics.append(row)
        dump_json(out / 'metrics.json', metrics)
        score = checkpoint_score(values, selection_metric)
        if score > best:
            best = score
            torch.save(dict(format='sam_adapters_v1', run_info=info, epoch=epoch,
                            metrics=row, model=checkpoint_model_state(model)), out / 'best_adapters.pth')
        state = dict(format='sam_adapters_v1', run_info=info, epoch=epoch, best=best,
                     trainable_names=trainable_names, model=checkpoint_model_state(model),
                     optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                              torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                              loader=generator.get_state()))
        temporary = out / 'last_state.tmp.pth'
        torch.save(state, temporary)
        os.replace(temporary, out / 'last_state.pth')
        del state
        dump_json(out / 'status.json', dict(status='checkpointed', epoch=epoch, stop_epoch=stop_epoch,
                                            selection_metric=selection_metric, best_score=best, last=row))
    print(f"COMPLETE {config['variant']} epochs={stop_epoch} best_{selection_metric}={best:.6f}", flush=True)


if __name__ == '__main__':
    main()

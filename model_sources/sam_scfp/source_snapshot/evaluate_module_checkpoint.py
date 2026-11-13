"""Reconstruct an adapter-format checkpoint on top of its original SAM weights."""

import argparse
import json
from pathlib import Path

import torch
import yaml

import models
from train_module_screen import configure_plan_a, evaluate, make_loader, seed_all


def load_model_for_evaluation(config_path, checkpoint_path):
    config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    seed_all(config['seed'])
    model = models.make(config['model'])
    base = torch.load(config['sam_checkpoint'], map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(base, strict=False)
    del base
    # Only load locally generated, trusted experiment checkpoints with pickle enabled.
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
    if state.get('format') != 'sam_adapters_v1':
        raise ValueError('Expected a locally generated sam_adapters_v1 checkpoint')
    if state['run_info']['variant'] != config['variant']:
        raise ValueError('Checkpoint and config variants differ')
    configure_plan_a(model)
    required = {name for name, p in model.named_parameters() if p.requires_grad}
    required.update(name for name, _ in model.named_buffers())
    if not required.issubset(state['model']):
        raise ValueError(f'Missing adaptation tensors: {sorted(required - set(state["model"]))}')
    loaded = model.load_state_dict(state['model'], strict=False)
    if loaded.unexpected_keys:
        raise ValueError(loaded.unexpected_keys)
    model.requires_grad_(False)
    return config, model.cuda().eval(), state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config, model, state = load_model_for_evaluation(args.config, args.checkpoint)
    _, loader = make_loader(config['val_dataset'], False)
    metrics = evaluate(loader, model, torch.bfloat16)
    result = dict(variant=config['variant'], saved_epoch=state['epoch'], metrics=metrics)
    if 'metrics' in state:
        result['absolute_building_iou_reproduction_delta'] = abs(
            metrics['building_iou'] - state['metrics']['building_iou'])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

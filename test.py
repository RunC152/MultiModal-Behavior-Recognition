"""
Evaluate a trained MultiModalBehaviorRecognition model.

Usage:
    # Evaluate best model with default config
    python test.py --checkpoint checkpoints/best_model.pth

    # Collect quality weights for visualization
    python test.py --checkpoint checkpoints/best_model.pth --save-weights weights.npz

    # Use a specific config (overrides checkpoint config)
    python test.py --config configs/default.yaml --checkpoint checkpoints/best_model.pth
"""

import argparse
import json
import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast
from tqdm import tqdm

from datasets.ntu_dataset import NTUDataset
from models.multimodal_model import MultiModalModel
from utils.metrics import accuracy


def load_checkpoint(path, device='cpu'):
    """Load checkpoint and extract model state + config."""
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get('config', None)
    epoch = checkpoint.get('epoch', 'unknown')
    best_acc1 = checkpoint.get('best_acc1', 0.0)
    return checkpoint['model_state_dict'], config, epoch, best_acc1


def evaluate(model, dataloader, criterion, device, config, collect_weights=False):
    """
    Evaluate model on a dataset.
    Returns metrics dict. If collect_weights, also returns arrays of w_rgb, w_ir, labels.
    """
    model.eval()

    total_loss = 0
    total_acc1 = 0
    total_acc5 = 0
    all_preds = []
    all_labels = []

    weights_rgb = []
    weights_ir = []
    weight_labels = []

    pbar = tqdm(dataloader, desc='Evaluating', leave=False)
    for batch in pbar:
        rgb_slow = batch['rgb_slow'].to(device)
        rgb_fast = batch['rgb_fast'].to(device)
        ir_slow = batch['ir_slow'].to(device)
        ir_fast = batch['ir_fast'].to(device)
        labels = batch['label'].to(device)

        with autocast(enabled=config['train']['use_amp']):
            if collect_weights:
                logits, w_rgb, w_ir = model(
                    rgb=[rgb_slow, rgb_fast], ir=[ir_slow, ir_fast],
                    return_weights=True)
            else:
                logits = model(rgb=[rgb_slow, rgb_fast], ir=[ir_slow, ir_fast])
            loss = criterion(logits, labels)

        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        _, preds = logits.topk(1, dim=1)

        total_loss += loss.item()
        total_acc1 += acc1.item()
        total_acc5 += acc5.item()
        all_preds.append(preds.cpu().squeeze(-1))
        all_labels.append(labels.cpu())

        if collect_weights:
            weights_rgb.append(w_rgb.cpu().squeeze(-1))
            weights_ir.append(w_ir.cpu().squeeze(-1))
            weight_labels.append(labels.cpu())

        pbar.set_postfix(loss=loss.item(), acc1=acc1.item())

    n = len(dataloader)
    metrics = {
        'loss': total_loss / n,
        'acc1': total_acc1 / n,
        'acc5': total_acc5 / n,
    }

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    metrics['per_class'] = per_class_accuracy(all_preds, all_labels)

    if collect_weights:
        weights = {
            'w_rgb': torch.cat(weights_rgb).numpy(),
            'w_ir': torch.cat(weights_ir).numpy(),
            'labels': torch.cat(weight_labels).numpy(),
        }
        return metrics, weights

    return metrics, None


def per_class_accuracy(preds, labels, num_classes=60):
    """Compute accuracy for each class (no sklearn dependency)."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, l in zip(preds, labels):
        cm[l, p] += 1
    class_totals = cm.sum(axis=1)
    acc = np.where(class_totals > 0,
                   cm.diagonal() / class_totals,
                   0.0)
    return {str(i): float(acc[i]) for i in range(num_classes)}


def print_metrics(metrics, title="Evaluation Results"):
    """Pretty print evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Loss:           {metrics['loss']:.4f}")
    print(f"  Top-1 Accuracy: {metrics['acc1']:.2f}%")
    print(f"  Top-5 Accuracy: {metrics['acc5']:.2f}%")

    per_class = metrics.get('per_class', {})
    if per_class:
        accs = list(per_class.values())
        print(f"  Per-class mean:  {np.mean(accs):.2f}%")
        print(f"  Per-class std:   {np.std(accs):.2f}%")
        print(f"  Per-class min:   {np.min(accs):.2f}% (class {int(np.argmin(accs))})")
        print(f"  Per-class max:   {np.max(accs):.2f}% (class {int(np.argmax(accs))})")

        # Bottom 5 classes
        sorted_idx = np.argsort(accs)
        print(f"\n  Bottom 5 classes:")
        for i in sorted_idx[:5]:
            print(f"    class {i:>2d}: {accs[i]:.2f}%")
        print(f"\n  Top 5 classes:")
        for i in sorted_idx[-5:][::-1]:
            print(f"    class {i:>2d}: {accs[i]:.2f}%")


def save_analysis(metrics, weights, output_dir):
    """Save per-class metrics and quality weights to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Per-class accuracy as JSON
    with open(os.path.join(output_dir, 'per_class_accuracy.json'), 'w') as f:
        json.dump(metrics['per_class'], f, indent=2)

    # Summary
    summary = {
        'loss': metrics['loss'],
        'acc1': metrics['acc1'],
        'acc5': metrics['acc5'],
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Quality weights for visualization
    if weights is not None:
        np.savez(os.path.join(output_dir, 'quality_weights.npz'), **weights)
        # Print quality stats
        w_rgb = weights['w_rgb']
        w_ir = weights['w_ir']
        print(f"\n  Quality weights stats:")
        print(f"    w_rgb: mean={w_rgb.mean():.4f} std={w_rgb.std():.4f}  "
              f"min={w_rgb.min():.4f} max={w_rgb.max():.4f}")
        print(f"    w_ir:  mean={w_ir.mean():.4f} std={w_ir.std():.4f}  "
              f"min={w_ir.min():.4f} max={w_ir.max():.4f}")

    print(f"\n  Results saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint (.pth)')
    parser.add_argument('--config', type=str, default=None,
                        help='Config file (overrides checkpoint config)')
    parser.add_argument('--save-weights', action='store_true',
                        help='Collect and save quality weights for visualization')
    parser.add_argument('--output-dir', type=str, default='eval_results',
                        help='Directory to save evaluation output')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size for evaluation')
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    state_dict, config, epoch, best_acc1 = load_checkpoint(args.checkpoint, device)
    if config is None and args.config is None:
        raise ValueError("No config found in checkpoint, please provide --config")
    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    print(f"  Trained epoch: {epoch}  |  Checkpoint best acc1: {best_acc1:.2f}%")

    # Override batch size for eval
    eval_batch_size = args.batch_size or config['data']['batch_size']

    # ── Data ──
    dataset = NTUDataset(
        rgb_dir=config['data']['rgb_dir'],
        ir_dir=config['data']['ir_dir'],
        slow_num_frames=config['data']['slow_num_frames'],
        fast_num_frames=config['data']['fast_num_frames'],
        side_size=config['data']['side_size'],
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config['train']['seed']),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        collate_fn=collate_valid_only,
    )
    print(f"Validation samples: {len(val_dataset)}")

    # ── Model ──
    model = MultiModalModel(
        model_type=config['model'].get('type', 'slowfast'),
        model_variant=config['model'].get('variant', None),
        rgb_weight=config['model']['rgb_weight'],
        ir_weight=config['model']['ir_weight'],
        feature_dim=config['model']['feature_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_classes=config['model']['num_classes'],
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # ── Evaluate ──
    criterion = nn.CrossEntropyLoss()
    metrics, weights = evaluate(
        model, val_loader, criterion, device, config,
        collect_weights=args.save_weights,
    )

    print_metrics(metrics)

    # ── Save ──
    save_analysis(metrics, weights, args.output_dir)


def collate_valid_only(batch):
    """Filter out invalid samples (label == -1)."""
    if not batch:
        return {}
    valid = [s for s in batch if s['label'] != -1]
    if not valid:
        return {}
    keys = valid[0].keys()
    return {k: torch.stack([s[k] for s in valid]) for k in keys}


if __name__ == '__main__':
    main()

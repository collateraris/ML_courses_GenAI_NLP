"""Shared checkpoint loading and atomic saving for training and sampling."""

import os
from pathlib import Path
import re

import torch

from model import ModelArgs


def load_latest_checkpoint(directory):
    """Return the checkpoint with the highest saved iteration and its path."""
    directory = Path(directory)
    periodic = [
        path for path in directory.glob('ckpt_iter_*.pt')
        if path.is_file() and re.fullmatch(r'ckpt_iter_\d+\.pt', path.name)
    ]
    # Prefer last_ckpt on ties: it also contains RNG and best-loss state.
    candidates = [directory / 'last_ckpt.pt']
    if periodic:
        candidates.append(max(periodic, key=lambda path: int(path.stem.split('_')[-1])))
    candidates.append(directory / 'best_ckpt.pt')

    latest_iteration = None
    latest_path = None
    metrics = []
    for path in candidates:
        if not path.is_file():
            continue
        # mmap avoids reading several multi-GB checkpoints into RAM just to
        # compare their iteration numbers. ModelArgs supports existing saves.
        with torch.serialization.safe_globals([ModelArgs]):
            checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        metrics.append((
            checkpoint.get('loss_version', 1),
            float(checkpoint.get('best_val_loss', checkpoint.get('val_loss', float('inf')))),
            float(checkpoint.get('best_val_main_loss', checkpoint.get('val_main_loss', float('inf')))),
        ))
        if latest_iteration is None or checkpoint['iter_num'] > latest_iteration:
            latest_iteration = checkpoint['iter_num']
            latest_path = path
        del checkpoint

    if latest_path is None:
        return None, None
    # Release the mappings before loading the selected state. Keeping mapped
    # optimizer tensors alive would prevent replacing this file on Windows.
    with torch.serialization.safe_globals([ModelArgs]):
        checkpoint = torch.load(latest_path, map_location='cpu', weights_only=True)
    # Scores from the old, doubly shifted objective are not comparable with
    # the corrected next-token objective. Merge only matching loss versions.
    loss_version = checkpoint.get('loss_version', 1)
    comparable_metrics = [metric for metric in metrics if metric[0] == loss_version]
    checkpoint['best_val_loss'] = min(metric[1] for metric in comparable_metrics)
    checkpoint['best_val_main_loss'] = min(metric[2] for metric in comparable_metrics)
    return checkpoint, latest_path


def save_checkpoint(checkpoint, path):
    """Replace a checkpoint only after its new contents have been written."""
    path = os.fspath(path)
    temporary_path = path + '.tmp'
    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

#!/usr/bin/env python3
"""
MemFlowDiT Checkpoint Manager
Keeps: recent N checkpoints (rolling) + milestone checkpoints + latest
"""
import sys
import os
from pathlib import Path
import re

def manage_checkpoints(
    checkpoint_dir: str,
    keep_recent: int = 20,
    milestone_every: int = 1000,
    current_step: int = None,
):
    """
    Args:
        checkpoint_dir: checkpoint directory
        keep_recent: keep last N regular checkpoints (rolling window)
        milestone_every: save milestone every N steps (permanent)
        current_step: current training step
    """
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        return
    
    # Pattern: step_XXXXXX.pt
    step_pattern = re.compile(r'^step_(\d{6})\.pt$')
    milestone_pattern = re.compile(r'^milestone_(\d{6})\.pt$')
    
    regular_ckpts = []
    milestone_ckpts = []
    
    for f in ckpt_path.iterdir():
        if not f.is_file():
            continue
        match_regular = step_pattern.match(f.name)
        match_milestone = milestone_pattern.match(f.name)
        
        if match_regular:
            step_num = int(match_regular.group(1))
            regular_ckpts.append((step_num, f))
        elif match_milestone:
            step_num = int(match_milestone.group(1))
            milestone_ckpts.append((step_num, f))
    
    # Sort by step number
    regular_ckpts.sort(key=lambda x: x[0])
    milestone_ckpts.sort(key=lambda x: x[0])
    
    # Promote milestone checkpoints
    if current_step and milestone_every > 0:
        if current_step % milestone_every == 0:
            regular_name = ckpt_path / f"step_{current_step:06d}.pt"
            milestone_name = ckpt_path / f"milestone_{current_step:06d}.pt"
            if regular_name.exists() and not milestone_name.exists():
                regular_name.rename(milestone_name)
                print(f"[CheckpointManager] Promoted step_{current_step:06d}.pt to milestone", flush=True)
                # Refresh list
                regular_ckpts = [(s, f) for s, f in regular_ckpts if s != current_step]
                milestone_ckpts.append((current_step, milestone_name))
                milestone_ckpts.sort(key=lambda x: x[0])
    
    # Keep only recent N regular checkpoints
    if len(regular_ckpts) > keep_recent:
        to_delete = regular_ckpts[:-keep_recent]
        for step_num, fpath in to_delete:
            try:
                fpath.unlink()
                print(f"[CheckpointManager] Deleted old checkpoint: {fpath.name}", flush=True)
            except Exception as e:
                print(f"[CheckpointManager] Failed to delete {fpath.name}: {e}", flush=True)
    
    # Report status
    remaining_regular = len(regular_ckpts) - len([s for s, f in regular_ckpts if not f.exists()])
    remaining_milestone = len([s for s, f in milestone_ckpts if f.exists()])
    print(f"[CheckpointManager] Status: {remaining_regular} regular + {remaining_milestone} milestones", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", help="Checkpoint directory")
    parser.add_argument("--keep_recent", type=int, default=20, help="Keep recent N checkpoints")
    parser.add_argument("--milestone_every", type=int, default=1000, help="Milestone every N steps")
    parser.add_argument("--current_step", type=int, default=None, help="Current step (for milestone promotion)")
    args = parser.parse_args()
    
    manage_checkpoints(
        args.checkpoint_dir,
        keep_recent=args.keep_recent,
        milestone_every=args.milestone_every,
        current_step=args.current_step,
    )

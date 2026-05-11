#!/usr/bin/env python3
"""
Launch and monitor the stable GRPO long run.

The monitor evaluates every saved checkpoint interval, tracks the best full-test
checkpoint, and stops training when full-test quality drops below the base line.
"""

import datetime
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / 'configs' / 'btcv_diffusion_dit_v3_4_fm_yolom_grpo_stable_long.yaml'
OUT_DIR = ROOT / 'data' / 'outputs' / 'btcv_diffusion_dit_v3_4_fm_yolom_grpo_stable_long'
CKPT_DIR = OUT_DIR / 'checkpoints'
LOG_DIR = OUT_DIR / 'posttrain_grpo'
MONITOR_LOG = LOG_DIR / 'monitor.jsonl'
MONITOR_SUMMARY = LOG_DIR / 'monitor_summary.json'

BASE_IOU = float(os.environ.get('GRPO_BASE_IOU', '0.892484'))
BASE_MBOUNDF = float(os.environ.get('GRPO_BASE_MBOUNDF', '0.775513'))
STEP300_IOU = float(os.environ.get('GRPO_STEP300_IOU', '0.894166'))


def now():
    return datetime.datetime.now().isoformat(timespec='seconds')


def write_event(event):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    event = {'timestamp': now(), **event}
    with MONITOR_LOG.open('a', encoding='utf-8') as f:
        f.write(json.dumps(event, ensure_ascii=False) + '\n')
    print(json.dumps(event, ensure_ascii=False), flush=True)


def latest_step(ckpt_path):
    if not ckpt_path.exists():
        return None
    try:
        obj = torch.load(str(ckpt_path), map_location='cpu')
        return int(obj.get('step', 0) or 0)
    except Exception as exc:
        write_event({'event': 'checkpoint_read_failed', 'path': str(ckpt_path), 'error': str(exc)})
        return None


def load_evaluated_steps():
    done = set()
    if not MONITOR_LOG.exists():
        return done
    with MONITOR_LOG.open('r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
            except Exception:
                continue
            if item.get('event') == 'eval_done' and 'step' in item:
                done.add(int(item['step']))
    return done


def read_recent_training_stats(n=120):
    log_path = LOG_DIR / 'logs.jsonl'
    if not log_path.exists():
        return {}
    rows = []
    with log_path.open('r', encoding='utf-8') as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if not rows:
        return {}
    by_step = {}
    for row in rows:
        if isinstance(row.get('step'), int):
            by_step[row['step']] = row
    rows = [by_step[k] for k in sorted(by_step)][-n:]
    out = {'last_step': rows[-1].get('step')}
    for key in ('reward_mean', 'final_score_mean', 'kl_loss', 'approx_kl', 'clipfrac'):
        vals = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
        if vals:
            out[f'{key}_mean'] = sum(vals) / len(vals)
            out[f'{key}_last'] = vals[-1]
    return out


def query_gpu(index):
    cmd = [
        'nvidia-smi',
        '--query-gpu=index,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits',
    ]
    out = subprocess.check_output(cmd, text=True)
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 3:
            continue
        if parts[0] == str(index):
            return {'free_mb': int(parts[1]), 'util_pct': int(parts[2])}
    raise RuntimeError(f'GPU {index} not found')


def wait_for_gpu(index, role):
    min_free_mb = int(os.environ.get('MONITOR_GPU_MIN_FREE_MB', '30000'))
    max_util_pct = int(os.environ.get('MONITOR_GPU_MAX_UTIL', '30'))
    interval = int(os.environ.get('MONITOR_GPU_WAIT_SECONDS', '60'))
    while True:
        try:
            stat = query_gpu(index)
        except Exception as exc:
            write_event({'event': 'gpu_query_failed', 'gpu': str(index), 'role': role, 'error': str(exc)})
            time.sleep(interval)
            continue
        if stat['free_mb'] >= min_free_mb and stat['util_pct'] <= max_util_pct:
            write_event({'event': 'gpu_ready', 'gpu': str(index), 'role': role, **stat})
            return
        write_event({'event': 'gpu_wait', 'gpu': str(index), 'role': role, **stat, 'min_free_mb': min_free_mb, 'max_util_pct': max_util_pct})
        time.sleep(interval)


def run_eval(step, eval_gpu, ode_steps):
    wait_for_gpu(eval_gpu, role='eval')
    ckpt = CKPT_DIR / 'latest.pt'
    save_dir = ROOT / 'visual' / f'v3_4_fm_yolom_grpo_stable_long_eval_step{step:04d}'
    env = os.environ.copy()
    env.update({
        'CFG_FILE': str(CFG),
        'CKPT': str(ckpt),
        'CUDA_VISIBLE_DEVICES': str(eval_gpu),
        'EVAL_GPU': str(eval_gpu),
        'ODE_STEPS': str(ode_steps),
        'SAVE_VISUALS': '0',
        'SAVE_DIR': str(save_dir),
        'EVAL_SEED': env.get('EVAL_SEED', '20260504'),
    })
    cmd = [sys.executable, str(ROOT / 'scripts' / 'eval_v37_full_iou.py')]
    write_event({'event': 'eval_start', 'step': step, 'cmd': ' '.join(cmd), 'save_dir': str(save_dir)})
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    eval_log = LOG_DIR / f'eval_step{step:04d}.log'
    eval_log.write_text(proc.stdout, encoding='utf-8')
    if proc.returncode != 0:
        raise RuntimeError(f'eval failed for step {step}, see {eval_log}')
    summaries = sorted(glob.glob(str(save_dir / 'v3_7_full_test_iou_*.json')))
    if not summaries:
        raise RuntimeError(f'eval summary not found in {save_dir}')
    with open(summaries[-1], 'r', encoding='utf-8') as f:
        summary = json.load(f)
    return {
        'summary_path': summaries[-1],
        'iou': float(summary.get('mean_iou_sample_avg', 0.0)),
        'mboundf': float(summary.get('mean_mboundf_sample_avg', 0.0)),
        'dice': float(summary.get('mean_dice_sample_avg', 0.0)),
        'failed': int(summary.get('failed_samples', 0)),
    }


def terminate(proc):
    if proc is None or proc.poll() is not None:
        return
    write_event({'event': 'terminate_train', 'pid': proc.pid})
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.kill(proc.pid, signal.SIGKILL)


def main():
    train_gpu = os.environ.get('TRAIN_GPU', '4')
    eval_gpu = os.environ.get('EVAL_GPU_MONITOR', '5')
    poll_seconds = int(os.environ.get('MONITOR_POLL_SECONDS', '30'))
    eval_every_step = int(os.environ.get('MONITOR_EVAL_EVERY_STEP', '200'))
    ode_steps = int(os.environ.get('MONITOR_ODE_STEPS', '10'))
    attach = os.environ.get('MONITOR_ATTACH', '0') == '1'

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    proc = None
    if not attach:
        wait_for_gpu(train_gpu, role='train')
        train_env = os.environ.copy()
        train_env.update({
            'CFG_FILE': str(CFG),
            'CUDA_VISIBLE_DEVICES': str(train_gpu),
            'GRPO_TRAIN_STEPS': train_env.get('GRPO_TRAIN_STEPS', '2000'),
            'GRPO_SAVE_EVERY': train_env.get('GRPO_SAVE_EVERY', '100'),
        })
        train_log = LOG_DIR / f'train_stable_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        train_f = train_log.open('w', encoding='utf-8')
        cmd = [sys.executable, str(ROOT / 'grpo_train.py')]
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=train_env, stdout=train_f, stderr=subprocess.STDOUT, text=True)
        write_event({'event': 'train_start', 'pid': proc.pid, 'train_gpu': train_gpu, 'train_log': str(train_log)})
    else:
        write_event({'event': 'monitor_attach'})

    evaluated = load_evaluated_steps()
    bad_iou_count = 0
    bad_mbf_count = 0
    best_iou = STEP300_IOU
    best_step = 300
    stop_reason = ''

    try:
        while True:
            if proc is not None and proc.poll() is not None:
                write_event({'event': 'train_exited', 'returncode': proc.returncode})
                break

            step = latest_step(CKPT_DIR / 'latest.pt')
            stats = read_recent_training_stats()
            if step is not None:
                write_event({'event': 'heartbeat', 'step': step, **stats})
                if step > 0 and step % eval_every_step == 0 and step not in evaluated:
                    try:
                        result = run_eval(step, eval_gpu=eval_gpu, ode_steps=ode_steps)
                    except Exception as exc:
                        write_event({'event': 'eval_failed', 'step': step, 'error': str(exc)})
                        time.sleep(poll_seconds)
                        continue
                    evaluated.add(step)
                    write_event({'event': 'eval_done', 'step': step, **result})

                    if result['iou'] > best_iou:
                        best_iou = result['iou']
                        best_step = step
                        shutil.copy2(CKPT_DIR / 'latest.pt', CKPT_DIR / 'best.pt')
                        write_event({'event': 'best_update', 'step': step, 'iou': best_iou, 'path': str(CKPT_DIR / 'best.pt')})

                    bad_iou_count = bad_iou_count + 1 if result['iou'] < BASE_IOU else 0
                    bad_mbf_count = bad_mbf_count + 1 if result['mboundf'] < BASE_MBOUNDF else 0
                    if bad_iou_count >= 2:
                        stop_reason = f'IoU below base for 2 evals: last={result["iou"]:.6f}, base={BASE_IOU:.6f}'
                        break
                    if bad_mbf_count >= 2:
                        stop_reason = f'mBoundF below base for 2 evals: last={result["mboundf"]:.6f}, base={BASE_MBOUNDF:.6f}'
                        break

            if stats.get('last_step', 0) and stats.get('last_step', 0) >= 300:
                if abs(float(stats.get('kl_loss_mean', 0.0))) < 1e-12:
                    stop_reason = 'KL loss stayed zero after warmup; fixed reference constraint is not active'
                    break

            target = int(os.environ.get('GRPO_TRAIN_STEPS', '2000'))
            if step is not None and step >= target:
                write_event({'event': 'target_reached', 'step': step})
                break
            time.sleep(poll_seconds)
    finally:
        if stop_reason:
            write_event({'event': 'stop_triggered', 'reason': stop_reason})
            terminate(proc)
        summary = {
            'timestamp': now(),
            'best_step': best_step,
            'best_iou': best_iou,
            'stop_reason': stop_reason,
            'base_iou': BASE_IOU,
            'base_mboundf': BASE_MBOUNDF,
        }
        MONITOR_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        write_event({'event': 'monitor_done', **summary})


if __name__ == '__main__':
    main()

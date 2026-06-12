#!/usr/bin/env python3
import os
import sys
import time

import torch

_THIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _THIS_DIR)

if not os.environ.get('CFG_FILE'):
    os.environ['CFG_FILE'] = os.path.join(
        _THIS_DIR,
        'configs',
        '1232_final_v8_heatmap_extreme_diffusion_from_detext_gpu3.yaml',
    )

from lib.config import cfg
from lib.datasets import make_data_loader
from lib.networks import make_network
from lib.train.trainers import make_trainer


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(label, fn):
    sync()
    t0 = time.time()
    out = fn()
    sync()
    print(f'{label}: {time.time() - t0:.3f}s', flush=True)
    return out


def move_batch_to_cuda(batch):
    for k in list(batch.keys()):
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].cuda(non_blocking=True)
    return batch


def main():
    batch_size = int(os.environ.get('DEBUG_BATCH_SIZE', '1'))
    cfg.train.batch_size = batch_size
    cfg.train.num_workers = 0
    cfg.dataloader_persistent_workers = False
    cfg.dataloader_prefetch_factor = 0

    print(f'cfg={os.environ.get("CFG_FILE")} batch_size={batch_size}', flush=True)
    print(f'detector_backend={getattr(cfg, "detector_backend", "")}', flush=True)

    net = make_network(cfg)
    trainer = make_trainer(cfg, net)
    wrapper = trainer.network.cuda().train()

    loader = make_data_loader(cfg, is_train=True, is_distributed=False)
    batch = timed('load_batch', lambda: next(iter(loader)))
    batch = move_batch_to_cuda(batch)

    for key in ('inp', 'ct_hm', 'wh', 'ct_ind', 'ct_01', 'i_gt_py'):
        val = batch.get(key)
        if isinstance(val, torch.Tensor):
            print(f'batch[{key}] shape={tuple(val.shape)} dtype={val.dtype}', flush=True)

    output = timed('network_forward', lambda: wrapper.net(batch['inp'], batch))
    for key in ('ct_hm', 'wh', 'ct', 'detection', 'ex_pred', 'i_gt_4py', 'diff_loss', 'py'):
        val = output.get(key)
        if isinstance(val, torch.Tensor):
            print(f'output[{key}] shape={tuple(val.shape)} dtype={val.dtype}', flush=True)
        else:
            print(f'output[{key}] type={type(val).__name__}', flush=True)

    losses = []
    if wrapper.ct_crit is not None and 'ct_hm' in output and 'wh' in output:
        def det_loss_fn():
            ct_target = batch['ct_hm'].to(output['ct_hm'].device)
            wh_target = batch['wh'].to(output['wh'].device)
            ct_ind = batch['ct_ind'].to(output['wh'].device)
            ct_mask = batch['ct_01'].to(output['wh'].device)
            ct_loss = wrapper.ct_crit(output['ct_hm'], ct_target)
            wh_loss = wrapper.wh_crit(output['wh'], wh_target, ct_ind, ct_mask)
            return ct_loss + float(getattr(cfg, 'heatmap_wh_weight', 0.1)) * wh_loss, ct_loss, wh_loss

        det_loss, ct_loss, wh_loss = timed('det_loss', det_loss_fn)
        print(
            f'det_loss={float(det_loss.detach().item()):.6f} '
            f'ct={float(ct_loss.detach().item()):.6f} wh={float(wh_loss.detach().item()):.6f}',
            flush=True,
        )
        losses.append(det_loss)

    if 'ex_pred' in output and 'i_gt_4py' in output:
        ex_target = output['i_gt_4py'].to(device=output['ex_pred'].device, dtype=output['ex_pred'].dtype)
        ex_loss = timed('ex_loss', lambda: wrapper.ex_crit(output['ex_pred'], ex_target))
        print(f'ex_loss={float(ex_loss.detach().item()):.6f}', flush=True)
        losses.append(ex_loss)

    if 'diff_loss' in output:
        diff_loss = output['diff_loss']
        print(f'diff_loss={float(diff_loss.detach().item()):.6f}', flush=True)
        losses.append(diff_loss)

    total = sum(losses)
    print(f'total_loss={float(total.detach().item()):.6f}', flush=True)
    timed('backward', lambda: total.backward())


if __name__ == '__main__':
    main()

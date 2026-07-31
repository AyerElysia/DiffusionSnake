"""Diagnose which coordinate space the predicted contours live in.

Loads a few foreground val slices with GT boxes, then prints, for one sample:
  - GT detection boxes returned by use_gt_detection (input space, 512)
  - the initial contour handed to evolution (output space, 128)
  - the final network output['py'][-1]
  - the same contour after *down_ratio and inverse affine, vs the GT mask bbox

Run:
  CFG_FILE=configs/sagittal_2d_v4_6c_moonvit_train.yaml \
  CUDA_VISIBLE_DEVICES=7 python tools/diag_contour_space.py --ckpt <path>
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg-file', default='configs/sagittal_2d_v4_6c_moonvit_train.yaml')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--max-slices', type=int, default=40)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


ARGS = _parse_args()
os.environ['CFG_FILE'] = ARGS.cfg_file
# lib.config parses sys.argv on import; hide our own flags from it.
sys.argv = [sys.argv[0]]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lib.config import cfg  # noqa: E402
from lib.datasets.make_dataset import make_data_loader  # noqa: E402
from lib.networks import make_network  # noqa: E402
from lib.evaluators.sagittal_2d_fixed.snake import (  # noqa: E402
    configure_box_mode,
    inverse_affine_points,
)
from lib.utils.snake import snake_config  # noqa: E402


def _rng(name, arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        print('  {:<22} empty'.format(name))
        return
    print('  {:<22} x[{:8.2f},{:8.2f}] y[{:8.2f},{:8.2f}]'.format(
        name, arr[..., 0].min(), arr[..., 0].max(), arr[..., 1].min(), arr[..., 1].max()))


def main():
    cfg.test.dataset = 'SagittalPseudo3DVal'
    cfg.test.batch_size = 1
    cfg.train.num_workers = 0
    configure_box_mode(cfg, 'gt')

    device = torch.device(ARGS.device if torch.cuda.is_available() else 'cpu')
    loader = make_data_loader(cfg, is_train=False)
    network = make_network(cfg).to(device)
    state = torch.load(ARGS.ckpt, map_location='cpu')
    net_state = state.get('net', state)
    missing, unexpected = network.load_state_dict(net_state, strict=False)
    print('[ckpt] epoch={} missing={} unexpected={}'.format(
        state.get('epoch'), len(missing), len(unexpected)))
    network.eval()

    shown = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= ARGS.max_slices or shown >= 2:
                break
            ct_01 = batch['ct_01'].bool()
            if int(ct_01.sum()) == 0:
                continue
            inp = batch['inp'].to(device)
            moved = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and not k.startswith('locate_feat'):
                    moved[k] = v.to(device)
                else:
                    moved[k] = v
            output = network(inp, moved)

            print('=== sample {} inp={} n_gt={}'.format(
                i, tuple(inp.shape), int(ct_01.sum())))
            det = output['detection'].detach().cpu().numpy()
            print('  feat_hw={} down_ratio={} ro={}'.format(
                output.get('feat_hw'), snake_config.down_ratio, snake_config.ro))
            if det.size:
                _rng('gt_detection(box)', det[0, :, :4].reshape(-1, 2))
            for key in ('it_ex', 'it_py'):
                if key in output:
                    _rng(key, output[key].detach().cpu().numpy())
            pys = output.get('py')
            if isinstance(pys, torch.Tensor):
                pys = [pys]
            if pys is not None and len(pys) > 0:
                _rng('py[0]', pys[0].detach().cpu().numpy())
                _rng('py[-1]', pys[-1].detach().cpu().numpy())
                final = pys[-1].detach().cpu().numpy()
                _rng('py[-1]*down_ratio', final * float(snake_config.down_ratio))

            meta = batch['meta']
            inv = np.asarray(meta['inv_trans_input'], dtype=np.float32).reshape(2, 3)
            orig_hw = np.asarray(meta['orig_hw']).reshape(-1)[:2]
            print('  orig_hw={}'.format(tuple(int(x) for x in orig_hw)))
            if pys:
                restored = inverse_affine_points(
                    final[0] * float(snake_config.down_ratio), inv, orig_hw, flipped=False)
                _rng('restored[0]', restored)
                nodr = inverse_affine_points(final[0], inv, orig_hw, flipped=False)
                _rng('restored[0] no *dr', nodr)

            # GT poly / box in original space for reference
            if 'i_gt_py' in batch:
                _rng('batch.i_gt_py', batch['i_gt_py'].detach().cpu().numpy())
            wh = batch['wh'][0][ct_01[0]].detach().cpu().numpy()
            print('  gt wh(feat) mean={}'.format(np.round(wh.mean(axis=0), 2)))
            shown += 1


if __name__ == '__main__':
    main()

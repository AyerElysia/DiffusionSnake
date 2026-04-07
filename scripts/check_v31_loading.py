import torch
import sys
import os

os.chdir('/home/medteam/Zhrch/DiffusionSnake-12-30')
sys.path.insert(0, '.')

os.environ['CFG_FILE'] = 'configs/btcv_diffusion_dit_v3_1.yaml'

from lib.config import cfg, args
cfg.merge_from_file('configs/btcv_diffusion_dit_v3_1.yaml')

from lib.networks import make_network
from lib.train.trainers import make_trainer

network = make_network(cfg)
trainer = make_trainer(cfg, network)

ckpt = torch.load('data/outputs/btcv_diffusion_dit_v3_1/checkpoints/latest.pt', map_location='cpu')
sd = ckpt['state_dict']

wrapper = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
info = wrapper.load_state_dict(sd, strict=False)

print('=== Missing Keys ===')
print(f'Count: {len(info.missing_keys)}')
for k in info.missing_keys[:20]:
    print(f'  {k}')

print('\n=== Unexpected Keys ===')
print(f'Count: {len(info.unexpected_keys)}')
for k in info.unexpected_keys[:20]:
    print(f'  {k}')
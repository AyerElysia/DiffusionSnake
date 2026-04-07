"""Check V3.1 PatchifyEmbedding weights"""
import torch

# Load V3.1 checkpoint (correct path)
ckpt_path = 'data/outputs/btcv_diffusion_dit_v3_1/checkpoints/latest.pt'
ckpt = torch.load(ckpt_path, map_location='cpu')

# Get the model state dict
if 'model_state_dict' in ckpt:
    state_dict = ckpt['model_state_dict']
elif 'state_dict' in ckpt:
    state_dict = ckpt['state_dict']
else:
    state_dict = ckpt

print('=== V3.1 PatchifyEmbedding weights analysis ===')
for key in state_dict.keys():
    if 'image_embed' in key:
        w = state_dict[key]
        print(f'{key}: shape={w.shape}, mean={w.mean():.6f}, std={w.std():.6f}, min={w.min():.6f}, max={w.max():.6f}')

# Compare with V3.0 Perceiver
ckpt_v3_path = 'data/outputs/btcv_diffusion_dit_v3/checkpoints/latest.pt'
ckpt_v3 = torch.load(ckpt_v3_path, map_location='cpu')

if 'model_state_dict' in ckpt_v3:
    state_dict_v3 = ckpt_v3['model_state_dict']
elif 'state_dict' in ckpt_v3:
    state_dict_v3 = ckpt_v3['state_dict']
else:
    state_dict_v3 = ckpt_v3

print('\n=== V3.0 Perceiver weights analysis ===')
for key in state_dict_v3.keys():
    if 'global_compressor' in key:
        w = state_dict_v3[key]
        print(f'{key}: shape={w.shape}, mean={w.mean():.6f}, std={w.std():.6f}')
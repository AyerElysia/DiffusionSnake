import argparse
import json
import re
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Merge heatmap detector and diffusion checkpoints.")
    parser.add_argument("--detector-ckpt", required=True)
    parser.add_argument("--diffusion-ckpt", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def unwrap_state_dict(state_dict):
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("module.net.") for k in keys):
        return {k[len("module.net."):]: v for k, v in state_dict.items()}
    if all(k.startswith("net.") for k in keys):
        return {k[4:]: v for k, v in state_dict.items()}
    if all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_state(path):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("state_dict") or ckpt.get("model") or ckpt.get("net") or ckpt
    return ckpt, unwrap_state_dict(state)


def remap_legacy_state_dict(sd):
    legacy_re = re.compile(r'(\.?)time_emb_(\d)(\..*)')
    remapped = {}
    for key, value in sd.items():
        if legacy_re.search(key):
            remapped[legacy_re.sub(r'\1time_emb_net.\2\3', key)] = value
        else:
            remapped[key] = value
    return remapped


def add_net_prefix(state_dict):
    prefixed = {}
    for key, value in state_dict.items():
        if key.startswith("net."):
            prefixed[key] = value
        else:
            prefixed[f"net.{key}"] = value
    return prefixed


def main():
    args = parse_args()
    detector_path = Path(args.detector_ckpt).resolve()
    diffusion_path = Path(args.diffusion_ckpt).resolve()
    output_path = Path(args.output).resolve()

    det_ckpt, det_state = load_state(detector_path)
    diff_ckpt, diff_state = load_state(diffusion_path)

    diff_state = remap_legacy_state_dict(diff_state)
    diff_state = add_net_prefix(diff_state)
    det_state = add_net_prefix(det_state)

    merged = {}
    merged.update(diff_state)
    for key, value in det_state.items():
        if key.startswith("net.heatmap_detector."):
            merged[key] = value

    payload = {
        "state_dict": merged,
        "step": 0,
        "source": {
            "detector_ckpt": str(detector_path),
            "diffusion_ckpt": str(diffusion_path),
            "detector_keys": len([k for k in det_state if k.startswith("net.heatmap_detector.")]),
            "diffusion_keys": len(diff_state),
            "merged_keys": len(merged),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(json.dumps(payload["source"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

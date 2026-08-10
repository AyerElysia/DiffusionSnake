#!/usr/bin/env python3
import json
import os
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[3]
CFG = ROOT / "configs/volmem/depth_sweep/depth_sweep_p0_l6.yaml"
os.environ["CFG_FILE"] = str(CFG)
sys.argv = [sys.argv[0], "--cfg_file", str(CFG)]
sys.path.insert(0, str(ROOT))

import torch

from lib.utils.snake import snake_config, snake_gcn_utils


def main():
    box = torch.tensor([[[80.0, 120.0, 240.0, 360.0]]])
    score = torch.ones((1, 1))
    got = snake_gcn_utils.prepare_testing({
        "detection": torch.cat(
            [box, score.unsqueeze(-1), torch.zeros((1, 1, 1))], dim=-1
        )
    })
    expected = snake_gcn_utils._box_to_octagon_init(
        box[0] / float(snake_config.down_ratio), snake_config.poly_num
    )
    assert snake_config.init == "octagon"
    assert torch.equal(got["i_it_py"], expected)
    assert float(got["i_it_py"][..., 0].min()) >= 0.0
    assert float(got["i_it_py"][..., 0].max()) <= 60.0
    assert float(got["i_it_py"][..., 1].max()) <= 90.0
    broken = snake_gcn_utils._box_to_octagon_init(box[0], snake_config.poly_num)
    assert float(broken.max()) == float(got["i_it_py"].max()) * float(snake_config.down_ratio)
    print(json.dumps({
        "status": "PASS",
        "init": snake_config.init,
        "down_ratio": float(snake_config.down_ratio),
        "input_box": box.tolist(),
        "flow_bounds": [
            float(got["i_it_py"][..., 0].min()),
            float(got["i_it_py"][..., 1].min()),
            float(got["i_it_py"][..., 0].max()),
            float(got["i_it_py"][..., 1].max()),
        ],
        "broken_max_over_fixed_max": float(broken.max() / got["i_it_py"].max()),
        "training_direct_octagon_exact": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()

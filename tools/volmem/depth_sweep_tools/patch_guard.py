"""Make the MemFlow checkpoint-compatibility guard config-driven.

Default stays 0.80 so mainline behaviour is bit-identical. Only a config that
explicitly sets train.memflow_min_compat_ratio can relax it -- used for the
L=12 depth arm, where 6 of 12 DiT layers are deliberately fresh (448/610 =
73.4% pretrained), which the fixed 80% guard cannot express.

Idempotent: re-running is a no-op.
"""
import io
import os
import re
import sys

TRAINER = "tools/volmem/train_memflowdit.py"
CFG = "configs/volmem/depth_sweep/depth_sweep_p3_l12.yaml"

OLD = '''    info = model.load_state_dict(compatible, strict=False)
    if len(compatible) < int(len(target) * 0.80):
        raise RuntimeError(
            "MemFlow checkpoint compatibility below 80%: {}/{}".format(
                len(compatible), len(target)
            )
        )
'''

NEW = '''    info = model.load_state_dict(compatible, strict=False)
    # Depth-sweep: the guard exists to catch a mis-specified checkpoint. A
    # deliberate depth increase legitimately lowers the pretrained fraction
    # (each new DiT layer adds 18 block keys + 9 memflow-adapter keys), so the
    # threshold is config-driven. Default 0.80 == previous hardcoded behaviour.
    min_ratio = float(getattr(cfg.train, "memflow_min_compat_ratio", 0.80))
    if len(compatible) < int(len(target) * min_ratio):
        raise RuntimeError(
            "MemFlow checkpoint compatibility below {:.1f}%: {}/{}".format(
                min_ratio * 100.0, len(compatible), len(target)
            )
        )
'''


def patch_trainer():
    with io.open(TRAINER, "r", encoding="utf-8") as handle:
        src = handle.read()
    if "memflow_min_compat_ratio" in src:
        print("[trainer] already patched, skipping")
        return False
    if src.count(OLD) != 1:
        print("[trainer] FAIL: expected exactly 1 guard block, found {}".format(
            src.count(OLD)))
        sys.exit(1)
    src = src.replace(OLD, NEW)
    with io.open(TRAINER, "w", encoding="utf-8") as handle:
        handle.write(src)
    print("[trainer] guard is now config-driven (default 0.80)")
    return True


def patch_cfg():
    with io.open(CFG, "r", encoding="utf-8") as handle:
        src = handle.read()
    if "memflow_min_compat_ratio" in src:
        print("[cfg] already patched, skipping")
        return False
    # anchor on the key my earlier depth patch already injects
    anchor = "  new_layer_base_depth: 6\n"
    if src.count(anchor) != 1:
        print("[cfg] FAIL: expected exactly 1 anchor, found {}".format(
            src.count(anchor)))
        sys.exit(1)
    src = src.replace(
        anchor,
        anchor + "  memflow_min_compat_ratio: 0.70  # L=12: 448/610=73.4% pretrained\n",
    )
    with io.open(CFG, "w", encoding="utf-8") as handle:
        handle.write(src)
    print("[cfg] memflow_min_compat_ratio: 0.70 injected")
    return True


if __name__ == "__main__":
    if not os.path.isfile(TRAINER):
        print("run me from the worktree root")
        sys.exit(1)
    patch_trainer()
    patch_cfg()

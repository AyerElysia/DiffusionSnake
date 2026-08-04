"""Phase0-A: dump per-slice contours + GT for 3D refinement diagnostics.

Read-only probe. Loads a fixed VolMem MemFlowDiT checkpoint, runs predict_step
over whole volumes, and dumps for every slice:
  - predicted contours (original image space, float)
  - matched GT contour points (nearest-neighbour on GT boundary)
  - GT mask boundary points
  - per-slice bookkeeping (class label, box, foreground pixel counts)

Output is an .npz per volume plus an index.json, consumed by the 3D metric
suite and the oracle action-family analysis.

Nothing here writes into the training loop. Peak GPU memory is ~2.5GB.
"""
import argparse
import contextlib
import json
import os
import sys
import time
from collections import OrderedDict, defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--box-mode", choices=("gt", "predicted"), default="gt")
    parser.add_argument("--memory-mode",
                        choices=("autoregressive", "oracle", "off"),
                        default="autoregressive")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cache-root", default=None,
                        help="Override cfg.locate_feat_cache_root. Point this at "
                             "a tmpfs copy when the shared disk is saturated.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-volumes", type=int, default=None)
    parser.add_argument("--volume-start", type=int, default=0)
    parser.add_argument("--only-volumes", default=None,
                        help="Comma-separated case ids. Lets a run be pinned to "
                             "volumes already staged in tmpfs.")
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


ARGS = parse_args()
os.environ["CFG_FILE"] = ARGS.cfg_file
# The inherited config module parses argv during import.
sys.argv = [sys.argv[0], "--cfg_file", ARGS.cfg_file]

import cv2
import numpy as np
import torch

from scipy.spatial import cKDTree

from lib.config import cfg
from lib.datasets.collate_batch import snake_collator
from lib.evaluators.sagittal_2d_fixed import Evaluator, configure_box_mode
from lib.evaluators.sagittal_2d_fixed.snake import inverse_affine_points
from lib.networks import make_network
from lib.train.trainers.make_trainer import _wrapper_factory
from lib.utils.snake import snake_config
from volmem.adapters import (
    V46cContourAdapter,
    build_detection_provider,
    configure_single_slice_compatibility,
    make_single_slice_dataset_class,
)
from volmem.adapters.legacy_dataset import align_mask_to_token_grid
from volmem.models import MemFlowDiTSnake, SliceSequenceMeta


def amp_context(use_amp):
    """Fresh autocast (or no-op) context, matching the eval script's dtype."""
    if use_amp:
        return torch.cuda.amp.autocast(dtype=torch.float16)
    return contextlib.nullcontext()


def move_batch(batch, device):
    for key, value in list(batch.items()):
        if key == "locate_feat" or str(key).startswith("locate_feat_"):
            continue
        if torch.is_tensor(value):
            batch[key] = value.to(device=device, non_blocking=True)
    batch["locate_feat"] = [
        feature.to(device=device, dtype=torch.float16, non_blocking=True)
        for feature in batch["locate_feat"]
    ]
    return batch


def load_checkpoint_strict(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("state_dict") or checkpoint
    model_state = model.state_dict()
    clean = OrderedDict()
    for key, value in state.items():
        normalized = str(key)
        while normalized.startswith("module."):
            normalized = normalized[len("module."):]
        if normalized in model_state and tuple(value.shape) == tuple(
                model_state[normalized].shape):
            clean[normalized] = value
    minimum = max(1, len(model_state) // 2)
    if len(clean) < minimum:
        raise RuntimeError("incompatible checkpoint: matched {}/{}".format(
            len(clean), len(model_state)))
    model.load_state_dict(clean, strict=False)
    print("[checkpoint] matched={}/{} step={}".format(
        len(clean), len(model_state), checkpoint.get("step", -1)), flush=True)
    return int(checkpoint.get("step", -1))


def build_model(device):
    base_network = make_network(cfg)
    slice_wrapper = _wrapper_factory(cfg, base_network)
    detection_cache, detection_policy = build_detection_provider(cfg)
    adapter = V46cContourAdapter(
        slice_wrapper,
        detection_cache=detection_cache,
        detection_policy=detection_policy,
    )
    model = MemFlowDiTSnake(
        contour_adapter=adapter,
        feature_dim=int(cfg.locate_feat_dim),
        memory_dim=int(cfg.volmem.memory_dim),
        memory_capacity=int(cfg.volmem.memory_capacity),
        memory_heads=int(cfg.volmem.memory_heads),
        mask_channels=int(getattr(cfg.volmem, "mask_channels", 1)),
        memory_pool_size=int(cfg.volmem.memory_pool_size),
        dit_state_dim=int(cfg.dit_state_dim),
        distance_scale=float(cfg.volmem.relative_distance_scale),
    ).to(device)
    step = load_checkpoint_strict(model, ARGS.ckpt)
    return model.eval(), step


def group_volume_indices(records):
    grouped = defaultdict(list)
    for dataset_index, record in enumerate(records):
        grouped[str(record["case_id"])].append(
            (int(record["slice_idx"]), dataset_index))
    ordered = []
    for volume_id in sorted(grouped):
        items = sorted(grouped[volume_id])
        ordered.append((volume_id, items))
    if ARGS.only_volumes:
        wanted = [v.strip() for v in ARGS.only_volumes.split(",") if v.strip()]
        ordered = [(vid, items) for vid, items in ordered if vid in wanted]
        missing = sorted(set(wanted) - {vid for vid, _ in ordered})
        if missing:
            raise RuntimeError("requested volumes not in split: {}".format(missing))
    if ARGS.volume_start > 0:
        ordered = ordered[ARGS.volume_start:]
    if ARGS.max_volumes is not None and ARGS.max_volumes > 0:
        ordered = ordered[:ARGS.max_volumes]
    return ordered
def build_gt_boundary_index(gt_label_mask):
    """One pass over the slice: {label -> (points[N,2], cKDTree)}.

    Called once per slice instead of once per predicted contour; cropping to
    each label's bounding box keeps findContours off the full-resolution mask.
    """
    index = {}
    labels = [int(v) for v in np.unique(gt_label_mask) if v > 0]
    for label in labels:
        binary = (gt_label_mask == label)
        rows = np.flatnonzero(binary.any(axis=1))
        cols = np.flatnonzero(binary.any(axis=0))
        if rows.size == 0 or cols.size == 0:
            continue
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        x0, x1 = int(cols[0]), int(cols[-1]) + 1
        crop = np.ascontiguousarray(binary[y0:y1, x0:x1].astype(np.uint8))
        contours, _ = cv2.findContours(
            crop, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        points = np.concatenate(
            [c.reshape(-1, 2) for c in contours], axis=0).astype(np.float32)
        points += np.asarray([x0, y0], dtype=np.float32)  # back to image space
        index[label] = (points, cKDTree(points))
    return index


def nearest_gt_targets(pred_xy, gt_entry):
    """For each predicted point, the nearest GT boundary point and its distance.

    This is the 'GT residual' used for oracle action-family analysis. Nearest
    neighbour is a lower bound on the true correspondence error, which is the
    right notion here: it measures how far a point *could* usefully move.
    """
    if gt_entry is None or pred_xy.shape[0] == 0:
        nan = np.full(pred_xy.shape, np.nan, dtype=np.float32)
        return nan, np.full((pred_xy.shape[0],), np.nan, dtype=np.float32)
    points, tree = gt_entry
    dist, idx = tree.query(pred_xy, k=1)
    return points[idx].astype(np.float32), np.asarray(dist, dtype=np.float32)


def contour_normals(poly):
    """Outward-ish unit normals from the closed-polygon tangent."""
    prev_pt = np.roll(poly, 1, axis=0)
    next_pt = np.roll(poly, -1, axis=0)
    tangent = next_pt - prev_pt
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-6)
    tangent = tangent / norm
    normals = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    # Orient outward relative to the centroid so sign is comparable across slices.
    centroid = poly.mean(axis=0, keepdims=True)
    outward = poly - centroid
    flip = (normals * outward).sum(axis=1) < 0
    normals[flip] *= -1.0
    return normals.astype(np.float32)


def token_evidence(sample, pred_label_mask, device, mask_channels):
    """Mask evidence for the memory write, shaped exactly [1,C,H,W].

    SliceMemoryEncoder.forward hard-asserts dim()==4 and size(0)==1, so the
    batch axis is added here explicitly rather than relying on whatever
    align_mask_to_token_grid happens to return.
    """
    if ARGS.memory_mode == "oracle":
        grid = sample["volmem_mask_grid"]
    else:
        grid = align_mask_to_token_grid(
            pred_label_mask, sample, mask_channels=mask_channels)
    tensor = torch.as_tensor(grid, device=device, dtype=torch.float32)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 4 or tensor.size(0) != 1:
        raise RuntimeError(
            "mask evidence must be [1,C,H,W], got {}".format(tuple(tensor.shape)))
    if tensor.size(1) != mask_channels:
        raise RuntimeError("mask evidence channels {} != cfg {}".format(
            tensor.size(1), mask_channels))
    return tensor


def main():
    np.random.seed(ARGS.seed)
    torch.manual_seed(ARGS.seed)
    torch.cuda.manual_seed_all(ARGS.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Must precede configure_single_slice_compatibility: the dataset shim reads
    # the cache root off cfg when it is constructed.
    if ARGS.cache_root:
        if not os.path.isdir(ARGS.cache_root):
            raise RuntimeError("cache root does not exist: " + ARGS.cache_root)
        print("[cache] locate_feat_cache_root {} -> {}".format(
            cfg.locate_feat_cache_root, ARGS.cache_root), flush=True)
        cfg.locate_feat_cache_root = ARGS.cache_root

    configure_single_slice_compatibility(cfg)
    configure_box_mode(cfg, ARGS.box_mode)
    cfg.test.dataset = "VolMemVal" if ARGS.split == "val" else "VolMemTest"
    cfg.test.batch_size = 1
    cfg.train.num_workers = 0
    os.makedirs(ARGS.out_dir, exist_ok=True)
    cfg.result_dir = os.path.join(ARGS.out_dir, "_evaluator")

    device = torch.device(ARGS.device)
    dataset = make_single_slice_dataset_class()(
        ann_file=str(cfg.volmem.manifest_file),
        data_root=str(cfg.volmem.data_root),
        split=ARGS.split,
    )
    volumes = group_volume_indices(dataset.records)
    model, checkpoint_step = build_model(device)
    evaluator = Evaluator(cfg.result_dir)
    mask_channels = int(getattr(cfg.volmem, "mask_channels", 1))
    use_amp = bool(cfg.use_amp)

    index = {
        "checkpoint": ARGS.ckpt,
        "checkpoint_step": checkpoint_step,
        "split": ARGS.split,
        "box_mode": ARGS.box_mode,
        "memory_mode": ARGS.memory_mode,
        "seed": ARGS.seed,
        "poly_num": int(cfg.poly_num),
        "down_ratio": float(snake_config.down_ratio),
        "volumes": [],
    }
    processed = 0
    timing = defaultdict(float)
    with torch.no_grad():
        for volume_number, (volume_id, items) in enumerate(volumes, start=1):
            bank = model.new_banks([volume_id])
            rows = []
            for slice_index, dataset_index in items:
                t_mark = time.time()
                sample = dataset[dataset_index]
                batch = move_batch(snake_collator([sample]), device)
                timing["load"] += time.time() - t_mark
                t_mark = time.time()
                meta = SliceSequenceMeta(
                    volume_id=volume_id,
                    slice_index=slice_index,
                    slice_position=float(slice_index),
                    position_unit="index",
                    sequence_direction="ascending",
                )
                if ARGS.memory_mode == "off":
                    model.set_slice_memory([], [])
                with amp_context(use_amp):
                    output, raw_features, read_delta = model.predict_step(
                        batch, [meta], bank)

                timing["infer"] += time.time() - t_mark
                t_mark = time.time()

                record = dataset.records[dataset_index]
                gt_label_mask = evaluator._read_mask(record["mask_path"])
                gt_index = build_gt_boundary_index(gt_label_mask)
                timing["gt_index"] += time.time() - t_mark
                t_mark = time.time()

                predictions = evaluator._prepare_predictions(output, 1)[0]
                img_path = batch["img_path"]
                img_path = (img_path[0] if isinstance(img_path, (list, tuple))
                            else img_path)
                eval_record = evaluator._record_for_path(img_path)
                _, inv_trans, orig_hw, flipped = evaluator._sample_metadata(
                    batch, 0, 1, eval_record, gt_label_mask.shape)

                pred_label_mask = np.zeros(gt_label_mask.shape, dtype=np.uint16)
                per_contour = []
                for contour, label, score in predictions:
                    restored = inverse_affine_points(
                        contour * float(snake_config.down_ratio),
                        inv_trans, orig_hw, flipped=flipped)
                    poly = np.asarray(restored, dtype=np.float32)
                    if poly.shape[0] >= 3:
                        cv2.fillPoly(pred_label_mask,
                                     [np.rint(poly).astype(np.int32)],
                                     int(label) + 1)
                    gt_entry = gt_index.get(int(label) + 1)
                    target, dist = nearest_gt_targets(poly, gt_entry)
                    per_contour.append({
                        "label": int(label) + 1,
                        "score": float(score),
                        "poly": poly,
                        "gt_target": target,
                        "gt_dist": dist,
                        "normal": contour_normals(poly),
                        "n_gt_boundary": (
                            0 if gt_entry is None else int(gt_entry[0].shape[0])),
                    })
                rows.append({
                    "slice_idx": int(slice_index),
                    "read_delta": float(read_delta),
                    "gt_labels": sorted(gt_index.keys()),
                    "gt_fg": int((gt_label_mask > 0).sum()),
                    "pred_fg": int((pred_label_mask > 0).sum()),
                    "contours": per_contour,
                })
                timing["match"] += time.time() - t_mark
                t_mark = time.time()

                if ARGS.memory_mode != "off":
                    evidence = token_evidence(
                        sample, pred_label_mask, device, mask_channels)
                    with amp_context(use_amp):
                        model.write_step(raw_features, [evidence], [meta], bank)
                    model.detach_banks(bank, keep_recent=0)
                timing["write"] += time.time() - t_mark
                processed += 1
                if processed % ARGS.log_every == 0:
                    print("[probe] vol={}/{} slices={} contours={} t/slice={:.2f}s "
                          "[{}]".format(
                              volume_number, len(volumes), processed,
                              len(per_contour),
                              sum(timing.values()) / processed,
                              " ".join("{}={:.2f}".format(k, v / processed)
                                       for k, v in sorted(timing.items()))),
                          flush=True)

            save_volume(ARGS.out_dir, volume_id, rows, index)

    with open(os.path.join(ARGS.out_dir, "index.json"), "w") as handle:
        json.dump(index, handle, indent=1)
    print("[probe] done volumes={} slices={}".format(
        len(index["volumes"]), processed), flush=True)


def save_volume(out_dir, volume_id, rows, index):
    """Flatten per-contour arrays into one npz per volume."""
    flat = {k: [] for k in ("poly", "gt_target", "gt_dist", "normal")}
    meta_slice, meta_label, meta_score, meta_ngt = [], [], [], []
    slice_rows = []
    for row in rows:
        slice_rows.append({
            "slice_idx": row["slice_idx"],
            "read_delta": row["read_delta"],
            "gt_labels": row["gt_labels"],
            "gt_fg": row["gt_fg"],
            "pred_fg": row["pred_fg"],
            "n_contours": len(row["contours"]),
        })
        for contour in row["contours"]:
            for key in flat:
                flat[key].append(contour[key])
            meta_slice.append(row["slice_idx"])
            meta_label.append(contour["label"])
            meta_score.append(contour["score"])
            meta_ngt.append(contour["n_gt_boundary"])
    payload = {}
    for key, values in flat.items():
        payload[key] = (np.stack(values, axis=0) if values
                        else np.zeros((0, int(cfg.poly_num), 2), dtype=np.float32))
    payload["slice_idx"] = np.asarray(meta_slice, dtype=np.int32)
    payload["label"] = np.asarray(meta_label, dtype=np.int32)
    payload["score"] = np.asarray(meta_score, dtype=np.float32)
    payload["n_gt_boundary"] = np.asarray(meta_ngt, dtype=np.int32)
    path = os.path.join(out_dir, "{}.npz".format(volume_id))
    np.savez_compressed(path, **payload)
    index["volumes"].append({
        "volume_id": volume_id,
        "npz": os.path.basename(path),
        "n_slices": len(rows),
        "n_contours": int(payload["label"].shape[0]),
        "slices": slice_rows,
    })


if __name__ == "__main__":
    main()

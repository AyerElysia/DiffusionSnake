import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from lib.config import cfg
from lib.utils.snake import snake_config, snake_gcn_utils, snake_voc_utils


_SAM_MODEL_CACHE: Dict[Tuple[str, str, int], Any] = {}
_EFFICIENT_SAM_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}
_WARNED: set = set()


def _warn_once(key: str, msg: str) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    print(msg)


def sam_init_enabled() -> bool:
    method = str(getattr(cfg, "contour_init_method", "octagon")).strip().lower()
    return method in ("sam", "efficient_sam")


def _sam_backend() -> str:
    method = str(getattr(cfg, "contour_init_method", "octagon")).strip().lower()
    backend = str(getattr(cfg, "sam_backend", "")).strip().lower()
    if backend:
        return backend
    if method == "efficient_sam":
        return "efficient_sam"
    return "sam"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_weight_path(attr_name: str = "sam_weight") -> str:
    weight = str(getattr(cfg, attr_name, "")).strip()
    if not weight:
        return ""
    p = Path(weight)
    if p.is_absolute():
        return str(p)
    repo_path = _repo_root() / p
    if repo_path.exists():
        return str(repo_path)
    return weight


def _load_sam_model(device: torch.device):
    weight = _resolve_weight_path("sam_weight")
    if not weight:
        _warn_once("sam_no_weight", "[V5.0/SAM] sam_weight is empty; fallback to octagon initialization.")
        return None

    allow_download = bool(getattr(cfg, "sam_allow_download", False))
    if (not allow_download) and (not os.path.exists(weight)):
        _warn_once(
            "sam_missing_weight",
            f"[V5.0/SAM] SAM weight not found: {weight}; fallback to octagon initialization.",
        )
        return None

    imgsz = int(getattr(cfg, "sam_imgsz", 1024))
    device_name = "cpu" if device.type == "cpu" else f"cuda:{device.index or 0}"
    cache_key = (weight, device_name, imgsz)
    if cache_key in _SAM_MODEL_CACHE:
        return _SAM_MODEL_CACHE[cache_key]

    yoloe_root = _repo_root() / "yoloe"
    if yoloe_root.exists() and str(yoloe_root) not in sys.path:
        sys.path.insert(0, str(yoloe_root))

    try:
        from ultralytics import SAM

        model = SAM(weight)
        _SAM_MODEL_CACHE[cache_key] = model
        return model
    except Exception as exc:
        _warn_once("sam_load_failed", f"[V5.0/SAM] failed to load SAM ({exc}); fallback to octagon initialization.")
        _SAM_MODEL_CACHE[cache_key] = None
        return None


def _resolve_efficient_sam_weight_path() -> str:
    weight = str(getattr(cfg, "efficient_sam_weight", "")).strip()
    if weight:
        return _resolve_weight_path("efficient_sam_weight")
    return _resolve_weight_path("sam_weight")


def _load_efficient_sam_model(device: torch.device):
    weight = _resolve_efficient_sam_weight_path()
    if not weight:
        _warn_once(
            "efficient_sam_no_weight",
            "[V5.2/EfficientSAM] efficient_sam_weight is empty; fallback to octagon initialization.",
        )
        return None
    if (not os.path.exists(weight)) or os.path.getsize(weight) < 1024 * 1024:
        _warn_once(
            "efficient_sam_missing_weight",
            f"[V5.2/EfficientSAM] weight not found: {weight}; fallback to octagon initialization.",
        )
        return None

    device_name = "cpu" if device.type == "cpu" else f"cuda:{device.index or 0}"
    cache_key = (weight, device_name)
    if cache_key in _EFFICIENT_SAM_MODEL_CACHE:
        return _EFFICIENT_SAM_MODEL_CACHE[cache_key]

    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from SAMSnake.network.EfficientSAM.efficient_sam.efficient_sam import build_efficient_sam

        model = build_efficient_sam(
            encoder_patch_embed_dim=int(getattr(cfg, "efficient_sam_encoder_dim", 192)),
            encoder_num_heads=int(getattr(cfg, "efficient_sam_encoder_heads", 3)),
            checkpoint=weight,
        )
        model = model.to(device=device, dtype=torch.float32).eval()
        for param in model.parameters():
            param.requires_grad = False
        _EFFICIENT_SAM_MODEL_CACHE[cache_key] = model
        return model
    except Exception as exc:
        _warn_once(
            "efficient_sam_load_failed",
            f"[V5.2/EfficientSAM] failed to load EfficientSAM ({exc}); fallback to octagon initialization.",
        )
        _EFFICIENT_SAM_MODEL_CACHE[cache_key] = None
        return None


def _as_numpy_image(img: Any) -> np.ndarray:
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _predict_sam_masks(image: np.ndarray, boxes: np.ndarray, device: torch.device) -> Optional[np.ndarray]:
    if _sam_backend() == "efficient_sam":
        return _predict_efficient_sam_masks(image, boxes, device)

    if boxes.size == 0:
        return np.zeros((0, image.shape[0], image.shape[1]), dtype=np.uint8)

    model = _load_sam_model(device)
    if model is None:
        return None

    dev_arg: Any = "cpu" if device.type == "cpu" else int(device.index or 0)
    try:
        results = model.predict(
            image,
            bboxes=boxes.astype(float).tolist(),
            imgsz=int(getattr(cfg, "sam_imgsz", 1024)),
            device=dev_arg,
            verbose=False,
            save=False,
            retina_masks=True,
        )
        if not results or getattr(results[0], "masks", None) is None:
            return None
        masks = results[0].masks.data
        if torch.is_tensor(masks):
            masks = masks.detach().cpu().numpy()
        return (masks > 0.5).astype(np.uint8)
    except Exception as exc:
        _warn_once("sam_predict_failed", f"[V5.0/SAM] SAM prediction failed ({exc}); fallback to octagon initialization.")
        return None


def _predict_efficient_sam_masks(image: np.ndarray, boxes: np.ndarray, device: torch.device) -> Optional[np.ndarray]:
    if boxes.size == 0:
        return np.zeros((0, image.shape[0], image.shape[1]), dtype=np.uint8)

    model = _load_efficient_sam_model(device)
    if model is None:
        return None

    image = _as_numpy_image(image)
    if bool(getattr(cfg, "efficient_sam_bgr_to_rgb", True)):
        image = image[..., ::-1].copy()
    image_t = torch.from_numpy(image).to(device=device, dtype=torch.float32)
    image_t = image_t.permute(2, 0, 1).unsqueeze(0) / 255.0

    boxes_t = torch.as_tensor(boxes, device=device, dtype=torch.float32).reshape(1, -1, 2, 2)
    labels = torch.tensor([[[2, 3]]], dtype=torch.float32, device=device).repeat(1, boxes_t.size(1), 1)

    predicted_logits = None
    predicted_iou = None
    pre_masks = None
    try:
        model_dtype = next(model.parameters()).dtype
        with torch.no_grad():
            predicted_logits, predicted_iou = model(
                image_t.to(dtype=model_dtype),
                boxes_t.to(dtype=model_dtype),
                labels.to(dtype=model_dtype),
            )
            pre_masks = predicted_logits[0] >= float(getattr(cfg, "efficient_sam_mask_threshold", 0.0))
            mode = str(getattr(cfg, "efficient_sam_multimask_select", "area")).strip().lower()
            if mode == "iou":
                get_index = predicted_iou[0].argmax(dim=1)
            else:
                get_index = pre_masks.sum(dim=(2, 3)).argmax(dim=1)
            pre_masks = pre_masks[torch.arange(get_index.size(0), device=device), get_index]
        return pre_masks.detach().cpu().numpy().astype(np.uint8)
    except Exception as exc:
        _warn_once(
            "efficient_sam_predict_failed",
            f"[V5.2/EfficientSAM] prediction failed ({exc}); fallback to octagon initialization.",
        )
        return None
    finally:
        # EfficientSAM is frozen but its temporary CUDA buffers are large enough to starve
        # the downstream FM backward pass if they stay in the caching allocator.
        del image_t, boxes_t, labels, predicted_logits, predicted_iou, pre_masks
        if device.type == "cuda" and bool(getattr(cfg, "efficient_sam_empty_cache_after_predict", True)):
            torch.cuda.empty_cache()


def _fallback_octagon_from_box(box_xyxy: Sequence[float], num_points: int, out_h: int, out_w: int) -> np.ndarray:
    dr = float(snake_config.down_ratio)
    box = np.asarray(box_xyxy, dtype=np.float32).copy() / dr
    box[[0, 2]] = np.clip(box[[0, 2]], 0, max(out_w - 1, 0))
    box[[1, 3]] = np.clip(box[[1, 3]], 0, max(out_h - 1, 0))
    if box[2] <= box[0]:
        box[2] = min(box[0] + 1.0, max(out_w - 1, 1))
    if box[3] <= box[1]:
        box[3] = min(box[1] + 1.0, max(out_h - 1, 1))
    base = snake_voc_utils.get_init(box)
    return snake_voc_utils.uniformsample(base.astype(np.float32), num_points).astype(np.float32)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep_label = int(np.argmax(areas)) + 1
    return (labels == keep_label).astype(np.uint8)


def _mask_to_poly(mask: np.ndarray, num_points: int, out_h: int, out_w: int) -> Optional[np.ndarray]:
    mask = _largest_component(mask)
    min_area = int(getattr(cfg, "sam_min_mask_area", 16))
    if int(mask.sum()) < min_area:
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    if contour.shape[0] < 3 or cv2.contourArea(contour) < min_area:
        return None

    contour /= float(snake_config.down_ratio)
    contour[:, 0] = np.clip(contour[:, 0], 0, max(out_w - 1, 0))
    contour[:, 1] = np.clip(contour[:, 1], 0, max(out_h - 1, 0))
    if len(np.unique(np.round(contour), axis=0)) < 3:
        return None

    try:
        return snake_voc_utils.uniformsample(contour, num_points).astype(np.float32)
    except Exception:
        return None


def _polys_by_image_from_boxes(
    orig_imgs: Sequence[Any],
    boxes_by_image: Sequence[np.ndarray],
    device: torch.device,
    out_h: int,
    out_w: int,
    num_points: int,
) -> Tuple[List[np.ndarray], int]:
    polys_by_image: List[np.ndarray] = []
    fallback_count = 0

    for img, boxes in zip(orig_imgs, boxes_by_image):
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        image = _as_numpy_image(img)
        img_h, img_w = image.shape[:2]
        if boxes.size:
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, max(img_w - 1, 0))
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, max(img_h - 1, 0))

        fallback_polys = [
            _fallback_octagon_from_box(box, num_points=num_points, out_h=out_h, out_w=out_w)
            for box in boxes
        ]
        masks = _predict_sam_masks(image, boxes, device=device)

        polys: List[np.ndarray] = []
        for idx, fallback_poly in enumerate(fallback_polys):
            poly = None
            if masks is not None and idx < len(masks):
                poly = _mask_to_poly(masks[idx], num_points=num_points, out_h=out_h, out_w=out_w)
            if poly is None:
                poly = fallback_poly
                fallback_count += 1
            polys.append(poly)

        if polys:
            polys_by_image.append(np.stack(polys, axis=0).astype(np.float32))
        else:
            polys_by_image.append(np.zeros((0, num_points, 2), dtype=np.float32))

    return polys_by_image, fallback_count


def build_sam_testing_init(output: Dict[str, Any], batch: Dict[str, Any], device: torch.device) -> Optional[Dict[str, Any]]:
    if not sam_init_enabled():
        return None
    if batch is None or "orig_img" not in batch or "detection" not in output:
        return None

    detection = output["detection"].detach()
    if detection.numel() == 0:
        return None

    score_thresh = float(getattr(cfg, "sam_det_score_thresh", 1e-4))
    boxes_by_image: List[np.ndarray] = []
    counts: List[int] = []
    for b in range(detection.size(0)):
        keep = detection[b, :, 4] > score_thresh
        boxes = detection[b, keep, :4].detach().cpu().numpy().astype(np.float32)
        boxes_by_image.append(boxes)
        counts.append(int(boxes.shape[0]))

    if sum(counts) == 0:
        return None

    out_h, out_w = output.get("feat_hw", (0, 0))
    if not out_h or not out_w:
        img0 = _as_numpy_image(batch["orig_img"][0])
        out_h, out_w = img0.shape[0] // snake_config.down_ratio, img0.shape[1] // snake_config.down_ratio

    polys_by_image, fallback_count = _polys_by_image_from_boxes(
        batch["orig_img"],
        boxes_by_image,
        device=device,
        out_h=int(out_h),
        out_w=int(out_w),
        num_points=int(snake_config.poly_num),
    )

    polys = [torch.from_numpy(p) for p in polys_by_image if p.shape[0] > 0]
    if polys:
        i_it_py = torch.cat(polys, dim=0).to(device=device, dtype=detection.dtype)
    else:
        i_it_py = torch.zeros((0, snake_config.poly_num, 2), device=device, dtype=detection.dtype)

    py_ind = torch.cat(
        [torch.full((n,), i, dtype=torch.long, device=device) for i, n in enumerate(counts) if n > 0],
        dim=0,
    ) if sum(counts) > 0 else torch.zeros((0,), dtype=torch.long, device=device)
    c_it_py = snake_gcn_utils.img_poly_to_can_poly(i_it_py)
    centers = []
    dr = float(snake_config.down_ratio)
    for boxes in boxes_by_image:
        if boxes.shape[0] == 0:
            continue
        centers.append(torch.from_numpy((boxes[:, :2] + boxes[:, 2:4]) * (0.5 / dr)))
    sam_ct = torch.cat(centers, dim=0).to(device=device, dtype=detection.dtype) if centers else torch.zeros(
        (0, 2), device=device, dtype=detection.dtype
    )

    return {
        "sam_i_it_py": i_it_py,
        "sam_c_it_py": c_it_py,
        "sam_py_ind": py_ind,
        "sam_ct": sam_ct,
        "sam_init_fallback_count": fallback_count,
    }


def attach_sam_testing_init(output: Dict[str, Any], batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    init = build_sam_testing_init(output, batch, device)
    if init is not None:
        output.update(init)
    return output


def build_sam_polys_from_boxes(
    orig_imgs: Sequence[Any],
    boxes_by_image: Sequence[np.ndarray],
    device: torch.device,
    out_h: int,
    out_w: int,
    num_points: int,
) -> Tuple[List[np.ndarray], int]:
    """Public helper for diagnostic GT-box SAM initialization tests."""
    return _polys_by_image_from_boxes(
        orig_imgs,
        boxes_by_image,
        device=device,
        out_h=out_h,
        out_w=out_w,
        num_points=num_points,
    )


def maybe_use_output_sam_training_init(
    train_dict: Dict[str, torch.Tensor],
    output: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    i_it_py = output.get("sam_i_it_py", None)
    out_py_ind = output.get("sam_py_ind", None)
    if not torch.is_tensor(i_it_py) or not torch.is_tensor(out_py_ind):
        return train_dict
    if i_it_py.numel() == 0 or train_dict["i_it_py"].numel() == 0:
        return train_dict

    target_py_ind = train_dict["py_ind"].to(device=out_py_ind.device, dtype=torch.long)
    out_py_ind = out_py_ind.to(device=i_it_py.device, dtype=torch.long)
    selected = []
    cursors: Dict[int, int] = {}
    for img_idx_t in target_py_ind.detach().cpu().tolist():
        img_idx = int(img_idx_t)
        candidates = (out_py_ind == img_idx).nonzero(as_tuple=False).view(-1)
        cursor = cursors.get(img_idx, 0)
        if cursor >= int(candidates.numel()):
            return train_dict
        selected.append(candidates[cursor])
        cursors[img_idx] = cursor + 1

    if not selected:
        return train_dict
    sel = torch.stack(selected, dim=0).to(device=i_it_py.device)
    if sel.numel() != train_dict["i_it_py"].size(0):
        return train_dict

    new_i_it_py = i_it_py[sel].to(device=train_dict["i_it_py"].device, dtype=train_dict["i_it_py"].dtype)
    train_dict["i_it_py"] = new_i_it_py
    train_dict["c_it_py"] = snake_gcn_utils.img_poly_to_can_poly(new_i_it_py)
    train_dict["i_it_4py"] = new_i_it_py[:, : snake_config.init_poly_num]
    train_dict["c_it_4py"] = snake_gcn_utils.img_poly_to_can_poly(train_dict["i_it_4py"])
    return train_dict


def _box_iou_np(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    area_a = max((box[2] - box[0]) * (box[3] - box[1]), 0.0)
    area_b = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 0.0)
    return inter / np.maximum(area_a + area_b - inter, 1e-6)


def _select_training_prompt_boxes(
    train_dict: Dict[str, torch.Tensor],
    output: Dict[str, Any],
    source: str,
) -> List[np.ndarray]:
    py_ind = train_dict["py_ind"].detach().cpu().numpy().astype(np.int64)
    gt_polys = train_dict["i_gt_py"].detach().cpu().numpy().astype(np.float32)
    gt_boxes = np.concatenate([gt_polys.min(axis=1), gt_polys.max(axis=1)], axis=1)
    gt_boxes *= float(snake_config.down_ratio)

    if source != "yolo_box" or "detection" not in output:
        return [gt_boxes[py_ind == b] for b in range(int(py_ind.max()) + 1 if py_ind.size else 0)]

    det = output["detection"].detach().cpu().numpy().astype(np.float32)
    score_thresh = float(getattr(cfg, "sam_train_det_score_thresh", 1e-4))
    iou_min = float(getattr(cfg, "sam_train_match_iou_min", 0.10))
    boxes_by_image: List[List[np.ndarray]] = [[] for _ in range(det.shape[0])]

    for idx, img_idx in enumerate(py_ind.tolist()):
        candidates = det[img_idx]
        candidates = candidates[candidates[:, 4] > score_thresh]
        prompt_box = gt_boxes[idx]
        if candidates.size > 0:
            ious = _box_iou_np(prompt_box, candidates[:, :4])
            best = int(np.argmax(ious))
            if float(ious[best]) >= iou_min:
                prompt_box = candidates[best, :4]
        boxes_by_image[img_idx].append(prompt_box.astype(np.float32))

    return [
        np.stack(items, axis=0).astype(np.float32) if items else np.zeros((0, 4), dtype=np.float32)
        for items in boxes_by_image
    ]


def maybe_replace_training_init(
    train_dict: Dict[str, torch.Tensor],
    output: Dict[str, Any],
    batch: Dict[str, Any],
    device: torch.device,
    out_h: int,
    out_w: int,
) -> Dict[str, torch.Tensor]:
    if not sam_init_enabled() or not bool(getattr(cfg, "sam_use_in_train", True)):
        return train_dict
    if batch is None or "orig_img" not in batch or train_dict["i_it_py"].numel() == 0:
        return train_dict

    source = str(getattr(cfg, "sam_train_prompt_source", "gt_box")).strip().lower()
    boxes_by_image = _select_training_prompt_boxes(train_dict, output, source=source)
    if not boxes_by_image:
        return train_dict

    polys_by_image, _ = _polys_by_image_from_boxes(
        batch["orig_img"],
        boxes_by_image,
        device=device,
        out_h=out_h,
        out_w=out_w,
        num_points=int(train_dict["i_it_py"].size(1)),
    )

    py_ind = train_dict["py_ind"].detach().cpu().numpy().astype(np.int64)
    per_image_cursor = [0 for _ in polys_by_image]
    new_polys = []
    for img_idx in py_ind.tolist():
        cursor = per_image_cursor[img_idx]
        if img_idx >= len(polys_by_image) or cursor >= len(polys_by_image[img_idx]):
            new_polys.append(train_dict["i_it_py"][len(new_polys)].detach().cpu().numpy())
        else:
            new_polys.append(polys_by_image[img_idx][cursor])
        per_image_cursor[img_idx] += 1

    i_it_py = torch.from_numpy(np.stack(new_polys, axis=0)).to(
        device=device,
        dtype=train_dict["i_it_py"].dtype,
    )
    train_dict["i_it_py"] = i_it_py
    train_dict["c_it_py"] = snake_gcn_utils.img_poly_to_can_poly(i_it_py)
    return train_dict

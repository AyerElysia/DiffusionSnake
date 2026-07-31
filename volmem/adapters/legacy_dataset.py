import cv2
import numpy as np

from lib.config import cfg


def configure_single_slice_compatibility(cfg) -> None:
    """Confine inherited dataset field names to the adapter boundary."""
    cfg.pseudo3d_input_mode = "center_repeat"
    cfg.pseudo3d_mean = 0.0
    cfg.pseudo3d_std = 1.0
    cfg.pseudo3d_color_aug = False
    cfg.pseudo3d_lr_flip = False
    cfg.pseudo3d_random_crop = False
    cfg.prev_contour_init_prob = 0.0
    feature_keys = list(getattr(cfg, "locate_feat_keys", ["layer_18"]))
    cfg.sagittal_moonvit_feature_key = ",".join(str(key) for key in feature_keys)
    cfg.sagittal_moonvit_fusion_mode = "center_only"


def align_mask_to_token_grid(mask, locate_metadata, mask_channels=1):
    """Map a label mask to the cached MoonViT grid.

    ``mask_channels=1`` preserves the legacy binary foreground evidence.
    Larger values produce class-aware soft occupancy channels, with channel 0
    reserved for background and vertebra labels kept at their dataset ids.
    """
    mask = np.asarray(mask)
    resized_hw = np.asarray(locate_metadata["locate_feat_resized_hw"]).reshape(-1)
    padded_hw = np.asarray(locate_metadata["locate_feat_padded_hw"]).reshape(-1)
    pad = np.asarray(locate_metadata["locate_feat_pad"]).reshape(-1)
    grid_hw = np.asarray(locate_metadata["locate_feat_grid_hw"]).reshape(-1)
    if not (
        resized_hw.size == 2
        and padded_hw.size == 2
        and pad.size == 4
        and grid_hw.size == 2
    ):
        raise ValueError("invalid MoonViT cache geometry metadata")

    resized_h, resized_w = (int(resized_hw[0]), int(resized_hw[1]))
    padded_h, padded_w = (int(padded_hw[0]), int(padded_hw[1]))
    pad_left, pad_top, pad_right, pad_bottom = [int(value) for value in pad]
    grid_h, grid_w = (int(grid_hw[0]), int(grid_hw[1]))
    if resized_h <= 0 or resized_w <= 0 or grid_h <= 0 or grid_w <= 0:
        raise ValueError("MoonViT geometry dimensions must be positive")
    if pad_top + resized_h + pad_bottom != padded_h:
        raise ValueError("vertical MoonViT padding metadata is inconsistent")
    if pad_left + resized_w + pad_right != padded_w:
        raise ValueError("horizontal MoonViT padding metadata is inconsistent")

    mask_channels = int(mask_channels)
    if mask_channels <= 0:
        raise ValueError("mask_channels must be positive")
    if mask_channels == 1:
        labels = [(0, np.asarray(mask > 0, dtype=np.uint8))]
    else:
        present = [
            int(label)
            for label in np.unique(mask)
            if 0 < int(label) < mask_channels
        ]
        labels = [
            (label, np.asarray(mask == label, dtype=np.uint8))
            for label in present
        ]

    token_mask = np.zeros(
        (mask_channels, grid_h, grid_w),
        dtype=np.float32,
    )
    for channel, binary in labels:
        resized = cv2.resize(
            binary,
            (resized_w, resized_h),
            interpolation=cv2.INTER_NEAREST,
        )
        padded = np.zeros((padded_h, padded_w), dtype=np.float32)
        padded[
            pad_top:pad_top + resized_h,
            pad_left:pad_left + resized_w,
        ] = resized.astype(np.float32, copy=False)
        token_mask[channel] = cv2.resize(
            padded,
            (grid_w, grid_h),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)
    return np.ascontiguousarray(token_mask)


def make_single_slice_dataset_class():
    """Build a compatibility class that never reads adjacent-slice images."""
    from lib.datasets.sagittal_2d_fixed.snake import Dataset as LegacyDataset

    class SingleSliceCompatibilityDataset(LegacyDataset):
        def _neighbor_rows(self, center_row):
            return [center_row, center_row, center_row]

        def __getitem__(self, index):
            sample = super().__getitem__(index)
            row = self.records[index]
            original_image = self._read_grayscale_image(row["image_path"])
            original_mask = self._read_mask(row["mask_path"], original_image.shape)
            sample["volmem_mask_grid"] = align_mask_to_token_grid(
                original_mask,
                sample,
                mask_channels=int(getattr(cfg.volmem, "mask_channels", 1)),
            )
            return sample

    return SingleSliceCompatibilityDataset

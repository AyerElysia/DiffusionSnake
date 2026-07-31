import torch.nn as nn
from .evolve import Evolution
from .snake import Snake
from lib.utils import net_utils, data_utils
from lib.utils.snake import snake_decode, snake_gcn_utils, snake_config
import torch
import torch.nn.functional as F
from lib.config import cfg
import warnings
from lib.networks.YOLOV8.nn.tasks import DetectionModel, attempt_load_one_weight, yaml_model_load
import glob
import os
import sys
import types
import importlib.machinery
from torchvision import models as tv_models
from torchvision.ops import nms as tv_nms

warnings.filterwarnings("ignore")


# 网络拼接主程序（之前的都是准备）--------------------------------------------------------------------------------------------


def _make_torchvision_resnet(backbone_name, pretrained):
    backbone_name = str(backbone_name).strip().lower()
    if backbone_name not in ("resnet18", "resnet34"):
        raise ValueError(f"Unsupported heatmap backbone: {backbone_name}")

    constructor = getattr(tv_models, backbone_name)
    if not pretrained:
        try:
            return constructor(weights=None)
        except TypeError:
            return constructor(pretrained=False)

    try:
        weight_enum_name = "ResNet18_Weights" if backbone_name == "resnet18" else "ResNet34_Weights"
        weights = getattr(tv_models, weight_enum_name).IMAGENET1K_V1
        return constructor(weights=weights)
    except AttributeError:
        return constructor(pretrained=True)


def _load_moonvit_pretrained_state(pretrained_path, target_state):
    pretrained_path = str(pretrained_path or '').strip()
    if not pretrained_path:
        return {}

    paths = []
    if os.path.isdir(pretrained_path):
        paths = sorted(glob.glob(os.path.join(pretrained_path, '*.safetensors')))
        if not paths:
            paths = sorted(glob.glob(os.path.join(pretrained_path, '*.pt')) + glob.glob(os.path.join(pretrained_path, '*.pth')))
    elif pretrained_path.endswith('.safetensors'):
        paths = [pretrained_path]

    reusable_state = {}
    prefixes = (
        'vision_model.',
        'model.vision_model.',
        'module.vision_model.',
        'backbone.',
        'net.heatmap_detector.backbone.',
    )

    def clean_key_name(key):
        clean = key
        for prefix in prefixes:
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
                break
        return clean

    def maybe_add(key, value):
        clean = clean_key_name(key)
        if clean in target_state and tuple(value.shape) == tuple(target_state[clean].shape):
            reusable_state[clean] = value

    if paths and paths[0].endswith('.safetensors'):
        try:
            from safetensors import safe_open
        except Exception as e:
            raise ImportError(f"safetensors is required to load MoonViT pretrained weights: {e}")
        for path in paths:
            with safe_open(path, framework='pt', device='cpu') as f:
                for key in f.keys():
                    clean = clean_key_name(key)
                    if clean in target_state:
                        value = f.get_tensor(key)
                        if tuple(value.shape) == tuple(target_state[clean].shape):
                            reusable_state[clean] = value
        return reusable_state

    if paths:
        state_dict = {}
        for path in paths:
            ckpt = torch.load(path, map_location='cpu')
            shard = ckpt.get('model', ckpt.get('state_dict', ckpt)) if isinstance(ckpt, dict) else ckpt
            state_dict.update(shard)
    else:
        ckpt = torch.load(pretrained_path, map_location='cpu')
        state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt)) if isinstance(ckpt, dict) else ckpt

    for key, value in state_dict.items():
        maybe_add(key, value)
    return reusable_state


class HeatmapResNetDetector(nn.Module):
    def __init__(
            self,
            num_classes,
            head_conv=256,
            backbone_name="resnet18",
            pretrained=False,
            feat_channels=64,
            mask_classes=0):
        super().__init__()

        backbone = _make_torchvision_resnet(backbone_name, pretrained)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.lat1 = nn.Conv2d(64, feat_channels, kernel_size=1, bias=False)
        self.lat2 = nn.Conv2d(128, feat_channels, kernel_size=1, bias=False)
        self.lat3 = nn.Conv2d(256, feat_channels, kernel_size=1, bias=False)
        self.lat4 = nn.Conv2d(512, feat_channels, kernel_size=1, bias=False)
        self.out = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )

        self.ct_head = nn.Sequential(
            nn.Conv2d(feat_channels, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_classes, kernel_size=1, bias=True),
        )
        self.wh_head = nn.Sequential(
            nn.Conv2d(feat_channels, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, kernel_size=1, bias=True),
        )
        self.mask_head = None
        if int(mask_classes) > 0:
            self.mask_head = nn.Sequential(
                nn.Conv2d(feat_channels, head_conv, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, int(mask_classes), kernel_size=1, bias=True),
            )
            nn.init.constant_(self.mask_head[-1].bias, -4.0)
        nn.init.constant_(self.ct_head[-1].bias, -2.19)

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5 = self.lat4(c5)
        p4 = self.lat3(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.lat2(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lat1(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        feat = self.out(p2)

        ct_hm = net_utils.sigmoid(self.ct_head(feat))
        wh = F.relu(self.wh_head(feat))
        mask_logits = self.mask_head(feat) if self.mask_head is not None else None
        return feat, ct_hm, wh, mask_logits


class LocateFeatAdapter(nn.Module):
    def __init__(self, in_channels=2304, hidden_channels=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(int(in_channels), int(hidden_channels), kernel_size=1, bias=False),
            nn.GroupNorm(8, int(hidden_channels)),
            nn.GELU(),
            nn.Conv2d(int(hidden_channels), int(hidden_channels), kernel_size=3, padding=1, bias=True),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, x):
        return self.proj(x)


class LocateFeatReplacer(nn.Module):
    """Project cached MoonViT features and upsample them by ``upscale``.

    MoonViT grids are coarse (patch 14 over a 448-long side, i.e. roughly 10.6
    original pixels per cell), so the upsampling factor directly controls how
    many distinct feature cells a 128-point contour can address. ``upscale`` is
    exposed via ``cfg.locate_feat_replace_upscale`` and must stay in sync with
    the ``source_scale`` used to build the resampling grid.
    """

    def __init__(self, in_channels=2304, hidden_channels=256, out_channels=64, upscale=2,
                 input_layers=1):
        super().__init__()
        hidden_channels = int(hidden_channels)
        upscale = int(upscale)
        input_layers = int(input_layers)
        if upscale < 1:
            raise ValueError("LocateFeatReplacer upscale must be >= 1.")
        if hidden_channels % 16 != 0:
            raise ValueError("LocateFeatReplacer hidden_channels must be divisible by 16.")
        if upscale > 1 and hidden_channels % (upscale * upscale) != 0:
            raise ValueError(
                "LocateFeatReplacer hidden_channels must be divisible by upscale^2 "
                f"(got hidden_channels={hidden_channels}, upscale={upscale})."
            )
        self.upscale = upscale
        out_channels = int(out_channels)
        in_channels = int(in_channels)
        # Cached MoonViT layers live on very different scales (measured on the
        # sagittal cache: layer_18 std ~0.93, layer_26 std ~2.64), so a bare
        # concat lets the deeper layer dominate the 1x1 projection's gradients.
        # GroupNorm over `input_layers` contiguous groups is exactly per-layer
        # standardisation. It is kept OUTSIDE `self.proj` and has affine=False so
        # every existing `proj.N.*` key keeps its index and checkpoints still load.
        self.input_norm = None
        if input_layers > 1:
            if in_channels % input_layers != 0:
                raise ValueError(
                    "LocateFeatReplacer in_channels must be divisible by input_layers "
                    f"(got in_channels={in_channels}, input_layers={input_layers})."
                )
            self.input_norm = nn.GroupNorm(input_layers, in_channels, affine=False)
        layers = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.GroupNorm(16, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, hidden_channels),
            nn.GELU(),
        ]
        if upscale <= 2:
            # Legacy graph (upscale 1 and 2). Kept byte-identical so existing
            # checkpoints trained with PixelShuffle(2) still load cleanly.
            if upscale > 1:
                layers.append(nn.PixelShuffle(upscale))
                head_in = hidden_channels // (upscale * upscale)
            else:
                head_in = hidden_channels
            layers.append(nn.Conv2d(head_in, out_channels, kernel_size=3, padding=1, bias=True))
        else:
            # For larger factors a bare PixelShuffle would leave only
            # hidden/upscale^2 channels (e.g. 512/16 = 32) feeding the head.
            # Expand to out_channels * upscale^2 first so the post-shuffle map
            # keeps full width.
            layers.extend([
                nn.Conv2d(hidden_channels, out_channels * upscale * upscale, kernel_size=1, bias=False),
                nn.PixelShuffle(upscale),
                nn.GroupNorm(16, out_channels),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            ])
        self.proj = nn.Sequential(*layers)

    def forward(self, x):
        if self.input_norm is not None:
            x = self.input_norm(x)
        return self.proj(x)


class HeatmapConvNeXtDetector(nn.Module):
    def __init__(
            self,
            num_classes,
            head_conv=256,
            model_name="convnextv2_tiny",
            pretrained=False,
            pretrained_path="",
            feat_channels=64,
            mask_classes=0,
            allow_fallback=True,
            fallback_model_name="convnext_tiny"):
        super().__init__()
        try:
            import timm
        except Exception as e:
            raise ImportError(f"timm is required for ConvNeXt heatmap detection: {e}")

        requested = str(model_name).strip()
        fallback = str(fallback_model_name or "convnext_tiny").strip()
        available = set(timm.list_models())
        actual_name = requested
        if requested not in available:
            if bool(allow_fallback) and fallback in available:
                print(
                    f"[ConvNeXt] requested model={requested} is unavailable in this timm; "
                    f"fallback to {fallback}",
                    flush=True,
                )
                actual_name = fallback
            else:
                raise ValueError(f"ConvNeXt model {requested} is unavailable in timm and fallback is disabled.")
        self.model_name = actual_name

        self.backbone = timm.create_model(
            actual_name,
            pretrained=bool(pretrained) and not str(pretrained_path or '').strip(),
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        if str(pretrained_path or '').strip():
            ckpt = torch.load(str(pretrained_path), map_location='cpu')
            state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt)) if isinstance(ckpt, dict) else ckpt
            target_state = self.backbone.state_dict()
            reusable_state = {
                k: v for k, v in state_dict.items()
                if k in target_state and tuple(v.shape) == tuple(target_state[k].shape)
            }
            missing, unexpected = self.backbone.load_state_dict(reusable_state, strict=False)
            print(
                f"[ConvNeXt] Loaded pretrained path={pretrained_path} "
                f"reused={len(reusable_state)}/{len(target_state)} "
                f"missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )

        channels = [int(info['num_chs']) for info in self.backbone.feature_info.get_dicts()]
        if len(channels) < 4:
            raise RuntimeError(f"ConvNeXt features_only expected 4 feature maps, got {len(channels)}.")
        self.lats = nn.ModuleList([
            nn.Conv2d(ch, feat_channels, kernel_size=1, bias=False)
            for ch in channels[:4]
        ])
        self.out = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )

        self.ct_head = nn.Sequential(
            nn.Conv2d(feat_channels, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_classes, kernel_size=1, bias=True),
        )
        self.wh_head = nn.Sequential(
            nn.Conv2d(feat_channels, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, kernel_size=1, bias=True),
        )
        self.mask_head = None
        if int(mask_classes) > 0:
            self.mask_head = nn.Sequential(
                nn.Conv2d(feat_channels, head_conv, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, int(mask_classes), kernel_size=1, bias=True),
            )
            nn.init.constant_(self.mask_head[-1].bias, -4.0)
        nn.init.constant_(self.ct_head[-1].bias, -2.19)

    def forward(self, x):
        feats = self.backbone(x)
        p4 = self.lats[3](feats[3])
        p3 = self.lats[2](feats[2]) + F.interpolate(p4, size=feats[2].shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lats[1](feats[1]) + F.interpolate(p3, size=feats[1].shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.lats[0](feats[0]) + F.interpolate(p2, size=feats[0].shape[-2:], mode="bilinear", align_corners=False)
        feat = self.out(p1)

        ct_hm = net_utils.sigmoid(self.ct_head(feat))
        wh = F.relu(self.wh_head(feat))
        mask_logits = self.mask_head(feat) if self.mask_head is not None else None
        return feat, ct_hm, wh, mask_logits


def _ensure_eagle_transformers_compat():
    try:
        import transformers  # noqa: F401
        return
    except Exception:
        pass

    transformers_mod = types.ModuleType("transformers")
    activations_mod = types.ModuleType("transformers.activations")
    modeling_utils_mod = types.ModuleType("transformers.modeling_utils")
    utils_mod = types.ModuleType("transformers.utils")
    configuration_utils_mod = types.ModuleType("transformers.configuration_utils")
    transformers_mod.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)
    activations_mod.__spec__ = importlib.machinery.ModuleSpec("transformers.activations", loader=None)
    modeling_utils_mod.__spec__ = importlib.machinery.ModuleSpec("transformers.modeling_utils", loader=None)
    utils_mod.__spec__ = importlib.machinery.ModuleSpec("transformers.utils", loader=None)
    configuration_utils_mod.__spec__ = importlib.machinery.ModuleSpec("transformers.configuration_utils", loader=None)

    def _pytorch_gelu_tanh():
        return nn.GELU()

    class _PreTrainedModel(nn.Module):
        config_class = None

        def __init__(self, config=None, *args, **kwargs):
            super().__init__()
            self.config = config

        def post_init(self):
            return None

    class _PretrainedConfig:
        model_type = ""

        def __init__(self, **kwargs):
            self._attn_implementation = kwargs.pop("_attn_implementation", "eager")
            for key, value in kwargs.items():
                setattr(self, key, value)

        def to_dict(self):
            return dict(self.__dict__)

    activations_mod.PytorchGELUTanh = _pytorch_gelu_tanh
    modeling_utils_mod.PreTrainedModel = _PreTrainedModel
    utils_mod.is_flash_attn_2_available = lambda: False
    configuration_utils_mod.PretrainedConfig = _PretrainedConfig

    transformers_mod.activations = activations_mod
    transformers_mod.modeling_utils = modeling_utils_mod
    transformers_mod.utils = utils_mod
    transformers_mod.configuration_utils = configuration_utils_mod

    sys.modules.setdefault("transformers", transformers_mod)
    sys.modules.setdefault("transformers.activations", activations_mod)
    sys.modules.setdefault("transformers.modeling_utils", modeling_utils_mod)
    sys.modules.setdefault("transformers.utils", utils_mod)
    sys.modules.setdefault("transformers.configuration_utils", configuration_utils_mod)


class HeatmapMoonViTDetector(nn.Module):
    def __init__(
            self,
            num_classes,
            head_conv=256,
            eagle_root="Eagle/Embodied",
            pretrained_path="",
            freeze_backbone=False,
            patch_size=14,
            num_layers=6,
            num_heads=6,
            hidden_size=384,
            intermediate_size=1536,
            pos_h=64,
            pos_w=64,
            feat_channels=64,
            mask_classes=0,
            target_stride=4,
            max_input_size=0,
            input_norm='none'):
        super().__init__()

        _ensure_eagle_transformers_compat()
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        eagle_root = str(eagle_root or "Eagle/Embodied")
        if not os.path.isabs(eagle_root):
            eagle_root = os.path.join(repo_root, eagle_root)
        if eagle_root not in sys.path:
            sys.path.insert(0, eagle_root)

        from eaglevl.model.moon_vit.modeling_vit import MoonViTConfig, MoonVitPretrainedModel

        self.patch_size = int(patch_size)
        self.merge_kernel_size = (1, 1)
        self.target_stride = max(int(target_stride), 1)
        self.max_input_size = int(max_input_size or 0)
        self.input_norm = str(input_norm or 'none').strip().lower()
        self.hidden_size = int(hidden_size)
        moon_cfg = MoonViTConfig(
            patch_size=self.patch_size,
            init_pos_emb_height=int(pos_h),
            init_pos_emb_width=int(pos_w),
            num_attention_heads=int(num_heads),
            num_hidden_layers=int(num_layers),
            hidden_size=self.hidden_size,
            intermediate_size=int(intermediate_size),
            merge_kernel_size=self.merge_kernel_size,
            _attn_implementation="eager",
        )
        self.backbone = MoonVitPretrainedModel(moon_cfg)
        self.model_name = f"moonvit_l{int(num_layers)}_h{self.hidden_size}_p{self.patch_size}"

        if str(pretrained_path or '').strip():
            target_state = self.backbone.state_dict()
            reusable_state = _load_moonvit_pretrained_state(pretrained_path, target_state)
            missing, unexpected = self.backbone.load_state_dict(reusable_state, strict=False)
            print(
                f"[MoonViT] Loaded pretrained path={pretrained_path} "
                f"reused={len(reusable_state)}/{len(target_state)} "
                f"missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )

        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.proj = nn.Sequential(
            nn.Conv2d(self.hidden_size, feat_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, feat_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels),
            nn.ReLU(inplace=True),
        )
        self.ct_head = nn.Sequential(
            nn.Conv2d(feat_channels, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, num_classes, kernel_size=1, bias=True),
        )
        self.wh_head = nn.Sequential(
            nn.Conv2d(feat_channels, head_conv, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 2, kernel_size=1, bias=True),
        )
        self.mask_head = None
        if int(mask_classes) > 0:
            self.mask_head = nn.Sequential(
                nn.Conv2d(feat_channels, head_conv, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, int(mask_classes), kernel_size=1, bias=True),
            )
            nn.init.constant_(self.mask_head[-1].bias, -4.0)
        nn.init.constant_(self.ct_head[-1].bias, -2.19)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _prepare_moonvit_input(self, x):
        if self.input_norm in ('snake_to_locate', 'snake_bgr_to_locate_rgb'):
            mean = x.new_tensor([0.40789654, 0.44719302, 0.47026115]).view(1, 3, 1, 1)
            std = x.new_tensor([0.28863828, 0.27408164, 0.27809835]).view(1, 3, 1, 1)
            x = (x * std + mean).clamp(0.0, 1.0)
            if self.input_norm == 'snake_bgr_to_locate_rgb':
                x = x[:, [2, 1, 0], :, :]
            x = (x - 0.5) / 0.5
        return x

    def forward(self, x):
        b, _, h, w = x.shape
        x = self._prepare_moonvit_input(x)
        if self.max_input_size > 0 and max(h, w) > self.max_input_size:
            scale = float(self.max_input_size) / float(max(h, w))
            moon_h = max(int(round(h * scale / self.patch_size)) * self.patch_size, self.patch_size)
            moon_w = max(int(round(w * scale / self.patch_size)) * self.patch_size, self.patch_size)
            x_moon = F.interpolate(x, size=(moon_h, moon_w), mode="bilinear", align_corners=False)
        else:
            x_moon = x
        moon_h, moon_w = x_moon.shape[-2:]
        grid_h = max(moon_h // self.patch_size, 1)
        grid_w = max(moon_w // self.patch_size, 1)
        crop_h = grid_h * self.patch_size
        crop_w = grid_w * self.patch_size
        if crop_h != moon_h or crop_w != moon_w:
            x_in = x_moon[:, :, :crop_h, :crop_w]
        else:
            x_in = x_moon
        grid_hws = torch.tensor([[grid_h, grid_w]], device=x.device, dtype=torch.int32).repeat(b, 1)
        patches = F.unfold(
            x_in,
            kernel_size=(self.patch_size, self.patch_size),
            stride=(self.patch_size, self.patch_size),
        )
        patches = patches.transpose(1, 2).reshape(
            b * grid_h * grid_w, x.size(1), self.patch_size, self.patch_size
        )
        tokens = self.backbone.patch_embed(patches, grid_hws)
        tokens = self.backbone.encoder(tokens, grid_hws)
        feat = tokens.view(b, grid_h, grid_w, self.hidden_size).permute(0, 3, 1, 2).contiguous()
        feat = self.proj(feat)

        target_h = max(h // self.target_stride, 1)
        target_w = max(w // self.target_stride, 1)
        if feat.shape[-2:] != (target_h, target_w):
            feat = F.interpolate(feat, size=(target_h, target_w), mode="bilinear", align_corners=False)

        ct_hm = net_utils.sigmoid(self.ct_head(feat))
        wh = F.relu(self.wh_head(feat))
        mask_logits = self.mask_head(feat) if self.mask_head is not None else None
        return feat, ct_hm, wh, mask_logits


class SwinSnakeFeatureExtractor(nn.Module):
    def __init__(
            self,
            model_name="swin_tiny_patch4_window7_224",
            pretrained=False,
            pretrained_path="",
            img_size=672,
            out_channels=64):
        super().__init__()
        try:
            import timm
        except Exception as e:
            raise ImportError(f"timm is required for Swin snake feature extraction: {e}")

        self.img_size = int(img_size)
        if self.img_size % 224 != 0:
            raise ValueError("swin_img_size must be a multiple of 224 for window-7 Swin-T on this timm version.")
        self.swin = timm.create_model(
            str(model_name),
            pretrained=bool(pretrained) and not str(pretrained_path or '').strip(),
            img_size=self.img_size,
        )
        if str(pretrained_path or '').strip():
            ckpt = torch.load(str(pretrained_path), map_location='cpu')
            state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt)) if isinstance(ckpt, dict) else ckpt
            target_state = self.swin.state_dict()
            reusable_state = {
                k: v for k, v in state_dict.items()
                if k in target_state and tuple(v.shape) == tuple(target_state[k].shape)
            }
            missing, unexpected = self.swin.load_state_dict(reusable_state, strict=False)
            print(
                f"[Snake] Loaded Swin pretrained path={pretrained_path} "
                f"reused={len(reusable_state)}/{len(target_state)} "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )

        self.p2_channels = int(getattr(self.swin.layers[0], 'dim'))
        self.p3_channels = int(getattr(self.swin.layers[1], 'dim'))
        self.proj_p2 = nn.Conv2d(self.p2_channels, out_channels, kernel_size=1, bias=False)
        self.proj_p3 = nn.Conv2d(self.p3_channels, out_channels, kernel_size=1, bias=False)
        nn.init.zeros_(self.proj_p3.weight)

    def forward(self, x):
        h_in, w_in = x.shape[-2:]
        if h_in > self.img_size or w_in > self.img_size:
            raise ValueError(f"Input {(h_in, w_in)} exceeds configured Swin img_size={self.img_size}")
        pad_h = self.img_size - h_in
        pad_w = self.img_size - w_in
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)

        y = self.swin.patch_embed(x)
        y = self.swin.pos_drop(y)
        for block in self.swin.layers[0].blocks:
            y = block(y)

        grid_p2 = self.img_size // 4
        p2 = y.view(y.size(0), grid_p2, grid_p2, self.p2_channels).permute(0, 3, 1, 2).contiguous()
        out_h, out_w = h_in // 4, w_in // 4
        p2 = p2[:, :, :out_h, :out_w]
        feat = self.proj_p2(p2)

        p3_tokens = self.swin.layers[0].downsample(y)
        grid_p3 = self.img_size // 8
        p3 = p3_tokens.view(p3_tokens.size(0), grid_p3, grid_p3, self.p3_channels).permute(0, 3, 1, 2).contiguous()
        p3 = p3[:, :, :max(out_h // 2, 1), :max(out_w // 2, 1)]
        p3 = F.interpolate(p3, size=(out_h, out_w), mode="bilinear", align_corners=False)
        return feat + self.proj_p3(p3)


class Network(nn.Module):
    def __init__(self, num_layers, heads, head_conv=256, down_ratio=4, det_dir=''):
        super(Network, self).__init__()

        self.detector_backend = str(getattr(cfg, 'detector_backend', 'yolo') or 'yolo').strip().lower()
        self.down_ratio = float(down_ratio)
        nc = int(getattr(cfg, 'yolo_num_classes', 0) or heads.get('ct_hm', 1))
        self.detector_num_classes = nc
        self.freeze_snake = bool(getattr(cfg, 'freeze_snake', False))
        self.freeze_heatmap_detector = bool(getattr(cfg, 'freeze_heatmap_detector', False))
        self.freeze_yolo = bool(getattr(cfg, 'freeze_yolo', False))
        self.use_p3_features = False
        self.use_swin_snake_feature = False
        self.swin_snake_feature = None
        self.yolo = None
        self.samsnake_dla = None
        self.samsnake_refine = None
        self.use_extreme_refine = bool(getattr(cfg, 'use_extreme_refine', False))
        self.extreme_fuse = None
        self.extreme_refiner = None
        self.locate_feat_inject = bool(getattr(cfg, 'locate_feat_inject', False))
        self.locate_feat_replace = bool(getattr(cfg, 'locate_feat_replace', False))
        self.locate_feat_adapter = None
        self.locate_feat_replacer = None
        self.locate_feat_replace_upscale = int(getattr(cfg, 'locate_feat_replace_upscale', 2))
        if self.locate_feat_inject and self.locate_feat_replace:
            raise ValueError("locate_feat_inject and locate_feat_replace are mutually exclusive.")

        if self.detector_backend == 'yolo':
            # 使用本地 YOLOv8 检测模型替换 DLA，输出检测与特征
            # 选择包含 P2 的结构以获得 stride=4 的特征图，空间大小与原来 DLA 的 136x136 对齐（当输入是 544x544）
            yolo_yaml = 'lib/networks/YOLOV8/cfg/models/v8/yolov8-p2.yaml'
            yolo_cfg = yaml_model_load(yolo_yaml)
            yolo_scale = str(getattr(cfg, 'yolo_model_scale', '') or '').strip().lower()
            if yolo_scale:
                yolo_cfg['scale'] = yolo_scale
            self.yolo = DetectionModel(cfg=yolo_cfg, ch=3, nc=nc, verbose=False)

            try:
                actual_scale = str(getattr(self.yolo, 'yaml', {}).get('scale', ''))
                print(f"[YOLO] yaml={yolo_yaml} requested_scale={yolo_scale or 'default'} actual_scale={actual_scale or 'default'} nc={nc}")
            except Exception:
                pass

            # 加载 YOLO 预训练权重（测试阶段无需加载，统一依赖整体checkpoint；训练可通过开关启用）
            yolov8_pt = str(getattr(cfg, 'yolo_pretrained', '') or '').strip()
            load_yolo_pretrained = bool(getattr(cfg, 'load_yolo_pretrained', False))
            require_yolo_pretrained = bool(getattr(cfg, 'require_yolo_pretrained', False))
            if load_yolo_pretrained:
                try:
                    if not yolov8_pt:
                        raise FileNotFoundError("cfg.yolo_pretrained is empty")
                    if require_yolo_pretrained and not os.path.isfile(yolov8_pt):
                        raise FileNotFoundError(f"YOLO pretrained weights not found: {yolov8_pt}")
                    weights_model, _ = attempt_load_one_weight(
                        yolov8_pt, device=None, inplace=True, fuse=False
                    )
                    self.yolo.load(weights_model)
                    print(f"[YOLO] Loaded pretrained weights: {yolov8_pt}", flush=True)
                    if bool(getattr(cfg, 'pseudo3d_symmetric_stem_init', False)):
                        stem_weight = self.yolo.model[0].conv.weight
                        with torch.no_grad():
                            stem_weight.copy_(self.make_symmetric_stem_weight(stem_weight))
                        print(
                            "[YOLO] Applied symmetric pseudo-3D initialization to the first stem convolution.",
                            flush=True,
                        )
                except Exception as e:
                    message = f"Failed to load YOLO pretrained weights from {yolov8_pt or '<empty>'}: {e}"
                    if require_yolo_pretrained:
                        raise RuntimeError(message) from e
                    print(f"[WARN] {message}", flush=True)

# 将 P2 级别的特征通道压到 snake_feature_dim，供 Snake 的 GCN 使用
            # YOLO Detect 头拼接后的通道数为 regmax*4 + nc（默认 regmax=16 -> 64）
            _snake_feat_dim = int(getattr(cfg, 'snake_feature_dim', 64))
            in_ch = 64 + nc
            self.cnn_proj = nn.Conv2d(in_ch, _snake_feat_dim, kernel_size=1, bias=False)
            self.use_p3_features = bool(
                getattr(cfg, 'v3_4_use_p3_features', False)
                or getattr(cfg, 'v3_7_use_p3_features', False)
                or getattr(cfg, 'v4_use_p3_features', False)
                or getattr(cfg, 'v4_1_use_p3_features', False)
                or getattr(cfg, 'v4_2_use_p3_features', False)
            )
            if self.use_p3_features:
                self.cnn_proj_p3 = nn.Conv2d(in_ch, _snake_feat_dim, kernel_size=1, bias=False)
                nn.init.zeros_(self.cnn_proj_p3.weight)
                print("[Snake] P3 feature fusion enabled with zero-init residual.")
            self.use_swin_snake_feature = bool(getattr(cfg, 'use_swin_snake_feature', False))
            if self.use_swin_snake_feature:
                self.swin_snake_feature = SwinSnakeFeatureExtractor(
                    model_name=str(getattr(cfg, 'swin_model_name', 'swin_tiny_patch4_window7_224')),
                    pretrained=bool(getattr(cfg, 'swin_pretrained', False)),
                    pretrained_path=str(getattr(cfg, 'swin_pretrained_path', '') or ''),
                    img_size=int(getattr(cfg, 'swin_img_size', 672)),
                    out_channels=64,
                )
                if bool(getattr(cfg, 'swin_freeze', False)):
                    self.swin_snake_feature.eval()
                    for p in self.swin_snake_feature.parameters():
                        p.requires_grad = False
                print(
                    f"[Snake] Swin feature replacement enabled: "
                    f"model={getattr(cfg, 'swin_model_name', 'swin_tiny_patch4_window7_224')} "
                    f"pretrained={getattr(cfg, 'swin_pretrained', False)} "
                    f"pretrained_path={getattr(cfg, 'swin_pretrained_path', '') or 'none'} "
                    f"img_size={getattr(cfg, 'swin_img_size', 672)}"
                )
        elif self.detector_backend.startswith('moonvit'):
            self.heatmap_detector = HeatmapMoonViTDetector(
                num_classes=nc,
                head_conv=int(getattr(cfg, 'moonvit_head_conv', head_conv)),
                eagle_root=str(getattr(cfg, 'moonvit_root', 'Eagle/Embodied')),
                pretrained_path=str(getattr(cfg, 'moonvit_pretrained_path', '') or ''),
                freeze_backbone=bool(getattr(cfg, 'moonvit_freeze', False)),
                patch_size=int(getattr(cfg, 'moonvit_patch_size', 14)),
                num_layers=int(getattr(cfg, 'moonvit_num_layers', 6)),
                num_heads=int(getattr(cfg, 'moonvit_num_heads', 6)),
                hidden_size=int(getattr(cfg, 'moonvit_hidden_size', 384)),
                intermediate_size=int(getattr(cfg, 'moonvit_intermediate_size', 1536)),
                pos_h=int(getattr(cfg, 'moonvit_pos_h', 64)),
                pos_w=int(getattr(cfg, 'moonvit_pos_w', 64)),
                feat_channels=int(getattr(cfg, 'moonvit_out_channels', 64)),
                mask_classes=nc if bool(getattr(cfg, 'use_heatmap_mask_head', False)) else 0,
                target_stride=int(getattr(cfg, 'moonvit_target_stride', 4)),
                max_input_size=int(getattr(cfg, 'moonvit_max_input_size', 0)),
                input_norm=str(getattr(cfg, 'moonvit_input_norm', 'none') or 'none'),
            )
            print(
                f"[MoonViT] backend={self.detector_backend} "
                f"model={self.heatmap_detector.model_name} "
                f"freeze={getattr(cfg, 'moonvit_freeze', False)} "
                f"pretrained_path={getattr(cfg, 'moonvit_pretrained_path', '') or 'none'} "
                f"input_norm={getattr(cfg, 'moonvit_input_norm', 'none')} "
                f"nc={nc} mask_head={bool(getattr(cfg, 'use_heatmap_mask_head', False))}",
                flush=True,
            )
        elif self.detector_backend.startswith('convnext'):
            self.heatmap_detector = HeatmapConvNeXtDetector(
                num_classes=nc,
                head_conv=int(getattr(cfg, 'convnext_head_conv', head_conv)),
                model_name=str(getattr(cfg, 'convnext_model_name', 'convnextv2_tiny')),
                pretrained=bool(getattr(cfg, 'convnext_pretrained', False)),
                pretrained_path=str(getattr(cfg, 'convnext_pretrained_path', '') or ''),
                feat_channels=int(getattr(cfg, 'convnext_out_channels', 64)),
                mask_classes=nc if bool(getattr(cfg, 'use_heatmap_mask_head', False)) else 0,
                allow_fallback=bool(getattr(cfg, 'convnext_allow_fallback', True)),
                fallback_model_name=str(getattr(cfg, 'convnext_fallback_model_name', 'convnext_tiny')),
            )
            print(
                f"[ConvNeXt] backend={self.detector_backend} "
                f"model={getattr(cfg, 'convnext_model_name', 'convnextv2_tiny')} "
                f"actual={self.heatmap_detector.model_name} "
                f"nc={nc} mask_head={bool(getattr(cfg, 'use_heatmap_mask_head', False))}"
            )
        elif self.detector_backend.startswith('heatmap_'):
            self.heatmap_detector = HeatmapResNetDetector(
                num_classes=nc,
                head_conv=head_conv,
                backbone_name=str(getattr(cfg, 'heatmap_backbone', 'resnet18')),
                pretrained=bool(getattr(cfg, 'heatmap_pretrained', False)),
                feat_channels=int(getattr(cfg, 'heatmap_feat_channels', 64)),
                mask_classes=nc if bool(getattr(cfg, 'use_heatmap_mask_head', False)) else 0,
            )
            print(
                f"[Heatmap] backend={self.detector_backend} "
                f"backbone={getattr(cfg, 'heatmap_backbone', 'resnet18')} "
                f"nc={nc} mask_head={bool(getattr(cfg, 'use_heatmap_mask_head', False))}"
            )
        elif self.detector_backend == 'samsnake_fm':
            from SAMSnake.network.backbone.dla import DLASeg
            from .samsnake_refine import SAMSnakeRefine

            dla_layers = num_layers if int(num_layers) > 0 else 34
            self.samsnake_dla = DLASeg(
                f'dla{dla_layers}',
                heads,
                pretrained=bool(getattr(cfg, 'samsnake_dla_pretrained', False)),
                down_ratio=int(down_ratio),
                final_kernel=1,
                last_level=int(getattr(cfg, 'samsnake_dla_last_level', 5)),
                head_conv=head_conv,
                use_dcn=bool(getattr(cfg, 'samsnake_use_dcn', False)),
            )
            self.samsnake_refine = SAMSnakeRefine(
                c_in=64,
                num_points=int(getattr(cfg, 'poly_num', 128)),
                stride=float(getattr(cfg, 'samsnake_refine_stride', 4.0)),
            )
            self.samsnake_freeze_dla = bool(getattr(cfg, 'samsnake_freeze_dla', False))
            if self.samsnake_freeze_dla:
                self.samsnake_dla.eval()
                for p in self.samsnake_dla.parameters():
                    p.requires_grad = False
            print(
                f"[V5.2] backend=samsnake_fm DLA{dla_layers} "
                f"pretrained={getattr(cfg, 'samsnake_dla_pretrained', False)} "
                f"use_dcn={getattr(cfg, 'samsnake_use_dcn', False)} "
                f"freeze_dla={getattr(cfg, 'samsnake_freeze_dla', False)}"
            )
        else:
            raise ValueError(f"Unsupported detector backend: {self.detector_backend}")

        if self.locate_feat_inject:
            self.locate_feat_adapter = LocateFeatAdapter(
                in_channels=int(getattr(cfg, 'locate_feat_dim', 2304)),
                hidden_channels=int(getattr(cfg, 'locate_feat_adapt_hidden_channels', 64)),
            )
            print(
                f"[LocateFeat] injection enabled dim={getattr(cfg, 'locate_feat_dim', 2304)} "
                f"root={getattr(cfg, 'locate_feat_cache_root', 'data/locate_feat_cache')}",
                flush=True,
            )
        if self.locate_feat_replace:
            self.locate_feat_replace_upscale = int(getattr(cfg, 'locate_feat_replace_upscale', 2))
            self.locate_feat_replacer = LocateFeatReplacer(
                in_channels=int(getattr(cfg, 'locate_feat_dim', 2304)),
                hidden_channels=int(getattr(cfg, 'locate_feat_replace_hidden_dim', 256)),
                out_channels=int(getattr(cfg, 'locate_feat_replace_out_channels', 64)),
                upscale=self.locate_feat_replace_upscale,
                input_layers=int(getattr(cfg, 'locate_feat_input_layers', 1)),
            )
            param_count = sum(p.numel() for p in self.locate_feat_replacer.parameters())
            print(
                f"[LocateFeat] replacement enabled dim={getattr(cfg, 'locate_feat_dim', 2304)} "
                f"upscale={self.locate_feat_replace_upscale} "
                f"input_layers={int(getattr(cfg, 'locate_feat_input_layers', 1))} "
                f"root={getattr(cfg, 'locate_feat_cache_dir', '') or getattr(cfg, 'locate_feat_cache_root', 'data/locate_feat_cache')} "
                f"params={param_count / 1e6:.3f}M",
                flush=True,
            )

        # Choose between original evolution and diffusion evolution
        use_diffusion = getattr(cfg, 'use_diffusion_evolution', False)

        if bool(getattr(cfg, 'detector_only_warmup', False)):
            self.gcn = None
            print("[Snake] Detector-only warmup enabled; evolution module is not built.", flush=True)
        elif use_diffusion:
            # 延迟导入，避免与 diffusion.evolution -> snake.snake 的循环依赖
            from lib.networks.diffusion import make_evolution
            use_flow_matching = bool(
                getattr(cfg, 'use_flow_matching', False)
                or getattr(cfg, 'use_dit_v3_6', False)
            )
            self.gcn = make_evolution(
                use_grpo=getattr(cfg, 'use_grpo', False),
                state_dim=128,
                feature_dim=int(getattr(cfg, 'snake_feature_dim', 64)),
                num_points=128,
                num_timesteps=getattr(cfg, 'diffusion_timesteps', 1000),
                use_ddim_inference=getattr(cfg, 'use_ddim_inference', True),
                loss_weight=getattr(cfg, 'diffusion_loss_weight', 1.0),
                loss_type=getattr(cfg, 'diffusion_loss_type', 'adaptive'),
                # DiT 去噪器参数
                use_dit_denoiser=getattr(cfg, 'use_dit_denoiser', False),
                use_flow_matching=use_flow_matching,
                flow_ode_steps=getattr(cfg, 'flow_ode_steps', 10),
                dit_num_layers=getattr(cfg, 'dit_num_layers', 6),
                dit_num_heads=getattr(cfg, 'dit_num_heads', 8),
                dit_state_dim=getattr(cfg, 'dit_state_dim', 256),
            )
            self.diffusion_loss_fn = None
        else:
            self.gcn = Evolution()
            self.diffusion_loss_fn = None

        if self.use_extreme_refine:
            # V4.6c detext 配置启用该分支：bbox 先变 40 点初始轮廓，
            # extreme_refiner 再预测 refined extreme points，供 diffusion inference 重建 octagon。
            self.extreme_fuse = nn.Conv1d(128, 64, kernel_size=1)
            self.extreme_refiner = Snake(state_dim=128, feature_dim=64 + 2, conv_type='dgrid')
            print("[Snake] Extreme-point refinement head enabled for octagon initialization.")

        # 冻结 Snake 相关模块（只训练 YOLO）
        if self.freeze_snake:
            if self.detector_backend.startswith('heatmap_'):
                # Detector-only warmup must not update the cached-feature adapter
                # or diffusion branch. Keep only the heatmap detector trainable.
                for name, parameter in self.named_parameters():
                    parameter.requires_grad = name.startswith('heatmap_detector.')
            else:
                if self.detector_backend == 'yolo':
                    modules_to_freeze = [self.gcn, self.cnn_proj] if not use_diffusion else [self.cnn_proj]
                    if hasattr(self, 'cnn_proj_p3'):
                        modules_to_freeze.append(self.cnn_proj_p3)
                elif self.detector_backend == 'samsnake_fm':
                    modules_to_freeze = [self.gcn]
                    if self.samsnake_refine is not None:
                        modules_to_freeze.append(self.samsnake_refine)
                else:
                    modules_to_freeze = [self.gcn]
                for m in modules_to_freeze:
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False
            # 确保 YOLO 可训练（除非显式要求冻结）
            if self.yolo is not None:
                for p in self.yolo.parameters():
                    p.requires_grad = not self.freeze_yolo

        # 单独的 YOLO 冻结开关
        if self.freeze_yolo and self.yolo is not None:
            for p in self.yolo.parameters():
                p.requires_grad = False
        if self.freeze_heatmap_detector and self.heatmap_detector is not None:
            for p in self.heatmap_detector.parameters():
                p.requires_grad = False

    # Detection filtering uses torchvision.ops.nms on decoded axis-aligned xyxy boxes.

    @staticmethod
    def should_use_gt_detection(use_gt_det, train_only, is_training, batch):
        return bool(use_gt_det) and batch is not None and (
            bool(is_training) or not bool(train_only)
        )

    @staticmethod
    def offset_gt_detection_classes(ct_cls, class_offset):
        return ct_cls - float(class_offset)

    @staticmethod
    def make_symmetric_stem_weight(weight):
        if not torch.is_tensor(weight) or weight.ndim != 4 or weight.size(1) != 3:
            shape = tuple(weight.shape) if torch.is_tensor(weight) else type(weight).__name__
            raise ValueError(f"Expected a [out_channels,3,kH,kW] stem weight, got {shape}")
        grayscale_weight = weight.sum(dim=1, keepdim=True) / 3.0
        return grayscale_weight.expand_as(weight).clone()

    @staticmethod
    def attach_py_detection_metadata(output, fail_on_mismatch=False, num_classes=None):
        detection = output.get('detection')
        py = output.get('py')
        if not torch.is_tensor(detection) or py is None:
            return output
        final_py = py[-1] if isinstance(py, (list, tuple)) and py else py
        if not torch.is_tensor(final_py):
            return output

        valid = detection[..., 4] > 1e-4
        scores = detection[..., 4][valid]
        classes = detection[..., 5][valid].long()
        if int(final_py.size(0)) != int(scores.numel()):
            if fail_on_mismatch:
                raise RuntimeError(
                    "Cannot associate final contours with detections: "
                    f"py={int(final_py.size(0))}, detections={int(scores.numel())}"
                )
            return output
        if num_classes is not None and classes.numel() > 0:
            num_classes = int(num_classes)
            if bool(((classes < 0) | (classes >= num_classes)).any()):
                raise RuntimeError(
                    f"Detection classes must be in [0, {num_classes - 1}], "
                    f"got min={int(classes.min())}, max={int(classes.max())}"
                )
        output['py_score'] = scores
        output['py_cls'] = classes
        return output

    @staticmethod
    def _batch_meta_tensor(batch, key, device, dtype):
        if batch is None or 'meta' not in batch or key not in batch['meta']:
            return None
        value = batch['meta'][key]
        if torch.is_tensor(value):
            return value.to(device=device, dtype=dtype)
        return torch.as_tensor(value, device=device, dtype=dtype)

    @staticmethod
    def _batch_tensor(batch, key, device, dtype):
        if batch is None or key not in batch:
            return None
        value = batch[key]
        if torch.is_tensor(value):
            return value.to(device=device, dtype=dtype)
        return torch.as_tensor(value, device=device, dtype=dtype)

    def _build_locate_feature_grid(self, batch, target_h, target_w, device, dtype, source_scale=1.0, sample_idx=None):
        inv_trans = self._batch_meta_tensor(batch, 'inv_trans_input', device, dtype)
        orig_hw = self._batch_meta_tensor(batch, 'orig_hw', device, dtype)
        flipped = self._batch_meta_tensor(batch, 'flipped', device, dtype)
        locate_scale = self._batch_tensor(batch, 'locate_feat_scale', device, dtype)
        grid_hw = self._batch_tensor(batch, 'locate_feat_grid_hw', device, dtype)
        patch_size = self._batch_tensor(batch, 'locate_feat_patch_size', device, dtype)
        locate_pad = self._batch_tensor(batch, 'locate_feat_pad', device, dtype)

        if inv_trans is None or orig_hw is None or locate_scale is None or grid_hw is None:
            raise KeyError(
                "Locate feature injection requires meta.inv_trans_input/meta.orig_hw and "
                "locate_feat_scale/locate_feat_grid_hw in batch"
            )

        # Slice to a single sample when called in per-sample mode
        if sample_idx is not None:
            i = sample_idx
            inv_trans = inv_trans[i:i+1]
            orig_hw = orig_hw[i:i+1]
            if flipped is not None:
                flipped = flipped[i:i+1]
            locate_scale = locate_scale[i:i+1]
            grid_hw = grid_hw[i:i+1]
            if patch_size is not None:
                patch_size = patch_size[i:i+1]
            if locate_pad is not None:
                locate_pad = locate_pad[i:i+1]

        bsz = int(inv_trans.size(0))
        if inv_trans.dim() != 3 or inv_trans.size(1) != 2 or inv_trans.size(2) != 3:
            raise ValueError(f"meta.inv_trans_input must be [B,2,3], got {tuple(inv_trans.shape)}")
        if patch_size is None:
            patch_size = torch.full((bsz, 1), 14.0, device=device, dtype=dtype)
        if locate_pad is None:
            locate_pad = torch.zeros((bsz, 4), device=device, dtype=dtype)
        if locate_scale.dim() == 1:
            locate_scale = locate_scale[:, None]
        if patch_size.dim() == 1:
            patch_size = patch_size[:, None]
        if flipped is None:
            flipped = torch.zeros((bsz, 1), device=device, dtype=dtype)
        if flipped.dim() == 1:
            flipped = flipped[:, None]

        ys = (torch.arange(target_h, device=device, dtype=dtype) + 0.5) * float(self.down_ratio) - 0.5
        xs = (torch.arange(target_w, device=device, dtype=dtype) + 0.5) * float(self.down_ratio) - 0.5
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        ones = torch.ones_like(xx)
        input_xy1 = torch.stack([xx, yy, ones], dim=0).view(3, -1)
        input_xy1 = input_xy1.unsqueeze(0).expand(bsz, -1, -1)
        src_xy = torch.bmm(inv_trans, input_xy1).view(bsz, 2, target_h, target_w)

        src_x = src_xy[:, 0]
        src_y = src_xy[:, 1]
        orig_w = orig_hw[:, 1].view(bsz, 1, 1)
        flip_mask = flipped.view(bsz, 1, 1) > 0.5
        src_x = torch.where(flip_mask, orig_w - src_x - 1.0, src_x)

        scale = locate_scale.view(bsz, 1, 1)
        source_scale = max(float(source_scale), 1e-6)
        patch = (patch_size.view(bsz, 1, 1) / source_scale).clamp(min=1e-6)
        pad_left = locate_pad[:, 0].view(bsz, 1, 1)
        pad_top = locate_pad[:, 1].view(bsz, 1, 1)
        feat_x = (src_x * scale + pad_left) / patch - 0.5
        feat_y = (src_y * scale + pad_top) / patch - 0.5

        gh = (grid_hw[:, 0].view(bsz, 1, 1) * source_scale).clamp(min=1.0)
        gw = (grid_hw[:, 1].view(bsz, 1, 1) * source_scale).clamp(min=1.0)
        norm_x = torch.where(gw > 1.0, (feat_x / (gw - 1.0)) * 2.0 - 1.0, torch.zeros_like(feat_x))
        norm_y = torch.where(gh > 1.0, (feat_y / (gh - 1.0)) * 2.0 - 1.0, torch.zeros_like(feat_y))
        return torch.stack([norm_x, norm_y], dim=-1)

    def apply_locate_feature_injection(self, cnn_feature, batch=None):
        if not getattr(self, 'locate_feat_inject', False) or getattr(self, 'locate_feat_adapter', None) is None:
            return cnn_feature, {}
        if batch is None or 'locate_feat' not in batch:
            raise KeyError(
                "cfg.locate_feat_inject=True but batch has no locate_feat. "
                "Check locate_feat_cache_root and dataset split."
            )

        feat = batch['locate_feat']
        device = cnn_feature.device
        dtype = cnn_feature.dtype
        target_h = int(cnn_feature.size(2))
        target_w = int(cnn_feature.size(3))
        bsz = int(cnn_feature.size(0))
        resample_padding = str(
            getattr(cfg, 'locate_feat_resample_padding_mode', 'zeros')
        ).strip().lower()
        if resample_padding not in ('zeros', 'border', 'reflection'):
            raise ValueError(
                'locate_feat_resample_padding_mode must be zeros/border/reflection, '
                'got {!r}'.format(resample_padding)
            )

        # Variable-size MoonViT caches are collated as a Python list of [C,H,W]
        # tensors. Process each sample independently so grid_sample stays exact.
        if isinstance(feat, (list, tuple)):
            if len(feat) != bsz:
                raise ValueError(
                    f"locate_feat list length {len(feat)} != batch size {bsz}"
                )
            # The feature maps may have different spatial sizes, but the target
            # grid and all batch metadata are shared by this replacement call.
            # Build the grids once instead of repeating full-batch transfers and
            # meshgrid/bmm work for every sample.
            grid = self._build_locate_feature_grid(
                batch, target_h, target_w, device, dtype,
            )
            aligned_list = []
            for i, f in enumerate(feat):
                f = f.to(device=device, dtype=dtype, non_blocking=True)
                if f.dim() == 3:
                    f = f.unsqueeze(0)  # [1, C, H, W]
                adapted_i = self.locate_feat_adapter(f)  # [1, C', H', W']
                grid_i = grid[i:i + 1]
                aligned_i = F.grid_sample(
                    adapted_i, grid_i,
                    mode='bilinear', padding_mode=resample_padding, align_corners=True,
                )
                aligned_list.append(aligned_i)
            aligned = torch.cat(aligned_list, dim=0)
        else:
            if not torch.is_tensor(feat):
                feat = torch.stack([torch.as_tensor(f) for f in feat])
            feat = feat.to(device=device, dtype=dtype, non_blocking=True)
            adapted = self.locate_feat_adapter(feat)
            grid = self._build_locate_feature_grid(
                batch, target_h, target_w, device, dtype,
            )
            aligned = F.grid_sample(
                adapted, grid,
                mode='bilinear', padding_mode=resample_padding, align_corners=True,
            )

        stats = {
            'locate_feat_residual_absmax': aligned.detach().abs().max(),
            'locate_feat_adapter_last_absmax': self.locate_feat_adapter.proj[-1].weight.detach().abs().max(),
        }
        return cnn_feature + aligned, stats

    def apply_locate_feature_replacement(self, cnn_feature, batch=None):
        if not self.locate_feat_replace or self.locate_feat_replacer is None:
            return cnn_feature, {}
        if batch is None or 'locate_feat' not in batch:
            raise KeyError(
                "cfg.locate_feat_replace=True but batch has no locate_feat. "
                "Check locate_feat_cache_dir/locate_feat_cache_root and dataset split."
            )

        feat = batch['locate_feat']
        device = cnn_feature.device
        dtype = cnn_feature.dtype
        target_h = int(cnn_feature.size(2))
        target_w = int(cnn_feature.size(3))
        bsz = int(cnn_feature.size(0))
        resample_padding = str(
            getattr(cfg, 'locate_feat_resample_padding_mode', 'zeros')
        ).strip().lower()
        if resample_padding not in ('zeros', 'border', 'reflection'):
            raise ValueError(
                'locate_feat_resample_padding_mode must be zeros/border/reflection, '
                'got {!r}'.format(resample_padding)
            )
        # The replacer upsamples the MoonViT grid by this factor, so the
        # resampling grid must be built against the upsampled cell size.
        source_scale = float(getattr(self.locate_feat_replacer, 'upscale', 2))

        # Variable-size MoonViT caches are collated as a Python list of [C,H,W]
        # tensors. Process each sample independently so grid_sample stays exact.
        if isinstance(feat, (list, tuple)):
            if len(feat) != bsz:
                raise ValueError(
                    "locate_feat list length {} != batch size {}".format(len(feat), bsz)
                )
            # Feature-map sizes can differ across samples, so the feature path
            # remains per-sample. Coordinate construction does not need to be.
            grid = self._build_locate_feature_grid(
                batch, target_h, target_w, device, dtype, source_scale=source_scale,
            )
            outs = []
            absmax = cnn_feature.new_tensor(0.0)
            for i in range(bsz):
                fi = feat[i]
                if not torch.is_tensor(fi):
                    fi = torch.as_tensor(fi)
                fi = fi.to(device=device, dtype=dtype, non_blocking=True)
                if fi.dim() == 3:
                    fi = fi.unsqueeze(0)
                ri = self.locate_feat_replacer(fi)
                grid_i = grid[i:i + 1]
                ri = F.grid_sample(
                    ri, grid_i, mode='bilinear',
                    padding_mode=resample_padding, align_corners=True,
                )
                outs.append(ri)
                absmax = torch.maximum(absmax, ri.detach().abs().max())
            replaced = torch.cat(outs, dim=0)
            return replaced, {'locate_feat_replace_absmax': absmax}

        if not torch.is_tensor(feat):
            feat = torch.as_tensor(feat)
        feat = feat.to(device=device, dtype=dtype, non_blocking=True)
        replaced = self.locate_feat_replacer(feat)
        grid = self._build_locate_feature_grid(
            batch, target_h, target_w, device, dtype,
            source_scale=float(self.locate_feat_replace_upscale),
        )
        replaced = F.grid_sample(
            replaced, grid, mode='bilinear',
            padding_mode=resample_padding, align_corners=True,
        )
        stats = {
            'locate_feat_replace_absmax': replaced.detach().abs().max(),
        }
        return replaced, stats

    def maybe_jitter_extreme_training_init(self, i_it_4py, c_it_4py, cnn_feature):
        scale_amp = max(float(getattr(cfg, 'ex_box_jitter_scale', 0.0)), 0.0)
        shift_amp = max(float(getattr(cfg, 'ex_box_jitter_shift', 0.0)), 0.0)
        if (
            (scale_amp <= 0.0 and shift_amp <= 0.0)
            or not torch.is_tensor(i_it_4py)
            or i_it_4py.numel() == 0
        ):
            return i_it_4py, c_it_4py, i_it_4py.new_tensor(0.0)

        xy_min = torch.min(i_it_4py, dim=1)[0]
        xy_max = torch.max(i_it_4py, dim=1)[0]
        wh = (xy_max - xy_min).clamp(min=1e-3)
        center = (xy_min + xy_max) * 0.5

        if scale_amp > 0.0:
            scale = 1.0 + (torch.rand((i_it_4py.size(0), 1), device=i_it_4py.device, dtype=i_it_4py.dtype) * 2.0 - 1.0) * scale_amp
        else:
            scale = torch.ones((i_it_4py.size(0), 1), device=i_it_4py.device, dtype=i_it_4py.dtype)
        if shift_amp > 0.0:
            shift = (torch.rand((i_it_4py.size(0), 2), device=i_it_4py.device, dtype=i_it_4py.dtype) * 2.0 - 1.0) * shift_amp * wh
        else:
            shift = torch.zeros((i_it_4py.size(0), 2), device=i_it_4py.device, dtype=i_it_4py.dtype)

        jitter_wh = (wh * scale).clamp(min=1e-3)
        jitter_center = center + shift
        x1y1 = jitter_center - jitter_wh * 0.5
        x2y2 = jitter_center + jitter_wh * 0.5
        max_xy = i_it_4py.new_tensor([
            max(float(cnn_feature.size(3) - 1), 1.0),
            max(float(cnn_feature.size(2) - 1), 1.0),
        ])
        x1y1 = torch.maximum(torch.minimum(x1y1, max_xy), torch.zeros_like(x1y1))
        x2y2 = torch.maximum(torch.minimum(x2y2, max_xy), torch.zeros_like(x2y2))
        lo = torch.minimum(x1y1, x2y2)
        hi = torch.maximum(x1y1, x2y2)
        hi = torch.maximum(hi, lo + 1e-3)
        hi = torch.minimum(hi, max_xy)
        lo = torch.minimum(lo, hi - 1e-3).clamp(min=0.0)
        jitter_box = torch.cat([lo, hi], dim=1)

        jitter_init = snake_decode.get_init(jitter_box.unsqueeze(0))
        jitter_init = snake_gcn_utils.uniform_upsample(jitter_init, snake_config.init_poly_num)[0]
        jitter_can = snake_gcn_utils.img_poly_to_can_poly(jitter_init)
        return jitter_init, jitter_can, i_it_4py.new_tensor(float(i_it_4py.size(0)))

    def predict_extreme_points(self, cnn_feature, i_it_4py, c_it_4py, ind):
        # extreme refine 头：输入是 bbox-derived 40 点初始轮廓和 CNN feature，
        # 输出是 refined 的 4 个 extreme points。它不是重新找目标类别，
        # 而是在 detector bbox 给出的局部区域内微调上/左/下/右关键点。
        if (
            self.extreme_refiner is None
            or self.extreme_fuse is None
            or not torch.is_tensor(i_it_4py)
            or i_it_4py.numel() == 0
        ):
            return None

        h, w = cnn_feature.size(2), cnn_feature.size(3)
        point_feat = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_4py, ind, h, w)
        center = (torch.min(i_it_4py, dim=1)[0] + torch.max(i_it_4py, dim=1)[0]) * 0.5
        center_feat = snake_gcn_utils.get_gcn_feature(cnn_feature, center[:, None], ind, h, w)
        point_feat = torch.cat([point_feat, center_feat.expand_as(point_feat)], dim=1)
        point_feat = self.extreme_fuse(point_feat)

        init_input = torch.cat([point_feat, c_it_4py.permute(0, 2, 1)], dim=1)
        adj = snake_gcn_utils.get_adj_ind(snake_config.adj_num, init_input.size(2), init_input.device)
        offset, _ = self.extreme_refiner(init_input, adj, i_it_4py)
        refined = i_it_4py + offset.permute(0, 2, 1)
        stride = max(int(snake_config.init_poly_num // 4), 1)
        # init_poly_num=40 时每 10 个点取一个，得到 4 个 refined extreme points。
        # 后续 diffusion inference 会用这些点重新构造 octagon 初始化。
        return refined[:, ::stride][:, :4]

    def attach_extreme_prediction(self, output, cnn_feature, batch=None):
        # 把 extreme refine 的结果挂到 output 字典上。
        # 推理时 output['detection'] 已包含 [x1,y1,x2,y2,score,class0]；
        # 这里调用 prepare_testing_init 将 bbox 变成初始 polygon，再预测 refined extreme。
        # 如果没有有效 bbox，就不会生成 output['ex']，后续 evolution 会得到空 contour。
        if not self.use_extreme_refine:
            return output

        if self.training and batch is not None:
            init = snake_gcn_utils.prepare_training_init(output, batch)
            i_it_4py = init['i_it_4py'].to(device=cnn_feature.device, dtype=cnn_feature.dtype)
            c_it_4py = init['c_it_4py'].to(device=cnn_feature.device, dtype=cnn_feature.dtype)
            ind = init['ind'].to(device=cnn_feature.device)
            i_it_4py, c_it_4py, jitter_count = self.maybe_jitter_extreme_training_init(
                i_it_4py,
                c_it_4py,
                cnn_feature,
            )
            ex_pred = self.predict_extreme_points(cnn_feature, i_it_4py, c_it_4py, ind)
            if ex_pred is not None:
                output['ex_pred'] = ex_pred
                output['i_gt_4py'] = init['i_gt_4py'].to(device=cnn_feature.device, dtype=cnn_feature.dtype)
                output['ex_box_jitter_count'] = jitter_count
            return output

        init = snake_gcn_utils.prepare_testing_init(output['detection'][..., :4], output['detection'][..., 4])
        ex = self.predict_extreme_points(
            cnn_feature,
            init['i_it_4py'].to(device=cnn_feature.device, dtype=cnn_feature.dtype),
            init['c_it_4py'].to(device=cnn_feature.device, dtype=cnn_feature.dtype),
            init['ind'].to(device=cnn_feature.device),
        )
        if ex is not None:
            output['ex'] = ex
            output['ex_py_ind'] = init['ind'].to(device=cnn_feature.device)
        return output

    def decode_detection_from_yolo(self, yolo_y, h, w):
        # yolo_y: (B, no, HW) where first 4*reg_max decoded to xywh already by head, but here yolo_y stores [xywh, cls_logits]
        # 将其转置为 (B, HW, C)
        y = yolo_y.permute(0, 2, 1).contiguous()
        xywh = y[..., :4]
        cls_logits = y[..., 4:]
        # 分数与类别
        cls_prob = cls_logits.sigmoid()
        score, cls_idx = cls_prob.max(dim=-1, keepdim=True)
        # 转 xyxy 并裁剪
        x, y_c, w_box, h_box = xywh.unbind(-1)
        x1 = x - w_box / 2
        y1 = y_c - h_box / 2
        x2 = x + w_box / 2
        y2 = y_c + h_box / 2
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        boxes = data_utils.clip_to_image(boxes, h, w)
        detection = torch.cat([boxes, score, cls_idx.float()], dim=-1)
        return detection

    def decode_detection_from_heatmap(self, ct_hm, wh):
        max_det = max(int(getattr(cfg, 'det_max_det', 100)) * 4, 100)
        class_offset = int(getattr(cfg, 'heatmap_class_offset', 0))
        ct_hm_decode = ct_hm[:, class_offset:, :, :] if class_offset > 0 else ct_hm
        if ct_hm_decode.size(1) <= 0:
            raise ValueError(f"heatmap_class_offset={class_offset} removes all heatmap channels")
        ct, detection = snake_decode.decode_ct_hm(ct_hm_decode, wh, K=max_det)
        h, w = ct_hm.shape[-2:]
        h_img, w_img = int(round(h * self.down_ratio)), int(round(w * self.down_ratio))

        ct = ct * self.down_ratio
        detection = detection.clone()
        detection[..., :4] = detection[..., :4] * self.down_ratio
        detection[..., :4] = data_utils.clip_to_image(detection[..., :4], h_img, w_img)
        return ct, detection

    def filter_detection_candidates(self, detection):
        conf_thres = float(getattr(cfg, 'det_conf_thresh', 0.20))
        iou_thres = float(getattr(cfg, 'det_iou_thresh', 0.30))
        max_det = int(getattr(cfg, 'det_max_det', 300))
        per_class = bool(getattr(cfg, 'per_class_nms', True))
        use_nms = bool(getattr(cfg, 'use_nms_for_snake', True))

        filtered = []
        for det_b in detection:
            keep = det_b[:, 4] >= conf_thres
            det_b = det_b[keep]
            if det_b.numel() == 0:
                filtered.append(det_b.new_zeros((0, 6)))
                continue

            det_b = det_b[torch.argsort(det_b[:, 4], descending=True)]
            if use_nms:
                if per_class:
                    kept_idx = []
                    for cls_id in det_b[:, 5].unique():
                        cls_mask = det_b[:, 5] == cls_id
                        cls_inds = cls_mask.nonzero(as_tuple=False).squeeze(1)
                        cls_keep = tv_nms(det_b[cls_inds, :4], det_b[cls_inds, 4], iou_thres)
                        kept_idx.append(cls_inds[cls_keep])
                    keep_idx = torch.cat(kept_idx, dim=0) if kept_idx else det_b.new_zeros((0,), dtype=torch.long)
                else:
                    keep_idx = tv_nms(det_b[:, :4], det_b[:, 4], iou_thres)
                if keep_idx.numel() > 0:
                    keep_idx = keep_idx[torch.argsort(det_b[keep_idx, 4], descending=True)]
                    det_b = det_b[keep_idx]
            filtered.append(det_b[:max_det])

        max_len = max((d.size(0) for d in filtered), default=0)
        if max_len == 0:
            return detection.new_zeros((detection.size(0), 0, 6))

        packed = detection.new_zeros((detection.size(0), max_len, 6))
        for b, det_b in enumerate(filtered):
            if det_b.size(0) > 0:
                packed[b, :det_b.size(0)] = det_b
        return packed

    def use_gt_detection(self, output, batch):
        ct_01 = batch['ct_01'].bool()
        batch_size = int(ct_01.size(0))
        feat_h, feat_w = output.get('feat_hw', (0, 0))
        if not feat_h or not feat_w:
            feat_h, feat_w = batch['ct_hm'].shape[-2:]

        counts = ct_01.long().sum(dim=1)
        max_len = int(counts.max().item()) if counts.numel() > 0 else 0
        device = batch['inp'].device
        dtype = batch['inp'].dtype
        if max_len == 0:
            output['ct'] = torch.zeros((batch_size, 0, 2), device=device, dtype=dtype)
            output['detection'] = torch.zeros((batch_size, 0, 6), device=device, dtype=dtype)
            return output

        packed_ct = torch.zeros((batch_size, max_len, 2), device=device, dtype=dtype)
        packed_det = torch.zeros((batch_size, max_len, 6), device=device, dtype=dtype)
        dr = float(self.down_ratio)

        for b in range(batch_size):
            keep = ct_01[b]
            n = int(keep.sum().item())
            if n == 0:
                continue
            ct_ind = batch['ct_ind'][b, keep].to(device=device)
            xs = (ct_ind % int(feat_w)).to(dtype=dtype)
            ys = (ct_ind // int(feat_w)).to(dtype=dtype)
            ct_feat = torch.stack([xs, ys], dim=1)
            wh_feat = batch['wh'][b, keep].to(device=device, dtype=dtype)
            bboxes_feat = torch.cat([
                xs[:, None] - wh_feat[..., 0:1] / 2,
                ys[:, None] - wh_feat[..., 1:2] / 2,
                xs[:, None] + wh_feat[..., 0:1] / 2,
                ys[:, None] + wh_feat[..., 1:2] / 2,
            ], dim=1)
            bboxes = bboxes_feat * dr
            bboxes = data_utils.clip_to_image(
                bboxes,
                int(round(feat_h * dr)),
                int(round(feat_w * dr)),
            )
            score = torch.ones((n, 1), device=device, dtype=dtype)
            ct_cls = batch['ct_cls'][b, keep].to(device=device, dtype=dtype).view(n, 1)
            class_offset = int(getattr(cfg, 'gt_detection_class_offset', 0))
            if class_offset == 0 and self.detector_backend.startswith('heatmap_'):
                # Backward compatibility for legacy heatmap configs; subtract once only.
                class_offset = int(getattr(cfg, 'heatmap_class_offset', 0))
            ct_cls = self.offset_gt_detection_classes(ct_cls, class_offset)
            packed_ct[b, :n] = ct_feat * dr
            packed_det[b, :n] = torch.cat([bboxes, score, ct_cls], dim=1)

        output['ct'] = packed_ct
        output['detection'] = packed_det

        return output

    def forward(self, x, batch=None):
        if self.detector_backend == 'samsnake_fm':
            if bool(getattr(cfg, 'samsnake_freeze_dla', False)):
                self.samsnake_dla.eval()
                with torch.no_grad():
                    dla_out, cnn_feature = self.samsnake_dla(x)
            else:
                dla_out, cnn_feature = self.samsnake_dla(x)
            h, w = cnn_feature.size(2), cnn_feature.size(3)
            ct_hm = net_utils.sigmoid(dla_out['ct_hm']) if 'ct_hm' in dla_out else None
            wh = F.relu(dla_out['wh']) if 'wh' in dla_out else None
            if ct_hm is not None and wh is not None:
                ct, raw_det = self.decode_detection_from_heatmap(ct_hm, wh)
                detection = self.filter_detection_candidates(raw_det)
            else:
                ct = torch.zeros((x.size(0), 0, 2), device=x.device, dtype=x.dtype)
                detection = torch.zeros((x.size(0), 0, 6), device=x.device, dtype=x.dtype)

            output = {
                'ct_hm': ct_hm,
                'wh': wh,
                'ct': ct,
                'detection': detection,
                'feat_hw': (h, w),
                'cnn_feature': cnn_feature,
            }
            if self.should_use_gt_detection(
                getattr(cfg, 'use_gt_det', False),
                getattr(cfg, 'use_gt_det_train_only', False),
                self.training,
                batch,
            ):
                self.use_gt_detection(output, batch)
            output = self.attach_extreme_prediction(output, cnn_feature, batch)

            if batch is not None:
                from lib.utils.snake.sam_init import attach_sam_testing_init, sam_init_enabled
                if sam_init_enabled():
                    output = attach_sam_testing_init(output, batch, device=x.device)

            if (
                self.samsnake_refine is not None
                and bool(getattr(cfg, 'v5_2_use_samsnake_refine', True))
                and torch.is_tensor(output.get('sam_i_it_py', None))
            ):
                raw_init = output['sam_i_it_py']
                output['sam_raw_i_it_py'] = raw_init
                centers = output.get('sam_ct', raw_init.mean(dim=1))
                py_ind = output.get(
                    'sam_py_ind',
                    torch.zeros((raw_init.size(0),), dtype=torch.long, device=raw_init.device),
                )
                coarse = self.samsnake_refine(
                    cnn_feature,
                    centers,
                    raw_init,
                    py_ind,
                    ignore=bool(getattr(cfg, 'samsnake_refine_ignore', False)),
                )
                output['samsnake_coarse_i_it_py'] = coarse
                output['sam_i_it_py'] = coarse
                output['sam_c_it_py'] = snake_gcn_utils.img_poly_to_can_poly(coarse)

            if (
                self.gcn is not None
                and (not self.freeze_snake)
                and (not bool(getattr(cfg, 'skip_diffusion_forward', False)))
            ):
                output = self.gcn(output, cnn_feature, batch)
            output['feat_hw'] = (h, w)
            output['cnn_feature'] = cnn_feature
            return output

        if (
            self.detector_backend.startswith('heatmap_')
            or self.detector_backend.startswith('convnext')
            or self.detector_backend.startswith('moonvit')
        ):
            use_gt_detection = self.should_use_gt_detection(
                getattr(cfg, 'use_gt_det', False),
                getattr(cfg, 'use_gt_det_train_only', False),
                self.training,
                batch,
            )
            skip_heatmap_detector = (
                use_gt_detection
                and bool(getattr(cfg, 'skip_heatmap_detector_when_gt', False))
                and self.locate_feat_replace
                and self.locate_feat_replacer is not None
                and not self.use_extreme_refine
            )
            if skip_heatmap_detector:
                det_loss_weight = float(getattr(cfg, 'loss_scales', {}).get('det', 1.0))
                if self.training and det_loss_weight != 0.0:
                    raise RuntimeError(
                        'skip_heatmap_detector_when_gt requires loss_scales.det=0 during training'
                    )
                stride = max(int(round(self.down_ratio)), 1)
                feature_h = (int(x.size(2)) + stride - 1) // stride
                feature_w = (int(x.size(3)) + stride - 1) // stride
                feature_channels = int(getattr(cfg, 'heatmap_feat_channels', 256))
                feature_dtype = (
                    torch.get_autocast_gpu_dtype()
                    if torch.is_autocast_enabled() else x.dtype
                )
                cnn_feature = x.new_zeros(
                    (x.size(0), feature_channels, feature_h, feature_w), dtype=feature_dtype
                )
                ct_hm = x.new_zeros(
                    (x.size(0), self.detector_num_classes, feature_h, feature_w), dtype=feature_dtype
                )
                wh = x.new_zeros((x.size(0), 2, feature_h, feature_w), dtype=feature_dtype)
                mask_logits = None
            else:
                cnn_feature, ct_hm, wh, mask_logits = self.heatmap_detector(x)
                if mask_logits is not None:
                    mask_guidance_alpha = float(getattr(cfg, 'heatmap_mask_guidance_alpha', 0.0))
                    if mask_guidance_alpha > 0.0:
                        mask_guidance = torch.sigmoid(mask_logits).amax(dim=1, keepdim=True)
                        cnn_feature = cnn_feature * (1.0 + mask_guidance_alpha * mask_guidance)
            det_cnn_feature = cnn_feature
            snake_feature, locate_feat_stats = self.apply_locate_feature_injection(det_cnn_feature, batch)
            if self.freeze_snake:
                replace_stats = {}
            else:
                snake_feature, replace_stats = self.apply_locate_feature_replacement(snake_feature, batch)
            locate_feat_stats.update(replace_stats)
            h, w = det_cnn_feature.size(2), det_cnn_feature.size(3)
            if use_gt_detection:
                # Training will replace these with GT detections below. Avoid
                # building an unused top-k/NMS graph from the heatmap outputs.
                ct = ct_hm.new_zeros((ct_hm.size(0), 0, 2))
                detection = ct_hm.new_zeros((ct_hm.size(0), 0, 6))
            else:
                ct, raw_det = self.decode_detection_from_heatmap(ct_hm, wh)
                detection = self.filter_detection_candidates(raw_det)

            output = {
                'ct_hm': ct_hm,
                'wh': wh,
                'ct': ct,
                'detection': detection,
                'feat_hw': (h, w),
                'cnn_feature': snake_feature,
            }
            output.update(locate_feat_stats)
            if mask_logits is not None:
                output['mask_logits'] = mask_logits
            if use_gt_detection:
                self.use_gt_detection(output, batch)
            output = self.attach_extreme_prediction(output, det_cnn_feature, batch)
            if (not self.training) and str(getattr(cfg, 'contour_init_method', 'octagon')).strip().lower() == 'sam':
                from lib.utils.snake.sam_init import attach_sam_testing_init
                output = attach_sam_testing_init(output, batch, device=x.device)
            if (not self.training) and batch is not None and batch.get('prev_contour_cache'):
                from lib.utils.snake.prev_contour_init import attach_prev_contour_testing_init
                output = attach_prev_contour_testing_init(output, batch, device=x.device)
            det_side_output = {
                k: output[k]
                for k in ('ct_hm', 'wh', 'ct', 'detection', 'mask_logits')
                if k in output
            }
            if (
                self.gcn is not None
                and (not self.freeze_snake)
                and (not bool(getattr(cfg, 'skip_diffusion_forward', False)))
            ):
                output = self.gcn(output, snake_feature, batch)
            output.update(det_side_output)
            output['feat_hw'] = (h, w)
            output['cnn_feature'] = snake_feature
            return output

        # YOLO 前向：返回 (y, feats)，其中 feats 为多尺度 head 特征列表
        yolo_out = self.yolo(x)
        # Detect 头推理默认返回 (y, feats)。y 是张量，feats 是多尺度特征列表
        if isinstance(yolo_out, tuple) and len(yolo_out) >= 2:
            yolo_y, yolo_feats = yolo_out[0], yolo_out[1]
        else:
            # 兼容返回单个张量的情况（导出/特殊路径）
            yolo_y, yolo_feats = yolo_out, []

        if self.use_swin_snake_feature:
            cnn_feature = self.swin_snake_feature(x)
            p2 = cnn_feature
        else:
            # 选择 P2 特征（最细一层，对应 stride=4，索引取 0）并压到 64 通道
            p2 = yolo_feats[0] if isinstance(yolo_feats, (list, tuple)) and len(yolo_feats) > 0 else None
            if p2 is None:
                raise RuntimeError("YOLO head features are not available; expected a list with P2 at index 0.")
            cnn_feature = self.cnn_proj(p2)
            if self.use_p3_features:
                p3 = yolo_feats[1] if isinstance(yolo_feats, (list, tuple)) and len(yolo_feats) > 1 else None
                if p3 is not None:
                    p3_up = F.interpolate(p3, size=p2.shape[-2:], mode='bilinear', align_corners=False)
                    cnn_feature = cnn_feature + self.cnn_proj_p3(p3_up)

        det_cnn_feature = cnn_feature
        snake_feature, locate_feat_stats = self.apply_locate_feature_injection(det_cnn_feature, batch)

        # 从 YOLO 输出构建 detection (B, N, 6) => [x1,y1,x2,y2,score,cls]
        # 并按配置执行阈值+NMS，确保训练/测试阶段一致地给 Snake 提供精简候选
        h, w = det_cnn_feature.size(2), det_cnn_feature.size(3) # 这个h,w是特征图的尺寸，相比图像尺寸缩小了4倍
        h_img, w_img = h*4, w*4
        raw_det = self.decode_detection_from_yolo(yolo_y, h_img, w_img)  # [B, HW, 6]

        detection = self.filter_detection_candidates(raw_det)

        # 构造与下游一致的 output 字典
        output = {}
        output.update({'detection': detection})
        output.update(locate_feat_stats)
        # 记录特征图尺寸，供可视化/坐标缩放使用（不再依赖 ct_hm）
        output['feat_hw'] = (h, w)
        output['cnn_feature'] = snake_feature

        # 训练时可选择使用 GT 框替换
        if self.should_use_gt_detection(
            getattr(cfg, 'use_gt_det', False),
            getattr(cfg, 'use_gt_det_train_only', False),
            self.training,
            batch,
        ):
            self.use_gt_detection(output, batch)

        output = self.attach_extreme_prediction(output, det_cnn_feature, batch)

        if (not self.training) and str(getattr(cfg, 'contour_init_method', 'octagon')).strip().lower() == 'sam':
            from lib.utils.snake.sam_init import attach_sam_testing_init
            output = attach_sam_testing_init(output, batch, device=x.device)

        if (not self.training) and batch is not None and batch.get('prev_contour_cache'):
            from lib.utils.snake.prev_contour_init import attach_prev_contour_testing_init
            output = attach_prev_contour_testing_init(output, batch, device=x.device)

        detector_side_output = {
            key: output[key]
            for key in (
                'ct_hm', 'wh', 'ct', 'detection', 'ex', 'ex_py_ind',
                'sam_i_it_py', 'sam_c_it_py', 'sam_py_ind', 'sam_ct',
                'sam_raw_i_it_py', 'samsnake_coarse_i_it_py',
            )
            if key in output
        }

        # 传入 Snake 进行演化。部分 evolution 实现返回一个新字典，
        # 因而需要像 heatmap 分支一样恢复 detector-side keys。
        if self.gcn is not None and not self.freeze_snake:
            output = self.gcn(output, snake_feature, batch)
        output.update(detector_side_output)
        output.update(locate_feat_stats)
        output['feat_hw'] = (h, w)
        output['cnn_feature'] = snake_feature

        # 暴露 YOLO 原始预测供损失使用
        output['yolo_preds'] = (yolo_y, yolo_feats)

        if (
            not self.training
            and str(getattr(cfg, 'contour_init_method', 'octagon')).strip().lower() == 'octagon'
        ):
            yolo_nc = int(
                getattr(self.yolo, 'yaml', {}).get(
                    'nc', getattr(cfg, 'yolo_num_classes', 0) or 0
                )
                or 0
            )
            output = self.attach_py_detection_metadata(
                output,
                fail_on_mismatch=True,
                num_classes=yolo_nc if yolo_nc > 0 else None,
            )

        return output


def get_network(num_layers, heads, head_conv=256, down_ratio=4, det_dir=''):
    network = Network(num_layers, heads, head_conv, down_ratio, det_dir)
    return network

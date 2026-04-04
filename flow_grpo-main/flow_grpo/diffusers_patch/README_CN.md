# diffusers_patch 中文说明

本目录提供了一组在 Hugging Face diffusers 管线基础上的“带对数概率（log-prob）回传”的轻量补丁，覆盖 SD3、FLUX、QwenImage 与 WAN 等模型/管线。同时包含若干 DreamBooth/LoRA 训练中用到的文本编码辅助函数。

核心思想是在每个去噪/采样时间步内，通过自定义的 SDE 反演函数计算并返回该步对数概率与中间潜变量，以便于强化学习（如 GRPO/IPO/DPO 等）或对齐训练时使用。

## 文件说明

- `sd3_sde_with_logprob.py`
  - 单步 SDE 反演核心函数 `sde_step_with_logprob`，适配 `FlowMatchEulerDiscreteScheduler`。
  - 支持 `sde` 与 `cps` 两种形式，返回：`prev_sample`、该步样本级 `log_prob`、该步均值与标准差信息。
  - 数值注意：内部统一转 `float32` 计算，避免 bf16 在均值/方差计算时溢出。

- `sd3_pipeline_with_logprob.py`
  - 基于 Stable Diffusion 3 的管线扩展：记录每一步 latent 与 log-prob，最终返回图像与这些中间量。
  - 保持与官方 SD3 管线相同的输入/CFG/解码流程，便于替换与复现。

- `sd3_pipeline_with_logprob_fast.py`
  - 与上类似，但支持“窗口化”记录策略：通过 `sde_window_size` 与 `sde_window_range`，仅在某个时间段记录/训练，避免最后一步接近图像分布时的数值尖峰。

- `flux_pipeline_with_logprob.py`
  - FLUX 文生图管线扩展：返回图像、所有中间 latent、文本/图像 ids 与每步 log-prob。

- `flux_kontext_pipeline_with_logprob.py`
  - FLUX-Kontext（图像-文本联合上下文）版本：在编码输入图像后与文本一同送入 transformer，记录每步 log-prob 与潜变量轨迹。

- `qwenimage_pipeline_with_logprob.py`
  - QwenImage 文生图：采用 True-CFG（条件/负条件合成后做范数重标定），支持窗口化记录，返回图像与中间量字典。

- `qwenimage_edit_pipeline_with_logprob.py`
  - QwenImage 图像编辑版本：对输入图像进行预处理/编码，与文本一同驱动生成，仅对生成 latent 部分做 SDE 更新与记录。

- `wan_pipeline_with_logprob.py`
  - WAN 视频生成：自定义 `sde_step_with_logprob` 适配 `UniPCMultistepScheduler`，在采样过程中记录 latent/log-prob，并可选计算 KL 奖励（与参考预测对比）。

- `wan_prompt_embedding.py`
  - WAN 用 T5 文本编码辅助，按 `max_sequence_length` 进行 pad 并扩展到 `num_videos_per_prompt`。

- `train_dreambooth_lora_sd3.py`
  - SD3/多路文本编码（两路 CLIP + 一路 T5）的 DreamBooth/LoRA 训练辅助，输出与 SD3 管线对齐的嵌入格式。

- `train_dreambooth_lora_flux.py`
  - FLUX 文本编码辅助（CLIP pooled + T5 token 级），用于 LoRA 训练或微调。

## 快速开始示例

以下示例展示如何在不改动原类方法的情况下，导入本目录的函数并以“函数形式”调用（传入 `self=pipe`）。

> 提示：确保所用调度器类型与对应补丁兼容（SD3 使用 `FlowMatchEulerDiscreteScheduler`，WAN 使用 `UniPCMultistepScheduler` 等）。

### SD3（标准版）

```python
import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob as sd3_with_lp

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers").to("cuda")
pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)

image, all_latents, all_log_probs = sd3_with_lp(
    pipe,
    prompt="a cat playing piano",
    num_inference_steps=28,
    guidance_scale=7.0,
    noise_level=0.7,
)
image[0].save("sd3_sample.png")
```

### SD3（窗口化 fast 版）

```python
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob_fast import pipeline_with_logprob as sd3_fast

image, all_latents, all_log_probs, all_timesteps = sd3_fast(
    pipe,
    prompt="a cozy cabin in snow",
    num_inference_steps=28,
    sde_window_size=8,
    sde_window_range=(0, 28),
    noise_level=0.7,
)
```

### FLUX

```python
from diffusers import FluxPipeline
from flow_grpo.diffusers_patch.flux_pipeline_with_logprob import pipeline_with_logprob as flux_with_lp

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev").to("cuda")

image, all_latents, latent_image_ids, text_ids, all_log_probs = flux_with_lp(
    pipe,
    prompt="ultra-detailed watercolor landscape",
    num_inference_steps=28,
)
image[0].save("flux_sample.png")
```

### FLUX-Kontext（图像+文本）

```python
from PIL import Image
from flow_grpo.diffusers_patch.flux_kontext_pipeline_with_logprob import pipeline_with_logprob as flux_kontext

img = Image.open("input.jpg").convert("RGB")
image, all_latents, latent_ids, text_ids, all_log_probs, image_latents = flux_kontext(
    pipe,
    image=img,
    prompt="describe and enhance the given scene",
    num_inference_steps=28,
)
```

### QwenImage（文生图）

```python
from flow_grpo.diffusers_patch.qwenimage_pipeline_with_logprob import pipeline_with_logprob as qwenimg

ret = qwenimg(
    pipe,
    prompt=["a panda in bamboo forest"],
    negative_prompt=["low quality"],
    true_cfg_scale=4.0,
    num_inference_steps=50,
    noise_level=0.7,
    sde_window_size=10,
    sde_window_range=(0, 50),
)
ret["images"][0].save("qwenimage_sample.png")
```

### QwenImage Edit（图像编辑）

```python
from PIL import Image
from flow_grpo.diffusers_patch.qwenimage_edit_pipeline_with_logprob import pipeline_with_logprob as qwenedit

img = Image.open("to_edit.jpg").convert("RGB")
ret = qwenedit(
    pipe,
    image=img,
    prompt=["make it sunset"],
    negative_prompt=["overexposed"],
    num_inference_steps=50,
    noise_level=0.7,
)
ret["images"][0].save("qwenedit_sample.png")
```

### WAN（视频生成）

```python
from flow_grpo.diffusers_patch.wan_pipeline_with_logprob import wan_pipeline_with_logprob as wan_with_lp

video_output, all_latents, all_log_probs, all_kl = wan_with_lp(
    pipe,
    prompt=["a flying dragon"],
    height=480,
    width=832,
    num_frames=81,
    num_inference_steps=50,
    guidance_scale=5.0,
    kl_reward=0.0,
)
# video_output.frames: 根据 pipeline 返回类型进行保存/可视化
```

## 训练/对齐建议

- log-prob 聚合：本实现默认按除 batch 维外的所有维度进行均值，得到“样本级”对数概率；
- 窗口化记录：通过 `sde_window_size`/`sde_window_range` 避免最后一步高斯过尖导致的数值溢出；
- 精度建议：SDE 步进内部统一使用 `float32` 做均值/方差计算；
- `noise_level`：对数概率的尺度因子，可根据具体任务微调（0.6~0.8 常见）。

## 兼容性与注意事项

- 请使用与管线相匹配的调度器：如 SD3 对应 `FlowMatchEulerDiscreteScheduler`，WAN 对应 `UniPCMultistepScheduler`；
- 以上示例均以“函数形式”调用：将 pipeline 实例作为第一个参数传入；
- 若需完全替换原 `__call__`，可自行将函数绑定为实例方法，但请谨慎操作以免影响其他功能。

## 许可

文件顶部保留了来源与许可证信息，遵循上游仓库（Hugging Face diffusers 等）的开源许可。以上补丁仅为研究用途提供便利。


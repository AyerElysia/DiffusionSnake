"""ImageReward 评分器

该模块封装了 ImageReward 模型，用于对「文本描述-图像」对进行相关性/质量评分。

主要包含一个简单的打分器类 `ImageRewardScorer`，提供可调用接口，
输入一组 `prompts` 与对应的 `PIL.Image` 列表，返回每对的分数（rewards）。
"""

from transformers import AutoProcessor, AutoModel  # 目前未使用，保留以兼容潜在扩展
from PIL import Image
import torch
import ImageReward as RM

class ImageRewardScorer(torch.nn.Module):
    """基于 ImageReward 的评分器。

    参数
    - device: 将模型加载到的设备，例如 "cuda"、"cuda:0" 或 "cpu"。
    - dtype: 模型权重与计算的数据类型，例如 `torch.float32`、`torch.float16`。

    说明
    - 默认加载 `ImageReward-v1.0` 权重，并设为 eval 模式与 no-grad，
      只用于推理打分，不进行训练与反向传播。
    """

    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        # 模型权重名称（HuggingFace / ImageReward 提供的检查点标识）
        self.model_path = "ImageReward-v1.0"
        # 推理设备与精度配置
        self.device = device
        self.dtype = dtype
        # 加载 ImageReward 模型，设为评估模式并切换精度
        self.model = RM.load(self.model_path, device=device).eval().to(dtype=dtype)
        # 关闭梯度以避免不必要的内存与计算开销
        self.model.requires_grad_(False)
        
    @torch.no_grad()
    def __call__(self, prompts, images):
        """对一批 prompt-图像对进行评分。

        参数
        - prompts: 文本描述列表，长度应与 images 一致。
        - images: `PIL.Image.Image` 列表，长度应与 prompts 一致。

        返回
        - rewards: 浮点分数列表，对应每个 (prompt, image) 的相关性/质量得分。
        """
        rewards = []
        # 逐对进行推理打分；ImageReward 的 inference_rank 支持排序场景，
        # 这里传入单图列表 [image]，返回 (排序索引, 分数)
        for prompt, image in zip(prompts, images):
            _, reward = self.model.inference_rank(prompt, [image])
            rewards.append(reward)
        return rewards

# Usage example
def main():
    """使用示例

    准备若干图像文件与对应文本描述，调用评分器得到分数列表。
    """
    scorer = ImageRewardScorer(
        device="cuda",        # 可改为 "cpu" 以在无 GPU 环境运行
        dtype=torch.float32    # 可根据显存与速度需求选择 fp16/fp32
    )

    # 待评估的图像文件路径列表
    images = [
        "astronaut.jpg",
    ]
    # 打开为 PIL.Image 列表
    pil_images = [Image.open(img) for img in images]

    # 与上方图像按顺序一一对应的文本描述
    prompts = [
        'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
    ]

    # 打印每个 (prompt, image) 的分数
    print(scorer(prompts, pil_images))

if __name__ == "__main__":
    main()

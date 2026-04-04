"""
CLIP评分器模块

该模块基于CLIP（Contrastive Language-Image Pre-training）模型，
实现了图像-文本对齐度评分和图像相似度计算功能。

主要功能：
1. 文本-图像对齐度评分：评估生成图像与文本提示的匹配程度
2. 图像相似度评分：计算两幅图像之间的视觉相似性
3. 多模态特征提取：提取图像和文本的嵌入向量

适用场景：
- 扩散模型的强化学习训练（GRPO）
- 图像生成质量评估
- 图像编辑任务的一致性评估
- 多模态AI系统评估

参考：https://github.com/RE-N-Y/imscore/blob/main/src/imscore/preference/model.py
"""

from importlib import resources
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from transformers import AutoImageProcessor, CLIPProcessor, CLIPModel
import numpy as np
from PIL import Image

def get_size(size):
    """
    解析图像尺寸配置
    
    将不同格式的尺寸配置统一转换为(height, width)格式。
    支持多种配置格式以适配不同的CLIP模型版本。
    
    Args:
        size: 尺寸配置，可以是：
            - int: 正方形图像的边长，如224 -> (224, 224)
            - dict: 包含"height"和"width"的字典
            - dict: 包含"shortest_edge"的字典（保持宽高比）
    
    Returns:
        tuple: (height, width)格式的尺寸
    
    Raises:
        ValueError: 当size格式无效时抛出异常
    """
    if isinstance(size, int):
        # 正方形图像
        return (size, size)
    elif "height" in size and "width" in size:
        # 明确指定高度和宽度
        return (size["height"], size["width"])
    elif "shortest_edge" in size:
        # 保持宽高比，指定最短边长度
        return size["shortest_edge"]
    else:
        raise ValueError(f"Invalid size: {size}")
    

def get_image_transform(processor: AutoImageProcessor):
    """
    根据CLIP处理器配置构建图像预处理变换管道
    
    根据CLIP模型的预处理配置，构建对应的torchvision变换管道，
    确保输入图像格式与模型预训练时一致。
    
    Args:
        processor: CLIP的图像处理器，包含预处理配置信息
        
    Returns:
        torchvision.transforms.Compose: 图像预处理管道
        
    管道包含：
        - Resize: 调整图像尺寸（如果do_resize=True）
        - CenterCrop: 中心裁剪（如果do_center_crop=True）
        - Normalize: 归一化（如果do_normalize=True）
    """
    config = processor.to_dict()
    
    # 构建尺寸调整变换
    resize = T.Resize(get_size(config.get("size"))) if config.get("do_resize") else nn.Identity()
    
    # 构建中心裁剪变换
    crop = T.CenterCrop(get_size(config.get("crop_size"))) if config.get("do_center_crop") else nn.Identity()
    
    # 构建归一化变换
    normalise = T.Normalize(mean=processor.image_mean, std=processor.image_std) if config.get("do_normalize") else nn.Identity()

    return T.Compose([resize, crop, normalise])

class ClipScorer(torch.nn.Module):
    """
    CLIP评分器主类
    
    基于OpenAI的CLIP模型实现多模态评分功能。使用CLIP-ViT-Large-Patch14模型，
    该模型在大规模图像-文本对数据集上进行预训练，具有强大的多模态理解能力。
    
    主要功能：
    1. 文本-图像对齐度评分：计算图像与文本描述的匹配程度
    2. 图像相似度评分：计算两幅图像的视觉相似性
    3. 特征提取：提取图像和文本的嵌入向量
    
    Args:
        device (str or torch.device): 计算设备，如'cuda'或'cpu'
    
    Attributes:
        device: 计算设备
        model: CLIP模型实例
        processor: CLIP文本和图像处理器
        tform: 图像预处理变换管道
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
        
        # 加载CLIP模型（ViT-Large-Patch14版本）
        # 该模型具有强大的视觉理解能力和文本理解能力
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        
        # 加载CLIP处理器，负责文本tokenization和图像预处理
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        # 构建图像预处理管道，确保与模型预训练时的预处理一致
        self.tform = get_image_transform(self.processor.image_processor)
        
        # 设置为评估模式，禁用dropout等训练特有的操作
        self.eval()
    
    def _process(self, pixels):
        """
        图像预处理
        
        对输入图像进行CLIP模型所需的预处理，包括：
        1. 调整尺寸
        2. 中心裁剪
        3. 归一化
        
        Args:
            pixels (torch.Tensor): 输入图像张量，形状为(N, C, H, W)或(C, H, W)
                                    值域应在[0, 1]范围内
        
        Returns:
            torch.Tensor: 预处理后的图像张量，形状与输入相同
        """
        dtype = pixels.dtype  # 保存原始数据类型
        pixels = self.tform(pixels)  # 应用预处理变换
        pixels = pixels.to(dtype=dtype)  # 恢复原始数据类型

        return pixels

    @torch.no_grad()
    def __call__(self, pixels, prompts, return_img_embedding=False):
        """
        计算文本-图像对齐度评分
        
        这是CLIP评分器的主要方法，用于评估图像与文本描述的匹配程度。
        通过计算图像嵌入和文本嵌入之间的余弦相似度来得到对齐度分数。
        
        Args:
            pixels (torch.Tensor): 输入图像张量，形状为(N, C, H, W)
                                       值域应为[0, 1]
            prompts (list): 文本提示列表，长度应与images的第一维度相同
            return_img_embedding (bool): 是否返回图像嵌入向量
                                          - True: 返回(评分, 图像嵌入)
                                          - False: 只返回评分
        
        Returns:
            如果return_img_embedding=True:
                tuple: (scores, image_embeds)
                    - scores (torch.Tensor): 图像-文本对齐度分数，形状为(N,)
                    - image_embeds (torch.Tensor): 图像嵌入向量，形状为(N, D)
            如果return_img_embedding=False:
                torch.Tensor: 图像-文本对齐度分数，形状为(N,)
                
        技术细节：
        1. 文本处理器进行tokenization、padding和truncation
        2. 图像进行预处理（调整尺寸、裁剪、归一化）
        3. CLIP模型计算图像-文本相似度矩阵
        4. 提取对角线元素得到一对一的相似度分数
        5. 除以30进行缩放（经验性参数，调整分数范围）
        """
        # 文本预处理：tokenization、padding、truncation
        texts = self.processor(
            text=prompts, 
            padding='max_length',    # 填充到最大长度
            truncation=True,         # 截断过长序列
            return_tensors="pt"     # 返回PyTorch张量
        ).to(self.device)
        
        # 图像预处理并移动到指定设备
        pixels = self._process(pixels).to(self.device)
        
        # CLIP模型前向传播，计算图像-文本相似度
        outputs = self.model(pixel_values=pixels, **texts)
        
        if return_img_embedding:
            # 返回评分和图像嵌入向量
            return outputs.logits_per_image.diagonal()/30, outputs.image_embeds
        else:
            # 只返回评分分数
            return outputs.logits_per_image.diagonal()/30

    @torch.no_grad()
    def image_similarity(self, pixels, ref_pixels):
        """
        计算图像相似度
        
        计算两组图像之间的视觉相似度，主要用于：
        1. 图像编辑任务：评估编辑前后图像的一致性
        2. 图像检索：找到视觉相似的图像
        3. 质量评估：评估生成图像与参考图像的相似程度
        
        Args:
            pixels (torch.Tensor): 目标图像张量，形状为(N, C, H, W)
            ref_pixels (torch.Tensor): 参考图像张量，形状为(N, C, H, W)
                                      必须与pixels的第一维度相同
        
        Returns:
            torch.Tensor: 相似度分数，形状为(N,)
                          每个值表示对应位置两幅图像的余弦相似度
                          值域为[-1, 1]，越接近1表示越相似
                          
        算法步骤：
        1. 对两组图像进行相同的预处理
        2. 使用CLIP提取图像特征向量
        3. 对特征向量进行L2归一化
        4. 计算余弦相似度矩阵
        5. 提取对角线元素得到一对一的相似度
        """
        # 预处理目标图像和参考图像
        pixels = self._process(pixels).to(self.device)
        ref_pixels = self._process(ref_pixels).to(self.device)

        # 提取目标图像的视觉特征向量
        pixel_embeds = self.model.get_image_features(pixel_values=pixels)
        
        # 提取参考图像的视觉特征向量
        ref_embeds = self.model.get_image_features(pixel_values=ref_pixels)

        # 对特征向量进行L2归一化（单位向量）
        # 这样余弦相似度就等于点积
        pixel_embeds = pixel_embeds / pixel_embeds.norm(p=2, dim=-1, keepdim=True)
        ref_embeds = ref_embeds / ref_embeds.norm(p=2, dim=-1, keepdim=True)

        # 计算相似度矩阵：sim[i,j] = pixels[i] · ref_pixels[j]
        sim = pixel_embeds @ ref_embeds.T
        
        # 提取对角线元素：sim[i] = pixels[i] · ref_pixels[i]
        sim = torch.diagonal(sim, 0)
        return sim


def main():
    """
    演示和测试函数
    
    展示ClipScorer的基本用法，包括：
    1. 创建CLIP评分器实例
    2. 加载和预处理图像
    3. 计算文本-图像对齐度评分
    4. 验证模块功能的正确性
    
    这是一个简单的测试用例，演示了如何使用CLIP评分器评估
    生成图像与文本描述的匹配程度。
    """
    # 创建CLIP评分器实例，使用CUDA加速
    scorer = ClipScorer(device='cuda')

    # 测试图像路径（使用相同的图像进行测试）
    images = [
        "assets/test.jpg",
        "assets/test.jpg"
    ]
    
    # 加载PIL图像
    pil_images = [Image.open(img) for img in images]
    
    # 对应的文本提示（一个匹配，一个不匹配）
    prompts = [
        'an image of cat',
        'not an image of cat'
    ]
    
    # 将PIL图像转换为numpy数组
    images = [np.array(img) for img in pil_images]
    images = np.array(images)
    
    # 调整维度顺序：从NHWC转换为NCHW格式
    images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
    
    # 转换为PyTorch张量并归一化到[0, 1]范围
    images = torch.tensor(images, dtype=torch.uint8) / 255.0
    
    # 计算文本-图像对齐度评分
    scores = scorer(images, prompts)
    print("CLIP Scores:", scores)
    
    # 预期结果：
    # - 第一个分数应该较高（图像确实包含猫）
    # - 第二个分数应该较低（图像不匹配"不是猫"的描述）


if __name__ == "__main__":
    # 运行演示测试
    main()
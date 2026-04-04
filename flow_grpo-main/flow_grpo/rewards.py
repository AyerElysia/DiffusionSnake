"""
奖励函数模块 - Flow-GRPO项目

该模块实现了各种奖励函数，用于评估生成图像的质量。
这些奖励函数涵盖了美学评分、文本-图像对齐、OCR识别、人类偏好等多个维度。

支持的奖励类型：
- JPEG压缩性：衡量图像的压缩效率
- 美学评分：基于CLIP的美学质量预测
- CLIP评分：文本-图像相似度
- PickScore：基于人类偏好的通用评分
- ImageReward：综合质量评估
- QwenVL：多模态大模型评分
- OCR：文本渲染质量
- GenEval：复杂组合提示评估
- UnifiedReward：统一奖励模型
"""

from PIL import Image
import io
import numpy as np
import torch
from collections import defaultdict

def jpeg_incompressibility():
    """
    JPEG不可压缩性奖励函数
    
    计算图像在JPEG压缩后的文件大小，作为图像复杂度的代理指标。
    更复杂的图像压缩后文件更大，因此不可压缩性更高。
    
    返回:
        function: 接受(images, prompts, metadata)参数的奖励函数
                 返回(文件大小数组KB, 元数据字典)
    """
    def _fn(images, prompts, metadata):
        """
        内部奖励计算函数
        
        Args:
            images: 输入图像，可以是torch.Tensor(NCHW格式)或numpy数组(NHWC格式)
            prompts: 文本提示列表（此函数不使用）
            metadata: 元数据字典（此函数不使用）
            
        Returns:
            tuple: (文件大小数组KB, 空元数据字典)
        """
        # 处理torch.Tensor输入，转换为NHWC格式的numpy数组
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            
        # 转换为PIL图像对象
        images = [Image.fromarray(image) for image in images]
        
        # 创建内存缓冲区用于JPEG压缩
        buffers = [io.BytesIO() for _ in images]
        
        # 以高质量(95)保存JPEG格式到内存缓冲区
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
            
        # 获取压缩后的文件大小（KB）
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    """
    JPEG可压缩性奖励函数
    
    jpeg_incompressibility的负向版本，值越小表示图像越容易压缩（质量可能较低）。
    通过除以500进行归一化，使奖励值在合理范围内。
    
    返回:
        function: 返回负向归一化的JPEG大小作为奖励
    """
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        """
        计算JPEG可压缩性奖励
        
        Args:
            images: 输入图像
            prompts: 文本提示
            metadata: 元数据
            
        Returns:
            tuple: (负向归一化的文件大小, 元数据)
        """
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew/500, meta  # 负号且除以500进行归一化

    return _fn

def aesthetic_score():
    """
    美学评分奖励函数
    
    使用预训练的美学评分模型对图像进行美学质量评估。
    基于CLIP模型的线性回归预测器，输出1-10的美学分数。
    
    返回:
        function: 返回图像美学评分的奖励函数
    """
    from flow_grpo.aesthetic_scorer import AestheticScorer

    # 初始化美学评分器，使用float32精度并加载到CUDA
    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        """
        计算美学评分
        
        Args:
            images: 输入图像，torch.Tensor或numpy数组格式
            prompts: 文本提示列表（此函数不使用）
            metadata: 元数据字典（此函数不使用）
            
        Returns:
            tuple: (美学评分数组, 空元数据字典)
        """
        # 处理torch.Tensor输入，转换为uint8格式
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            # 处理numpy数组输入，从NHWC转换为NCHW
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
            
        # 使用评分器计算美学分数
        scores = scorer(images)
        return scores, {}

    return _fn

def clip_score():
    """
    CLIP评分奖励函数
    
    使用CLIP模型计算图像与文本提示之间的相似度分数。
    CLIP能够理解图像内容并与文本进行语义对齐，
    是评估文本到图像生成质量的基础指标。
    
    返回:
        function: 返回CLIP相似度评分的奖励函数
    """
    from flow_grpo.clip_scorer import ClipScorer

    # 初始化CLIP评分器，使用float32精度并加载到CUDA
    scorer = ClipScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        """
        计算CLIP文本-图像相似度评分
        
        Args:
            images: 输入图像，torch.Tensor或numpy数组格式
            prompts: 对应的文本提示列表
            metadata: 元数据字典（此函数不使用）
            
        Returns:
            tuple: (CLIP相似度分数数组, 空元数据字典)
        """
        # 处理numpy数组输入，从NHWC转换为NCHW并归一化到[0,1]
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)/255.0
            
        # 计算图像与文本的CLIP相似度分数
        scores = scorer(images, prompts)
        return scores, {}

    return _fn

def image_similarity_score(device):
    """
    图像相似度评分函数
    
    使用CLIP模型计算两幅图像之间的相似度，
    主要用于图像编辑任务中评估编辑前后图像的一致性。
    
    Args:
        device: 计算设备 (cuda/cpu)
        
    返回:
        function: 返回图像间相似度评分的函数
    """
    from flow_grpo.clip_scorer import ClipScorer

    # 初始化CLIP评分器
    scorer = ClipScorer(device=device).cuda()

    def _fn(images, ref_images):
        """
        计算图像相似度评分
        
        Args:
            images: 生成的图像数组
            ref_images: 参考图像数组
            
        Returns:
            tuple: (图像相似度分数数组, 空元数据字典)
        """
        # 处理生成图像格式转换
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)/255.0
            
        # 处理参考图像格式转换
        if not isinstance(ref_images, torch.Tensor):
            # 转换PIL图像为numpy数组
            ref_images = [np.array(img) for img in ref_images]
            ref_images = np.array(ref_images)
            ref_images = ref_images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            ref_images = torch.tensor(ref_images, dtype=torch.uint8)/255.0
            
        # 计算图像间相似度分数
        scores = scorer.image_similarity(images, ref_images)
        return scores, {}

    return _fn

def pickscore_score(device):
    """
    PickScore评分奖励函数
    
    PickScore是基于人类偏好训练的文本到图像评分模型，
    能够准确评估生成图像与人类审美的对齐程度。
    这是目前最通用的图像质量评估指标之一。
    
    Args:
        device: 计算设备 (cuda/cpu)
        
    返回:
        function: 返回PickScore评分的奖励函数
    """
    from flow_grpo.pickscore_scorer import PickScoreScorer

    # 初始化PickScore评分器
    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        """
        计算PickScore评分
        
        Args:
            images: 输入图像，torch.Tensor或numpy数组格式
            prompts: 对应的文本提示列表
            metadata: 元数据字典（此函数不使用）
            
        Returns:
            tuple: (PickScore分数数组, 空元数据字典)
        """
        # 处理torch.Tensor输入，转换为PIL图像列表
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
            
        # 计算PickScore分数
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def imagereward_score(device):
    """
    ImageReward评分奖励函数
    
    ImageReward是一个综合性的文本到图像质量评估模型，
    同时考虑文本图像对齐、视觉保真度和安全性等多个维度。
    
    Args:
        device: 计算设备 (cuda/cpu)
        
    返回:
        function: 返回ImageReward评分的奖励函数
    """
    from flow_grpo.imagereward_scorer import ImageRewardScorer

    # 初始化ImageReward评分器
    scorer = ImageRewardScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        """
        计算ImageReward评分
        
        Args:
            images: 输入图像，torch.Tensor或numpy数组格式
            prompts: 对应的文本提示列表
            metadata: 元数据字典（此函数不使用）
            
        Returns:
            tuple: (ImageReward分数数组, 空元数据字典)
        """
        # 处理torch.Tensor输入，转换为PIL图像列表
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
            
        # 确保prompts是字符串列表格式
        prompts = [prompt for prompt in prompts]
        
        # 计算ImageReward分数
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def qwenvl_score(device):
    """
    QwenVL评分奖励函数
    
    使用Qwen-VL多模态大模型对生成图像进行评分。
    QwenVL具有强大的视觉理解和推理能力，
    能够评估复杂场景和细粒度的图像质量。
    
    Args:
        device: 计算设备 (cuda/cpu)
        
    返回:
        function: 返回QwenVL评分的奖励函数
    """
    from flow_grpo.qwenvl import QwenVLScorer

    # 初始化QwenVL评分器，使用bfloat16精度以提高推理效率
    scorer = QwenVLScorer(dtype=torch.bfloat16, device=device)

    def _fn(images, prompts, metadata):
        """
        计算QwenVL评分
        
        Args:
            images: 输入图像，torch.Tensor或numpy数组格式
            prompts: 对应的文本提示列表
            metadata: 元数据字典（此函数不使用）
            
        Returns:
            tuple: (QwenVL分数数组, 空元数据字典)
        """
        # 处理torch.Tensor输入，转换为PIL图像列表
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
            
        # 确保prompts是字符串列表格式
        prompts = [prompt for prompt in prompts]
        
        # 计算QwenVL分数
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

    
def ocr_score(device):
    """
    OCR评分奖励函数
    
    使用OCR技术评估图像中文本渲染的准确性和可读性。
    主要用于评估生成图像中的文字质量，如字体、清晰度、
    文本内容准确性等。
    
    Args:
        device: 计算设备 (cuda/cpu)
        
    返回:
        function: 返回OCR准确率评分的奖励函数
    """
    from flow_grpo.ocr import OcrScorer

    # 初始化OCR评分器
    scorer = OcrScorer()

    def _fn(images, prompts, metadata):
        """
        计算OCR评分
        
        Args:
            images: 输入图像，torch.Tensor或numpy数组格式
            prompts: 包含文本内容的提示列表
            metadata: 元数据字典（此函数不使用）
            
        Returns:
            tuple: (OCR准确率分数数组, 空元数据字典)
        """
        # 处理torch.Tensor输入，转换为numpy数组格式
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            
        # 使用OCR评分器计算文本识别准确率
        scores = scorer(images, prompts)
        # 将tensor转换为列表格式返回
        return scores, {}

    return _fn

def video_ocr_score(device):
    """
    视频/图像OCR评分奖励函数
    
    支持对视频序列和单帧图像进行OCR评分。
    能够处理视频中的文本渲染质量评估，
    用于文本到视频生成任务的优化。
    
    Args:
        device: 计算设备 (cuda/cpu)
        
    返回:
        function: 返回视频OCR评分的奖励函数
    """
    from flow_grpo.ocr import OcrScorer_video_or_image

    # 初始化视频OCR评分器
    scorer = OcrScorer_video_or_image()

    def _fn(images, prompts, metadata):
        """
        计算视频/图像OCR评分
        
        Args:
            images: 输入数据，可以是图像(4D)或视频(5D)格式的torch.Tensor
            prompts: 包含文本内容的提示列表
            metadata: 元数据字典（此函数不使用）
            
        Returns:
            tuple: (OCR准确率分数数组, 空元数据字典)
        """
        # 处理不同维度的torch.Tensor输入
        if isinstance(images, torch.Tensor):
            if images.dim() == 4 and images.shape[1] == 3:
                # 单帧图像: (N, C, H, W) -> (N, H, W, C)
                images = images.permute(0, 2, 3, 1) 
            elif images.dim() == 5 and images.shape[2] == 3:
                # 视频序列: (N, T, C, H, W) -> (N, T, H, W, C)
                images = images.permute(0, 1, 3, 4, 2)
                
            # 归一化到uint8格式
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            
        # 使用OCR评分器计算文本识别准确率
        scores = scorer(images, prompts)
        # 将tensor转换为列表格式返回
        return scores, {}

    return _fn

def deqa_score_remote(device):
    """
    DeQA远程评分奖励函数
    
    通过远程API调用DeQA(Design Quality Assessment)模型进行图像质量评估。
    DeQA是一个基于多模态LLM的图像质量评估模型，能够测量
    失真和纹理损伤对感知质量的影响。
    
    Args:
        device: 计算设备 (cuda/cpu)
        
    返回:
        function: 返回DeQA质量评分的奖励函数
        
    注意:
        需要预先启动DeQA服务器在 http://127.0.0.1:18086
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    # 配置参数
    batch_size = 64  # 批处理大小，避免内存溢出
    url = "http://127.0.0.1:18086"  # DeQA服务器地址
    
    # 配置HTTP会话，设置重试策略
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        """
        计算DeQA质量评分
        
        Args:
            images: 输入图像，torch.Tensor或numpy数组格式
            prompts: 文本提示列表（此函数不使用）
            metadata: 元数据字典（此函数不使用）
            
        Returns:
            tuple: (DeQA质量分数数组, 空元数据字典)
        """
        del prompts  # DeQA评估不需要文本提示
        
        # 处理torch.Tensor输入，转换为numpy数组格式
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            
        # 将图像分批处理，避免单次请求数据量过大
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        all_scores = []
        
        for image_batch in images_batched:
            jpeg_images = []

            # 使用JPEG格式压缩图像以减少传输数据量
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # 准备发送给DeQA服务器的数据格式
            data = {
                "images": jpeg_images,
            }
            data_bytes = pickle.dumps(data)

            # 向DeQA服务器发送请求
            response = sess.post(url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)

            # 收集评分结果
            all_scores += response_data["outputs"]

        return all_scores, {}

    return _fn

def geneval_score(device):
    """
    GenEval复杂组合提示评估奖励函数
    
    通过远程API调用GenEval模型评估复杂组合提示的生成质量。
    GenEval专门用于评估T2I模型在复杂组合提示上的表现，
    能够检测对象数量、空间关系、属性绑定等复杂语义。
    
    Args:
        device: 计算设备 (cuda/cpu)
        
    返回:
        function: 返回GenEval综合评估结果的函数
        
    返回值包含:
        - scores: 基础评分
        - rewards: 普通奖励
        - strict_rewards: 严格模式奖励（训练时使用）
        - group_rewards: 分组普通奖励
        - group_strict_rewards: 分组严格奖励
        
    注意:
        需要预先启动GenEval服务器在 http://127.0.0.1:18085
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    # 配置参数
    batch_size = 64  # 批处理大小
    url = "http://127.0.0.1:18085"  # GenEval服务器地址
    
    # 配置HTTP会话，设置重试策略
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadatas, only_strict):
        """
        计算GenEval复杂组合提示评估
        
        Args:
            images: 输入图像，torch.Tensor或numpy数组格式
            prompts: 文本提示列表（此函数不使用，信息包含在metadatas中）
            metadatas: 包含复杂提示结构的元数据列表
            only_strict: 是否只计算严格奖励（训练时为True以减少计算量）
            
        Returns:
            tuple: (基础评分, 普通奖励, 严格奖励, 分组普通奖励, 分组严格奖励)
        """
        del prompts  # 提示信息已在metadatas中
        
        # 处理torch.Tensor输入，转换为numpy数组格式
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            
        # 将图像和元数据分批处理
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadatas_batched = np.array_split(metadatas, np.ceil(len(metadatas) / batch_size))
        
        # 初始化结果收集器
        all_scores = []              # 基础评分
        all_rewards = []             # 普通奖励
        all_strict_rewards = []      # 严格奖励
        all_group_strict_rewards = []  # 分组严格奖励
        all_group_rewards = []       # 分组普通奖励
        
        # 分批处理图像和元数据
        for image_batch, metadata_batched in zip(images_batched, metadatas_batched):
            jpeg_images = []

            # 使用JPEG格式压缩图像
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # 准备发送给GenEval服务器的数据格式
            data = {
                "images": jpeg_images,
                "meta_datas": list(metadata_batched),
                "only_strict": only_strict,
            }
            data_bytes = pickle.dumps(data)

            # 向GenEval服务器发送请求
            response = sess.post(url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)

            # 收集各类评估结果
            all_scores += response_data["scores"]
            all_rewards += response_data["rewards"]
            all_strict_rewards += response_data["strict_rewards"]
            all_group_strict_rewards.append(response_data["group_strict_rewards"])
            all_group_rewards.append(response_data["group_rewards"])
            
        # 将分组奖励合并为字典格式
        all_group_strict_rewards_dict = defaultdict(list)
        all_group_rewards_dict = defaultdict(list)
        
        # 合并严格奖励
        for current_dict in all_group_strict_rewards:
            for key, value in current_dict.items():
                all_group_strict_rewards_dict[key].extend(value)
        all_group_strict_rewards_dict = dict(all_group_strict_rewards_dict)

        # 合并普通奖励
        for current_dict in all_group_rewards:
            for key, value in current_dict.items():
                all_group_rewards_dict[key].extend(value)
        all_group_rewards_dict = dict(all_group_rewards_dict)

        return all_scores, all_rewards, all_strict_rewards, all_group_rewards_dict, all_group_strict_rewards_dict

    return _fn

def unifiedreward_score_remote(device):
    """
    UnifiedReward远程评分奖励函数
    
    通过远程API调用UnifiedReward模型进行图像质量评估。
    UnifiedReward是当前最先进的多模态理解和生成奖励模型，
    在人类偏好排行榜上名列前茅。
    
    Args:
        device: 计算设备 (cuda/cpu)
        
    返回:
        function: 返回UnifiedReward质量评分的奖励函数
        
    注意:
        需要预先启动UnifiedReward服务器在指定地址
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    # 配置参数
    batch_size = 64  # 批处理大小
    url = "http://10.82.120.15:18085"  # UnifiedReward服务器地址
    
    # 配置HTTP会话，设置重试策略
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        """
        计算UnifiedReward质量评分
        
        Args:
            images: 输入图像，torch.Tensor或numpy数组格式
            prompts: 对应的文本提示列表
            metadata: 元数据字典（此函数不使用）
            
        Returns:
            tuple: (UnifiedReward质量分数数组, 空元数据字典)
        """
        # 处理torch.Tensor输入，转换为numpy数组格式
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            
        # 将图像和提示分批处理
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        
        # 分批处理图像和提示
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # 使用JPEG格式压缩图像以减少传输数据量
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # 准备发送给UnifiedReward服务器的数据格式
            data = {
                "images": jpeg_images,
                "prompts": prompt_batch
            }
            data_bytes = pickle.dumps(data)

            # 向UnifiedReward服务器发送请求
            response = sess.post(url, data=data_bytes, timeout=120)
            print("response: ", response)
            print("response: ", response.content)
            response_data = pickle.loads(response.content)

            # 收集评分结果
            all_scores += response_data["outputs"]

        return all_scores, {}

    return _fn

def unifiedreward_score_sglang(device):
    """
    UnifiedReward SGLang评分奖励函数
    
    使用SGLang部署的UnifiedReward模型进行图像质量评估。
    UnifiedReward是一个先进的多模态评估模型，能够综合评估
    文本-图像对齐度、视觉质量和美学等方面。
    
    Args:
        device: 计算设备 (cuda/cpu)
        
    返回:
        function: 返回UnifiedReward质量评分的奖励函数
        
    注意:
        需要预先启动SGLang服务器在 http://127.0.0.1:17140
    """
    import asyncio
    from openai import AsyncOpenAI
    import base64
    from io import BytesIO
    import re 

    def pil_image_to_base64(image):
        """
        将PIL图像转换为base64编码格式
        
        Args:
            image: PIL图像对象
            
        Returns:
            str: base64编码的图像数据URL
        """
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        return base64_qwen

    def _extract_scores(text_outputs):
        """
        从模型输出中提取评分分数
        
        Args:
            text_outputs: 模型文本输出列表
            
        Returns:
            list: 提取的评分数组
        """
        scores = []
        # 使用正则表达式匹配 "Final Score: X.X" 格式的分数
        pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
        for text in text_outputs:
            match = re.search(pattern, text)
            if match:
                try:
                    scores.append(float(match.group(1)))
                except ValueError:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores

    client = AsyncOpenAI(base_url="http://127.0.0.1:17140/v1", api_key="flowgrpo")
        
    async def evaluate_image(prompt, image):
        question = f"<image>\nYou are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nBased on the above criteria, assign a score from 1 to 5 after \'Final Score:\'.\nYour task is provided as follows:\nText Caption: [{prompt}]"
        images_base64 = pil_image_to_base64(image)
        response = await client.chat.completions.create(
            model="UnifiedReward-7b-v1.5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": images_base64},
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    async def evaluate_batch_image(images, prompts):
        tasks = [evaluate_image(prompt, img) for prompt, img in zip(prompts, images)]
        results = await asyncio.gather(*tasks)
        return results

    def _fn(images, prompts, metadata):
        # 处理Tensor类型转换
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        
        # 转换为PIL Image并调整尺寸
        images = [Image.fromarray(image).resize((512, 512)) for image in images]

        # 执行异步批量评估
        text_outputs = asyncio.run(evaluate_batch_image(images, prompts))
        score = _extract_scores(text_outputs)
        score = [sc/5.0 for sc in score]
        return score, {}
    
    return _fn

def multi_score(device, score_dict):
    """
    多奖励评分组合函数
    
    支持同时使用多个奖励函数，并按权重组合它们的评分结果。
    这是Flow-GRPO框架的核心奖励计算接口，支持灵活的多目标优化。
    
    Args:
        device: 计算设备 (cuda/cpu)
        score_dict: 奖励函数字典，格式为 {奖励名称: 权重}
                   例如: {"pickscore": 0.5, "ocr": 0.3, "aesthetic": 0.2}
                   
    返回:
        function: 返回多奖励组合评分的函数
        
    支持的奖励类型:
        - deqa: DeQA质量评估
        - ocr: 文本渲染质量
        - video_ocr: 视频/图像OCR
        - imagereward: ImageReward综合评估
        - pickscore: PickScore人类偏好
        - qwenvl: QwenVL多模态评估
        - aesthetic: 美学评分
        - jpeg_compressibility: 压缩性评估
        - unifiedreward: UnifiedReward统一评估
        - geneval: GenEval复杂提示评估
        - clipscore: CLIP文本-图像相似度
        - image_similarity: 图像相似度（用于编辑任务）
    """
    # 定义所有可用的奖励函数映射
    score_functions = {
        "deqa": deqa_score_remote,
        "ocr": ocr_score,
        "video_ocr": video_ocr_score,
        "imagereward": imagereward_score,
        "pickscore": pickscore_score,
        "qwenvl": qwenvl_score,
        "aesthetic": aesthetic_score,
        "jpeg_compressibility": jpeg_compressibility,
        "unifiedreward": unifiedreward_score_sglang,
        "geneval": geneval_score,
        "clipscore": clip_score,
        "image_similarity": image_similarity_score,
    }
    
    # 根据score_dict动态初始化所需的奖励函数
    score_fns = {}
    for score_name, weight in score_dict.items():
        # 检查函数是否需要device参数，并进行相应初始化
        score_fns[score_name] = score_functions[score_name](device) if 'device' in score_functions[score_name].__code__.co_varnames else score_functions[score_name]()

    def _fn(images, prompts, metadata, ref_images=None, only_strict=True):
        """
        计算多奖励组合评分
        
        Args:
            images: 输入图像，torch.Tensor或numpy数组格式
            prompts: 对应的文本提示列表
            metadata: 元数据字典
            ref_images: 参考图像列表（仅image_similarity使用）
            only_strict: 是否只计算严格奖励（仅geneval使用，训练时为True以减少计算量）
            
        Returns:
            tuple: (详细评分字典, 空元数据字典)
                  详细评分字典包含各奖励函数的分数和加权平均值
        """
        total_scores = []      # 加权总分累加器
        score_details = {}     # 详细评分结果存储
        
        # 遍历所有配置的奖励函数
        for score_name, weight in score_dict.items():
            if score_name == "geneval":
                # GenEval需要特殊处理，返回多维度评估结果
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = score_fns[score_name](images, prompts, metadata, only_strict)
                
                # 存储GenEval的详细评估结果
                score_details['accuracy'] = rewards
                score_details['strict_accuracy'] = strict_rewards
                
                # 存储分组评估结果
                for key, value in group_strict_rewards.items():
                    score_details[f'{key}_strict_accuracy'] = value
                for key, value in group_rewards.items():
                    score_details[f'{key}_accuracy'] = value
                    
            elif score_name == "image_similarity":
                # 图像相似度评估需要参考图像
                scores, rewards = score_fns[score_name](images, ref_images)
            else:
                # 其他常规奖励函数
                scores, rewards = score_fns[score_name](images, prompts, metadata)
                
            # 存储原始分数
            score_details[score_name] = scores
            
            # 计算加权分数
            weighted_scores = [weight * score for score in scores]
            
            # 累加到总分数中
            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]
        
        # 存储加权平均分数
        score_details['avg'] = total_scores
        return score_details, {}

    return _fn

def main():
    """
    主函数 - 奖励函数模块测试示例
    
    演示如何使用multi_score函数对图像进行多维度质量评估。
    这是一个完整的测试用例，展示了从图像加载到评分计算的完整流程。
    """
    import torchvision.transforms as transforms

    # 测试图像路径列表
    image_paths = [
        "nasa.jpg",  # 示例：NASA宇航员手套图像
    ]

    # 定义图像预处理变换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为torch.Tensor格式并归一化到[0,1]
    ])

    # 加载并预处理图像
    images = torch.stack([transform(Image.open(image_path).convert('RGB')) for image_path in image_paths])
    
    # 对应的文本提示
    prompts = [
        'A astronaut's glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    
    # 元数据字典（可根据需要添加额外信息）
    metadata = {}
    
    # 配置奖励函数权重字典
    # 这里只使用UnifiedReward进行演示，权重为1.0
    score_dict = {
        "unifiedreward": 1.0
    }
    # 初始化计算设备（优先使用CUDA）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化多奖励评分函数
    scoring_fn = multi_score(device, score_dict)
    
    # 执行评分计算
    scores, _ = scoring_fn(images, prompts, metadata)
    
    # 打印评分结果
    print("Scores:", scores)


if __name__ == "__main__":
    # 当脚本作为主程序运行时执行测试
    main()
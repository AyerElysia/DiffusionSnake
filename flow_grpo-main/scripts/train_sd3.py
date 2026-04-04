"""
Flow-GRPO SD3 训练脚本

这是Flow-GRPO框架的核心训练脚本，专门用于Stable Diffusion 3模型的强化学习训练。
该脚本实现了完整的GRPO（Group Relative Policy Optimization）算法流程，
包括数据采样、奖励计算、策略更新和模型评估等关键组件。

主要功能：
- SD3扩散模型的GRPO强化学习训练
- 支持多种奖励函数（OCR、PickScore、GenEval等）
- 实现了Flow-GRPO和Flow-GRPO-Fast两种变体
- 支持分布式训练和LoRA微调
- 集成了EMA、统计跟踪等训练优化技术

使用方法：
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml scripts/train_sd3.py --config config/grpo.py:general_ocr_sd3
"""

from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
import hashlib

# 命令行参数和配置管理
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger

# 扩散模型相关
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import is_compiled_module

# Flow-GRPO内部模块
import flow_grpo.prompts        # 提示生成模块
import flow_grpo.rewards         # 奖励函数模块
from flow_grpo.stat_tracking import PerPromptStatTracker  # 统计跟踪
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob  # 带对数概率的SD3管道
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob  # SDE采样
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt  # 提示编码

# 深度学习框架
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image

# LoRA微调和数据加载
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper  # 指数移动平均

# 配置tqdm进度条
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

# 命令行参数定义
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "训练配置文件路径")

# 获取logger
logger = get_logger(__name__)

class TextPromptDataset(Dataset):
    """
    简单文本提示数据集类
    
    用于加载普通的文本提示数据，每行一个提示。
    适用于OCR、PickScore等基础任务的训练。
    
    Args:
        dataset: 数据集根目录路径
        split: 数据集分割类型（'train' 或 'test'）
    """
    def __init__(self, dataset, split='train'):
        # 构建数据文件路径
        self.file_path = os.path.join(dataset, f'{split}.txt')
        # 加载所有文本提示，去除首尾空白字符
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.prompts)
    
    def __getitem__(self, idx):
        """获取单个样本，返回提示和空元数据字典"""
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        """
        批次整理函数
        
        Args:
            examples: 样本列表，每个样本包含prompt和metadata
            
        Returns:
            tuple: (提示列表, 元数据列表)
        """
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class GenevalPromptDataset(Dataset):
    """
    GenEval复杂提示数据集类
    
    专门用于加载GenEval任务的复杂组合提示数据。
    GenEval数据集包含复杂的元数据信息，用于评估模型的组合理解能力。
    
    Args:
        dataset: 数据集根目录路径
        split: 数据集分割类型（'train' 或 'test'）
    """
    def __init__(self, dataset, split='train'):
        # 构建元数据文件路径（jsonl格式）
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        # 加载包含复杂结构的元数据
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            # 从元数据中提取文本提示
            self.prompts = [item['prompt'] for item in self.metadatas]
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.prompts)
    
    def __getitem__(self, idx):
        """获取单个样本，返回提示和对应的复杂元数据"""
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        """
        批次整理函数
        
        Args:
            examples: 样本列表，每个样本包含prompt和复杂metadata
            
        Returns:
            tuple: (提示列表, 复杂元数据列表)
        """
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class DistributedKRepeatSampler(Sampler):
    """
    分布式K重复采样器
    
    这是GRPO训练中的关键组件，用于实现每个提示的多次采样。
    在强化学习中，我们需要对同一个prompt生成多个样本（k次重复），
    以计算该prompt的优势值并进行策略更新。
    
    Args:
        dataset: 数据集对象
        batch_size: 每个GPU/副本的批次大小
        k: 每个样本的重复次数（GRPO中的group size）
        num_replicas: 总副本数（GPU数量）
        rank: 当前副本的rank
        seed: 随机种子，确保所有副本同步
    """
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # 每个副本的批次大小
        self.k = k                    # 每个样本的重复次数
        self.num_replicas = num_replicas  # 总副本数
        self.rank = rank              # 当前副本的rank
        self.seed = seed              # 用于同步的随机种子
        
        # 计算每次迭代需要的唯一样本数量
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k无法整除n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # 唯一样本数量
        self.epoch = 0

    def __iter__(self):
        """
        无限迭代器，每次生成一个批次的样本索引
        
        算法流程：
        1. 随机选择m个唯一样本
        2. 每个样本重复k次，生成n*b个总样本
        3. 随机打乱以确保均匀分布
        4. 分配给各个副本
        """
        while True:
            # 生成确定性的随机序列，确保所有副本同步
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # 随机选择m个唯一样本
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            
            # 每个样本重复k次，生成n*b个总样本
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # 随机打乱以确保均匀分布
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
            # 分配样本给各个副本
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            
            # 返回当前副本的样本索引
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        """设置epoch数，用于跨epoch的随机状态同步"""
        self.epoch = epoch


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    """
    计算文本嵌入向量
    
    将文本提示转换为模型可以理解的嵌入向量，这是扩散模型中的关键预处理步骤。
    SD3使用多个文本编码器（CLIP、T5等），需要分别编码后组合。
    
    Args:
        prompt: 文本提示字符串
        text_encoders: 文本编码器列表
        tokenizers: 对应的分词器列表
        max_sequence_length: 最大序列长度
        device: 计算设备（cuda/cpu）
        
    Returns:
        tuple: (prompt_embeds, pooled_prompt_embeds)
               - prompt_embeds: 文本嵌入向量
               - pooled_prompt_embeds: 池化后的文本嵌入
    """
    with torch.no_grad():
        # 使用SD3专用的编码函数处理文本
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        # 将嵌入向量移动到指定设备
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds

def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    计算奖励标准差为零的提示比例
    
    这是GRPO训练中的关键诊断函数，用于评估训练过程的稳定性。
    如果过多提示的标准差为零，说明模型对这些提示的输出过于一致，
    可能缺乏多样性或梯度消失。
    
    Args:
        prompts: 提示列表
        gathered_rewards: 包含奖励的字典，必须包含'ori_avg'键
        
    Returns:
        tuple: (zero_std_ratio, prompt_std_devs)
               - zero_std_ratio: 标准差为零的提示比例
               - prompt_std_devs: 所有唯一提示的平均标准差
    """
    # Convert prompt list to NumPy array
    prompt_array = np.array(prompts)
    
    # Get unique prompts and their group information
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    
    # Group rewards for each prompt
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    
    # Calculate standard deviation for each group
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    
    # Calculate the ratio of zero standard deviation
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    
    return zero_std_ratio, prompt_std_devs.mean()

def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        # Use a stable hash (SHA256), then convert it to an integer seed
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')  # Take the first 4 bytes as part of the seed
        seed = (base_seed + prompt_hash_int) % (2**31) # Ensure the number is within a valid range
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators

        
def compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config):
    if config.train.cfg:
        noise_pred = transformer(
            hidden_states=torch.cat([sample["latents"][:, j]] * 2),
            timestep=torch.cat([sample["timesteps"][:, j]] * 2),
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = (
            noise_pred_uncond
            + config.sample.guidance_scale
            * (noise_pred_text - noise_pred_uncond)
        )
    else:
        noise_pred = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
    
    # compute the log prob of next_latents given latents under the current model
    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t

def eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    # test_dataloader = itertools.islice(test_dataloader, 2)
    all_rewards = defaultdict(list)
    for test_batch in tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, 
            text_encoders, 
            tokenizers, 
            max_sequence_length=128, 
            device=accelerator.device
        )
        # The last batch may not be full batch_size
        if len(prompt_embeds)<len(sample_neg_prompt_embeds):
            sample_neg_prompt_embeds = sample_neg_prompt_embeds[:len(prompt_embeds)]
            sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:len(prompt_embeds)]
        with autocast():
            with torch.no_grad():
                images, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution, 
                    noise_level=0,
                )
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        # yield to to make sure reward computation starts
        time.sleep(0)
        rewards, reward_metadata = rewards.result()

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)
    
    last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu().numpy()
    last_batch_prompt_ids = tokenizers[0](
        prompts,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    )
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_images_gather))
            # sample_indices = random.sample(range(len(images)), num_samples)
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]
            for key, value in all_rewards.items():
                print(key, value.shape)
            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | " + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
                },
                step=global_step,
            )
    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)

def main(_):
    """
    训练主函数
    
    这是整个GRPO训练流程的入口点，负责：
    1. 初始化分布式训练环境
    2. 加载SD3模型和相关组件
    3. 配置LoRA微调（如需要）
    4. 启动训练循环
    
    Args:
        _: 命令行参数（未使用，保持接口兼容性）
    """
    # === 基础设置：配置文件和运行参数 ===
    config = FLAGS.config

    # 生成唯一的运行ID，用于区分不同的训练实验
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # 计算每个轨迹中需要训练的时间步数
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    # === 分布式训练配置 ===
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    # 初始化Accelerator（HuggingFace分布式训练框架）
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # 梯度累积步数：时间步数 × 样本累积步数
        # 这确保我们在所有时间步上进行梯度累积
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    
    # === 日志和监控初始化 ===
    if accelerator.is_main_process:
        # 初始化Wandb实验跟踪
        wandb.init(project="flow_grpo")
        logger.info(f"\n配置信息：\n{config}")

    # === 随机种子设置 ===
    # device_specific确保不同GPU获得不同的随机序列，避免数据重复
    set_seed(config.seed, device_specific=True)

    # === 模型加载和初始化 ===
    # 加载预训练的SD3管道
    pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
    
    # 冻结非训练相关的模型参数以节省内存
    # VAE和文本编码器在训练中不需要更新梯度
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    # 只有在不使用LoRA时才冻结transformer
    pipeline.transformer.requires_grad_(not config.use_lora)

    # 提取文本编码器和分词器用于后续处理
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    # 禁用安全检查器（训练中不需要）
    pipeline.safety_checker = None
      # 配置进度条显示
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # === 混合精度配置 ===
    # 对于混合精度训练，将非训练权重（VAE、非LoRA文本编码器）转换为半精度
    # 这些权重仅用于推理，保持全精度不必要且浪费内存
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # 将模型组件移动到设备并设置数据类型
    pipeline.vae.to(accelerator.device, dtype=torch.float32)  # VAE保持float32
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_3.to(accelerator.device, dtype=inference_dtype)
    pipeline.transformer.to(accelerator.device)  # transformer保持默认精度

      # === LoRA微调配置 ===
    if config.use_lora:
        # 定义需要应用LoRA的注意力层目标模块
        # 这些是transformer中的关键注意力计算层
        target_modules = [
            "attn.add_k_proj",    # 键投影层
            "attn.add_q_proj",    # 查询投影层
            "attn.add_v_proj",    # 值投影层
            "attn.to_add_out",   # 输出投影层
            "attn.to_k",         # 键映射层
            "attn.to_out.0",     # 输出层第一部分
            "attn.to_q",         # 查询映射层
            "attn.to_v",         # 值映射层
        ]
        
        # LoRA配置参数
        transformer_lora_config = LoraConfig(
            r=32,                    # LoRA rank（秩）
            lora_alpha=64,           # LoRA缩放因子
            init_lora_weights="gaussian",  # 使用高斯分布初始化权重
            target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
            # After loading with PeftModel.from_pretrained, all parameters have requires_grad set to False. You need to call set_adapter to enable gradients for the adapter parameters.
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
    
    transformer = pipeline.transformer
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    # This ema setting affects the previous 20 × 8 = 160 steps on average.
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # 准备提示生成函数和奖励函数
    # multi_score函数支持多种奖励的组合，按权重计算最终奖励
    reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
    eval_reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)

    # 根据配置的数据集类型加载数据
    if config.prompt_fn == "general_ocr":
        # 加载OCR任务的训练和测试数据集
        train_dataset = TextPromptDataset(config.dataset, 'train')
        test_dataset = TextPromptDataset(config.dataset, 'test')

        # 创建无限循环的DataLoader，用于GRPO训练
        # 使用DistributedKRepeatSampler实现每个prompt的多次重复采样
        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,  # 每GPU批次大小
            k=config.sample.num_image_per_prompt,        # 每个prompt重复次数（group size）
            num_replicas=accelerator.num_processes,      # GPU数量
            rank=accelerator.process_index,              # 当前GPU rank
            seed=42                                        # 固定种子确保同步
        )

        # 创建训练DataLoader
        # 注意：不需要shuffle，因为Sampler已经控制了采样顺序
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,  # 使用自定义采样器
            num_workers=1,                # 工作进程数
            collate_fn=TextPromptDataset.collate_fn,  # 批次整理函数
            # persistent_workers=True  # 可选：保持工作进程活跃
        )

        # 创建测试DataLoader（常规模式）
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=TextPromptDataset.collate_fn,
            shuffle=False,      # 测试时不打乱
            num_workers=8,      # 可以使用更多工作进程加速
        )
    
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, 'train')
        test_dataset = GenevalPromptDataset(config.dataset, 'test')

        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=GenevalPromptDataset.collate_fn,
            # persistent_workers=True
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=GenevalPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    else:
        raise NotImplementedError("Only general_ocr is supported with dataset")


    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, test_dataloader = accelerator.prepare(transformer, optimizer, train_dataloader, test_dataloader)

    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # Train!
    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    # assert config.sample.train_batch_size >= config.train.batch_size
    # assert config.sample.train_batch_size % config.train.batch_size == 0
    # assert samples_per_epoch % total_train_batch_size == 0

    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)

    while True:
        #################### EVAL ####################
        pipeline.transformer.eval()
        if epoch % config.eval_freq == 0:
            eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, eval_reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters)
        if epoch % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
            save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)

        #################### SAMPLING ####################
        pipeline.transformer.eval()
        samples = []
        prompts = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, 
                text_encoders, 
                tokenizers, 
                max_sequence_length=128, 
                device=accelerator.device
            )
            prompt_ids = tokenizers[0](
                prompts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)

            # sample
            if config.sample.same_latent:
                generator = create_generator(prompts, base_seed=epoch*10000+i)
            else:
                generator = None
            with autocast():
                with torch.no_grad():
                    images, latents, log_probs = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution, 
                        noise_level=config.sample.noise_level,
                        generator=generator
                )

            latents = torch.stack(
                latents, dim=1
            )  # (batch_size, num_steps + 1, 16, 96, 96)
            log_probs = torch.stack(log_probs, dim=1)  # shape after stack (batch_size, num_steps)

            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.train_batch_size, 1
            )  # (batch_size, num_steps)

            # compute rewards asynchronously
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            # yield to to make sure reward computation starts
            time.sleep(0)

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[
                        :, :-1
                    ],  # each entry is the latent before timestep t
                    "next_latents": latents[
                        :, 1:
                    ],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            # accelerator.print(reward_metadata)
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        if epoch % 10 == 0 and accelerator.is_main_process:
            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_samples)

                for idx, i in enumerate(sample_indices):
                    image = images[i]
                    pil = Image.fromarray(
                        (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))  # 使用新的索引

                sampled_prompts = [prompts[i] for i in sample_indices]
                sampled_rewards = [rewards['avg'][i] for i in sample_indices]

                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompt:.100} | avg: {avg_reward:.2f}",
                            )
                            for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                        ],
                    },
                    step=global_step,
                )
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        # The purpose of repeating `adv` along the timestep dimension here is to make it easier to introduce timestep-dependent advantages later, such as adding a KL reward.
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        # gather rewards across processes
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}
        # log rewards and images
        if accelerator.is_main_process:
            wandb.log(
                {
                    "epoch": epoch,
                    **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items() if '_strict_accuracy' not in key and '_accuracy' not in key},
                },
                step=global_step,
            )

        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )
            advantages = stat_tracker.update(prompts, gathered_rewards['avg'])
            if accelerator.is_local_main_process:
                print("len(prompts)", len(prompts))
                print("len unique prompts", len(set(prompts)))

            group_size, trained_prompt_num = stat_tracker.get_stats()

            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts, gathered_rewards)

            if accelerator.is_main_process:
                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / (gathered_rewards['avg'].std() + 1e-4)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())

        del samples["rewards"]
        del samples["prompt_ids"]

        # Get the mask for samples where all advantages are zero across the time dimension
        mask = (samples["advantages"].abs().sum(dim=1) != 0)
        
        # If the number of True values in mask is not divisible by config.sample.num_batches_per_epoch,
        # randomly change some False values to True to make it divisible
        num_batches = config.sample.num_batches_per_epoch
        true_count = mask.sum()
        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True
        if accelerator.is_main_process:
            wandb.log(
                {
                    "actual_batch_size": mask.sum().item()//config.sample.num_batches_per_epoch,
                },
                step=global_step,
            )
        # Filter out samples where the entire time dimension of advantages is zero
        samples = {k: v[mask] for k, v in samples.items()}

        total_batch_size, num_timesteps = samples["timesteps"].shape
        # assert (
        #     total_batch_size
        #     == config.sample.train_batch_size * config.sample.num_batches_per_epoch
        # )
        assert num_timesteps == config.sample.num_steps

        #################### GRPO训练阶段 ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # 沿批次维度打乱样本，增加训练随机性
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # 重新分批以适应训练
            samples_batched = {
                k: v.reshape(-1, total_batch_size//config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }

            # 将字典的列表转换为列表的字典，便于迭代
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # 设置transformer为训练模式
            pipeline.transformer.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if config.train.cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = torch.cat(
                        [train_neg_prompt_embeds[:len(sample["prompt_embeds"])], sample["prompt_embeds"]]
                    )
                    pooled_embeds = torch.cat(
                        [train_neg_pooled_prompt_embeds[:len(sample["pooled_prompt_embeds"])], sample["pooled_prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]
                    pooled_embeds = sample["pooled_prompt_embeds"]

                train_timesteps = [step_index  for step_index in range(num_train_timesteps)]
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        with autocast():
                            prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config)
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    with transformer.module.disable_adapter():
                                        _, _, prev_sample_mean_ref, _ = compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config)

                        # GRPO算法核心逻辑 - 策略梯度优化
                        # 1. 限制优势值范围，防止训练不稳定
                        advantages = torch.clamp(
                            sample["advantages"][:, j],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        
                        # 2. 计算重要性采样比率 r = π_θ(a|s) / π_θ_old(a|s)
                        # 这是GRPO中的关键比率，用于新旧策略之间的比较
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        
                        # 3. 计算未裁剪的损失: -A * r
                        unclipped_loss = -advantages * ratio
                        
                        # 4. 计算裁剪后的损失: -A * clip(r, 1-ε, 1+ε)
                        # 这是PPO/GRPO的核心裁剪机制，防止策略更新过大
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,  # 下界：1-ε
                            1.0 + config.train.clip_range,  # 上界：1+ε
                        )
                        
                        # 5. 选择最大值作为最终策略损失（保守更新）
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        
                        # 6. 可选的KL散度正则化，防止策略偏离过远
                        if config.train.beta > 0:
                            # 计算当前策略与参考策略之间的KL散度
                            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * std_dev_t ** 2)
                            kl_loss = torch.mean(kl_loss)
                            # 总损失 = 策略损失 + KL正则化项
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss

                        # 收集训练统计信息，用于监控训练状态
                        info["approx_kl"].append(
                            0.5 * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                            # 近似KL散度，用于监控策略变化幅度
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (torch.abs(ratio - 1.0) > config.train.clip_range).float()
                            )
                            # 裁剪比例：有多少样本被裁剪了
                        )
                        info["clipfrac_gt_one"].append(
                            torch.mean(
                                (ratio - 1.0 > config.train.clip_range).float()
                            )
                            # 比率>1的裁剪比例（策略变好的样本）
                        )
                        info["clipfrac_lt_one"].append(
                            torch.mean(
                                (1.0 - ratio > config.train.clip_range).float()
                            )
                            # 比率<1的裁剪比例（策略变差的样本）
                        )
                        info["policy_loss"].append(policy_loss)  # 策略损失
                        if config.train.beta > 0:
                            info["kl_loss"].append(kl_loss)       # KL散度损失

                        info["loss"].append(loss)               # 总损失

                        # 反向传播和优化步骤
                        # 反向传播计算梯度
                        accelerator.backward(loss)
                        
                        # 梯度同步时进行梯度裁剪和参数更新
                        if accelerator.sync_gradients:
                            # 梯度裁剪，防止梯度爆炸
                            accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                            # 更新模型参数
                            optimizer.step()
                            # 清空梯度缓存
                            optimizer.zero_grad()

                    # 检查accelerator是否在后台执行了优化步骤
                    if accelerator.sync_gradients:
                        # 汇总统计信息并记录到wandb
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")  # 多GPU平均
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        
                        if accelerator.is_main_process:
                            wandb.log(info, step=global_step)  # 记录训练指标
                        global_step += 1
                        info = defaultdict(list)  # 重置统计信息收集器
                        
                # EMA（指数移动平均）更新，提高模型稳定性
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)
            # 确保在inner epoch结束时执行了优化步骤
            # assert accelerator.sync_gradients
        
        epoch+=1  # 进入下一个epoch
        
if __name__ == "__main__":
    # 启动训练主程序
    app.run(main)


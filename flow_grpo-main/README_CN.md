# Flow-GRPO: 基于在线强化学习的流匹配模型训练框架

## 项目概述

Flow-GRPO 是一个创新的深度学习项目，通过在线强化学习方法训练流匹配（Flow Matching）模型。该项目支持多种先进的文本到图像生成模型，包括 Stable Diffusion 3.5、FLUX、Qwen-Image 等，并提供了完整的训练、评估和部署解决方案。

### 🎯 核心特性

- **多模型支持**: 支持 SD3.5、FLUX.1-dev、FLUX.1-Kontext-dev、Qwen-Image、Qwen-Image-Edit、Wan2.1 等
- **强化学习训练**: 基于 GRPO（Generalized Relative Preference Optimization）算法
- **Flow-GRPO-Fast**: 高效训练变体，仅需1-2个去噪步骤的训练
- **多奖励函数**: 支持 GenEval、OCR、PickScore、ImageReward、CLIPScore 等多种奖励模型
- **分布式训练**: 完整的多节点、多GPU训练支持
- **灵活配置**: 基于 ml_collections 的模块化配置系统

---

## 📁 项目结构

```
flow_grpo-main/
├── config/                     # 配置文件目录
│   ├── base.py                # 基础配置
│   ├── grpo.py                # GRPO 训练配置
│   ├── dpo.py                 # DPO 训练配置
│   └── sft.py                 # SFT 训练配置
├── dataset/                    # 数据集目录
│   ├── counting_edit/         # 计数编辑任务数据
│   ├── drawbench/             # 绘图基准测试数据
│   ├── geneval/               # GenEval 评估数据
│   ├── ocr/                   # OCR 文本渲染数据
│   └── pickscore/             # PickScore 偏好数据
├── flow_grpo/                  # 核心模块目录
│   ├── diffusers_patch/       # Diffusers 库补丁
│   ├── assets/                # 模型资源和可视化文件
│   ├── aesthetic_scorer.py    # 美学评分器
│   ├── clip_scorer.py         # CLIP 评分器
│   ├── ocr.py                 # OCR 评估工具
│   ├── rewards.py             # 奖励函数集合
│   └── prompts.py             # 提示词处理
├── scripts/                    # 训练和部署脚本
│   ├── accelerate_configs/    # Accelerate 分布式配置
│   ├── demo/                  # 演示脚本
│   ├── multi_node/            # 多节点训练脚本
│   └── single_node/           # 单节点训练脚本
└── setup.py                    # 安装配置文件
```

---

## 🚀 快速开始

### 1. 环境设置

#### 基础环境
```bash
# 克隆仓库
git clone https://github.com/yifan123/flow_grpo.git
cd flow_grpo

# 创建并激活 conda 环境
conda create -n flow_grpo python=3.10.16
conda activate flow_grpo

# 安装项目依赖
pip install -e .
```

#### 模型下载
为了避免多GPU训练时的重复下载和存储浪费，请预先下载所需模型：

**基础生成模型**
- SD3.5: `stabilityai/stable-diffusion-3.5-medium`
- FLUX: `black-forest-labs/FLUX.1-dev`

**奖励模型**
- PickScore: `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`, `yuvalkirstain/PickScore_v1`
- CLIPScore: `openai/clip-vit-large-patch14`
- Aesthetic Score: `openai/clip-vit-large-patch14`

### 2. 奖励模型环境配置

#### GenEval 奖励
```bash
# 创建新环境并安装依赖
conda create -n geneval python=3.10.16
conda activate geneval
# 按照 reward-server 仓库说明安装 GenEval 相关依赖
```

#### OCR 奖励
```bash
pip install paddlepaddle-gpu==2.6.2
pip install paddleocr==2.9.1
pip install python-Levenshtein

# 预下载 OCR 模型
python -c "from paddleocr import PaddleOCR; ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False, show_log=False)"
```

#### ImageReward 奖励
```bash
pip install image-reward
pip install git+https://github.com/openai/CLIP.git
```

#### UnifiedReward 奖励
```bash
conda create -n sglang python=3.10.16
conda activate sglang
pip install "sglang[all]"

# 启动 UnifiedReward 服务
python -m sglang.launch_server \
    --model-path CodeGoat24/UnifiedReward-7b-v1.5 \
    --api-key flowgrpo --port 17140 \
    --chat-template chatml-llava \
    --enable-p2p-check --mem-fraction-static 0.85
```

### 3. 训练启动

#### 单节点训练
```bash
# SD3.5 模型训练
bash scripts/single_node/grpo.sh

# FLUX 模型训练
bash scripts/single_node/grpo_flux.sh
```

#### 多节点训练

**SD3.5 多节点训练**
```bash
# 主节点
bash scripts/multi_node/sd3/main.sh
# 其他节点
bash scripts/multi_node/sd3/main1.sh
bash scripts/multi_node/sd3/main2.sh
bash scripts/multi_node/sd3/main3.sh
```

**FLUX 多节点训练**
```bash
# 主节点
bash scripts/multi_node/flux/main.sh
# 其他节点
bash scripts/multi_node/flux/main1.sh
bash scripts/multi_node/flux/main2.sh
bash scripts/multi_node/flux/main3.sh
```

---

## 🎯 核心功能详解

### 1. Flow-GRPO 算法

Flow-GRPO 是一种基于广义相对偏好优化的强化学习算法，专门用于流匹配模型的训练：

#### 算法特点
- **在线强化学习**: 通过在线采样和奖励反馈优化模型
- **流匹配兼容**: 专门适配 Flow Matching 模型的训练需求
- **稳定训练**: 通过 KL 散度正则化确保训练稳定性

#### 训练流程
1. **提示词采样**: 从数据集中采样训练提示词
2. **图像生成**: 使用当前模型生成候选图像
3. **奖励计算**: 使用多种奖励函数评估生成质量
4. **策略更新**: 基于 GRPO 算法更新模型参数
5. **重复迭代**: 循环执行直到收敛

### 2. Flow-GRPO-Fast 高速训练

Flow-GRPO-Fast 是项目的重要创新，大幅提升训练效率：

#### 核心思想
- **分叉采样**: 在 ODE 采样的中间步骤注入噪声，切换到 SDE 采样
- **局部训练**: 仅在1-2个随机步骤上进行训练，大幅减少计算成本
- **确定性分支**: 分叉前后的采样保持确定性，确保一致性

#### 优势特性
- **训练效率**: 相比完整训练提速5-10倍
- **性能保持**: 在多项任务上接近或超越完整训练效果
- **内存友好**: 显著降低显存需求

#### 配置示例
```python
# 在 config/grpo.py 中配置 Flow-GRPO-Fast
config.sample.sde_window_size = 2        # SDE 窗口大小
config.sample.sde_window_range = [5, 15] # 窗口位置范围
config.train.clip_range = 1e-3           # 较小的裁剪范围
```

### 3. 多奖励函数系统

项目支持丰富的奖励函数组合，实现多目标优化：

#### 奖励函数类型

**文本理解与对齐**
- **PickScore**: 基于人类偏好的通用 T2I 评估
- **ImageReward**: 文本-图像对齐、视觉保真度和安全性评估
- **QwenVL**: 基于大语言模型的实验性评估

**图像质量评估**
- **Aesthetic**: 基于 CLIP 的美学评分预测
- **DeQA**: 基于多模态 LLM 的图像质量评估
- **JPEG_Compressibility**: 图像压缩质量代理指标

**任务特定评估**
- **GenEval**: 复杂组合提示词的 T2I 模型评估
- **OCR**: 基于 OCR 的文本渲染质量评估
- **UnifiedReward**: 最先进的多模态理解与生成奖励模型

#### 多奖励组合配置
```python
# 在配置文件中设置多奖励权重
config.reward_fn = {
    "pickscore": 0.5,     # 人类偏好权重
    "ocr": 0.2,           # 文本渲染质量权重
    "aesthetic": 0.3      # 美学质量权重
}
```

### 4. 模型支持详解

#### Stable Diffusion 3.5
- **模型路径**: `stabilityai/stable-diffusion-3.5-medium`
- **特点**: 高质量图像生成，支持复杂提示词理解
- **配置示例**: 
  ```python
  config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
  config.sample.num_steps = 10
  config.sample.guidance_scale = 4.5
  ```

#### FLUX.1-dev
- **模型路径**: `black-forest-labs/FLUX.1-dev`
- **特点**: 先进的流匹配架构，优秀的生成质量
- **环境要求**: 需要从主分支安装 diffusers
  ```bash
  pip install git+https://github.com/huggingface/diffusers.git
  ```

#### Qwen-Image 系列
- **Qwen-Image**: 通用的文本到图像生成模型
- **Qwen-Image-Edit**: 图像编辑专用模型
- **特点**: 统一了 Flow-GRPO 和 Flow-GRPO-Fast 实现
- **配置参数**:
  ```python
  config.sample.sde_window_size = 2
  config.sample.sde_window_range = [5, 15]
  ```

#### Wan2.1 视频生成
- **模型路径**: `hf_cache/Wan2.1-T2V-1.3B-Diffusers`
- **应用场景**: 文本到视频生成任务
- **特殊配置**:
  ```python
  config.height = 240
  config.width = 416
  config.frames = 33
  config.sample.num_steps = 20
  ```

---

## ⚙️ 配置系统详解

### 1. 配置文件结构

项目采用 ml_collections 进行模块化配置管理：

```python
# config/grpo.py 示例配置
def general_ocr_sd3():
    config = base.get_config()
    
    # 模型配置
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    
    # 训练参数
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2
    
    # 采样参数
    config.sample.num_steps = 10
    config.sample.guidance_scale = 4.5
    
    # 奖励函数
    config.reward_fn = {"ocr": 1.0}
    
    return config
```

### 2. 关键超参数说明

#### 训练相关参数
- **batch_size**: 训练批次大小
- **gradient_accumulation_steps**: 梯度累积步数
- **learning_rate**: 学习率设置
- **num_epochs**: 训练轮数
- **clip_range**: GRPO 裁剪范围

#### 采样相关参数
- **num_steps**: 去噪步数
- **guidance_scale**: 引导尺度
- **num_image_per_prompt**: 每个提示词生成的图像数量
- **same_latent**: 是否对相同提示词使用相同噪声

#### Flow-GRPO-Fast 参数
- **sde_window_size**: SDE 采样窗口大小
- **sde_window_range**: SDE 窗口位置范围
- **noise_level**: 噪声水平控制

### 3. 分布式训练配置

#### 多节点配置示例
```yaml
# scripts/accelerate_configs/multi_node.yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_process_ip: 192.168.1.100  # 主节点 IP
main_process_port: 29500
mixed_precision: fp16
num_machines: 4                 # 机器数量
num_processes: 32               # 总进程数
rdzv_backend: static
```

---

## 📊 训练监控与评估

### 1. 训练指标监控

项目集成了 WandB 进行训练过程监控：

- **奖励曲线**: 各项奖励函数的变化趋势
- **KL 散度**: 策略变化的 KL 散度监控
- **损失函数**: 训练损失的详细分解
- **生成样本**: 定期保存生成的图像样本

### 2. 评估指标

#### GenEval 评估
- 复杂组合提示词的理解能力
- 多对象生成和空间关系处理
- 属性绑定和计数能力

#### OCR 评估
- 文本渲染准确性
- 字符识别率
- 文本-图像对齐质量

#### PickScore 评估
- 人类偏好对齐程度
- 图像美学质量
- 整体生成质量

### 3. 性能优化建议

#### 训练效率优化
- 使用 Flow-GRPO-Fast 进行快速实验
- 合理设置 batch_size 和 gradient_accumulation_steps
- 采用混合精度训练（fp16/bf16）

#### 内存优化
- 使用梯度检查点（gradient checkpointing）
- 合理配置 num_image_per_prompt
- 启用模型并行和数据并行

---

## 🔧 高级功能与扩展

### 1. 自定义奖励函数

#### 添加新的奖励函数
```python
# 在 flow_grpo/rewards.py 中添加新奖励
class CustomReward:
    def __init__(self, model_path, device="cuda"):
        self.model = load_model(model_path)
        self.device = device
    
    def __call__(self, prompts, images):
        # 实现奖励计算逻辑
        scores = self.compute_scores(prompts, images)
        return scores
```

#### 注册奖励函数
```python
# 在配置文件中使用自定义奖励
config.reward_fn = {
    "custom_reward": 0.5,
    "pickscore": 0.5
}
```

### 2. 支持新模型

#### 添加模型支持步骤

1. **创建管道补丁文件**
   ```python
   # flow_grpo/diffusers_patch/new_model_pipeline_with_logprob.py
   # 基于原始 pipeline 添加对数概率计算支持
   ```

2. **创建训练脚本**
   ```python
   # scripts/train_new_model.py
   # 基于 DreamBooth 示例创建训练脚本
   ```

3. **实现 SDE 采样**
   ```python
   # flow_grpo/diffusers_patch/new_model_sde_with_logprob.py
   # 实现带对数概率的 SDE 采样
   ```

4. **验证实现**
   - 检查 SDE 采样正确性
   - 确保 on-policy 一致性
   - 调整噪声水平和其他超参数

### 3. 数据集处理

#### 支持的数据集格式
- **文本文件**: 每行一个提示词
- **JSONL 格式**: 包含提示词和元数据
- **图像-文本对**: 用于特定任务的数据

#### 数据集配置示例
```python
# 在配置文件中指定数据集
config.dataset = "path/to/your/dataset"
config.prompt_fn = "custom_prompt_function"  # 提示词处理函数
```

---

## 🛠️ 故障排除

### 1. 常见问题

#### 训练不稳定
- **症状**: 奖励曲线剧烈波动，训练发散
- **解决方案**: 
  - 减小 `clip_range` 参数
  - 调整 `learning_rate`
  - 检查奖励函数是否正常工作

#### 显存不足
- **症状**: CUDA out of memory 错误
- **解决方案**:
  - 减小 `batch_size`
  - 增加 `gradient_accumulation_steps`
  - 启用梯度检查点

#### 奖励函数错误
- **症状**: 奖励计算失败或返回异常值
- **解决方案**:
  - 检查奖励模型环境配置
  - 验证 API 服务是否正常运行
  - 确认输入数据格式正确

### 2. 性能调优建议

#### 精度选择
- **fp16**: 更高精度，更小的对数概率误差（推荐用于 SD3.5）
- **bf16**: 更大数值范围，适用于大型模型（FLUX、Wan 必需）

#### 批次大小优化
```python
# 经验配置公式
group_number = 48
group_size = 24
config.sample.train_batch_size * num_gpu / config.sample.num_image_per_prompt * config.sample.num_batches_per_epoch = 48
config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
```

---

## 📈 实验结果与性能

### 1. Flow-GRPO vs Flow-GRPO-Fast

基于 PickScore 评估的对比实验：

- **Flow-GRPO**: 完整10步训练，基线性能
- **Flow-GRPO-Fast (2步)**: 性能相当，训练速度提升5倍
- **Flow-GRPO-Fast (1步)**: 性能略降，训练速度提升10倍

### 2. 不同任务性能

#### GenEval 任务
- CPS 采样显著提升性能
- 无 CFG 训练有效进行 CFG 蒸馏
- Flow-GRPO-Fast 在 GenEval 上表现优异

#### OCR 文本渲染
- 文本识别准确率大幅提升
- 字符级和单词级对齐改善
- 支持复杂字体和布局

#### PickScore 人类偏好
- 生成图像更符合人类审美
- 提示词跟随能力增强
- 整体质量显著改善

### 3. 计算效率

#### 训练速度对比
- **Flow-GRPO**: 完整训练，基准速度
- **Flow-GRPO-Fast**: 5-10倍加速
- **CPS 采样**: 额外20-30%速度提升

#### 资源需求
- **单节点**: 4-8 GPU 配置
- **多节点**: 支持跨节点并行
- **显存需求**: 根据模型和批次大小调整

---

## 🤝 社区与贡献

### 1. 项目致谢

本项目基于以下优秀开源项目：
- [ddpo-pytorch](https://github.com/kvablack/ddpo-pytorch): 在线强化学习训练框架
- [diffusers](https://github.com/huggingface/diffusers): 先进的扩散模型库
- 特别感谢 Kevin Black 对 ddpo-pytorch 项目的贡献

### 2. 引用格式

如果 Flow-GRPO 对您的研究有帮助，请引用以下论文：

```bibtex
@article{liu2025flow,
  title={Flow-grpo: Training flow matching models via online rl},
  author={Liu, Jie and Liu, Gongye and Liang, Jiajun and Li, Yangguang and Liu, Jiaheng and Wang, Xintao and Wan, Pengfei and Zhang, Di and Ouyang, Wanli},
  journal={arXiv preprint arXiv:2505.05470},
  year={2025}
}
```

如果使用了 Flow-DPO 相关工作，请引用：

```bibtex
@article{liu2025improving,
  title={Improving video generation with human feedback},
  author={Liu, Jie and Liu, Gongye and Liang, Jiajun and Yuan, Ziyang and Liu, Xiaokun and Zheng, Mingwu and Wu, Xiele and Wang, Qiulin and Qin, Wenyu and Xia, Menghan and others},
  journal={arXiv preprint arXiv:2501.13918},
  year={2025}
}
```

### 3. 贡献指南

欢迎社区贡献！参与方式包括：

- **报告问题**: 在 GitHub Issues 中报告 bug 或提出改进建议
- **代码贡献**: 提交 Pull Request 添加新功能或修复问题
- **模型支持**: 为新的生成模型添加支持
- **奖励函数**: 贡献新的评估奖励函数
- **文档改进**: 完善文档和教程

---

## 📄 许可证

本项目采用开源许可证，具体条款请参见 [LICENSE](LICENSE) 文件。

---

## 🔗 相关链接

- **项目主页**: [GitHub Repository](https://github.com/yifan123/flow_grpo)
- **论文链接**: [ArXiv](https://arxiv.org/abs/2505.05470)
- **项目页面**: [Visualization](https://gongyeliu.github.io/Flow-GRPO/)
- **在线演示**: [HuggingFace Spaces](https://huggingface.co/spaces/jieliu/SD3.5-M-Flow-GRPO)
- **模型下载**: [HuggingFace Models](https://huggingface.co/collections/jieliu/sd35m-flowgrpo-68298ec27a27af64b0654120)
- **奖励服务器**: [reward-server](https://github.com/yifan123/reward-server)

---

*Flow-GRPO 项目代表了在文本到图像生成领域的重要进展，通过创新的强化学习方法显著提升了生成质量和训练效率。欢迎加入我们的社区，共同推进 AIGC 技术的发展！*
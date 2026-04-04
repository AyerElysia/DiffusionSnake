# Flow-GRPO: 基于在线强化学习的流匹配模型训练框架

[![ArXiv](https://img.shields.io/badge/ArXiv-red?logo=arxiv)](https://arxiv.org/abs/2505.05470)
[![可视化页面](https://img.shields.io/badge/可视化-green?logo=github)](https://gongyeliu.github.io/Flow-GRPO/)
[![代码仓库](https://img.shields.io/badge/代码-9E95B7?logo=github)](https://github.com/yifan123/flow_grpo)
[![模型下载](https://img.shields.io/badge/模型-blue?logo=huggingface)](https://huggingface.co/collections/jieliu/sd35m-flowgrpo-68298ec27a27af64b0654120)
[![在线演示](https://img.shields.io/badge/演示-blue?logo=huggingface)](https://huggingface.co/spaces/jieliu/SD3.5-M-Flow-GRPO)

## 📋 项目简介

Flow-GRPO是一个先进的在线强化学习框架，专门用于训练流匹配模型（Flow Matching Models）。该项目结合了最新的强化学习技术和扩散模型理论，为文本到图像生成、文本到视频生成等任务提供了高效的训练解决方案。

### 🎯 核心特性

- **Flow-GRPO算法**: 基于在线强化学习的流匹配模型训练方法
- **Flow-GRPO-Fast**: 加速变体，仅需训练1-2个去噪步骤
- **多模型支持**: 支持SD3.5、FLUX、Qwen-Image、Wan2.1等主流模型
- **多奖励函数**: 集成PickScore、OCR、GenEval、ImageReward等多种奖励模型
- **分布式训练**: 支持单节点和多节点分布式训练
- **高效采样**: 采用CPS（系数保持采样）和窗口机制优化训练效率

## 🏗️ 项目架构

```
flow_grpo/
├── flow_grpo/                 # 核心代码库
│   ├── diffusers_patch/      # 扩散模型补丁和管道
│   ├── rewards/              # 奖励函数实现
│   ├── assets/               # 资源文件
│   ├── aesthetic_scorer.py   # 美学评分器
│   ├── clip_scorer.py        # CLIP评分器
│   ├── pickscore_scorer.py   # PickScore评分器
│   ├── ocr.py                # OCR相关功能
│   └── ema.py                # 指数移动平均
├── configs/                  # 配置文件
│   ├── base.py              # 基础配置
│   ├── grpo.py              # GRPO算法配置
│   ├── dpo.py               # DPO算法配置
│   └── sft.py               # SFT算法配置
├── scripts/                  # 训练脚本
│   ├── train_*.py           # 各模型训练脚本
│   ├── single_node/         # 单节点训练脚本
│   ├── multi_node/          # 多节点训练脚本
│   └── demo/                # 演示脚本
└── dataset/                  # 数据集目录
    ├── ocr/                 # OCR数据集
    ├── geneval/             # GenEval数据集
    ├── pickscore/           # PickScore数据集
    └── counting_edit/       # 计数编辑数据集
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆仓库
git clone https://github.com/yifan123/flow_grpo.git
cd flow_grpo

# 创建conda环境
conda create -n flow_grpo python=3.10.16
conda activate flow_grpo

# 安装依赖
pip install -e .
```

### 2. 模型下载

为了避免多GPU训练时的重复下载和存储浪费，请提前下载所需模型：

**基础模型**
- **SD3.5**: `stabilityai/stable-diffusion-3.5-medium`
- **FLUX**: `black-forest-labs/FLUX.1-dev`
- **Qwen-Image**: `Qwen/Qwen2-VL-7B-Instruct`

**奖励模型**
- **PickScore**: `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`, `yuvalkirstain/PickScore_v1`
- **CLIPScore**: `openai/clip-vit-large-patch14`
- **Aesthetic Score**: `openai/clip-vit-large-patch14`

### 3. 奖励模型配置

由于不同奖励模型可能依赖不同版本，建议为每个奖励模型创建独立环境：

#### OCR奖励
```bash
pip install paddlepaddle-gpu==2.6.2
pip install paddleocr==2.9.1
pip install python-Levenshtein

# 预下载模型
python -c "from paddleocr import PaddleOCR; ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False, show_log=False)"
```

#### ImageReward
```bash
pip install image-reward
pip install git+https://github.com/openai/CLIP.git
```

#### UnifiedReward
```bash
conda create -n sglang python=3.10.16
conda activate sglang
pip install "sglang[all]"

# 启动服务
python -m sglang.launch_server --model-path CodeGoat24/UnifiedReward-7b-v1.5 --api-key flowgrpo --port 17140 --chat-template chatml-llava --enable-p2p-check --mem-fraction-static 0.85
```

### 4. 开始训练

#### 单节点训练
```bash
# SD3模型训练
bash scripts/single_node/grpo.sh

# FLUX模型训练
bash scripts/single_node/grpo_flux.sh

# Qwen-Image训练
bash scripts/single_node/grpo_qwenimage.sh
```

#### 多节点训练

**SD3多节点训练:**
```bash
# 主节点
bash scripts/multi_node/sd3/main.sh
# 其他节点
bash scripts/multi_node/sd3/main1.sh
bash scripts/multi_node/sd3/main2.sh
bash scripts/multi_node/sd3/main3.sh
```

**FLUX多节点训练:**
```bash
# 主节点
bash scripts/multi_node/flux/main.sh
# 其他节点
bash scripts/multi_node/flux/main1.sh
bash scripts/multi_node/flux/main2.sh
bash scripts/multi_node/flux/main3.sh
```

## 🎯 支持的任务和模型

### 支持的生成模型
- **Stable Diffusion 3.5 Medium**: 高质量文本到图像生成
- **FLUX.1-dev**: 先进的流匹配图像生成模型
- **FLUX.1-Kontext-dev**: 支持图像编辑的变体
- **Qwen-Image**: 多模态视觉语言模型
- **Qwen-Image-Edit**: 图像编辑专用模型
- **Wan2.1**: 文本到视频生成模型

### 支持的奖励函数
| 奖励模型 | 用途 | 特点 |
|---------|------|------|
| **PickScore** | 通用图像质量评估 | 基于人类偏好训练 |
| **OCR** | 文本渲染质量 | 光学字符识别准确率 |
| **GenEval** | 复杂组合提示评估 | 对象计数和空间关系 |
| **ImageReward** | 文本图像对齐 | 综合质量评估 |
| **Aesthetic** | 美学评分 | 基于CLIP的美学预测 |
| **JPEG_Compressibility** | 图像压缩质量 | 代理质量指标 |
| **UnifiedReward** | 统一奖励模型 | 多模态理解领先水平 |

### 支持的训练算法
- **Flow-GRPO**: 核心强化学习算法
- **Flow-GRPO-Fast**: 加速训练变体
- **Flow-DPO**: 直接偏好优化
- **Flow-SFT**: 监督微调

## ⚙️ 核心算法特性

### Flow-GRPO-Fast加速机制

Flow-GRPO-Fast通过以下机制显著提升训练效率：

1. **窗口机制**: 仅在1-2个去噪步骤上进行训练
2. **确定轨迹**: 使用ODE采样生成确定轨迹
3. **随机注入**: 在中间步骤随机注入噪声切换到SDE
4. **局部随机性**: 将随机性限制在少数步骤

### CPS（系数保持采样）

- **噪声级别**: 推荐`noise_level = 0.8`
- **无需调参**: 适用于不同模型和步数
- **质量提升**: 在GenEval任务上表现显著改善

### 无CFG训练

- **CFG蒸馏**: RL过程实现有效的分类器自由引导蒸馏
- **训练效率**: 移除CFG显著提升训练速度
- **推理优化**: 支持无CFG推理，降低计算成本

## 📊 性能表现

### 训练效率对比

Flow-GRPO-Fast相比原版Flow-GRPO：
- **训练速度**: 提升5-10倍
- **内存占用**: 显著降低
- **奖励性能**: 在PickScore上达到相当水平

### 质量评估

在不同任务上的表现：
- **GenEval**: 复杂组合提示理解
- **OCR**: 文本渲染准确性
- **PickScore**: 人类偏好对齐

## 🔧 高级配置

### 多奖励训练

支持多奖励函数加权组合：

```python
reward_config = {
    "pickscore": 0.5,
    "ocr": 0.2,
    "aesthetic": 0.3
}
```

### 超参数调优

关键超参数建议：
- **组大小**: `group_number=48`, `group_size=24`
- **梯度累积**: `gradient_accumulation_steps = num_batches_per_epoch // 2`
- **剪辑范围**: Fast版本使用较小的`clip_range`

### 分布式训练配置

**FSDP配置**:
```yaml
# scripts/accelerate_configs/deepspeed_zero2.yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_forward_prefetch: FALSE
  fsdp_offload_params: false
  fsdp_sharding_strategy: 1
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: []
```

## 🛠️ 扩展新模型

要为新模型添加支持，请遵循以下步骤：

### 1. 创建模型适配文件
```python
# flow_grpo/diffusers_patch/your_model_pipeline_with_logprob.py
# 基于diffusers中的原始pipeline进行修改
```

### 2. 创建训练脚本
```python
# scripts/train_your_model.py
# 基于现有的train_sd3.py进行适配
```

### 3. 验证SDE采样
```bash
# 使用demo脚本验证SDE实现
python scripts/demo/your_model_sde_demo.py
```

### 4. 配置参数
```python
# config/grpo.py中添加配置
def your_model_config():
    config = base.get_config()
    config.pretrained.model = "your-model-path"
    # 其他配置...
    return config
```

## ❓ 常见问题

### Q: 使用fp16还是bf16？
A: 优先使用fp16，精度更高。对于FLUX和Wan模型，由于fp16推理无法生成有效图像，必须使用bf16。

### Q: Flow-GRPO-Fast训练崩溃怎么办？
A: 减小`clip_range`参数，避免训练不稳定。

### Q: 多GPU训练输出不一致？
A: 确保训练和数据收集使用相同的batch size，检查是否有torch.compile等差异。

### Q: 如何验证on-policy一致性？
A: 设置`num_batches_per_epoch=1`和`gradient_accumulation_steps=1`，ratio应保持为1。

## 📚 相关资源

### 论文引用
```bibtex
@article{liu2025flow,
  title={Flow-grpo: Training flow matching models via online rl},
  author={Liu, Jie and Liu, Gongye and Liang, Jiajun and Li, Yangguang and Liu, Jiaheng and Wang, Xintao and Wan, Pengfei and Zhang, Di and Ouyang, Wanli},
  journal={arXiv preprint arXiv:2505.05470},
  year={2025}
}
```

### 相关项目
- [ddpo-pytorch](https://github.com/kvablack/ddpo-pytorch): 扩散模型强化学习训练框架
- [diffusers](https://github.com/huggingface/diffusers): HuggingFace扩散模型库
- [MixGRPO](https://www.arxiv.org/abs/2507.21802): 混合GRPO算法

### 可视化演示
- [项目主页](https://gongyeliu.github.io/Flow-GRPO/): 训练过程可视化
- [在线演示](https://huggingface.co/spaces/jieliu/SD3.5-M-Flow-GRPO): HuggingFace Spaces演示

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出改进建议！

### 开发环境设置
```bash
# 安装开发依赖
pip install -e ".[dev]"

# 代码格式化
black .

# 运行测试
pytest
```

### 提交流程
1. Fork仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 📄 许可证

本项目基于Apache 2.0许可证开源。详见[LICENSE](../LICENSE)文件。

## 🙏 致谢

- 感谢[ddpo-pytorch](https://github.com/kvablack/ddpo-pytorch)项目提供的优秀框架基础
- 感谢[HuggingFace Diffusers](https://github.com/huggingface/diffusers)社区的支持
- 特别感谢Kevin Black对ddpo-pytorch项目的杰出贡献

---

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交[GitHub Issue](https://github.com/yifan123/flow_grpo/issues)
- 发送邮件至项目维护者

**Flow-GRPO** - 让流匹配模型训练更高效、更智能！ 🚀
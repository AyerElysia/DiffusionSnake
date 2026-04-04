# DiffusionSnake

基于扩散模型与DeepSnake的医学图像分割框架，集成YOLO目标检测与GRPO强化学习训练。

## 项目简介

DiffusionSnake是一个用于脊柱MRI图像分割的深度学习框架，主要特点：

- **DeepSnake轮廓进化**: 基于极值点的主动轮廓模型，实现精确的目标边界分割
- **YOLO检测集成**: 使用YOLOv8进行初始检测框定位，为Snake提供初始轮廓
- **扩散模型进化**: 集成扩散模型（Diffusion Model）进行轮廓进化优化
- **GRPO强化学习**: 支持Group Relative Policy Optimization训练策略
- **多类别分割**: 支持脊柱52类结构分割（椎体、椎间盘、脊髓等）

## 项目结构

```
DiffusionSnake-12-30/
├── configs/                    # 配置文件
│   └── grpo_snake.yaml        # GRPO训练配置
├── lib/                       # 核心库
│   ├── networks/              # 网络模型定义
│   │   ├── snake/            # Snake网络
│   │   ├── YOLOV8/           # YOLOv8检测网络
│   │   └── vision_mamba2/    # Vision Mamba模块
│   ├── datasets/              # 数据集处理
│   │   ├── sbd/              # SBD数据集
│   │   ├── coco/             # COCO数据集
│   │   └── voc_test/         # VOC测试集
│   ├── train/                 # 训练模块
│   ├── evaluators/            # 评估器
│   ├── utils/                 # 工具函数
│   └── csrc/                  # C++扩展
├── flow_grpo-main/            # GRPO训练框架
│   ├── config/               # GRPO配置
│   ├── dataset/              # 训练数据集
│   ├── flow_grpo/            # GRPO核心实现
│   └── scripts/              # 训练脚本
├── assets/                    # 资源文件
├── run.py                     # 主运行脚本
├── train_net.py               # 训练脚本
├── grpo_train.py              # GRPO训练脚本
└── demo.py                    # 演示脚本
```

## 环境配置

### 依赖安装

```bash
# PyTorch (请根据CUDA版本选择)
pip install torch torchvision

# 其他依赖
pip install opencv-python numpy pyyaml tqdm
pip install ultralytics  # YOLOv8
```

### 编译C++扩展

```bash
cd lib/csrc/extreme_utils/
python setup.py build_ext --inplace
```

## 快速开始

### 训练

使用传统训练模式：

```bash
python train_net.py --cfg_file configs/grpo_snake.yaml
```

使用GRPO强化学习训练：

```bash
python grpo_train.py
```

### 测试/评估

```bash
python run.py --type evaluate --cfg_file configs/grpo_snake.yaml
```

### 可视化

```bash
python run.py --type visualize --cfg_file configs/grpo_snake.yaml
```

### 演示

```bash
python demo.py
```

## 配置说明

主要配置文件`configs/grpo_snake.yaml`关键参数：

```yaml
model: 'sbd'                    # 模型类型
network: 'ro_34'                # 网络架构
task: 'snake'                   # 任务类型

# GRPO配置
use_grpo: true                  # 启用GRPO
grpo_first_contour_only: true   # 仅对首个轮廓应用GRPO
grpo_window_size: 1             # GRPO窗口大小
grpo_window_range: [15, 20]     # 窗口范围

# 扩散模型配置
use_diffusion_evolution: true   # 启用扩散进化
diffusion_timesteps: 1000       # 扩散时间步数
use_ddim_inference: true        # 使用DDIM推理

# 训练配置
train:
  optim: 'adam'
  lr: 1e-7
  batch_size: 64
  epoch: 500
  data_path: '/path/to/data'

# YOLO检测配置
det_conf_thresh: 0.01           # 置信度阈值
det_iou_thresh: 0.45            # NMS IoU阈值
```

## 支持的分割类别

项目支持脊柱MRI图像的52类结构分割：

| 类别ID | 结构名称 |
|--------|----------|
| 1-25 | 椎体 (S1, L5-L1, T12-T1, C7-C1) |
| 26-48 | 椎间盘 (S1/L5 - C3/C2) |
| 50 | 脊髓 |
| 51 | 附件结构 |

详细类别描述参见配置文件中的`class_descriptions`字段。

## 核心模块

### 1. Snake网络

主动轮廓模型，通过迭代更新轮廓点坐标实现精确分割。

### 2. YOLO检测器

提供初始目标检测框，为Snake提供初始轮廓。

### 3. 扩散进化模块

使用扩散模型优化轮廓进化过程，提升分割精度。

### 4. GRPO训练器

基于强化学习的训练策略，优化轮廓进化决策。

## 运行模式

`run.py`支持多种运行模式：

- `dataset`: 数据集测试
- `network`: 网络性能测试
- `evaluate`: 评估模式
- `visualize`: 可视化模式
- `sbd`: SBD数据转换
- `demo`: 演示模式
- `test_medical`: 医学图像测试

```bash
python run.py --type <mode> --cfg_file <config_path>
```

## 模型权重

预训练模型应放置在配置文件指定的`model_dir`路径下。

## 可视化结果

训练过程中的可视化结果保存在`visual/`目录下，包括：
- 检测框（绿色）
- 预测轮廓（红色）
- 真值轮廓（蓝色）
- 初始轮廓（黄色）

## 参考文献

- DeepSnake: [Deep Snake for Real-Time Instance Segmentation](https://arxiv.org/abs/2001.01629)
- YOLOv8: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Diffusion Models: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

## License

本项目仅供学术研究使用。

## 致谢

感谢DeepSnake、YOLOv8等开源项目的贡献。

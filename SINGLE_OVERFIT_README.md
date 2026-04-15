# 单样本过拟合训练配置说明

## 配置文件列表

### V2系列 (DiT V2架构)
- `btcv_diffusion_dit_v2_single_overfit.yaml` - V2.0, GPU 5
- `btcv_diffusion_dit_v2_1_single_overfit.yaml` - V2.1, GPU 5
- `btcv_diffusion_dit_v2_2_single_overfit.yaml` - V2.2, GPU 5
- `btcv_diffusion_dit_v2_3_single_overfit_gpu6.yaml` - V2.3, GPU 6

### V3系列 (DiT V3架构)
- `btcv_diffusion_dit_v3_single_overfit_gpu7.yaml` - V3.0, GPU 7 (已运行)
- `btcv_diffusion_dit_v3_1_single_overfit.yaml` - V3.1, GPU 6
- `btcv_diffusion_dit_v3_2_single_overfit.yaml` - V3.2, GPU 6
- `btcv_diffusion_dit_v3_4_single_overfit.yaml` - V3.4, GPU 7, 1W epoch

## GPU分配

- **GPU 5**: V2.0, V2.1, V2.2 (串行运行)
- **GPU 6**: V2.3, V3.1, V3.2 (串行运行)
- **GPU 7**: V3.0 (已在运行), V3.4（单样本）

## 启动脚本

### 一键启动所有训练
```bash
bash run_all_single_overfit.sh
```

### 分GPU启动
```bash
# 只启动GPU 5上的任务
bash run_gpu5_overfit.sh

# 续跑GPU 5上未完成的任务
bash run_gpu5_resume_overfit.sh

# 只启动GPU 6上的任务
bash run_gpu6_overfit.sh

# 只启动GPU 7上的任务
bash run_gpu7_overfit.sh
```

### 单独启动某个版本
```bash
cd /mnt/sdb1/leijh/DiffusionSnake/DiffusionSnake-12-30
conda activate snake1

# 例如启动V2.1
export CFG_FILE=configs/btcv_diffusion_dit_v2_1_single_overfit.yaml
CUDA_VISIBLE_DEVICES=5 python diffusion_train.py
```

## 日志文件

所有训练日志保存在 `logs/` 目录:
- `v2_0_single_overfit_gpu5.log`
- `v2_1_single_overfit_gpu5.log`
- `v2_2_single_overfit_gpu5.log`
- `v2_3_single_overfit_gpu6.log`
- `v3_1_single_overfit_gpu6.log`
- `v3_2_single_overfit_gpu6.log`
- `v3_4_single_overfit_gpu7.log`

## 模型输出目录

训练的checkpoint和输出保存在:
- `data/model/btcv_diffusion_dit_v2_single_overfit_gpu5/`
- `data/model/btcv_diffusion_dit_v2_1_single_overfit_gpu5/`
- `data/model/btcv_diffusion_dit_v2_2_single_overfit_gpu5/`
- `data/model/btcv_diffusion_dit_v2_3_single_overfit_gpu6/`
- `data/model/btcv_diffusion_dit_v3_overfit_single_gpu7/`
- `data/model/btcv_diffusion_dit_v3_1_single_overfit_gpu6/`
- `data/model/btcv_diffusion_dit_v3_2_single_overfit_gpu6/`
- `data/model/btcv_diffusion_dit_v3_4_single_overfit_gpu7/`

## 训练参数

所有单样本训练使用相同的基础参数:
- 数据集: BtcvMini (单样本)
- 数据路径: `/mnt/sdb1/leijh/DiffusionSnake/Datasets/BTCV/btcv_png_single_overfit`
- Batch size: 1
- Learning rate: 5e-5
- Epochs: 10000
- Save interval: 100 epochs
- Optimizer: AdamW
- Warmup steps: 100

## 监控训练

```bash
# 查看GPU使用情况
nvidia-smi

# 实时查看某个版本的训练日志
tail -f logs/v2_1_single_overfit_gpu5.log

# 查看所有训练进程
ps aux | grep diffusion_train.py
```

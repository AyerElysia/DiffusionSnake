# DiffusionSnake V3/V3.1 Bug Fix Report

## 概述

V3 将初始轮廓从矩形改为八边形后，推理结果完全混乱（轮廓爆炸到图像外）。
经过系统排查，发现 **4 个关键 Bug**，全部已修复并验证。

---

## Bug 1（最严重）：Checkpoint 权重键名不匹配

### 问题
所有 Denoiser（V1–V3.1）在重构过程中将：
```python
# 旧代码
self.time_emb_1 = nn.Linear(dim, dim)
self.time_emb_3 = nn.SiLU()
```
改为：
```python
# 新代码
self.time_emb_net = nn.Sequential(
    nn.Linear(dim, dim * 4),  # [0]
    nn.Linear(dim, dim),      # [1]  (实际结构)
    ...
    nn.SiLU(),                # [3]
)
```

但已保存的 checkpoint 中，权重键名仍为 `time_emb_1.weight`、`time_emb_3.weight`。
新代码期望的是 `time_emb_net.1.weight`、`time_emb_net.3.weight`。

### 影响
- 时间步嵌入权重 **完全不加载**（被 `strict=False` 静默忽略）
- 扩散模型 **没有时间步条件**，DDIM 去噪等于在噪声上做无条件变换
- 结果：输出是随机噪声级别的位移

### 修复
在 `lib/networks/diffusion/pretrain_evolution.py` 中新增 `remap_legacy_state_dict()` 函数：
```python
def remap_legacy_state_dict(state_dict):
    """将旧版 time_emb_1/time_emb_3 映射到 time_emb_net.1/time_emb_net.3"""
    new_sd = {}
    pat = re.compile(r'(\.?)time_emb_(\d)(\..*)')
    for k, v in state_dict.items():
        new_k = pat.sub(r'\1time_emb_net.\2\3', k)
        new_sd[new_k] = v
    return new_sd
```
在所有 checkpoint 加载点（训练 resume、所有推理脚本）调用此函数。

---

## Bug 2：推理脚本双重反归一化

### 问题
`pretrain_evolution.py` 的 `sample_disp()` 方法 **内部已调用** `denormalize_disp()`：
```python
# sample_disp() 末尾 (line ~304)
return self.denormalize_disp(pred_x0)  # 已经反归一化了！
```
但 `infer_v3_refinement_sync.py` 和 `scripts/infer_v3_refinement.py` 又调用了一次：
```python
disp = evo.denormalize_disp(disp)  # 重复！变换被应用两次
```

### 影响
归一化公式 `(x+1)*0.5*(max-min)+min` 被应用两次：
- 正确位移 `±30px` → 第二次反归一化后变成 `±2000px` → 轮廓远超图像边界

### 修复
删除推理脚本中多余的 `denormalize_disp()` 调用。

---

## Bug 3：V3.1 初始轮廓不匹配

### 问题
`lib/utils/snake/snake_config.py` 第 29 行：
```python
# 旧代码（只检查 use_dit_v3）
if getattr(cfg, 'use_dit_v3', False):
    snake_config.init = 'octagon'
```
V3.1 的配置文件设置的是 `use_dit_v3_1: true`（而非 `use_dit_v3`），所以条件不满足。

### 影响
- 训练时：`prepare_evolution()` 总是使用八边形（正确）
- 推理时：`snake_config.init` 未被设置为 `octagon`，使用了默认的四边形
- 训练用八边形 12 点，推理用四边形 4 点 → 形状完全不匹配

### 修复
```python
if getattr(cfg, 'use_dit_v3', False) or getattr(cfg, 'use_dit_v3_1', False):
    snake_config.init = 'octagon'
```

---

## Bug 4：推理脚本 `net.` 前缀处理错误

### 问题
训练保存 checkpoint 时，`trainer.network` 是 `NetworkWrapper`，其内部模型是 `self.net`。
所以 checkpoint 键名带有 `net.` 前缀（如 `net.gcn.denoiser.xxx`）。

推理脚本错误地 **剥离了前缀**，然后尝试加载到 `NetworkWrapper`：
```python
# 错误
new_sd = {k.replace('net.', ''): v for k, v in sd.items()}
model.load_state_dict(new_sd)  # model 是 NetworkWrapper，期望 net.xxx
```
导致 500+ missing keys，500+ unexpected keys。

### 修复
保留 `net.` 前缀，直接加载到 `NetworkWrapper`：
```python
sd = remap_legacy_state_dict(sd)
wrapper = trainer.network.module if hasattr(trainer.network, 'module') else trainer.network
wrapper.load_state_dict(sd, strict=False)
```

---

## 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `lib/networks/diffusion/pretrain_evolution.py` | 新增 `remap_legacy_state_dict()` |
| `lib/utils/snake/snake_config.py` | 添加 `use_dit_v3_1` 八边形条件 |
| `diffusion_train.py` | 训练 resume 时调用 remap |
| `infer_v3_refinement_sync.py` | 修复加载 + 去除双重反归一化 |
| `scripts/infer_v3_refinement.py` | 同上 |
| `scripts/infer_v3_final.py` | 修复加载 |
| `scripts/infer_v3_2_refinement.py` | 修复加载 |

---

## 验证结果

### 修复前
```
Loading checkpoint...  Missing: 502, Unexpected: 572
Disp Stats: mean_abs = 5745  ← 完全随机
```

### 修复后
```
Loading checkpoint...  Missing: 0, Unexpected: 0  ✅
Disp Stats: mean_abs = 86    ← 合理范围
```

### 训练状态

| 版本 | 已训练步数 | 目标步数 | 完成度 | diff_loss |
|------|-----------|---------|--------|-----------|
| V2.1 | 12,000 | 23,000 | 52% | 0.0015 |
| V3.0 | 2,961 | 23,000 | 12.8% | 0.007 |
| V3.1 | 1,513 | 23,000 | 6.6% | 0.013 |

V3/V3.1 模型严重欠训练（仅 ~126/1000 epochs），需要继续训练到至少 500 epochs 才能公平评估。

---

## 根因总结

V3 推理混乱的根本原因是 **Bug 1 + Bug 4 叠加**：

1. 权重键名不匹配 → 时间步嵌入完全丢失 → 模型无时间步条件
2. `net.` 前缀剥离 → 所有权重都加载失败 → 模型等于随机初始化

这两个 Bug 加在一起，使模型在推理时等于一个完全未训练的随机网络。

Bug 2（双重反归一化）和 Bug 3（形状不匹配）进一步放大了错误。

---

## 建议

1. **立即恢复 V3.1 训练**：当前 checkpoint 已经正确保存了 126 epochs 的权重（训练时代码正常工作），现在推理加载也修复了。继续训练到 1000 epochs。
2. **训练完成后重新评估**：V3 八边形初始化理论上优于矩形（更接近目标形状），应在充分训练后再比较 V2 vs V3。
3. **未来避免类似问题**：重构网络层时，添加向后兼容的 checkpoint 加载逻辑或同时更新已有 checkpoint。

# V3.5 傅里叶空间扩散 — 最终结果 (2026-04-19)

## 核心成果：毛边问题彻底解决 ✅

V3.5 通过在傅里叶空间进行扩散（而非空间域128个点），从根本上消除了V3.0的锯齿/毛边问题。

### 数值对比

| 版本 | Epoch | Avg Smoothness | 相对V3.0 | 状态 |
|------|-------|---------------|----------|------|
| V3.0 基线 | 10000 | 6486 | 1× | 严重毛边 |
| V3.0 + K=8后处理 | 10000 | 484 | 13× | 有改善但信息丢失 |
| V3.3a (Conv1d) | 9000 | 更差 | <1× | 失败 |
| V3.3b (smooth loss) | 7600 | 爆炸 | N/A | 灾难性失败 |
| **V3.5 ep1000** | 1000 | 52 | 125× | 收敛中 |
| **V3.5 ep2500** | 2500 | 21 | 309× | 良好 |
| **V3.5 ep3500** | 3500 | **12.2** | **532×** | 🏆 优秀 |

### 关键指标 (epoch 3500)
- 7个器官轮廓平均平滑度: **12.2**
- 最佳轮廓 (C5): smoothness=6.1
- 最差轮廓 (C6): smoothness=24.9
- 所有轮廓均平滑、无自相交、形状合理

---

## 关键Bug修复：傅里叶归一化

### 问题
原始实现中，`normalize_disp_fourier()` 使用空间域位移的统计量来归一化傅里叶系数。

```
空间域统计量: range=103.7px → 归一化除以 6639
实际傅里叶系数: std=27.05
归一化后: std = 27.05/6639 = 0.004  ← 信号几乎为零！
```

扩散模型的噪声是 N(0,1)，但GT信号只有std=0.004，相当于信号被噪声淹没250倍。
模型学到的是"忽略条件，直接输出零"，推理时产生乱线。

### 修复
改用**傅里叶域统计量**直接做标准化：

```python
# 预计算 Fourier-domain 的 mean/std
fourier_global_mean = 0.19
fourier_global_std = 27.05

# 归一化
normalized = (fourier_coeffs - mean) / std  # → std=1.0 ✅
# 反归一化
fourier_coeffs = normalized * std + mean
```

### 修改文件
1. `lib/networks/diffusion/pretrain_evolution.py`:
   - `_load_fourier_stats()`: 加载预计算的Fourier统计量
   - `normalize_disp_fourier()`: 使用 `(x - mean) / std`
   - `denormalize_disp_fourier()`: 使用 `x * std + mean`
2. `lib/config/config.py`: 新增 `fourier_disp_stats` 配置项
3. `data/stats/btcv_fourier_stats_K16_single_overfit.json`: 统计量文件

---

## V3.5 架构回顾

### 核心思想
不预测128个点的xy位移，而是预测K=16个傅里叶系数（每个系数有实部+虚部×2个坐标=4维）。
IFFT保证输出只包含低频分量 → **数学上不可能产生毛边**。

### Denoiser: DiTDenoiserV3_5 (10.9M params)
```
输入:
  - 128点CNN特征 (来自YOLOv8 backbone)
  - K×4 带噪傅里叶系数
  - 时间步 t

处理:
  1. FourierPointBridge: 128点特征 → K个频率特征 (Cross-Attention)
  2. FourierCoeffEmbedding: 傅里叶系数嵌入
  3. 多层 DiTBlockV3: Self-Attention + FFN + AdaLN
  4. FourierFinalLayer: 零初始化输出 K×4 系数

输出: K×4 去噪后的傅里叶系数 → IFFT → 128×2 位移
```

---

## 训练状态

- **PID**: 905636, GPU 0 (~1.7GB)
- **当前**: epoch 3524/10000
- **Loss趋势**: 0.96→0.058(ep500)→0.024(ep1000)→0.004(ep2500)→0.007(ep3500)
- **预计完成**: ~2小时后

---

## 下一步

### 立即
1. 等待10k epoch训练完成，做最终推理对比
2. 与V3.0做side-by-side可视化对比

### 短期
3. **计算全数据集BTCV的傅里叶统计量** → 为全数据集训练做准备
4. 全数据集V3.5训练
5. 与V3.6 (Flow Matching) 结果对比

### 创新总结
- **V3.5的核心创新**: 将扩散过程从空间域转移到傅里叶域，通过频率截断实现结构性平滑保证
- 这与现有的后处理方法本质不同：不是"事后补救"，而是"先天免疫"
- 可与V3.4多步迭代组合使用

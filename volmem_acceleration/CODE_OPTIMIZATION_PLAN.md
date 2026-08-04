# 代码级优化方案

如果配置优化不足以达到2x加速，将执行以下代码改造：

## 优先级1: MoE 专家剪枝 (volmem/models/memflow_dit.py)

```python
# 修改1: 减少活跃专家数
class AdaptiveMoE(nn.Module):
    def __init__(self, num_experts=8, num_active=4):
        super().__init__()
        self.num_experts = num_experts
        self.num_active = num_active
        self.experts = nn.ModuleList([Expert() for _ in range(num_experts)])
        
    def forward(self, x):
        # 只使用前 num_active 个专家
        expert_outputs = []
        for i in range(self.num_active):  # 4 instead of 8
            expert_outputs.append(self.experts[i](x))
        # ...路由和聚合...
```

## 优先级2: 简化路由策略 (train_net.py)

```python
# 从 Top-k=2 改为 Top-k=1
top_k_routing: 1  # 只选最好的专家
enable_balance_loss: false  # 禁用辅助损失
```

## 优先级3: 混合精度 (train_net.py)

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast(dtype=torch.float16):  # MoE层和Memory用float16
        output = model(batch)
        loss = criterion(output)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## 优先级4: 多尺度推理 (新模块)

```python
class HierarchicalInference(nn.Module):
    def forward_coarse_medium_fine(self, batch):
        # Coarse: 间隔3取1 (3x加速)
        coarse_indices = torch.arange(0, len(batch), 3)
        coarse_output = self.model(batch[coarse_indices])
        
        # Medium: 细化
        medium_indices = torch.arange(1, len(batch), 2)
        medium_output = self.refine_medium(batch[medium_indices], coarse_output)
        
        # Fine: 本地细化
        fine_output = self.refine_fine(batch, medium_output)
        
        return fine_output
```

## 优先级5: 自蒸馏 (新模块)

```python
class DistilledMemFlow(nn.Module):
    def forward_multistep(self, batch):
        # 预测未来3步
        predictions = []
        current = batch
        
        for step in range(3):
            pred = self.model(current)
            predictions.append(pred)
            current = pred  # 用预测值作为下一步的输入
        
        return predictions
```


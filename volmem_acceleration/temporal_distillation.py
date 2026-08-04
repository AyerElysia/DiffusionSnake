"""
自蒸馏实现
- 多步预测
- 知识蒸馏损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalDistilledMemFlow(nn.Module):
    """支持多步预测的蒸馏框架"""
    
    def __init__(self, base_model, steps_ahead=2):
        super().__init__()
        self.base_model = base_model
        self.steps_ahead = steps_ahead
        
        # 多步预测头
        feature_dim = 256  # 或根据实际调整
        self.future_heads = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim) 
            for _ in range(steps_ahead)
        ])
    
    def forward_multistep(self, batch, metas, banks):
        """
        多步预测
        返回: [step0, step1, ..., step_k]
        """
        
        # 第一步
        step0_output = self.base_model(batch, metas, banks)
        predictions = [step0_output]
        
        # 未来步
        current = step0_output
        for i in range(self.steps_ahead):
            # 用当前预测作为下一步的特征
            if isinstance(current, dict):
                current_features = current.get('features', current)
            else:
                current_features = current
            
            # 预测未来一步
            future = self.future_heads[i](current_features)
            predictions.append(future)
            current = future
        
        return predictions
    
    def compute_distillation_loss(self, predictions, targets):
        """
        计算蒸馏损失
        
        Loss = task_loss + alpha * kl_divergence
        """
        
        # 任务损失（所有步的预测损失）
        task_loss = 0
        for pred, target in zip(predictions, targets):
            task_loss += F.mse_loss(pred, target)
        
        task_loss = task_loss / len(predictions)
        
        # 知识蒸馏损失 (KL散度 between steps)
        # 假设相邻步之间应该平滑
        kl_loss = 0
        for i in range(len(predictions) - 1):
            # 使相邻预测接近
            kl_loss += F.kl_div(
                F.log_softmax(predictions[i+1], dim=-1),
                F.softmax(predictions[i].detach(), dim=-1),
                reduction='batchmean'
            )
        
        total_loss = task_loss + 0.1 * kl_loss
        return total_loss


"""
Per-Prompt统计跟踪模块

该模块实现了Flow-GRPO算法中的关键组件：per-prompt统计跟踪和优势计算。
这是GRPO算法的核心创新之一，通过为每个文本提示单独计算统计量和优势值，
解决了传统RL方法在扩散模型训练中的数值不稳定问题。

主要功能：
- 为每个prompt单独维护历史奖励统计
- 计算per-prompt优势值，用于策略梯度优化
- 支持多种算法类型（GRPO、RWR、SFT、DPO）
- 提供训练过程统计信息监控
"""

import numpy as np
from collections import deque
import torch


class PerPromptStatTracker:
    """
    Per-Prompt统计跟踪器
    
    这是Flow-GRPO算法的核心组件，负责为每个文本提示单独维护奖励统计信息，
    并计算相应的优势值。相比传统的全局标准化，per-prompt方法能够：
    1. 提高训练稳定性
    2. 减少不同prompt间的干扰
    3. 更准确地评估每个提示的策略改进
    
    Args:
        global_std (bool): 是否使用全局标准差
                          - True: 使用所有奖励的全局标准差
                          - False: 使用每个prompt组内的标准差（推荐）
    """
    def __init__(self, global_std=False):
        self.global_std = global_std          # 标准差计算模式
        self.stats = {}                       # 存储每个prompt的奖励历史 {prompt: [rewards]}
        self.history_prompts = set()          # 存储所有出现过的prompt的hash值

    def update(self, prompts, rewards, type='grpo'):
        """
        更新统计信息并计算优势值
        
        这是整个模块的核心方法，执行以下步骤：
        1. 按prompt分组更新历史奖励统计
        2. 计算每个prompt组内样本的优势值
        3. 支持多种RL算法的优势计算方式
        
        Args:
            prompts (list): 文本提示列表
            rewards (list): 对应的奖励值列表
            type (str): 算法类型，支持以下选项：
                        - 'grpo': 使用标准化优势值 (默认)
                        - 'rwr': 使用原始奖励值作为优势
                        - 'sft': 使用winner-takes-all策略
                        - 'dpo': 使用偏好对比较优势
                        
        Returns:
            np.ndarray: 计算得到的优势值数组，形状与rewards相同
        """
        # 转换为numpy数组以提高计算效率
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        
        # 获取唯一的prompt列表
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards)*0.0  # 初始化优势数组
        
        # === 第一阶段：更新每个prompt的历史奖励统计 ===
        for prompt in unique:
            # 获取当前prompt对应的所有奖励
            prompt_rewards = rewards[prompts == prompt]
            
            # 如果是新prompt，初始化统计列表
            if prompt not in self.stats:
                self.stats[prompt] = []
                
            # 将新奖励添加到历史记录中
            self.stats[prompt].extend(prompt_rewards)
            
            # 记录prompt的hash值，用于统计历史prompt数量
            self.history_prompts.add(hash(prompt))
        
        # === 第二阶段：计算每个prompt的优势值 ===
        for prompt in unique:
            # 将列表转换为numpy数组以便计算统计量
            self.stats[prompt] = np.stack(self.stats[prompt])
            prompt_rewards = rewards[prompts == prompt]  # 当前批次的奖励
            
            # 计算均值和标准差
            mean = np.mean(self.stats[prompt], axis=0, keepdims=True)
            
            if self.global_std:
                # 使用全局标准差（不推荐，会增加不同prompt间的干扰）
                std = np.std(rewards, axis=0, keepdims=True) + 1e-4
            else:
                # 使用prompt组内标准差（推荐，减少组间干扰）
                std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4
            
            # === 根据算法类型计算优势值 ===
            if type == 'grpo':
                # GRPO算法：使用标准化优势值
                # 优势 = (奖励 - 均值) / 标准差
                advantages[prompts == prompt] = (prompt_rewards - mean) / std
                
            elif type == 'rwr':
                # Reward-Weighted Regression: 直接使用原始奖励
                # 注意：这里注释掉的标准化版本可能更适合某些场景
                advantages[prompts == prompt] = prompt_rewards
                # 可选：使用softmax归一化的奖励作为优势
                # advantages[prompts == prompt] = torch.softmax(torch.tensor(prompt_rewards), dim=0).numpy()
                
            elif type == 'sft':
                # Supervised Fine-Tuning: Winner-takes-all策略
                # 最高奖励的样本优势为1，其他为0
                advantages[prompts == prompt] = (torch.tensor(prompt_rewards) == torch.max(torch.tensor(prompt_rewards))).float().numpy()
                
            elif type == 'dpo':
                # Direct Preference Optimization: 偏好对比较
                # 在每个prompt组内，奖励最高的为胜者，最低的为败者
                prompt_advantages = torch.tensor(prompt_rewards)
                
                # 找到最高和最低奖励的索引
                max_idx = torch.argmax(prompt_advantages)
                min_idx = torch.argmin(prompt_advantages)
                
                # 处理所有奖励相同的特殊情况
                if max_idx == min_idx:
                    min_idx = 0  # 强制选择不同的索引
                    max_idx = 1
                    
                # 创建优势数组：胜者为1，败者为-1，其他为0
                result = torch.zeros_like(prompt_advantages).float()
                result[max_idx] = 1.0    # 胜者优势
                result[min_idx] = -1.0   # 败者优势
                advantages[prompts == prompt] = result.numpy()
            
        return advantages

    def get_stats(self):
        """
        获取统计信息
        
        用于监控训练过程的状态，包括：
        - 平均组大小：每个prompt平均有多少个样本
        - 历史prompt数量：总共遇到过的不同prompt数量
        
        Returns:
            tuple: (avg_group_size, history_prompts)
                   - avg_group_size (float): 平均每组样本数
                   - history_prompts (int): 历史prompt总数
        """
        # 计算平均组大小（每个prompt的平均样本数）
        avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats) if self.stats else 0
        # 获取历史prompt总数
        history_prompts = len(self.history_prompts)
        return avg_group_size, history_prompts
    
    def clear(self):
        """
        清空统计信息
        
        通常在每个epoch结束时调用，用于：
        1. 重置统计数据，开始新的统计周期
        2. 防止历史数据过度累积影响当前训练
        3. 保持算法的时效性
        
        注意：history_prompts不会被清空，用于统计总体训练覆盖度
        """
        self.stats = {}  # 清空每个prompt的奖励历史

def main():
    """
    演示和测试函数
    
    展示PerPromptStatTracker的基本用法，包括：
    1. 创建统计跟踪器
    2. 更新奖励统计并计算优势值
    3. 获取训练统计信息
    4. 清空统计数据
    
    这是一个简单的测试用例，用于验证模块功能的正确性。
    """
    # 创建统计跟踪器实例
    tracker = PerPromptStatTracker()
    
    # 测试数据：包含重复prompt的奖励
    prompts = ['a', 'b', 'a', 'c', 'b', 'a']  # 3个'a', 2个'b', 1个'c'
    rewards = [1, 2, 3, 4, 5, 6]
    
    # 更新统计信息并计算优势值
    advantages = tracker.update(prompts, rewards)
    print("Advantages:", advantages)
    
    # 获取统计信息
    avg_group_size, history_prompts = tracker.get_stats()
    print("Average Group Size:", avg_group_size)      # 平均每个prompt有多少样本
    print("History Prompts:", history_prompts)       # 历史上遇到的prompt总数
    
    # 清空统计数据
    tracker.clear()
    print("Stats after clear:", tracker.stats)       # 验证统计信息已被清空


if __name__ == "__main__":
    # 运行演示测试
    main()
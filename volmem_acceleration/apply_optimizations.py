"""
一键应用所有优化
"""

import argparse
import yaml
from pathlib import Path

def apply_optimization(config_path, optimization_type):
    """
    根据优化类型应用改动
    
    optimization_type: 
    - 'moe_pruning': MoE剪枝
    - 'mixed_precision': 混合精度
    - 'hierarchical': 多尺度推理
    - 'distillation': 自蒸馏
    - 'combined': 组合优化
    """
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    if optimization_type == 'moe_pruning':
        # 减少专家数
        config['moe_config'] = {
            'num_experts': 4,
            'top_k': 1,
            'enable_balance_loss': False,
        }
    
    elif optimization_type == 'mixed_precision':
        config['use_mixed_precision'] = True
        config['amp_dtype'] = 'float16'
    
    elif optimization_type == 'combined':
        # 组合所有优化
        config['moe_config'] = {'num_experts': 4, 'top_k': 1}
        config['use_mixed_precision'] = True
        config['chunks_per_step'] = 20
        config['locate_feat_dim'] = 576
        config['snake_feature_dim'] = 128
    
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--optimization', default='combined')
    parser.add_argument('--output', required=True)
    
    args = parser.parse_args()
    
    config = apply_optimization(args.config, args.optimization)
    
    with open(args.output, 'w') as f:
        yaml.dump(config, f)
    
    print(f"✅ 优化配置已保存到: {args.output}")


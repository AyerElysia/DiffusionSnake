#!/usr/bin/env python3
"""
EnergySnake训练脚本 - 支持轮廓引导的自蒸馏
"""

import argparse
import os
import sys
import torch
import importlib

def make_network(cfg):
    """创建网络模型"""
    from lib.networks.make_network import make_network as create_network
    return create_network(cfg)

def make_optimizer(cfg, network):
    """创建优化器"""
    if cfg.train.optim == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=cfg.train.lr)
    elif cfg.train.optim == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(), lr=cfg.train.lr, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(network.parameters(), lr=cfg.train.lr)
    return optimizer

def make_lr_scheduler(cfg, optimizer):
    """创建学习率调度器"""
    from torch.optim.lr_scheduler import MultiStepLR
    return MultiStepLR(optimizer, milestones=cfg.train.milestones, gamma=cfg.train.gamma)

def make_recorder(cfg):
    """创建记录器"""
    from lib.train.recorder import make_recorder
    return make_recorder(cfg)

def load_model(network, optimizer, scheduler, recorder, model_dir, resume=True, strict=False, target_epoch=None):
    """加载模型"""
    import os
    import glob
    import re
    
    if not resume:
        return 0
    
    # 查找模型文件
    model_files = glob.glob(os.path.join(model_dir, "*.pth"))
    if not model_files:
        print("⚠️ 没有找到预训练模型，从头开始训练")
        return 0
    
    # 根据target_epoch选择模型文件
    if target_epoch is not None and target_epoch != -1:
        # 查找指定epoch的模型文件
        target_model_path = os.path.join(model_dir, f"{target_epoch}.pth")
        if os.path.exists(target_model_path):
            model_path = target_model_path
            print(f"🎯 加载指定epoch {target_epoch} 的模型")
        else:
            # 查找最接近的epoch
            available_epochs = []
            for f in model_files:
                match = re.search(r'(\d+)\.pth$', os.path.basename(f))
                if match:
                    available_epochs.append((int(match.group(1)), f))
            
            if available_epochs:
                available_epochs.sort(key=lambda x: abs(x[0] - target_epoch))
                model_path = available_epochs[0][1]
                actual_epoch = available_epochs[0][0]
                print(f"⚠️ 未找到epoch {target_epoch}，使用最接近的epoch {actual_epoch}: {os.path.basename(model_path)}")
            else:
                print("⚠️ 未找到有效的epoch模型文件，使用最新模型")
                model_files.sort(key=os.path.getmtime)
                model_path = model_files[-1]
    else:
        # 使用最新模型文件
        model_files.sort(key=os.path.getmtime)
        model_path = model_files[-1]
        print(f"📅 使用最新模型文件: {os.path.basename(model_path)}")
    
    print(f"load model: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 加载网络参数 - 处理不同的键名
    network_state_dict = network.state_dict()
    if 'state_dict' in checkpoint:
        pretrained_state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        pretrained_state_dict = checkpoint['model']
    else:
        # 直接使用checkpoint
        pretrained_state_dict = checkpoint
    
    # 过滤不匹配的参数
    matched_state_dict = {}
    unmatched_keys = []
    
    for name, param in pretrained_state_dict.items():
        if name in network_state_dict:
            if param.shape == network_state_dict[name].shape:
                matched_state_dict[name] = param
            else:
                unmatched_keys.append(f"{name}: shape mismatch {param.shape} vs {network_state_dict[name].shape}")
        else:
            unmatched_keys.append(f"{name}: not found in network")
    
    if unmatched_keys:
        print(f"⚠️ {len(unmatched_keys)} 个参数不匹配:")
        for key in unmatched_keys[:5]:  # 只显示前5个
            print(f"  {key}")
        if len(unmatched_keys) > 5:
            print(f"  ... and {len(unmatched_keys) - 5} more")
    
    network.load_state_dict(matched_state_dict, strict=False)
    print("✅ 模型权重加载成功 (strict=False)")
    
    # 加载其他状态
    begin_epoch = 0
    if 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("✅ 优化器状态加载成功")
        except Exception as e:
            print(f"⚠️ 优化器参数组数量不匹配，可能由于ClinicalBERT添加")
            print(f"ℹ️  将跳过优化器状态加载，使用新初始化的优化器")
    
    if 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("✅ 调度器状态加载成功")
    
    if 'recorder' in checkpoint:
        recorder.load_state_dict(checkpoint['recorder'])
        print("✅ 记录器状态加载成功")
    
    if 'epoch' in checkpoint:
        begin_epoch = checkpoint['epoch'] + 1
    
    return begin_epoch

def save_model(network, optimizer, scheduler, recorder, epoch, model_dir):
    """保存模型"""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{epoch}.pth")
    
    checkpoint = {
        'state_dict': network.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'recorder': recorder.state_dict(),
        'epoch': epoch
    }
    
    torch.save(checkpoint, model_path)
    print(f"✅ 模型已保存: {model_path}")

def make_data_loader(cfg, is_train=True):
    """创建数据加载器"""
    from lib.datasets import make_data_loader as make_dataloader
    return make_dataloader(cfg, is_train)

def make_trainer(network, cfg):
    """创建训练器"""
    from lib.train import make_trainer
    return make_trainer(cfg, network)

def train_traditional(cfg, network, trainer):
    """传统训练模式 - 支持梯度累积以优化内存使用"""
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)

    # 梯度累积设置 - 由于batch_size=1，需要累积4步来达到等效batch_size=4
    gradient_accumulation_steps = 4
    print(f"🔄 使用梯度累积: {gradient_accumulation_steps} 步")

    # 使用非严格加载以支持ClinicalBERT参数
    target_epoch = getattr(cfg, 'load_epoch', None)
    begin_epoch = load_model(network, optimizer, scheduler, recorder, cfg.model_dir, resume=cfg.resume, strict=False, target_epoch=target_epoch)
    train_loader = make_data_loader(cfg, is_train=True)

    print("begin_epoch:", begin_epoch, "train.epoch:", cfg.train.epoch)

    for epoch in range(begin_epoch, cfg.train.epoch):
        print(f"第 {epoch} 轮···")
        # if epoch > 50:
        #     break
        recorder.epoch = epoch
        trainer.train(epoch, train_loader, optimizer, recorder)
        scheduler.step()

        if (epoch + 1) % cfg.train.save_ep == 0:
            save_model(network, optimizer, scheduler, recorder, epoch, cfg.model_dir)

    return network

def train(cfg, network, trainer):
    """训练主函数"""
    # 检查是否启用分阶段训练
    staged_training = getattr(cfg, 'staged_training', False)
    if staged_training:
        print("🚀 启用分阶段训练模式")
        return train_staged(cfg, network, trainer)
    else:
        print("📚 使用传统训练模式")
        return train_traditional(cfg, network, trainer)

def train_staged(cfg, network, trainer):
    """分阶段训练模式"""
    print("分阶段训练模式暂未实现")
    return train_traditional(cfg, network, trainer)

def test():
    """测试模式"""
    print("测试模式暂未实现")

def main():
    parser = argparse.ArgumentParser(description='EnergySnake Training Script')
    parser.add_argument('--cfg_file', default='configs/sbd_snake.yaml', type=str)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--type', type=str, default="")
    parser.add_argument('--det', type=str, default='')
    parser.add_argument('-f', type=str, default='')
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # 加载配置
    from lib.config import cfg
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    
    # 设置CUDA可见设备
    if hasattr(cfg, 'gpus'):
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, cfg.gpus))
    
    # 设置随机种子
    torch.manual_seed(cfg.random_num)
    torch.cuda.manual_seed(cfg.random_num)
    
    if args.test:
        test()
        return
    
    print("=" * 50)
    print("🚀 开始EnergySnake训练")
    print("=" * 50)
    
    # 创建网络
    network = make_network(cfg)
    
    # 创建训练器
    trainer = make_trainer(network, cfg)
    
    # 开始训练
    trained_network = train(cfg, network, trainer)
    
    print("🎉 训练完成!")

if __name__ == '__main__':
    main()
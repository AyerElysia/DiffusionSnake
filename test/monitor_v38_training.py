#!/usr/bin/env python
"""
V3.8训练监控脚本
自动监控训练进度，判断是否可以开始对比
"""
import json
import time
import subprocess
import sys

def get_latest_epoch_and_loss():
    """获取最新的epoch和loss"""
    try:
        with open('data/outputs/btcv_diffusion_dit_v3_8_single_overfit/logs.jsonl', 'r') as f:
            lines = f.readlines()
            if not lines:
                return None, None

            last_line = lines[-1]
            data = json.loads(last_line)
            return data['epoch'], data['loss']
    except Exception as e:
        print(f"读取日志失败: {e}")
        return None, None

def check_convergence(log_file, window=50):
    """检查训练是否收敛"""
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()

        if len(lines) < window:
            return False, None

        # 获取最近window个epoch的loss
        recent_losses = []
        for line in lines[-window:]:
            data = json.loads(line)
            recent_losses.append(data['loss'])

        # 计算loss的变化率
        avg_loss = sum(recent_losses) / len(recent_losses)
        loss_std = (sum((x - avg_loss) ** 2 for x in recent_losses) / len(recent_losses)) ** 0.5

        # 判断是否收敛：loss标准差小于均值的10%
        is_converged = loss_std < avg_loss * 0.1

        return is_converged, avg_loss
    except Exception as e:
        print(f"检查收敛失败: {e}")
        return False, None

def is_training_running():
    """检查训练进程是否还在运行"""
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )
        return 'diffusion_train.py' in result.stdout
    except:
        return False

def main():
    print("开始监控V3.8训练...")
    print("判断标准：")
    print("  1. 至少训练500 epoch")
    print("  2. 最近50个epoch的loss标准差 < 均值的10%（收敛）")
    print("  3. 或者训练到1000 epoch")
    print()

    log_file = 'data/outputs/btcv_diffusion_dit_v3_8_single_overfit/logs.jsonl'
    check_interval = 60  # 每60秒检查一次

    while True:
        epoch, loss = get_latest_epoch_and_loss()

        if epoch is None:
            print("无法读取训练进度，等待...")
            time.sleep(check_interval)
            continue

        # 检查训练是否还在运行
        if not is_training_running():
            print(f"\n训练进程已停止（epoch {epoch}）")
            print("开始对比...")
            sys.exit(0)

        # 检查是否达到最小epoch要求
        if epoch < 500:
            print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch}, Loss {loss:.6f} - 等待至少500 epoch...")
            time.sleep(check_interval)
            continue

        # 检查是否收敛
        is_converged, avg_loss = check_convergence(log_file, window=50)

        if is_converged:
            print(f"\n训练已收敛！")
            print(f"  当前epoch: {epoch}")
            print(f"  最近50 epoch平均loss: {avg_loss:.6f}")
            print(f"  开始对比...")
            sys.exit(0)

        # 如果达到1000 epoch，无论是否收敛都开始对比
        if epoch >= 1000:
            print(f"\n已达到1000 epoch，开始对比...")
            sys.exit(0)

        print(f"[{time.strftime('%H:%M:%S')}] Epoch {epoch}, Loss {loss:.6f}, Avg Loss (50): {avg_loss:.6f} - 继续等待...")
        time.sleep(check_interval)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n监控已停止")
        sys.exit(1)

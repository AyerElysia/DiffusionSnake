"""FSDP 训练与检查点工具

本模块提供基于 PyTorch Fully Sharded Data Parallel (FSDP) 的封装与实用函数：
- `FSDPConfig`: 配置对象，用于集中管理分片策略、混合精度、激活检查点等开关。
- `fsdp_wrapper`: 按配置将模型包装为 FSDP 模型，并可选择启用激活检查点与 device mesh。
- `save_fsdp_checkpoint`: 以 FULL_STATE_DICT 形式保存 safetensors 检查点（仅 rank 0 落盘）。
- 优化器状态迁移工具：`offload_optimizer_states_to_cpu` / `load_optimizer_states_to_gpu` 与
  包装器 `OptimizerOffload`，便于节省显存。
- `init_distributed`: 基于环境变量快速初始化分布式训练环境（NCCL）。
"""

import os
import functools
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch, MixedPrecision, CPUOffload
from torch.distributed.fsdp.api import StateDictType, FullStateDictConfig
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from safetensors.torch import save_file


class FSDPConfig:
    """FSDP 配置对象

    参数
    - sharding_strategy: 分片策略，字符串键，对应 `ShardingStrategy[...]`，可选：
      - "FULL_SHARD"（参数、梯度、优化器状态全分片）
      - "SHARD_GRAD_OP" 等其他策略
      - "HYBRID_SHARD"（需搭配 device mesh）
    - backward_prefetch: 反向传播预取策略，字符串键，对应 `BackwardPrefetch[...]`。
    - cpu_offload: 是否将参数/offload 到 CPU（通过 `CPUOffload`）。
    - num_replicate: HYBRID_SHARD 时的复制维度大小（数据并行维）。
    - num_shard: HYBRID_SHARD 时的分片维度大小（模型并行/分片维）。
    - mixed_precision_dtype: 混合精度 dtype（例如 `torch.bfloat16`）。
    - use_activation_checkpointing: 是否启用激活检查点以省显存。
    - use_device_mesh: 是否使用 `init_device_mesh` 创建 device mesh（HYBRID_SHARD 常用）。
    """
    def __init__(
        self,
        sharding_strategy="FULL_SHARD",
        backward_prefetch="BACKWARD_PRE", 
        cpu_offload=False,
        num_replicate=1,
        num_shard=8,
        mixed_precision_dtype=torch.bfloat16,
        use_activation_checkpointing=True,
        use_device_mesh=False,
    ):
        self.sharding_strategy = sharding_strategy
        self.backward_prefetch = backward_prefetch
        self.cpu_offload = cpu_offload
        self.num_replicate = num_replicate
        self.num_shard = num_shard
        self.mixed_precision_dtype = mixed_precision_dtype
        self.use_activation_checkpointing = use_activation_checkpointing
        self.use_device_mesh = use_device_mesh

def fsdp_wrapper(model, fsdp_config, get_transformer_layer_cls, ignored_modules=None):
    """将 `model` 包装为 FSDP 模型。

    参数
    - model: 需要并行化的 `nn.Module`。
    - fsdp_config: `FSDPConfig` 实例，控制分片策略、精度与附加功能。
    - get_transformer_layer_cls: 可调用，返回需要 auto-wrap 的 Transformer 层类元组。
      例如返回 `(DecoderLayer,)`，用于 `transformer_auto_wrap_policy`。
    - ignored_modules: 可选，训练中忽略 FSDP 处理的模块列表（如词表嵌入共享等）。

    返回
    - 包装后的 `FSDP` 模型。
    """
    if ignored_modules is None:
        ignored_modules = []
    
    # 若使用 HYBRID_SHARD 且启用 device mesh，则构建 2D mesh（replicate x shard）
    device_mesh = None
    if fsdp_config.sharding_strategy == 'HYBRID_SHARD' and fsdp_config.use_device_mesh:
        device_mesh = init_device_mesh(
            "cuda", 
            mesh_shape=(fsdp_config.num_replicate, fsdp_config.num_shard),
            mesh_dim_names=("replicate", "shard")
        )
    
    # 构建 FSDP 模型：
    # - auto_wrap_policy 指定需要自动包裹的 Transformer 层
    # - mixed_precision 控制参数/通信/缓冲区的 dtype
    # - device_id 将本 rank 绑定到对应 GPU
    # - sharding_strategy / backward_prefetch / cpu_offload / device_mesh 按配置启用
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=get_transformer_layer_cls(),
        ),
        ignored_modules=ignored_modules,
        mixed_precision=MixedPrecision(
            param_dtype=fsdp_config.mixed_precision_dtype,
            reduce_dtype=fsdp_config.mixed_precision_dtype,
            buffer_dtype=fsdp_config.mixed_precision_dtype,
        ),
        device_id=dist.get_rank() % torch.cuda.device_count(),
        sharding_strategy=ShardingStrategy[fsdp_config.sharding_strategy],
        backward_prefetch=BackwardPrefetch[fsdp_config.backward_prefetch],
        cpu_offload=CPUOffload(offload_params=fsdp_config.cpu_offload),
        device_mesh=device_mesh,
        use_orig_params=True,
    )
    
    # 若启用激活检查点，则对指定 Transformer 层应用 checkpoint，以降低显存占用
    if fsdp_config.use_activation_checkpointing:
        def grad_checkpoint_check_fn(module):
            """判定需要做激活检查点的模块（返回 True 则包裹）。"""
            return isinstance(module, tuple(get_transformer_layer_cls()))
        
        apply_activation_checkpointing(
            fsdp_model, 
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            ), 
            check_fn=grad_checkpoint_check_fn
        )
    
    return fsdp_model

    
def save_fsdp_checkpoint(save_dir, model, global_step, rank):
    """保存 FSDP 模型的 FULL_STATE_DICT 检查点（safetensors）。

    说明
    - 使用 `StateDictType.FULL_STATE_DICT`，并通过 `FullStateDictConfig` 设置仅 rank 0 落盘、
      且将权重 offload 到 CPU 后再保存，以减轻显存压力。
    - 其他 rank 调用后通过 barrier 等待。
    """
    save_path = os.path.join(save_dir, f"checkpoint-{global_step}")
    os.makedirs(save_path, exist_ok=True)
    
    # 仅在 rank 0 保存完整权重
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
    ):
        state_dict = model.state_dict()
        if rank == 0:
            save_file(state_dict, os.path.join(save_path, "model.safetensors"))
            print(f"Model saved as safetensors: {save_path}/model.safetensors")
        del state_dict
    
    dist.barrier()

def offload_optimizer_states_to_cpu(optimizer):
    """将优化器状态迁移到 CPU 以节省显存。

    注意：仅迁移张量类型的状态（如动量、二阶矩等）。
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param in optimizer.state:
                state = optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to('cpu', non_blocking=True)


def load_optimizer_states_to_gpu(optimizer):
    """将优化器状态迁回对应参数所在 GPU。"""
    for group in optimizer.param_groups:
        for param in group['params']:
            if param in optimizer.state:
                state = optimizer.state[param]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(param.device, non_blocking=True)


class OptimizerOffload:
    """优化器 CPU offload 包装器。

    用法：
    - 用该类包裹原始优化器后，`step()` 前自动将状态迁回 GPU，
      `step()` 执行后再 offload 回 CPU，减少显存占用峰值。
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer
        
    def step(self, *args, **kwargs):
        load_optimizer_states_to_gpu(self.optimizer)
        result = self.optimizer.step(*args, **kwargs)
        offload_optimizer_states_to_cpu(self.optimizer)
        return result
    
    def __getattr__(self, name):
        # 透传除 step 之外的其他属性/方法到原始优化器
        return getattr(self.optimizer, name)



def init_distributed():
    """初始化分布式训练环境（NCCL）。

    读取以下环境变量：
    - `RANK`, `WORLD_SIZE`, `LOCAL_RANK`（由启动器如 torchrun 设置）。

    返回
    - (is_distributed: bool, rank: int, world_size: int, local_rank: int)
    - 若未处于分布式环境，则返回 (False, 0, 1, 0)
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return False, 0, 1, 0
        
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    
    # 设置当前进程绑定的 CUDA 设备
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    
    return True, rank, world_size, local_rank

from .trainer import Trainer
import importlib.util
import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP


def _wrapper_factory(cfg, network):
    # Determine which wrapper to use based on configuration
    use_diffusion = getattr(cfg, 'use_diffusion_trainer', False)
    use_grpo = bool(getattr(cfg, 'use_grpo', False) or getattr(cfg, 'use_grpo_kl', False) or getattr(getattr(cfg, 'train', cfg), 'use_grpo', False))

    if use_diffusion:
        if use_grpo:
            module = '.'.join(['lib.train.trainers', 'diffusion_grpo_trainer'])
            path = os.path.join('lib/train/trainers', 'diffusion_grpo_trainer.py')
            wrapper_class = 'DiffusionGRPONetworkWrapper'
        else:
            module = '.'.join(['lib.train.trainers', 'diffusion_trainer'])
            path = os.path.join('lib/train/trainers', 'diffusion_trainer.py')
            wrapper_class = 'DiffusionPretrainNetworkWrapper'
    else:
        module = '.'.join(['lib.train.trainers', cfg.task])
        path = os.path.join('lib/train/trainers', cfg.task + '.py')
        wrapper_class = 'NetworkWrapper'

    # Load the appropriate wrapper class
    spec = importlib.util.spec_from_file_location(module, path)
    wrapper_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper_module)
    network_wrapper = getattr(wrapper_module, wrapper_class)(network)
    return network_wrapper


def make_trainer(cfg, network, distributed=False, local_rank=None):
    network = _wrapper_factory(cfg, network)
    if distributed:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError('distributed=True requires an initialized process group')
        if local_rank is None:
            local_rank = torch.distributed.get_rank()
        network = DDP(
            network,
            device_ids=[int(local_rank)],
            output_device=int(local_rank),
            broadcast_buffers=True,
            bucket_cap_mb=int(getattr(cfg, 'ddp_bucket_cap_mb', 25)),
            find_unused_parameters=bool(getattr(cfg, 'ddp_find_unused_parameters', True)),
            gradient_as_bucket_view=bool(getattr(cfg, 'ddp_gradient_as_bucket_view', True)),
        )
    return Trainer(network)

'''
这个项目简直就是套娃······
  make_trainer的作用大概如下：
      从（）传入 make_trainer 函数一个network，首先调用 _wrapper_factory 函数，_wrapper_factory会找到相应任务下的训练器封装文件（暂且叫这个）（比如lib/train/trainers/snake.py），
    并通过imp（）方法提取训练器封装文件中的NetworkWrapper对象，将network作为NetworkWrapper对象的初始化参数，从而返回一个NetworkWrapper实例，这个实例与输入前的network相比，增加了损失函数的计算和记录的方法。
      然后将第一次封装后的对象作为初始化参数，建立一个Trainer类（实际在进行二次封装），Trainer类中又添加正向传播、误差逆传播、评估等功能。
'''

from .yacs import CfgNode as CN
import argparse
import os
import sys
from pathlib import Path

# Modified by Zhang Ruicheng on 2024.04.21

cfg = CN()

cfg.train_split_rate = 0.8
cfg.multistage = False
cfg.random_num = 0

# 中间结果可视化
cfg.vis_zrc = 0

# model
cfg.model = 'ZRC'
cfg.model_dir = 'data/model'

# network
cfg.network = 'dla_34'

cfg.inital_drn22 = '/data/lyc/EnergeSnake/data/model/initial_darnet/drn_d_22-4bd2f8ea.pth'

# network heads
cfg.heads = CN()

# task
cfg.task = ''

# gpus
cfg.gpus = [0]

# if load the pretrained network
cfg.resume = True

# Diffusion/Snake integration switches
cfg.use_diffusion_evolution = False
cfg.use_diffusion_trainer = False
cfg.freeze_snake = False
cfg.diffusion_timesteps = 1000
cfg.use_ddim_inference = True
cfg.diffusion_loss_weight = 1.0
cfg.diffusion_disp_norm = False
cfg.diffusion_disp_stats = ''
cfg.use_dit_denoiser = False
cfg.use_dit_v2 = False
cfg.dit_num_layers = 6
cfg.dit_num_heads = 8
cfg.dit_state_dim = 256

# YOLO/NMS related toggles
cfg.use_nms_for_snake = True
cfg.det_conf_thresh = 0.20
cfg.det_iou_thresh = 0.30
cfg.det_max_det = 300
cfg.per_class_nms = True
cfg.yolo_pretrained = ''
cfg.load_yolo_pretrained = False

# demo
cfg.demo_vis = '/mnt/date/zhangrch/EnergeSnake/zrc_visual/2301/'
#cfg.demo_path = '/home/ub/PycharmProjects/EnergeSnake/multiple_segmentdata/808/'
cfg.demo_path = '/mnt/date/zhangrch/EnergeSnake/multiple_segmentdata/setA/'

# -----------------------------------------------------------------------------
# pretrain for drn
# -----------------------------------------------------------------------------
cfg.pretrain_drn = CN()
# 训练数据路径
cfg.pretrain_drn.train_images_path = '/mnt/date/zhangrch/EnergeSnake/multiple_segmentdata/newenergy/resized'
# 模型参数保存位置
cfg.pretrain_drn.state_dir = '/mnt/date/zhangrch/EnergeSnake/data/model/pretrain_drn/pretrain_epoch_100.pth'
# 训练数据数量
cfg.pretrain_drn.image_nums = 808
# batch_size
cfg.pretrain_drn.batch_size = 3

# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()

cfg.train.dataset = 'SbdTrain'
cfg.train.epoch = 140
cfg.train.num_workers = 8

# use adam as default
cfg.train.optim = 'adam'
cfg.train.lr = 1e-4
cfg.train.weight_decay = 5e-4

cfg.train.warmup = False
cfg.train.scheduler = ''
cfg.train.milestones = [80, 120, 200, 240]
cfg.train.gamma = 0.5

cfg.train.batch_size = 4

cfg.train.data_path = '/multiple_segmentdata/2301/'


# -----------------------------------------------------------------------------
# vis_GT
# -----------------------------------------------------------------------------
cfg.test = CN()
cfg.test.dataset = 'SbdMini'
cfg.test.batch_size = 1
cfg.test.epoch = -1
cfg.test.img_path = '/mnt/date/zhangrch/EnergeSnake/multiple_segmentdata/setA/'

# recorder
cfg.record_dir = 'data/record'

# result
cfg.result_dir = 'data/result'

# evaluation
cfg.skip_eval = False

cfg.save_ep = 100
cfg.eval_ep = 5

cfg.use_gt_det = False

# -----------------------------------------------------------------------------
# snake
# -----------------------------------------------------------------------------
cfg.ct_score = 0.05


def parse_cfg(cfg, args):
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')

    # assign the gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])

    cfg.det_dir = os.path.join(cfg.model_dir, cfg.task, args.det)

    # assign the network head conv
    cfg.head_conv = 64 if 'res' in cfg.network else 256

    #cfg.model_dir = os.path.join(cfg.model_dir)
    cfg.record_dir = os.path.join(cfg.record_dir, cfg.task, cfg.model)
    cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.model)


def _infer_default_cfg(args):
    """Select default config file based on argv or env."""
    if getattr(args, 'cfg_file', None):
        return args.cfg_file
    env_cfg = os.environ.get("CFG_FILE", "")
    if env_cfg:
        return env_cfg
    script = Path(sys.argv[0]).stem.lower()
    if "grpo_train" in script:
        return "configs/grpo_snake.yaml"
    if "diffusion_train" in script:
        return "configs/diffusion_snake.yaml"
    return "configs/sbd_snake.yaml"


def make_cfg(args):
    cfg_file = _infer_default_cfg(args)
    cfg.merge_from_file(cfg_file)  # 从推断的配置文件中加载配置项
    extra_opts = args.opts if args.opts is not None else []
    cfg.merge_from_list(extra_opts)  # 从命令行参数列表中合并额外的配置项
    parse_cfg(cfg, args)
    return cfg


parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", default="", type=str)
parser.add_argument("--ckpt", default="", type=str)
parser.add_argument('--vis_GT', action='store_true', dest='vis_GT', default=False)
parser.add_argument("--type", type=str, default="")
parser.add_argument('--det', type=str, default='')
parser.add_argument('-f', type=str, default='')
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()

if len(args.type) > 0:
    cfg.task = "run"
    print("!111！")
cfg = make_cfg(args)
